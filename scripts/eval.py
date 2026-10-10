"""Eval harness: run fixed questions through the real agent and model, one JSON line per question.

The model is whatever .env selects (LLM_PROVIDER, LLM_MODEL): change it there and rerun to compare
models. Questions and checks: eval/questions.yaml. Each answer is appended to
eval/results/<timestamp>-<model>.jsonl as soon as it is done, so a stopped run keeps its results.
Not run in CI (needs a model and the real database).

    uv run python scripts/eval.py                         # all questions
    uv run python scripts/eval.py --only q3_anchorage_long_haul,q4_sfo_unmet_demand
    uv run python scripts/eval.py --report eval/results/<file>.jsonl   # re-render the table
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import build_agent  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.render import render_reply  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.yaml"
# A commentary that stops mid-sentence (cut by the context window or max tokens) ends without this.
SENTENCE_END = re.compile(r"[.!?)\]\"'*`:]\s*$")


def load_questions(only: list[str] | None) -> list[dict]:
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    unknown = set(only or []) - {q["id"] for q in questions}
    if unknown:
        sys.exit(f"Unknown question id(s): {', '.join(sorted(unknown))}")
    return [q for q in questions if not only or q["id"] in only]


def _tools_ok(q: dict, calls: list[dict]) -> bool | None:
    """Expected tools called, each `tool_args` entry matched by at least one call of that tool."""
    expected = q.get("tools")
    if expected is None:
        return None
    names = [c["name"] for c in calls]
    if expected == []:
        return not names
    args_ok = all(any(c["name"] == name and all(c["args"].get(k) == v for k, v in want.items()) for c in calls)
                  for name, want in (q.get("tool_args") or {}).items())
    return set(expected) <= set(names) and args_ok


def _reasons(r: dict, q: dict, unsupported: list[dict]) -> list[str]:
    """Which checks failed, in words (empty when the answer passes)."""
    reasons = []
    if r["tools_ok"] is False:
        want = ", ".join(q.get("tools") or []) or "no tool"
        args = q.get("tool_args")
        reasons.append(f"tools: expected {want}{f' with {args}' if args else ''}, called "
                       f"{', '.join(r['tools_called']) or 'none'}")
    if r["missing_facts"]:
        reasons.append(f"missing facts: {', '.join(r['missing_facts'])}")
    if r["forbidden"]:
        reasons.append(f"forbidden text shown: {', '.join(repr(s) for s in r['forbidden'])}")
    if unsupported:
        reasons.append("figure check removed: " + "; ".join(x for u in unsupported for x in u["reasons"]))
    if r["empty"]:
        reasons.append("empty reply (fixed fallback text)")
    if r["truncated"]:
        reasons.append("reply cut mid-sentence")
    return reasons


def score(q: dict, reply, seconds: float) -> dict:
    """Checks for one answer: expected tools, must-have facts, forbidden text, figure check, empty or cut.

    Forbidden text is checked in what the user sees (rendered cards and the commentary after the
    figure check). A forbidden string that only the removed sentences contained is the guardrail
    working: recorded in `forbidden_filtered`, not a failure. With `answer_from_card`, an answer
    with no tool call passes the tool and fact checks when it has figures and the figure check
    (against the earlier cards of the conversation) removed none of them."""
    calls = [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls]
    called = [c["name"] for c in calls]
    labels = [f.get("label", "") for env in reply.tool_results if isinstance(env, dict) for f in env.get("facts") or []]
    removed = reply.figure_check.get("removed", [])
    # In restricted mode removing figures is the design, not an unsupported claim: not a failure.
    unsupported = removed if reply.figure_check.get("mode") != "restricted" else []
    visible = f"{render_reply(reply.tool_calls, reply.tool_results)}\n{reply.commentary}".lower()
    raw = " ".join([reply.commentary, *(r["sentence"] for r in removed)]).lower()
    must_not = q.get("must_not", [])
    from_card = (q.get("answer_from_card") and not called and not unsupported and not reply.fallback
                 and any(ch.isdigit() for ch in reply.commentary))
    result = {
        "tools_called": called, "tool_calls": calls,
        "tools_ok": True if from_card else _tools_ok(q, calls), "answered_from_card": bool(from_card),
        "missing_facts": [] if from_card else [m for m in q.get("must_have", []) if not any(m in x for x in labels)],
        "forbidden": [s for s in must_not if s.lower() in visible],
        "forbidden_filtered": [s for s in must_not if s.lower() in raw and s.lower() not in visible],
        "figure_check": reply.figure_check, "empty": reply.fallback,
        "truncated": bool(reply.commentary) and not SENTENCE_END.search(reply.commentary),
        "seconds": round(seconds, 1), "commentary": reply.commentary,
    }
    reasons = _reasons(result, q, unsupported)
    result["pass"] = not reasons
    result["reason"] = "; ".join(reasons)
    return result


def run(questions: list[dict], out_dir: Path) -> Path:
    settings = get_settings()
    agent = build_agent(settings)
    model = f"{settings.llm_provider}-{settings.llm_model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9.]+', '-', model)}.jsonl"
    run_id = uuid.uuid4().hex[:8]
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['id']} ...", file=sys.stderr, flush=True)
        base = {"id": q["id"], "group": q.get("group"), "question": q["question"], "model": model}
        start = time.perf_counter()
        try:
            reply = agent.ask(q["question"], f"eval-{run_id}-{q.get('group') or q['id']}")
            record = {**base, **score(q, reply, time.perf_counter() - start)}
        except Exception as exc:  # noqa: BLE001 - e.g. the model server stopped; keep going
            error = f"{type(exc).__name__}: {exc}"
            record = {**base, "error": error, "pass": False, "reason": f"error: {error}",
                      "seconds": round(time.perf_counter() - start, 1)}
        with path.open("a", encoding="utf-8") as f:  # one line per question, written at once
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _mark(value) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def report(path: Path) -> str:
    """Markdown table of a results file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [f"Model: {rows[0]['model'] if rows else 'n/a'} ({path.name})", "",
             "| Question | Tools called | Seconds | Expected tools | Missing facts | Figures removed "
             "| Empty/cut | Pass | Reason |", "|---|---|---|---|---|---|---|---|---|"]
    notes = []
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['id']} | error: {r['error'][:60]} | {r['seconds']} | | | | | no | error |")
            continue
        removed = len(r["figure_check"].get("removed", []))
        cut = "empty" if r["empty"] else ("cut" if r["truncated"] else "no")
        missing = ", ".join(r["missing_facts"] + [f"wrote '{s}'" for s in r["forbidden"]]) or "none"
        lines.append(f"| {r['id']} | {', '.join(r['tools_called']) or 'none'} | {r['seconds']} | "
                     f"{_mark(r['tools_ok'])} | {missing} | {removed} | {cut} | {_mark(r['pass'])} | "
                     f"{_cell(r.get('reason', ''))} |")
        if r.get("forbidden_filtered"):
            notes.append(f"- {r['id']}: the raw reply contained {', '.join(map(repr, r['forbidden_filtered']))}; "
                         "the figure check removed it (guardrail working, not a failure).")
        if r.get("answered_from_card"):
            notes.append(f"- {r['id']}: answered from an earlier card (no tool call, every figure supported).")
    passed = sum(bool(r.get("pass")) for r in rows)
    seconds = sum(r.get("seconds", 0) for r in rows)
    lines += ["", f"Passed {passed} of {len(rows)}; total {seconds:.0f} s."]
    if notes:
        lines += ["", "Notes:", *notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval questions against the model in .env.")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--report", type=Path, help="print the table of a saved .jsonl and exit")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    path = args.report or run(load_questions(args.only.split(",") if args.only else None), args.out_dir)
    print(report(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
