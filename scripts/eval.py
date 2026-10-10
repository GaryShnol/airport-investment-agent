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

QUESTIONS = ROOT / "eval" / "questions.yaml"
# A commentary that stops mid-sentence (cut by the context window or max tokens) ends without this.
SENTENCE_END = re.compile(r"[.!?)\]\"'*`:]\s*$")


def load_questions(only: list[str] | None) -> list[dict]:
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    unknown = set(only or []) - {q["id"] for q in questions}
    if unknown:
        sys.exit(f"Unknown question id(s): {', '.join(sorted(unknown))}")
    return [q for q in questions if not only or q["id"] in only]


def score(q: dict, reply, seconds: float) -> dict:
    """Checks for one answer: expected tools, must-have facts, forbidden text, figure check, empty or cut."""
    called = [c["name"] for c in reply.tool_calls]
    labels = [f.get("label", "") for env in reply.tool_results if isinstance(env, dict) for f in env.get("facts") or []]
    expected = q.get("tools")
    tools_ok = None if expected is None else (not called if expected == [] else set(expected) <= set(called))
    removed = reply.figure_check.get("removed", [])
    said = " ".join([reply.commentary, *(r["sentence"] for r in removed)])
    result = {
        "tools_called": called, "tools_ok": tools_ok,
        "missing_facts": [m for m in q.get("must_have", []) if not any(m in label for label in labels)],
        "forbidden": [s for s in q.get("must_not", []) if s.lower() in said.lower()],
        "figure_check": reply.figure_check, "empty": reply.fallback,
        "truncated": bool(reply.commentary) and not SENTENCE_END.search(reply.commentary),
        "seconds": round(seconds, 1), "commentary": reply.commentary,
    }
    result["pass"] = (tools_ok is not False and not result["missing_facts"] and not result["forbidden"]
                      and not removed and not result["empty"] and not result["truncated"])
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
            record = {**base, "error": f"{type(exc).__name__}: {exc}", "pass": False,
                      "seconds": round(time.perf_counter() - start, 1)}
        with path.open("a", encoding="utf-8") as f:  # one line per question, written at once
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _mark(value) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def report(path: Path) -> str:
    """Markdown table of a results file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [f"Model: {rows[0]['model'] if rows else 'n/a'} ({path.name})", "",
             "| Question | Tools called | Seconds | Expected tools | Missing facts | Figures removed "
             "| Empty/cut | Pass |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['id']} | error: {r['error'][:60]} | {r['seconds']} | | | | | no |")
            continue
        removed = len(r["figure_check"].get("removed", []))
        cut = "empty" if r["empty"] else ("cut" if r["truncated"] else "no")
        missing = ", ".join(r["missing_facts"] + [f"wrote '{s}'" for s in r["forbidden"]]) or "none"
        lines.append(f"| {r['id']} | {', '.join(r['tools_called']) or 'none'} | {r['seconds']} | "
                     f"{_mark(r['tools_ok'])} | {missing} | {removed} | {cut} | {_mark(r['pass'])} |")
    passed = sum(bool(r.get("pass")) for r in rows)
    seconds = sum(r.get("seconds", 0) for r in rows)
    lines += ["", f"Passed {passed} of {len(rows)}; total {seconds:.0f} s."]
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
