"""scripts/eval.py without a model: the question file, the per-answer checks and the report table."""
import importlib.util
import json

from src.agent import Reply
from src.config import ROOT

spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
eval_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_script)


def reply(commentary="BOS leads.", calls=("rank_airports",), labels=("Weights used",), removed=(), fallback=False):
    return Reply(text=commentary, tool_calls=[{"name": c, "args": {}} for c in calls],
                 tool_results=[{"facts": [{"label": label, "value": "x"} for label in labels]}],
                 commentary=commentary, fallback=fallback,
                 figure_check={"removed": [{"sentence": s, "reasons": ["r"]} for s in removed], "withheld": False})


def test_question_file_is_valid():
    questions = eval_script.load_questions(None)
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)) and 12 <= len(ids) <= 14
    assert {"q1_new_england", "q2_la_vs_santa_ana", "q3_anchorage_long_haul", "q4_sfo_unmet_demand"} <= set(ids)
    assert all(set(q) <= {"id", "question", "group", "tools", "must_have", "must_not"} for q in questions)


def test_pass_needs_tools_facts_and_a_clean_commentary():
    q = {"id": "q", "tools": ["rank_airports"], "must_have": ["Weights used"]}
    assert eval_script.score(q, reply(), 1.0)["pass"] is True
    assert eval_script.score(q, reply(calls=("airport_metrics",)), 1.0)["tools_ok"] is False
    assert eval_script.score(q, reply(labels=()), 1.0)["missing_facts"] == ["Weights used"]
    assert eval_script.score(q, reply(removed=("BOS has 99%.",)), 1.0)["pass"] is False
    assert eval_script.score(q, reply(commentary="BOS leads because"), 1.0)["truncated"] is True
    assert eval_script.score(q, reply(fallback=True), 1.0)["pass"] is False


def test_no_tool_expected_and_forbidden_text():
    q = {"id": "q", "tools": [], "must_not": ["99%"]}
    assert eval_script.score(q, reply(calls=(), labels=()), 1.0)["pass"] is True
    assert eval_script.score(q, reply(calls=("rank_airports",)), 1.0)["tools_ok"] is False
    # forbidden text counts also when the figure check removed it
    assert eval_script.score(q, reply(calls=(), removed=("SFO has a 99% load factor.",)), 1.0)["forbidden"] == ["99%"]


def test_report_table_from_a_results_file(tmp_path):
    path = tmp_path / "run.jsonl"
    rows = [{"id": "q1", "model": "ollama-m", **eval_script.score({"tools": ["rank_airports"]}, reply(), 12.34)},
            {"id": "q2", "model": "ollama-m", "error": "ConnectError: refused", "pass": False, "seconds": 0.1}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    table = eval_script.report(path)
    assert "| q1 | rank_airports | 12.3 | yes | none | 0 | no | yes |" in table
    assert "| q2 | error: ConnectError: refused | 0.1 | | | | | no |" in table
    assert table.endswith("Passed 1 of 2; total 12 s.")
