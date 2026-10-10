"""The agent: a LangChain `create_agent` ReAct loop over the tools in src/tools.py.

The model picks tools and phrases their envelopes; it never computes numbers. Conversation
memory is an in-memory checkpointer keyed by `thread_id` (follow-up questions see earlier turns).

CLI: `uv run python -m src.agent "question"` prints the tools called, the data rendered from their
envelopes (src/render.py), then the model's commentary: the same layout as the Streamlit app.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src import store
from src.config import ConfigError, ScoringConfig, Settings, get_settings, load_scoring
from src.llm import get_chat_model
from src.prompts import SYSTEM_PROMPT
from src.render import render_reply
from src.tools import make_tools

# Reasoning models (qwen3, gpt-oss via some servers) may prefix the answer with a <think> block,
# sometimes without the opening tag. Neither must reach the user.
_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_THINK_TAIL = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


class InputRejected(ValueError):
    """The question was refused before reaching the model (empty or too long)."""


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks (and an unopened leading one) from model output."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_TAIL.sub("", text)
    return text.strip()


def check_input(text: str | None, max_chars: int) -> str:
    """Light guard: non-empty and at most `max_chars` characters (MAX_INPUT_CHARS)."""
    question = (text or "").strip()
    if not question:
        raise InputRejected("Please type a question.")
    if len(question) > max_chars:
        raise InputRejected(f"The question is too long ({len(question):,} characters; the limit is {max_chars:,}).")
    return question


FACTS_FOOTER = "Key figures (from the data tool):"
# A period such as 2025-01 in a fact value: the answer only has to name its year(s), since models
# write "January to December 2025". In the answer, such date tokens (any dash) are removed before
# numbers are compared, so "2025-12" never counts as the number 12.
_PERIOD_IN_FACT = re.compile(r"\b(\d{4})-\d{2}\b")
_DATE_IN_ANSWER = re.compile(r"\b\d{4}[-\u2010\u2011\u2012\u2013\u2014/]\d{1,2}\b")
_NUMBER_IN_FACT = re.compile(r"\d[\d,]*(?:\.\d+)?")     # tools write 2,714,358 and 43.2
_NUMBER_IN_ANSWER = re.compile(r"\d+(?:[.,]\d+)*")


def _answer_numbers(answer: str) -> set[Decimal]:
    """Every number in the answer, read both ways where ambiguous: '2,714,358' and '43.2' as
    usual, and '43,2' (single comma, no dot) also as a decimal comma."""
    numbers = set()
    for token in _NUMBER_IN_ANSWER.findall(_DATE_IN_ANSWER.sub(" ", answer)):
        readings = [token.replace(",", "")]
        if token.count(",") == 1 and "." not in token:
            readings.append(token.replace(",", "."))
        for reading in readings:
            try:
                numbers.add(Decimal(reading))
            except InvalidOperation:
                pass  # e.g. "1.2.3"
    return numbers


def _number_present(target: Decimal, numbers: set[Decimal]) -> bool:
    """Tolerance rule: the exact value (trailing zeros ignored: 0.10 = 0.1); for values above 10
    also the nearest integer, half up (43.2 -> 43, 11.5 -> 12); for values below 1 also the
    value written as a percentage (0.385 -> 38.5), since weights are often phrased that way."""
    accepted = {target}
    if abs(target) > 10:
        accepted.add(target.quantize(Decimal(1), rounding=ROUND_HALF_UP))
    if abs(target) < 1:
        accepted.add(target * 100)
    return bool(accepted & numbers)


def _fact_present(value: str, answer: str, numbers: set[Decimal]) -> bool:
    years = _PERIOD_IN_FACT.findall(value)
    if years:
        return all(re.search(rf"\b{y}\b", answer) for y in dict.fromkeys(years))
    targets = _NUMBER_IN_FACT.findall(value)
    if not targets:  # a word such as the confidence level
        return value.lower() in answer.lower()
    return all(_number_present(Decimal(t.replace(",", "")), numbers) for t in targets)


def missing_facts(answer: str, envelopes: list[Any]) -> list[dict]:
    """Required facts (from tool envelopes) whose numbers or words the answer does not contain.

    Facts are taken in envelope order; a later fact with the same label replaces an earlier one
    (the confidence of airport_metrics wins over that of resolve_airport in the same turn).
    """
    required: dict[str, dict] = {}
    for env in envelopes:
        if isinstance(env, dict):
            for f in env.get("facts") or []:
                if f.get("required") and f.get("value"):
                    required.pop(f["label"], None)
                    required[f["label"]] = f
    numbers = _answer_numbers(answer)
    return [f for f in required.values() if not _fact_present(str(f["value"]), answer, numbers)]


def ensure_facts(answer: str, envelopes: list[Any]) -> str:
    """Append a short block with the required facts the answer left out (no retry, deterministic).

    Mandatory content (both ANC long-haul shares, the threshold sensitivity, the weights used,
    data period and confidence) is guaranteed here by code, not by the prompt.
    """
    missing = missing_facts(answer, envelopes)
    if not missing:
        return answer
    lines = "\n".join(f"- {f['label']}: {f['value']}" for f in missing)
    return f"{answer}\n\n{FACTS_FOOTER}\n{lines}".strip()


def _parse(content: Any) -> Any:
    """Tool envelopes reach the message history as JSON text; give callers the dict back."""
    try:
        return json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return content


@dataclass
class Reply:
    """One turn. Interfaces render `tool_results` (src/render.py) first and show `commentary`
    after it; `text` is the commentary plus the ensure_facts footer, for text-only consumers."""
    text: str
    tool_calls: list[dict] = field(default_factory=list)     # {"name", "args"} in call order
    tool_results: list[Any] = field(default_factory=list)    # envelopes, same order
    missing_facts: list[dict] = field(default_factory=list)  # required facts appended by ensure_facts
    commentary: str = ""                                     # the model's text, reasoning stripped


@dataclass
class AirportAgent:
    graph: Any                 # compiled LangGraph from create_agent
    max_input_chars: int

    def ask(self, question: str, thread_id: str) -> Reply:
        """Run one turn in `thread_id`; returns the cleaned answer and this turn's tool calls."""
        question = check_input(question, self.max_input_chars)
        state = self.graph.invoke({"messages": [HumanMessage(question)]},
                                  config={"configurable": {"thread_id": thread_id}})
        messages = state["messages"]
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        turn = messages[last_human + 1:]
        calls = [{"name": c["name"], "args": c["args"]} for m in turn if isinstance(m, AIMessage)
                 for c in m.tool_calls]
        results = [_parse(m.content) for m in turn if isinstance(m, ToolMessage)]
        final = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
        answer = strip_reasoning(final.text if final else "")
        return Reply(ensure_facts(answer, results), calls, results, missing_facts(answer, results), answer)


def build_agent(
    settings: Settings,
    model: BaseChatModel | None = None,
    connect: Callable[[], sqlite3.Connection] | None = None,
    cfg: ScoringConfig | None = None,
) -> AirportAgent:
    """Agent wired from settings. `model`, `connect` and `cfg` can be injected (tests)."""
    cfg = cfg or load_scoring(settings.weights_path)
    if model is None:
        try:
            model = get_chat_model(settings)
        except ConfigError as exc:
            raise ConfigError(f"The language model is not configured: {exc} "
                              "Set LLM_PROVIDER and LLM_MODEL in .env (see .env.example).") from exc
    if connect is None:
        if not settings.db_path.exists():
            raise ConfigError(f"Database not found at {settings.db_path}. Run `uv run python -m src.ingest` first.")
        db_path = settings.db_path

        def connect() -> sqlite3.Connection:
            return store.connect_ro(db_path)

    graph = create_agent(model, make_tools(connect, cfg), system_prompt=SYSTEM_PROMPT,
                         checkpointer=InMemorySaver())
    return AirportAgent(graph, settings.max_input_chars)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the airport agent one question.")
    parser.add_argument("question")
    parser.add_argument("--thread", default="cli", help="conversation id (memory is per process)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # answers can be Hebrew or other non-ASCII text
    try:
        settings = get_settings()
        agent = build_agent(settings)
        reply = agent.ask(args.question, args.thread)
    except (ConfigError, InputRejected) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - e.g. the model server is not running
        print(f"Error calling the model ({settings.llm_provider}/{settings.llm_model}): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Model: {settings.llm_provider}/{settings.llm_model}")
    print("Tools called:" if reply.tool_calls else "Tools called: none")
    for call in reply.tool_calls:
        print(f"  - {call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})")
    print(f"Required facts the commentary omits: {len(reply.missing_facts)} (all are in the data below)")
    print("\nData (from tools):\n" + (render_reply(reply.tool_calls, reply.tool_results) or "none"))
    print("\nCommentary (model):\n" + reply.commentary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
