"""The agent: a LangChain `create_agent` ReAct loop over the tools in src/tools.py.

The model picks tools and phrases their envelopes; it never computes numbers. Conversation
memory is an in-memory checkpointer keyed by `thread_id` (follow-up questions see earlier turns).

CLI: `uv run python -m src.agent "question"` prints the tools called, the data rendered from their
envelopes (src/render.py), then the model's commentary: the same layout as the Streamlit app.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src import store
from src.config import ConfigError, ScoringConfig, Settings, get_settings, load_scoring
from src.figure_check import check_commentary, finish_commentary, removal_note, restrict_commentary
from src.llm import get_chat_model, llm_error_message
from src.prompts import build_system_prompt
from src.render import render_reply
from src.tools import make_tools, region_named, unsupported_region

log = logging.getLogger(__name__)

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


def drop_restated_headlines(commentary: str, envelopes: list[Any]) -> str:
    """Remove verbatim copies of tool headlines from the commentary (case and spacing ignored).

    The headline is rendered above the data card, so a model that copies it adds nothing;
    paraphrases are left alone (the prompt asks for commentary, not a restatement)."""
    for env in envelopes:
        headline = env.get("headline") if isinstance(env, dict) else None
        if headline:
            pattern = r"(?:^|(?<=\s))(?:>\s*)?" + r"\s+".join(map(re.escape, headline.split()))
            commentary = re.sub(pattern, "", commentary, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", commentary).strip()


def _parse(content: Any) -> Any:
    """Tool envelopes reach the message history as JSON text; give callers the dict back."""
    try:
        return json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return content


# --- model-call guard: compact history, context warning, empty-reply retry -------------------

# Rough token estimate for the context warning: about 4 characters per token for English and JSON.
CHARS_PER_TOKEN = 4
# Warn when the estimated prompt reaches this share of OLLAMA_NUM_CTX.
CONTEXT_WARN_RATIO = 0.8
# What an earlier turn's tool result keeps in the prompt: the card the user saw (headline and
# facts, which include one line per ranked airport), not the raw `data`.
COMPACT_KEYS = ("headline", "facts", "data_period", "confidence", "error", "candidates")
RETRY_NUDGE = "Your last reply was empty. Reply to the user now, in text, using the tool results above."
EMPTY_WITH_DATA = "The model returned no commentary; the data above comes from the tools."
EMPTY_NO_DATA = "The model returned no answer. Please try again or rephrase the question."
ONLY_HEADLINE = "The model's reply only repeated the headline above."


def compact_envelope(content: Any) -> Any:
    """An envelope reduced to COMPACT_KEYS (JSON text); anything else is returned unchanged."""
    env = _parse(content)
    if not isinstance(env, dict):
        return content
    return json.dumps({k: env[k] for k in COMPACT_KEYS if env.get(k) not in (None, [], "")}, ensure_ascii=False)


def compact_history(messages: list) -> list:
    """Tool results from earlier turns (before the last user message) in compact form.

    The checkpointer keeps every ToolMessage, so follow-ups ("which of these has the highest
    load factor?") can be answered from the earlier card; full envelopes would fill a local
    model's context in two or three turns, and Ollama then silently cuts the oldest messages.
    Only the prompt is compacted; the stored history and the Reply keep the full envelopes."""
    humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    last_human = humans[-1] if humans else len(messages)
    return [m.model_copy(update={"content": compact_envelope(m.content)})
            if isinstance(m, ToolMessage) and i < last_human else m for i, m in enumerate(messages)]


def _chars(content: Any) -> int:
    return len(content if isinstance(content, str) else json.dumps(content, default=str))


def estimate_tokens(request: ModelRequest) -> tuple[int, int]:
    """(whole prompt, largest tool result) in estimated tokens: system prompt, messages, tool schemas."""
    system = _chars(request.system_message.content) if request.system_message else 0
    schemas = sum(len(getattr(t, "description", "") or "") + _chars(getattr(t, "args", {})) for t in request.tools)
    tool_results = [_chars(m.content) for m in request.messages if isinstance(m, ToolMessage)]
    total = system + schemas + sum(_chars(m.content) for m in request.messages)
    return total // CHARS_PER_TOKEN, max(tool_results, default=0) // CHARS_PER_TOKEN


def _last_message(response: Any) -> Any:
    return response.result[-1] if isinstance(response, ModelResponse) else response


def is_empty_reply(message: Any) -> bool:
    """An AI message with no tool call and no text once reasoning is stripped."""
    return isinstance(message, AIMessage) and not message.tool_calls and not strip_reasoning(message.text)


def turn_guard(num_ctx: int | None):
    """Middleware around every model call: compact earlier tool results, warn when the prompt
    nears `num_ctx` (OLLAMA_NUM_CTX; no check when unset), and retry an empty reply once with
    a short nudge (the nudge is sent to the model only, never stored)."""
    @wrap_model_call(name="TurnGuard")
    def guard(request: ModelRequest, handler: Callable) -> Any:
        request = request.override(messages=compact_history(request.messages))
        if num_ctx:
            total, largest = estimate_tokens(request)
            if total >= CONTEXT_WARN_RATIO * num_ctx:
                log.warning("Prompt is about %s tokens (%.0f%% of OLLAMA_NUM_CTX=%s; largest tool result about %s); "
                            "the model server may cut the oldest messages. Raise OLLAMA_NUM_CTX or start a new chat.",
                            f"{total:,}", 100 * total / num_ctx, f"{num_ctx:,}", f"{largest:,}")
        response = handler(request)
        if is_empty_reply(_last_message(response)):
            log.warning("The model returned an empty reply; retrying once.")
            response = handler(request.override(messages=[*request.messages, HumanMessage(RETRY_NUDGE)]))
        return response
    return guard


@dataclass
class Reply:
    """One turn. Interfaces render `tool_results` (src/render.py) first and show `commentary`
    after it; `text` is the commentary plus the ensure_facts footer, for text-only consumers."""
    text: str
    tool_calls: list[dict] = field(default_factory=list)     # {"name", "args"} in call order
    tool_results: list[Any] = field(default_factory=list)    # envelopes, same order
    missing_facts: list[dict] = field(default_factory=list)  # required facts appended by ensure_facts
    commentary: str = ""                                     # the model's text, reasoning stripped
    fallback: bool = False                                   # commentary is a fixed message (empty reply)
    figure_check: dict = field(default_factory=dict)         # commentary check: mode, removed, hidden, note


def _fallback(raw: str, results: list[Any]) -> str:
    """Deterministic commentary when the model's text is empty after cleaning (never blank)."""
    if raw:
        return ONLY_HEADLINE
    return EMPTY_WITH_DATA if results else EMPTY_NO_DATA


REGION_GUARD = ("{state} is not a configured region, so no ranking is shown (the model asked for region "
                "'{region}'). Ask for the national ranking or about a specific airport.")
REGION_CAVEAT = ("the question also names {state}, which this {region} ranking does not cover ({state} is not "
                 "a configured region); ask about a specific {state} airport or the national ranking")


def _with_region_caveat(envelope: Any, state: str, region: str) -> Any:
    """The ranking envelope with a caveat (first assumption and a fact) for a state the question
    mentions outside the ranked region."""
    if not isinstance(envelope, dict) or envelope.get("error"):
        return envelope
    caveat = REGION_CAVEAT.format(state=state, region=region)
    return {**envelope, "assumptions": [caveat, *(envelope.get("assumptions") or [])],
            "facts": [{"label": "Not covered by this ranking", "value": caveat, "required": False},
                      *(envelope.get("facts") or [])]}


def guard_regions(question: str, calls: list[dict], results: list[Any], cfg: ScoringConfig | None,
                  rank_tool: Any) -> tuple[list[dict], list[Any], str | None]:
    """Replace every ranking that stands in for a state the question names but the region passed
    does not contain (California answered with the national ranking) by the tool's deterministic
    configured-regions envelope for that state. Returns the calls, the results and the fixed
    commentary (None when nothing was replaced). Code, not the prompt: P6.1 showed one model
    passing a configured region for California and getting a national ranking silently.

    Only the question's target region decides: when the question itself names the configured
    region passed (`region_named`: "Rank the New England airports ... as busy as New York?"), the
    ranking is kept and the other state gets a caveat instead."""
    if cfg is None or rank_tool is None:
        return calls, results, None
    calls, results, note = list(calls), list(results), None
    for i, c in enumerate(calls):
        if c["name"] != "rank_airports" or i >= len(results):
            continue
        passed = str(c["args"].get("region", ""))
        state = unsupported_region(question, passed, cfg)
        if state and region_named(question, passed, cfg):
            results[i] = _with_region_caveat(results[i], state, passed)
        elif state:
            calls[i] = {"name": "rank_airports", "args": {"region": state}}
            results[i] = _parse(rank_tool.invoke({"region": state}))
            note = REGION_GUARD.format(state=state, region=passed)
    return calls, results, note


@dataclass
class AirportAgent:
    graph: Any                 # compiled LangGraph from create_agent
    max_input_chars: int
    restricted_tools: tuple[str, ...] = ()   # commentary.restricted_tools: caveats and next steps only
    cfg: ScoringConfig | None = None         # regions, for the region guard
    rank_tool: Any = None                    # the rank_airports tool, for the region guard

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
        calls, results, region_note = guard_regions(question, calls, results, self.cfg, self.rank_tool)
        if region_note:  # the model's text was about the replaced ranking: fixed text, no check needed
            log.info("Region guard replaced a ranking: %s", region_note)
            return Reply(ensure_facts(region_note, results), calls, results, missing_facts(region_note, results),
                         region_note, False, {"mode": "region_guard", "removed": [], "withheld": False, "hidden": False})
        final = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
        raw = strip_reasoning(final.text if final else "")
        # Every tool result of the thread: a follow-up may quote an earlier card.
        thread = [_parse(m.content) for m in messages if isinstance(m, ToolMessage)]
        cleaned = drop_restated_headlines(raw, results)
        restricted = any(c["name"] in self.restricted_tools for c in calls)
        check = restrict_commentary(cleaned, thread) if restricted else check_commentary(cleaned, thread)
        note = ""
        if self.cfg is not None:
            check = finish_commentary(check, self.cfg.commentary_min_words, self.cfg.commentary_fallback)
            note = removal_note(check.removed, self.cfg.commentary_removed_note)
        if check.removed:  # the full detail stays in the log; the user sees the compact note
            log.info("Commentary check (%s) removed %d sentence(s): %s", check.mode, len(check.removed),
                     [item["reasons"] for item in check.removed])
            log.debug("Removed sentences: %s", [item["sentence"] for item in check.removed])
        answer = check.text
        fallback = not answer and not check.hidden   # hidden: the card is the whole answer
        if fallback:
            answer = _fallback(raw, results)
        return Reply(ensure_facts(answer, results), calls, results, missing_facts(answer, results), answer, fallback,
                     {**check.as_dict(), "note": note})


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
            raise ConfigError(f"The language model is not configured: {exc}") from exc
    if connect is None:
        if not settings.db_path.exists():
            raise ConfigError(f"Database not found at {settings.db_path}. Run `uv run python -m src.ingest` first.")
        db_path = settings.db_path

        def connect() -> sqlite3.Connection:
            return store.connect_ro(db_path)

    num_ctx = settings.ollama_num_ctx if settings.llm_provider == "ollama" else None
    tools = make_tools(connect, cfg)
    graph = create_agent(model, tools, system_prompt=build_system_prompt(tools),
                         middleware=[turn_guard(num_ctx)], checkpointer=InMemorySaver())
    rank_tool = next(t for t in tools if t.name == "rank_airports")
    return AirportAgent(graph, settings.max_input_chars, cfg.commentary_restricted_tools, cfg, rank_tool)


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
        print(f"Error: {llm_error_message(settings, exc)}", file=sys.stderr)
        return 1
    print(f"Model: {settings.llm_provider}/{settings.llm_model}")
    print("Tools called:" if reply.tool_calls else "Tools called: none")
    for call in reply.tool_calls:
        print(f"  - {call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})")
    print(f"Required facts the commentary omits: {len(reply.missing_facts)} (all are in the data below)")
    print("\nData (from tools):\n" + (render_reply(reply.tool_calls, reply.tool_results) or "none"))
    print("\nCommentary (model):\n" + (reply.commentary or "none (the card above is the answer)"))
    for item in reply.figure_check.get("removed", []):
        print(f"  [removed by the commentary check: {'; '.join(item['reasons'])}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
