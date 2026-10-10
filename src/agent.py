"""The agent: a LangChain `create_agent` ReAct loop over the tools in src/tools.py.

The model picks tools and phrases their envelopes; it never computes numbers. Conversation
memory is an in-memory checkpointer keyed by `thread_id` (follow-up questions see earlier turns).

CLI: `uv run python -m src.agent "question"` prints the tools called and the final answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src import store
from src.config import ConfigError, ScoringConfig, Settings, get_settings, load_scoring
from src.llm import get_chat_model
from src.prompts import SYSTEM_PROMPT
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


def _parse(content: Any) -> Any:
    """Tool envelopes reach the message history as JSON text; give callers the dict back."""
    try:
        return json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return content


@dataclass
class Reply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)     # {"name", "args"} in call order
    tool_results: list[Any] = field(default_factory=list)    # envelopes, same order


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
        return Reply(strip_reasoning(final.text if final else ""), calls, results)


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
    print("\nAnswer:\n" + reply.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
