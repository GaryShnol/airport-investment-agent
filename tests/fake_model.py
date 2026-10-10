"""A scripted chat model for agent tests: replays predefined AI messages, records what it saw."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def call(name: str, call_id: str = "c1", **args: Any) -> AIMessage:
    """An AI message asking for one tool call."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """Returns `script` messages in order, one per model call; `seen` keeps every prompt received.

    `bind_tools` returns the model itself: the script, not the tool schema, decides the calls.
    """
    script: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        if len(self.seen) > len(self.script):
            raise AssertionError(f"model called {len(self.seen)} times, script has {len(self.script)}")
        return ChatResult(generations=[ChatGeneration(message=self.script[len(self.seen) - 1])])
