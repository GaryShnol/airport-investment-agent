"""LLM access. The rest of the project only calls get_chat_model(settings).

A registry of provider builders, not a class hierarchy: LangChain already gives every
provider the same BaseChatModel interface. Adding a provider = one builder + one entry.
Provider packages are imported lazily so an unused provider needs no setup.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel

from .config import ConfigError, Settings


def _openai(s: Settings) -> BaseChatModel:
    # Also covers any OpenAI-compatible server (Ollama /v1, LM Studio, vLLM) via LLM_BASE_URL.
    if not s.openai_api_key and not s.llm_base_url:
        raise ConfigError(
            "OPENAI_API_KEY is required for LLM_PROVIDER=openai "
            "(or set LLM_BASE_URL to use a local OpenAI-compatible server)."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s.llm_model,
        temperature=s.llm_temperature,
        api_key=s.openai_api_key or "not-needed",
        base_url=s.llm_base_url,
    )


def _ollama(s: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    # num_ctx only when set: otherwise the server's default context window applies.
    extra = {"num_ctx": s.ollama_num_ctx} if s.ollama_num_ctx else {}
    return ChatOllama(
        model=s.llm_model,
        temperature=s.llm_temperature,
        base_url=s.ollama_base_url,
        **extra,
    )


PROVIDERS: dict[str, Callable[[Settings], BaseChatModel]] = {
    "openai": _openai,
    "ollama": _ollama,
}


def get_chat_model(settings: Settings) -> BaseChatModel:
    try:
        build = PROVIDERS[settings.llm_provider]
    except KeyError:
        raise ConfigError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Supported: {', '.join(sorted(PROVIDERS))}"
        ) from None
    return build(settings)
