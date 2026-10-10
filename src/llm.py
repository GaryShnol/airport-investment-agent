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
            "OPENAI_API_KEY is not set: add it to .env for LLM_PROVIDER=openai "
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
        base_url=model_url(s),
        **extra,
    )


PROVIDERS: dict[str, Callable[[Settings], BaseChatModel]] = {
    "openai": _openai,
    "ollama": _ollama,
}


def model_url(s: Settings) -> str:
    """Server the provider calls: LLM_BASE_URL for either provider; for Ollama, OLLAMA_BASE_URL if unset."""
    if s.llm_base_url:
        return s.llm_base_url
    return s.ollama_base_url if s.llm_provider == "ollama" else "https://api.openai.com/v1"


# Exception class names (anywhere in the cause chain) that mean the server could not be reached.
_CONNECTION_ERRORS = {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout"}


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def llm_error_message(settings: Settings, exc: BaseException) -> str:
    """One line for the user when the model cannot be built or called, naming the .env setting to fix.

    Never a traceback: the app and the CLI show this instead. Classified by exception type names and
    `status_code`, so no provider package has to be imported here."""
    if isinstance(exc, ConfigError):
        return str(exc)
    chain = _chain(exc)
    status = next((getattr(e, "status_code") for e in chain if isinstance(getattr(e, "status_code", None), int)), None)
    if any(type(e).__name__ in _CONNECTION_ERRORS or isinstance(e, (ConnectionError, TimeoutError)) for e in chain):
        setting = ("LLM_BASE_URL" if settings.llm_base_url or settings.llm_provider == "openai"
                   else "LLM_BASE_URL (or OLLAMA_BASE_URL)")
        return (f"Cannot reach the model server at {model_url(settings)}: check {setting} in .env "
                "(from Docker, a host Ollama is http://host.docker.internal:11434) and that the server is running.")
    if status in (401, 403) or any(type(e).__name__ == "AuthenticationError" for e in chain):
        return "The model server rejected the API key: check OPENAI_API_KEY in .env."
    if status == 404 or any(type(e).__name__ == "NotFoundError" for e in chain):
        hint = (f" (for Ollama, run `ollama pull {settings.llm_model}` on the host)"
                if settings.llm_provider == "ollama" else "")
        return f"Model '{settings.llm_model}' not found: check LLM_MODEL in .env{hint}."
    return (f"Error calling the model ({settings.llm_provider}/{settings.llm_model}): {type(exc).__name__}. "
            "Check LLM_PROVIDER, LLM_MODEL and the key or LLM_BASE_URL in .env.")


def get_chat_model(settings: Settings) -> BaseChatModel:
    try:
        build = PROVIDERS[settings.llm_provider]
    except KeyError:
        raise ConfigError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}': set it in .env to one of {', '.join(sorted(PROVIDERS))}."
        ) from None
    return build(settings)
