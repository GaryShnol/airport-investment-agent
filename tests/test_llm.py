"""LLM errors become one line naming the .env setting to fix (P5.10): the app and the CLI never show a traceback."""
import httpx
import openai
import pytest
from ollama import ResponseError

from src.config import ConfigError, Settings
from src.llm import get_chat_model, llm_error_message, model_url

OLLAMA = Settings.from_env({"LLM_PROVIDER": "ollama", "LLM_MODEL": "lfm2.5:8b",
                            "LLM_BASE_URL": "http://host.docker.internal:11434"})
OPENAI = Settings.from_env({"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "sk-test"})
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(cls, code):
    return cls("rejected", response=httpx.Response(code, request=REQUEST), body=None)


def wrapped(exc):
    """As LangChain raises them: the provider error as the cause of another exception."""
    try:
        try:
            raise exc
        except Exception as inner:
            raise RuntimeError("model call failed") from inner
    except RuntimeError as outer:
        return outer


def test_llm_base_url_serves_both_providers():
    assert model_url(OLLAMA) == "http://host.docker.internal:11434"
    assert get_chat_model(OLLAMA).base_url == "http://host.docker.internal:11434"
    assert model_url(Settings.from_env({"LLM_PROVIDER": "ollama"})) == "http://localhost:11434"  # OLLAMA_BASE_URL
    assert model_url(OPENAI) == "https://api.openai.com/v1"


def test_missing_key_names_openai_api_key():
    with pytest.raises(ConfigError) as err:
        get_chat_model(Settings.from_env({"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o-mini"}))
    message = llm_error_message(OPENAI, err.value)
    assert message.startswith("OPENAI_API_KEY is not set") and "\n" not in message


@pytest.mark.parametrize("settings, exc, expected", [
    (OLLAMA, wrapped(httpx.ConnectError("Connection refused")),
     "Cannot reach the model server at http://host.docker.internal:11434: check LLM_BASE_URL in .env"),
    (OLLAMA, ConnectionError("Failed to connect to Ollama"), "Cannot reach the model server at http://host.docker"),
    (OPENAI, wrapped(openai.APIConnectionError(request=REQUEST)),
     "Cannot reach the model server at https://api.openai.com/v1: check LLM_BASE_URL in .env"),
    (OPENAI, wrapped(status_error(openai.AuthenticationError, 401)),
     "The model server rejected the API key: check OPENAI_API_KEY in .env."),
    (OPENAI, status_error(openai.NotFoundError, 404), "Model 'gpt-4o-mini' not found: check LLM_MODEL in .env."),
    (OLLAMA, ResponseError("model 'lfm2.5:8b' not found", 404),
     "Model 'lfm2.5:8b' not found: check LLM_MODEL in .env (for Ollama, run `ollama pull lfm2.5:8b` on the host)."),
    (OPENAI, ValueError("something else"),
     "Error calling the model (openai/gpt-4o-mini): ValueError. Check LLM_PROVIDER, LLM_MODEL and the key"),
])
def test_runtime_errors_name_the_setting(settings, exc, expected):
    message = llm_error_message(settings, exc)
    assert message.startswith(expected) and "\n" not in message
    assert "sk-test" not in message  # never the key itself


def test_unknown_provider_names_llm_provider():
    with pytest.raises(ConfigError) as err:
        get_chat_model(Settings.from_env({"LLM_PROVIDER": "nope"}))
    assert llm_error_message(OPENAI, err.value) == "Unknown LLM_PROVIDER 'nope': set it in .env to one of ollama, openai."
