"""Voice input (P8.1): a recording becomes text through a stubbed transcription call; problems are one line."""
import io
import wave
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.config import ConfigError, Settings
from src.voice import VoiceError, recording_seconds, stt_url, transcribe, unavailable_reason

OPENAI = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "STT_MODEL": "stt-test",
                            "STT_MAX_SECONDS": "5"})
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")


def wav(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class FakeClient:
    """Stands in for openai.OpenAI: records the call, returns `text` or raises `error`."""

    def __init__(self, text="", error=None):
        self.calls, self.text, self.error = [], text, error
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


def test_success_returns_the_transcript_and_sends_the_configured_model():
    client = FakeClient(text="  Compare LA and Santa Ana congestion.  ")
    audio = wav(2)
    assert transcribe(OPENAI, audio, client) == "Compare LA and Santa Ana congestion."
    (call,) = client.calls
    assert call["model"] == "stt-test" and call["file"] == ("recording.wav", audio, "audio/wav")


@pytest.mark.parametrize("audio", [b"", None, wav(0)])
def test_empty_audio_is_rejected_before_any_call(audio):
    client = FakeClient(text="never")
    with pytest.raises(VoiceError, match="The recording is empty"):
        transcribe(OPENAI, audio, client)
    assert client.calls == []


def test_unreadable_audio_is_one_line():
    with pytest.raises(VoiceError, match="could not be read as WAV"):
        transcribe(OPENAI, b"not a wav file", FakeClient(text="never"))


def test_silent_audio_gives_an_empty_transcript_and_a_message():
    with pytest.raises(VoiceError, match="No speech was recognized"):
        transcribe(OPENAI, wav(2), FakeClient(text="   "))


def test_too_long_recording_names_the_setting_and_sends_nothing():
    client = FakeClient(text="never")
    assert recording_seconds(wav(6)) == 6
    with pytest.raises(VoiceError) as err:
        transcribe(OPENAI, wav(6), client)
    assert str(err.value) == "The recording is 6 s long; the limit is 5 s (STT_MAX_SECONDS in .env)."
    assert client.calls == []


def test_no_key_hides_the_control_and_refuses_to_transcribe():
    no_key = Settings.from_env({"LLM_PROVIDER": "openai"})
    assert unavailable_reason(no_key) == "Voice input is off: set OPENAI_API_KEY (or STT_BASE_URL) in .env."
    with pytest.raises(VoiceError, match="OPENAI_API_KEY"):
        transcribe(no_key, wav(1), FakeClient(text="never"))


def test_other_provider_hides_the_control_unless_an_stt_endpoint_is_set():
    ollama = Settings.from_env({"LLM_PROVIDER": "ollama", "OPENAI_API_KEY": "sk-test"})
    assert "LLM_PROVIDER is ollama" in unavailable_reason(ollama)
    local = Settings.from_env({"LLM_PROVIDER": "ollama", "STT_BASE_URL": "http://localhost:9000/v1"})
    assert unavailable_reason(local) is None and stt_url(local) == "http://localhost:9000/v1"
    assert unavailable_reason(OPENAI) is None and stt_url(OPENAI) == "https://api.openai.com/v1"


@pytest.mark.parametrize("error, expected", [
    (openai.APIConnectionError(request=REQUEST),
     "Cannot reach the transcription server at https://api.openai.com/v1: check the network and STT_BASE_URL "
     "(or LLM_BASE_URL) in .env."),
    (openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None),
     "The transcription API rejected the key: check OPENAI_API_KEY in .env."),
    (openai.NotFoundError("no model", response=httpx.Response(404, request=REQUEST), body=None),
     "The transcription API refused the request for model 'stt-test': check STT_MODEL in .env."),
    (RuntimeError("boom"),
     "Transcription failed (RuntimeError): check STT_MODEL, OPENAI_API_KEY and STT_BASE_URL in .env."),
])
def test_api_errors_become_one_line_naming_the_setting(error, expected):
    with pytest.raises(VoiceError) as err:
        transcribe(OPENAI, wav(1), FakeClient(error=error))
    assert str(err.value) == expected and "Traceback" not in str(err.value)


def test_vocabulary_hint_is_sent_when_set_and_omitted_when_empty():
    client = FakeClient(text="Compare LA and Santa Ana congestion.")
    transcribe(OPENAI, wav(1), client)
    assert client.calls[0]["prompt"] == OPENAI.stt_prompt and "Santa Ana" in OPENAI.stt_prompt
    custom = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "STT_PROMPT": " LAX, SNA "})
    transcribe(custom, wav(1), client)
    assert client.calls[1]["prompt"] == "LAX, SNA"
    off = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "STT_PROMPT": ""})
    transcribe(off, wav(1), client)
    assert off.stt_prompt == "" and "prompt" not in client.calls[2]


def test_stt_settings_have_defaults_and_are_validated():
    s = Settings.from_env({})
    assert (s.stt_model, s.stt_max_seconds, s.stt_base_url) == ("gpt-4o-mini-transcribe", 60.0, None)
    with pytest.raises(ConfigError, match="STT_MAX_SECONDS"):
        Settings.from_env({"STT_MAX_SECONDS": "0"})
