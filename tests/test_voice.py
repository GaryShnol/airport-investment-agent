"""Voice input (P8.1): a recording becomes text through a stubbed transcription call; problems are one line."""
import io
import logging
import wave
from types import SimpleNamespace

import httpx
import openai
import pytest

from src import config
from src.config import ConfigError, Settings
from src.voice import VoiceError, is_no_speech, recording_seconds, stt_url, transcribe, unavailable_reason

OPENAI = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "STT_MODEL": "stt-test",
                            "STT_MAX_SECONDS": "5"})
HINTED = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test",
                            "STT_PROMPT": "Questions about US airports like LAX, Santa Ana (SNA), ANC and SFO."})
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


@pytest.mark.parametrize("text", ["", "   ", " ... "])
def test_silence_gives_an_empty_result(text):
    assert transcribe(OPENAI, wav(3), FakeClient(text=text)) == ""


def test_a_transcript_that_echoes_the_vocabulary_hint_is_no_speech():
    # Manual test: 3 s of silence came back as the hint itself and was sent to the agent as a question.
    hint = HINTED.stt_prompt
    for echo in (hint, hint.upper().replace(",", "").replace(":", " "), f"{hint} Thank you."):
        assert transcribe(HINTED, wav(3), FakeClient(text=echo)) == "", echo
    assert not is_no_speech("Compare LAX and SNA congestion.", hint)          # a question naming the codes
    assert not is_no_speech("Compare LAX and SNA congestion.", "")            # no hint: only empty is no speech
    assert is_no_speech("Airport questions: LAX, SNA.", "airport questions -- LAX SNA")


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


def api_error(cls, status, message):
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)


@pytest.mark.parametrize("error, expected", [
    (openai.APIConnectionError(request=REQUEST),
     "Transcription failed: network error, cannot reach https://api.openai.com/v1. Check the network and "
     "STT_BASE_URL (or LLM_BASE_URL) in .env."),
    (api_error(openai.AuthenticationError, 401, "Incorrect API key provided"),
     "Transcription failed: invalid API key. Check OPENAI_API_KEY in .env."),
    (api_error(openai.PermissionDeniedError, 403, "Project does not have access"),
     "Transcription failed: invalid API key. Check OPENAI_API_KEY in .env."),
    (api_error(openai.NotFoundError, 404, "no model"),
     "Transcription failed: model 'stt-test' not available. Check STT_MODEL in .env."),
    (api_error(openai.BadRequestError, 400, "Invalid model stt-test for transcription"),
     "Transcription failed: model 'stt-test' not available. Check STT_MODEL in .env."),
    (api_error(openai.BadRequestError, 400, "Invalid language 'xx'. Language parameter must be ISO-639-1"),
     "Transcription failed: invalid STT_LANGUAGE 'en'. Use a supported two-letter code or leave it empty to "
     "auto-detect."),
    (api_error(openai.BadRequestError, 400, "Audio file is too short. Minimum audio length is 0.1 seconds."),
     "Transcription failed: audio too long or empty. Record again (limit STT_MAX_SECONDS=5 s)."),
    (api_error(openai.APIStatusError, 413, "Maximum content size limit exceeded"),
     "Transcription failed: audio too long or empty. Record again (limit STT_MAX_SECONDS=5 s)."),
    (RuntimeError("boom"),
     "Transcription failed (RuntimeError): see the app logs for details."),
])
def test_api_errors_become_one_line_naming_the_cause(error, expected):
    with pytest.raises(VoiceError) as err:
        transcribe(OPENAI, wav(1), FakeClient(error=error))
    assert str(err.value) == expected and "Traceback" not in str(err.value)


def test_a_failed_call_is_logged_with_its_traceback_but_no_key_or_audio(caplog):
    audio = wav(1)
    with caplog.at_level(logging.ERROR, logger="src.voice"), pytest.raises(VoiceError):
        transcribe(OPENAI, audio, FakeClient(error=RuntimeError("boom")))
    [record] = caplog.records
    assert record.exc_info and "RuntimeError: boom" in caplog.text and "stt-test" in caplog.text
    assert "sk-test" not in caplog.text and repr(audio[:16]) not in caplog.text


def test_the_shipped_defaults_send_model_and_language_and_no_hint():
    defaults = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"})
    client = FakeClient(text="Compare LA and Santa Ana congestion.")
    assert transcribe(defaults, wav(1), client) == "Compare LA and Santa Ana congestion."
    [kwargs] = client.calls
    assert kwargs["model"] == "gpt-4o-mini-transcribe" and kwargs["language"] == "en"
    assert "prompt" not in kwargs and config.STT_PROMPT == "" and kwargs["file"][0] == "recording.wav"


def test_vocabulary_hint_is_sent_when_set_and_omitted_when_empty():
    client = FakeClient(text="Compare LA and Santa Ana congestion.")
    transcribe(HINTED, wav(1), client)
    assert client.calls[0]["prompt"] == HINTED.stt_prompt and "Santa Ana" in HINTED.stt_prompt
    custom = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "STT_PROMPT": " LAX, SNA "})
    transcribe(custom, wav(1), client)
    assert client.calls[1]["prompt"] == "LAX, SNA"
    off = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "STT_PROMPT": ""})
    transcribe(off, wav(1), client)
    assert off.stt_prompt == "" and "prompt" not in client.calls[2]


def test_language_is_sent_by_default_and_omitted_when_empty():
    client = FakeClient(text="Compare LA and Santa Ana congestion.")
    transcribe(OPENAI, wav(1), client)
    assert client.calls[0]["language"] == "en"
    auto = Settings.from_env({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "STT_LANGUAGE": ""})
    transcribe(auto, wav(1), client)
    assert "language" not in client.calls[1]


def test_stt_settings_have_defaults_and_are_validated():
    s = Settings.from_env({})
    assert (s.stt_model, s.stt_max_seconds, s.stt_base_url, s.stt_language) == ("gpt-4o-mini-transcribe", 60.0,
                                                                                  None, "en")
    assert s.stt_prompt == ""                  # no vocabulary hint unless STT_PROMPT is set
    assert Settings.from_env({"STT_LANGUAGE": " ES "}).stt_language == "es"
    with pytest.raises(ConfigError, match="STT_MAX_SECONDS"):
        Settings.from_env({"STT_MAX_SECONDS": "0"})
    with pytest.raises(ConfigError, match="STT_LANGUAGE"):
        Settings.from_env({"STT_LANGUAGE": "english"})
