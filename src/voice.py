"""Voice input for the app: a recording becomes text through the OpenAI audio transcription API.

The transcript then takes the typed-text path (`AirportAgent.ask`, with the MAX_INPUT_CHARS guard), so
voice adds no second way into the agent. The audio is held in memory for one call only: it is never
written to disk or kept in the session state. Problems become a `VoiceError` with one line naming the
setting to fix, never a traceback.
"""
from __future__ import annotations

import io
import wave
from typing import Any

from .config import Settings
from .llm import CONNECTION_ERRORS, error_chain


class VoiceError(ValueError):
    """A recording that cannot be turned into a question, with an actionable one-line message."""


def unavailable_reason(s: Settings) -> str | None:
    """Why the microphone is hidden, or None when transcription is configured.

    Shown when STT_BASE_URL is set, or with the OpenAI provider and a key (the existing LLM credentials)."""
    if s.stt_base_url:
        return None
    if s.llm_provider != "openai":
        return (f"Voice input is off: it uses the OpenAI transcription API, and LLM_PROVIDER is {s.llm_provider} "
                "(set STT_BASE_URL in .env for another transcription server).")
    if not s.openai_api_key:
        return "Voice input is off: set OPENAI_API_KEY (or STT_BASE_URL) in .env."
    return None


def stt_url(s: Settings) -> str:
    """Transcription server: STT_BASE_URL, else LLM_BASE_URL for the OpenAI provider, else the OpenAI API."""
    if s.stt_base_url:
        return s.stt_base_url
    if s.llm_provider == "openai" and s.llm_base_url:
        return s.llm_base_url
    return "https://api.openai.com/v1"


def recording_seconds(audio: bytes) -> float:
    """Length of a WAV recording (st.audio_input records WAV)."""
    with wave.open(io.BytesIO(audio)) as w:
        rate = w.getframerate()
        return w.getnframes() / rate if rate else 0.0


def check_recording(audio: bytes | None, max_seconds: float) -> float:
    """Reject empty, unreadable or too long recordings before anything is sent."""
    if not audio:
        raise VoiceError("The recording is empty: record again.")
    try:
        seconds = recording_seconds(audio)
    except (wave.Error, EOFError) as exc:
        raise VoiceError("The recording could not be read as WAV audio: record again.") from exc
    if seconds <= 0:
        raise VoiceError("The recording is empty: record again.")
    if seconds > max_seconds:
        raise VoiceError(f"The recording is {seconds:.0f} s long; the limit is {max_seconds:g} s "
                         "(STT_MAX_SECONDS in .env).")
    return seconds


def error_message(s: Settings, exc: BaseException) -> str:
    """One line for a failed transcription call, naming the .env setting to fix."""
    chain = error_chain(exc)
    status = next((getattr(e, "status_code") for e in chain if isinstance(getattr(e, "status_code", None), int)), None)
    if any(type(e).__name__ in CONNECTION_ERRORS or isinstance(e, (ConnectionError, TimeoutError)) for e in chain):
        return (f"Cannot reach the transcription server at {stt_url(s)}: check the network and STT_BASE_URL "
                "(or LLM_BASE_URL) in .env.")
    if status in (401, 403) or any(type(e).__name__ == "AuthenticationError" for e in chain):
        return "The transcription API rejected the key: check OPENAI_API_KEY in .env."
    if status in (400, 404) or any(type(e).__name__ in ("NotFoundError", "BadRequestError") for e in chain):
        return f"The transcription API refused the request for model '{s.stt_model}': check STT_MODEL in .env."
    return (f"Transcription failed ({type(exc).__name__}): check STT_MODEL, OPENAI_API_KEY and STT_BASE_URL "
            "in .env.")


def _client(s: Settings) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=s.openai_api_key or "not-needed", base_url=stt_url(s))


def transcribe(s: Settings, audio: bytes | None, client: Any = None) -> str:
    """The text of one recording; raises VoiceError for anything the user has to act on."""
    if reason := unavailable_reason(s):
        raise VoiceError(reason)
    check_recording(audio, s.stt_max_seconds)
    client = client or _client(s)
    try:
        result = client.audio.transcriptions.create(model=s.stt_model, file=("recording.wav", audio, "audio/wav"))
    except Exception as exc:  # noqa: BLE001 - every failure becomes one line naming the setting
        raise VoiceError(error_message(s, exc)) from exc
    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise VoiceError("No speech was recognized (silent recording?): record again, closer to the microphone.")
    return text
