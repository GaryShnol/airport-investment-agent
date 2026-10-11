"""Voice input for the app: a recording becomes text through the OpenAI audio transcription API.

The transcript then takes the typed-text path (`AirportAgent.ask`, with the MAX_INPUT_CHARS guard), so
voice adds no second way into the agent. The audio is held in memory for one call only: it is never
written to disk or kept in the session state. Problems become a `VoiceError` with one line naming the
setting to fix, never a traceback (that goes to the log). STT_LANGUAGE fixes the language (an accent is
not read as Spanish) and STT_PROMPT, a one-sentence vocabulary hint, is sent with each recording when set.
On silence a model can answer with the hint itself, so a transcript that is empty or repeats the hint
counts as no speech: the result is "" and nothing reaches the agent. The app shows the transcript for review before it is sent.
"""
from __future__ import annotations

import io
import logging
import re
import wave
from typing import Any

from .config import Settings
from .llm import CONNECTION_ERRORS, error_chain


NO_SPEECH = "Didn't catch that - please try again."
log = logging.getLogger(__name__)


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
    """"Transcription failed: <cause>" for a failed transcription call, naming the .env setting to fix.

    The cause is read from the exception type, the HTTP status and the API's error text; the traceback goes
    to the log (see `transcribe`), never to the user."""
    chain = error_chain(exc)
    status = next((getattr(e, "status_code") for e in chain if isinstance(getattr(e, "status_code", None), int)), None)
    names = {type(e).__name__ for e in chain}
    text = " ".join(str(e) for e in chain).lower()
    if names & CONNECTION_ERRORS or any(isinstance(e, (ConnectionError, TimeoutError)) for e in chain):
        return (f"Transcription failed: network error, cannot reach {stt_url(s)}. Check the network and "
                "STT_BASE_URL (or LLM_BASE_URL) in .env.")
    if status in (401, 403) or "AuthenticationError" in names or "PermissionDeniedError" in names:
        return "Transcription failed: invalid API key. Check OPENAI_API_KEY in .env."
    if status == 404 or "NotFoundError" in names or (status == 400 and "model" in text):
        return f"Transcription failed: model '{s.stt_model}' not available. Check STT_MODEL in .env."
    if status == 400 and "language" in text:
        return (f"Transcription failed: invalid STT_LANGUAGE '{s.stt_language}'. Use a supported two-letter code "
                "or leave it empty to auto-detect.")
    if status == 413 or (status == 400 and any(w in text for w in ("audio", "duration", "too long", "file"))):
        return (f"Transcription failed: audio too long or empty. Record again "
                f"(limit STT_MAX_SECONDS={s.stt_max_seconds:g} s).")
    return f"Transcription failed ({type(exc).__name__}): see the app logs for details."


def normalize(text: str) -> str:
    """Lowercase words without punctuation, single-spaced: the form compared by the echo guard."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def is_no_speech(text: str, prompt: str) -> bool:
    """True when a transcript holds no question: empty, or equal to / containing the vocabulary hint."""
    said = normalize(text)
    return not said or bool(prompt and normalize(prompt) in said)


def _client(s: Settings) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=s.openai_api_key or "not-needed", base_url=stt_url(s))


def transcribe(s: Settings, audio: bytes | None, client: Any = None) -> str:
    """The text of one recording, "" when no speech was detected; raises VoiceError for anything the user has to
    act on."""
    if reason := unavailable_reason(s):
        raise VoiceError(reason)
    check_recording(audio, s.stt_max_seconds)
    client = client or _client(s)
    try:
        options = {"prompt": s.stt_prompt} if s.stt_prompt else {}     # STT_PROMPT: vocabulary hint, optional
        if s.stt_language:                                              # STT_LANGUAGE: "" = auto-detect
            options["language"] = s.stt_language
        result = client.audio.transcriptions.create(model=s.stt_model, file=("recording.wav", audio, "audio/wav"),
                                                    **options)
    except Exception as exc:  # noqa: BLE001 - logged in full, shown to the user as one line naming the cause
        log.exception("Transcription call failed (model %s, server %s)", s.stt_model, stt_url(s))
        raise VoiceError(error_message(s, exc)) from exc
    text = (getattr(result, "text", None) or "").strip()
    return "" if is_no_speech(text, s.stt_prompt) else text
