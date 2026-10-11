"""Smoke test of app.py with Streamlit's AppTest, a scripted fake model and the tiny test DB."""
import logging
import sqlite3
from types import SimpleNamespace

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from src import agent as agent_mod
from src import config, voice
from src.config import ROOT, ConfigError
from tests.conftest import seed_analysis_db
from tests.fake_model import ScriptedChatModel, call
from tests.test_voice import FakeClient, wav

APP = str(ROOT / "app.py")


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    with sqlite3.connect(db_path) as conn:
        seed_analysis_db(conn)
    conn.close()
    monkeypatch.setenv("DB_PATH", str(db_path))
    config.get_settings.cache_clear()
    st.cache_resource.clear()
    yield
    config.get_settings.cache_clear()
    st.cache_resource.clear()


def fake_build(monkeypatch, scoring_cfg, *script):
    real = agent_mod.build_agent
    model = ScriptedChatModel(script=list(script))
    monkeypatch.setattr(agent_mod, "build_agent", lambda settings: real(settings, model=model, cfg=scoring_cfg))


def test_sample_question_shows_data_then_commentary(app_env, monkeypatch, scoring_cfg):
    fake_build(monkeypatch, scoring_cfg, call("rank_airports", region="new_england", top_n=3),
               AIMessage("Scores are relative to the peer set."))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "2025-01..2025-12" in at.sidebar.markdown[0].value
    next(b for b in at.sidebar.button if "New England" in b.label).click().run()
    assert not at.exception
    texts = [m.value for m in at.main.markdown]
    data, commentary = texts.index("##### Data (from tools)"), texts.index("##### Commentary (model)")
    assert data < commentary and texts[commentary + 1] == "Scores are relative to the peer set."
    assert any("| Rank | Airport |" in t for t in texts[data:commentary])
    assert [e.label for e in at.expander] == ["Excluded airports (1)", "Tools called (1)"]


def test_ranking_commentary_with_everything_removed_shows_the_fallback_and_one_note(app_env, monkeypatch,
                                                                                    scoring_cfg):
    fake_build(monkeypatch, scoring_cfg, call("rank_airports", region="new_england", top_n=3),
               AIMessage("BOS leads with a score of 0.9. BOS is first. The order is:"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in at.sidebar.button if "New England" in b.label).click().run()
    assert not at.exception
    texts = [m.value for m in at.main.markdown]
    assert texts[texts.index("##### Commentary (model)") + 1] == scoring_cfg.commentary_fallback
    notes = [c.value for c in at.main.caption if c.value.startswith("Commentary check")]
    assert notes == ["Commentary check: 3 sentence(s) removed (contains a figure; names an airport; "
                     "fragment left after the removals)."]


def test_missing_model_config_shows_a_friendly_error(app_env, monkeypatch):
    def broken(settings):
        raise ConfigError("The language model is not configured: set LLM_MODEL in .env.")
    monkeypatch.setattr(agent_mod, "build_agent", broken)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.error[0].value == "The language model is not configured: set LLM_MODEL in .env."


def test_unreachable_model_shows_one_line_naming_the_setting(app_env, monkeypatch, scoring_cfg):
    import httpx
    monkeypatch.setenv("LLM_BASE_URL", "http://host.docker.internal:11434")
    fake_build(monkeypatch, scoring_cfg)

    def unreachable(self, question, thread_id):
        raise httpx.ConnectError("Connection refused")
    monkeypatch.setattr(agent_mod.AirportAgent, "ask", unreachable)
    at = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in at.sidebar.button if "New England" in b.label).click().run()
    assert not at.exception
    assert at.error[0].value.startswith("Cannot reach the model server at http://host.docker.internal:11434: "
                                        "check LLM_BASE_URL in .env")


# --- Voice input (P8.1). AppTest cannot record audio, so st.audio_input is replaced by a FakeMic and the
# transcription client is stubbed.

class FakeMic:
    """Stands in for st.audio_input like a browser: a recording belongs to the widget on screen (the last key
    the script rendered) and keeps its upload id across reruns; a widget with another key starts empty."""

    def __init__(self, audio):
        self.keys, self.count, self.recording, self.owner = [], 0, None, "voice_0"
        self.record(audio)

    def record(self, audio):
        self.count += 1
        self.owner = self.keys[-1] if self.keys else "voice_0"
        self.recording = SimpleNamespace(file_id=f"rec-{self.count}", getvalue=lambda: audio)

    def __call__(self, label, *, key=None, **kwargs):
        self.keys.append(key)
        return self.recording if key == self.owner else None


def fake_microphone(monkeypatch, audio):
    mic = FakeMic(audio)
    monkeypatch.setattr(st, "audio_input", mic)
    return mic


def test_no_key_hides_the_microphone_with_a_caption(app_env, monkeypatch, scoring_cfg):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    fake_build(monkeypatch, scoring_cfg)
    mic = fake_microphone(monkeypatch, wav(1))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and mic.keys == []
    assert "Voice input is off: set OPENAI_API_KEY (or STT_BASE_URL) in .env." in [c.value for c in at.caption]


def voice_app(monkeypatch, scoring_cfg, transcript, *script, env=()):
    """The app with OpenAI voice on, a scripted model, a stubbed transcription and one recording.

    Returns the AppTest after its first run, the questions the agent was asked, the transcription client and
    the microphone."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for name, value in env:
        monkeypatch.setenv(name, value)
    fake_build(monkeypatch, scoring_cfg, *script)
    asked = []
    real_ask = agent_mod.AirportAgent.ask
    monkeypatch.setattr(agent_mod.AirportAgent, "ask",
                        lambda self, q, t: asked.append(q) or real_ask(self, q, t))
    client = FakeClient(text=transcript)
    monkeypatch.setattr(voice, "_client", lambda settings: client)
    mic = fake_microphone(monkeypatch, wav(2))
    return AppTest.from_file(APP, default_timeout=30).run(), asked, client, mic


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_recording_is_shown_for_review_and_not_sent_until_send(app_env, monkeypatch, scoring_cfg):
    at, asked, client, _ = voice_app(monkeypatch, scoring_cfg, "Which airports in New England are candidates?",
                                  call("rank_airports", region="new_england", top_n=3),
                                  AIMessage("Scores are relative to the peer set."))
    assert not at.exception and len(client.calls) == 1 and asked == []
    assert at.text_area[0].value == "Which airports in New England are candidates?"
    assert [b.label for b in at.button if b.label in ("Send", "Discard")] == ["Send", "Discard"]
    assert not any(m.value.startswith("You said:") for m in at.main.markdown)
    assert at.session_state.voice_seen == "rec-1" and at.session_state.voice_n == 0   # same widget until Send

    at.text_area[0].input("Which New England airports are strong candidates?")
    button(at, "Send").click().run()
    assert not at.exception and len(client.calls) == 1
    assert asked == ["Which New England airports are strong candidates?"]      # the edited text, once
    texts = [m.value for m in at.main.markdown]
    assert "You said: Which New England airports are strong candidates?" in texts
    assert "##### Data (from tools)" in texts
    assert not at.text_area                    # the review box is gone after sending
    assert at.session_state.voice_n == 1       # and the microphone is a fresh widget
    at.run()
    assert len(asked) == 1 and len(client.calls) == 1


def test_discard_clears_the_draft_and_resets_the_microphone(app_env, monkeypatch, scoring_cfg):
    at, asked, client, _ = voice_app(monkeypatch, scoring_cfg, "Compare LA and Santa Ana congestion.")
    button(at, "Discard").click().run()
    assert not at.exception and asked == [] and not at.text_area
    assert at.session_state.voice_n == 1
    assert "voice_draft" not in at.session_state and "voice_sent" not in at.session_state
    assert not any(m.value.startswith("You said:") for m in at.main.markdown)
    at.run()
    assert len(client.calls) == 1 and not at.text_area and asked == []


def test_prompt_echo_shows_a_short_message_and_sends_nothing(app_env, monkeypatch, scoring_cfg):
    hint = "Questions about LAX and SNA."
    at, asked, _, _ = voice_app(monkeypatch, scoring_cfg, hint, env=[("STT_PROMPT", hint)])
    assert not at.exception and asked == [] and not at.text_area
    assert [w.value for w in at.warning] == [voice.NO_SPEECH]


def test_silent_recording_after_a_chat_and_a_pending_transcript_shows_only_the_message(app_env, monkeypatch,
                                                                                       scoring_cfg):
    # Manual test: with messages on screen, a new silent recording left the old transcript and review box.
    at, asked, client, mic = voice_app(monkeypatch, scoring_cfg, "Which airports in New England are candidates?",
                                       call("rank_airports", region="new_england", top_n=3),
                                       AIMessage("Scores are relative to the peer set."))
    button(at, "Send").click().run()
    client.text = "Compare LA and Santa Ana congestion."
    mic.record(wav(2))
    at.run()
    assert [t.value for t in at.text_area] == ["Compare LA and Santa Ana congestion."]   # pending transcript

    client.text = ""
    mic.record(wav(3))
    at.run()
    assert not at.exception and len(client.calls) == 3
    assert [w.value for w in at.warning] == [voice.NO_SPEECH]
    assert not at.text_area and not [b for b in at.button if b.label in ("Send", "Discard")]
    assert "voice_draft" not in at.session_state and asked == ["Which airports in New England are candidates?"]
    assert "You said: Which airports in New England are candidates?" in [m.value for m in at.main.markdown]
    at.run()                                   # a rerun with the same recording does not transcribe it again
    assert len(client.calls) == 3 and not at.text_area and len(asked) == 1


def test_a_new_transcript_replaces_the_pending_one(app_env, monkeypatch, scoring_cfg):
    at, asked, client, mic = voice_app(monkeypatch, scoring_cfg, "Compare LA and Santa Ana congestion.")
    at.text_area[0].input("Compare LAX and SNA congestion.")     # an edit in the old box does not survive
    client.text = "What is the unmet demand at SFO?"
    mic.record(wav(2))
    at.run()
    assert not at.exception and len(client.calls) == 2 and asked == []
    assert [t.value for t in at.text_area] == ["What is the unmet demand at SFO?"] and not at.warning


def test_long_transcript_hits_the_input_guard(app_env, monkeypatch, scoring_cfg):
    at, asked, _, _ = voice_app(monkeypatch, scoring_cfg, "a spoken question that is too long",
                             env=[("MAX_INPUT_CHARS", "20")])
    button(at, "Send").click().run()
    assert not at.exception
    assert at.error[0].value == "The question is too long (34 characters; the limit is 20)."


def test_transcription_error_is_one_line(app_env, monkeypatch, scoring_cfg):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("STT_MAX_SECONDS", "1")
    fake_build(monkeypatch, scoring_cfg)
    fake_microphone(monkeypatch, wav(3))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [e.value for e in at.error] == ["The recording is 3 s long; the limit is 1 s (STT_MAX_SECONDS in .env)."]
    assert not any(m.value.startswith("You said:") for m in at.main.markdown)



def test_an_unexpected_voice_failure_is_logged_and_shown_as_one_line(app_env, monkeypatch, scoring_cfg, caplog):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_build(monkeypatch, scoring_cfg)
    fake_microphone(monkeypatch, wav(1))

    def broken(*args, **kwargs):
        raise KeyError("voice_draft")
    monkeypatch.setattr(voice, "transcribe", broken)
    with caplog.at_level(logging.ERROR):
        at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [e.value for e in at.error] == ["Transcription failed (KeyError): see the app logs for details."]
    assert "Voice input failed" in caplog.text and "KeyError: 'voice_draft'" in caplog.text
