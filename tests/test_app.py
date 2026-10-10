"""Smoke test of app.py with Streamlit's AppTest, a scripted fake model and the tiny test DB."""
import sqlite3

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from src import agent as agent_mod
from src import config
from src.config import ROOT, ConfigError
from tests.conftest import seed_analysis_db
from tests.fake_model import ScriptedChatModel, call

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


def test_ranking_commentary_with_only_figures_and_names_shows_no_commentary_block(app_env, monkeypatch, scoring_cfg):
    fake_build(monkeypatch, scoring_cfg, call("rank_airports", region="new_england", top_n=3),
               AIMessage("BOS leads with a score of 0.9."))
    at = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in at.sidebar.button if "New England" in b.label).click().run()
    assert not at.exception
    texts = [m.value for m in at.main.markdown]
    assert "##### Data (from tools)" in texts and "##### Commentary (model)" not in texts


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
