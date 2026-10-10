"""Agent tests with a scripted fake chat model and a tiny SQLite DB (no LLM, no network)."""
import sqlite3
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import agent as agent_mod
from src.agent import InputRejected, build_agent, strip_reasoning
from src.config import ConfigError, Settings
from src.prompts import SYSTEM_PROMPT
from tests.conftest import seed_analysis_db
from tests.fake_model import ScriptedChatModel, call


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env({"DB_PATH": str(tmp_path / "agent.db"), "MAX_INPUT_CHARS": "200"})


@pytest.fixture
def connect(settings):
    with sqlite3.connect(settings.db_path) as conn:
        seed_analysis_db(conn)
    conn.close()
    return lambda: sqlite3.connect(settings.db_path)


def make(settings, connect, scoring_cfg, *script):
    model = ScriptedChatModel(script=list(script))
    return build_agent(settings, model=model, connect=connect, cfg=scoring_cfg), model


def test_ranking_question_routes_to_rank_airports(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg,
                        call("rank_airports", region="new_england", top_n=3),
                        AIMessage("BOS ranks first (2025-01..2025-12, medium confidence)."))
    reply = agent.ask("Which airports in New England are strong candidates for terminal expansion?", "t1")
    assert [c["name"] for c in reply.tool_calls] == ["rank_airports"]
    envelope = reply.tool_results[0]
    assert envelope["data"]["region"] == "new_england" and envelope["data"]["weights_used"]
    assert isinstance(model.seen[0][0], SystemMessage) and model.seen[0][0].content == SYSTEM_PROMPT
    assert "BOS ranks first" in reply.text


def test_la_triggers_resolve_airport_and_answer_states_definition(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg,
                        call("resolve_airport", text="LA"),
                        AIMessage("I used LA = LAX. Other LA airports: BUR, LGB, ONT, SNA."))
    reply = agent.ask("How busy is LA?", "t2")
    assert reply.tool_calls == [{"name": "resolve_airport", "args": {"text": "LA"}}]
    data = reply.tool_results[0]["data"]
    assert data["ambiguous"] is True and data["default"] == "LAX"
    # the model's second call sees the ambiguity envelope it has to act on
    assert '"ambiguous": true' in model.seen[1][-1].content
    assert "LA = LAX" in reply.text


def test_out_of_scope_question_makes_no_tool_call(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg,
                        AIMessage("Sorry, that is out of scope: I answer questions about US airports."))
    reply = agent.ask("Write me a poem about cats", "t3")
    assert reply.tool_calls == [] and reply.tool_results == []
    assert len(model.seen) == 1 and "out of scope" in reply.text
    assert "out of scope" in SYSTEM_PROMPT


def test_think_block_is_stripped(settings, connect, scoring_cfg):
    agent, _ = make(settings, connect, scoring_cfg,
                    AIMessage("<think>The user wants data, but I will keep this hidden.</think>\nHello!"))
    assert agent.ask("Hi", "t4").text == "Hello!"


@pytest.mark.parametrize("raw, clean", [
    ("<think>a\nb</think>Answer", "Answer"),
    ("hidden reasoning</think>\n\nAnswer", "Answer"),      # opening tag missing
    ("<THINK>x</THINK> A <think>y</think>B", "A B"),
    ("<think>never closed", ""),
    ("No reasoning here", "No reasoning here"),
])
def test_strip_reasoning(raw, clean):
    assert strip_reasoning(raw) == clean


@pytest.mark.parametrize("question", ["", "   ", "x" * 201])
def test_empty_and_too_long_input_rejected_before_the_model(settings, connect, scoring_cfg, question):
    agent, model = make(settings, connect, scoring_cfg)
    with pytest.raises(InputRejected):
        agent.ask(question, "t5")
    assert model.seen == []


def test_input_limit_comes_from_settings(settings, connect, scoring_cfg):
    agent, _ = make(settings, connect, scoring_cfg, AIMessage("ok"))
    assert agent.max_input_chars == 200
    assert agent.ask("x" * 200, "t6").text == "ok"


def test_second_turn_with_same_thread_sees_the_first(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg,
                        call("airport_metrics", airport="ANC"),
                        AIMessage("ANC: 50.8% of all departures are long haul."),
                        AIMessage("That figure includes cargo flights."),
                        AIMessage("New thread, no history."))
    agent.ask("What share of flights from Anchorage are long haul?", "same")
    reply = agent.ask("Does that include cargo?", "same")
    second_prompt = model.seen[2]
    humans = [m.content for m in second_prompt if isinstance(m, HumanMessage)]
    assert humans == ["What share of flights from Anchorage are long haul?", "Does that include cargo?"]
    assert any("50.8%" in m.content for m in second_prompt if isinstance(m, AIMessage))
    assert reply.tool_calls == []  # only this turn's calls are reported

    agent.ask("Hello", "other")
    assert [m.content for m in model.seen[3] if isinstance(m, HumanMessage)] == ["Hello"]


def test_missing_llm_key_gives_a_clear_config_error(settings, connect, scoring_cfg):
    openai = replace(settings, llm_provider="openai", llm_model="gpt-4o-mini", openai_api_key=None, llm_base_url=None)
    with pytest.raises(ConfigError, match="not configured.*OPENAI_API_KEY"):
        build_agent(openai, connect=connect, cfg=scoring_cfg)


def test_missing_database_gives_a_clear_config_error(settings, scoring_cfg):
    with pytest.raises(ConfigError, match="ingest"):
        build_agent(settings, model=ScriptedChatModel(script=[]), cfg=scoring_cfg)


def test_cli_prints_tools_and_answer(monkeypatch, capsys, settings, connect, scoring_cfg):
    agent, _ = make(settings, connect, scoring_cfg, call("data_status"), AIMessage("<think>x</think>Data: 2025."))
    monkeypatch.setattr(agent_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_mod, "build_agent", lambda s: agent)
    assert agent_mod.main(["What data do you have?"]) == 0
    out = capsys.readouterr().out
    assert "- data_status()" in out and "Data: 2025." in out and "<think>" not in out
