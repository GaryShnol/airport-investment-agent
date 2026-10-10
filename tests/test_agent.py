"""Agent tests with a scripted fake chat model and a tiny SQLite DB (no LLM, no network)."""
import json
import sqlite3
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src import agent as agent_mod
from src.agent import FACTS_FOOTER, InputRejected, build_agent, ensure_facts, missing_facts, strip_reasoning
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


# --- ensure_facts: required facts are guaranteed by code -----------------------------------



def f(label, value, required=True):
    return {"label": label, "value": value, "required": required}


ANC = {"facts": [
    f("Long-haul threshold", "2,500 mi"),
    f("Long-haul share >= 2,500 mi, all flights (including cargo)", "43.2%"),
    f("Long-haul share >= 2,500 mi, passenger service", "11.5%"),
    f("Sensitivity: long-haul share >= 1,500 mi", "all flights 49.1%, passenger service 19.3%"),
    f("Sensitivity: long-haul share >= 3,000 mi", "all flights 31.0%, passenger service 4.3%"),
    f("Passengers (passenger service)", "2,714,358", required=False),
    f("Data period", "2025-01..2025-12"),
    f("Confidence", "medium"),
]}
ANC_FULL = ("For 2025-01..2025-12, at 2,500 mi, 43.2% of all departures from ANC are long haul "
            "(cargo included) and 11.5% of passenger-service departures. Sensitivity: at 1,500 mi "
            "49.1% / 19.3%, at 3,000 mi 31.0% / 4.3%. Confidence: medium.")


def test_answer_with_all_required_facts_gets_no_footer():
    assert ensure_facts(ANC_FULL, [ANC]) == ANC_FULL


def test_missing_passenger_service_share_gets_exactly_that_fact():
    answer = ANC_FULL.replace(" and 11.5% of passenger-service departures", "")
    out = ensure_facts(answer, [ANC])
    assert out == f"{answer}\n\n{FACTS_FOOTER}\n- Long-haul share >= 2,500 mi, passenger service: 11.5%"


def test_real_model_answer_gets_the_missing_facts():
    # what lfm2.5:8b answered in the first acceptance run (non-breaking hyphens in the period)
    answer = ("According to the 2025‑01..2025‑12 data, at a 2500‑mile threshold, 43.2 % of departures "
              "from Ted Stevens Anchorage International Airport are long‑haul (including cargo). "
              "Confidence in the figure is medium.")
    labels = [m["label"] for m in missing_facts(answer, [ANC])]
    assert labels == ["Long-haul share >= 2,500 mi, passenger service",
                      "Sensitivity: long-haul share >= 1,500 mi", "Sensitivity: long-haul share >= 3,000 mi"]


@pytest.mark.parametrize("written", ["43.2%", "43.2 %", "43,2%", "43%", "43.20%"])
def test_rounding_and_decimal_comma_tolerance(written):
    env = {"facts": [f("share", "43.2%")]}
    assert missing_facts(f"The share is {written}.", [env]) == []


@pytest.mark.parametrize("written", ["44%", "4.32%", "143.2%", "43.7%"])
def test_wrong_numbers_are_not_accepted(written):
    env = {"facts": [f("share", "43.2%")]}
    assert len(missing_facts(f"The share is {written}.", [env])) == 1


def test_small_values_need_the_exact_number_but_dates_never_count():
    env = {"facts": [f("passenger service", "11.5%"), f("small", "4.3%")]}
    # 11.5 may round to 12, but the 12 in "2025-12" is a date, not a figure
    assert [m["label"] for m in missing_facts("Data 2025-01..2025-12, 4%.", [env])] == ["passenger service", "small"]
    assert missing_facts("Shares: 12% and 4.3%.", [env]) == []


def test_weights_mix_up_footer_lists_the_weights_used():
    env = {"facts": [f("Weights used (renormalized without congestion, demand_context)",
                       "utilization 0.385, scale 0.231, growth 0.231, long_haul 0.154"),
                     f("Data period", "2025-01..2025-12"), f("Confidence", "medium")]}
    mixed = ("Data 2025-01..2025-12, confidence medium. Ranking weights: utilization 0.385, congestion 0.25, "
             "scale 0.15, growth 0.15, long-haul 0.10, demand-context 0.10.")
    out = ensure_facts(mixed, [env])
    assert out.endswith(f"{FACTS_FOOTER}\n- Weights used (renormalized without congestion, demand_context): "
                        "utilization 0.385, scale 0.231, growth 0.231, long_haul 0.154")
    # weights phrased as percentages are accepted
    assert missing_facts("Weights: utilization 38.5%, scale 23.1%, growth 23.1%, long haul 15.4%; "
                         "2025, medium.", [env]) == []


def test_later_fact_with_the_same_label_wins():
    resolve = {"facts": [f("Confidence", "high")]}
    metrics = {"facts": [f("Confidence", "medium")]}
    assert missing_facts("Confidence: medium.", [resolve, metrics]) == []


def test_ensure_facts_is_deterministic_and_ignores_non_envelopes():
    answer = "Nothing useful."
    envelopes = [ANC, "not json", {"data": None, "facts": [], "error": "x"}]
    assert ensure_facts(answer, envelopes) == ensure_facts(answer, envelopes)
    assert ensure_facts("Hi", []) == "Hi"


def test_reply_keeps_commentary_apart_from_the_footer(settings, connect, scoring_cfg):
    agent, _ = make(settings, connect, scoring_cfg,
                    call("airport_metrics", airport="ANC", topic="demand"),
                    AIMessage("ANC is busy."))
    reply = agent.ask("What is the load factor at Anchorage?", "t-topic")
    assert reply.commentary == "ANC is busy."
    assert [f["label"] for f in reply.missing_facts] == [
        "Passengers (passenger service)", "Load factor (passenger service, a demand-pressure proxy)",
        "Data period", "Confidence"]
    assert reply.text.startswith("ANC is busy.\n\n" + FACTS_FOOTER)


# --- headline (P2.9c) ------------------------------------------------------------------------

def test_commentary_drops_a_verbatim_headline(settings, connect, scoring_cfg):
    headline = ("50.8% of all departures from ANC (incl. cargo) are >= 2,500 mi (744 of 1,464); "
                "passenger service: 16.7% (144 of 864).")
    agent, _ = make(settings, connect, scoring_cfg,
                    call("airport_metrics", airport="ANC", topic="long_haul"),
                    AIMessage(f"{headline.upper()}\n\nCargo flights to Asia explain the gap."))
    reply = agent.ask("What share of ANC flights are long haul?", "t-headline")
    assert reply.tool_results[0]["headline"] == headline
    assert reply.commentary == "Cargo flights to Asia explain the gap."


def test_drop_restated_headlines_keeps_paraphrases():
    env = {"headline": "BOS leads."}
    assert agent_mod.drop_restated_headlines("Boston is first. > BOS  leads.", [env]) == "Boston is first."
    assert agent_mod.drop_restated_headlines("BOS clearly leads.", [env, "x"]) == "BOS clearly leads."


# --- agent robustness (P2.14) ------------------------------------------------------------------

def test_tool_messages_are_kept_across_turns_in_compact_form(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg,
                        call("rank_airports", region="new_england"), AIMessage("BOS leads."),
                        AIMessage("PWM has the highest load factor in the card."))
    first = agent.ask("Rank New England airports for expansion", "t-history")
    agent.ask("Which of these has the highest load factor?", "t-history")
    tool_messages = [m for m in model.seen[2] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1  # the first turn's result reaches the second turn's prompt
    card = json.loads(tool_messages[0].content)
    assert set(card) == {"headline", "facts", "data_period", "confidence"}  # compact: no raw data
    assert card["headline"] == first.tool_results[0]["headline"]
    assert any(f["label"].startswith("#1 ") and "utilization (load factor)" in f["value"] for f in card["facts"])
    # within the turn that called it, the model sees the full envelope
    assert "data" in json.loads(next(m for m in model.seen[1] if isinstance(m, ToolMessage)).content)
    # the stored history keeps the full envelope
    stored = agent.graph.get_state({"configurable": {"thread_id": "t-history"}}).values["messages"]
    assert "data" in json.loads(next(m for m in stored if isinstance(m, ToolMessage)).content)


def test_compact_envelope_leaves_non_envelopes_alone():
    assert agent_mod.compact_envelope("not json") == "not json"
    env = {"headline": "h", "facts": [], "data": {"big": 1}, "error": None, "confidence": "low"}
    assert json.loads(agent_mod.compact_envelope(json.dumps(env))) == {"headline": "h", "confidence": "low"}


def test_empty_reply_is_retried_once_with_a_nudge(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg, AIMessage(""), AIMessage("Hello, ask me about airports."))
    reply = agent.ask("Hi", "t-empty")
    assert reply.commentary == "Hello, ask me about airports." and reply.fallback is False
    assert model.seen[1][-1].content == agent_mod.RETRY_NUDGE
    stored = agent.graph.get_state({"configurable": {"thread_id": "t-empty"}}).values["messages"]
    assert agent_mod.RETRY_NUDGE not in [m.content for m in stored]  # the nudge is never stored


def test_reasoning_only_reply_counts_as_empty(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg, AIMessage("<think>hmm</think>"), AIMessage("Answer."))
    assert agent.ask("Hi", "t-think").commentary == "Answer." and len(model.seen) == 2


def test_still_empty_after_retry_gives_a_deterministic_message(settings, connect, scoring_cfg):
    agent, model = make(settings, connect, scoring_cfg, AIMessage(""), AIMessage(""))
    reply = agent.ask("Hi", "t-empty2")
    assert reply.commentary == agent_mod.EMPTY_NO_DATA and reply.fallback is True and len(model.seen) == 2

    agent, _ = make(settings, connect, scoring_cfg, call("data_status"), AIMessage(""), AIMessage(" "))
    reply = agent.ask("What data do you have?", "t-empty3")
    assert reply.commentary == agent_mod.EMPTY_WITH_DATA and reply.tool_results


def test_reply_that_only_repeats_the_headline_is_not_blank(settings, connect, scoring_cfg):
    agent, _ = make(settings, connect, scoring_cfg, call("data_status"),
                    AIMessage("Data loaded for 2025-01..2025-12 (12 of 12 months); freshness unknown."))
    reply = agent.ask("What data do you have?", "t-only-headline")
    assert reply.commentary == agent_mod.ONLY_HEADLINE and reply.fallback is True


def test_context_warning_near_num_ctx(tmp_path, connect, scoring_cfg, caplog):
    tight = Settings.from_env({"DB_PATH": str(tmp_path / "agent.db"), "OLLAMA_NUM_CTX": "500"})
    agent, _ = make(tight, connect, scoring_cfg, call("rank_airports", region="new_england"), AIMessage("ok"))
    with caplog.at_level("WARNING", logger="src.agent"):
        agent.ask("Rank New England", "t-ctx")
    assert any("OLLAMA_NUM_CTX=500" in r.getMessage() for r in caplog.records)


def test_no_context_warning_without_num_ctx(settings, connect, scoring_cfg, caplog):
    agent, _ = make(settings, connect, scoring_cfg, call("rank_airports", region="new_england"), AIMessage("ok"))
    with caplog.at_level("WARNING", logger="src.agent"):
        agent.ask("Rank New England", "t-noctx")
    assert not any("OLLAMA_NUM_CTX" in r.getMessage() for r in caplog.records)


def test_prompt_rules():
    for rule in ["do not repeat it", "Use only figures that appear in the card",
                 "No conclusion without supporting data", "always call resolve_airport first",
                 "ask the user which airport they mean"]:
        assert rule in SYSTEM_PROMPT
