"""Golden tests (P3.7): the four exam questions, and the edge cases, through the agent's tools.

No model: each test calls the LangChain tools the agent uses (`make_tools`, JSON in and out) on a
small fixture DB (tests/conftest.py: T-100 2024-2025, On-Time 2025, SFO routes and delay causes).
Expected values are worked out from the fixture by hand in the comments. Runs in CI.
"""
import json
import sqlite3

import pytest

from src.tools import make_tools
from tests.conftest import seed_analysis_db, seed_ontime, seed_sfo


@pytest.fixture
def tools(tmp_path, scoring_cfg):
    path = tmp_path / "golden.db"
    with sqlite3.connect(path) as conn:
        seed_analysis_db(conn)
        seed_ontime(conn)
        seed_sfo(conn)
    conn.close()
    by_name = {t.name: t for t in make_tools(lambda: sqlite3.connect(path), scoring_cfg)}

    def ask(name: str, **args) -> dict:
        result = by_name[name].invoke(args)
        json.dumps(result)  # what the model receives must be JSON
        return result
    return ask


def values(result: dict) -> dict:
    return {f["label"]: f["value"] for f in result["facts"]}


def required(result: dict) -> set[str]:
    return {f["label"] for f in result["facts"] if f["required"]}


# --- Q1: New England expansion candidates ------------------------------------------------------

def test_q1_new_england_ranking(tools):
    r = tools("rank_airports", region="new_england")
    assert r["headline"] == ("new_england terminal-expansion ranking by score: 1 BOS, 2 PWM, 3 BTV "
                             "(3 ranked, 1 excluded).")
    assert [(x["iata"], x["score"]) for x in r["data"]["ranked"]] == [("BOS", 0.738), ("PWM", 0.537), ("BTV", 0.291)]
    # HYA: 2 departures a month, round(5 x 1.1) = 6 passengers x 12 months, below the test floor of 1,000
    assert r["data"]["excluded"] == [{"iata": "HYA", "name": "Cape Cod Gateway Airport",
                                      "reason": "72 passenger-service passengers, below the 1,000 floor"}]
    # BTV has 600 On-Time departures, under min_ontime_flights: scored without congestion
    assert r["data"]["ranked"][2]["missing"] == ["congestion"]
    assert values(r)["Weights used (renormalized without demand_context)"] == (
        "utilization 0.278, congestion 0.278, scale 0.167, growth 0.167, long_haul 0.111")
    assert values(r)["Stability of the top airport"] == "BOS first in 20 of 20 runs"
    assert [(x["stability"]["in_top_n_pct"], x["stability"]["min_rank"], x["stability"]["max_rank"])
            for x in r["data"]["ranked"]] == [(100.0, 1, 1), (100.0, 2, 2), (100.0, 3, 3)]
    assert {"Data period", "Confidence", "Weights used (renormalized without demand_context)"} <= required(r)
    assert r["confidence"] == "medium"
    # drivers: why each airport ranks where it does. BOS ties congestion and scale at 100 (congestion is
    # listed first in the weights); PWM has the highest load factor but the lowest congestion. Every
    # airport grows 10%, so growth sits on the tie floor (percentile 50 for all) and is never BTV's
    # strongest; neither is long haul (BTV has none, a floor shared with PWM); utilization is next (22)
    assert [values(r)[f"Drivers: {c}"] for c in ("BOS", "PWM", "BTV")] == [
        "strongest: congestion, percentile 100; weakest: utilization (load factor), percentile 44",
        "strongest: utilization (load factor), percentile 100; weakest: congestion, percentile 17",
        "strongest: utilization (load factor), percentile 22; "
        "weakest: scale (annual passengers), percentile 17"]


# --- Q2: LA vs Santa Ana congestion ------------------------------------------------------------

def test_q2_la_vs_santa_ana(tools):
    r = tools("compare_airports", a="LA", b="Santa Ana")
    # LAX 250 of 1,000 delayed (25.3% of 990 reporting), SNA 40 of 198 (20.2%); taxi-out 18 vs 16 min;
    # NAS 2.0 vs 1.5 min; cancellations 10 of 1,000 and 2 of 200 are both 1.0%: no meaningful difference
    assert r["headline"] == (
        "LAX vs SNA, 2025-01..2025-12 (95% margins on On-Time figures): delayed departures: LAX higher "
        "(LAX 25.3%, SNA 20.2%); taxi-out: LAX higher (LAX 18.0 min, SNA 16.0 min); NAS delay per arrival: "
        "LAX higher (LAX 2.0 min, SNA 1.5 min); no meaningful difference in cancellations (not scored) "
        "(LAX 1.0%, SNA 1.0%). Overall: LAX is higher on every measure with a meaningful difference. "
        "Note: LAX has 10.0 times SNA's departures (passenger service): rates compare per flight, but the "
        "airports differ in scale.")
    assert r["assumptions"][0] == "'LA' = LAX (configured default; other candidates: BUR, LGB, ONT, SNA)"
    v = values(r)
    assert v["LAX: Departures delayed >= 15 min"] == "25.3% ± 0.78 pts (11,880 departures)"
    assert v["SNA: Departures delayed >= 15 min"] == "20.2% ± 1.61 pts (2,376 departures)"
    assert v["Verdict: Cancelled departures (shown, not scored)"] == "no meaningful difference"
    assert {"LAX: Departures delayed >= 15 min", "SNA: Departures delayed >= 15 min"} <= required(r)


# --- Q3: long-haul share out of Anchorage ------------------------------------------------------

def test_q3_anchorage_long_haul_both_bases_with_counts(tools):
    r = tools("airport_metrics", airport="Anchorage", topic="long_haul")
    # per month: SEA 60 pax (1,448 mi), ORD 10 pax + 50 cargo (2,846 mi), HNL 2 combi (2,781 mi)
    # all flights: 62 of 122 a month; passenger service (pax + combi): 12 of 72
    assert r["headline"] == ("50.8% of all departures from ANC (incl. cargo) are >= 2,500 mi (744 of 1,464); "
                             "passenger service: 16.7% (144 of 864).")
    v = values(r)
    assert v["Long-haul share >= 2,500 mi, all flights (including cargo)"] == "50.8%"
    assert v["Long-haul share >= 2,500 mi, passenger service"] == "16.7%"
    assert v["Departures >= 2,500 mi, all flights"] == "744 of 1,464"
    assert v["Departures >= 2,500 mi, passenger service"] == "144 of 864"
    assert v["Sensitivity: long-haul share >= 3,000 mi"] == "all flights 0.0%, passenger service 0.0%"
    assert {"Long-haul threshold", "Long-haul share >= 2,500 mi, all flights (including cargo)",
            "Long-haul share >= 2,500 mi, passenger service", "Sensitivity: long-haul share >= 1,500 mi",
            "Sensitivity: long-haul share >= 3,000 mi"} <= required(r)


# --- Q4: SFO unmet demand and why ---------------------------------------------------------------

def test_q4_sfo_demand_gap_is_a_proxy_with_confidence_at_most_medium(tools):
    r = tools("explain_demand_gap", airport="SFO")
    assert r["headline"] == (
        "PROXY, not a measurement (unmet demand is not observable in this data): SFO load factor was above the "
        "80% target in 3 of 12 months in 2025-01..2025-12 (peak 2025-06 81.7%); spill proxy 450 passengers above "
        "the 80% load level (excess over 80% of seats per route and month; range 600 at 75% to 300 at 85%).")
    assert r["confidence"] in ("medium", "low")
    assert values(r)["Unmet demand"] == "not observable in this data; the figures below are a proxy"
    assert r["data"]["why"][0].startswith("Airport-wide load factor was above the 80% target in 3 of 12 months")
    assert r["data"]["why"][2] == "The largest delay cause on arrivals is NAS (50.0% of delay minutes)."


# --- edge cases -----------------------------------------------------------------------------

def test_unknown_airport(tools):
    r = tools("airport_metrics", airport="Atlantis")
    assert r["data"] is None
    assert r["headline"] == ("No airport with passenger service matches 'Atlantis'. Try the 3-letter IATA code "
                             "(e.g. SFO) or the city the airport serves.")
    assert values(r)["Match"] == "no match for 'Atlantis'"


def test_unsupported_region(tools):
    r = tools("rank_airports", region="California")
    assert r["data"] is None
    assert r["headline"].startswith("Region 'California' is not configured. Configured regions: new_england "
                                    "(MA, CT, RI, NH, VT, ME). Use region 'national'")
    assert r["candidates"] == ["new_england", "national"]


def test_scenario_weights(tools):
    r = tools("rank_airports", region="new_england", weights={"growth": 0.3})
    assert r["data"]["mode"] == "scenario"
    assert r["headline"] == ("Scenario weights: new_england terminal-expansion ranking by score: "
                             "1 BOS (base 1), 2 PWM (base 2), 3 BTV (base 3) (3 ranked, 1 excluded).")
    # growth 0.30 instead of 0.15, the rest as configured (total 1.05 without demand_context), renormalized
    assert values(r)["Scenario weights used (renormalized without demand_context)"] == (
        "utilization 0.238, congestion 0.238, scale 0.143, growth 0.286, long_haul 0.095")


def test_ambiguity_is_returned_not_guessed(tools):
    for r in (tools("airport_metrics", airport="Chicago"), tools("compare_airports", a="Chicago", b="LAX"),
              tools("explain_demand_gap", airport="Chicago")):
        assert r["data"] is None
        assert r["headline"] == ("'Chicago' matches 2 airports: best match ORD (Chicago O'Hare International "
                                 "Airport), alternatives MDW.")
        assert r["candidates"] == ["ORD", "MDW"]
    la = tools("resolve_airport", text="LA")
    assert la["data"]["ambiguous"] is True and la["data"]["default"] == "LAX"
