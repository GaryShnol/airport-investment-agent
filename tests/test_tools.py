"""Tool tests on an in-memory SQLite DB (tiny synthetic data, never data/raw)."""
import json
import sqlite3

import pytest

from src import tools
from tests.conftest import seed_analysis_db

ENVELOPE_KEYS = {"facts", "data", "data_period", "freshness", "assumptions", "confidence"}


def facts(result):
    """label -> {value, required}"""
    return {f["label"]: {"value": f["value"], "required": f["required"]} for f in result["facts"]}


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    seed_analysis_db(conn)
    yield conn
    conn.close()


def check_envelope(result):
    assert ENVELOPE_KEYS <= set(result)
    json.dumps(result)  # JSON-serializable, no numpy types
    return result


# --- envelope and freshness --------------------------------------------------------------

def test_data_status(db, scoring_cfg):
    r = check_envelope(tools.data_status(db, scoring_cfg))
    assert r["data_period"] == "2025-01..2025-12"
    assert r["freshness"] == "unknown"  # ingest --check never ran
    assert r["data"]["months_in_window"] == 12
    assert "congestion" in r["data"]["score_components_not_built"]
    assert r["data"]["regions"] == {"new_england": ["MA", "CT", "RI", "NH", "VT", "ME"]}


@pytest.mark.parametrize("upstream, expected", [(None, "unknown"), ("2025-12", "current"), ("2026-07", "behind")])
def test_freshness_from_meta(db, upstream, expected):
    db.execute("UPDATE meta SET upstream_latest = ?", (upstream,))
    assert tools.freshness(db) == expected


# --- resolve_airport ---------------------------------------------------------------------

def test_resolve_exact_iata(db, scoring_cfg):
    d = check_envelope(tools.resolve_airport(db, scoring_cfg, "anc"))["data"]
    assert (d["iata"], d["ambiguous"]) == ("ANC", False)


def test_resolve_la_is_ambiguous_with_default(db, scoring_cfg):
    r = check_envelope(tools.resolve_airport(db, scoring_cfg, "LA"))
    assert r["data"]["ambiguous"] is True
    assert r["data"]["default"] == "LAX"
    assert [c["iata"] for c in r["data"]["candidates"]] == ["LAX", "BUR", "LGB", "ONT", "SNA"]
    assert any("LA = LAX" in a for a in r["assumptions"])


def test_resolve_by_name_ignores_airports_without_service(db, scoring_cfg):
    # MRI is also in Anchorage but has no passenger service; ANC matches on its name first.
    assert tools.resolve_airport(db, scoring_cfg, "Anchorage")["data"]["iata"] == "ANC"


def test_resolve_by_municipality(db, scoring_cfg):
    assert tools.resolve_airport(db, scoring_cfg, "santa ana")["data"]["iata"] == "SNA"


def test_resolve_several_matches_is_ambiguous(db, scoring_cfg):
    d = tools.resolve_airport(db, scoring_cfg, "Chicago")["data"]
    assert d["ambiguous"] is True and d["default"] is None
    assert [c["iata"] for c in d["candidates"]] == ["ORD", "MDW"]  # largest first


@pytest.mark.parametrize("text", ["", "Atlantis", "100%"])
def test_resolve_unknown_returns_error_envelope(db, scoring_cfg, text):
    r = check_envelope(tools.resolve_airport(db, scoring_cfg, text))
    assert r["data"] is None and r["error"]


# --- airport_metrics ---------------------------------------------------------------------

def test_airport_metrics_long_haul_both_definitions(db, scoring_cfg):
    r = check_envelope(tools.airport_metrics(db, scoring_cfg, "ANC"))
    d = r["data"]
    lh = d["long_haul"]
    assert lh["threshold_mi"] == 2500
    # per month: all flights 122 departures, 62 >= 2,500 mi; passenger service 72, 12 >= 2,500 mi
    assert lh["all_flights"]["share_pct"] == round(62 / 122 * 100, 1)
    assert lh["passenger_service"]["share_pct"] == round(12 / 72 * 100, 1)
    assert [s["threshold_mi"] for s in lh["sensitivity"]] == [1500, 2500, 3000]
    assert lh["sensitivity"][-1] == {"threshold_mi": 3000, "all_flights_pct": 0.0, "passenger_service_pct": 0.0}
    assert d["load_factor_pct"] == round((7700 + 1760 + 165) / 11200 * 100, 1)  # 2025 passengers / seats
    assert d["growth_pct"] == pytest.approx(10.0, abs=0.1)
    assert r["data_period"] == "2025-01..2025-12"
    assert any("cargo" in a for a in r["assumptions"])


def required(result):
    return {f["label"]: f["value"] for f in result["facts"] if f["required"]}


def test_airport_metrics_required_facts(db, scoring_cfg):
    f = facts(tools.airport_metrics(db, scoring_cfg, "ANC", focus="long_haul"))
    required = {k: v["value"] for k, v in f.items() if v["required"]}
    assert required == {
        "Long-haul threshold": "2,500 mi",
        "Long-haul share >= 2,500 mi, all flights (including cargo)": f"{round(62 / 122 * 100, 1)}%",
        "Long-haul share >= 2,500 mi, passenger service": f"{round(12 / 72 * 100, 1)}%",
        "Sensitivity: long-haul share >= 1,500 mi": "all flights 50.8%, passenger service 16.7%",
        "Sensitivity: long-haul share >= 3,000 mi": "all flights 0.0%, passenger service 0.0%",
        "Data period": "2025-01..2025-12",
        "Confidence": "medium",
    }
    assert f["Passengers (passenger service)"] == {"value": "115,500", "required": False}


def test_airport_metrics_focus_load_factor_does_not_require_long_haul(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "ANC", focus="load_factor")
    assert required(r) == {"Load factor (passenger service, a demand-pressure proxy)": "85.9%",
                           "Data period": "2025-01..2025-12", "Confidence": "medium"}
    assert facts(r)["Long-haul threshold"] == {"value": "2,500 mi", "required": False}  # still shown


def test_airport_metrics_focus_growth_and_all(db, scoring_cfg):
    growth = required(tools.airport_metrics(db, scoring_cfg, "ANC", focus="growth"))
    assert set(growth) == {"Passenger growth vs 2024-01..2024-12", "Data period", "Confidence"}
    every = required(tools.airport_metrics(db, scoring_cfg, "ANC"))  # default focus = all
    assert set(every) == set(required(tools.airport_metrics(db, scoring_cfg, "ANC", focus="long_haul"))) | set(
        growth) | {"Load factor (passenger service, a demand-pressure proxy)"}


def test_airport_metrics_unknown_focus_is_an_error(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "ANC", focus="delays")
    assert r["data"] is None and "focus must be one of" in r["error"]


def test_error_envelope_has_no_facts(db, scoring_cfg):
    assert tools.rank_airports(db, scoring_cfg, "Texas")["facts"] == []


def test_airport_metrics_custom_threshold_is_added_to_sensitivity(db, scoring_cfg):
    lh = tools.airport_metrics(db, scoring_cfg, "Anchorage", threshold_mi=2000)["data"]["long_haul"]
    assert lh["threshold_mi"] == 2000
    assert [s["threshold_mi"] for s in lh["sensitivity"]] == [1500, 2000, 2500, 3000]


def test_airport_metrics_ambiguous_returns_candidates(db, scoring_cfg):
    r = check_envelope(tools.airport_metrics(db, scoring_cfg, "LA"))
    assert r["data"] is None
    assert r["candidates"] == ["LAX", "BUR", "LGB", "ONT", "SNA"] and r["default"] == "LAX"


def test_airport_metrics_without_service(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "MRI")
    assert r["data"] is None and "no passenger service" in r["error"]


# --- rank_airports -----------------------------------------------------------------------

def test_rank_airports_region(db, scoring_cfg):
    r = check_envelope(tools.rank_airports(db, scoring_cfg, "New England", top_n=2))
    d = r["data"]
    assert d["region"] == "new_england"
    assert [x["rank"] for x in d["ranked"]] == [1, 2] and d["ranked_total"] == 3
    assert {x["iata"] for x in d["ranked"]} <= {"BOS", "PWM", "BTV"}
    assert d["excluded"] == [e for e in d["excluded"] if e.startswith("HYA")] and d["excluded_total"] == 1
    assert "weights_configured" not in d  # only the weights applied reach the model
    assert set(d["weights_used"]) == {"utilization", "scale", "growth", "long_haul"}
    weights = facts(r)["Weights used (renormalized without congestion, demand_context)"]
    assert weights == {"value": "utilization 0.385, scale 0.231, growth 0.231, long_haul 0.154", "required": True}
    assert {"Data period", "Confidence"} <= {k for k, v in facts(r).items() if v["required"]}
    assert not any(v["required"] for k, v in facts(r).items() if k.startswith("#"))
    assert "scale (annual passengers)" in facts(r)[f"#1 {d['ranked'][0]['iata']} ({d['ranked'][0]['name']})"]["value"]
    assert sum(d["weights_used"].values()) == pytest.approx(1.0, abs=0.01)
    assert set(d["components_unavailable"]) == {"congestion", "demand_context"}
    top = d["ranked"][0]["components"]
    assert set(top) == set(d["weights_used"]) and "percentile" in top["utilization"]
    assert r["confidence"] == "medium"  # renormalized without congestion and demand context


def test_rank_airports_unknown_region(db, scoring_cfg):
    r = check_envelope(tools.rank_airports(db, scoring_cfg, "Texas"))
    assert r["data"] is None and r["candidates"] == ["new_england"]


# --- LangChain wrappers ------------------------------------------------------------------

def test_make_tools_errors_become_envelopes(scoring_cfg):
    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    by_name = {t.name: t for t in tools.make_tools(broken, scoring_cfg)}
    assert set(by_name) == {"data_status", "resolve_airport", "airport_metrics", "rank_airports"}
    r = by_name["airport_metrics"].invoke({"airport": "ANC"})
    assert r["data"] is None and "unable to open" in r["error"]


# --- headline (P2.9c) --------------------------------------------------------------------

def test_every_envelope_has_a_headline(db, scoring_cfg):
    results = [tools.data_status(db, scoring_cfg), tools.resolve_airport(db, scoring_cfg, "LA"),
               tools.resolve_airport(db, scoring_cfg, "Chicago"), tools.resolve_airport(db, scoring_cfg, "Atlantis"),
               tools.airport_metrics(db, scoring_cfg, "BOS"), tools.rank_airports(db, scoring_cfg, "new_england"),
               tools.rank_airports(db, scoring_cfg, "Texas")]
    for r in results:
        assert isinstance(r["headline"], str) and r["headline"]
    assert results[0]["headline"] == "Data loaded for 2025-01..2025-12 (12 of 12 months); freshness unknown."


def test_long_haul_headline_states_definition_both_bases_and_counts(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "ANC", focus="long_haul")
    # per year: all flights 1,464 departures, 744 >= 2,500 mi; passenger service 864, 144
    assert r["headline"] == ("50.8% of all departures from ANC (incl. cargo) are >= 2,500 mi (744 of 1,464); "
                             "passenger service: 16.7% (144 of 864).")
    assert facts(r)["Departures >= 2,500 mi, all flights"] == {"value": "744 of 1,464", "required": False}


def test_metrics_headline_follows_focus(db, scoring_cfg):
    load = tools.airport_metrics(db, scoring_cfg, "ANC", focus="load_factor")["headline"]
    assert load.startswith("ANC load factor (passenger service) is 85.9%") and ">= 2,500" not in load
    growth = tools.airport_metrics(db, scoring_cfg, "ANC", focus="growth")["headline"]
    assert growth.startswith("ANC passengers changed +10.0% in 2025-01..2025-12")


def test_ranking_headline_names_the_top(db, scoring_cfg):
    r = tools.rank_airports(db, scoring_cfg, "new_england")
    top = ", ".join(f"{x['rank']} {x['iata']}" for x in r["data"]["ranked"])
    assert r["headline"] == f"new_england terminal-expansion ranking by score: {top} (3 ranked, 1 excluded)."


# --- rank_airports: sort_by and scenario weights (P2.11) ----------------------------------

def test_rank_default_is_base_mode_with_config_weights(db, scoring_cfg):
    d = tools.rank_airports(db, scoring_cfg, "new_england")["data"]
    assert d["mode"] == "base" and d["sort_by"] == "score"
    assert all("base_rank" not in x and "score_rank" not in x for x in d["ranked"])


@pytest.mark.parametrize("sort_by, column", [("growth", "growth_pct"), ("load_factor", "load_factor_pct"),
                                             ("passengers", "passengers")])
def test_rank_sort_by_metric_is_done_in_code(db, scoring_cfg, sort_by, column):
    r = tools.rank_airports(db, scoring_cfg, "new_england", sort_by=sort_by)
    ranked = r["data"]["ranked"]
    values = [x["sort_value"] for x in ranked]
    assert values == sorted(values, reverse=True)
    assert [x["rank"] for x in ranked] == [1, 2, 3]
    base = {x["iata"]: x["rank"] for x in tools.rank_airports(db, scoring_cfg, "new_england")["data"]["ranked"]}
    assert {x["iata"]: x["score_rank"] for x in ranked} == base  # score rank kept next to the sort order
    assert facts(r)["Sorted by"]["required"] is True
    assert r["data"]["ties"] == []


def test_rank_sort_by_passengers_order(db, scoring_cfg):
    ranked = tools.rank_airports(db, scoring_cfg, "new_england", sort_by="passengers")["data"]["ranked"]
    assert [x["iata"] for x in ranked] == ["BOS", "PWM", "BTV"]  # 2025 passengers: BOS > PWM > BTV


def test_rank_unknown_sort_by_is_an_error(db, scoring_cfg):
    r = tools.rank_airports(db, scoring_cfg, "new_england", sort_by="delays")
    assert r["data"] is None and "sort_by must be one of" in r["error"]


def test_scenario_weights_are_labeled_and_show_base_vs_scenario_rank(db, scoring_cfg):
    r = check_envelope(tools.rank_airports(db, scoring_cfg, "new_england", weights={"growth": 0.6, "scale": 0}))
    d = r["data"]
    assert d["mode"] == "scenario" and r["headline"].startswith("Scenario weights: new_england")
    assert d["weights_requested"] == {"growth": 0.6, "scale": 0.0}
    # merged with config (utilization 0.25, long_haul 0.10), renormalized over built components
    total = 0.25 + 0.6 + 0.10
    assert d["weights_used"] == {"utilization": round(0.25 / total, 3), "scale": 0.0,
                                 "growth": round(0.6 / total, 3), "long_haul": round(0.10 / total, 3)}
    assert d["weights_base"] == tools.rank_airports(db, scoring_cfg, "new_england")["data"]["weights_used"]
    base = {x["iata"]: x["rank"] for x in tools.rank_airports(db, scoring_cfg, "new_england")["data"]["ranked"]}
    assert {x["iata"]: x["base_rank"] for x in d["ranked"]} == base
    f = facts(r)
    assert f["Ranking"]["value"].startswith("scenario") and f["Ranking"]["required"]
    assert any(k.startswith("Scenario weights used") and v["required"] for k, v in f.items())
    assert any(a.startswith("scenario:") for a in r["assumptions"])


def test_scenario_weight_for_unbuilt_component_is_ignored_and_said(db, scoring_cfg):
    r = tools.rank_airports(db, scoring_cfg, "new_england", weights={"congestion": 0.5})
    assert "congestion" not in r["data"]["weights_used"]
    assert any("ignored (component not built): congestion" in a for a in r["assumptions"])


@pytest.mark.parametrize("weights, message", [
    ({"price": 0.2}, "Unknown weight(s) price"),
    ({"growth": -0.1}, "non-negative"),
    ({"growth": "high"}, "non-negative"),
    ({"growth": True}, "non-negative"),
    ({"utilization": 0, "scale": 0, "growth": 0, "long_haul": 0}, "no weight to any available component"),
])
def test_invalid_scenario_weights_are_errors(db, scoring_cfg, weights, message):
    r = check_envelope(tools.rank_airports(db, scoring_cfg, "new_england", weights=weights))
    assert r["data"] is None and message in r["error"]


def test_scenario_does_not_change_config(db, scoring_cfg):
    before = dict(scoring_cfg.weights)
    tools.rank_airports(db, scoring_cfg, "new_england", weights={"growth": 0.9})
    assert scoring_cfg.weights == before


def test_rank_list_is_capped_by_config(db, scoring_cfg):
    from dataclasses import replace
    r = tools.rank_airports(db, replace(scoring_cfg, max_ranked=2), "new_england", top_n=50)
    assert len(r["data"]["ranked"]) == 2 and r["data"]["ranked_total"] == 3
    assert "list capped at 2 airports" in r["assumptions"]


def test_rank_tool_schema_exposes_sort_by_and_weights(scoring_cfg):
    rank = {t.name: t for t in tools.make_tools(lambda: None, scoring_cfg)}["rank_airports"]
    props = rank.args
    assert props["sort_by"]["enum"] == ["score", "growth", "load_factor", "passengers"]
    assert "weights" in props and "growth" in rank.description
