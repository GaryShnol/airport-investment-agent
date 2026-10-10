"""Tool tests on an in-memory SQLite DB (tiny synthetic data, never data/raw)."""
import json
import sqlite3

import pytest

from src import scoring, store, tools
from tests.conftest import seed_analysis_db, seed_census, seed_ontime, seed_sfo

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
    assert "congestion" in r["data"]["score_components_unavailable"]
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
    f = facts(tools.airport_metrics(db, scoring_cfg, "ANC", topic="long_haul"))
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
    assert "Passengers (passenger service)" not in f  # long-haul topic: no demand rows


def test_airport_metrics_topic_demand_shows_no_long_haul_rows(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "ANC", topic="demand")
    assert required(r) == {"Passengers (passenger service)": "115,500",
                           "Load factor (passenger service, a demand-pressure proxy)": "85.9%",
                           "Data period": "2025-01..2025-12", "Confidence": "medium"}
    assert not any("ong-haul" in label or label.startswith("Departures >=") for label in facts(r))
    assert "long_haul" not in r["data"] and r["data"]["topic"] == "demand"
    assert not any("long haul" in a or "cargo" in a for a in r["assumptions"])
    assert ">=" not in r["headline"]


def test_airport_metrics_topic_growth_rows(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "ANC", topic="growth")
    assert list(facts(r)) == ["Airport", "Passengers 2025-01..2025-12 (passenger service)",
                              "Passengers 2024-01..2024-12", "Passenger growth vs 2024-01..2024-12",
                              "Data period", "Confidence"]
    assert set(required(r)) == {"Passenger growth vs 2024-01..2024-12", "Data period", "Confidence"}
    assert "load_factor_pct" not in r["data"] and "long_haul" not in r["data"]


def test_airport_metrics_topic_all_is_the_union(db, scoring_cfg):
    def labels(topic, only_required=False):
        r = tools.airport_metrics(db, scoring_cfg, "ANC", topic=topic)
        return set(required(r) if only_required else facts(r))
    every = labels("all", True)
    assert every == (labels("long_haul", True) | labels("demand", True) | labels("growth", True)
                     | labels("congestion", True))
    # growth's passenger row is folded into the demand row when both are shown
    assert labels("all") == (labels("long_haul") | labels("demand") | labels("growth") | labels("congestion")) - {
        "Passengers 2025-01..2025-12 (passenger service)"}


@pytest.mark.parametrize("topic", ["long_haul", "demand", "growth", "all"])
def test_airport_metrics_always_has_the_data_period(db, scoring_cfg, topic):
    r = tools.airport_metrics(db, scoring_cfg, "BOS", topic=topic)
    assert required(r)["Data period"] == "2025-01..2025-12" and r["data_period"] == "2025-01..2025-12"


def test_airport_metrics_unknown_topic_is_an_error(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "ANC", topic="delays")
    assert r["data"] is None and "topic must be one of" in r["error"]


def test_plain_error_envelope_has_no_facts(db, scoring_cfg):
    assert tools.airport_metrics(db, scoring_cfg, "ANC", topic="delays")["facts"] == []


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
    assert r["data"] is None and r["candidates"] == ["new_england", "national"]


# --- LangChain wrappers ------------------------------------------------------------------

def test_make_tools_errors_become_envelopes(scoring_cfg):
    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    by_name = {t.name: t for t in tools.make_tools(broken, scoring_cfg)}
    assert set(by_name) == {"data_status", "resolve_airport", "airport_metrics", "rank_airports", "compare_airports",
                            "explain_demand_gap"}
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
    r = tools.airport_metrics(db, scoring_cfg, "ANC", topic="long_haul")
    # per year: all flights 1,464 departures, 744 >= 2,500 mi; passenger service 864, 144
    assert r["headline"] == ("50.8% of all departures from ANC (incl. cargo) are >= 2,500 mi (744 of 1,464); "
                             "passenger service: 16.7% (144 of 864).")
    assert facts(r)["Departures >= 2,500 mi, all flights"] == {"value": "744 of 1,464", "required": False}


def test_metrics_headline_follows_topic(db, scoring_cfg):
    load = tools.airport_metrics(db, scoring_cfg, "ANC", topic="demand")["headline"]
    assert load == "ANC in 2025-01..2025-12: 115,500 passengers on 134,400 seats (passenger service), load factor 85.9%."
    growth = tools.airport_metrics(db, scoring_cfg, "ANC", topic="growth")["headline"]
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


# --- resolve_airport ranking, no match, regions (P2.12) -------------------------------------

def test_no_match_is_explicit_with_suggestions(db, scoring_cfg):
    r = check_envelope(tools.resolve_airport(db, scoring_cfg, "Anchorge"))
    assert r["data"] is None and r["error"] == "No airport with passenger service matches 'Anchorge'"
    assert facts(r)["Match"] == {"value": "no match for 'Anchorge'", "required": True}
    assert r["suggestions"][0]["iata"] == "ANC" and r["candidates"][0] == "ANC"
    assert r["headline"].startswith("No airport with passenger service matches 'Anchorge'. Did you mean: ANC (")


def test_no_match_without_close_names_still_gives_a_hint(db, scoring_cfg):
    r = tools.resolve_airport(db, scoring_cfg, "Atlantis")
    assert r["suggestions"] == [] and "IATA code" in r["headline"]
    assert facts(r)["Match"]["required"] is True


def test_several_matches_return_best_and_alternatives(db, scoring_cfg):
    r = tools.resolve_airport(db, scoring_cfg, "Chicago")
    d = r["data"]
    assert d["ambiguous"] is True and d["best"]["iata"] == "ORD"
    assert [a["iata"] for a in d["alternatives"]] == ["MDW"] and d["matches_total"] == 2
    assert r["headline"] == ("'Chicago' matches 2 airports: best match ORD (Chicago O'Hare International "
                             "Airport), alternatives MDW.")


def test_name_and_city_matches_rank_by_passengers(db, scoring_cfg):
    # a small airport with "Boston" in its name must not beat BOS (city Boston, far more passengers)
    db.execute("INSERT INTO airports (iata, name, state, municipality) VALUES "
               "('BSF', 'Boston Small Field', 'MA', 'Quincy')")
    db.executemany("INSERT INTO t100_route_month VALUES ('BSF', NULL, 'BOS', 2025, ?, 'pax', 10, 1, 9, 5)",
                   [(m,) for m in range(1, 13)])
    d = tools.resolve_airport(db, scoring_cfg, "boston")["data"]
    assert d["best"]["iata"] == "BOS" and [a["iata"] for a in d["alternatives"]] == ["BSF"]


def test_exact_iata_wins_over_name_matches(db, scoring_cfg):
    d = tools.resolve_airport(db, scoring_cfg, "ORD")["data"]
    assert d == {"query": "ORD", "ambiguous": False, "iata": "ORD", "name": "Chicago O'Hare International Airport",
                 "state": "IL"}


def test_metrics_for_an_ambiguous_name_return_best_and_alternatives(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "Chicago")
    assert r["data"] is None and r["best"] == "ORD" and r["alternatives"] == ["MDW"]
    assert r["headline"].startswith("'Chicago' matches 2 airports")


def test_unsupported_region_lists_configured_regions_and_suggests_national(db, scoring_cfg):
    r = check_envelope(tools.rank_airports(db, scoring_cfg, "California"))
    assert r["data"] is None
    assert r["error"] == ("Region 'California' is not configured. Configured regions: new_england "
                          "(MA, CT, RI, NH, VT, ME). Use region 'national' for a ranking of all US airports, "
                          "or ask about a specific airport (airport_metrics).")
    assert r["candidates"] == ["new_england", "national"]
    assert facts(r)["Configured regions"] == {"value": "new_england (MA, CT, RI, NH, VT, ME)", "required": True}


def test_national_ranking_covers_the_whole_peer_set(db, scoring_cfg):
    r = check_envelope(tools.rank_airports(db, scoring_cfg, "national", top_n=50))
    d = r["data"]
    assert d["region"] == "national" and d["states"] == "all US states and territories"
    assert d["ranked_total"] == d["peer_set_size"] == 8  # the 9 seeded origins except HYA
    assert d["excluded"] == [e for e in d["excluded"] if e.startswith("HYA")]


# --- congestion (On-Time loaded) -----------------------------------------------------------

@pytest.fixture
def db_ontime(db):
    seed_ontime(db)
    return db


def test_congestion_topic_required_facts_and_headline(db_ontime, scoring_cfg):
    r = check_envelope(tools.airport_metrics(db_ontime, scoring_cfg, "SNA", topic="congestion"))
    assert required(r) == {
        "Departures delayed >= 15 min": f"{round(40 / 198 * 100, 1)}%",
        "Mean taxi-out (departures)": "16.0 min",
        "NAS-delay minutes per arriving flight": "1.5",
        "Cancelled departures (shown, not scored)": "1.0% (24 of 2,400)",
        "Data period": "2025-01..2025-12",
        "Confidence": "medium",
    }
    assert r["headline"] == ("SNA in 2025-01..2025-12: 20.2% of departures delayed >= 15 min, mean taxi-out "
                             "16.0 min, 1.5 NAS-delay minutes per arriving flight; 1.0% of departures cancelled "
                             "(not scored).")
    assert "Passengers (passenger service)" not in facts(r)   # congestion topic only
    assert any("large reporting carriers" in a for a in r["assumptions"])
    assert any("arrivals" in a and "departures" in a for a in r["assumptions"])
    c = r["data"]["congestion"]
    assert c["status"] == "ok" and c["departures_reported"] == 2400 and c["arrivals_reported"] == 2376
    assert set(c["percentiles"]) == {"dep_delay_rate", "taxi_out", "nas_delay_per_arrival", "congestion"}


def test_congestion_topic_without_enough_flights(db_ontime, scoring_cfg):
    r = tools.airport_metrics(db_ontime, scoring_cfg, "BTV", topic="congestion")
    assert required(r)["Congestion"] == "not available: 600 reported departures, below the 1,000 minimum"
    r = tools.airport_metrics(db_ontime, scoring_cfg, "ANC", topic="congestion")
    assert required(r)["Congestion"] == "not available: no flights of the On-Time reporting carriers"
    assert "no data" in r["headline"]


def test_congestion_topic_before_ontime_is_loaded(db, scoring_cfg):
    r = tools.airport_metrics(db, scoring_cfg, "SNA", topic="congestion")
    assert required(r)["Congestion"] == "not available: On-Time data not loaded"


def test_rank_with_ontime_uses_the_congestion_weight(db_ontime, scoring_cfg):
    r = tools.rank_airports(db_ontime, scoring_cfg, "new_england")
    d = r["data"]
    assert set(d["weights_used"]) == {"utilization", "congestion", "scale", "growth", "long_haul"}
    assert set(d["components_unavailable"]) == {"demand_context"}
    assert facts(r)["Weights used (renormalized without demand_context)"]["value"] == (
        "utilization 0.278, congestion 0.278, scale 0.167, growth 0.167, long_haul 0.111")
    by = {x["iata"]: x for x in d["ranked"]}
    assert by["BOS"]["components"]["congestion"]["dep_delay_rate_pct"] == round(200 / 780 * 100, 1)
    # BTV has too few On-Time flights: congestion missing, scored on the other components
    assert by["BTV"]["missing"] == ["congestion"] and "congestion" not in by["BTV"]["weights_used"]
    assert any("large reporting carriers" in a for a in r["assumptions"])
    assert any("cancellations are shown, not scored" in a for a in r["assumptions"])
    assert "congestion (delayed departures" in facts(r)["#1 " + f"{d['ranked'][0]['iata']} ({d['ranked'][0]['name']})"]["value"]


def test_data_status_lists_congestion_only_until_ontime_is_loaded(db, scoring_cfg):
    assert "congestion" in tools.data_status(db, scoring_cfg)["data"]["score_components_unavailable"]
    seed_ontime(db)
    assert "congestion" not in tools.data_status(db, scoring_cfg)["data"]["score_components_unavailable"]


# --- compare_airports (P3.3) -------------------------------------------------------------------

def comparison(result):
    """metric key -> (verdict, {airport: display})"""
    return {m["key"]: (m["verdict"], {c: v["display"] for c, v in m["cells"].items()})
            for m in result["data"]["metrics"]}


def test_compare_la_uses_the_configured_default_and_states_it(db_ontime, scoring_cfg):
    r = check_envelope(tools.compare_airports(db_ontime, scoring_cfg, "LA", "Santa Ana"))
    assert r["data"]["airports"] == ["LAX", "SNA"]
    assert r["assumptions"][0] == "'LA' = LAX (configured default; other candidates: BUR, LGB, ONT, SNA)"
    assert r["confidence"] == "medium"


def test_compare_verdicts_follow_the_margin(db_ontime, scoring_cfg):
    m = comparison(tools.compare_airports(db_ontime, scoring_cfg, "LAX", "SNA"))
    # LAX 25.3% of 11,880 vs SNA 20.2% of 2,376: 5.1 points, combined margin about 1.8 -> LAX higher
    assert m["dep_delay_rate"] == ("LAX higher", {"LAX": "25.3% ± 0.78 pts (11,880 departures)",
                                                  "SNA": "20.2% ± 1.61 pts (2,376 departures)"})
    assert m["nas_min_per_arrival"][0] == "LAX higher"
    assert m["congestion_percentile"][0] == "not tested (a rank, no margin)"
    assert m["passengers"][0] == "LAX higher (full count)"


def test_compare_inside_the_margin_is_no_meaningful_difference(db_ontime, scoring_cfg):
    # PWM 18.4% of 1,764 vs SNA 20.2% of 2,376: 1.8 points inside the combined margin of about 2.4
    r = tools.compare_airports(db_ontime, scoring_cfg, "PWM", "SNA")
    assert comparison(r)["dep_delay_rate"][0] == "no meaningful difference"
    assert "no meaningful difference in delayed departures (PWM 18.4%, SNA 20.2%)" in r["headline"]
    assert "PWM higher" not in r["headline"].split(";")[0] and "SNA higher" not in r["headline"].split(";")[0]


def test_compare_facts_are_per_airport_and_cancellations_not_scored(db_ontime, scoring_cfg):
    r = tools.compare_airports(db_ontime, scoring_cfg, "LAX", "SNA")
    by = facts(r)
    assert by["LAX: Departures delayed >= 15 min"]["required"] and by["SNA: Departures delayed >= 15 min"]["required"]
    assert by["Verdict: Departures delayed >= 15 min"]["value"] == "LAX higher"
    assert "SNA: Cancelled departures (shown, not scored)" in by
    assert {f["airport"] for f in r["facts"] if "airport" in f} == {"LAX", "SNA"}
    assert any("cancellations are shown, not scored" in a for a in r["assumptions"])


def test_compare_overall_sentence_from_the_scored_verdicts(db_ontime, scoring_cfg):
    # LAX higher on all three scored measures (cancellations are not scored and do not count)
    r = tools.compare_airports(db_ontime, scoring_cfg, "LAX", "SNA")
    assert r["data"]["overall"] == "Overall: LAX is higher on every measure with a meaningful difference."
    # PWM vs SNA: delays inside the margin, taxi-out and NAS differ in opposite directions -> mixed
    assert tools.compare_airports(db_ontime, scoring_cfg, "PWM", "SNA")["data"]["overall"] == (
        "Overall: mixed, no airport is higher on every measure.")
    assert tools.compare_airports(db_ontime, scoring_cfg, "BOS", "BTV")["data"]["overall"] == (
        "Overall: not comparable, no scored congestion figure for both airports.")


def test_compare_size_note_follows_the_configured_ratio(db_ontime, scoring_cfg):
    from dataclasses import replace
    # passenger-service departures a month: LAX 100, SNA 10
    note = " Note: LAX has 10.0 times SNA's departures (passenger service): rates compare per flight, but the airports differ in scale."
    assert tools.compare_airports(db_ontime, scoring_cfg, "LAX", "SNA")["headline"].endswith(note)
    wide = replace(scoring_cfg, comparison_size_ratio_note=20.0)
    assert "Note:" not in tools.compare_airports(db_ontime, wide, "LAX", "SNA")["headline"]


def test_compare_without_congestion_figures_is_not_comparable(db_ontime, scoring_cfg):
    r = tools.compare_airports(db_ontime, scoring_cfg, "BOS", "BTV")   # BTV under min_ontime_flights
    assert comparison(r)["dep_delay_rate"][0] == "not comparable"
    assert "BTV has no congestion figures (too few flights)." in r["headline"]


def test_compare_ambiguous_name_returns_the_ambiguity(db_ontime, scoring_cfg):
    r = tools.compare_airports(db_ontime, scoring_cfg, "Chicago", "BOS")
    assert r["data"] is None and r["candidates"] == ["ORD", "MDW"] and r["best"] == "ORD"


def test_compare_unknown_and_same_airport(db_ontime, scoring_cfg):
    assert "No airport" in tools.compare_airports(db_ontime, scoring_cfg, "Atlantis", "BOS")["error"]
    assert "two different airports" in tools.compare_airports(db_ontime, scoring_cfg, "LAX", "LAX")["error"]


# --- explain_demand_gap (P3.4) -----------------------------------------------------------------

@pytest.fixture
def db_sfo(db_ontime):
    seed_sfo(db_ontime)
    return db_ontime


def test_demand_gap_is_a_labeled_proxy_with_capped_confidence(db_sfo, scoring_cfg):
    r = check_envelope(tools.explain_demand_gap(db_sfo, scoring_cfg, "SFO"))
    assert r["headline"].startswith("PROXY, not a measurement")
    assert r["data"]["proxy"] is True
    assert required(r)["Unmet demand"] == "not observable in this data; the figures below are a proxy"
    assert any("not observable" in a for a in r["assumptions"])
    assert any("booking" in a and "spill" in a for a in r["assumptions"])
    assert r["confidence"] in ("medium", "low")


def test_demand_gap_facts_from_the_data(db_sfo, scoring_cfg):
    r = tools.explain_demand_gap(db_sfo, scoring_cfg, "SFO")
    req = required(r)
    # June-August: (950 + 1,500) / 3,000 = 81.7% > 80%; other months 73.3%
    assert req["Months with load factor above 80%"] == "3 of 12"
    assert facts(r)["Peak months (load factor)"]["value"] == "2025-06 81.7%, 2025-07 81.7%, 2025-08 81.7%"
    # spill proxy: HNL (950 - 800) x 3 months = 450 of 27,150 passengers; LAX never above 80%
    # the excess over 80% of seats, not all passengers on full flights (2,850 flew on HNL in summer)
    assert req["Spill proxy: passengers above the 80% load level (excess over 80% of seats, per route and month)"] == (
        "450 (1.7% of passengers)")
    assert any("only the excess counts" in a for a in r["assumptions"])
    assert req["Spill proxy sensitivity (target load factor)"] == "600 at 75% to 300 at 85%"
    assert facts(r)["Routes with the most passengers above 80% load"]["value"] == "SFO-HNL 450 (100.0% of the spill proxy)"
    assert facts(r)["Routes with the most passengers (load factor)"]["value"].startswith("SFO-LAX 18,000 (75.0%)")
    # causes per month: NAS 300, late aircraft 180, carrier 100, weather 20, security 0 (of 600)
    assert req["Delay-cause mix (share of delay minutes, arrivals)"] == (
        "NAS 50.0%, late aircraft 30.0%, carrier 16.7%, weather 3.3%, security 0.0%")
    assert req["Passenger growth vs 2024-01..2024-12"] == "+10.4%"


def test_demand_gap_why_is_built_from_the_facts(db_sfo, scoring_cfg):
    why = tools.explain_demand_gap(db_sfo, scoring_cfg, "SFO")["data"]["why"]
    assert why == [
        "Airport-wide load factor was above the 80% target in 3 of 12 months, highest in "
        "2025-06 81.7%, 2025-07 81.7%, 2025-08 81.7%.",
        "Passengers above the target load are concentrated on routes to HNL (100.0% of the spill proxy on that route).",
        "The largest delay cause on arrivals is NAS (50.0% of delay minutes).",
        "Passengers changed +10.4% vs 2024-01..2024-12.",
    ]


def test_demand_gap_without_ontime_still_answers(db, scoring_cfg):
    seed_sfo(db)
    r = tools.explain_demand_gap(db, scoring_cfg, "SFO")
    assert required(r)["Delay-cause mix (share of delay minutes, arrivals)"] == "not available (no On-Time delay minutes)"


def test_demand_gap_target_comes_from_config(db_sfo, scoring_cfg):
    from dataclasses import replace
    r = tools.explain_demand_gap(db_sfo, replace(scoring_cfg, target_load_factor=0.75), "SFO")
    assert required(r)["Months with load factor above 75%"] == "3 of 12"
    assert "above the 75% load level" in r["headline"]


def test_demand_gap_ambiguous_or_unknown(db_sfo, scoring_cfg):
    assert tools.explain_demand_gap(db_sfo, scoring_cfg, "Chicago")["candidates"] == ["ORD", "MDW"]
    assert "No airport" in tools.explain_demand_gap(db_sfo, scoring_cfg, "Atlantis")["error"]


# --- rank stability (P3.5) -----------------------------------------------------------------------

def test_rank_airports_reports_stability_per_airport(db_ontime, scoring_cfg):
    r = tools.rank_airports(db_ontime, scoring_cfg, "new_england")
    for x in r["data"]["ranked"]:
        s = x["stability"]
        assert s["runs"] == scoring_cfg.sensitivity_runs and s["top_n"] == scoring_cfg.sensitivity_top_n
        assert s["min_rank"] <= x["rank"] <= s["max_rank"]
    by = facts(r)
    assert by["Rank stability (method)"]["value"] == "20 runs, each weight changed at random by up to ±25% and renormalized (seed 42)"
    top = r["data"]["ranked"][0]
    assert by["Stability of the top airport"]["value"] == f"{top['iata']} first in {top['stability']['first']} of 20 runs"
    assert "of stability runs, ranks" in by[f"#1 {top['iata']} ({top['name']})"]["value"]


def test_rank_stability_is_deterministic_in_the_tool(db_ontime, scoring_cfg):
    a = tools.rank_airports(db_ontime, scoring_cfg, "national")
    b = tools.rank_airports(db_ontime, scoring_cfg, "national")
    assert [x["stability"] for x in a["data"]["ranked"]] == [x["stability"] for x in b["data"]["ranked"]]


def test_unstable_top_airport_gets_a_headline_note(db_ontime, scoring_cfg, monkeypatch):
    real = tools.scoring.rank_stability

    def shaky(ranking, weights, cfg):
        out = real(ranking, weights, cfg)
        out[ranking.ranked["iata"].iloc[0]].update(first=12)
        return out

    monkeypatch.setattr(tools.scoring, "rank_stability", shaky)
    r = tools.rank_airports(db_ontime, scoring_cfg, "new_england")
    top = r["data"]["ranked"][0]["iata"]
    assert r["headline"].endswith(f"Not stable: {top} is first in only 12 of 20 runs with perturbed weights.")


def test_stable_top_airport_has_no_note(db_ontime, scoring_cfg):
    assert "Not stable" not in tools.rank_airports(db_ontime, scoring_cfg, "new_england")["headline"]


# --- drivers (P3.8c) ---------------------------------------------------------------------------

def test_rank_airports_drivers_block(db_ontime, scoring_cfg):
    r = tools.rank_airports(db_ontime, scoring_cfg, "new_england")
    window = store.analysis_window(db_ontime)
    result = scoring.score(store.scoring_metrics(db_ontime, window, scoring_cfg.long_haul_miles), scoring_cfg, window)
    for x in r["data"]["ranked"]:
        pcts = {c: v["percentile"] for c, v in x["components"].items() if v["percentile"] is not None}
        floor = set(next(row for _, row in result.table.iterrows() if row["iata"] == x["iata"])["at_floor"])
        assert x["drivers"]["strongest"]["percentile"] == max(p for c, p in pcts.items() if c not in floor)
        assert x["drivers"]["strongest"]["component"] not in floor
        assert x["drivers"]["weakest"]["percentile"] == min(pcts.values())
        f = next(f for f in r["facts"] if f["label"] == f"Drivers: {x['iata']}")
        assert f["ranked"] == x["iata"] and f["value"].startswith("strongest: ")
    assert any(a.startswith("drivers = ") for a in r["assumptions"])


# --- demand_context (P4.3) ---------------------------------------------------------------------

@pytest.fixture
def db_full(db_ontime):
    seed_census(db_ontime)
    return db_ontime


def labels(r):
    return {f["label"]: f["value"] for f in r["facts"]}


def test_demand_card_shows_the_metro_context(db_full, scoring_cfg):
    r = check_envelope(tools.airport_metrics(db_full, scoring_cfg, "BOS", topic="demand"))
    v = labels(r)
    assert v["Metro area (CBSA)"] == "Boston-Cambridge-Newton, MA-NH Metro Area (14460)"
    assert v["Metro population (ACS 5-year 2020-2024)"] == "4,900,000"
    assert v["Metro population growth vs ACS 5-year 2015-2019 (demand context)"] == "+2.1%"
    assert "Demand context percentile in the peer set" in v
    assert any(a.startswith("demand context = population growth") for a in r["assumptions"])
    assert any("not endorsed or certified by the Census Bureau" in a for a in r["assumptions"])


def test_airport_without_cbsa_says_so_and_is_medium(db_full, scoring_cfg):
    r = tools.airport_metrics(db_full, scoring_cfg, "BTV", topic="demand")
    assert labels(r)["Metro area (CBSA)"].startswith("none: the airport lies outside every metro and micro area")
    assert r["confidence"] == "medium"


def test_card_before_census_is_loaded(db_ontime, scoring_cfg):
    r = tools.airport_metrics(db_ontime, scoring_cfg, "BOS", topic="demand")
    assert labels(r)["Metro population (demand context)"].startswith("not available: Census population not loaded")
    assert "demand_context" in tools.data_status(db_ontime, scoring_cfg)["data"]["score_components_unavailable"]


def test_ranking_uses_demand_context_in_weights_and_drivers(db_full, scoring_cfg):
    r = tools.rank_airports(db_full, scoring_cfg, "new_england")
    assert labels(r)["Weights used"] == ("utilization 0.25, congestion 0.25, scale 0.15, growth 0.15, "
                                         "long_haul 0.1, demand_context 0.1")
    assert "demand_context" not in tools.data_status(db_full, scoring_cfg)["data"]["score_components_unavailable"]
    bos = next(x for x in r["data"]["ranked"] if x["iata"] == "BOS")
    assert bos["components"]["demand_context"]["metro_growth_pct"] == 2.1
    assert "demand context (metro population growth) 2.1%" in labels(r)[f"#{bos['rank']} BOS ({bos['name']})"]
    # BTV has no CBSA: scored without demand_context (and without congestion), its own weights shown
    btv = next(x for x in r["data"]["ranked"] if x["iata"] == "BTV")
    assert "demand_context" in btv["missing"] and "demand_context" not in btv["weights_used"]
    # drivers cover demand_context: PWM (fastest-growing metro) ties it with utilization at 100, and
    # utilization is listed first in the weights
    pwm = next(x for x in r["data"]["ranked"] if x["iata"] == "PWM")
    assert pwm["components"]["demand_context"]["percentile"] == 100
    assert labels(r)["Drivers: PWM"].startswith("strongest: utilization (load factor), percentile 100")
