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


def test_airport_metrics_required_facts(db, scoring_cfg):
    f = facts(tools.airport_metrics(db, scoring_cfg, "ANC"))
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
