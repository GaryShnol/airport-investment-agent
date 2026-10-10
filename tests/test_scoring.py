"""Scoring tests on small synthetic data (never data/raw)."""
import math
from dataclasses import replace

import pandas as pd
import pytest

from src import scoring
from src.config import ScoringConfig
from src.store import Window


# --- percentile rank ---------------------------------------------------------------------

def test_percentile_orders_values():
    pct = scoring.percentile_rank(pd.Series([10, 30, 20, 40], index=list("abcd")))
    assert list(pct.sort_values().index) == ["a", "c", "b", "d"]
    assert pct["d"] == 1.0 and pct["a"] == 0.25


def test_percentile_ties_share_the_average_rank():
    pct = scoring.percentile_rank(pd.Series([1, 2, 2, 3], index=list("abcd")))
    assert pct["b"] == pct["c"] == pytest.approx(2.5 / 4)


def test_percentile_ignores_missing():
    pct = scoring.percentile_rank(pd.Series([1.0, None, 3.0], index=list("abc")))
    assert math.isnan(pct["b"])
    assert (pct["a"], pct["c"]) == (0.5, 1.0)


# --- raw metrics -------------------------------------------------------------------------

def test_load_factor():
    lf = scoring.load_factor(pd.Series([80, 50]), pd.Series([100, 0]))
    assert lf[0] == 0.8 and math.isnan(lf[1])


def test_long_haul_share():
    share = scoring.long_haul_share(pd.Series([5, 0, 1]), pd.Series([20, 10, 0]))
    assert share[0] == 0.25 and share[1] == 0.0 and math.isnan(share[2])


def test_growth():
    g = scoring.growth(pd.Series([110, 90]), pd.Series([100, 100]))
    assert g.tolist() == pytest.approx([0.10, -0.10])


@pytest.mark.parametrize("previous", [0, None])
def test_growth_without_previous_is_missing(previous):
    g = scoring.growth(pd.Series([100.0]), pd.Series([previous], dtype=float))
    assert math.isnan(g[0])


# --- weight renormalization --------------------------------------------------------------

WEIGHTS = {"utilization": 0.25, "congestion": 0.25, "scale": 0.15,
           "growth": 0.15, "long_haul": 0.10, "demand_context": 0.10}


def test_renormalize_sums_to_one_and_keeps_proportions():
    used = scoring.renormalize(WEIGHTS, ["utilization", "scale", "growth", "long_haul"])
    assert set(used) == {"utilization", "scale", "growth", "long_haul"}
    assert sum(used.values()) == pytest.approx(1.0)
    assert used["utilization"] == pytest.approx(0.25 / 0.65)
    assert used["utilization"] / used["long_haul"] == pytest.approx(2.5)


def test_renormalize_all_available_is_unchanged():
    assert scoring.renormalize(WEIGHTS, WEIGHTS) == pytest.approx(WEIGHTS)


def test_renormalize_nothing_available():
    with pytest.raises(ValueError):
        scoring.renormalize(WEIGHTS, [])


# --- score and rank ----------------------------------------------------------------------

FULL = Window("2025-01", "2025-12", "2024-01", "2024-12", months=12, prev_months=12)


@pytest.fixture
def cfg():
    return ScoringConfig(
        weights=dict(WEIGHTS),
        congestion_weights={"dep_delay_rate": 0.4, "taxi_out": 0.3, "nas_delay_per_arrival": 0.3},
        long_haul_miles=2500, long_haul_sensitivity_miles=(1500, 2500, 3000),
        long_haul_min_share=0.02, min_passengers_per_year=5_000, min_ontime_flights=100, target_load_factor=0.8,
        sensitivity_runs=5, sensitivity_jitter=0.25, sensitivity_seed=1, comparison_confidence_level=0.95,
        regions={"north": ("AA", "BB"), "south": ("CC",)}, max_ranked=25,
    )


def airport(iata, state, departures, seats, passengers, long_haul, prev, months=12):
    return dict(iata=iata, name=f"{iata} Airport", state=state, region=None, departures=departures,
                seats=seats, passengers=passengers, long_haul_departures=long_haul,
                months_active=months, prev_passengers=prev)


@pytest.fixture
def metrics():
    return pd.DataFrame([
        airport("BIG", "AA", 1000, 100_000, 85_000, 100, 80_000),   # large, busy, some long haul
        airport("MID", "AA", 500, 40_000, 30_000, 0, 30_000),       # flat
        airport("NEW", "BB", 200, 10_000, 7_000, 0, 0),             # no previous year -> growth missing
        airport("HOT", "BB", 150, 10_000, 9_000, 0, 6_000),         # small, fast growth
        airport("TNY", "BB", 50, 3_000, 2_500, 0, 2_000),           # below the floor -> excluded
        airport("SUN", "CC", 800, 60_000, 50_000, 300, 45_000, months=8),  # other region, seasonal
        airport("YVR", None, 900, 90_000, 80_000, 50, 70_000),      # foreign origin: no state
    ])


def test_peer_set_needs_state_and_floor(metrics, cfg):
    result = scoring.score(metrics, cfg, FULL)
    peers = set(result.table.loc[result.table["in_peer_set"], "iata"])
    assert peers == {"BIG", "MID", "NEW", "HOT", "SUN"}
    assert result.peer_set_size == 5
    assert result.table.set_index("iata").loc[["TNY", "YVR"], "score"].isna().all()


def test_peer_set_floor_is_on_passengers_not_departures(cfg):
    commuter = airport("CMT", "AA", 5_000, 45_000, 4_999, 0, 4_000)   # many flights, tiny aircraft
    regional = airport("RGN", "AA", 40, 6_000, 5_000, 0, 4_000)       # few flights, enough passengers
    t = scoring.score(pd.DataFrame([commuter, regional]), cfg, FULL).table.set_index("iata")
    assert not t.at["CMT", "in_peer_set"] and t.at["RGN", "in_peer_set"]


def test_long_haul_share_below_the_floor_counts_as_zero(cfg):
    rows = pd.DataFrame([airport("NON", "AA", 10_000, 900_000, 800_000, 0, 700_000),
                         airport("LOW", "AA", 10_000, 900_000, 800_000, 199, 700_000),   # 1.99%
                         airport("TOP", "AA", 10_000, 900_000, 800_000, 200, 700_000)])  # 2.00%
    t = scoring.score(rows, cfg, FULL).table.set_index("iata")
    assert t.at["LOW", "long_haul_share"] == pytest.approx(0.0199)  # true share still shown
    assert t.at["LOW", "long_haul_share_scored"] == 0.0
    assert t.at["TOP", "long_haul_share_scored"] == pytest.approx(0.02)
    assert t.at["LOW", "pct_long_haul"] == t.at["NON", "pct_long_haul"] < t.at["TOP", "pct_long_haul"]


def test_floor_share_keeps_missing():
    out = scoring.floor_share(pd.Series([0.01, 0.02, None]), 0.02)
    assert out[0] == 0.0 and out[1] == 0.02 and math.isnan(out[2])


def test_missing_built_components_are_renormalized_and_reported(metrics, cfg):
    result = scoring.score(metrics, cfg, FULL)
    assert set(result.unavailable) == {"congestion", "demand_context"}
    assert set(result.weights_used) == {"utilization", "scale", "growth", "long_haul"}
    assert sum(result.weights_used.values()) == pytest.approx(1.0)
    assert result.weights_configured == WEIGHTS
    assert "renormalized" in result.notes[0]


def test_results_carry_raw_values_percentiles_and_score_range(metrics, cfg):
    t = scoring.score(metrics, cfg, FULL).table.set_index("iata")
    assert t.at["BIG", "load_factor"] == pytest.approx(0.85)
    assert t.at["BIG", "long_haul_share"] == pytest.approx(0.10)
    assert t.at["BIG", "growth"] == pytest.approx(0.0625)
    assert t.at["BIG", "pct_scale"] == 1.0
    assert {"pct_utilization", "pct_scale", "pct_growth", "pct_long_haul"} <= set(t.columns)
    assert t.loc[t["in_peer_set"], "score"].between(0, 1).all()


def test_missing_growth_is_a_missing_component_for_that_airport(metrics, cfg):
    t = scoring.score(metrics, cfg, FULL).table.set_index("iata")
    assert t.at["NEW", "missing"] == ["growth"]
    assert "growth" not in t.at["NEW", "weights_used"]
    assert sum(t.at["NEW", "weights_used"].values()) == pytest.approx(1.0)
    assert not math.isnan(t.at["NEW", "score"])
    assert any("growth" in r for r in t.at["NEW", "confidence_reasons"])


def test_confidence_is_at_most_medium_while_components_are_not_built(metrics, cfg):
    t = scoring.score(metrics, cfg, FULL).table
    assert set(t["confidence"]) == {"medium"}
    assert all("renormalized without congestion, demand_context" in r[0] for r in t["confidence_reasons"])


def test_confidence_high_when_every_component_is_available(metrics, cfg, monkeypatch):
    # Simulate the finished formula: pretend demand_context is built; congestion comes from On-Time.
    monkeypatch.setattr(scoring, "NOT_BUILT", {})
    monkeypatch.setitem(scoring.RAW_COLUMN, "demand_context", "demand_context")
    full = with_ontime(metrics, {i: (1000, 200, 15.0, 3.0, 10) for i in metrics["iata"]})
    full = full.assign(demand_context=range(len(metrics)))
    t = scoring.score(full, cfg, FULL).table.set_index("iata")
    assert t.at["BIG", "confidence"] == "high"
    assert t.at["SUN", "confidence"] == "medium"  # partial coverage: service in 8 of 12 months
    assert t.at["NEW", "confidence"] == "medium"  # growth missing


def test_confidence_low_with_fewer_than_12_months(metrics, cfg):
    t = scoring.score(metrics, cfg, replace(FULL, months=9)).table
    assert set(t["confidence"]) == {"low"}


def test_partial_previous_window_drops_growth(metrics, cfg):
    result = scoring.score(metrics, cfg, replace(FULL, prev_months=6))
    assert "growth" in result.unavailable
    assert "growth" not in result.weights_used


def test_rank_inside_region_and_excluded_list(metrics, cfg):
    ranking = scoring.rank(scoring.score(metrics, cfg, FULL), "north", cfg)
    assert set(ranking.ranked["iata"]) == {"BIG", "MID", "NEW", "HOT"}
    assert list(ranking.ranked["rank"]) == [1, 2, 3, 4]
    assert ranking.ranked["score"].is_monotonic_decreasing
    assert ranking.ranked.iloc[0]["iata"] == "BIG"
    assert list(ranking.excluded["iata"]) == ["TNY"]
    assert "below the 5,000 floor" in ranking.excluded.iloc[0]["reason"]


def test_region_percentiles_stay_national(metrics, cfg):
    south = scoring.rank(scoring.score(metrics, cfg, FULL), "south", cfg).ranked
    assert list(south["iata"]) == ["SUN"]
    assert south.iloc[0]["pct_scale"] < 1.0  # BIG, in another region, is still in the peer set


def test_unknown_region(metrics, cfg):
    with pytest.raises(ValueError, match="configured"):
        scoring.rank(scoring.score(metrics, cfg, FULL), "atlantis", cfg)


def test_same_input_same_output(metrics, cfg):
    a = scoring.score(metrics, cfg, FULL).table
    b = scoring.score(metrics.sample(frac=1, random_state=3), cfg, FULL).table
    pd.testing.assert_frame_equal(a, b)


def test_score_ties_break_by_iata(cfg):
    cfg = replace(cfg, min_passengers_per_year=0)
    twins = pd.DataFrame([airport("ZZZ", "AA", 200, 100, 80, 0, 70),
                          airport("AAA", "AA", 200, 100, 80, 0, 70)])
    t = scoring.score(twins, cfg, FULL).table
    assert t["score"].iloc[0] == t["score"].iloc[1]
    assert list(t["iata"]) == ["AAA", "ZZZ"]


def test_equal_scores_from_different_components_tie_exactly(cfg):
    cfg = replace(cfg, min_passengers_per_year=0)
    # scale and growth carry the same weight: high scale + low growth == low scale + high growth,
    # but summed in floating point the two can differ by 1e-16. The tie must still go by IATA.
    # Unrounded, this case gives 0.8557692307692308 vs 0.855769230769231.
    swap = pd.DataFrame([airport("ZZZ", "AA", 200, 1000, 800, 1, 790),   # bigger, slower growth
                         airport("AAA", "AA", 200, 875, 700, 1, 500),    # smaller, faster growth
                         airport("F0", "AA", 200, 1000, 101, 0, 100),
                         airport("F1", "AA", 200, 1000, 201, 1, 200)])
    t = scoring.score(swap, cfg, FULL).table
    assert t["score"].iloc[0] == t["score"].iloc[1]
    assert list(t["iata"][:2]) == ["AAA", "ZZZ"]


def test_custom_weights_change_the_ranking(metrics, cfg):
    default = scoring.rank(scoring.score(metrics, cfg, FULL), "north", cfg).ranked
    growth_only = replace(cfg, weights={**WEIGHTS, "utilization": 0.0, "scale": 0.0,
                                        "growth": 0.65, "long_haul": 0.0})
    custom = scoring.rank(scoring.score(metrics, growth_only, FULL), "north", growth_only).ranked
    assert default.iloc[0]["iata"] == "BIG"
    assert custom.iloc[0]["iata"] == "HOT"
    # NEW has no growth and every other built weight is zero: no score, listed last, not an error
    assert custom.iloc[-1]["iata"] == "NEW" and math.isnan(custom.iloc[-1]["score"])


# --- congestion (On-Time) ------------------------------------------------------------------

def with_ontime(metrics, values):
    """Add On-Time columns: iata -> (scheduled departures, delayed >= 15, mean taxi-out,
    NAS minutes per arrival, cancelled). Airports not listed have no On-Time rows (missing)."""
    rows = []
    for iata in metrics["iata"]:
        if iata not in values:
            rows.append({c: float("nan") for c in scoring.ONTIME_COLUMNS})
            continue
        dep, del15, taxi, nas, cancelled = values[iata]
        flown = dep - cancelled
        rows.append({"dep_flights": dep, "dep_cancelled": cancelled, "dep_del15": del15, "dep_del15_n": flown,
                     "taxi_out_sum": taxi * flown, "taxi_out_n": flown, "arr_flights": flown, "nas_min": nas * flown})
    return pd.concat([metrics.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


ONTIME = {  # peer airports: BIG, MID, NEW, HOT, SUN; TNY and YVR are outside the peer set
    "BIG": (1000, 300, 20.0, 6.0, 10),   # worst on every sub-metric
    "MID": (1000, 200, 15.0, 3.0, 50),   # many cancellations: must not matter
    "HOT": (1000, 100, 10.0, 1.0, 0),    # best on every sub-metric
    "SUN": (1000, 200, 25.0, 1.0, 0),    # mixed
    "TNY": (50, 40, 40.0, 9.0, 0),       # few flights
}


def test_congestion_sub_metrics_come_from_counts(metrics, cfg):
    t = scoring.score(with_ontime(metrics, ONTIME), cfg, FULL).table.set_index("iata")
    assert t.at["MID", "dep_delay_rate"] == pytest.approx(200 / 950)   # among departures reporting DEP_DEL15
    assert t.at["MID", "taxi_out_min"] == pytest.approx(15.0)
    assert t.at["MID", "nas_min_per_arrival"] == pytest.approx(3.0)
    assert t.at["MID", "cancellation_rate"] == pytest.approx(0.05)


def test_congestion_weight_is_used_when_ontime_is_loaded(metrics, cfg):
    result = scoring.score(with_ontime(metrics, ONTIME), cfg, FULL)
    assert "congestion" not in result.unavailable
    assert set(result.unavailable) == {"demand_context"}
    assert result.weights_used["congestion"] == pytest.approx(0.25 / 0.9)


def test_congestion_is_the_weighted_mean_of_sub_percentiles_then_ranked(metrics, cfg):
    t = scoring.score(with_ontime(metrics, ONTIME), cfg, FULL).table.set_index("iata")
    # sub-metric percentiles among the 4 peer airports with On-Time data (NEW has none)
    assert t.at["BIG", "pct_dep_delay_rate"] == 1.0 and t.at["HOT", "pct_dep_delay_rate"] == 0.25
    assert t.at["SUN", "pct_taxi_out"] == 1.0 and t.at["SUN", "pct_nas_delay_per_arrival"] == pytest.approx(0.375)
    expected = 0.4 * t.at["SUN", "pct_dep_delay_rate"] + 0.3 * 1.0 + 0.3 * 0.375
    assert t.at["SUN", "congestion_index"] == pytest.approx(expected)
    # the index is ranked like every other component: BIG most congested, HOT least
    assert t.at["BIG", "pct_congestion"] == 1.0 and t.at["HOT", "pct_congestion"] == 0.25


def test_cancellations_are_shown_but_not_scored(metrics, cfg):
    data = with_ontime(metrics, ONTIME)
    base = scoring.score(data, cfg, FULL).table.set_index("iata")
    hot = data["iata"] == "HOT"
    data.loc[hot, ["dep_flights", "dep_cancelled"]] = [1400, 400]   # 400 more cancellations, same operated flights
    other = scoring.score(data, cfg, FULL).table.set_index("iata")
    assert other.at["HOT", "cancellation_rate"] == pytest.approx(400 / 1400)
    assert other["score"].equals(base["score"])


def test_airport_without_ontime_is_renormalized_with_medium_confidence(metrics, cfg):
    t = scoring.score(with_ontime(metrics, ONTIME), cfg, FULL).table.set_index("iata")
    assert t.at["NEW", "ontime_status"] == "no_data"
    assert "congestion" in t.at["NEW", "missing"]
    assert "congestion" not in t.at["NEW", "weights_used"]
    assert sum(t.at["NEW", "weights_used"].values()) == pytest.approx(1.0)
    assert t.at["NEW", "confidence"] == "medium"
    assert any("congestion" in r for r in t.at["NEW", "confidence_reasons"])
    assert t.at["BIG", "missing"] == [] and "congestion" in t.at["BIG", "weights_used"]


def test_min_ontime_flights_leaves_congestion_missing(metrics, cfg):
    small = replace(cfg, min_passengers_per_year=2_000)    # TNY joins the peer set
    t = scoring.score(with_ontime(metrics, ONTIME), small, FULL).table.set_index("iata")
    assert t.at["TNY", "ontime_status"] == "too_few_flights"
    assert math.isnan(t.at["TNY", "dep_delay_rate"]) and "congestion" in t.at["TNY", "missing"]
    # TNY's noisy 80% delay rate does not enter the other airports' percentiles
    assert t.at["BIG", "pct_dep_delay_rate"] == 1.0
    t = scoring.score(with_ontime(metrics, ONTIME), replace(small, min_ontime_flights=10), FULL).table.set_index("iata")
    assert t.at["TNY", "ontime_status"] == "ok" and t.at["TNY", "pct_dep_delay_rate"] == 1.0


def test_congestion_changes_the_ranking_only_when_loaded(metrics, cfg):
    without = scoring.score(metrics, cfg, FULL)
    with_data = scoring.score(with_ontime(metrics, ONTIME), cfg, FULL)
    assert "pct_congestion" not in without.table.columns
    assert with_data.table.set_index("iata").at["BIG", "score"] != without.table.set_index("iata").at["BIG", "score"]


# --- comparison margins (P3.3) ---------------------------------------------------------------

def test_z_value_for_95_percent():
    assert scoring.z_value(0.95) == pytest.approx(1.95996, abs=1e-4)


def test_proportion_margin_from_the_flight_count():
    # 20% on 1,000 flights: 1.96 * sqrt(0.2 * 0.8 / 1000) = 2.48 points
    assert scoring.proportion_margin(0.2, 1000, 1.96) == pytest.approx(0.0248, abs=1e-4)
    assert scoring.proportion_margin(0.2, 0, 1.96) is None
    assert scoring.proportion_margin(None, 1000, 1.96) is None


def test_mean_margin_from_monthly_spread():
    assert scoring.mean_margin([10.0, 12.0, 14.0, 16.0], 2.0) == pytest.approx(2 * 2.581989 / 2, abs=1e-6)
    assert scoring.mean_margin([10.0], 1.96) is None          # one month: no spread
    assert scoring.mean_margin([10.0, float("nan")], 1.96) is None


@pytest.mark.parametrize("a, b, ma, mb, expected", [
    (0.20, 0.18, 0.01, 0.01, "a"),       # difference 2.0 > combined margin 1.41
    (0.18, 0.20, 0.01, 0.01, "b"),
    (0.20, 0.19, 0.01, 0.01, "same"),    # 1.0 inside 1.41: never a winner inside the margin
    (0.20, 0.186, 0.01, 0.01, "same"),   # 1.4 still inside
    (0.20, None, 0.01, 0.01, None),
    (0.20, 0.10, None, 0.01, None),
])
def test_compare_values_uses_the_combined_margin(a, b, ma, mb, expected):
    assert scoring.compare_values(a, b, ma, mb) == expected
