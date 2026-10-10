"""Scoring tests on small synthetic data (never data/raw)."""
import math

import pandas as pd
import pytest

from src import scoring


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
