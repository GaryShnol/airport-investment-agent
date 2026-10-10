"""Deterministic airport scoring. Pure functions: no database, no network, no LLM.

Definitions and methodology: docs/plan.md (Definitions, Scoring). Weights, thresholds and
regions come only from `ScoringConfig` (config/weights.yaml via src/config.py).
Helpers take pandas Series (one value per airport) and return Series aligned to the input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist, stdev
from typing import Iterable, Mapping

import pandas as pd

from src.config import ScoringConfig
from src.store import WINDOW_MONTHS, Window


def percentile_rank(values: pd.Series) -> pd.Series:
    """Percentile of each value inside the given set, in (0, 1]; higher value -> higher percentile.

    pandas `rank(method="average", pct=True)`: tied values share the mean of the ranks they
    span, so ties never depend on row order; the top value gets 1.0; missing values stay
    missing and do not count in the denominator.
    """
    return values.rank(method="average", pct=True)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, missing where the denominator is zero or missing."""
    return numerator / denominator.where(denominator > 0)


def load_factor(passengers: pd.Series, seats: pd.Series) -> pd.Series:
    """Passengers / seats on passenger service; missing when there are no seats."""
    return _ratio(passengers, seats)


def long_haul_share(long_haul_departures: pd.Series, departures: pd.Series) -> pd.Series:
    """Passenger-service departures at or above the threshold / all passenger-service departures."""
    return _ratio(long_haul_departures, departures)


def floor_share(share: pd.Series, min_share: float) -> pd.Series:
    """Shares below `min_share` become 0 (missing stays missing): a handful of charter flights
    should not lift an airport's long-haul percentile above airports with none."""
    return share.where(share.isna() | (share >= min_share), 0.0)


def growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    """current / previous - 1. Missing (not zero, not infinite) when previous is zero or missing:
    an airport without a previous year has no measurable growth, so the component is missing."""
    return _ratio(current, previous) - 1


def renormalize(weights: Mapping[str, float], available: Iterable[str]) -> dict[str, float]:
    """Keep only the available components and rescale their weights to sum to 1.

    Raises ValueError if no available component has a positive weight.
    """
    kept = {k: w for k, w in weights.items() if k in set(available)}
    total = sum(kept.values())
    if total <= 0:
        raise ValueError("No scoring component is available")
    return {k: w / total for k, w in kept.items()}


# Components not built yet, with the reason reported in every result. Remove an entry when the
# component is added (demand_context in M4 from Census).
NOT_BUILT = {
    "demand_context": "Census data not loaded yet (milestone M4)",
}
# Congestion is built but needs On-Time data: without the On-Time columns in the metrics it is
# unavailable for every airport and the weights are renormalized, as for a component not built.
ONTIME_NOT_LOADED = "On-Time data not loaded (run `uv run python -m src.ingest`)"
ONTIME_COLUMNS = ("dep_flights", "dep_cancelled", "dep_del15", "dep_del15_n", "taxi_out_sum", "taxi_out_n",
                  "arr_flights", "nas_min")
# Congestion sub-metric (config/weights.yaml `congestion` keys) -> raw column in the table.
CONGESTION_COLUMN = {"dep_delay_rate": "dep_delay_rate", "taxi_out": "taxi_out_min",
                     "nas_delay_per_arrival": "nas_min_per_arrival"}

# Region name for a ranking over every US airport in the peer set (no state filter).
NATIONAL = "national"

# Percentiles are k/n and weights are ratios, so exact ties in the score are common.
SCORE_DECIMALS = 9

# Raw value behind each component: the column shown next to its percentile.
# long_haul is ranked on the floored share (`long_haul_share_scored`); the true share is shown too.
# congestion is ranked on `congestion_index`, the weighted mean of its sub-metric percentiles.
RAW_COLUMN = {"utilization": "load_factor", "scale": "passengers", "growth": "growth",
              "long_haul": "long_haul_share_scored", "congestion": "congestion_index"}


def weighted_mean(values: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Row-wise weighted mean over the columns present (not missing), weights renormalized per row;
    missing when no weighted column is present."""
    w = pd.Series(weights, dtype=float)
    present = values.notna().mul(w).sum(axis=1)
    return (values * w).sum(axis=1) / present.where(present > 0)


def congestion_metrics(t: pd.DataFrame, min_flights: int) -> pd.DataFrame:
    """Congestion raw values from On-Time counts (columns ONTIME_COLUMNS).

    dep_delay_rate = DEP_DEL15 / departures reporting it; taxi_out_min = mean taxi-out (origin
    side); nas_min_per_arrival = NAS-delay minutes / arrivals operated (destination side, where BTS
    records causes); cancellation_rate = cancelled / scheduled departures (shown, never scored).
    Airports with fewer than `min_flights` scheduled departures in the window get no congestion
    values: On-Time covers only large reporting carriers, and a rate on a few flights is noise.
    `ontime_status` says which case applies: ok, too_few_flights or no_data.
    """
    dep = t["dep_flights"].fillna(0)
    enough = dep >= min_flights
    out = pd.DataFrame(index=t.index)
    out["ontime_status"] = ["ok" if ok else ("too_few_flights" if n > 0 else "no_data") for ok, n in zip(enough, dep)]
    out["dep_delay_rate"] = _ratio(t["dep_del15"], t["dep_del15_n"]).where(enough)
    out["taxi_out_min"] = _ratio(t["taxi_out_sum"], t["taxi_out_n"]).where(enough)
    out["nas_min_per_arrival"] = _ratio(t["nas_min"], t["arr_flights"]).where(enough)
    out["cancellation_rate"] = _ratio(t["dep_cancelled"], t["dep_flights"]).where(enough)
    return out


@dataclass(frozen=True)
class ScoreResult:
    """Scored airports plus everything an answer must state (window, weights, confidence).

    `table`: one row per airport in the metrics, with raw values, `pct_<component>` percentiles
    (0-1, peer set only), `score` (0-1, weighted mean of the percentiles; null outside the peer
    set), `in_peer_set`, `missing` (components missing for that airport), `weights_used`,
    `confidence` and `confidence_reasons`.
    """
    table: pd.DataFrame
    window: Window
    peer_set_size: int
    weights_configured: dict[str, float]
    weights_used: dict[str, float]          # after dropping unavailable components
    unavailable: dict[str, str]             # component -> reason
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RegionRanking:
    region: str
    states: tuple[str, ...]
    ranked: pd.DataFrame        # peer-set airports of the region, best first, with `rank`
    excluded: pd.DataFrame      # region airports outside the peer set, with `reason`
    result: ScoreResult


def _confidence(window: Window, missing: list[str], unavailable: dict[str, str],
                months_active: int, in_peer_set: bool) -> tuple[str, list[str]]:
    """Deterministic confidence (docs/plan.md, Scoring): high / medium / low with reasons."""
    if window.months < WINDOW_MONTHS:
        return "low", [f"only {window.months} of {WINDOW_MONTHS} months loaded in the window"]
    reasons = []
    if unavailable:
        reasons.append("renormalized without " + ", ".join(sorted(unavailable)))
    if missing:
        reasons.append("missing for this airport: " + ", ".join(missing))
    if months_active < WINDOW_MONTHS:
        reasons.append(f"passenger service in {months_active} of {WINDOW_MONTHS} months")
    if not in_peer_set:
        reasons.append("outside the peer set")
    return ("medium", reasons) if reasons else ("high", [])


def score(metrics: pd.DataFrame, cfg: ScoringConfig, window: Window) -> ScoreResult:
    """Score every peer-set airport; `metrics` has the columns of `store.airport_metrics`.

    Peer set = airports with a US state (`state`, from `airports`) and at least
    `min_passengers_per_year` passenger-service passengers. Each component is a percentile inside
    the peer set. Components not built yet (NOT_BUILT), and congestion when `metrics` has no
    On-Time columns (`store.scoring_metrics` adds them once On-Time is loaded), are dropped and
    the remaining weights renormalized. An airport missing a component (e.g. no previous-year
    passengers, so no growth; no On-Time data or fewer than `min_ontime_flights`, so no
    congestion) is scored on the components it has, with its own renormalized weights and
    medium confidence. Ties in the score are broken by IATA code.
    """
    unavailable = dict(NOT_BUILT)
    has_ontime = all(c in metrics.columns for c in ONTIME_COLUMNS)
    if not has_ontime:
        unavailable["congestion"] = ONTIME_NOT_LOADED
    if window.prev_months < WINDOW_MONTHS:
        unavailable["growth"] = (f"previous 12 months only partly loaded "
                                 f"({window.prev_months} of {WINDOW_MONTHS})")
    available = [c for c in cfg.weights if c not in unavailable]
    weights_used = renormalize(cfg.weights, available)
    notes = [f"Score uses {', '.join(available)}; weights renormalized without {', '.join(sorted(unavailable))}."
             if unavailable else "All components available."]

    t = metrics.copy()
    t["load_factor"] = load_factor(t["passengers"], t["seats"])
    t["long_haul_share"] = long_haul_share(t["long_haul_departures"], t["departures"])
    t["long_haul_share_scored"] = floor_share(t["long_haul_share"], cfg.long_haul_min_share)
    t["growth"] = growth(t["passengers"], t["prev_passengers"])
    t["in_peer_set"] = t["state"].notna() & (t["passengers"] >= cfg.min_passengers_per_year)

    peer = t["in_peer_set"]
    if has_ontime:
        t = t.join(congestion_metrics(t, cfg.min_ontime_flights))
        # Each sub-metric is a percentile in the peer set; the index is their weighted mean
        # (renormalized when one is missing), then ranked like every other component below.
        subs = pd.DataFrame({k: percentile_rank(t.loc[peer, col].astype(float))
                             for k, col in CONGESTION_COLUMN.items()}).reindex(t.index)
        for k in CONGESTION_COLUMN:
            t[f"pct_{k}"] = subs[k]
        t["congestion_index"] = weighted_mean(subs, cfg.congestion_weights)
    for component in available:
        t[f"pct_{component}"] = percentile_rank(t.loc[peer, RAW_COLUMN[component]].astype(float))

    # Weighted mean over the components each airport has: dividing by the sum of the weights
    # present renormalizes per airport. No weighted component present -> no score.
    pct = t[[f"pct_{c}" for c in available]].set_axis(available, axis=1)
    w = pd.Series(weights_used)
    present_weight = pct.notna().mul(w).sum(axis=1)
    # Rounded so float noise (1e-16) never decides an exact tie; ties then break by IATA.
    t["score"] = ((pct * w).sum(axis=1) / present_weight.where(peer & (present_weight > 0))).round(SCORE_DECIMALS)
    t["missing"] = [[c for c in available if pd.isna(pct.at[i, c])] if peer[i] else [] for i in t.index]
    t["weights_used"] = [{c: wt / present_weight[i] for c, wt in weights_used.items() if c not in miss}
                         if pd.notna(t.at[i, "score"]) else {} for i, miss in t["missing"].items()]
    conf = [_confidence(window, miss, unavailable, months, ok)
            for miss, months, ok in zip(t["missing"], t["months_active"], peer)]
    t["confidence"] = [c for c, _ in conf]
    t["confidence_reasons"] = [r for _, r in conf]

    t = t.sort_values(["score", "iata"], ascending=[False, True], na_position="last", kind="stable")
    return ScoreResult(t.reset_index(drop=True), window, int(peer.sum()), dict(cfg.weights),
                       weights_used, unavailable, notes)


def rank(result: ScoreResult, region: str, cfg: ScoringConfig) -> RegionRanking:
    """Rank the peer-set airports of a configured region (or NATIONAL: every airport with a US
    state, i.e. the whole peer set); list the region's other airports as excluded.

    Percentiles stay national (peer set); the region only filters. Region membership is by
    state, from `cfg.regions`; NATIONAL has no state list (`states` is empty).
    """
    if region == NATIONAL:
        states: tuple[str, ...] = ()
        t = result.table[result.table["state"].notna()]
    elif region in cfg.regions:
        states = cfg.regions[region]
        t = result.table[result.table["state"].isin(states)]
    else:
        raise ValueError(f"Unknown region {region!r}; configured: {sorted(cfg.regions)} or {NATIONAL!r}")
    ranked = t[t["in_peer_set"]].copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    excluded = t[~t["in_peer_set"]].sort_values("passengers", ascending=False, kind="stable").copy()
    excluded["reason"] = [f"{p:,} passenger-service passengers, below the {cfg.min_passengers_per_year:,} floor"
                          for p in excluded["passengers"]]
    return RegionRanking(region, states, ranked.reset_index(drop=True), excluded.reset_index(drop=True), result)


# --- comparison margins (compare_airports) -----------------------------------------------------

def z_value(confidence_level: float) -> float:
    """Two-sided normal quantile for a confidence level (0.95 -> 1.96)."""
    return NormalDist().inv_cdf(0.5 + confidence_level / 2)


def proportion_margin(share: float | None, n: float | None, z: float) -> float | None:
    """Margin of error of a share of flights (normal approximation of the binomial): z * sqrt(p(1-p)/n).

    Used for On-Time rates: On-Time covers only the large reporting carriers, so its flights are a
    sample of the airport's traffic. Missing when the share or the flight count is missing."""
    if share is None or n is None or share != share or n != n or n <= 0:
        return None
    return z * math.sqrt(share * (1 - share) / n)


def mean_margin(monthly_values: Iterable[float], z: float) -> float | None:
    """Margin of error of a mean in minutes from its month-to-month spread: z * sd / sqrt(months).

    Only monthly sums are stored, not per-flight values, so the spread of the monthly means is the
    available measure of noise (it also absorbs seasonality, so it is conservative). Missing with
    fewer than two months."""
    values = [float(v) for v in monthly_values if v is not None and v == v]
    if len(values) < 2:
        return None
    return z * stdev(values) / math.sqrt(len(values))


def compare_values(a: float | None, b: float | None, margin_a: float | None,
                   margin_b: float | None) -> str | None:
    """'a' or 'b' (the higher value) when the difference exceeds the combined margin
    sqrt(margin_a^2 + margin_b^2), 'same' when it does not, None when a value or margin is missing.
    The rule never names a higher airport inside the margin."""
    if None in (a, b, margin_a, margin_b):
        return None
    if abs(a - b) <= math.hypot(margin_a, margin_b):
        return "same"
    return "a" if a > b else "b"
