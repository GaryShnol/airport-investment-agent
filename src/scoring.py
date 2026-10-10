"""Deterministic airport scoring. Pure functions: no database, no network, no LLM.

Definitions and methodology: docs/plan.md (Definitions, Scoring). Weights, thresholds and
regions come only from `ScoringConfig` (config/weights.yaml via src/config.py).
Helpers take pandas Series (one value per airport) and return Series aligned to the input.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd


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
