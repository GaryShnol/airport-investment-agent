"""Agent tools. Each tool returns a small JSON-serializable envelope; the LLM only phrases it.

Envelope (docs/plan.md, Tools and envelope):
    {"data": ..., "data_period": "2025-01..2025-12", "freshness": "current|behind|unknown",
     "assumptions": [...], "confidence": "high|medium|low"}
Errors use the same shape with `data` null, an `error` message and `candidates` to choose from;
they are returned, never raised, so the model can ask the user instead of crashing.

The functions below take an open connection (tests pass an in-memory DB); `make_tools` wraps
them as LangChain tools that open the DB read-only per call. SQL lives in src/store.py, all
numbers in src/scoring.py. Shares are returned as percentages rounded for display.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Callable

from langchain_core.tools import BaseTool, tool

from src import scoring, store
from src.config import ScoringConfig

CONFIDENCE_ORDER = ("high", "medium", "low")
PASSENGER_SERVICE = "passenger service = aircraft config passenger or combi (BTS T-100), departures > 0"
# Display cap for candidate and excluded lists: keeps envelopes small for local models' short context.
MAX_LISTED = 5
LOAD_FACTOR_PROXY = "load factor = passengers / seats on passenger service: a demand-pressure proxy, not terminal utilization"


def _pct(value, digits: int = 1) -> float | None:
    """Share (0-1) as a rounded percentage; None for missing."""
    return None if value is None or value != value else round(float(value) * 100, digits)


def freshness(conn: sqlite3.Connection) -> str:
    """current / behind / unknown over all loaded sources (behind wins, then unknown)."""
    states = []
    for row in store.meta_rows(conn):
        if not row["upstream_latest"]:
            states.append("unknown")
        elif row["last_period"] and row["upstream_latest"] > row["last_period"]:
            states.append("behind")
        else:
            states.append("current")
    for state in ("behind", "unknown"):
        if state in states:
            return state
    return "current" if states else "unknown"


def _envelope(conn: sqlite3.Connection, window: store.Window | None, data, assumptions: list[str],
              confidence: str) -> dict:
    return {"data": data, "data_period": window.label if window else None, "freshness": freshness(conn),
            "assumptions": assumptions, "confidence": confidence}


def _error(conn: sqlite3.Connection | None, message: str, candidates: list | None = None, **extra) -> dict:
    return {"data": None, "error": message, "candidates": candidates or [], **extra,
            "data_period": None, "freshness": freshness(conn) if conn else "unknown",
            "assumptions": [], "confidence": "low"}


def _window(conn: sqlite3.Connection) -> store.Window:
    return store.analysis_window(conn)


# --- data_status -------------------------------------------------------------------------

def data_status(conn: sqlite3.Connection, cfg: ScoringConfig) -> dict:
    """What data is loaded, the analysis window, freshness and the scoring setup."""
    window = _window(conn)
    data = {
        "sources": store.meta_rows(conn),
        "months_in_window": window.months,
        "previous_window": f"{window.prev_start}..{window.prev_end} ({window.prev_months} months, for growth)",
        "score_components_not_built": dict(scoring.NOT_BUILT),
        "weights_configured": cfg.weights,
        "regions": {name: list(states) for name, states in cfg.regions.items()},
    }
    assumptions = ["freshness is unknown until `ingest --check` compares with upstream"] \
        if freshness(conn) == "unknown" else []
    confidence = "high" if window.months == store.WINDOW_MONTHS else "low"
    return _envelope(conn, window, data, assumptions, confidence)


# --- resolve_airport ---------------------------------------------------------------------

def _brief(row: dict) -> dict:
    return {"iata": row["iata"], "name": row["name"], "state": row["state"]}


def resolve_airport(conn: sqlite3.Connection, cfg: ScoringConfig, text: str) -> dict:
    """Name, city or code -> IATA. Order: configured metro alias (ambiguous, with a default),
    exact IATA, airport name, then municipality. Only airports with passenger service in the
    window count for name and city matches; several matches -> ambiguous with candidates."""
    query = (text or "").strip()
    if not query:
        return _error(conn, "Empty airport name")
    window = _window(conn)

    alias = cfg.metro_aliases.get(query.upper())
    if alias:
        candidates = []
        for code in alias["candidates"]:
            rows = store.find_airports(conn, "iata", code, window)
            candidates.append(_brief(rows[0]) if rows else {"iata": code})
        data = {"query": query, "ambiguous": True, "default": alias["default"], "candidates": candidates}
        assumptions = [f"'{query}' is ambiguous; default definition {query} = {alias['default']} (configured)"]
        return _envelope(conn, window, data, assumptions, "medium")

    if len(query) == 3 and query.isalpha():
        rows = store.find_airports(conn, "iata", query, window)
        if rows:
            return _envelope(conn, window, {"query": query, "ambiguous": False, **_brief(rows[0])}, [], "high")

    for column in ("name", "municipality"):
        rows = [r for r in store.find_airports(conn, column, query, window) if r["passengers"] > 0]
        if len(rows) == 1:
            return _envelope(conn, window, {"query": query, "ambiguous": False, **_brief(rows[0])},
                             [f"matched on airport {column}"], "high")
        if len(rows) > 1:
            data = {"query": query, "ambiguous": True, "default": None,
                    "candidates": [_brief(r) for r in rows[:MAX_LISTED]]}
            return _envelope(conn, window, data, [f"several airports match on {column}; ask the user"], "low")
    return _error(conn, f"No airport with passenger service matches '{query}'")


def _resolve_code(conn: sqlite3.Connection, cfg: ScoringConfig, text: str) -> tuple[str | None, dict | None]:
    """IATA code for `text`, or an error envelope (ambiguous or unknown) to return as is."""
    result = resolve_airport(conn, cfg, text)
    data = result["data"]
    if data is None:
        return None, result
    if data["ambiguous"]:
        return None, _error(conn, f"'{text}' is ambiguous; ask the user or state the definition used",
                            [c["iata"] for c in data["candidates"]], default=data["default"])
    return data["iata"], None


# --- airport_metrics ---------------------------------------------------------------------

def _shares(counts: dict) -> dict[int, float | None]:
    total = counts["departures"]
    return {t: (_pct(n / total) if total else None) for t, n in counts["long_haul"].items()}


def airport_metrics(conn: sqlite3.Connection, cfg: ScoringConfig, airport: str,
                    threshold_mi: int | None = None) -> dict:
    """Passengers, load factor, growth and long-haul share (all flights and passenger service)."""
    code, err = _resolve_code(conn, cfg, airport)
    if err:
        return err
    window = _window(conn)
    threshold = int(threshold_mi) if threshold_mi else cfg.long_haul_miles
    if threshold <= 0:
        return _error(conn, "threshold_mi must be a positive number of miles")

    result = scoring.score(store.airport_metrics(conn, window, cfg.long_haul_miles), cfg, window)
    rows = result.table[result.table["iata"] == code]
    if rows.empty:
        return _error(conn, f"{code} has no passenger service in {window.label}")
    row = rows.iloc[0]

    thresholds = sorted({threshold, *cfg.long_haul_sensitivity_miles})
    every = store.long_haul_counts(conn, code, window, thresholds)
    pax = store.long_haul_counts(conn, code, window, thresholds, store.PASSENGER_CONFIGS)
    every_pct, pax_pct = _shares(every), _shares(pax)
    data = {
        "airport": code, "name": row["name"], "state": row["state"],
        "passengers": int(row["passengers"]), "seats": int(row["seats"]),
        "departures_passenger_service": int(row["departures"]),
        "load_factor_pct": _pct(row["load_factor"]),
        "growth_pct": _pct(row["growth"]), "previous_passengers": int(row["prev_passengers"]),
        "months_active": int(row["months_active"]), "in_peer_set": bool(row["in_peer_set"]),
        "long_haul": {
            "threshold_mi": threshold,
            "all_flights": {"departures": every["departures"], "long_haul_departures": every["long_haul"][threshold],
                            "share_pct": every_pct[threshold]},
            "passenger_service": {"departures": pax["departures"], "long_haul_departures": pax["long_haul"][threshold],
                                  "share_pct": pax_pct[threshold]},
            "sensitivity": [{"threshold_mi": t, "all_flights_pct": every_pct[t], "passenger_service_pct": pax_pct[t]}
                            for t in thresholds],
        },
    }
    assumptions = [
        PASSENGER_SERVICE, LOAD_FACTOR_PROXY,
        "all flights = every aircraft config, including cargo and undocumented",
        f"long haul = segment distance >= {threshold} mi, share of departures",
        f"growth = passengers {window.label} vs {window.prev_start}..{window.prev_end}",
    ]
    return _envelope(conn, window, data, assumptions, row["confidence"])


# --- rank_airports -----------------------------------------------------------------------

def _region_key(region: str) -> str:
    return (region or "").strip().lower().replace("-", "_").replace(" ", "_")


def rank_airports(conn: sqlite3.Connection, cfg: ScoringConfig, region: str, top_n: int = 10) -> dict:
    """Rank a configured region's peer-set airports by the deterministic score."""
    key = _region_key(region)
    if key not in cfg.regions:
        return _error(conn, f"Unknown region '{region}'", sorted(cfg.regions))
    if int(top_n) < 1:
        return _error(conn, "top_n must be at least 1")
    window = _window(conn)
    result = scoring.score(store.airport_metrics(conn, window, cfg.long_haul_miles), cfg, window)
    ranking = scoring.rank(result, key, cfg)
    components = list(result.weights_used)

    ranked = []
    for _, r in ranking.ranked.head(int(top_n)).iterrows():
        raw = {"utilization": {"load_factor_pct": _pct(r["load_factor"])},
               "scale": {"passengers": int(r["passengers"])},
               "growth": {"growth_pct": _pct(r["growth"])},
               "long_haul": {"share_pct": _pct(r["long_haul_share"])}}
        item = {"rank": int(r["rank"]), "iata": r["iata"], "name": r["name"], "score": round(float(r["score"]), 3),
                "components": {c: {**raw[c], "percentile": _pct(r[f"pct_{c}"], 0)} for c in components},
                "confidence": r["confidence"]}
        if r["missing"]:
            item["missing"] = r["missing"]
            item["weights_used"] = {c: round(w, 3) for c, w in r["weights_used"].items()}
        ranked.append(item)

    scores = [x["score"] for x in ranked]
    ties = sorted({s for s in scores if scores.count(s) > 1})
    data = {
        "region": key, "states": list(ranking.states), "peer_set_size": result.peer_set_size,
        "ranked": ranked,
        "ranked_total": len(ranking.ranked),
        # Largest first; the rest are smaller still and below the same floor.
        "excluded": [f"{r['iata']}: {r['reason']}" for _, r in ranking.excluded.head(MAX_LISTED).iterrows()],
        "excluded_total": len(ranking.excluded),
        "weights_configured": cfg.weights,
        "weights_used": {c: round(w, 3) for c, w in result.weights_used.items()},
        "components_unavailable": result.unavailable,
        "ties": [[x["iata"] for x in ranked if x["score"] == s] for s in ties],
    }
    assumptions = [
        "score = weighted mean of national percentiles in the peer set (0-1); a capacity-pressure proxy, "
        "not a measured terminal need (no gate or terminal-area data)",
        f"peer set = US airports with >= {cfg.min_passengers_per_year:,} passenger-service passengers in the window",
        PASSENGER_SERVICE, LOAD_FACTOR_PROXY,
        f"long haul >= {cfg.long_haul_miles} mi; shares under {_pct(cfg.long_haul_min_share)}% count as 0 in the score",
        *result.notes,
    ]
    levels = [x["confidence"] for x in ranked] or ["low"]
    confidence = max(levels, key=CONFIDENCE_ORDER.index)
    return _envelope(conn, window, data, assumptions, confidence)


# --- LangChain wrappers ------------------------------------------------------------------

def make_tools(connect: Callable[[], sqlite3.Connection], cfg: ScoringConfig) -> list[BaseTool]:
    """LangChain tools over `connect` (a fresh read-only connection per call, closed after).

    Any exception becomes an error envelope: the model sees a message, the chat keeps going.
    """
    def run(fn, *args) -> dict:
        try:
            with closing(connect()) as conn:
                return fn(conn, cfg, *args)
        except Exception as exc:  # noqa: BLE001 - reported to the model, never raised
            return _error(None, f"{type(exc).__name__}: {exc}")

    @tool("data_status")
    def data_status_tool() -> dict:
        """Loaded data sources, the analysis period, freshness, configured weights and regions."""
        return run(lambda conn, c: data_status(conn, c))

    @tool("resolve_airport")
    def resolve_airport_tool(text: str) -> dict:
        """Turn an airport name, city or code (e.g. 'Anchorage', 'Santa Ana', 'LA') into an IATA code.
        Returns ambiguous=true with candidates and a default when the name covers several airports."""
        return run(resolve_airport, text)

    @tool("airport_metrics")
    def airport_metrics_tool(airport: str, threshold_mi: int | None = None) -> dict:
        """Metrics for one airport (IATA code or name): passengers, load factor, growth, and the
        long-haul share of departures for all flights and for passenger service, at the threshold
        (default from config) and at each sensitivity threshold."""
        return run(airport_metrics, airport, threshold_mi)

    @tool("rank_airports", description=(
        "Rank airports in a region as terminal-expansion candidates: scores, component percentiles "
        "with raw values, weights configured and used, excluded airports, confidence. "
        f"region must be one of: {', '.join(sorted(cfg.regions))}."))
    def rank_airports_tool(region: str, top_n: int = 10) -> dict:
        return run(rank_airports, region, top_n)

    return [data_status_tool, resolve_airport_tool, airport_metrics_tool, rank_airports_tool]
