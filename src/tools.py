"""Agent tools. Each tool returns a small JSON-serializable envelope; the LLM only phrases it.

Envelope (docs/plan.md, Tools and envelope):
    {"headline": "...", "data": ..., "facts": [...], "data_period": "2025-01..2025-12", "freshness": "current|behind|unknown",
     "assumptions": [...], "confidence": "high|medium|low"}
`headline` is one deterministic sentence with the answer in plain words (rendered above the facts
card; the model comments on it, never restates it). `facts` is a list of {label, value, required} with pre-formatted strings the model copies verbatim;
src/agent.py `ensure_facts` appends any required fact the answer leaves out, so mandatory content
does not depend on the model following the prompt.
Errors use the same shape with `data` null, an `error` message and `candidates` to choose from;
they are returned, never raised, so the model can ask the user instead of crashing.

The functions below take an open connection (tests pass an in-memory DB); `make_tools` wraps
them as LangChain tools that open the DB read-only per call. SQL lives in src/store.py, all
numbers in src/scoring.py. Shares are returned as percentages rounded for display.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Callable, Literal, get_args

from langchain_core.tools import BaseTool, tool

from src import scoring, store
from src.config import ScoringConfig

CONFIDENCE_ORDER = ("high", "medium", "low")
PASSENGER_SERVICE = "passenger service = aircraft config passenger or combi (BTS T-100), departures > 0"
# Display cap for candidate and excluded lists: keeps envelopes small for local models' short context.
MAX_LISTED = 5
# airport_metrics `focus`: which facts are required (ensure_facts appends them if the answer omits
# them). All facts are always returned; focus only decides what the answer must mention.
Focus = Literal["long_haul", "load_factor", "growth", "all"]
FOCUS_VALUES = get_args(Focus)
# Ranked airports named in a ranking headline; the table lists the rest.
HEADLINE_TOP = 3
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


def fact(label: str, value: str, required: bool = False) -> dict:
    return {"label": label, "value": value, "required": required}


def _miles(mi: int) -> str:
    return f"{mi:,} mi"


def _share(pct: float | None) -> str:
    return "n/a" if pct is None else f"{pct}%"


def _envelope(conn: sqlite3.Connection, window: store.Window | None, data, assumptions: list[str],
              confidence: str, facts: list[dict] | None = None, *, headline: str) -> dict:
    """Standard envelope. Data period and confidence are required facts for every tool."""
    period = window.label if window else None
    base = [fact("Data period", period, True)] if period else []
    base.append(fact("Confidence", confidence, True))
    return {"headline": headline, "facts": [*(facts or []), *base], "data": data, "data_period": period,
            "freshness": freshness(conn), "assumptions": assumptions, "confidence": confidence}


def _error(conn: sqlite3.Connection | None, message: str, candidates: list | None = None,
           headline: str | None = None, **extra) -> dict:
    """Error envelope; the headline is the message unless a fuller sentence is given."""
    return {"headline": headline or message, "data": None, "facts": [], "error": message,
            "candidates": candidates or [], **extra, "data_period": None,
            "freshness": freshness(conn) if conn else "unknown", "assumptions": [], "confidence": "low"}


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
    facts = [fact("Freshness", freshness(conn)),
             fact("Months loaded in the window", f"{window.months} of {store.WINDOW_MONTHS}")]
    headline = (f"Data loaded for {window.label} ({window.months} of {store.WINDOW_MONTHS} months); "
                f"freshness {freshness(conn)}.")
    return _envelope(conn, window, data, assumptions, confidence, facts, headline=headline)


# --- resolve_airport ---------------------------------------------------------------------

def _brief(row: dict) -> dict:
    return {"iata": row["iata"], "name": row["name"], "state": row["state"]}


def _airport_fact(row: dict) -> dict:
    return fact("Airport", f"{row['iata']} ({row['name']})")


def _resolved(query: str, row: dict) -> str:
    return f"'{query}' = {row['iata']} ({row['name']}, {row['state'] or 'no US state'})."


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
        facts = [fact("Default definition", f"{query} = {alias['default']}"),
                 fact("Other candidates", ", ".join(c["iata"] for c in candidates if c["iata"] != alias["default"]))]
        others = [c["iata"] for c in candidates if c["iata"] != alias["default"]]
        headline = f"'{query}' covers several airports; default {query} = {alias['default']}, others: {', '.join(others)}."
        return _envelope(conn, window, data, assumptions, "medium", facts, headline=headline)

    if len(query) == 3 and query.isalpha():
        rows = store.find_airports(conn, "iata", query, window)
        if rows:
            return _envelope(conn, window, {"query": query, "ambiguous": False, **_brief(rows[0])}, [], "high",
                             [_airport_fact(rows[0])], headline=_resolved(query, rows[0]))

    for column in ("name", "municipality"):
        rows = [r for r in store.find_airports(conn, column, query, window) if r["passengers"] > 0]
        if len(rows) == 1:
            return _envelope(conn, window, {"query": query, "ambiguous": False, **_brief(rows[0])},
                             [f"matched on airport {column}"], "high", [_airport_fact(rows[0])],
                             headline=_resolved(query, rows[0]))
        if len(rows) > 1:
            data = {"query": query, "ambiguous": True, "default": None,
                    "candidates": [_brief(r) for r in rows[:MAX_LISTED]]}
            codes = ", ".join(c["iata"] for c in data["candidates"])
            return _envelope(conn, window, data, [f"several airports match on {column}; ask the user"], "low",
                             [fact("Candidates", codes)],
                             headline=f"'{query}' matches several airports: {codes}.")
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
                    threshold_mi: int | None = None, focus: str = "all") -> dict:
    """Passengers, load factor, growth and long-haul share (all flights and passenger service).

    `focus` (FOCUS_VALUES) marks the facts the question is about as required, so a load-factor
    question does not get the long-haul sensitivity appended; "all" requires every metric."""
    focus = (focus or "all").strip().lower()
    if focus not in FOCUS_VALUES:
        return _error(conn, f"focus must be one of: {', '.join(FOCUS_VALUES)}")
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
    need = {f: focus in (f, "all") for f in FOCUS_VALUES}
    facts = [
        fact("Airport", f"{code} ({row['name']})"),
        fact("Long-haul threshold", _miles(threshold), need["long_haul"]),
        fact(f"Long-haul share >= {_miles(threshold)}, all flights (including cargo)",
             _share(every_pct[threshold]), need["long_haul"]),
        fact(f"Long-haul share >= {_miles(threshold)}, passenger service", _share(pax_pct[threshold]),
             need["long_haul"]),
        fact(f"Departures >= {_miles(threshold)}, all flights", _of(every, threshold)),
        fact(f"Departures >= {_miles(threshold)}, passenger service", _of(pax, threshold)),
        *[fact(f"Sensitivity: long-haul share >= {_miles(t)}",
               f"all flights {_share(every_pct[t])}, passenger service {_share(pax_pct[t])}", need["long_haul"])
          for t in thresholds if t != threshold],
        fact("Passengers (passenger service)", f"{data['passengers']:,}"),
        fact("Load factor (passenger service, a demand-pressure proxy)", _share(data["load_factor_pct"]),
             need["load_factor"]),
        fact(f"Passenger growth vs {window.prev_start}..{window.prev_end}", _share(data["growth_pct"]),
             need["growth"]),
    ]
    return _envelope(conn, window, data, assumptions, row["confidence"], facts,
                     headline=_metrics_headline(data, focus, window))


def _of(counts: dict, threshold: int) -> str:
    return f"{counts['long_haul'][threshold]:,} of {counts['departures']:,}"


def _signed(pct: float | None) -> str:
    return "n/a" if pct is None else f"{pct:+.1f}%"


def _metrics_headline(data: dict, focus: str, window: store.Window) -> str:
    """One sentence per focus; long haul states the definition and both bases with counts."""
    code, lh = data["airport"], data["long_haul"]
    every, pax = lh["all_flights"], lh["passenger_service"]
    long_haul = (f"{_share(every['share_pct'])} of all departures from {code} (incl. cargo) are "
                 f">= {_miles(lh['threshold_mi'])} ({every['long_haul_departures']:,} of {every['departures']:,}); "
                 f"passenger service: {_share(pax['share_pct'])} "
                 f"({pax['long_haul_departures']:,} of {pax['departures']:,}).")
    load = (f"{code} load factor (passenger service) is {_share(data['load_factor_pct'])}: "
            f"{data['passengers']:,} passengers on {data['seats']:,} seats in {window.label}.")
    growth = (f"{code} passengers changed {_signed(data['growth_pct'])} in {window.label} "
              f"({data['passengers']:,}) vs {window.prev_start}..{window.prev_end} ({data['previous_passengers']:,}).")
    summary = (f"{code} in {window.label}: {data['passengers']:,} passengers (passenger service), load factor "
               f"{_share(data['load_factor_pct'])}, growth {_signed(data['growth_pct'])}.")
    return {"long_haul": long_haul, "load_factor": load, "growth": growth}.get(focus, f"{summary} {long_haul}")


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
        # Only the weights actually applied: listing the configured ones too (with unavailable
        # components) made small models mix the two sets. Configured weights are in data_status.
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
    weights_label = "Weights used" + (f" (renormalized without {', '.join(sorted(result.unavailable))})"
                                      if result.unavailable else "")
    facts = [fact(weights_label, ", ".join(f"{c} {w}" for c, w in data["weights_used"].items()), True),
             *[fact(f"#{x['rank']} {x['iata']} ({x['name']})", _ranked_summary(x)) for x in ranked]]
    top = ", ".join(f"{x['rank']} {x['iata']}" for x in ranked[:HEADLINE_TOP])
    headline = (f"{key} terminal-expansion ranking by score: {top or 'no airport in the peer set'} "
                f"({data['ranked_total']} ranked, {data['excluded_total']} excluded).")
    return _envelope(conn, window, data, assumptions, confidence, facts, headline=headline)


COMPONENT_LABELS = {"utilization": "utilization (load factor)", "scale": "scale (annual passengers)",
                    "growth": "growth (passengers vs previous 12 months)", "long_haul": "long-haul share"}


def _ranked_summary(item: dict) -> str:
    """One line per ranked airport: score, then each component's raw value and percentile."""
    parts = [f"score {item['score']}"]
    for c, v in item["components"].items():
        raw = next(val for key, val in v.items() if key != "percentile")
        shown = f"{raw:,}" if isinstance(raw, int) else _share(raw)
        pct = "n/a" if v["percentile"] is None else f"{v['percentile']:.0f}"
        parts.append(f"{COMPONENT_LABELS.get(c, c)} {shown}, percentile {pct}")
    return "; ".join(parts)


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
    def airport_metrics_tool(airport: str, threshold_mi: int | None = None,
                             focus: Focus = "all") -> dict:
        """Metrics for one airport (IATA code or name): passengers, load factor, growth, and the
        long-haul share of departures for all flights and for passenger service, at the threshold
        (default from config) and at each sensitivity threshold. Set focus to what the question
        asks about (long_haul, load_factor, growth); use all when it asks for several."""
        return run(airport_metrics, airport, threshold_mi, focus)

    @tool("rank_airports", description=(
        "Rank airports in a region as terminal-expansion candidates: scores, component percentiles "
        "with raw values, the weights used, excluded airports, confidence. "
        f"region must be one of: {', '.join(sorted(cfg.regions))}."))
    def rank_airports_tool(region: str, top_n: int = 10) -> dict:
        return run(rank_airports, region, top_n)

    return [data_status_tool, resolve_airport_tool, airport_metrics_tool, rank_airports_tool]
