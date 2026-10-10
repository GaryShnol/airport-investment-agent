"""Agent tools. Each tool returns a small JSON-serializable envelope; the LLM only phrases it.

Envelope (docs/plan.md, Tools and envelope):
    {"headline": "...", "data": ..., "facts": [...], "data_period": "2025-01..2025-12", "freshness": "fresh|stale (upstream has ...)|unknown",
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
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Callable, Literal, get_args

from langchain_core.tools import BaseTool, tool

from src import census, scoring, store
from src.config import ScoringConfig

CONFIDENCE_ORDER = ("high", "medium", "low")
PASSENGER_SERVICE = "passenger service = aircraft config passenger or combi (BTS T-100), departures > 0"
# Display cap for candidate and excluded lists: keeps envelopes small for local models' short context.
MAX_LISTED = 5
# airport_metrics `topic`: which facts, data and assumptions are returned, and which facts are
# required (ensure_facts appends them if the answer omits them). demand = passengers, seats, load factor.
Topic = Literal["long_haul", "demand", "growth", "congestion", "all"]
TOPIC_VALUES = get_args(Topic)
# resolve_airport suggestions: difflib ratio needed to suggest an airport for an unmatched name.
SUGGEST_MIN_SIMILARITY = 0.6
# Ranked airports named in a ranking headline; the table lists the rest.
HEADLINE_TOP = 3
LOAD_FACTOR_PROXY = "load factor = passengers / seats on passenger service: a demand-pressure proxy, not terminal utilization"
ONTIME_COVERAGE = ("On-Time covers domestic flights of the large reporting carriers only, not every flight at the "
                   "airport; airports with few reported flights get no congestion value")
CONGESTION_SIDES = ("departure delay rate (>= 15 min), taxi-out and cancellations are measured on departures from "
                    "the airport; NAS-delay minutes on arrivals to it (BTS records delay causes for arrivals)")
CANCELLATIONS_NOT_SCORED = "cancellations are shown, not scored (mostly weather, not a capacity signal)"


def _pct(value, digits: int = 1) -> float | None:
    """Share (0-1) as a rounded percentage; None for missing."""
    return None if value is None or value != value else round(float(value) * 100, digits)


def freshness(conn: sqlite3.Connection) -> str:
    """fresh / stale (upstream has ...) / unknown over the loaded sources, from the last
    `ingest --check` stored in `meta` (store.overall_freshness). Never checks upstream itself."""
    return store.overall_freshness(store.meta_rows(conn))


STALE_WARNING = ("data is stale: upstream has newer data than loaded ({}); figures use the loaded data, "
                 "run `uv run python -m src.ingest` to refresh")


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
    fresh = freshness(conn)
    if fresh.startswith("stale"):  # still answered, with a warning (CLAUDE.md)
        assumptions = [*assumptions, STALE_WARNING.format(fresh.removeprefix("stale (upstream has ").rstrip(")"))]
    return {"headline": headline, "facts": [*(facts or []), *base], "data": data, "data_period": period,
            "freshness": fresh, "assumptions": assumptions, "confidence": confidence}


def _error(conn: sqlite3.Connection | None, message: str, candidates: list | None = None,
           headline: str | None = None, **extra) -> dict:
    """Error envelope; the headline is the message unless a fuller sentence is given."""
    return {"headline": headline or message, "data": None, "facts": [], "error": message,
            "candidates": candidates or [], **extra, "data_period": None,
            "freshness": freshness(conn) if conn else "unknown", "assumptions": [], "confidence": "low"}


def _window(conn: sqlite3.Connection) -> store.Window:
    return store.analysis_window(conn)


def _scoring_metrics(conn: sqlite3.Connection, cfg: ScoringConfig, window: store.Window):
    return store.scoring_metrics(conn, window, cfg.long_haul_miles, (cfg.census_vintage, cfg.census_base_vintage))


# --- data_status -------------------------------------------------------------------------

def _sources_missing(conn: sqlite3.Connection, cfg: ScoringConfig) -> dict[str, str]:
    """Components whose source is not loaded (scored without them, weights renormalized)."""
    missing = {}
    if not store.ontime_loaded(conn):
        missing["congestion"] = scoring.ONTIME_NOT_LOADED
    if not store.census_loaded(conn, (cfg.census_vintage, cfg.census_base_vintage)):
        missing["demand_context"] = scoring.census_not_loaded(cfg)
    return missing


def data_status(conn: sqlite3.Connection, cfg: ScoringConfig) -> dict:
    """What data is loaded, the analysis window, freshness and the scoring setup."""
    window = _window(conn)
    data = {
        "sources": store.meta_rows(conn),
        "months_in_window": window.months,
        "previous_window": f"{window.prev_start}..{window.prev_end} ({window.prev_months} months, for growth)",
        "score_components_unavailable": _sources_missing(conn, cfg),
        "weights_configured": cfg.weights,
        "regions": {name: list(states) for name, states in cfg.regions.items()},
    }
    assumptions = ["freshness is unknown until `ingest --check` compares with upstream (or it could not tell)"] \
        if freshness(conn) == store.UNKNOWN else []
    confidence = "high" if window.months == store.WINDOW_MONTHS else "low"
    facts = [fact("Freshness", freshness(conn)),
             *[fact(f"Freshness: {r['source']}", f"{store.source_freshness(r)}; loaded {r['last_period']}; "
                                                  f"checked {r['checked_at'] or 'never'}")
               for r in data["sources"] if r["last_period"]],
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


def _similarity(query: str, row: dict) -> float:
    """Best difflib ratio of the query against the airport's city, name and (for 3 letters) code."""
    q = query.lower()
    fields = [row["municipality"], row["name"]] + ([row["iata"]] if len(q) == 3 else [])
    return max((SequenceMatcher(None, q, f.lower()).ratio() for f in fields if f), default=0.0)


def _suggestions(conn: sqlite3.Connection, query: str, window: store.Window) -> list[dict]:
    """Closest airports with passenger service (typos such as 'Anchorge'), most similar first."""
    scored = [(_similarity(query, r), r) for r in store.served_airports(conn, window)] if query else []
    close = sorted(((s, r) for s, r in scored if s >= SUGGEST_MIN_SIMILARITY),
                   key=lambda sr: (-sr[0], -sr[1]["passengers"], sr[1]["iata"]))
    return [_brief(r) for _, r in close[:MAX_LISTED]]


def _no_match(conn: sqlite3.Connection, query: str, window: store.Window | None) -> dict:
    """Explicit "no match" envelope: a required fact, suggestions and a hint, never an empty result."""
    suggestions = _suggestions(conn, query, window) if window else []
    codes = [s["iata"] for s in suggestions]
    message = f"No airport with passenger service matches '{query}'" if query else "Empty airport name"
    named = ", ".join(f"{s['iata']} ({s['name']})" for s in suggestions)
    hint = (f"Did you mean: {named}?" if suggestions
            else "Try the 3-letter IATA code (e.g. SFO) or the city the airport serves.")
    return _error(conn, message, codes, headline=f"{message}. {hint}", suggestions=suggestions,
                  facts=[fact("Match", f"no match for '{query}'", True),
                         fact("Suggestions", ", ".join(codes) or "none close; " + hint)])


def _matches(conn: sqlite3.Connection, query: str, window: store.Window) -> tuple[list[dict], str]:
    """Matching airports, best first, and how they matched ("iata" or "name or city").

    An exact IATA code wins outright. Otherwise airports with passenger service whose name or
    city contains the query, ranked by passengers in the window (so "New York" puts JFK ahead
    of "New York Stewart International")."""
    if len(query) == 3 and query.isalpha():
        rows = store.find_airports(conn, "iata", query, window)
        if rows:
            return rows[:1], "iata"
    found: dict[str, dict] = {}
    for column in ("name", "municipality"):
        for r in store.find_airports(conn, column, query, window):
            if r["passengers"] > 0:
                found.setdefault(r["iata"], r)
    rows = sorted(found.values(), key=lambda r: (-r["passengers"], r["iata"]))
    return rows, "name or city"


def resolve_airport(conn: sqlite3.Connection, cfg: ScoringConfig, text: str) -> dict:
    """Name, city or code -> IATA. Order: configured metro alias (ambiguous, with a default),
    exact IATA, then name or city ranked by passengers. Several matches -> `ambiguous` with the
    best match and the alternatives (the agent asks which one); none -> an explicit no-match
    envelope with suggestions."""
    query = (text or "").strip()
    if not query:
        return _no_match(conn, query, None)
    window = _window(conn)

    alias = cfg.metro_aliases.get(query.upper())
    if alias:
        candidates = []
        for code in alias["candidates"]:
            rows = store.find_airports(conn, "iata", code, window)
            candidates.append(_brief(rows[0]) if rows else {"iata": code})
        others = [c["iata"] for c in candidates if c["iata"] != alias["default"]]
        best = next(c for c in candidates if c["iata"] == alias["default"])
        data = {"query": query, "ambiguous": True, "default": alias["default"], "best": best,
                "candidates": candidates, "alternatives": [c for c in candidates if c is not best]}
        assumptions = [f"'{query}' is ambiguous; default definition {query} = {alias['default']} (configured)"]
        facts = [fact("Default definition", f"{query} = {alias['default']}"),
                 fact("Other candidates", ", ".join(others))]
        headline = f"'{query}' covers several airports; default {query} = {alias['default']}, others: {', '.join(others)}."
        return _envelope(conn, window, data, assumptions, "medium", facts, headline=headline)

    rows, how = _matches(conn, query, window)
    if not rows:
        return _no_match(conn, query, window)
    if len(rows) == 1:
        return _envelope(conn, window, {"query": query, "ambiguous": False, **_brief(rows[0])},
                         [] if how == "iata" else [f"matched on airport {how}"], "high",
                         [_airport_fact(rows[0])], headline=_resolved(query, rows[0]))
    shown = [_brief(r) for r in rows[:MAX_LISTED]]
    best, alternatives = shown[0], shown[1:]
    data = {"query": query, "ambiguous": True, "default": None, "best": best, "candidates": shown,
            "alternatives": alternatives, "matches_total": len(rows)}
    alt = ", ".join(a["iata"] for a in alternatives)
    facts = [fact("Best match (most passengers)", f"{best['iata']} ({best['name']})"),
             fact("Alternatives", alt + (f" and {len(rows) - len(shown)} more" if len(rows) > len(shown) else ""))]
    return _envelope(conn, window, data,
                     [f"{len(rows)} airports match on {how}; ranked by passengers; ask the user which one"],
                     "low", facts,
                     headline=f"'{query}' matches {len(rows)} airports: best match {best['iata']} "
                              f"({best['name']}), alternatives {alt}.")


def _resolve_code(conn: sqlite3.Connection, cfg: ScoringConfig, text: str) -> tuple[str | None, dict | None]:
    """IATA code for `text`, or an error envelope (ambiguous or unknown) to return as is."""
    result = resolve_airport(conn, cfg, text)
    data = result["data"]
    if data is None:
        return None, result
    if data["ambiguous"]:
        return None, _error(conn, f"'{text}' is ambiguous; ask the user or state the definition used",
                            [c["iata"] for c in data["candidates"]], headline=result["headline"],
                            default=data["default"], best=data["best"]["iata"],
                            alternatives=[a["iata"] for a in data["alternatives"]])
    return data["iata"], None


# --- airport_metrics ---------------------------------------------------------------------

def _shares(counts: dict) -> dict[int, float | None]:
    total = counts["departures"]
    return {t: (_pct(n / total) if total else None) for t, n in counts["long_haul"].items()}


def airport_metrics(conn: sqlite3.Connection, cfg: ScoringConfig, airport: str,
                    threshold_mi: int | None = None, topic: str = "all") -> dict:
    """One airport's metrics for a `topic` (TOPIC_VALUES): long_haul (share of departures for all
    flights and for passenger service, with sensitivity), demand (passengers, seats, load factor),
    growth (passengers vs the previous 12 months), or all of them.

    The topic decides which facts, data and assumptions are returned (so a passenger question
    shows no long-haul rows) and which facts are required; the data period is always included."""
    topic = (topic or "all").strip().lower()
    if topic not in TOPIC_VALUES:
        return _error(conn, f"topic must be one of: {', '.join(TOPIC_VALUES)}")
    code, err = _resolve_code(conn, cfg, airport)
    if err:
        return err
    window = _window(conn)
    threshold = int(threshold_mi) if threshold_mi else cfg.long_haul_miles
    if threshold <= 0:
        return _error(conn, "threshold_mi must be a positive number of miles")

    result = scoring.score(_scoring_metrics(conn, cfg, window), cfg, window)
    rows = result.table[result.table["iata"] == code]
    if rows.empty:
        return _error(conn, f"{code} has no passenger service in {window.label}")
    row = rows.iloc[0]
    shows = {t: topic in (t, "all") for t in TOPIC_VALUES if t != "all"}

    data = {"airport": code, "name": row["name"], "state": row["state"], "topic": topic,
            "passengers": int(row["passengers"]), "months_active": int(row["months_active"]),
            "in_peer_set": bool(row["in_peer_set"])}
    facts = [fact("Airport", f"{code} ({row['name']})")]
    assumptions = [PASSENGER_SERVICE]
    headlines = []

    if shows["demand"]:
        data.update(seats=int(row["seats"]), departures_passenger_service=int(row["departures"]),
                    load_factor_pct=_pct(row["load_factor"]))
        facts += [fact("Passengers (passenger service)", f"{data['passengers']:,}", True),
                  fact("Seats (passenger service)", f"{data['seats']:,}"),
                  fact("Departures (passenger service)", f"{data['departures_passenger_service']:,}"),
                  fact("Load factor (passenger service, a demand-pressure proxy)", _share(data["load_factor_pct"]),
                       True)]
        assumptions.append(LOAD_FACTOR_PROXY)
        headlines.append(f"{code} in {window.label}: {data['passengers']:,} passengers on {data['seats']:,} seats "
                         f"(passenger service), load factor {_share(data['load_factor_pct'])}.")
        data["demand_context"] = _metro_data(row, result)
        facts += _metro_facts(data["demand_context"], cfg)
        assumptions += _metro_assumptions(data["demand_context"], cfg)

    if shows["growth"]:
        data.update(growth_pct=_pct(row["growth"]), previous_passengers=int(row["prev_passengers"]))
        previous = f"{window.prev_start}..{window.prev_end}"
        if not shows["demand"]:
            facts.append(fact(f"Passengers {window.label} (passenger service)", f"{data['passengers']:,}"))
        facts += [fact(f"Passengers {previous}", f"{data['previous_passengers']:,}"),
                  fact(f"Passenger growth vs {previous}", _share(data["growth_pct"]), True)]
        assumptions.append(f"growth = passengers {window.label} vs {previous}")
        headlines.append(f"{code} passengers changed {_signed(data['growth_pct'])} in {window.label} "
                         f"({data['passengers']:,}) vs {previous} ({data['previous_passengers']:,}).")

    if shows["long_haul"]:
        thresholds = sorted({threshold, *cfg.long_haul_sensitivity_miles})
        every = store.long_haul_counts(conn, code, window, thresholds)
        pax = store.long_haul_counts(conn, code, window, thresholds, store.PASSENGER_CONFIGS)
        every_pct, pax_pct = _shares(every), _shares(pax)
        data["long_haul"] = {
            "threshold_mi": threshold,
            "all_flights": {"departures": every["departures"], "long_haul_departures": every["long_haul"][threshold],
                            "share_pct": every_pct[threshold]},
            "passenger_service": {"departures": pax["departures"], "long_haul_departures": pax["long_haul"][threshold],
                                  "share_pct": pax_pct[threshold]},
            "sensitivity": [{"threshold_mi": t, "all_flights_pct": every_pct[t], "passenger_service_pct": pax_pct[t]}
                            for t in thresholds],
        }
        facts += [
            fact("Long-haul threshold", _miles(threshold), True),
            fact(f"Long-haul share >= {_miles(threshold)}, all flights (including cargo)",
                 _share(every_pct[threshold]), True),
            fact(f"Long-haul share >= {_miles(threshold)}, passenger service", _share(pax_pct[threshold]), True),
            fact(f"Departures >= {_miles(threshold)}, all flights", _of(every, threshold)),
            fact(f"Departures >= {_miles(threshold)}, passenger service", _of(pax, threshold)),
            *[fact(f"Sensitivity: long-haul share >= {_miles(t)}",
                   f"all flights {_share(every_pct[t])}, passenger service {_share(pax_pct[t])}", True)
              for t in thresholds if t != threshold],
        ]
        assumptions += ["all flights = every aircraft config, including cargo and undocumented",
                        f"long haul = segment distance >= {threshold} mi, share of departures"]
        headlines.append(_long_haul_headline(code, data["long_haul"]))

    if shows["congestion"]:
        data["congestion"] = _congestion_data(row, result)
        facts += _congestion_facts(data["congestion"], cfg)
        assumptions += [CONGESTION_SIDES, CANCELLATIONS_NOT_SCORED, ONTIME_COVERAGE,
                        f"congestion needs >= {cfg.min_ontime_flights:,} On-Time departures in the window"]
        headlines.append(_congestion_headline(code, window, data["congestion"]))

    return _envelope(conn, window, data, assumptions, row["confidence"], facts, headline=" ".join(headlines))


def _census_label(cfg: ScoringConfig, vintage: int) -> str:
    """'ACS 5-year 2020-2024' for acs/acs5, else the dataset path and vintage."""
    if cfg.census_dataset == "acs/acs5":
        return f"ACS 5-year {vintage - 4}-{vintage}"
    return f"{cfg.census_dataset} {vintage}"


def _metro_data(row, result: scoring.ScoreResult) -> dict:
    """The airport's CBSA, metro population in both vintages, growth and its percentile. `status`:
    ok, no_cbsa (outside every metro and micro area), no_base (CBSA code not in the base vintage)
    or not_loaded (Census not loaded, or no airport geocoded)."""
    if "demand_context" in result.unavailable:
        return {"status": "not_loaded", "reason": result.unavailable["demand_context"]}
    cbsa = row.get("cbsa")
    if cbsa is None or cbsa != cbsa:
        return {"status": "no_cbsa"}
    growth_pct = _pct(row["metro_growth"])
    return {"status": "ok" if growth_pct is not None else "no_base", "cbsa": cbsa, "cbsa_name": row.get("cbsa_name"),
            "population": _count(row["metro_population"]), "population_base": _count(row["metro_population_base"]),
            "growth_pct": growth_pct,
            "percentile": _pct(row["pct_demand_context"], 0) if row["in_peer_set"] else None}


def _metro_facts(m: dict, cfg: ScoringConfig) -> list[dict]:
    """Card rows for the demand context (not required: context, not the answer)."""
    if m["status"] == "not_loaded":
        return [fact("Metro population (demand context)", f"not available: {m['reason']}")]
    if m["status"] == "no_cbsa":
        return [fact("Metro area (CBSA)", "none: the airport lies outside every metro and micro area "
                                          "(demand context not scored for it)")]
    name = m["cbsa_name"] or "name not in this vintage"
    facts = [fact("Metro area (CBSA)", f"{name} ({m['cbsa']})"),
             fact(f"Metro population ({_census_label(cfg, cfg.census_vintage)})",
                  "n/a" if m["population"] is None else f"{m['population']:,}")]
    base = _census_label(cfg, cfg.census_base_vintage)
    if m["status"] == "no_base":
        return facts + [fact("Metro population growth (demand context)",
                             f"not available: CBSA {m['cbsa']} has no figure in {base} (delineation changed)")]
    facts.append(fact(f"Metro population growth vs {base} (demand context)", _signed(m["growth_pct"])))
    if m["percentile"] is not None:
        facts.append(fact("Demand context percentile in the peer set", f"{m['percentile']:.0f}"))
    return facts


def _metro_assumptions(m: dict, cfg: ScoringConfig) -> list[str]:
    if m["status"] == "not_loaded":
        return []
    return [f"demand context = population growth of the metro or micro area (CBSA) the airport lies in, "
            f"{_census_label(cfg, cfg.census_vintage)} vs {_census_label(cfg, cfg.census_base_vintage)} "
            "(non-overlapping periods); CBSA from the airport's coordinates (Census Geocoder); context for "
            "demand, not a measure of air travel demand",
            census.ATTRIBUTION]


def _num(value, digits: int = 1) -> float | None:
    """Rounded number; None for missing."""
    return None if value is None or value != value else round(float(value), digits)


def _count(value) -> int | None:
    return None if value is None or value != value else int(value)


def _congestion_data(row, result: scoring.ScoreResult) -> dict:
    """Raw congestion values, counts and percentiles for one scored row. `status`: ok, too_few_flights
    (below min_ontime_flights), no_data (no On-Time flights) or not_loaded (no On-Time at all)."""
    if "congestion" in result.unavailable:
        return {"status": "not_loaded", "reason": result.unavailable["congestion"]}
    data = {"status": row["ontime_status"],
            "departures_reported": _count(row["dep_flights"]), "arrivals_reported": _count(row["arr_flights"]),
            "dep_delay_rate_pct": _pct(row["dep_delay_rate"]), "taxi_out_min": _num(row["taxi_out_min"]),
            "nas_min_per_arrival": _num(row["nas_min_per_arrival"], 2),
            "cancellation_rate_pct": _pct(row["cancellation_rate"]), "cancelled": _count(row["dep_cancelled"])}
    if row["ontime_status"] == "ok" and row["in_peer_set"]:
        data["percentiles"] = {k: _pct(row[f"pct_{k}"], 0) for k in [*scoring.CONGESTION_COLUMN, "congestion"]}
    return data


def _congestion_facts(c: dict, cfg: ScoringConfig) -> list[dict]:
    """Required congestion facts (cancellations included, marked as not scored), or one required
    fact saying why there are none."""
    if c["status"] != "ok":
        why = {"not_loaded": "On-Time data not loaded",
               "no_data": "no flights of the On-Time reporting carriers",
               "too_few_flights": f"{c.get('departures_reported') or 0:,} reported departures, "
                                  f"below the {cfg.min_ontime_flights:,} minimum"}[c["status"]]
        return [fact("Congestion", f"not available: {why}", True)]
    facts = [
        fact("Departures delayed >= 15 min", _share(c["dep_delay_rate_pct"]), True),
        fact("Mean taxi-out (departures)", f"{c['taxi_out_min']} min", True),
        fact("NAS-delay minutes per arriving flight", f"{c['nas_min_per_arrival']}", True),
        fact("Cancelled departures (shown, not scored)",
             f"{_share(c['cancellation_rate_pct'])} ({c['cancelled']:,} of {c['departures_reported']:,})", True),
        fact("On-Time flights reported (large carriers only)",
             f"{c['departures_reported']:,} departures, {c['arrivals_reported']:,} arrivals"),
    ]
    if "percentiles" in c:
        p = c["percentiles"]
        facts.append(fact("Congestion percentile in the peer set",
                          f"{p['congestion']:.0f} (delay rate {p['dep_delay_rate']:.0f}, taxi-out {p['taxi_out']:.0f}, "
                          f"NAS delay {p['nas_delay_per_arrival']:.0f})"))
    return facts


def _congestion_headline(code: str, window: store.Window, c: dict) -> str:
    if c["status"] != "ok":
        return f"{code}: no congestion figures for {window.label} ({c['status'].replace('_', ' ')})."
    return (f"{code} in {window.label}: {_share(c['dep_delay_rate_pct'])} of departures delayed >= 15 min, "
            f"mean taxi-out {c['taxi_out_min']} min, {c['nas_min_per_arrival']} NAS-delay minutes per arriving "
            f"flight; {_share(c['cancellation_rate_pct'])} of departures cancelled (not scored).")


def _of(counts: dict, threshold: int) -> str:
    return f"{counts['long_haul'][threshold]:,} of {counts['departures']:,}"


def _signed(pct: float | None) -> str:
    return "n/a" if pct is None else f"{pct:+.1f}%"


def _long_haul_headline(code: str, lh: dict) -> str:
    """States the definition (threshold, share of departures) and both bases with counts."""
    every, pax = lh["all_flights"], lh["passenger_service"]
    return (f"{_share(every['share_pct'])} of all departures from {code} (incl. cargo) are "
            f">= {_miles(lh['threshold_mi'])} ({every['long_haul_departures']:,} of {every['departures']:,}); "
            f"passenger service: {_share(pax['share_pct'])} "
            f"({pax['long_haul_departures']:,} of {pax['departures']:,}).")


# --- rank_airports -----------------------------------------------------------------------

def _region_names(cfg: ScoringConfig) -> list[str]:
    return [*sorted(cfg.regions), scoring.NATIONAL]


def _unknown_region(conn: sqlite3.Connection, cfg: ScoringConfig, region: str) -> dict:
    """Deterministic answer for a region that is not configured (e.g. California): the configured
    regions with their states, and the two ways forward."""
    configured = "; ".join(f"{name} ({', '.join(states)})" for name, states in sorted(cfg.regions.items()))
    message = (f"Region '{region}' is not configured. Configured regions: {configured}. "
               f"Use region '{scoring.NATIONAL}' for a ranking of all US airports, "
               "or ask about a specific airport (airport_metrics).")
    return _error(conn, message, _region_names(cfg),
                  facts=[fact("Region", f"'{region}' is not configured", True),
                         fact("Configured regions", configured, True),
                         fact("Suggestion", f"national ranking (region '{scoring.NATIONAL}') or a specific airport")])


def _region_key(region: str) -> str:
    return (region or "").strip().lower().replace("-", "_").replace(" ", "_")


SortBy = Literal["score", "growth", "load_factor", "passengers"]
SORT_VALUES = get_args(SortBy)
SORT_LABELS = {"score": "expansion score", "growth": "passenger growth", "load_factor": "load factor",
               "passengers": "annual passengers"}


def scenario_weights(cfg: ScoringConfig, weights: dict | None) -> tuple[dict[str, float] | None, str | None]:
    """Validate a scenario override: (merged weights, None) or (None, error message).

    Keys must be weight names from config/weights.yaml, values non-negative numbers. Keys left
    out keep their configured weight; scoring.score then renormalizes over the available
    components, so the values need not sum to 1 ("growth 0.30" means double the default 0.15).
    """
    if not weights:
        return None, None
    if not isinstance(weights, dict):
        return None, 'weights must be an object such as {"growth": 0.3}'
    unknown = sorted(map(str, set(weights) - set(cfg.weights)))
    if unknown:
        return None, f"Unknown weight(s) {', '.join(unknown)}; valid: {', '.join(cfg.weights)}"
    merged = dict(cfg.weights)
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
            return None, f"Weight {key} must be a non-negative number (got {value!r})"
        merged[key] = float(value)
    return merged, None


def _rank_table(conn: sqlite3.Connection, cfg: ScoringConfig, key: str, window: store.Window):
    result = scoring.score(_scoring_metrics(conn, cfg, window), cfg, window)
    return result, scoring.rank(result, key, cfg)


def _weights_text(weights: dict[str, float]) -> str:
    return ", ".join(f"{c} {w}" for c, w in weights.items())


def rank_airports(conn: sqlite3.Connection, cfg: ScoringConfig, region: str, top_n: int = 10,
                  sort_by: str = "score", weights: dict | None = None) -> dict:
    """Rank a configured region's peer-set airports by the deterministic score.

    `sort_by` reorders the ranked airports by one raw metric (done here, never by the model);
    `weights` is a scenario override (see `scenario_weights`): the result is labeled "scenario"
    and every airport carries its base rank (configured weights) next to its scenario rank.
    The list is capped at `cfg.max_ranked` airports.
    """
    key = _region_key(region)
    if key not in cfg.regions and key != scoring.NATIONAL:
        return _unknown_region(conn, cfg, region)
    if int(top_n) < 1:
        return _error(conn, "top_n must be at least 1")
    sort_by = (sort_by or "score").strip().lower()
    if sort_by not in SORT_VALUES:
        return _error(conn, f"sort_by must be one of: {', '.join(SORT_VALUES)}")
    merged, problem = scenario_weights(cfg, weights)
    if problem:
        return _error(conn, problem, list(cfg.weights))
    window = _window(conn)
    base_result, base_ranking = _rank_table(conn, cfg, key, window)
    result, ranking = base_result, base_ranking
    if merged is not None:
        try:
            result, ranking = _rank_table(conn, replace(cfg, weights=merged), key, window)
        except ValueError:  # every available component weighted 0
            return _error(conn, "The scenario weights give no weight to any available component "
                                f"(not built: {', '.join(sorted(base_result.unavailable))})", list(cfg.weights))
    scenario = merged is not None
    components = list(result.weights_used)
    base_rank = dict(zip(base_ranking.ranked["iata"], base_ranking.ranked["rank"]))
    stability = scoring.rank_stability(ranking, result.weights_used, cfg)

    order = ranking.ranked
    if sort_by != "score":  # highest first, missing last, ties by score then IATA
        order = order.sort_values([sort_by, "score", "iata"], ascending=[False, False, True],
                                  na_position="last", kind="stable")
    shown = min(int(top_n), cfg.max_ranked)
    ranked = []
    for position, (_, r) in enumerate(order.head(shown).iterrows(), start=1):
        raw = {"utilization": {"load_factor_pct": _pct(r["load_factor"])},
               "scale": {"passengers": int(r["passengers"])},
               "growth": {"growth_pct": _pct(r["growth"])},
               "long_haul": {"share_pct": _pct(r["long_haul_share"])},
               "demand_context": {"metro_growth_pct": _pct(r.get("metro_growth"))}}
        if "congestion" in components:
            raw["congestion"] = {"dep_delay_rate_pct": _pct(r["dep_delay_rate"]),
                                 "taxi_out_min": _num(r["taxi_out_min"]),
                                 "nas_min_per_arrival": _num(r["nas_min_per_arrival"], 2)}
        item = {"rank": position, "iata": r["iata"], "name": r["name"], "score": round(float(r["score"]), 3),
                "components": {c: {**raw[c], "percentile": _pct(r[f"pct_{c}"], 0)} for c in components},
                "confidence": r["confidence"]}
        if sort_by != "score":
            item["score_rank"] = int(r["rank"])
            item["sort_value"] = {"growth": _pct(r["growth"]), "load_factor": _pct(r["load_factor"]),
                                  "passengers": int(r["passengers"])}[sort_by]
        if scenario:
            item["base_rank"] = int(base_rank[r["iata"]])
        if found := scoring.drivers(r, components):
            item["drivers"] = {k: None if cp is None else {"component": cp[0], "percentile": _pct(cp[1], 0)}
                               for k, cp in found.items()}
        s = stability[r["iata"]]
        item["stability"] = {"top_n": s["top_n"], "in_top_n_pct": _pct(s["in_top_n"] / s["runs"], 0),
                             "first": s["first"], "runs": s["runs"], "min_rank": s["min_rank"],
                             "max_rank": s["max_rank"]}
        if r["missing"]:
            item["missing"] = r["missing"]
            item["weights_used"] = {c: round(w, 3) for c, w in r["weights_used"].items()}
        ranked.append(item)

    scores = [x["score"] for x in ranked]
    ties = sorted({s for s in scores if scores.count(s) > 1}) if sort_by == "score" else []
    data = {
        "region": key, "states": list(ranking.states) or "all US states and territories",
        "peer_set_size": result.peer_set_size,
        "mode": "scenario" if scenario else "base", "sort_by": sort_by,
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
        "drivers_basis": "national percentile in the peer set (higher = stronger)",
        "stability": {"runs": cfg.sensitivity_runs, "weight_jitter": cfg.sensitivity_jitter,
                      "seed": cfg.sensitivity_seed, "top_n": cfg.sensitivity_top_n},
    }
    if scenario:
        data["weights_base"] = {c: round(w, 3) for c, w in base_result.weights_used.items()}
        data["weights_requested"] = {k: float(v) for k, v in weights.items()}
    assumptions = [
        "score = weighted mean of national percentiles in the peer set (0-1); a capacity-pressure proxy, "
        "not a measured terminal need (no gate or terminal-area data)",
        f"peer set = US airports with >= {cfg.min_passengers_per_year:,} passenger-service passengers in the window",
        PASSENGER_SERVICE, LOAD_FACTOR_PROXY,
        f"long haul >= {cfg.long_haul_miles} mi; shares under {_pct(cfg.long_haul_min_share)}% count as 0 in the score",
        *result.notes,
    ]
    if "congestion" in result.weights_used:
        assumptions += [f"congestion = {_weights_text(cfg.congestion_weights)} (percentiles in the peer set); "
                        + CANCELLATIONS_NOT_SCORED, CONGESTION_SIDES,
                        f"{ONTIME_COVERAGE} (minimum {cfg.min_ontime_flights:,} departures)"]
    if scenario:
        ignored = sorted(k for k, v in weights.items() if k in result.unavailable and v > 0)
        assumptions.insert(0, "scenario: weights changed by the user, not the configured methodology; "
                              "weights not given keep their configured value, then all are renormalized"
                              + (f"; ignored (component not built): {', '.join(ignored)}" if ignored else ""))
    if sort_by != "score":
        assumptions.insert(0, f"sorted by {SORT_LABELS[sort_by]} (highest first), not by the expansion score")
    assumptions.append("drivers = each airport's highest and lowest component percentile; tied values share one "
                       "percentile, so airports at the lowest value of a component shared by several airports (no "
                       "material long-haul service) get a middling percentile; such a component is never reported "
                       "as an airport's strongest")
    if int(top_n) > shown and len(ranking.ranked) > shown:
        assumptions.append(f"list capped at {cfg.max_ranked} airports")
    levels = [x["confidence"] for x in ranked] or ["low"]
    confidence = max(levels, key=CONFIDENCE_ORDER.index)
    renorm = f" (renormalized without {', '.join(sorted(result.unavailable))})" if result.unavailable else ""
    facts = []
    if scenario:
        facts += [fact("Ranking", "scenario (user weights, not the configured methodology)", True),
                  fact("Scenario weights used" + renorm, _weights_text(data["weights_used"]), True),
                  fact("Base weights (config)" + renorm, _weights_text(data["weights_base"]))]
    else:
        facts.append(fact("Weights used" + renorm, _weights_text(data["weights_used"]), True))
    if sort_by != "score":
        facts.append(fact("Sorted by", f"{SORT_LABELS[sort_by]}, highest first (not the score)", True))
    facts += [fact(f"#{x['rank']} {x['iata']} ({x['name']})", _ranked_summary(x)) for x in ranked]
    # Drivers block: why each airport ranks where it does, without model commentary (rendered as a table).
    facts += [fact(f"Drivers: {x['iata']}", _drivers_text(x["drivers"])) | {"ranked": x["iata"]}
              for x in ranked if x.get("drivers")]
    headline = _ranking_headline(data, key, sort_by, scenario)
    if ranked:
        facts.append(fact("Rank stability (method)", _stability_method(cfg)))
        assumptions.append(f"rank stability: {_stability_method(cfg)}; it tests the weights, not the data")
        top = next((x for x in ranked if x.get("score_rank", x["rank"]) == 1), ranked[0])  # the score's #1
        s = top["stability"]
        facts.append(fact("Stability of the top airport", f"{top['iata']} first in {s['first']} of {s['runs']} runs"))
        if sort_by == "score" and s["first"] < cfg.sensitivity_stable_share * s["runs"]:
            headline += (f" Not stable: {top['iata']} is first in only {s['first']} of {s['runs']} runs "
                         "with perturbed weights.")
    return _envelope(conn, window, data, assumptions, confidence, facts, headline=headline)


def _stability_method(cfg: ScoringConfig) -> str:
    return (f"{cfg.sensitivity_runs} runs, each weight changed at random by up to ±{cfg.sensitivity_jitter:.0%} "
            f"and renormalized (seed {cfg.sensitivity_seed})")


def _ranking_headline(data: dict, key: str, sort_by: str, scenario: bool) -> str:
    ranked = data["ranked"][:HEADLINE_TOP]
    if scenario:
        top = ", ".join(f"{x['rank']} {x['iata']} (base {x['base_rank']})" for x in ranked)
    else:
        top = ", ".join(f"{x['rank']} {x['iata']}" for x in ranked)
    what = "terminal-expansion ranking by score" if sort_by == "score" else f"airports by {SORT_LABELS[sort_by]}"
    prefix = "Scenario weights: " if scenario else ""
    return (f"{prefix}{key} {what}: {top or 'no airport in the peer set'} "
            f"({data['ranked_total']} ranked, {data['excluded_total']} excluded).")


COMPONENT_LABELS = {"utilization": "utilization (load factor)", "scale": "scale (annual passengers)",
                    "growth": "growth (passengers vs previous 12 months)", "long_haul": "long-haul share",
                    "congestion": "congestion", "demand_context": "demand context (metro population growth)"}


def _driver(d: dict | None) -> str:
    if d is None:
        return "none (every component at the lowest value shared by several airports)"
    return f"{COMPONENT_LABELS.get(d['component'], d['component'])}, percentile {d['percentile']:.0f}"


def _drivers_text(drivers: dict) -> str:
    return f"strongest: {_driver(drivers['strongest'])}; weakest: {_driver(drivers['weakest'])}"


def _ranked_summary(item: dict) -> str:
    """One line per ranked airport: score, then each component's raw value and percentile."""
    parts = [f"score {item['score']}"]
    for c, v in item["components"].items():
        pct = "n/a" if v["percentile"] is None else f"{v['percentile']:.0f}"
        if c == "congestion":
            taxi = "n/a" if v["taxi_out_min"] is None else f"{v['taxi_out_min']} min"
            nas = "n/a" if v["nas_min_per_arrival"] is None else f"{v['nas_min_per_arrival']} min"
            parts.append(f"congestion (delayed departures {_share(v['dep_delay_rate_pct'])}, taxi-out {taxi}, "
                         f"NAS delay {nas} per arrival), percentile {pct}")
            continue
        raw = next(val for key, val in v.items() if key != "percentile")
        shown = f"{raw:,}" if isinstance(raw, int) else _share(raw)
        parts.append(f"{COMPONENT_LABELS.get(c, c)} {shown}, percentile {pct}")
    if s := item.get("stability"):
        parts.append(f"top {s['top_n']} in {s['in_top_n_pct']:.0f}% of stability runs, ranks {s['min_rank']}-{s['max_rank']}")
    return "; ".join(parts)


# --- compare_airports --------------------------------------------------------------------

T100_FULL_COUNT = ("T-100 figures (load factor, passengers, growth, long-haul share) count every flight, "
                   "so they are compared as is, without a margin of error")
# On-Time rows of the comparison: (key in the score table, label, short name for the headline, kind).
# kind "share": a rate of flights, margin from the flight count; "minutes": a mean, margin from the
# spread of the monthly means (only monthly sums are stored).
ONTIME_COMPARED = (
    ("dep_delay_rate", "Departures delayed >= 15 min", "delayed departures", "share"),
    ("taxi_out_min", "Mean taxi-out (departures)", "taxi-out", "minutes"),
    ("nas_min_per_arrival", "NAS-delay minutes per arriving flight", "NAS delay per arrival", "minutes"),
    ("cancellation_rate", "Cancelled departures (shown, not scored)", "cancellations (not scored)", "share"),
)
# Denominator of each share (column of the score table) and the monthly sums behind each mean.
SHARE_DENOMINATOR = {"dep_delay_rate": "dep_del15_n", "cancellation_rate": "dep_flights"}
MONTHLY_MEAN = {"taxi_out_min": ("taxi_out_sum", "taxi_out_n"), "nas_min_per_arrival": ("nas_min", "arr_flights")}
MINUTE_DIGITS = {"taxi_out_min": 1, "nas_min_per_arrival": 2}
# T-100 rows of the comparison: (key in the score table, label, display kind).
T100_COMPARED = (
    ("load_factor", "Load factor (passenger service)", "share"),
    ("passengers", "Passengers (passenger service)", "count"),
    ("growth", "Passenger growth vs previous 12 months", "signed"),
    ("long_haul_share", "Long-haul share (passenger service)", "share"),
)


def _resolve_for_compare(conn: sqlite3.Connection, cfg: ScoringConfig,
                         text: str) -> tuple[str | None, str | None, dict | None]:
    """(IATA, assumption, error). A configured metro alias resolves to its default and the
    definition is stated (LA = LAX); any other ambiguity is returned as an error, never guessed."""
    data = resolve_airport(conn, cfg, text)["data"]
    if data and data["ambiguous"] and data["default"]:
        others = ", ".join(a["iata"] for a in data["alternatives"])
        return data["default"], f"'{text}' = {data['default']} (configured default; other candidates: {others})", None
    code, err = _resolve_code(conn, cfg, text)
    return code, None, err


def _ontime_cell(key: str, kind: str, row, monthly, z: float) -> dict:
    """Value, margin and display text of one On-Time metric for one airport. The raw (unrounded)
    value and margin decide the verdict; `short` is the value as the headline writes it."""
    value = None if row is None else row[key]
    if value is None or value != value:
        return {"value": None, "margin": None, "display": "n/a", "short": "n/a"}
    value = float(value)
    if kind == "share":
        n = row[SHARE_DENOMINATOR[key]]
        margin = scoring.proportion_margin(value, n, z)
        return {"value": value, "margin": margin, "short": _share(_pct(value)),
                "display": f"{_share(_pct(value))} ± {_num(margin * 100, 2)} pts ({int(n):,} departures)"}
    total, count = MONTHLY_MEAN[key]
    margin = scoring.mean_margin((monthly[total] / monthly[count].where(monthly[count] > 0)).tolist(), z)
    digits = MINUTE_DIGITS[key]
    spread = "n/a" if margin is None else _num(margin, digits)
    return {"value": value, "margin": margin, "short": f"{_num(value, digits)} min",
            "display": f"{_num(value, digits)} min ± {spread} ({len(monthly)} monthly means)"}


def _t100_display(kind: str, value) -> str:
    if value is None or value != value:
        return "n/a"
    if kind == "count":
        return f"{int(value):,}"
    return _signed(_pct(value)) if kind == "signed" else _share(_pct(value))


def compare_airports(conn: sqlite3.Connection, cfg: ScoringConfig, a: str, b: str) -> dict:
    """Two airports side by side: congestion (On-Time, a margin of error per figure and a
    deterministic verdict), load factor, passengers, growth and long-haul share (T-100, full counts).

    Verdict rule: a difference inside the combined margin is "no meaningful difference"; only a
    difference beyond it names the higher airport. Facts carry an `airport` key (one value per
    airport) or a `metric` key (the verdict), used by the table renderer and the figure check."""
    codes, notes = [], []
    for text in (a, b):
        code, note, err = _resolve_for_compare(conn, cfg, text)
        if err:
            return err
        codes.append(code)
        notes += [note] if note else []
    if codes[0] == codes[1]:
        return _error(conn, f"Both names resolve to {codes[0]}; name two different airports")
    window = _window(conn)
    result = scoring.score(_scoring_metrics(conn, cfg, window), cfg, window)
    table = result.table.set_index("iata")
    missing = [c for c in codes if c not in table.index]
    if missing:
        return _error(conn, f"{missing[0]} has no passenger service in {window.label}")
    rows = {c: table.loc[c] for c in codes}
    z = scoring.z_value(cfg.comparison_confidence_level)
    level = f"{cfg.comparison_confidence_level:.0%}"
    status = {c: (rows[c]["ontime_status"] if "congestion" not in result.unavailable else "not_loaded")
              for c in codes}

    metrics, facts, parts, scored = [], [], [], []
    for key, label, short, kind in ONTIME_COMPARED:
        cells = {c: _ontime_cell(key, kind, rows[c] if status[c] == "ok" else None,
                                 store.ontime_monthly(conn, c, window) if status[c] == "ok" else None, z)
                 for c in codes}
        winner = scoring.compare_values(*(cells[c]["value"] for c in codes), *(cells[c]["margin"] for c in codes))
        verdict = {"a": f"{codes[0]} higher", "b": f"{codes[1]} higher", "same": "no meaningful difference",
                   None: "not comparable"}[winner]
        if key in scoring.CONGESTION_COLUMN.values():  # scored congestion metrics (not cancellations)
            scored.append(winner)
        values = ", ".join(f"{c} {cells[c]['short']}" for c in codes)
        parts.append(f"no meaningful difference in {short} ({values})" if winner == "same"
                     else f"{short}: {verdict} ({values})")
        metrics.append({"key": key, "label": label, "source": "On-Time", "verdict": verdict,
                        "cells": {c: {"display": cells[c]["display"], "value": _num(cells[c]["value"], 4),
                                      "margin": _num(cells[c]["margin"], 4)} for c in codes}})
        facts += [fact(f"{c}: {label}", cells[c]["display"], key == "dep_delay_rate") | {"airport": c}
                  for c in codes]
        facts.append(fact(f"Verdict: {label}", verdict) | {"metric": key})

    pct = {c: _pct(rows[c]["pct_congestion"], 0) if status[c] == "ok" and rows[c]["in_peer_set"] else None
           for c in codes}
    shown = {c: "n/a" if pct[c] is None else f"{pct[c]:.0f}" for c in codes}
    metrics.append({"key": "congestion_percentile", "label": "Congestion percentile in the peer set",
                    "source": "On-Time", "verdict": "not tested (a rank, no margin)",
                    "cells": {c: {"display": shown[c], "value": pct[c]} for c in codes}})
    facts += [fact(f"{c}: Congestion percentile in the peer set", shown[c]) | {"airport": c} for c in codes]

    for key, label, kind in T100_COMPARED:
        values = {c: rows[c][key] for c in codes}
        known = all(v is not None and v == v for v in values.values())
        verdict = ("not comparable" if not known else "equal" if values[codes[0]] == values[codes[1]]
                   else f"{max(codes, key=lambda c: values[c])} higher (full count)")
        metrics.append({"key": key, "label": label, "source": "T-100", "verdict": verdict,
                        "cells": {c: {"display": _t100_display(kind, values[c])} for c in codes}})
        facts += [fact(f"{c}: {label}", _t100_display(kind, values[c])) | {"airport": c} for c in codes]

    data = {"airports": codes, "names": {c: rows[c]["name"] for c in codes}, "confidence_level": level,
            "ontime_status": status, "metrics": metrics}
    assumptions = [*notes,
                   f"On-Time figures carry a {level} margin of error: rates from the flight counts, minute "
                   "averages from the spread of the 12 monthly means; a difference inside the combined margin "
                   "(square root of the sum of the squared margins) is reported as no meaningful difference",
                   ONTIME_COVERAGE, CONGESTION_SIDES, CANCELLATIONS_NOT_SCORED, T100_FULL_COUNT,
                   PASSENGER_SERVICE, LOAD_FACTOR_PROXY]
    levels = [rows[c]["confidence"] for c in codes] + (["medium"] if notes else [])
    confidence = max(levels, key=CONFIDENCE_ORDER.index)
    no_figures = [f"{c} has no congestion figures ({status[c].replace('_', ' ')})" for c in codes if status[c] != "ok"]
    assumptions += no_figures
    overall = scoring.overall_verdict(scored)
    overall_text = {"none": "Overall: not comparable, no scored congestion figure for both airports.",
                    "same": "Overall: no meaningful difference.",
                    "a": f"Overall: {codes[0]} is higher on every measure with a meaningful difference.",
                    "b": f"Overall: {codes[1]} is higher on every measure with a meaningful difference.",
                    "mixed": "Overall: mixed, no airport is higher on every measure."}[overall]
    data["overall"] = overall_text
    assumptions.append("overall verdict over the scored congestion measures (delayed departures, taxi-out, NAS "
                       "delay per arrival; cancellations are not scored); higher = more congested")
    deps = {c: int(rows[c]["departures"]) for c in codes}
    big, small = sorted(codes, key=lambda c: -deps[c])
    size_note = ""
    if deps[small] and deps[big] / deps[small] > cfg.comparison_size_ratio_note:
        size_note = (f" Note: {big} has {deps[big] / deps[small]:.1f} times {small}'s departures (passenger "
                     "service): rates compare per flight, but the airports differ in scale.")
        assumptions.append(f"departures differ by more than {cfg.comparison_size_ratio_note:g} times "
                           "(comparison.size_ratio_note): per-flight rates compare, totals do not")
    headline = (f"{codes[0]} vs {codes[1]}, {window.label} ({level} margins on On-Time figures): "
                + "; ".join(parts) + "." + "".join(f" {n}." for n in no_figures) + f" {overall_text}" + size_note)
    return _envelope(conn, window, data, assumptions, confidence, facts, headline=headline)


# --- explain_demand_gap ------------------------------------------------------------------

UNMET_DEMAND_PROXY = ("unmet demand is not observable in this data: T-100 counts passengers carried, never "
                      "passengers turned away; every figure here is a PROXY, not a measurement")
NEEDED_TO_MEASURE = ("measuring unmet demand would need booking and denied-booking data, fares by date and "
                     "airline spill models")
# BTS delay-cause categories (fixed definitions, not an interpretation of this airport).
CAUSE_LABELS = {"nas": "NAS", "carrier": "carrier", "late_aircraft": "late aircraft", "weather": "weather",
                "security": "security"}
CAUSE_DEFINITIONS = ("delay causes as BTS defines them: NAS = national aviation system (traffic volume, airport "
                     "operations, non-extreme weather); carrier = within the airline's control; late aircraft = "
                     "a previous flight with the same aircraft arrived late; weather = extreme weather; shares of "
                     "delay minutes on arrivals at the airport")
# Proxy results are capped at this confidence: a proxy is never "high" (docs/plan.md, Scoring).
PROXY_CONFIDENCE_CAP = "medium"


def _month_label(year, month) -> str:
    return f"{int(year)}-{int(month):02d}"


def _lf(value) -> str:
    return _share(_pct(value))


def _target(value: float) -> str:
    """A target load factor from config as a whole percentage (0.8 -> 80%)."""
    return f"{value:.0%}"


def explain_demand_gap(conn: sqlite3.Connection, cfg: ScoringConfig, airport: str) -> dict:
    """A labeled PROXY for unmet demand at one airport, with a "why" built only from its facts.

    Unmet demand cannot be observed here (no booking or spill data). The proxy, all in code: months
    with load factor above the configured target and the peak months; the routes with the most
    passengers and with the most passengers above the target; passengers above the target load
    level per route and month (spill proxy: the excess over target x seats), also at each sensitivity target; the delay-cause mix
    (arrivals, On-Time); growth. Confidence is capped at medium."""
    code, err = _resolve_code(conn, cfg, airport)
    if err:
        return err
    window = _window(conn)
    result = scoring.score(_scoring_metrics(conn, cfg, window), cfg, window)
    rows = result.table[result.table["iata"] == code]
    routes = store.route_months(conn, code, window)
    if rows.empty or routes.empty:
        return _error(conn, f"{code} has no passenger service in {window.label}")
    row = rows.iloc[0]
    target, targets = cfg.target_load_factor, sorted({cfg.target_load_factor, *cfg.target_sensitivity})
    t_text = _target(target)

    months = routes.groupby(["year", "month"], as_index=False)[["passengers", "seats"]].sum()
    months["load_factor"] = scoring.load_factor(months["passengers"], months["seats"])
    months["label"] = [_month_label(y, m) for y, m in zip(months["year"], months["month"])]
    above = months[months["load_factor"] > target]
    peaks = months.sort_values(["load_factor", "label"], ascending=[False, True], kind="stable").head(cfg.peak_months)
    peak_text = ", ".join(f"{p['label']} {_lf(p['load_factor'])}" for _, p in peaks.iterrows())

    spill = {t: int(round(scoring.above_target(routes["passengers"], routes["seats"], t).sum())) for t in targets}
    routes["spill"] = scoring.above_target(routes["passengers"], routes["seats"], target)
    by_route = routes.groupby("dest").agg(passengers=("passengers", "sum"), seats=("seats", "sum"),
                                          spill=("spill", "sum"))
    by_route["load_factor"] = scoring.load_factor(by_route["passengers"], by_route["seats"])
    busiest = by_route.sort_values("passengers", ascending=False, kind="stable").head(cfg.top_routes)
    spilling = by_route[by_route["spill"] > 0].sort_values("spill", ascending=False, kind="stable").head(cfg.top_routes)
    total_pax = int(row["passengers"])
    spill_share = _pct(spill[target] / total_pax) if total_pax else None
    top_spill_share = _pct(spilling["spill"].sum() / spill[target]) if spill[target] else None

    causes = (scoring.delay_cause_mix({c: row[f"{c}_min"] for c in scoring.DELAY_CAUSES})
              if f"{scoring.DELAY_CAUSES[0]}_min" in row.index else None)
    growth_pct = _pct(row["growth"])
    previous = f"{window.prev_start}..{window.prev_end}"

    data = {
        "airport": code, "name": row["name"], "proxy": True, "target_load_factor": target,
        "load_factor_pct": _pct(row["load_factor"]), "passengers": total_pax, "growth_pct": growth_pct,
        "months": [{"month": m["label"], "load_factor_pct": _pct(m["load_factor"])} for _, m in months.iterrows()],
        "months_above_target": [m["label"] for _, m in above.iterrows()],
        "spill_proxy": {f"{t:.2f}": spill[t] for t in targets},
        "routes_by_passengers": [{"dest": d, "passengers": int(r["passengers"]), "load_factor_pct": _pct(r["load_factor"])}
                                 for d, r in busiest.iterrows()],
        "routes_by_spill": [{"dest": d, "spill_proxy": int(round(r["spill"])), "load_factor_pct": _pct(r["load_factor"])}
                            for d, r in spilling.iterrows()],
        "delay_cause_mix_pct": {c: _pct(s) for c, s in causes.items()} if causes else None,
    }
    range_text = (f"{spill[targets[0]]:,} at {_target(targets[0])} to {spill[targets[-1]]:,} at {_target(targets[-1])}"
                  if len(targets) > 1 else f"{spill[target]:,}")
    facts = [
        fact("Unmet demand", "not observable in this data; the figures below are a proxy", True),
        fact(f"Months with load factor above {t_text}", f"{len(above)} of {len(months)}", True),
        fact("Peak months (load factor)", peak_text),
        fact(f"Spill proxy: passengers above the {t_text} load level (excess over {t_text} of seats, per route "
             "and month)",
             f"{spill[target]:,} ({_share(spill_share)} of passengers)", True),
        fact("Spill proxy sensitivity (target load factor)", range_text, True),
        fact("Routes with the most passengers (load factor)",
             "; ".join(f"{code}-{d} {r['passengers']:,} ({_share(r['load_factor_pct'])})"
                       for d, r in zip(busiest.index, data["routes_by_passengers"]))),
        fact(f"Routes with the most passengers above {t_text} load",
             ("; ".join(f"{code}-{r['dest']} {r['spill_proxy']:,}" for r in data["routes_by_spill"])
              + f" ({_share(top_spill_share)} of the spill proxy)") if spilling.size else "none"),
        fact("Load factor (passenger service, a demand-pressure proxy)", _share(data["load_factor_pct"])),
        fact(f"Passenger growth vs {previous}", _signed(growth_pct), True),
        fact("Delay-cause mix (share of delay minutes, arrivals)",
             ", ".join(f"{CAUSE_LABELS[c]} {_share(p)}" for c, p in data["delay_cause_mix_pct"].items())
             if causes else "not available (no On-Time delay minutes)", True),
    ]

    why = [f"Airport-wide load factor was above the {t_text} target in {len(above)} of {len(months)} months, "
           f"highest in {peak_text}." if len(above) else
           f"Airport-wide load factor stayed at or below the {t_text} target in every month (highest {peak_text})."]
    if spilling.size:
        why.append(f"Passengers above the target load are concentrated on routes to {', '.join(spilling.index[:3])} "
                   f"({_share(top_spill_share)} of the spill proxy on "
                   + ("that route)." if len(spilling) == 1 else f"the top {len(spilling)} routes)."))
    if causes:
        top = next(iter(data["delay_cause_mix_pct"].items()))
        why.append(f"The largest delay cause on arrivals is {CAUSE_LABELS[top[0]]} ({_share(top[1])} of delay minutes).")
    why.append(f"Passengers changed {_signed(growth_pct)} vs {previous}.")
    data["why"] = why

    assumptions = [UNMET_DEMAND_PROXY, NEEDED_TO_MEASURE,
                   f"spill proxy = passengers in excess of {t_text} x seats, summed over the route-months whose load "
                   f"factor is above {t_text}: only the excess counts (a route-month at 85% adds 5% of its seats), "
                   "not every passenger on a full flight; sensitivity at "
                   f"{', '.join(_target(t) for t in targets)}",
                   PASSENGER_SERVICE, LOAD_FACTOR_PROXY, CAUSE_DEFINITIONS, ONTIME_COVERAGE,
                   f"confidence capped at {PROXY_CONFIDENCE_CAP}: a proxy, not a measurement"]
    confidence = max(row["confidence"], PROXY_CONFIDENCE_CAP, key=CONFIDENCE_ORDER.index)
    headline = (f"PROXY, not a measurement (unmet demand is not observable in this data): {code} load factor was "
                f"above the {t_text} target in {len(above)} of {len(months)} months in {window.label}"
                + (f" (peak {peaks.iloc[0]['label']} {_lf(peaks.iloc[0]['load_factor'])})" if len(above) else "")
                + f"; spill proxy {spill[target]:,} passengers above the {t_text} load level (excess over "
                  f"{t_text} of seats per route and month; range {range_text}).")
    return _envelope(conn, window, data, assumptions, confidence, facts, headline=headline)


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
                             topic: Topic = "all") -> dict:
        """Metrics for one airport (IATA code or name). Set topic to what the question asks about:
        long_haul (share of departures >= threshold, all flights and passenger service, with
        sensitivity), demand (passengers, seats, load factor), growth (passengers vs the previous
        12 months), congestion (departure delay rate, taxi-out, NAS delay per arrival, cancellations);
        all when it asks for several. threshold_mi only for long_haul (default from config)."""
        return run(airport_metrics, airport, threshold_mi, topic)

    @tool("rank_airports", description=(
        "Rank airports in a region as terminal-expansion candidates: scores, component percentiles "
        "with raw values, the weights used, excluded airports, confidence. "
        f"region must be one of: {', '.join(_region_names(cfg))} ({scoring.NATIONAL} = all US airports). "
        "sort_by: score (default), or growth, load_factor, passengers to list the airports by that metric. "
        "weights: only for what-if questions (e.g. 'what if growth weighs more'): a scenario override "
        f"such as {{\"growth\": 0.3}}; keys: {', '.join(cfg.weights)}. Omit it otherwise."))
    def rank_airports_tool(region: str, top_n: int = 10, sort_by: SortBy = "score",
                           weights: dict[str, float] | None = None) -> dict:
        return run(rank_airports, region, top_n, sort_by, weights)

    @tool("compare_airports")
    def compare_airports_tool(a: str, b: str) -> dict:
        """Compare two airports (IATA codes or names) side by side: congestion (delayed departures,
        taxi-out, NAS delay per arrival, cancellations) with a margin of error and a verdict per
        figure, plus load factor, passengers, growth and long-haul share. Use for "compare X and Y",
        "X vs Y", "is X more congested than Y"."""
        return run(compare_airports, a, b)

    @tool("explain_demand_gap")
    def explain_demand_gap_tool(airport: str) -> dict:
        """Unmet or excess demand at one airport (IATA code or name) and why: a labeled PROXY (unmet
        demand is not observable): months above the target load factor, peak months, densest routes,
        passengers above the target load level (spill proxy) with a sensitivity range, delay-cause
        mix and growth. Use for "unmet demand", "demand gap", "is X short of capacity and why"."""
        return run(explain_demand_gap, airport)

    return [data_status_tool, resolve_airport_tool, airport_metrics_tool, rank_airports_tool, compare_airports_tool,
            explain_demand_gap_tool]
