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
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Callable, Literal, get_args

from langchain_core.tools import BaseTool, tool

from src import scoring, store
from src.config import ScoringConfig

CONFIDENCE_ORDER = ("high", "medium", "low")
PASSENGER_SERVICE = "passenger service = aircraft config passenger or combi (BTS T-100), departures > 0"
# Display cap for candidate and excluded lists: keeps envelopes small for local models' short context.
MAX_LISTED = 5
# airport_metrics `topic`: which facts, data and assumptions are returned, and which facts are
# required (ensure_facts appends them if the answer omits them). demand = passengers, seats, load factor.
Topic = Literal["long_haul", "demand", "growth", "all"]
TOPIC_VALUES = get_args(Topic)
# resolve_airport suggestions: difflib ratio needed to suggest an airport for an unmatched name.
SUGGEST_MIN_SIMILARITY = 0.6
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

    result = scoring.score(store.airport_metrics(conn, window, cfg.long_haul_miles), cfg, window)
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

    return _envelope(conn, window, data, assumptions, row["confidence"], facts, headline=" ".join(headlines))


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
    result = scoring.score(store.airport_metrics(conn, window, cfg.long_haul_miles), cfg, window)
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
               "long_haul": {"share_pct": _pct(r["long_haul_share"])}}
        item = {"rank": position, "iata": r["iata"], "name": r["name"], "score": round(float(r["score"]), 3),
                "components": {c: {**raw[c], "percentile": _pct(r[f"pct_{c}"], 0)} for c in components},
                "confidence": r["confidence"]}
        if sort_by != "score":
            item["score_rank"] = int(r["rank"])
            item["sort_value"] = {"growth": _pct(r["growth"]), "load_factor": _pct(r["load_factor"]),
                                  "passengers": int(r["passengers"])}[sort_by]
        if scenario:
            item["base_rank"] = int(base_rank[r["iata"]])
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
    if scenario:
        ignored = sorted(k for k, v in weights.items() if k in result.unavailable and v > 0)
        assumptions.insert(0, "scenario: weights changed by the user, not the configured methodology; "
                              "weights not given keep their configured value, then all are renormalized"
                              + (f"; ignored (component not built): {', '.join(ignored)}" if ignored else ""))
    if sort_by != "score":
        assumptions.insert(0, f"sorted by {SORT_LABELS[sort_by]} (highest first), not by the expansion score")
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
    return _envelope(conn, window, data, assumptions, confidence, facts,
                     headline=_ranking_headline(data, key, sort_by, scenario))


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
                             topic: Topic = "all") -> dict:
        """Metrics for one airport (IATA code or name). Set topic to what the question asks about:
        long_haul (share of departures >= threshold, all flights and passenger service, with
        sensitivity), demand (passengers, seats, load factor), growth (passengers vs the previous
        12 months); all when it asks for several. threshold_mi only for long_haul (default from config)."""
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

    return [data_status_tool, resolve_airport_tool, airport_metrics_tool, rank_airports_tool]
