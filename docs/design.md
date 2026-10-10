# Design: Airport Investment Intelligence Agent

Values (weights, thresholds, floors) live in [config/weights.yaml](../config/weights.yaml), definitions and implementation detail in [plan.md](plan.md), decisions with their trade-offs in the decision log of [TASKS.md](../TASKS.md).

## 1. Problem and approach
An analyst asks in plain language where terminal investment is most likely to pay off and needs an answer they can defend.
Public data has no terminal-capacity figures, so the agent measures capacity pressure: load factor, congestion, volume, growth, long haul, metro growth.
A LangGraph ReAct agent reads the question and calls six small tools over a local SQLite DB built offline from BTS and Census data.
Every number, ranking and verdict is computed in SQL and a deterministic scoring module; the model picks tools and writes short commentary.
Every answer states its data period, assumptions, confidence, weights used and data freshness.

## 2. Architecture
Both diagrams are in the [README](../README.md#architecture). The Streamlit chat (or the CLI, `python -m src.agent`) passes the question to the agent, which calls `data_status`, `resolve_airport`, `airport_metrics`, `compare_airports`, `rank_airports` or `explain_demand_gap`. Each tool returns an envelope: a deterministic headline, labeled facts (some required), period, freshness, assumptions and confidence. The UI renders the card from the envelope in code and shows the model's commentary under its own label. `ingest` (offline, idempotent) builds the DB and `ingest --check` records whether upstream has newer months. The DB ships as a gzipped GitHub Release asset that `src.bootstrap` downloads and verifies on first start.

## 3. Scoring methodology
- **Components**: utilization (seat load factor, passenger service); congestion (departure-delay rate, taxi-out, NAS-delay minutes per arriving flight; cancellations shown, not scored); scale (annual passengers); growth (vs the previous 12 months); long haul (share of departures above the distance threshold, shares under a floor count as 0); demand context (population growth of the airport's metro or micro area, Census ACS 5-year).
- **Percentiles in the peer set**: each component is a percentile among US airports above a passenger floor, so all share one scale and the score (weighted mean, 0-1) reads as an average national percentile. Airports are then ranked inside the requested region.
- **Missing component** (too few On-Time flights, no metro area): the remaining weights are renormalized for that airport and its confidence drops to medium; the card says why.
- **Confidence**, set in code: high = every component over 12 months; medium = a component renormalized or partial coverage; low = fewer than 12 months in the window. The unmet-demand proxy is capped at medium.
- **Rank stability**: the weights are perturbed at random (seeded) and the region re-ranked many times; the card shows each airport's share of runs in the top 3 and its rank range, and flags an unstable #1. It tests the weights, not the data.
- **What the score is not**: a measure of terminal need. It is a capacity-pressure proxy and cannot see gates, terminal area, airline plans or costs.

## 4. Key tradeoffs
- **Deterministic scoring, not the LLM**: numbers must be reproducible and auditable; the price is more code.
- **Proxy for unmet demand**: T-100 counts passengers carried, never those turned away, so the proxy is passengers above a target load level, with a range, labeled as a proxy.
- **One analysis window** (trailing 12 complete months ending at the older BTS source): all answers use the same months; the faster source's newer months wait.
- **2% long-haul floor**: without it a few charters lifted small airports to the 65th percentile of a component meant to signal wide-body facilities.
- **ACS population as demand context**: the only Census series covering every metro and micro area; growth over non-overlapping 5-year periods, as consecutive vintages share four of five years.
- **Seed DB as a release asset**: keeps a 10 MB binary out of git and pins one data version per URL; first start needs the network once.
- **Restricted commentary for rankings and comparisons**: an ordering or verdict cannot be checked by a figure check, and an early eval run rewrote the New England order, so the model may only add caveats there.
- **Swappable model service**: the model is a service selected in `.env`, OpenAI for the reviewer quickstart (fast, reliable tool calls) and a local Ollama model for development (free, offline, slower); local models call tools less reliably, hence few typed tools. On the 14 eval questions (2026-10-10), gpt-4o-mini passed 14 of 14 in 34 s in total after M6 (12 of 14 before), gpt-5-mini 11 of 14 in 105-121 s before M6, so gpt-4o-mini is the reviewer default.

## 5. Where AI is used, and where it is not
Used for: understanding the question, choosing the tool, extracting parameters (airport names, region, topic, sort order, scenario weights such as "growth 0.30"), asking when a name is ambiguous, and short commentary.
Not used for: any number, ranking, comparison verdict, stability result or unmet-demand figure. These come from SQL and `src/scoring.py`, and the card is rendered from the envelopes by code.
Guardrails, in code: the figure check (`src/figure_check.py`) removes commentary sentences whose numbers are not in the tool results or are credited to the wrong airport; after rankings and comparisons only caveat sentences without digits or airport names are kept; `ensure_facts` appends required facts the commentary omits.

## 6. Answers on the shipped data
Tool output on the seed `data-2025`: window 2025-01..2025-12, freshness stale (BTS has published up to 2026-07). Newer data will change these.

**New England candidates** (`rank_airports`, confidence medium: HVN, ACK, ORH lack a component). BOS 0.755, BTV 0.663, PWM 0.605, BGR 0.590, PVD 0.489; 10 ranked, 45 below the passenger floor. Stability: BOS first in 20 of 20 runs; BOS and BTV always top 3, PWM in 80% of runs, BGR in 20%. Drivers: BOS long haul (96th percentile), BTV congestion (79th), PWM utilization (87th). With growth at 0.30, BGR moves to third and PWM to fourth. Caveat: a pressure ranking, not a business case.

**LAX vs SNA congestion** (`compare_airports`; asked as "LA", the card states LA = LAX and gives medium confidence for that assumption, high when the model passes LAX itself). Delayed departures: SNA higher (19.2% vs 18.1%). Taxi-out: LAX higher (17.7 vs 16.1 min). NAS delay per arrival: no meaningful difference (1.97 vs 1.76 min, inside the 95% margins). Overall: mixed. Caveat: On-Time covers the large carriers only, and LAX has 5.1 times SNA's departures.

**ANC long haul** (`airport_metrics`, confidence high). At 2,500 mi: 43.2% of all departures including cargo (37,969 of 87,988), 11.5% of passenger service (4,494 of 39,025). At 1,500 mi 49.1% / 19.3%; at 3,000 mi 31.0% / 4.3%. Caveat: a cargo hub, so the base decides the answer; both are always given.

**SFO demand gap** (`explain_demand_gap`, a PROXY, confidence medium). Load factor above 80% in 10 of 12 months (peak 2025-06, 89.4%). Proxy: 1,427,569 passengers above the 80% load level (5.4%), range 2,648,904 at 75% to 559,193 at 85%. Why: full flights most months, the proxy concentrated on routes to EWR, JFK, ORD, NAS the largest arrival delay cause (34.3% of minutes), passengers +4.7%. Caveat: undercounts true spill; shows where demand presses, not how much is lost.

## 7. Uncertainty, limits, next steps
Policy: state the period, definitions, confidence and freshness every time; ask about an ambiguous name or state the default used ("LA" = LAX); label proxies; answer stale data with a warning rather than refuse; report On-Time differences inside the margin of error as no difference.
Limits: no terminal or gate data; On-Time misses smaller carriers (15 of 233 peer airports have no congestion value); percentiles depend on the peer set and its floor; stability covers weights only; the binomial margins assume independent flights (optimistic).
Next steps: automated On-Time download (direct monthly files, already probed by `--check`); automated T-100 download (an ASP.NET form, manual files as fallback); scheduled ingest and seed release to move the window into 2026; voice input (`st.audio_input`).
