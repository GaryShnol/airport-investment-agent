# Build plan (read when starting a step; not auto-loaded)
Task status lives in `TASKS.md`. This file holds the design.

## Configuration (no hardcoded values)
- `.env` (see `.env.example`) -> `src/config.py` `Settings.from_env()`: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_TEMPERATURE`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `CENSUS_API_KEY`, `DB_PATH`, `SEED_DB_PATH`, `RAW_DATA_DIR`, `WEIGHTS_PATH`, `LOG_LEVEL`. Secrets are hidden from `repr`. Relative paths resolve against the project root.
- `config/weights.yaml` -> `load_scoring()` validates (weights sum to 1) and returns `ScoringConfig`: weights, congestion sub-weights, long-haul miles and sensitivity list, min departures, target load factor, sensitivity runs/jitter/seed, regions.
- Why split: env is per-deployment and secret; methodology is reviewable, versioned and part of the design doc.

## LLM provider
`src/llm.py`: `PROVIDERS = {"openai": ..., "ollama": ...}`; `get_chat_model(settings)` returns a LangChain `BaseChatModel`. Lazy imports. `openai` accepts `LLM_BASE_URL` for any OpenAI-compatible server (Ollama `/v1`, LM Studio, vLLM). No class hierarchy: LangChain already provides the common interface. Local models are weaker at tool calling, so tools are few, typed and return small JSON; the agent is tested with a fake model too.

## SQLite schema
- `meta(source, last_period, upstream_latest, checked_at, fetched_at)`
- `airports(iata, name, state, region, cbsa, lat, lon)`
- `t100_route_month(origin, dest, year, month, config, distance, departures, seats, passengers)`  config = pax | cargo | combi | unknown
- `ontime_airport_month(airport, year, month, dep_flights, dep_del15, dep_cancelled, taxi_out_sum, taxi_out_n, arr_flights, arr_del15, carrier_min, weather_min, nas_min, security_min, late_aircraft_min)`  departure side at the origin airport; arrival side and cause minutes at the destination airport
- `census_metro(cbsa, year, population, fetched_at)`

## Ingest (offline, idempotent, `src/ingest.py`)
- Reads files from `RAW_DATA_DIR`; skips (source, year, month) already in `meta`; validates ZIPs with `zipfile`; aggregates On-Time in a stream with `usecols`.
- `--check`: HEAD the next On-Time month (404 = not published) and read T-100 "Latest Available Data"; stores `upstream_latest`, `checked_at`; prints what is missing. No downloads.
- Optional (stretch): `--download` fetches missing On-Time months (direct URLs) and T-100 (form POST). Any failure falls back to local files.
- `seed/airport.db` is built from the aggregated tables and committed; the container copies it to the `DB_PATH` volume if absent.

## Tools and envelope (6 tools)
`data_status`, `resolve_airport`, `airport_metrics`, `compare_airports`, `rank_airports`, `explain_demand_gap`.
Every tool returns:
```json
{"data": {}, "data_period": "2025-01..2025-12", "freshness": "current|behind|unknown",
 "assumptions": ["long-haul >= 2500 mi", "passenger flights only"], "confidence": "high|medium|low"}
```
- `airport_metrics(airport)`: volume, load factor, congestion, long-haul share with and without cargo plus sensitivity at each configured threshold.
- `compare_airports(a, b)`: side by side with normalized metrics.
- `rank_airports(region, top_n)`: scores, component breakdown, weights, rank stability.
- `explain_demand_gap(airport)`: peak months, highest load-factor routes, spill proxy (seats needed to bring dense routes to the target load factor), delay-cause mix.
- `resolve_airport(text)`: name or city -> IATA; ambiguity returns candidates for the agent to ask about.

## Definitions (fixed; used by ingest, scoring and tools)
- Passenger service: T-100 rows with AIRCRAFT_CONFIG 1 (passenger) or 3 (combi) and DEPARTURES_PERFORMED > 0. Cargo (2) and undocumented (4) are excluded from passenger metrics; they count only in "all flights" figures.
- Peer set: airports with a US state (`ORIGIN_STATE_ABR` not null) and at least `min_departures_per_year` passenger-service departures in the window. Percentiles are computed inside this set.
- Window: trailing 12 complete months ending at `min(last_period)`; growth compares with the previous 12.
- On-Time sides: departure delay rate, taxi-out and cancellations belong to the origin airport. Arrival counts and delay-cause minutes belong to the destination airport, because BTS records delay causes for arrival delays.

## Scoring (`src/scoring.py`, pure functions, unit-tested)
Component = percentile inside the peer set; airports are ranked inside the requested region afterwards. Raw values are always shown next to the percentiles.
- Utilization = pax / seats (passenger service)
- Scale = annual passengers (passenger service)
- Congestion = weighted percentiles: departure-delay >=15 rate, mean taxi-out, NAS-delay minutes per arriving flight
- Growth = passengers in the window vs the previous 12 months
- LongHaul = departures >= threshold / passenger-service departures
- DemandContext = metro population growth (Census)
- Why Scale: profit depends on volume as well as pressure. Without it a small airport with fast growth outranks a large one (HVN +25% on 0.7M passengers vs BOS on 21M).
- Why LongHaul is only 0.10: it hints at wide-body and international facilities, but it is secondary and distorted by cargo.
- Missing component -> renormalize weights, lower confidence, say so.
- Question mapping: Q2 = Congestion; Q3 = LongHaul; Q4 = Utilization + Congestion + delay causes + spill proxy.
- Confidence (deterministic): high = all components, full 12 months, in the peer set; medium = a component renormalized or partial coverage; low = fewer than 12 months or proxy-only. Unmet demand is capped at medium.
- Rank stability: re-rank with N random weight perturbations (seeded); report each airport's rank range and whether the top-3 holds.

## Agent and UI
- LangGraph ReAct agent (`create_react_agent` from `langgraph.prebuilt`, or `create_agent` from `langchain.agents` if that is the current API in the installed version; same engine) with a checkpointer and `thread_id` for follow-ups. System prompt: scope rules, assumptions policy, clarify when ambiguous, reply in the user's language.
- Output check: numbers in the answer should appear in tool results (tolerant to rounding); a soft warning, not a retry loop.
- Streamlit chat, sidebar with data period and freshness, Census attribution, and a collapsible panel listing the tools called and their envelopes (shows the reasoning and that numbers come from tools). Streamlit reruns the script on every message, so the agent and its memory live in `st.cache_resource` and the `thread_id` in `st.session_state`. SQLite is opened read-only (`mode=ro`) per call. Voice is a stretch goal (`st.audio_input`).

## Packaging
Dockerfile + `docker-compose.yml` (app service, `env_file: .env`, volume for the DB, entrypoint seeds the DB). Local model from a container: `OLLAMA_BASE_URL=http://host.docker.internal:11434`. GitHub Actions runs `pytest`. README: two-command quickstart. `docs/design.md`: scoring methodology, tradeoffs, where AI is used, limits, production path (scheduled ingest, tracing, secrets, evals).

## Acceptance values (calendar 2025; become golden tests in P3.5)
These came from the exploratory pass (passenger-only = config 1; SFO load factor = rows with seats). Re-baseline in P2.2 under the Definitions above; the ANC passenger-only share may move by about 0.1 point because combi flights are now included, and the new numbers replace these.
- T-100 2025 ANC, all configs: 87,988 departures; share >= 2,500 mi = 43.2%; passenger-config only = 11.4%.
- T-100 2025 SFO load factor = 82.3%.
- On-Time 2025: 7,001,619 rows; January = 539,747.

## Data validation (first pass on T-100 2024-2025, done before building)
- Profile: 1,121,004 rows, 2,189 origins. Rows with 0 departures: 0.2%, none with passengers. No negative values. Config counts: pax 984,737, cargo 117,440, undocumented 9,905, both 8,922.
- Cross-check against the ArcGIS NTAD 2024 layer (T-100 / ArcGIS): passengers ANC 1.01, SNA 1.04, BOS 1.26, SFO 1.43, LAX 1.45; departures 1.03 to 1.33. The layer looks like a subset, with the largest hubs most affected (hypothesis; confirm against FAA enplanement totals). T-100 is used as the source of truth.
- ANC share of departures by distance (all configs / passenger-only), 2025: >=1,500 mi 49.1% / 19.4%; >=2,500 mi 43.2% / 11.4%; >=3,000 mi 31.0% / 4.0%. Cargo-config alone: 68.4% at 2,500 mi. 2024 is lower (40.0% / 8.2% at 2,500 mi).
- SFO load factor 82.3% in 2025 (83.2% in 2024), 6th of 55 airports with at least 5M seats. Monthly range 77.2% (January) to 89.4% (June). Densest routes: HNL 90.6%, EWR 89.2%, MIA 89.2%, JFK 88.2%.
