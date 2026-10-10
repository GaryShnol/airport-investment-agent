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
- `t100_route_month(origin, dest, year, month, config, distance, departures, seats, passengers)`  config = pax | cargo | both
- `ontime_airport_month(airport, year, month, flights, dep_del15, cancelled, taxi_out_sum, taxi_out_n, carrier_min, weather_min, nas_min, security_min, late_aircraft_min)`
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

## Scoring (`src/scoring.py`, pure functions, unit-tested)
Component = percentile vs all US passenger airports with >= `min_departures` in the window; rank inside the region afterwards.
- Utilization = pax / seats (passenger config)
- Congestion = weighted percentiles: departure-delay >=15 rate, taxi-out, NAS-delay minutes per flight
- LongHaul = departures >= threshold / passenger departures
- Growth = YoY passengers (window vs previous 12 months)
- DemandContext = metro population growth (Census)
- Missing component -> renormalize weights, lower confidence, say so.
- Question mapping: Q2 = Congestion; Q3 = LongHaul; Q4 = Utilization + Congestion + delay causes + spill proxy.
- Confidence (deterministic): high = all components, full 12 months, above min departures; medium = a component renormalized or partial coverage; low = fewer than 12 months or proxy-only. Unmet demand is capped at medium.
- Rank stability: re-rank with N random weight perturbations (seeded); report each airport's rank range and whether the top-3 holds.

## Agent and UI
- LangGraph ReAct (`create_react_agent`) with a checkpointer and `thread_id` for follow-ups. System prompt: scope rules, assumptions policy, clarify when ambiguous, reply in the user's language.
- Output check: numbers in the answer should appear in tool results (tolerant to rounding); a soft warning, not a retry loop.
- Streamlit chat, sidebar with data period and freshness, Census attribution. Voice is a stretch goal (`st.audio_input`).

## Packaging
Dockerfile + `docker-compose.yml` (app service, `env_file: .env`, volume for the DB, entrypoint seeds the DB). Local model from a container: `OLLAMA_BASE_URL=http://host.docker.internal:11434`. GitHub Actions runs `pytest`. README: two-command quickstart. `docs/design.md`: scoring methodology, tradeoffs, where AI is used, limits, production path (scheduled ingest, tracing, secrets, evals).

## Acceptance values (from data already analyzed, calendar 2025)
- T-100 2025 ANC, all configs: 87,988 departures; share >= 2,500 mi = 43.2%; passenger-config only = 11.4%.
- T-100 2025 SFO load factor = 82.3%.
- On-Time 2025: 7,001,619 rows; January = 539,747.
