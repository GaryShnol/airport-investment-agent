# Airport Investment Agent

LangGraph ReAct agent over a local SQLite DB, with deterministic scoring. Data: BTS T-100 Segment, BTS On-Time, Census API. 24h take-home for a Deloitte Digital FDE role: clarity and reasoning over completeness. Everything here is reviewed in an interview, so keep it simple, explainable and tidy.
Must answer: (1) New England expansion candidates (2) LAX vs SNA congestion (3) % long-haul out of ANC (4) SFO unmet demand and why.
Plan, schema, tools, scoring, acceptance values: `docs/plan.md` (read before building a step). Task board and status: `TASKS.md` (update it as you work).

## Commands (uv; no pip, no manual venv). Commands marked (planned) are not built yet; see TASKS.md
- `uv sync` install · `uv add <pkg>` add a dependency · `uv run pytest`
- (planned) `uv run python -m src.ingest` load new files into SQLite (idempotent) · `uv run python -m src.ingest --check` report whether upstream has newer months
- (planned) `uv run streamlit run app.py` · `docker compose up` (first start downloads the seed DB from `SEED_DB_URL`, a GitHub Release asset; needs only `.env` with an LLM key or a local model)
- Do not run uv or pip inside the cloud sandbox against the user's folder: it would create a Linux `.venv` in a Windows project.

## Rules
- The LLM never computes numbers. It picks tools and phrases results; all metrics come from SQL and `src/scoring.py`.
- No hardcoded values. Infra and secrets (provider, model, URLs, keys, paths) come from `.env` through `src/config.py`. Scoring methodology (weights, thresholds, regions, sensitivity) lives in `config/weights.yaml`. Code reads both only via `src/config.py`.
- LLM access only through `src/llm.py` (`get_chat_model(settings)`): a registry of provider builders (openai, ollama). OpenAI-compatible local servers work through `LLM_BASE_URL`. Adding a provider = one builder function + one registry entry.
- No network calls at chat time. Network is used only by `ingest` (offline refresh).
- One analysis window for all questions: trailing 12 complete months ending at `min(last_period)` across sources. Growth compares to the previous 12 months.
- Stale = upstream has a newer month than the loaded data (from the last `ingest --check`, stored in `meta`). Never checked -> say "freshness unknown". Still answer, with a warning.
- Every answer states data period, assumptions, confidence and the weights used.
- Mandatory content is guaranteed by code, not by the prompt: tools mark required `facts`, and `ensure_facts` in `src/agent.py` appends any the answer omits. Do not fix a missing figure by tuning the prompt.
- Ambiguous entity (e.g. "LA") -> ask, or state the definition used (LA = LAX).
- Reply in the language of the user's question (default English). Code, docs and tool output in English.
- Never commit `data/` or `.env`; never print keys. The seed DB (`uv run python -m src.seed build` -> `data/seed/airport.db.gz` + `.sha256`) is not committed: it is published as a GitHub Release asset and fetched by `src.bootstrap`.
- Diagrams (architecture, request flow) live only in `README.md` as mermaid blocks, which GitHub renders. Edit them there; do not copy them elsewhere.
- Work in small steps; update `TASKS.md` and commit per task.

## Traps
- ANC is cargo-dominated: long-haul share differs hugely with/without cargo (all configs 43.2% vs passenger-only at 2,500 mi, 2025: 11.4% for config 1 only, 11.5% for passenger service = config 1 and 3). Always report both, the threshold and a sensitivity (1,500 / 2,500 / 3,000 mi).
- T-100: read AIRCRAFT_CONFIG/CLASS/AIRCRAFT_TYPE as strings; config 1=pax, 2=cargo, 3=both, 4=undocumented. Drop rows with DEPARTURES_PERFORMED = 0. ORIGIN_STATE_ABR is null for most foreign origins but holds province codes for Canada (YYZ = ON) and TT for Guam/Saipan; use `airports.state` for "US". T-100 totals are higher than the ArcGIS NTAD layer (ratio 1.0-1.45); T-100 is the source of truth, see Data validation in `docs/plan.md`.
- On-Time: ~7M rows for 2025. Stream and aggregate, never load all; read only needed columns. Numbers are strings like `0.00`; FL_DATE looks like `1/1/2025 12:00:00 AM`; delay-cause columns are empty unless the flight was delayed (empty is not zero). Domestic, large carriers only, so small airports can be missing: renormalize and lower confidence.
- Load factor = passengers / seats on passenger flights; it is a demand-pressure proxy, not terminal utilization (no gate or terminal-area data). "Unmet demand" is a proxy (spill estimate), never a field.
- Definitions (passenger service, peer set, window, On-Time origin vs destination sides) are fixed in `docs/plan.md`; do not redefine them in code. BTS records delay causes for arrival delays, so cause minutes belong to the destination airport; departure delay and taxi-out belong to the origin.
- Delay causes: carrier delay is airline-driven, not infrastructure; weather is not fixable by a terminal. Congestion uses delay rate, taxi-out and NAS-delay minutes per flight.
- BTS On-Time direct files exist: `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YYYY}_{M}.zip` (~30 MB, full columns). A missing month returns 404 with `text/html`; validate the ZIP and Content-Type. Latest available as of 2026-10-10: 2026-07 (publication lag about 1.5-2 months).
- BTS T-100 Segment download is an ASP.NET form (`DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45`): POST with `__VIEWSTATE`, `__VIEWSTATEGENERATOR`, `__EVENTVALIDATION`, `cboYear`, `cboPeriod`, field checkboxes, `chkDownloadZip`, `btnDownload`. Page says "Latest Available Data: July 2026". Not yet verified end to end; manual download + local file is the fallback. One request at a time; validate ZIPs with `zipfile`.
- Census: key in the query string (`key=`), kept in `.env` as `CENSUS_API_KEY` (needed only by ingest); ACS 5-year is smoothed and values come back as strings; Boston CBSA = 14460. UI must show: "This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau."
- The cloud sandbox cannot reach external sites (api.census.gov, transtats.bts.gov, davidmegginson.github.io); run network fetches on the user's machine.
- `airports.csv` (OurAirports) is needed for IATA -> name/state/lat/lon. It has no CBSA: airport -> CBSA comes from the Census Geocoder (lat/lon), cached in the DB (and so in the seed).
- FAA NAS Status was considered and dropped: it is live operational status only, no history, passengers or distances.

## Defaults (all in `config/weights.yaml`, change there)
- `Score = 0.25 Utilization + 0.25 Congestion + 0.15 Scale + 0.15 Growth + 0.10 LongHaul + 0.10 DemandContext`
- Congestion = 0.40 departure-delay rate + 0.30 taxi-out + 0.30 NAS-delay minutes per arriving flight (cancellations shown, not scored: mostly weather).
- Long-haul threshold 2,500 mi; min 100,000 passenger-service passengers/yr to be ranked; min 1,000 On-Time departures in the window to score congestion; long-haul share under 2% counts as 0 in the score; unmet-demand target load factor 0.80; New England = MA CT RI NH VT ME.
