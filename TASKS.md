# Tasks

Task board for the Airport Investment Intelligence Agent (Deloitte Digital FDE exam, 24h).
Status legend: [x] done · [~] in progress / written, not yet verified · [ ] todo · (stretch) only if time allows.
Last updated: 2026-10-10. Rule: finish a task, tick it here, commit with the task id (for example `P2.1: T-100 ingest`).

## Phase 0 - Discovery and decisions
- [x] P0.1 Read the exam brief and the job post; derive what is evaluated (reasoning, clarity, production thinking, not completeness)
- [x] P0.2 Choose sources: BTS T-100 Segment (volume, seats, distance), BTS On-Time (delays, taxi, delay causes), Census API (metro population)
- [x] P0.3 Download T-100 2024-2025 and the 12 On-Time months of 2025
- [x] P0.4 First-pass analysis and cross-check against ArcGIS NTAD (`reports/t100_first_pass.txt`), acceptance values recorded
- [x] P0.5 Document traps (cargo at ANC, config codes, empty vs zero delay causes, proxies) in `CLAUDE.md`
- [x] P0.6 Feasibility checks: On-Time direct files work (404 when unpublished, latest 2026-07); T-100 download is a scriptable form (latest 2026-07); Census key works
- [x] P0.7 Design review and decisions (see Decision log)
- [x] P0.8 Align `CLAUDE.md`, `docs/plan.md` and the README diagrams (single source, no duplicates)

## Phase 1 - Foundations
- [x] P1.0 Repo hygiene: cleanup, `.gitignore`, `.gitattributes` (LF), first commit
- [x] P1.1 Config: `.env.example`, `config/weights.yaml`, `src/config.py` (no hardcoded values)
- [x] P1.2 LLM provider registry `src/llm.py` (openai, ollama, OpenAI-compatible via base URL)
- [~] P1.3 Unit tests for config and provider (`tests/test_config.py`) - written, run `uv run pytest`
- [~] P1.4 `pyproject.toml` deps and `.gitignore` updated - run `uv sync` to create the lock file
- [ ] P1.5 Download `airports.csv` (OurAirports) on the user's machine
- [ ] P1.6 Create the real `.env` (key or local model) and smoke-test `get_chat_model`

## Phase 2 - Data layer
- [ ] P2.1 `src/store.py`: SQLite schema and queries (schema in `docs/plan.md`)
- [ ] P2.2 T-100 ingest to `t100_route_month`; verify ANC 87,988 / 43.2% / 11.4% and SFO 82.3%
- [ ] P2.3 On-Time stream ingest to `ontime_airport_month`; verify 7,001,619 rows and January 539,747
- [ ] P2.4 `ingest --check` (HEAD next On-Time month, T-100 latest) and `meta` freshness fields
- [ ] P2.5 Census ingest: Geocoder lat/lon to CBSA (cached) and population (kept last, see Decision log)
- [ ] P2.6 Build and commit `seed/airport.db`
- [ ] P2.7 (stretch) `ingest --download`: On-Time direct URLs and T-100 form POST, fallback to local files
- [ ] P2.8 (stretch) extend the window to 2026-07 using P2.7

## Phase 3 - Scoring (deterministic)
- [ ] P3.1 Tests first: percentile, load factor, long-haul share, growth, renormalization when a component is missing
- [ ] P3.2 `src/scoring.py`: components, weights from YAML, deterministic confidence
- [ ] P3.3 Rank stability under weight perturbation (seeded)
- [ ] P3.4 Spill proxy for unmet demand (target load factor from YAML)
- [ ] P3.5 Golden tests on the four exam questions

## Phase 4 - Tools and agent
- [ ] P4.1 Tool envelope and `data_status`, `resolve_airport`
- [ ] P4.2 `airport_metrics`, `compare_airports`, `rank_airports`, `explain_demand_gap`
- [ ] P4.3 `src/agent.py`: LangGraph ReAct agent, memory by `thread_id`, system prompt (scope, assumptions, clarify, user language)
- [ ] P4.4 Soft output check (numbers appear in tool results)
- [ ] P4.5 Agent test with a fake model (tool routing, ambiguity "LA", stale warning)
- [ ] P4.6 Try a local model (Ollama) and note tool-calling quality in the design doc

## Phase 5 - UI
- [ ] P5.1 `app.py` Streamlit chat with follow-ups
- [ ] P5.2 Sidebar: data period, freshness, provider; Census attribution
- [ ] P5.3 (stretch) voice input via `st.audio_input`

## Phase 6 - Packaging and documentation
- [ ] P6.1 `Dockerfile`, `docker-compose.yml`, entrypoint that seeds the DB volume
- [ ] P6.2 GitHub Actions: `pytest`
- [~] P6.3 `README.md`: draft with diagrams (architecture and flow, single source), scoring and limits is in; finalize the quickstart when the app runs
- [ ] P6.4 `docs/design.md`: scoring methodology, key tradeoffs, where AI is used, limits and assumptions, production path
- [x] P6.5 Cleanup: removed one-off download helpers, a duplicate ZIP, an empty temp folder and caches (done at the first commit)
- [ ] P6.6 Final check: fresh clone, `docker compose up`, run the four questions plus follow-ups

## Decision log (for the interview)
| Decision | Why | Trade-off |
|---|---|---|
| LLM never computes numbers; tools return an envelope | Trust and auditability, exam requires deterministic scoring | More code than letting the model reason over tables |
| SQLite with pre-aggregated tables and a committed seed DB | Runs offline in seconds, reproducible for the reviewer | Data frozen until `ingest` is rerun |
| Ingest is offline and idempotent, with `--check` | Chat stays fast and stable; freshness is measured, not guessed | No live data in chat |
| One analysis window (trailing 12 months, `min(last_period)`) | Consistent across sources, easy to explain | Newer months wait for the slower source |
| Stale means upstream has a newer month | Honest (publication lag is about 2 months) | Needs the network once to know |
| Provider registry in `src/llm.py`, selected by `.env` | Swap OpenAI and local models without code changes, cheap tests | Local models call tools less reliably |
| Methodology in `config/weights.yaml`, infra in `.env` | Reviewable and versioned vs per-deployment and secret | Two config sources, one loader |
| Score is a capacity-pressure proxy, not a terminal-need measure | No gate or terminal-area data in public sources | Stated as a limit in every answer |
| Congestion = delay rate, taxi-out, NAS delay per flight; cancellations not scored | Carrier delay is airline-driven, cancellations are mostly weather | Fewer signals |
| Census kept (10%), built last; FAA NAS Status dropped | Census works with a key; FAA is live status only, no history | Needs an airport-to-CBSA mapping (Geocoder, cached) |
| Replies in the user's language, English default | Global reviewers; Hebrew works too | None |
| Six small tools | Reliable tool choice, also for local models | Less flexible than free-form SQL |

## Demo script
1. Which airports in New England are strong candidates for terminal expansion? (ranking, weights, stability, confidence)
   - Follow-up: why is BDL below PVD? What if growth weighs more?
2. Compare LA and Santa Ana congestion. (agent states LA = LAX, offers BUR/LGB/ONT)
3. What percentage of long-haul flights leave Anchorage? (both numbers, thresholds, cargo explanation)
4. What is the unmet demand at SFO and why? (proxy, peak months, dense routes, delay causes, confidence capped at medium)
