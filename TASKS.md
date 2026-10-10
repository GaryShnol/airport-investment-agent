# Tasks

Task board for the Airport Investment Intelligence Agent (Deloitte Digital FDE exam, 24h budget).
Legend: [x] done · [~] written, not yet verified · [ ] todo · (stretch) only if time allows.
Last updated: 2026-10-10. Current milestone: **M2**.

## Working agreements
- Build in vertical slices: after M2 the app already answers questions end to end; later milestones add data and depth.
- One branch per milestone (`m2-vertical-slice`), merged into `main` by pull request. `main` stays green: CI runs `pytest`.
- Commit per task, message prefixed with the task id (for example `P2.2: T-100 ingest`). Tick the task here in the same commit.
- Tests first for scoring. No hardcoded values: `.env` through `src/config.py`, methodology in `config/weights.yaml`.

| Milestone | Goal | Rough time |
|---|---|---|
| M0 Discovery | Decisions, data downloaded, plan | done |
| M1 Foundations | Config, provider, CI, environment works | 1 h |
| M2 Vertical slice | T-100 only: store, scoring, tools, agent, chat UI | 5 h |
| M3 Congestion | On-Time: congestion, LAX vs SNA, SFO "why", rank stability | 4 h |
| M4 Context and freshness | Census, `ingest --check`, seed DB | 3 h |
| M5 Package and document | Docker, README, design doc, fresh-clone test | 4 h |
| Buffer | Debugging, polish, demo rehearsal | 4 h |

## M0 - Discovery and decisions
- [x] P0.1 Read the exam brief and job post; derive what is evaluated (reasoning, clarity, production thinking)
- [x] P0.2 Choose sources: BTS T-100 Segment, BTS On-Time, Census API
- [x] P0.3 Download T-100 2024-2025 and the 12 On-Time months of 2025
- [x] P0.4 First-pass analysis and cross-check against ArcGIS NTAD; findings and acceptance values in `docs/plan.md` (Data validation); the exploratory script and raw report are in git history
- [x] P0.5 Document data traps in `CLAUDE.md`
- [x] P0.6 Feasibility: On-Time direct files work (latest 2026-07); T-100 download is a scriptable form (latest 2026-07); Census key works
- [x] P0.7 Design review and decisions (Decision log)
- [x] P0.8 Align `CLAUDE.md`, `docs/plan.md` and the README diagrams (single source)
- [x] P0.9 Repo hygiene: cleanup, `.gitignore`, `.gitattributes`, first commits, pushed to GitHub

## M1 - Foundations
- [x] P1.1 Config: `.env.example`, `config/weights.yaml`, `src/config.py`
- [x] P1.2 LLM provider registry `src/llm.py` (openai, ollama, OpenAI-compatible via base URL)
- [x] P1.3 Config and provider tests (`tests/test_config.py`): 8 of 8 pass with `uv run pytest`
- [x] P1.4 `uv sync` and commit `uv.lock` (reproducible installs)
- [x] P1.5 GitHub Actions: `uv run pytest` on push and pull request
- [x] P1.6 Download `airports.csv` (OurAirports) on the user's machine (86,231 rows, 2,036 US airports with IATA)
- [x] P1.7 Create the real `.env`; smoke-test `get_chat_model` with a local Ollama model (works; finding: the reply contained a `<think>` block, handled in P2.7 and P2.10)
- [x] P1.8 Check which agent constructor the installed LangGraph/LangChain version provides

## M2 - Vertical slice on T-100 (Q3 complete, Q1 and Q4 partial) (current)
- [x] P2.1 `src/store.py`: SQLite schema and queries (schema in `docs/plan.md`)
- [x] P2.2 Ingest T-100 to `t100_route_month`; verify ANC 87,988 departures and the long-haul shares, and SFO load factor; re-baseline the acceptance values in `docs/plan.md` under the passenger-service definition
- [x] P2.3 Ingest `airports.csv` to the `airports` table (IATA, name, state, region, lat, lon)
- [x] P2.4 Scoring tests first: percentile, load factor, long-haul share, growth, renormalization when a component is missing
- [x] P2.5 `src/scoring.py` with the available components (utilization, scale, growth, long haul) and deterministic confidence
- [ ] P2.6 Tool envelope and tools: `data_status`, `resolve_airport` (handles "LA" ambiguity), `airport_metrics`, `rank_airports`
- [ ] P2.7 `src/agent.py`: LangGraph ReAct agent, memory by `thread_id`, system prompt (scope, assumptions, clarify, user language), light input guard; strip any reasoning (`<think>`) block from model output before it reaches the user
- [ ] P2.8 Agent test with a fake model: tool routing, ambiguity, out-of-scope, missing LLM key message
- [ ] P2.9 `app.py` Streamlit chat with follow-ups; agent and memory in `st.cache_resource`, `thread_id` in `st.session_state`, SQLite read-only per call; sidebar with data period and provider; collapsible panel with the tools called and their envelopes
- [ ] P2.10 Small eval (8-10 fixed questions with the expected tool) on local models (qwen3:4b-instruct-2507, lfm2.5:8b, gpt-oss:20b); record tool-choice accuracy and latency; set `num_ctx` in config; decide the reasoning setting (`reasoning` parameter of ChatOllama, verify it exists in the installed version)

## M3 - Congestion and depth (Q2 complete, Q4 complete)
- [ ] P3.1 Ingest On-Time (stream, `usecols`) to `ontime_airport_month` (departure side at origin, arrival side and cause minutes at destination); verify 7,001,619 rows and January 539,747
- [ ] P3.2 Congestion component (delay rate, taxi-out, NAS delay per arriving flight) in scoring, with tests
- [ ] P3.3 `compare_airports` (LAX vs SNA, normalized)
- [ ] P3.4 `explain_demand_gap` for SFO: peak months, dense routes, spill proxy, delay-cause mix; confidence capped at medium
- [ ] P3.5 Rank stability under weight perturbation (seeded)
- [ ] P3.6 Soft output check: numbers appear in tool results (rounding tolerant)
- [ ] P3.7 Golden tests on the four exam questions

## M4 - Context and freshness
- [ ] P4.1 Census ingest: airport to CBSA via the Census Geocoder (cached), population from the API; attribution in the UI
- [ ] P4.2 `demand_context` component and weights back to the full formula
- [ ] P4.3 `ingest --check` (HEAD next On-Time month, T-100 "Latest Available Data") and `meta` freshness fields; `data_status` reports current, behind or unknown
- [ ] P4.4 Build and commit `seed/airport.db`
- [ ] P4.5 (stretch) `ingest --download` for On-Time (direct URLs) and T-100 (form POST), falling back to local files
- [ ] P4.6 (stretch) Extend the window to 2026-07 using P4.5

## M5 - Package and document
- [ ] P5.1 `Dockerfile`, `docker-compose.yml`, entrypoint that seeds the DB volume
- [ ] P5.2 Finalize `README.md` quickstart and local-model option
- [ ] P5.3 `docs/design.md`: scoring methodology, key tradeoffs, where AI is used, limits and assumptions, data validation (from `docs/plan.md`), production path
- [ ] P5.4 Final check: fresh clone, `docker compose up`, run the four questions plus follow-ups
- [ ] P5.5 (stretch) Voice input via `st.audio_input`

## Decision log (for the interview)
| Decision | Why | Trade-off |
|---|---|---|
| LLM never computes numbers; tools return an envelope | Trust and auditability, exam requires deterministic scoring | More code than letting the model reason over tables |
| LangGraph prebuilt ReAct agent on `langchain-core` models | The job post names LangChain-family frameworks; memory by `thread_id` comes free; no custom graph needed for one agent and six tools | A framework dependency; a manual tool loop would be about 60 lines |
| SQLite with pre-aggregated tables and a committed seed DB | Runs offline in seconds, reproducible for the reviewer | Data frozen until `ingest` is rerun |
| Ingest is offline and idempotent, with `--check` | Chat stays fast and stable; freshness is measured, not guessed | No live data in chat |
| One analysis window (trailing 12 months, `min(last_period)`) | Consistent across sources, easy to explain | Newer months wait for the slower source |
| Stale means upstream has a newer month | Honest (publication lag is about 2 months) | Needs the network once to know |
| Provider registry in `src/llm.py`, selected by `.env`; local Ollama model by default, OpenAI ready but commented out | Free and offline for development and for the reviewer's first run; OpenAI is one uncommented key away, no code change | Local models call tools less reliably |
| Methodology in `config/weights.yaml`, infra in `.env` | Reviewable and versioned vs per-deployment and secret | Two config sources, one loader |
| Score is a capacity-pressure proxy, not a terminal-need measure | No gate or terminal-area data in public sources | Stated as a limit in every answer |
| Congestion = delay rate, taxi-out, NAS delay per arriving flight; cancellations not scored | Carrier delay is airline-driven, cancellations are mostly weather | Fewer signals |
| Census kept (10%), built after the vertical slice; FAA NAS Status dropped | Census works with a key; FAA is live status only, no history | Needs an airport-to-CBSA mapping (Geocoder, cached) |
| Replies in the user's language, English default | Global reviewers; Hebrew works too | None |
| Six small tools | Reliable tool choice, also for local models | Less flexible than free-form SQL |
| Scale component (0.15) in the score | Profit depends on volume, not only pressure and growth; stops tiny fast-growing airports from topping the ranking | One more weight to defend (it is in YAML and covered by the stability check) |
| Passenger service = config 1 or 3, peer set = US airports above the departure floor | One explicit definition shared by ingest, scoring and tools | Acceptance values must be re-baselined once |
| On-Time split into origin and destination sides | BTS records delay causes for arrivals; mixing sides would misattribute them | Slightly wider table |
| Agent in `st.cache_resource`, SQLite opened read-only per call | Streamlit reruns the script per message; keeps memory and avoids thread errors | None |
| Agent constructor: `create_agent` from `langchain.agents` (add the `langchain` package in P2.7) | Installed LangGraph 1.2.14 still ships `create_react_agent`, but it warns it is deprecated since V1.0 and removed in V2.0, moved to `langchain.agents` | One more dependency (`langchain`) |

## Demo script
1. Which airports in New England are strong candidates for terminal expansion? (ranking, weights, stability, confidence)
   - Follow-up: why is BDL below PVD? What if growth weighs more?
2. Compare LA and Santa Ana congestion. (agent states LA = LAX, offers BUR/LGB/ONT)
3. What percentage of long-haul flights leave Anchorage? (both numbers, thresholds, cargo explanation)
4. What is the unmet demand at SFO and why? (proxy, peak months, dense routes, delay causes, confidence capped at medium)
