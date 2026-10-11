# Airport Investment Intelligence Agent

A conversational agent that helps an analyst find US airports where renovation or terminal expansion is most likely to pay off, judged by passenger and flight capacity pressure. The model understands the question and picks tools; every number, ranking and verdict comes from SQL and a deterministic scoring module. Built as a 24-hour take-home for a Deloitte Digital Forward Deployed Engineer role.

Design doc: [docs/design.md](docs/design.md) (methodology, tradeoffs, the four answers). Build plan and definitions: [docs/plan.md](docs/plan.md). Task board and decision log: [TASKS.md](TASKS.md).

![Compare LA and Santa Ana: the compare_airports data card with the per-metric verdicts](docs/img/lax-sna-verdict.jpg)

<details>
<summary><strong>More screenshots and a short demo</strong></summary>

Demo: the LAX vs SNA comparison from question to commentary.

![Demo: asking the app to compare LA and Santa Ana congestion](docs/img/demo-lax-vs-sna.gif)

![Start screen: data period, freshness, model and the four sample questions](docs/img/home.jpg)
![Compare LA and Santa Ana: the side-by-side table with margins of error and a verdict per metric](docs/img/lax-sna-table.jpg)
![Compare LA and Santa Ana: the labeled model commentary with one commentary-check note](docs/img/lax-sna-commentary.jpg)
![New England ranking: the rank_airports data card with the weights used and rank stability](docs/img/new-england-top.jpg)
![New England ranking: the assumptions and the labeled model commentary with one commentary-check note](docs/img/new-england-commentary.jpg)

</details>

## What it answers
1. Which airports in New England are strong candidates for terminal expansion? (ranking with the weights used, drivers and rank stability)
2. Compare LA (LAX) and Santa Ana (SNA) congestion. (side by side, with margins of error and a verdict per measure)
3. What share of flights out of Anchorage is long haul? (all flights including cargo, and passenger service only, with a distance sensitivity)
4. What is the unmet flight demand at SFO and why? (a labeled proxy: unmet demand is not observable in public data)

Follow-up questions work in the same chat ("what if growth weighs more?", "which of these has the highest load factor?"). Each answer shows the data card rendered from the tool results first, then the model's commentary, labeled as such, and states the data period, assumptions, confidence, the weights used and whether newer data exists upstream.

## How it works
- **The LLM never computes numbers.** It chooses tools and phrases results. Metrics come from SQL (`src/store.py`, `src/tools.py`) and `src/scoring.py`; the card on screen is rendered by code (`src/render.py`).
- **Guardrails in code, not in the prompt.** Required facts missing from the commentary are appended (`ensure_facts`); commentary sentences with figures that are not in the tool results are removed (`src/figure_check.py`); rankings and comparisons get caveat-only commentary.
- **Offline data.** An idempotent `ingest` command loads BTS and Census data into a local SQLite DB. Nothing is fetched at chat time.
- **No hardcoded values.** Infrastructure and secrets come from `.env` ([.env.example](.env.example)); scoring methodology lives in [config/weights.yaml](config/weights.yaml). Both are read only through `src/config.py`.
- **Swappable model.** The model is a service chosen in `.env`: OpenAI for the quickstart, a local Ollama model or any OpenAI-compatible server for development (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`), with no code change.

## Architecture

```mermaid
classDiagram
    direction LR

    class ChatUI {
        <<Streamlit>>
        +send(message, thread_id)
        +data_banner()
    }

    class Settings {
        <<config>>
        +from_env() Settings
        +load_scoring() ScoringConfig
    }

    class LLMFactory {
        <<factory>>
        +get_chat_model(settings) BaseChatModel
    }

    class OpenAIBackend {
        <<openai or compatible>>
    }

    class OllamaBackend {
        <<local model>>
    }

    class ReActAgent {
        -llm
        -memory
        +run(message, thread_id) Answer
    }

    class Tools {
        <<6 tools>>
        +data_status()
        +resolve_airport(text)
        +airport_metrics(airport)
        +compare_airports(a, b)
        +rank_airports(region)
        +explain_demand_gap(airport)
    }

    class ScoringEngine {
        <<deterministic>>
        +score(metrics, weights) Score
        +rank(region) Ranking
        +rank_stability(region) Stability
    }

    class DataStore {
        <<SQLite>>
        +metrics(airport, window)
        +delays(airport, window)
        +airports_in(region)
        +freshness() Report
    }

    class Ingest {
        <<CLI offline>>
        +load_files()
        +check() Report
        +download_missing()
    }

    ChatUI --> ReActAgent
    ReActAgent --> LLMFactory
    LLMFactory --> OpenAIBackend
    LLMFactory --> OllamaBackend
    LLMFactory ..> Settings : reads .env
    ReActAgent --> Tools : tool calls
    Tools --> DataStore : local SQL
    Tools --> ScoringEngine
    ScoringEngine --> DataStore
    ScoringEngine ..> Settings : weights.yaml
    Ingest --> DataStore : writes aggregates and meta
    Ingest ..> Settings : keys and paths
```

## Request flow

```mermaid
graph TD
    A["User query + thread_id"] --> B["Light guard in code: empty, length limit"]
    B --> C["ReAct agent (LLM from LLM_PROVIDER + system prompt: scope rules, assumptions, clarify when ambiguous, reply in user language)"]

    C --> D{"Next step?"}
    D -- "Ambiguous or unknown entity" --> E["Ask clarifying question"]
    D -- "Out of scope" --> F["Polite out-of-scope reply"]
    D -- "Need data" --> G["Call tool"]

    G --> H["Tools (6): data_status, resolve_airport, airport_metrics, compare_airports, rank_airports, explain_demand_gap"]
    H --> I["SQLite (local) and ScoringEngine (deterministic, weights from config/weights.yaml)"]
    I --> J["Tool result envelope: data, data_period, freshness, assumptions, confidence"]
    J --> C

    D -- "Enough data" --> K["LLM writes answer using only envelope numbers"]
    K --> L["Figure check in code: sentences with figures not in the tool results are removed; ensure_facts adds missing required facts"]
    L --> N["Final response: answer, period, assumptions, confidence, weights, freshness warning if stale"]

    subgraph OFFLINE ["Offline refresh (CLI, never during chat)"]
        O["ingest --check"] --> P{"Upstream has newer month than meta.last_period?"}
        P -- "Yes" --> Q["Report missing months, store upstream_latest in meta"]
        P -- "No" --> R["Mark data current"]
        Q --> S["ingest: load new files, aggregate, update SQLite and last_period"]
    end

    S -.-> I
    Q -.-> J
```

## Quickstart
The data comes ready-made (see First start); you only provide the language model.

### Docker with OpenAI (recommended)
Set `OPENAI_API_KEY` in `.env`, then `docker compose up`:
```bash
cp .env.example .env
# in .env, fill in the one empty line:  OPENAI_API_KEY=sk-...
docker compose up --build
```
Open http://localhost:8501 (or the `APP_PORT` you set). `.env.example` already has `LLM_PROVIDER=openai` and `LLM_MODEL=gpt-4o-mini`; change `LLM_MODEL` for another OpenAI model. If the key is missing or wrong, or the model server cannot be reached, the app shows one line naming the setting to fix (`OPENAI_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_PROVIDER`).

One question from the command line, inside the running container:
```bash
docker compose exec app python -m src.agent "Compare LA and Santa Ana airport congestion levels."
```

### Local (uv)
```bash
cp .env.example .env              # same LLM settings as above
uv sync
uv run python -m src.bootstrap    # downloads the seed DB to data/airport.db (skipped if a valid DB is there)
uv run streamlit run app.py
```
The CLI equivalent is `uv run python -m src.agent "<question>"`.

<details>
<summary><strong>Optional: a local model (Ollama)</strong></summary>

For development without an API key. The model runs in Ollama on the host, not in the container (the compose file has no Ollama service), and models come from the Ollama registry, not from Docker Hub or the image:
1. Install Ollama on the host and pull a model with tool calling, for example `ollama pull llama3.1`.
2. In `.env`: `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1`, `LLM_BASE_URL=http://host.docker.internal:11434` (for a local `uv run`, `http://localhost:11434`), and remove or leave empty `OPENAI_API_KEY`. The commented lines in `.env.example` show these settings; `OLLAMA_NUM_CTX=16384` keeps follow-ups inside the context window.
3. Docker Desktop resolves `host.docker.internal`; on Linux the `extra_hosts` line in `docker-compose.yml` does, and Ollama must listen beyond localhost (`OLLAMA_HOST=0.0.0.0`).

On CPU only, local models are slow: one to four minutes per answer with 8-20B models on a laptop in our runs, and their commentary is less reliable (the data card is not affected). Any OpenAI-compatible server also works: `LLM_PROVIDER=openai` with `LLM_BASE_URL=http://host.docker.internal:11434/v1`.

</details>

### Voice
Next to the chat input there is a microphone ("Or ask by voice"). Record a question, stop, and the app transcribes it with the OpenAI audio transcription API and shows the transcript in an editable box. Nothing is sent yet: fix a misheard word and press **Send**, or press **Discard** to record again. A sent transcript appears in the chat as "You said: ..." and is answered exactly like a typed question (same agent, same `MAX_INPUT_CHARS` limit).
- Needs: `LLM_PROVIDER=openai` with `OPENAI_API_KEY` (the same key and `LLM_BASE_URL` as the chat), or `STT_BASE_URL` pointing to another OpenAI-compatible transcription server. Otherwise the microphone is hidden and a one-line caption says why.
- Settings: `STT_MODEL` (default `gpt-4o-mini-transcribe`), `STT_MAX_SECONDS` (default 60; longer recordings are refused before anything is sent), `STT_BASE_URL` (optional), `STT_LANGUAGE` (default `en`, so an accent is not transcribed as another language; set it empty to auto-detect), `STT_PROMPT` (a one-sentence vocabulary hint with airport codes and domain words, on by default; set it empty to turn it off). For accented speech, `STT_MODEL=gpt-4o-transcribe` is a more accurate (and pricier) option. See `.env.example`.
- No speech: when the transcript is empty or only repeats the vocabulary hint (what a model can return for silence), the app says "Didn't catch that - please try again." and sends nothing.
- Errors (no key, server unreachable, rejected key or model, empty or too long recording) are one line naming the setting to fix.
- Privacy: audio is sent to the OpenAI API (or `STT_BASE_URL`) only when you record a question. It is held in memory for that one request and is never written to disk or kept in the session; only the transcript stays in the chat.

## First start
The data is not in git or in the image. `src.bootstrap` (the container entrypoint, or the command above) checks `DB_PATH`; if the DB is missing or invalid it downloads the seed DB from `SEED_DB_URL` (a gzipped SQLite file published as a GitHub Release asset), verifies its sha256 and installs it atomically. Later starts reuse it (in Docker it lives in the `data` volume). It then runs `ingest --check` with a short timeout so the sidebar can say whether newer data exists upstream; offline, freshness is "unknown" and the app still starts. If no data can be obtained it stops with three options: set `SEED_DB_URL`, mount an existing `airport.db` at `/app/data/airport.db`, or build the DB with `ingest` from the raw files.

## Data sources
| Source | Used for |
|---|---|
| [BTS T-100 Segment, All Carriers](https://www.transtats.bts.gov/) | Departures, seats, passengers, distance, aircraft configuration (passenger vs cargo); the analysis window and the 12 months before it (growth) |
| [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/) | Departure delays, taxi-out, cancellations, delay causes; domestic flights of the large reporting carriers only |
| [US Census Bureau Data API](https://www.census.gov/data/developers.html), ACS 5-year | Population of metro and micro areas (CBSAs); airport to CBSA via the Census Geocoder; vintages in `config/weights.yaml` (`census`) |
| [OurAirports](https://ourairports.com/data/) | Airport names, states, coordinates (public domain) |

One analysis window serves every question: the trailing 12 complete months ending at the latest month loaded for both BTS sources. The window is printed on every answer and in the sidebar; for the shipped seed it is 2025-01..2025-12.

This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau. BTS data is public (US Department of Transportation, Bureau of Transportation Statistics).

## Scoring
Each component is a percentile among US airports in the peer set (a minimum number of passenger-service passengers in the window), so a score of 0.75 means "on average at the 75th percentile nationally". Airports are then ranked inside the requested region. The score is a capacity-pressure proxy, not a measured terminal need: public data has no gate or terminal-area figures. Weights, thresholds, the peer-set floor and the regions are in [config/weights.yaml](config/weights.yaml); definitions are in [docs/plan.md](docs/plan.md) (Definitions, Scoring); the reasoning is in [docs/design.md](docs/design.md).

| Component | Measures | Source |
|---|---|---|
| Utilization | Seat load factor on passenger service (passengers / seats): demand pressure, not terminal use | T-100 |
| Congestion | Departure-delay rate, mean taxi-out, NAS-delay minutes per arriving flight (cancellations shown, not scored) | On-Time |
| Scale | Annual passengers: keeps the ranking about volume, not only pressure | T-100 |
| Growth | Passenger growth, window vs the previous 12 months | T-100 |
| Long haul | Share of passenger-service departures at or above the long-haul distance threshold | T-100 |
| Demand context | Population growth of the metro or micro area (CBSA) the airport lies in | Census |

A missing component (no On-Time coverage, no CBSA) renormalizes the remaining weights for that airport and lowers its confidence. Every ranking also reports rank stability: how often each airport stays near the top when the weights are perturbed at random (seeded, so the same question gives the same card). `rank_airports` accepts scenario weights ("what if growth weighs more?"), applied by the scoring code, not by the model.

## Testing and evaluation
- `uv run pytest`: unit and golden tests, no model and no network (CI runs them and a `docker build` on every push to `main` and every pull request). The golden tests (`tests/test_golden.py`) put the four exam questions and the edge cases (unknown airport, unsupported region, scenario weights, ambiguity) through the tools on a small fixture DB; agent tests use a scripted fake model.
- `uv run ruff check`: lint.
- `uv run python scripts/eval.py`: the eval harness and model-comparison tool. It asks the questions in [eval/questions.yaml](eval/questions.yaml) (the four exam questions, follow-ups in one conversation, single airports, an unknown airport, an unsupported region, an out-of-scope question and a prompt-injection attempt) to the real agent with the model set in `.env`, and checks per question: tools called (and their arguments) vs expected, must-have facts, forbidden text in what the user sees (the rendered cards and the commentary after the figure check), the commentary figure check, an empty or cut-off reply, and the time taken; every failed question gets a reason. Results are appended to `eval/results/<timestamp>-<model>.jsonl` (not committed) and a markdown table is printed. `--only q3_anchorage_long_haul,q4_sfo_unmet_demand` runs some questions; `--report <file>.jsonl` re-prints a saved run. Not in CI: it needs a model and the real DB.

<details>
<summary><strong>Updating the data</strong></summary>

Ingest runs offline, never during a chat, and is idempotent (a month already loaded is skipped; a changed On-Time file reloads its month only).
```bash
uv run python -m src.ingest           # load the files in RAW_DATA_DIR (default data/raw) into DB_PATH
uv run python -m src.ingest --check   # ask upstream for its latest period per source; no downloads
uv run python -m src.ingest --update  # download and load the newer BTS months (opt-in, below)
```
Raw files: T-100 exports as `data/raw/t100_segment/t100*.csv`, On-Time monthly TranStats downloads as `data/raw/ontime/**/ontime_*.zip`, OurAirports as `data/raw/airports.csv`; Census needs `CENSUS_API_KEY` in `.env`.

### Opt-in update from BTS (`--update`)
The shipped seed holds 2025-01..2025-12 and stays that way: every figure in the docs and the demo comes from that window. `--update` is for a user who wants newer months in their own DB.
```bash
uv run python -m src.ingest --update --dry-run           # the plan only: months, rough size and time
uv run python -m src.ingest --update --through 2026-01   # load up to a month (asks before downloading)
uv run python -m src.ingest --update --yes               # up to the latest month upstream, no prompt
```
What it does, per source (On-Time and T-100): the `--check` logic finds the latest month upstream; every month after the loaded one, up to `--through` (default: that latest month), is downloaded into a temporary folder outside the project, validated and loaded through the same aggregation as the raw files, one month at a time. On-Time comes from the BTS monthly PREZIP file (its CamelCase columns are renamed through the mapping in [config/sources.yaml](config/sources.yaml)); T-100 comes from the BTS download form (one GET for the form state, one POST for the month). A download that is not a ZIP (BTS answers errors with an HTML page), lacks a needed column or holds another month is refused. Each month is one transaction: the first failure stops the run, months loaded before it stay, and the next run continues from there. A month already loaded is never reloaded, so later BTS revisions of a loaded month are not picked up. Without a terminal, `--yes` is required. Afterwards it prints the loaded periods, the new analysis window and the freshness line. The chat itself never downloads anything.

Size and time: one On-Time month is about 30 MB (several minutes: BTS served about 100 KB/s when this was built), one T-100 month about 1 MB (about 15 s). Catching up from 2025-12 to 2026-07 is about 200 MB and 25-40 minutes. Timeouts, the pause between requests, retries and the User-Agent are `DOWNLOAD_*` and `HTTP_USER_AGENT` in `.env`; file formats and estimates are in `config/sources.yaml`.

In Docker the DB lives in the `data` volume, so run the update inside the running container (or in a one-off container on the same volume) and the app uses the new window on the next question:
```bash
docker compose exec app python -m src.ingest --update --yes
docker compose run --rm --entrypoint python app -m src.ingest --update --yes   # app not running
```
To return to the shipped data, remove the volume (`docker compose down -v`); the next start downloads the seed again.

What changes when the window moves: every answer (window, growth base, percentiles, rankings, confidence) is computed from the new 12 months, so the figures differ from those in [docs/design.md](docs/design.md), the demo script and the seed; the window is printed on every answer. Growth needs the 12 months before the window; the seed holds T-100 from 2024-01 and `--update` adds every month in between, so growth stays complete.

Freshness, per source and overall, from the last `--check` (stored in the DB, shown in the sidebar and in every answer):
- **fresh**: upstream has nothing newer than the loaded period.
- **stale (upstream has YYYY-MM)**: a newer month exists; answers still use the loaded data and say so. The shipped seed is stale on purpose: it holds a complete calendar year, and BTS has published later months since.
- **unknown**: never checked, or the check could not reach upstream.

</details>

## Known limits
- The score is a capacity-pressure proxy: load factor and delays show where demand presses, not how much terminal space is missing.
- "Unmet demand" is a labeled proxy (passengers above a target load level per route and month, with a range), never an observed value: there is no booking or spill data.
- On-Time covers domestic flights of the large reporting carriers only; airports with few reported flights are scored without congestion (lower confidence). On-Time figures carry margins of error; T-100 figures are full counts.
- Cargo-heavy airports such as ANC need passenger-service and all-flights figures side by side; the tools always give both.
- Percentiles are relative to the peer set, and the passenger floor that defines it is a policy line (airports just under it are listed as excluded, with the reason).
- The shipped seed stops at 2025-12. `ingest --update` loads newer BTS months on demand, but a loaded month is never reloaded (BTS revisions are missed), and it depends on the BTS download form, which can change without notice (manual files in `data/raw` are the fallback).
- The chat UI is in English; answers follow the language of the question.
- Voice input was tested with a live microphone in a browser; transcription accuracy depends on accent and audio quality (the transcript is shown as "You said: ..." before the answer). A silent recording is caught only when the model returns no text.

<details>
<summary><strong>Maintainers: publishing a new seed DB</strong></summary>

The seed DB is a GitHub Release asset, not a file in git. Each data version gets its own release tag, so an old URL keeps serving the data it was published with.
1. Refresh the data: `uv run python -m src.ingest` (raw files in `data/raw`), then `uv run python -m src.ingest --check`.
2. Rebuild the seed: `uv run python -m src.seed build` writes `data/seed/airport.db.gz` and `airport.db.gz.sha256` and prints the sizes (it stops above `SEED_MAX_MB`).
3. Publish a new release with both files (new tag, e.g. `data-2026-07`):
   ```bash
   gh release create data-2026-07 data/seed/airport.db.gz data/seed/airport.db.gz.sha256 \
     --title "Seed data 2026-07" --notes "Aggregated SQLite seed DB, window ending 2026-07. sha256 in airport.db.gz.sha256."
   ```
4. Point `SEED_DB_URL` in `.env.example` to `https://github.com/<owner>/<repo>/releases/download/<tag>/airport.db.gz` and commit. Existing volumes keep their DB until it is invalid (bump `store.SCHEMA_VERSION` when the schema changes) or the volume is removed (`docker compose down -v`).

</details>

## Project layout
```
app.py      Streamlit chat (data from tools first, model commentary second)
Dockerfile, docker-compose.yml, docker/   container image, one-service compose, entrypoint
src/        config, LLM provider registry, ingest and upstream check, Census, store, scoring,
            tools, render, figure check, agent (also the CLI), voice input, seed build, bootstrap
config/     scoring methodology (weights.yaml), upstream file formats for --update (sources.yaml)
docs/       design.md (design doc), plan.md (build plan and definitions)
tests/      unit and golden tests (CI)
eval/       eval questions; results/ (not committed) from scripts/eval.py
scripts/    eval.py (eval harness, model comparison)
data/       raw downloads, airport.db, seed/ (built seed DB); not committed
```
