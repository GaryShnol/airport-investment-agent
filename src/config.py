"""Single source of configuration.

- Environment (.env): infrastructure and secrets (LLM provider, keys, paths).
- config/weights.yaml: scoring methodology (weights, thresholds, regions).
- config/sources.yaml: upstream file formats for `ingest --update` (column mapping, form fields, estimates).
Nothing else in the project should read os.environ or hardcode these values.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
_TOLERANCE = 1e-6
# Default upstream URLs (override in .env). A missing On-Time month returns 404 with text/html.
ONTIME_PREZIP_URL = ("https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
                     "{year}_{month}.zip")
T100_PAGE_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45"
HTTP_USER_AGENT = "airport-investment-agent/1.0 (offline data refresh; https://github.com/GaryShnol/airport-investment-agent)"


class ConfigError(ValueError):
    """Raised for missing or invalid configuration, with an actionable message."""


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_base_url: str | None
    openai_api_key: str | None = field(repr=False)
    ollama_base_url: str
    ollama_num_ctx: int | None     # Ollama context window in tokens; None = the server's default
    census_api_key: str | None = field(repr=False)
    db_path: Path
    seed_db_path: Path           # gzipped seed DB written by `src.seed build`
    seed_max_mb: float           # `src.seed build` stops if the compressed seed is larger
    seed_db_url: str | None      # `src.bootstrap` downloads the seed from here when DB_PATH is not usable
    seed_db_sha256: str | None   # expected sha256 of the seed; None = read `<SEED_DB_URL>.sha256`
    bootstrap_check_timeout_s: float  # per request in the `ingest --check` run at startup
    app_port: int                # Streamlit port in the container (docker-compose publishes it)
    raw_data_dir: Path
    weights_path: Path
    sources_path: Path           # config/sources.yaml: upstream formats for `ingest --update`
    log_level: str
    max_input_chars: int
    # Voice input (app only): speech-to-text through the OpenAI audio transcription API.
    stt_model: str               # transcription model, e.g. gpt-4o-mini-transcribe
    stt_max_seconds: float       # longest recording accepted, checked before anything is sent
    stt_base_url: str | None     # a separate transcription endpoint; None = the OpenAI API (or LLM_BASE_URL)
    # Upstream endpoints and HTTP behaviour, used only by `src.ingest` (never at chat time).
    census_api_url: str          # Census Data API root, e.g. https://api.census.gov/data
    census_geocoder_url: str     # Census Geocoder, coordinates -> geographies
    ontime_prezip_url: str       # BTS On-Time monthly ZIP, with {year} and {month} placeholders
    t100_page_url: str           # BTS T-100 Segment download page ("Latest Available Data")
    http_timeout_s: float        # per request
    geocoder_delay_s: float      # pause between Geocoder requests (rate limit)
    # `ingest --update` downloads (one request at a time).
    download_timeout_s: float    # per request (connect and each read); a BTS form answer takes about 15 s
    download_delay_s: float      # pause before every download request after the first
    download_retries: int        # extra attempts after a network error or a 5xx answer (never after a 404)
    http_user_agent: str         # User-Agent sent with every ingest request

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ

        def get(name: str, default: str | None = None) -> str | None:
            value = env.get(name, default)
            return value.strip() if isinstance(value, str) and value.strip() else default

        try:
            temperature = float(get("LLM_TEMPERATURE", "0"))
        except ValueError as exc:
            raise ConfigError("LLM_TEMPERATURE must be a number") from exc
        try:
            max_input_chars = int(get("MAX_INPUT_CHARS", "2000"))
        except ValueError as exc:
            raise ConfigError("MAX_INPUT_CHARS must be a whole number") from exc
        if max_input_chars <= 0:
            raise ConfigError("MAX_INPUT_CHARS must be positive")
        try:
            http_timeout_s = float(get("HTTP_TIMEOUT_S", "30"))
            geocoder_delay_s = float(get("GEOCODER_DELAY_S", "0.25"))
        except ValueError as exc:
            raise ConfigError("HTTP_TIMEOUT_S and GEOCODER_DELAY_S must be numbers of seconds") from exc
        if http_timeout_s <= 0 or geocoder_delay_s < 0:
            raise ConfigError("HTTP_TIMEOUT_S must be positive and GEOCODER_DELAY_S not negative")
        try:
            seed_max_mb = float(get("SEED_MAX_MB", "50"))
        except ValueError as exc:
            raise ConfigError("SEED_MAX_MB must be a number of megabytes") from exc
        if seed_max_mb <= 0:
            raise ConfigError("SEED_MAX_MB must be positive")
        seed_db_sha256 = get("SEED_DB_SHA256")
        if seed_db_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", seed_db_sha256):
            raise ConfigError("SEED_DB_SHA256 must be 64 hex characters (a sha256), or unset")
        try:
            bootstrap_check_timeout_s = float(get("BOOTSTRAP_CHECK_TIMEOUT_S", "5"))
            app_port = int(get("APP_PORT", "8501"))
        except ValueError as exc:
            raise ConfigError("BOOTSTRAP_CHECK_TIMEOUT_S must be seconds and APP_PORT a whole number") from exc
        if bootstrap_check_timeout_s <= 0 or not 1 <= app_port <= 65535:
            raise ConfigError("BOOTSTRAP_CHECK_TIMEOUT_S must be positive and APP_PORT between 1 and 65535")
        try:
            stt_max_seconds = float(get("STT_MAX_SECONDS", "60"))
        except ValueError as exc:
            raise ConfigError("STT_MAX_SECONDS must be a number of seconds") from exc
        if stt_max_seconds <= 0:
            raise ConfigError("STT_MAX_SECONDS must be positive")
        try:
            download_timeout_s = float(get("DOWNLOAD_TIMEOUT_S", "120"))
            download_delay_s = float(get("DOWNLOAD_DELAY_S", "2"))
            download_retries = int(get("DOWNLOAD_RETRIES", "2"))
        except ValueError as exc:
            raise ConfigError("DOWNLOAD_TIMEOUT_S and DOWNLOAD_DELAY_S must be seconds and DOWNLOAD_RETRIES "
                              "a whole number") from exc
        if download_timeout_s <= 0 or download_delay_s < 0 or download_retries < 0:
            raise ConfigError("DOWNLOAD_TIMEOUT_S must be positive, DOWNLOAD_DELAY_S and DOWNLOAD_RETRIES not negative")
        num_ctx = get("OLLAMA_NUM_CTX")
        try:
            num_ctx = int(num_ctx) if num_ctx is not None else None
        except ValueError as exc:
            raise ConfigError("OLLAMA_NUM_CTX must be a whole number of tokens") from exc
        if num_ctx is not None and num_ctx <= 0:
            raise ConfigError("OLLAMA_NUM_CTX must be positive")

        return cls(
            llm_provider=get("LLM_PROVIDER", "ollama").lower(),
            llm_model=get("LLM_MODEL", "llama3.1"),
            llm_temperature=temperature,
            llm_base_url=get("LLM_BASE_URL"),
            openai_api_key=get("OPENAI_API_KEY"),
            ollama_base_url=get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_num_ctx=num_ctx,
            census_api_key=get("CENSUS_API_KEY"),
            db_path=_resolve(get("DB_PATH", "data/airport.db")),
            seed_db_path=_resolve(get("SEED_DB_PATH", "data/seed/airport.db.gz")),
            seed_max_mb=seed_max_mb,
            seed_db_url=get("SEED_DB_URL"),
            seed_db_sha256=seed_db_sha256.lower() if seed_db_sha256 else None,
            bootstrap_check_timeout_s=bootstrap_check_timeout_s,
            app_port=app_port,
            raw_data_dir=_resolve(get("RAW_DATA_DIR", "data/raw")),
            weights_path=_resolve(get("WEIGHTS_PATH", "config/weights.yaml")),
            sources_path=_resolve(get("SOURCES_PATH", "config/sources.yaml")),
            log_level=get("LOG_LEVEL", "INFO").upper(),
            max_input_chars=max_input_chars,
            stt_model=get("STT_MODEL", "gpt-4o-mini-transcribe"),
            stt_max_seconds=stt_max_seconds,
            stt_base_url=get("STT_BASE_URL"),
            census_api_url=get("CENSUS_API_URL", "https://api.census.gov/data").rstrip("/"),
            census_geocoder_url=get("CENSUS_GEOCODER_URL",
                                    "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"),
            ontime_prezip_url=get("ONTIME_PREZIP_URL", ONTIME_PREZIP_URL),
            t100_page_url=get("T100_PAGE_URL", T100_PAGE_URL),
            http_timeout_s=http_timeout_s,
            geocoder_delay_s=geocoder_delay_s,
            download_timeout_s=download_timeout_s,
            download_delay_s=download_delay_s,
            download_retries=download_retries,
            http_user_agent=get("HTTP_USER_AGENT", HTTP_USER_AGENT),
        )


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, float]
    congestion_weights: dict[str, float]
    long_haul_miles: int
    long_haul_min_share: float
    long_haul_sensitivity_miles: tuple[int, ...]
    min_passengers_per_year: int
    min_ontime_flights: int    # On-Time departures in the window needed to score congestion
    target_load_factor: float
    target_sensitivity: tuple[float, ...]   # spill proxy also at these target load factors (range)
    peak_months: int                        # explain_demand_gap: months listed as peaks
    top_routes: int                         # explain_demand_gap: routes listed
    sensitivity_runs: int
    sensitivity_jitter: float
    sensitivity_seed: int
    sensitivity_top_n: int              # rank stability: share of runs each airport spends in this top N
    sensitivity_stable_share: float     # the #1 airport is "stable" if first in at least this share of runs
    comparison_confidence_level: float  # compare_airports: margin of error for On-Time figures (0.95)
    comparison_size_ratio_note: float   # compare_airports: note when departures differ by more than this ratio
    commentary_restricted_tools: tuple[str, ...]  # commentary limited to caveats and next steps
    census_dataset: str          # Census Data API dataset path, e.g. acs/acs5
    census_variable: str         # population variable, e.g. B01003_001E
    census_vintage: int          # population vintage scored (demand_context)
    census_base_vintage: int     # earlier vintage, base for metro population growth
    geocoder_benchmark: str      # Census Geocoder benchmark
    geocoder_vintage: str        # Census Geocoder vintage (CBSA delineation) for airport -> CBSA
    regions: dict[str, tuple[str, ...]]
    max_ranked: int            # longest ranking a tool returns (keeps tool results small)
    non_us_airport_note: str   # scope.non_us_airport: caveat for airports outside the US (partial T-100)
    # name -> {"default": IATA, "candidates": (IATA, ...)}; e.g. LA -> LAX by default
    metro_aliases: dict[str, dict] = field(default_factory=dict)


_WEIGHT_KEYS = {"utilization", "congestion", "scale", "growth", "long_haul", "demand_context"}
_CONGESTION_KEYS = {"dep_delay_rate", "taxi_out", "nas_delay_per_arrival"}


def _check_weights(name: str, weights: Mapping[str, float], expected: set[str]) -> dict[str, float]:
    if set(weights) != expected:
        raise ConfigError(f"{name} must have exactly these keys: {sorted(expected)}")
    if any(v < 0 for v in weights.values()):
        raise ConfigError(f"{name} must not contain negative values")
    total = sum(weights.values())
    if abs(total - 1.0) > _TOLERANCE:
        raise ConfigError(f"{name} must sum to 1.0 (got {total:.4f})")
    return {k: float(v) for k, v in weights.items()}


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive whole number (got {value!r})")
    return value


def _share(name: str, value) -> float:
    """A number strictly between 0 and 1 (a probability or a load factor)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ConfigError(f"{name} must be a number between 0 and 1 (got {value!r})")
    return float(value)


def _ratio(name: str, value) -> float:
    """A ratio greater than 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 1:
        raise ConfigError(f"{name} must be a number greater than 1 (got {value!r})")
    return float(value)


def _text(name: str, value) -> str:
    """A non-empty string (wording shown to the user)."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string (got {value!r})")
    return value.strip()


def _base_vintage(census: Mapping) -> int:
    """The growth base must be an earlier vintage than the one scored."""
    base = _positive_int("census.base_vintage", census["base_vintage"])
    if base >= census["vintage"]:
        raise ConfigError(f"census.base_vintage ({base}) must be earlier than census.vintage ({census['vintage']})")
    return base


def load_scoring(path: Path) -> ScoringConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Scoring config not found: {path}") from exc

    try:
        return ScoringConfig(
            weights=_check_weights("weights", raw["weights"], _WEIGHT_KEYS),
            congestion_weights=_check_weights("congestion", raw["congestion"], _CONGESTION_KEYS),
            long_haul_miles=int(raw["long_haul_miles"]),
            long_haul_min_share=float(raw["long_haul_min_share"]),
            long_haul_sensitivity_miles=tuple(int(x) for x in raw["long_haul_sensitivity_miles"]),
            min_passengers_per_year=int(raw["min_passengers_per_year"]),
            min_ontime_flights=_positive_int("min_ontime_flights", raw["min_ontime_flights"]),
            target_load_factor=_share("unmet_demand.target_load_factor", raw["unmet_demand"]["target_load_factor"]),
            target_sensitivity=tuple(_share("unmet_demand.target_sensitivity", x)
                                     for x in raw["unmet_demand"]["target_sensitivity"]),
            peak_months=_positive_int("unmet_demand.peak_months", raw["unmet_demand"]["peak_months"]),
            top_routes=_positive_int("unmet_demand.top_routes", raw["unmet_demand"]["top_routes"]),
            sensitivity_runs=int(raw["sensitivity"]["runs"]),
            sensitivity_jitter=float(raw["sensitivity"]["weight_jitter"]),
            sensitivity_seed=int(raw["sensitivity"]["seed"]),
            sensitivity_top_n=_positive_int("sensitivity.top_n", raw["sensitivity"]["top_n"]),
            sensitivity_stable_share=_share("sensitivity.stable_share", raw["sensitivity"]["stable_share"]),
            comparison_confidence_level=_share("comparison.confidence_level", raw["comparison"]["confidence_level"]),
            comparison_size_ratio_note=_ratio("comparison.size_ratio_note", raw["comparison"]["size_ratio_note"]),
            commentary_restricted_tools=tuple(str(t) for t in raw["commentary"]["restricted_tools"] or ()),
            census_dataset=str(raw["census"]["dataset"]),
            census_variable=str(raw["census"]["variable"]),
            census_vintage=_positive_int("census.vintage", raw["census"]["vintage"]),
            census_base_vintage=_base_vintage(raw["census"]),
            geocoder_benchmark=str(raw["census"]["geocoder_benchmark"]),
            geocoder_vintage=str(raw["census"]["geocoder_vintage"]),
            regions={k: tuple(v) for k, v in raw["regions"].items()},
            max_ranked=_positive_int("max_ranked", raw["max_ranked"]),
            non_us_airport_note=_text("scope.non_us_airport", raw["scope"]["non_us_airport"]),
            metro_aliases={k.upper(): {"default": v["default"], "candidates": tuple(v["candidates"])}
                           for k, v in (raw.get("metro_aliases") or {}).items()},
        )
    except KeyError as exc:
        raise ConfigError(f"Missing key in {path}: {exc}") from exc


@dataclass(frozen=True)
class SourcesConfig:
    """Upstream file formats for `ingest --update` (config/sources.yaml)."""
    ontime_prezip_columns: dict[str, str]   # PREZIP column -> our column name
    ontime_expected_rows: dict[str, int]    # reference BTS flight counts by year or YYYY-MM
    ontime_month_mb: float
    t100_form: dict[str, str]               # fixed form values (geography, zip, button)
    t100_fields: tuple[str, ...]            # column checkboxes ticked in the form
    t100_zip_csv: str                       # data file inside the T-100 ZIP
    t100_month_mb: float
    assumed_kb_per_s: float                 # throughput for the plan's time estimate
    max_download_mb: float                  # larger downloads are refused


def _positive_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{name} must be a positive number (got {value!r})")
    return float(value)


def _period_key(name: str, key) -> str:
    key = str(key)
    if not re.fullmatch(r"\d{4}(-(0[1-9]|1[0-2]))?", key):
        raise ConfigError(f"{name}: {key!r} must be a year (2025) or a month (2025-01)")
    return key


def load_sources(path: Path) -> SourcesConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Sources config not found: {path}") from exc
    try:
        columns = {str(k): str(v) for k, v in raw["ontime"]["prezip_columns"].items()}
        if len(set(columns.values())) != len(columns):
            raise ConfigError("ontime.prezip_columns must map each PREZIP column to a different name")
        return SourcesConfig(
            ontime_prezip_columns=columns,
            ontime_expected_rows={_period_key("ontime.expected_rows", k): _positive_int("ontime.expected_rows", v)
                                  for k, v in (raw["ontime"].get("expected_rows") or {}).items()},
            ontime_month_mb=_positive_number("ontime.month_mb", raw["ontime"]["month_mb"]),
            t100_form={str(k): str(v) for k, v in raw["t100"]["form"].items()},
            t100_fields=tuple(str(f) for f in raw["t100"]["fields"]),
            t100_zip_csv=str(raw["t100"]["zip_csv"]),
            t100_month_mb=_positive_number("t100.month_mb", raw["t100"]["month_mb"]),
            assumed_kb_per_s=_positive_number("download.assumed_kb_per_s", raw["download"]["assumed_kb_per_s"]),
            max_download_mb=_positive_number("download.max_mb", raw["download"]["max_mb"]),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Missing or invalid key in {path}: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load .env once (real environment variables win) and return the settings."""
    load_dotenv(ROOT / ".env", override=False)
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_scoring() -> ScoringConfig:
    return load_scoring(get_settings().weights_path)
