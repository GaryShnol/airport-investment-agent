"""Single source of configuration.

- Environment (.env): infrastructure and secrets (LLM provider, keys, paths).
- config/weights.yaml: scoring methodology (weights, thresholds, regions).
Nothing else in the project should read os.environ or hardcode these values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
_TOLERANCE = 1e-6


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
    seed_db_path: Path
    raw_data_dir: Path
    weights_path: Path
    log_level: str
    max_input_chars: int

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
            seed_db_path=_resolve(get("SEED_DB_PATH", "seed/airport.db")),
            raw_data_dir=_resolve(get("RAW_DATA_DIR", "data/raw")),
            weights_path=_resolve(get("WEIGHTS_PATH", "config/weights.yaml")),
            log_level=get("LOG_LEVEL", "INFO").upper(),
            max_input_chars=max_input_chars,
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
    sensitivity_runs: int
    sensitivity_jitter: float
    sensitivity_seed: int
    regions: dict[str, tuple[str, ...]]
    max_ranked: int            # longest ranking a tool returns (keeps tool results small)
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
            target_load_factor=float(raw["unmet_demand"]["target_load_factor"]),
            sensitivity_runs=int(raw["sensitivity"]["runs"]),
            sensitivity_jitter=float(raw["sensitivity"]["weight_jitter"]),
            sensitivity_seed=int(raw["sensitivity"]["seed"]),
            regions={k: tuple(v) for k, v in raw["regions"].items()},
            max_ranked=_positive_int("max_ranked", raw["max_ranked"]),
            metro_aliases={k.upper(): {"default": v["default"], "candidates": tuple(v["candidates"])}
                           for k, v in (raw.get("metro_aliases") or {}).items()},
        )
    except KeyError as exc:
        raise ConfigError(f"Missing key in {path}: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load .env once (real environment variables win) and return the settings."""
    load_dotenv(ROOT / ".env", override=False)
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_scoring() -> ScoringConfig:
    return load_scoring(get_settings().weights_path)
