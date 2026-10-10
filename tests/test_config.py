import pytest

from src.config import ConfigError, Settings, load_scoring
from src.llm import PROVIDERS, get_chat_model


def test_defaults_and_overrides():
    s = Settings.from_env({})
    assert s.llm_provider == "ollama"
    assert s.db_path.name == "airport.db"

    s = Settings.from_env({"LLM_PROVIDER": "OLLAMA", "LLM_MODEL": "llama3.1", "LLM_TEMPERATURE": "0.2"})
    assert (s.llm_provider, s.llm_model, s.llm_temperature) == ("ollama", "llama3.1", 0.2)


def test_secrets_not_in_repr():
    s = Settings.from_env({"OPENAI_API_KEY": "sk-secret", "CENSUS_API_KEY": "census-secret"})
    assert "sk-secret" not in repr(s) and "census-secret" not in repr(s)


def test_bad_temperature():
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_TEMPERATURE": "hot"})


def test_unknown_provider():
    with pytest.raises(ConfigError, match="Supported"):
        get_chat_model(Settings.from_env({"LLM_PROVIDER": "nope"}))


def test_openai_requires_key_or_base_url():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_chat_model(Settings.from_env({"LLM_PROVIDER": "openai"}))


def test_registry_lists_providers():
    assert {"openai", "ollama"} <= set(PROVIDERS)


def test_shipped_weights_are_valid():
    cfg = load_scoring(Settings.from_env({}).weights_path)
    assert abs(sum(cfg.weights.values()) - 1.0) < 1e-6
    assert cfg.regions["new_england"] == ("MA", "CT", "RI", "NH", "VT", "ME")


def test_weights_must_sum_to_one(tmp_path):
    bad = tmp_path / "w.yaml"
    bad.write_text(
        """
weights: {utilization: 0.5, congestion: 0.5, scale: 0.5, long_haul: 0.0, growth: 0.0, demand_context: 0.0}
congestion: {dep_delay_rate: 0.4, taxi_out: 0.3, nas_delay_per_arrival: 0.3}
long_haul_miles: 2500
long_haul_min_share: 0.02
long_haul_sensitivity_miles: [2500]
min_passengers_per_year: 100000
unmet_demand: {target_load_factor: 0.8, target_sensitivity: [0.8], peak_months: 3, top_routes: 5}
sensitivity: {runs: 5, weight_jitter: 0.2, seed: 1, top_n: 3, stable_share: 0.9}
comparison: {confidence_level: 0.95}
regions: {x: [MA]}
"""
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_scoring(bad)


def test_ollama_num_ctx_from_env():
    assert Settings.from_env({}).ollama_num_ctx is None
    s = Settings.from_env({"OLLAMA_NUM_CTX": "8192"})
    assert s.ollama_num_ctx == 8192 and get_chat_model(s).num_ctx == 8192
    assert get_chat_model(Settings.from_env({})).num_ctx is None  # server default
    for bad in ("big", "0", "-1"):
        with pytest.raises(ConfigError, match="OLLAMA_NUM_CTX"):
            Settings.from_env({"OLLAMA_NUM_CTX": bad})
