"""Configuration loading, validation and failure messaging."""

from __future__ import annotations

import pytest

from config.settings import (
    AppSettings,
    load_settings,
    search_queries,
    validate_settings,
)


def _clear_env(monkeypatch):
    for key in (
        "APIFY_TOKEN",
        "APIFY_ACTOR_ID",
        "APIFY_DATASET_ID",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
        "DEMO_MODE",
        "MODELS",
        "ASSET_SYMBOL",
        "ASSET_NAME",
        "ASSET_KEYWORDS",
        "FORECAST_HORIZONS",
        "BAR_INTERVAL",
        "SENTIMENT_BACKEND",
        "ANNUALIZATION_PERIODS",
        "PERIOD_DAYS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_defaults_when_no_env_is_set(monkeypatch):
    _clear_env(monkeypatch)
    settings = load_settings()
    assert settings.asset_symbol == "BTC-USD"
    assert settings.period_days == 180
    assert settings.bar_interval == "1d"
    assert settings.models == ("naive", "random_forest", "hist_gbm")
    assert settings.horizons == (1, 2, 3, 4, 5)
    assert settings.demo_mode is False


def test_cli_flag_overrides_demo_mode(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEMO_MODE", "false")
    assert load_settings(demo=True).demo_mode is True
    assert load_settings(demo=False).demo_mode is False


def test_live_run_without_credentials_is_rejected_with_guidance(monkeypatch):
    _clear_env(monkeypatch)
    settings = load_settings()
    report = validate_settings(settings)
    assert not report.ok
    joined = " ".join(report.errors + report.disabled)
    assert "APIFY_TOKEN" in joined
    assert "TAVILY_API_KEY" in joined
    assert "EXA_API_KEY" in joined
    assert "--demo" in joined


def test_missing_individual_credentials_disable_only_their_feature(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("APIFY_TOKEN", "tok")
    monkeypatch.setenv("APIFY_ACTOR_ID", "someone/actor")
    monkeypatch.setenv("TAVILY_API_KEY", "tav")
    settings = load_settings()
    report = validate_settings(settings)
    assert report.ok
    assert settings.has_apify and settings.has_tavily and not settings.has_exa
    assert any("Exa" in item for item in report.disabled)
    assert not any("Apify" in item for item in report.disabled)


def test_dataset_id_alone_is_enough_for_apify(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("APIFY_TOKEN", "tok")
    monkeypatch.setenv("APIFY_DATASET_ID", "ds-123")
    settings = load_settings()
    assert settings.has_apify


def test_demo_mode_warns_and_skips_credential_errors(monkeypatch):
    _clear_env(monkeypatch)
    settings = load_settings(demo=True)
    report = validate_settings(settings)
    assert report.ok
    assert any("DEMO_MODE" in item for item in report.warnings)


def test_invalid_model_list_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MODELS", "naive,deep_magic")
    with pytest.raises(ValueError, match="deep_magic"):
        load_settings()


def test_invalid_interval_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BAR_INTERVAL", "3d")
    with pytest.raises(ValueError, match="BAR_INTERVAL"):
        load_settings()


def test_invalid_backend_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SENTIMENT_BACKEND", "gpt-vibes")
    with pytest.raises(ValueError, match="SENTIMENT_BACKEND"):
        load_settings()


def test_invalid_horizons_raise(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FORECAST_HORIZONS", "1,x,3")
    with pytest.raises(ValueError, match="FORECAST_HORIZONS"):
        load_settings()
    monkeypatch.setenv("FORECAST_HORIZONS", "0,1")
    with pytest.raises(ValueError, match="positive integers"):
        load_settings()


def test_too_short_history_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PERIOD_DAYS", "10")
    with pytest.raises(ValueError, match="PERIOD_DAYS"):
        load_settings()


def test_malformed_actor_input_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("APIFY_ACTOR_INPUT_JSON", "{not json}")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_settings()
    monkeypatch.setenv("APIFY_ACTOR_INPUT_JSON", "[1,2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_settings()


def test_actor_input_is_parsed_into_a_dict(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("APIFY_ACTOR_INPUT_JSON", '{"maxItems": 100, "searchType": "Tweet"}')
    settings = load_settings()
    assert settings.apify_actor_input == {"maxItems": 100, "searchType": "Tweet"}


def test_annualisation_auto_detects_crypto_and_equity(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_SYMBOL", "BTC-USD")
    assert load_settings().resolved_annualization == 365
    monkeypatch.setenv("ASSET_SYMBOL", "NVDA")
    assert load_settings().resolved_annualization == 252
    monkeypatch.setenv("ANNUALIZATION_PERIODS", "52")
    assert load_settings().resolved_annualization == 52


def test_search_queries_cover_cashtag_ticker_name_and_keywords(monkeypatch):
    _clear_env(monkeypatch)
    settings = load_settings()
    queries = search_queries(settings)
    assert "$BTC" in queries
    assert "BTC" in queries
    assert "Bitcoin" in queries
    assert len(queries) == len({query.lower() for query in queries})


def test_custom_keywords_override_defaults(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ASSET_SYMBOL", "NVDA")
    monkeypatch.setenv("ASSET_NAME", "Nvidia Corp")
    monkeypatch.setenv("ASSET_KEYWORDS", "gpu, ai chips")
    settings = load_settings()
    assert settings.asset_name == "Nvidia Corp"
    assert settings.asset_keywords == ("gpu", "ai chips")


def test_short_history_warning_mentions_sample_size(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PERIOD_DAYS", "75")
    monkeypatch.setenv("MIN_TRAIN_SAMPLES", "60")
    report = validate_settings(load_settings(demo=True))
    assert any("walk-forward steps" in item for item in report.warnings)


def test_settings_is_immutable():
    settings = AppSettings()
    with pytest.raises(Exception):
        settings.asset_symbol = "TSLA"


def test_ensure_directories_creates_everything(tmp_path):
    settings = AppSettings(
        db_path=tmp_path / "nested" / "db.sqlite3",
        raw_dir=tmp_path / "nested" / "raw",
        processed_dir=tmp_path / "nested" / "processed",
        forecast_dir=tmp_path / "nested" / "forecasts",
        chart_dir=tmp_path / "nested" / "charts",
        report_dir=tmp_path / "nested" / "reports",
    )
    settings.ensure_directories()
    assert settings.chart_dir.exists()
    assert settings.report_dir.exists()


def test_configured_search_terms_win_over_generated_ones(monkeypatch):
    """A query written in APIFY_ACTOR_INPUT_JSON must survive as the only run."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        "APIFY_ACTOR_INPUT_JSON",
        '{"searchTerms": ["(Bitcoin OR BTC OR $BTC) -filter:retweets"], "sort": "Latest"}',
    )
    settings = load_settings()
    queries = search_queries(settings)
    assert queries == ["(Bitcoin OR BTC OR $BTC) -filter:retweets"]


def test_multiple_configured_terms_are_deduplicated(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("APIFY_ACTOR_INPUT_JSON", '{"searchTerms": ["$BTC", "$btc", " BTC "]}')
    settings = load_settings()
    # case-insensitive duplicates collapse; surrounding whitespace is stripped
    assert search_queries(settings) == ["$BTC", "BTC"]


def test_search_terms_fall_back_to_the_asset_when_unset(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("APIFY_ACTOR_INPUT_JSON", '{"sort": "Latest", "maxItems": 10}')
    settings = load_settings()
    assert search_queries(settings) == ["$BTC", "BTC", "Bitcoin", "crypto"]


def test_empty_search_terms_fall_back_to_the_asset(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("APIFY_ACTOR_INPUT_JSON", '{"searchTerms": []}')
    settings = load_settings()
    assert search_queries(settings)[0] == "$BTC"
