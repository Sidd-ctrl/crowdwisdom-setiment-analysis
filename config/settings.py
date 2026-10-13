"""Configuration loading and validation.

Everything the pipeline needs at runtime is read from environment variables
(optionally seeded from a .env file) into a single frozen dataclass. Nothing in
this module performs network or disk I/O beyond reading the .env file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_MODELS = ("naive", "random_forest", "hist_gbm", "xgboost")
VALID_INTERVALS = ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo")
VALID_BACKENDS = ("vader", "finbert")

CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "USD", "USDT")


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_list(key: str, default: Iterable[str]) -> list[str]:
    raw = _env(key)
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class AppSettings:
    """Resolved runtime configuration."""

    asset_symbol: str = "BTC-USD"
    asset_name: str = "Bitcoin"
    asset_keywords: tuple[str, ...] = ("bitcoin", "btc")
    period_days: int = 180
    bar_interval: str = "1d"
    annualization_periods: int | None = None
    transaction_cost_bps: float = 10.0

    apify_token: str = ""
    apify_actor_id: str = ""
    apify_actor_input: dict = field(default_factory=dict)
    apify_dataset_id: str = ""
    apify_search_key: str = "searchTerms"
    apify_max_items: int = 5000
    apify_timeout_s: int = 600

    tavily_api_key: str = ""
    exa_api_key: str = ""
    news_max_results: int = 15
    news_lookback_days: int = 30

    sentiment_backend: str = "vader"

    models: tuple[str, ...] = ("naive", "random_forest", "hist_gbm")
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5)
    min_train_samples: int = 60
    refit_every: int = 5
    random_seed: int = 42

    demo_mode: bool = False
    log_level: str = "INFO"

    project_root: Path = PROJECT_ROOT
    db_path: Path = PROJECT_ROOT / "data" / "pipeline.sqlite3"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    forecast_dir: Path = PROJECT_ROOT / "outputs" / "forecasts"
    chart_dir: Path = PROJECT_ROOT / "outputs" / "charts"
    report_dir: Path = PROJECT_ROOT / "outputs" / "reports"

    @property
    def resolved_annualization(self) -> int:
        if self.annualization_periods:
            return int(self.annualization_periods)
        upper = self.asset_symbol.upper()
        return 365 if any(upper.endswith(sfx) for sfx in CRYPTO_SUFFIXES) else 252

    @property
    def has_apify(self) -> bool:
        return bool(self.apify_token) and bool(self.apify_actor_id or self.apify_dataset_id)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_exa(self) -> bool:
        return bool(self.exa_api_key)

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.forecast_dir,
            self.chart_dir,
            self.report_dir,
            self.db_path.parent,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigReport:
    """Outcome of startup configuration validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_text(self, settings: AppSettings) -> str:
        lines = [
            "Configuration check",
            f"  asset           : {settings.asset_symbol}",
            f"  window          : {settings.period_days} days @ {settings.bar_interval} bars",
            f"  annualisation   : {settings.resolved_annualization} periods/year",
            f"  cost assumption : {settings.transaction_cost_bps:.1f} bps one-way",
            f"  sentiment       : {settings.sentiment_backend}",
            f"  models          : {', '.join(settings.models)}",
            f"  horizons        : {', '.join(f'n+{h}' for h in settings.horizons)}",
            f"  mode            : {'DEMO (synthetic)' if settings.demo_mode else 'LIVE'}",
        ]
        for item in self.disabled:
            lines.append(f"  unavailable     : {item}")
        for item in self.warnings:
            lines.append(f"  warning         : {item}")
        for item in self.errors:
            lines.append(f"  ERROR           : {item}")
        if self.ok and not self.warnings and not self.disabled:
            lines.append("  status          : OK")
        return "\n".join(lines)


ASSET_ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "BTC-USD": ("Bitcoin", ("bitcoin", "btc", "crypto")),
    "ETH-USD": ("Ethereum", ("ethereum", "eth", "crypto")),
    "SOL-USD": ("Solana", ("solana", "sol", "crypto")),
    "TSLA": ("Tesla", ("tesla", "elon musk", "electric vehicle")),
    "NVDA": ("Nvidia", ("nvidia", "gpu", "ai chips")),
    "AAPL": ("Apple", ("apple", "iphone", "aapl")),
    "SPY": ("S&P 500 ETF", ("s&p 500", "spy", "stock market")),
}


def base_ticker(symbol: str) -> str:
    """BTC-USD -> BTC, NVDA -> NVDA."""
    return symbol.split("-")[0].strip()


def default_asset_name(symbol: str) -> str:
    return ASSET_ALIASES.get(symbol.upper(), (base_ticker(symbol), ()))[0]


def default_asset_keywords(symbol: str) -> tuple[str, ...]:
    known = ASSET_ALIASES.get(symbol.upper(), (None, ()))[1]
    return known or (base_ticker(symbol).lower(),)


def search_queries(settings: "AppSettings") -> list[str]:
    """Search terms for the X actor. Each term becomes its own actor run.

    Terms written explicitly in APIFY_ACTOR_INPUT_JSON take precedence, so a
    query such as "(Bitcoin OR BTC OR $BTC) -filter:retweets" is passed through
    untouched instead of being replaced by the generated terms. When no terms
    are configured, they are derived from the asset symbol.
    """
    configured = settings.apify_actor_input.get(settings.apify_search_key)
    if isinstance(configured, str):
        candidates = [configured]
    elif isinstance(configured, (list, tuple)):
        candidates = [str(item) for item in configured]
    else:
        candidates = []

    if not candidates:
        ticker = base_ticker(settings.asset_symbol)
        candidates = [f"${ticker}", ticker, settings.asset_name, *settings.asset_keywords]

    seen: list[str] = []
    for query in candidates:
        cleaned = query.strip()
        if cleaned and cleaned.lower() not in [s.lower() for s in seen]:
            seen.append(cleaned)
    return seen


def _resolve_path(raw: str, project_root: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root / path


def _parse_actor_input(raw: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"APIFY_ACTOR_INPUT_JSON is not valid JSON: {exc}. "
            "Use a JSON object such as {\"maxItems\": 2000}."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("APIFY_ACTOR_INPUT_JSON must be a JSON object, not a list or scalar.")
    return parsed


def load_settings(env_file: str | os.PathLike | None = None, demo: bool | None = None) -> AppSettings:
    """Read configuration from the environment (and .env when present).

    `demo` overrides DEMO_MODE so the CLI flag always wins over the env file.
    """
    default_env = PROJECT_ROOT / ".env"
    if env_file:
        load_dotenv(env_file, override=True)
    elif default_env.exists():
        load_dotenv(default_env, override=False)

    project_root = Path(_env("PROJECT_ROOT", str(PROJECT_ROOT))).expanduser().resolve()
    raw_models = [m.lower() for m in _env_list("MODELS", ("naive", "random_forest", "hist_gbm"))]
    unknown = [m for m in raw_models if m not in VALID_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model(s) in MODELS: {', '.join(unknown)}. Choose from {', '.join(VALID_MODELS)}."
        )

    backend = _env("SENTIMENT_BACKEND", "vader").lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"SENTIMENT_BACKEND must be one of {VALID_BACKENDS}, got '{backend}'.")

    interval = _env("BAR_INTERVAL", "1d")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"BAR_INTERVAL must be one of {VALID_INTERVALS}, got '{interval}'.")

    horizons = tuple(sorted({_parse_horizon(h) for h in _env_list("FORECAST_HORIZONS", ("1", "2", "3", "4", "5"))}))
    demo_flag = _env_bool("DEMO_MODE", False) if demo is None else bool(demo)

    settings = AppSettings(
        asset_symbol=_env("ASSET_SYMBOL", "BTC-USD").upper(),
        asset_name=_env("ASSET_NAME") or default_asset_name(_env("ASSET_SYMBOL", "BTC-USD")),
        asset_keywords=tuple(
            k.lower()
            for k in _env_list("ASSET_KEYWORDS", default_asset_keywords(_env("ASSET_SYMBOL", "BTC-USD")))
        ),
        period_days=_env_int("PERIOD_DAYS", 180),
        bar_interval=interval,
        annualization_periods=_env_int("ANNUALIZATION_PERIODS", 0) or None,
        transaction_cost_bps=_env_float("TRANSACTION_COST_BPS", 10.0),
        apify_token=_env("APIFY_TOKEN"),
        apify_actor_id=_env("APIFY_ACTOR_ID"),
        apify_actor_input=_parse_actor_input(_env("APIFY_ACTOR_INPUT_JSON", "{}")),
        apify_dataset_id=_env("APIFY_DATASET_ID"),
        apify_search_key=_env("APIFY_SEARCH_KEY", "searchTerms"),
        apify_max_items=_env_int("APIFY_MAX_ITEMS", 5000),
        apify_timeout_s=_env_int("APIFY_TIMEOUT_S", 600),
        tavily_api_key=_env("TAVILY_API_KEY"),
        exa_api_key=_env("EXA_API_KEY"),
        news_max_results=_env_int("NEWS_MAX_RESULTS", 15),
        news_lookback_days=_env_int("NEWS_LOOKBACK_DAYS", 30),
        sentiment_backend=backend,
        models=tuple(raw_models),
        horizons=horizons,
        min_train_samples=_env_int("MIN_TRAIN_SAMPLES", 60),
        refit_every=max(1, _env_int("REFIT_EVERY", 5)),
        random_seed=_env_int("RANDOM_SEED", 42),
        demo_mode=demo_flag,
        log_level=_env("LOG_LEVEL", "INFO").upper(),
        project_root=project_root,
        db_path=_resolve_path(_env("DB_PATH", "data/pipeline.sqlite3"), project_root),
        raw_dir=_resolve_path(_env("RAW_DIR", "data/raw"), project_root),
        processed_dir=_resolve_path(_env("PROCESSED_DIR", "data/processed"), project_root),
        forecast_dir=_resolve_path(_env("FORECAST_DIR", "outputs/forecasts"), project_root),
        chart_dir=_resolve_path(_env("CHART_DIR", "outputs/charts"), project_root),
        report_dir=_resolve_path(_env("REPORT_DIR", "outputs/reports"), project_root),
    )

    if settings.period_days < 30:
        raise ValueError("PERIOD_DAYS must be at least 30 to leave room for warm-up lags and validation.")
    return settings


def _parse_horizon(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"FORECAST_HORIZONS must be integers, got '{raw}'.") from exc
    if value < 1:
        raise ValueError("FORECAST_HORIZONS must be positive integers (n+1 ... n+k).")
    return value


def validate_settings(settings: AppSettings) -> ConfigReport:
    """Report what can and cannot run with the current configuration."""
    report = ConfigReport()

    if settings.demo_mode:
        report.warnings.append(
            "DEMO_MODE is on: every price, post and metric is synthetic and clearly "
            "labelled. It must never be presented as a live market result."
        )
        report.warnings.extend(_sample_size_warnings(settings))
        return report

    if not settings.has_apify:
        report.disabled.append(
            "X/Twitter ingestion. Set APIFY_TOKEN plus APIFY_ACTOR_ID (or APIFY_DATASET_ID "
            "to reuse an existing dataset). See README 'Configuring the Apify actor'."
        )
    if not settings.has_tavily:
        report.disabled.append("Tavily news context. Set TAVILY_API_KEY.")
    if not settings.has_exa:
        report.disabled.append("Exa supplementary search. Set EXA_API_KEY.")

    if not (settings.has_apify or settings.has_tavily or settings.has_exa):
        report.errors.append(
            "No data source is configured and DEMO_MODE is off. Add credentials to .env, "
            "or run with --demo to exercise the pipeline on labelled synthetic data."
        )

    if settings.sentiment_backend == "finbert":
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            report.errors.append(
                "SENTIMENT_BACKEND=finbert requires the optional 'transformers' and 'torch' "
                "packages. Install them or switch back to SENTIMENT_BACKEND=vader."
            )

    if "xgboost" in settings.models:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            report.warnings.append(
                "xgboost is listed in MODELS but not installed; it will be skipped."
            )

    report.warnings.extend(_sample_size_warnings(settings))
    return report


def _sample_size_warnings(settings: AppSettings) -> list[str]:
    """Flag windows too short to support meaningful out-of-sample metrics."""
    budget = settings.period_days - max(settings.horizons) - settings.min_train_samples
    if budget >= 20:
        return []
    return [
        f"Only ~{budget} out-of-sample walk-forward steps are available with "
        f"PERIOD_DAYS={settings.period_days}, MIN_TRAIN_SAMPLES={settings.min_train_samples} "
        f"and horizons up to n+{max(settings.horizons)}. Metrics will be noisy; treat them "
        "as indicative, not as evidence of an edge."
    ]
