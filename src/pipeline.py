"""Pipeline orchestration.

Each stage reads what it needs from SQLite and writes its output back, so any
stage can be re-run on its own. `run_pipeline` executes them in order for the
one-command entry point.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.logging_config import get_logger
from config.settings import AppSettings, search_queries
from src.features.builder import FEATURE_COLUMNS, build_features
from src.ingestion.apify_x import ApifyIngestionError, ApifyXClient
from src.ingestion.news import collect_news
from src.ingestion.prices import PriceIngestionError, fetch_prices
from src.ingestion.synthetic import generate_synthetic_posts, generate_synthetic_prices
from src.models.train import fit_on_history, forecast_ahead, importance_table
from src.reporting.evaluation_report import build_report_blocks, write_markdown, write_pdf
from src.reporting.outlook import build_outlook, write_outlook
from src.sentiment.aggregator import aggregate_bars, score_posts
from src.storage.database import SOURCE_APIFY, SOURCE_SYNTHETIC, SOURCE_YFINANCE, Database
from src.validation.metrics import evaluate_predictions
from src.validation.walk_forward import run_walk_forward
from src.visualization import charts

log = get_logger(__name__)


@dataclass
class StageResult:
    name: str
    ok: bool
    summary: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        status = "ok" if self.ok else "FAILED"
        parts = ", ".join(f"{key}={value}" for key, value in self.summary.items())
        text = f"[{self.name}] {status}" + (f" · {parts}" if parts else "")
        for note in self.notes:
            text += f"\n    - {note}"
        return text


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def post_source(settings: AppSettings) -> str:
    return SOURCE_SYNTHETIC if settings.demo_mode else SOURCE_APIFY


def price_source(settings: AppSettings) -> str:
    return SOURCE_SYNTHETIC if settings.demo_mode else SOURCE_YFINANCE


# --------------------------------------------------------------------- stage 1
def stage_ingest(settings: AppSettings, db: Database) -> StageResult:
    notes: list[str] = []
    queries = search_queries(settings)

    # ---- prices
    try:
        if settings.demo_mode:
            prices = generate_synthetic_prices(
                settings.asset_symbol, settings.period_days, settings.bar_interval, settings.random_seed
            )
        else:
            prices = fetch_prices(settings.asset_symbol, settings.period_days, settings.bar_interval)
        inserted = db.upsert_prices(prices, settings.asset_symbol, settings.bar_interval, price_source(settings))
        log.info("prices: %d row(s) stored (%d new)", len(prices), inserted)
    except PriceIngestionError as exc:
        log.error("price ingestion failed: %s", exc)
        return StageResult("ingest", False, {}, [str(exc)])

    # ---- X posts
    post_frame = pd.DataFrame()
    if settings.demo_mode:
        post_frame = generate_synthetic_posts(prices, settings.asset_symbol, settings.bar_interval, settings.random_seed)
    elif settings.has_apify:
        client = ApifyXClient(
            token=settings.apify_token,
            actor_id=settings.apify_actor_id,
            input_json=settings.apify_actor_input,
            dataset_id=settings.apify_dataset_id,
            search_key=settings.apify_search_key,
            max_items=settings.apify_max_items,
            timeout_s=settings.apify_timeout_s,
        )
        try:
            post_frame = client.collect(queries, settings.asset_symbol, settings.bar_interval)
        except ApifyIngestionError as exc:
            notes.append(f"Apify ingestion failed: {exc}")
            log.error("Apify ingestion failed: %s", exc)
    else:
        notes.append("X ingestion skipped: APIFY_TOKEN / APIFY_ACTOR_ID not configured.")

    inserted_posts = db.upsert_posts(post_frame) if not post_frame.empty else 0
    if not post_frame.empty:
        log.info("posts: %d row(s) stored (%d new)", len(post_frame), inserted_posts)
    else:
        notes.append("No X posts were collected in this run.")

    # ---- news (context only)
    news_frame = pd.DataFrame()
    if settings.demo_mode:
        notes.append("News lookup skipped in demo mode (no provider calls).")
    else:
        news_queries = [f"{settings.asset_name} {settings.asset_symbol.split('-')[0]} news"]
        news_frame, news_problems = collect_news(
            news_queries,
            settings.asset_symbol,
            tavily_key=settings.tavily_api_key,
            exa_key=settings.exa_api_key,
            max_results=settings.news_max_results,
            lookback_days=settings.news_lookback_days,
        )
        notes.extend(news_problems)
        if not news_frame.empty:
            db.upsert_news(news_frame)
            log.info("news: %d unique item(s) stored", len(news_frame))

    return StageResult(
        "ingest",
        True,
        {
            "price_bars": len(prices),
            "posts": len(post_frame),
            "new_posts": inserted_posts,
            "news": len(news_frame),
        },
        notes,
    )


# --------------------------------------------------------------------- stage 2
def stage_sentiment(settings: AppSettings, db: Database) -> StageResult:
    prices = db.load_prices(settings.asset_symbol, settings.bar_interval, price_source(settings))
    if prices.empty:
        return StageResult("sentiment", False, {}, ["No prices in the database; run the ingest stage first."])

    posts = db.load_posts(settings.asset_symbol, post_source(settings))
    notes: list[str] = []
    if posts.empty:
        notes.append("No posts found in the database; sentiment bars will be empty.")

    scored = score_posts(posts, settings.sentiment_backend)
    if not scored.empty:
        db.upsert_posts(scored)

    sentiment = aggregate_bars(scored, prices.index, settings.bar_interval)
    db.upsert_sentiment_bars(sentiment, settings.asset_symbol, settings.bar_interval, post_source(settings))
    coverage = int((sentiment["post_count"] > 0).sum())
    return StageResult(
        "sentiment",
        True,
        {"posts_scored": len(scored), "bars": len(sentiment), "bars_with_posts": coverage},
        notes,
    )


# --------------------------------------------------------------------- stage 3
def stage_features(settings: AppSettings, db: Database) -> StageResult:
    prices = db.load_prices(settings.asset_symbol, settings.bar_interval, price_source(settings))
    sentiment = db.load_sentiment_bars(settings.asset_symbol, settings.bar_interval, post_source(settings))
    if prices.empty:
        return StageResult("features", False, {}, ["No prices in the database; run the ingest stage first."])
    if sentiment.empty:
        sentiment = aggregate_bars(pd.DataFrame(), prices.index, settings.bar_interval)

    features, target_returns, target_prices, base_prices = build_features(
        prices, sentiment, settings.horizons
    )
    if features.empty:
        return StageResult(
            "features",
            False,
            {"features": len(FEATURE_COLUMNS)},
            ["Every bar was dropped during warm-up; increase PERIOD_DAYS."],
        )

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    path = settings.processed_dir / f"{settings.asset_symbol}_{settings.bar_interval}_features.csv"
    features.assign(base_price=base_prices).to_csv(path)
    log.info("feature matrix written to %s (%d rows x %d cols)", path, *features.shape)
    return StageResult(
        "features",
        True,
        {"rows": len(features), "columns": len(FEATURE_COLUMNS), "last_bar": str(features.index[-1])},
    )


# --------------------------------------------------------------------- stage 4
def stage_model(settings: AppSettings, db: Database, run_id: str | None = None) -> StageResult:
    run_id = run_id or new_run_id()
    prices = db.load_prices(settings.asset_symbol, settings.bar_interval, price_source(settings))
    sentiment = db.load_sentiment_bars(settings.asset_symbol, settings.bar_interval, post_source(settings))
    if prices.empty:
        return StageResult("model", False, {"run_id": run_id}, ["No prices in the database."])
    if sentiment.empty:
        sentiment = aggregate_bars(pd.DataFrame(), prices.index, settings.bar_interval)

    features, target_returns, target_prices, base_prices = build_features(
        prices, sentiment, settings.horizons
    )
    if features.empty:
        return StageResult("model", False, {"run_id": run_id}, ["Feature matrix is empty."])

    fitted = fit_on_history(features, target_returns, settings.models, settings.random_seed)
    latest = features.iloc[[-1]]
    forecast = forecast_ahead(
        fitted,
        latest,
        float(base_prices.iloc[-1]),
        settings.horizons,
        features.index[-1],
        settings.bar_interval,
    )
    forecast["run_id"] = run_id
    forecast["mode"] = "demo" if settings.demo_mode else "live"
    forecast["symbol"] = settings.asset_symbol
    forecast["interval"] = settings.bar_interval
    db.save_forecasts(forecast[["run_id", "mode", "symbol", "interval", "model", "horizon", "origin_ts", "target_ts", "base_price", "predicted_return", "predicted_price", "actual_return", "actual_price"]])
    forecast.to_csv(settings.forecast_dir / f"{settings.asset_symbol}_forecasts_{run_id}.csv", index=False)

    walk = run_walk_forward(
        features,
        target_returns,
        base_prices,
        settings.models,
        settings.horizons,
        settings.min_train_samples,
        settings.refit_every,
        settings.random_seed,
    )
    notes: list[str] = []
    if walk.empty:
        notes.append("Walk-forward validation produced no forecasts; metrics will be empty.")
        metrics = pd.DataFrame()
    else:
        stored = walk.predictions.copy()
        stored["run_id"] = run_id
        stored["mode"] = "demo" if settings.demo_mode else "live"
        stored["symbol"] = settings.asset_symbol
        stored["interval"] = settings.bar_interval
        db.save_forecasts(stored[["run_id", "mode", "symbol", "interval", "model", "horizon", "origin_ts", "target_ts", "base_price", "predicted_return", "predicted_price", "actual_return", "actual_price"]])
        walk.predictions.to_csv(settings.forecast_dir / f"{settings.asset_symbol}_walkforward_{run_id}.csv", index=False)
        metrics = evaluate_predictions(walk.predictions, settings.resolved_annualization, settings.transaction_cost_bps)
        metrics["run_id"] = run_id
        metrics["mode"] = "demo" if settings.demo_mode else "live"
        metrics["symbol"] = settings.asset_symbol
        db.save_metrics(metrics[["run_id", "mode", "symbol", "model", "horizon", "metric_name", "metric_value", "n_samples", "notes"]])

    importance = importance_table(fitted, FEATURE_COLUMNS)
    return StageResult(
        "model",
        True,
        {
            "run_id": run_id,
            "forecasts": len(forecast),
            "walk_forward_rows": len(walk.predictions),
            "origins": walk.steps,
            "refits": sum(walk.fits.values()),
            "metrics": len(metrics),
        },
        notes,
    )


# --------------------------------------------------------------------- stage 5
def stage_charts(settings: AppSettings, db: Database, run_id: str | None = None) -> StageResult:
    prices = db.load_prices(settings.asset_symbol, settings.bar_interval, price_source(settings))
    sentiment = db.load_sentiment_bars(settings.asset_symbol, settings.bar_interval, post_source(settings))
    if prices.empty:
        return StageResult("charts", False, {}, ["No prices in the database."])
    if sentiment.empty:
        sentiment = aggregate_bars(pd.DataFrame(), prices.index, settings.bar_interval)

    forecasts = db.load_forecasts(run_id=run_id, symbol=settings.asset_symbol)
    predictions = forecasts[forecasts["actual_price"].notna()] if not forecasts.empty else pd.DataFrame()
    forward = forecasts[forecasts["actual_price"].isna()] if not forecasts.empty else pd.DataFrame()
    metrics = db.load_metrics(run_id=run_id) if run_id else db.load_metrics()

    written: list[Path] = []
    written += charts.price_vs_sentiment(prices, sentiment, settings.asset_symbol, settings.chart_dir, settings.demo_mode)
    written += charts.sentiment_and_volume(prices, sentiment, settings.asset_symbol, settings.chart_dir, settings.demo_mode)
    written += charts.sentiment_price_matrix(
        prices, sentiment, settings.asset_symbol, settings.chart_dir, settings.demo_mode, predictions=predictions
    )
    written += charts.sentiment_return_density(prices, sentiment, settings.asset_symbol, settings.chart_dir, settings.demo_mode)
    if not predictions.empty:
        written += charts.actual_vs_predicted(predictions, settings.asset_symbol, settings.chart_dir, settings.demo_mode)
    if not metrics.empty:
        written += charts.evaluation_summary(metrics, settings.asset_symbol, settings.chart_dir, settings.demo_mode)
    if not forward.empty:
        written += charts.horizon_forecast(forward, settings.asset_symbol, settings.chart_dir, settings.demo_mode)

    return StageResult("charts", True, {"charts": len(written)})


# --------------------------------------------------------------------- stage 6
def stage_report(settings: AppSettings, db: Database, run_id: str | None = None) -> StageResult:
    run_id = run_id or db.latest_run_id(settings.asset_symbol)
    if not run_id:
        return StageResult("report", False, {}, ["No stored run found; run the model stage first."])

    prices = db.load_prices(settings.asset_symbol, settings.bar_interval, price_source(settings))
    sentiment = db.load_sentiment_bars(settings.asset_symbol, settings.bar_interval, post_source(settings))
    posts = db.load_posts(settings.asset_symbol, post_source(settings))
    news = db.load_news(settings.asset_symbol)
    forecasts = db.load_forecasts(run_id=run_id, symbol=settings.asset_symbol)
    metrics = db.load_metrics(run_id=run_id)
    predictions = forecasts[forecasts["actual_price"].notna()] if not forecasts.empty else pd.DataFrame()
    forward = forecasts[forecasts["actual_price"].isna()] if not forecasts.empty else pd.DataFrame()

    coverage = _coverage(prices, sentiment, posts, news, settings)
    walk_summary = {
        "steps": int(predictions["origin_ts"].nunique()) if not predictions.empty else 0,
        "first_origin": str(predictions["origin_ts"].min()) if not predictions.empty else "n/a",
        "last_origin": str(predictions["origin_ts"].max()) if not predictions.empty else "n/a",
        "total_forecasts": len(predictions),
        "skipped_text": "not persisted; see the run log",
    }

    outlook_md = build_outlook(settings, prices, sentiment, posts, news, forward, metrics, run_id)
    outlook_path = write_outlook(outlook_md, settings, run_id)

    chart_paths = sorted(
        path
        for path in settings.chart_dir.glob(f"{settings.asset_symbol}_*.png")
    )
    blocks = build_report_blocks(
        settings, coverage, walk_summary, metrics, forward, _importance_from_db(db), chart_paths, run_id
    )
    report_path = write_markdown(blocks, settings.report_dir / f"{settings.asset_symbol}_evaluation_report_{run_id}.md")
    pdf_path = write_pdf(blocks, settings.report_dir / f"{settings.asset_symbol}_evaluation_report_{run_id}.pdf")

    summary = {"outlook": outlook_path.name, "report": report_path.name}
    notes = [] if pdf_path else ["PDF report skipped: install reportlab to enable it."]
    if pdf_path:
        summary["pdf"] = pdf_path.name
    return StageResult("report", True, summary, notes)


def _importance_from_db(db: Database) -> pd.DataFrame:
    """Feature importance is not persisted; the report notes when it is absent."""
    return pd.DataFrame(columns=["model", "horizon", "rank", "feature", "importance"])


def _coverage(
    prices: pd.DataFrame, sentiment: pd.DataFrame, posts: pd.DataFrame, news: pd.DataFrame, settings: AppSettings
) -> dict:
    bars_with_posts = int((sentiment["post_count"] > 0).sum()) if not sentiment.empty else 0
    return {
        "price_bars": len(prices),
        "price_range": f"{prices.index.min()} → {prices.index.max()}" if not prices.empty else "n/a",
        "post_count": len(posts),
        "post_range": (
            f"{posts['posted_at'].min()} → {posts['posted_at'].max()}" if not posts.empty else "n/a"
        ),
        "bars_with_posts": bars_with_posts,
        "bars_without_posts": max(0, len(sentiment) - bars_with_posts),
        "news_count": len(news),
        "news_range": (
            f"{news['published_at'].min()} → {news['published_at'].max()}" if not news.empty else "n/a"
        ),
        "posts_dropped": 0,
        "ingestion_notes": "demo mode (synthetic)" if settings.demo_mode else "live sources",
    }


# ------------------------------------------------------------------------ run
def run_pipeline(settings: AppSettings, db: Database | None = None, stages: list[str] | None = None) -> dict:
    db = db or Database(settings.db_path)
    settings.ensure_directories()
    run_id = new_run_id()
    order = ["ingest", "sentiment", "features", "model", "charts", "report"]
    selected = stages or order

    results: list[StageResult] = []
    log.info("=== pipeline start · %s · mode=%s · run=%s ===",
             settings.asset_symbol, "demo" if settings.demo_mode else "live", run_id)
    for stage in order:
        if stage not in selected:
            continue
        if stage == "ingest":
            result = stage_ingest(settings, db)
        elif stage == "sentiment":
            result = stage_sentiment(settings, db)
        elif stage == "features":
            result = stage_features(settings, db)
        elif stage == "model":
            result = stage_model(settings, db, run_id)
        elif stage == "charts":
            result = stage_charts(settings, db, run_id)
        elif stage == "report":
            result = stage_report(settings, db, run_id)
        else:
            raise ValueError(f"unknown stage '{stage}'")
        results.append(result)
        log.info(result.describe())
        if not result.ok:
            log.error("stage '%s' failed; stopping the pipeline", stage)
            break

    failed = [result.name for result in results if not result.ok]
    log.info(
        "=== pipeline finished · %d/%d stage(s) ok · db=%s ===",
        len(results) - len(failed),
        len(results),
        db.path,
    )
    return {
        "run_id": run_id,
        "results": results,
        "failed": failed,
        "counts": db.table_counts(),
    }
