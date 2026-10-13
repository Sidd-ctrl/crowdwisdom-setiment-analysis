"""End-to-end smoke test of the demo pipeline.

Runs every stage on synthetic data with a small model set, then asserts that
the artefacts a reviewer would open actually exist and contain real numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import AppSettings
from src.pipeline import run_pipeline
from src.storage.database import Database

REQUIRED_CHARTS = (
    "price_vs_sentiment.png",
    "sentiment_volume.png",
    "sentiment_price_matrix.png",
    "sentiment_return_density.png",
    "actual_vs_predicted",
    "evaluation_summary.png",
    "forward_projection.png",
)


@pytest.fixture(scope="module")
def smoke_settings(tmp_path_factory):
    """One shared demo run for the whole module -- a full pipeline pass is slow."""
    root = tmp_path_factory.mktemp("smoke")
    return AppSettings(
        asset_symbol="BTC-USD",
        period_days=120,
        bar_interval="1d",
        demo_mode=True,
        models=("naive", "random_forest"),
        horizons=(1, 2, 3),
        min_train_samples=25,
        refit_every=5,
        random_seed=42,
        db_path=root / "pipeline.sqlite3",
        raw_dir=root / "raw",
        processed_dir=root / "processed",
        forecast_dir=root / "forecasts",
        chart_dir=root / "charts",
        report_dir=root / "reports",
    )


@pytest.fixture(scope="module")
def smoke_run(smoke_settings):
    db = Database(smoke_settings.db_path)
    outcome = run_pipeline(smoke_settings, db)
    assert outcome["failed"] == [], [result.describe() for result in outcome["results"]]
    return db, outcome


def test_full_demo_pipeline_produces_every_artefact(smoke_settings, smoke_run):
    _, outcome = smoke_run
    counts = outcome["counts"]
    assert counts["prices"] > 0
    assert counts["x_posts"] > 0
    assert counts["sentiment_bars"] > 0
    assert counts["forecasts"] > 0
    assert counts["evaluation_metrics"] > 0

    charts = [path.name for path in smoke_settings.chart_dir.iterdir()]
    for expected in REQUIRED_CHARTS:
        assert any(expected in name for name in charts), f"missing chart: {expected}"

    reports = [path.name for path in smoke_settings.report_dir.iterdir()]
    assert any("sentiment_outlook" in name for name in reports)
    assert any("evaluation_report" in name and name.endswith(".md") for name in reports)

    forecast_files = list(smoke_settings.forecast_dir.glob("*_forecasts_*.csv"))
    assert forecast_files, "no forward forecast CSV written"


def test_stored_metrics_cover_every_required_measure(smoke_settings, smoke_run):
    db, _ = smoke_run
    metrics = db.load_metrics()
    assert not metrics.empty
    names = set(metrics["metric_name"])
    assert {"mae", "rmse", "direction_accuracy", "sharpe", "sortino", "max_drawdown"} <= names
    assert (metrics[metrics.metric_name == "mae"]["n_samples"] > 0).all()
    assert set(metrics["mode"]) == {"demo"}


def test_demo_data_is_never_labelled_live(smoke_settings, smoke_run):
    db, _ = smoke_run
    posts = db.load_posts(smoke_settings.asset_symbol, "synthetic")
    assert not posts.empty
    assert db.load_posts(smoke_settings.asset_symbol, "apify").empty
    prices = db.load_prices(smoke_settings.asset_symbol, "1d", "synthetic")
    assert not prices.empty
    assert db.load_prices(smoke_settings.asset_symbol, "1d", "yfinance").empty


def test_reingesting_does_not_duplicate_rows(smoke_settings):
    from dataclasses import replace

    isolated = replace(smoke_settings, db_path=smoke_settings.db_path.parent / "reingest.sqlite3")
    db = Database(isolated.db_path)
    first = run_pipeline(isolated, db, stages=["ingest", "sentiment"])
    counts_after_first = db.table_counts()
    run_pipeline(isolated, db, stages=["ingest", "sentiment"])
    counts_after_second = db.table_counts()
    assert counts_after_first["prices"] == counts_after_second["prices"]
    assert counts_after_first["x_posts"] == counts_after_second["x_posts"]
    assert counts_after_first["sentiment_bars"] == counts_after_second["sentiment_bars"]
    assert first["failed"] == []


def test_outlook_separates_measurement_from_interpretation(smoke_settings, smoke_run):
    outlook = next(smoke_settings.report_dir.glob("*sentiment_outlook*.md")).read_text(encoding="utf-8")
    assert "## 1. Measured snapshot" in outlook
    assert "## 2. Interpretation" in outlook
    assert "## 5. Risks and data limitations" in outlook
    assert "DEMO MODE" in outlook
    assert "n+0" in outlook


def test_evaluation_report_states_its_data_mode(smoke_settings, smoke_run):
    report = next(
        path for path in smoke_settings.report_dir.glob("*evaluation_report*.md")
    ).read_text(encoding="utf-8")
    for section in (
        "1. Methodology",
        "2. Data sources and coverage",
        "3. Feature engineering",
        "4. Model selection",
        "5. Walk-forward validation",
        "6. Accuracy metrics",
        "7. Baseline comparison",
        "8. Strategy risk metrics",
        "12. Limitations and conclusions",
    ):
        assert section in report, f"missing section: {section}"
    assert "DEMO MODE" in report
    assert "sharpe" not in report.split("## 8")[1].lower() or "Sharpe" in report


def test_pipeline_stops_cleanly_without_data(smoke_settings):
    """A model run on an empty database must fail with a message, not a traceback."""
    from src.pipeline import stage_model

    db = Database(smoke_settings.db_path.parent / "empty.sqlite3")
    result = stage_model(smoke_settings, db)
    assert not result.ok
    assert result.notes
    assert "No prices" in result.notes[0]


def test_forecast_table_treats_n0_as_observed(smoke_settings, smoke_run):
    db, outcome = smoke_run
    forecasts = db.load_forecasts(run_id=outcome["run_id"])
    observed = forecasts[forecasts["horizon"] == 0]
    assert len(observed) == 1
    assert observed["model"].iloc[0] == "observed"
    assert observed["predicted_price"].iloc[0] == pytest.approx(observed["base_price"].iloc[0])
    assert set(forecasts["horizon"]) == {0, *smoke_settings.horizons}


def test_processed_feature_matrix_is_written(smoke_settings, smoke_run):
    files = list(smoke_settings.processed_dir.glob("*_features.csv"))
    assert files
    frame = pd.read_csv(files[0])
    assert "base_price" in frame.columns
    assert "sent_mean" in frame.columns
    assert "ret_1" in frame.columns
