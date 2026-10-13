"""SQLite storage: schema, idempotent upserts, source separation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.storage.database import Database

PRICE_COLUMNS = ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]


def _prices():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "adj_close": [1.2, 2.2],
            "volume": [100.0, 200.0],
        }
    )


def _posts(source="apify"):
    return pd.DataFrame(
        [
            {
                "post_id": "a1",
                "symbol": "BTC-USD",
                "search_query": "$BTC",
                "post_text": "hello",
                "posted_at": pd.Timestamp("2024-01-01 03:00", tz="UTC"),
                "bar_ts": pd.Timestamp("2024-01-01", tz="UTC"),
                "likes": 3,
                "reposts": 1,
                "views": 90,
                "author_handle": "u1",
                "author_followers": 400,
                "sentiment_score": 0.4,
                "sentiment_label": "positive",
                "data_source": source,
            }
        ]
    )


def test_schema_is_created(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    counts = db.table_counts()
    assert set(counts) == {
        "prices",
        "x_posts",
        "sentiment_bars",
        "news_items",
        "forecasts",
        "evaluation_metrics",
    }
    assert all(value == 0 for value in counts.values())


def test_price_upsert_is_idempotent(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    assert db.upsert_prices(_prices(), "BTC-USD", "1d", "yfinance") == 2
    assert db.upsert_prices(_prices(), "BTC-USD", "1d", "yfinance") == 0
    assert db.table_counts()["prices"] == 2


def test_demo_and_live_rows_are_kept_separate(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.upsert_prices(_prices(), "BTC-USD", "1d", "yfinance")
    db.upsert_prices(_prices(), "BTC-USD", "1d", "synthetic")
    assert db.table_counts()["prices"] == 4
    assert len(db.load_prices("BTC-USD", "1d", "synthetic")) == 2
    assert len(db.load_prices("BTC-USD", "1d", "yfinance")) == 2


def test_load_prices_returns_utc_index(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.upsert_prices(_prices(), "BTC-USD", "1d", "yfinance")
    frame = db.load_prices("BTC-USD", "1d", "yfinance")
    assert frame.index.tz is not None
    assert list(frame.columns) == ["open", "high", "low", "close", "adj_close", "volume"]


def test_upsert_prices_validates_columns(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    with pytest.raises(ValueError, match="missing columns"):
        db.upsert_prices(pd.DataFrame({"timestamp": ["2024-01-01"]}), "BTC-USD", "1d", "yfinance")


def test_post_upsert_is_idempotent_and_keeps_sentiment(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    assert db.upsert_posts(_posts()) == 1
    assert db.upsert_posts(_posts()) == 0
    loaded = db.load_posts("BTC-USD", "apify")
    assert len(loaded) == 1
    assert loaded["sentiment_score"].iloc[0] == pytest.approx(0.4)
    assert loaded["posted_at"].iloc[0].tz is not None


def test_sentiment_bar_upsert_and_roundtrip(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    bars = pd.DataFrame(
        {
            "post_count": [3, 0],
            "mean_score": [0.3, np.nan],
            "pos_ratio": [0.7, np.nan],
            "neg_ratio": [0.1, np.nan],
            "neu_ratio": [0.2, np.nan],
            "weighted_score": [0.25, np.nan],
            "likes": [30, 0],
            "reposts": [4, 0],
            "views": [900, 0],
            "avg_followers": [500.0, np.nan],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    )
    assert db.upsert_sentiment_bars(bars, "BTC-USD", "1d", "apify") == 2
    assert db.upsert_sentiment_bars(bars, "BTC-USD", "1d", "apify") == 0
    loaded = db.load_sentiment_bars("BTC-USD", "1d", "apify")
    assert len(loaded) == 2
    assert loaded["mean_score"].iloc[0] == pytest.approx(0.3)
    assert np.isnan(loaded["mean_score"].iloc[1])


def test_news_deduplication_on_insert(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    news = pd.DataFrame(
        [
            {"news_id": "abc", "symbol": "BTC-USD", "source": "tavily", "title": "A", "url": "u1", "published_at": None, "summary": "s", "relevance": 0.5},
            {"news_id": "abc", "symbol": "BTC-USD", "source": "tavily", "title": "A duplicate", "url": "u1", "published_at": None, "summary": "s", "relevance": 0.5},
        ]
    )
    assert db.upsert_news(news) == 1
    assert db.table_counts()["news_items"] == 1


def test_forecast_and_metric_roundtrip_by_run(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    forecast = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "mode": "demo",
                "symbol": "BTC-USD",
                "interval": "1d",
                "model": "naive",
                "horizon": 1,
                "origin_ts": pd.Timestamp("2024-01-05", tz="UTC"),
                "target_ts": pd.Timestamp("2024-01-06", tz="UTC"),
                "base_price": 100.0,
                "predicted_return": 0.0,
                "predicted_price": 100.0,
                "actual_return": 0.01,
                "actual_price": 101.0,
            }
        ]
    )
    db.save_forecasts(forecast)
    db.save_forecasts(forecast)
    assert db.table_counts()["forecasts"] == 1
    assert db.latest_run_id("BTC-USD") == "run-1"
    assert len(db.load_forecasts(run_id="run-1")) == 1
    assert db.load_forecasts(run_id="other").empty

    metrics = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "mode": "demo",
                "symbol": "BTC-USD",
                "model": "naive",
                "horizon": 1,
                "metric_name": "mae",
                "metric_value": 1.0,
                "n_samples": 10,
                "notes": "price level",
            }
        ]
    )
    db.save_metrics(metrics)
    db.save_metrics(metrics)
    assert db.table_counts()["evaluation_metrics"] == 1
    assert float(db.load_metrics("run-1")["metric_value"].iloc[0]) == 1.0


def test_nan_values_are_stored_as_null(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    frame = _prices()
    frame.loc[0, "volume"] = np.nan
    db.upsert_prices(frame, "BTC-USD", "1d", "yfinance")
    loaded = db.load_prices("BTC-USD", "1d", "yfinance")
    assert pd.isna(loaded["volume"].iloc[0])


def test_empty_frames_are_safe(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    empty = pd.DataFrame()
    assert db.upsert_posts(empty) == 0
    assert db.upsert_news(empty) == 0
    assert db.save_forecasts(empty) == 0
    assert db.save_metrics(empty) == 0
    assert db.load_posts("BTC-USD", "apify").empty


def test_all_null_sentiment_metrics_load_back_as_float(tmp_path):
    """An empty window stores every metric as NULL; the read must not give object dtype."""
    db = Database(tmp_path / "test.sqlite3")
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True)
    bars = pd.DataFrame(
        {
            "post_count": [0, 0, 0],
            "mean_score": [np.nan] * 3,
            "pos_ratio": [np.nan] * 3,
            "neg_ratio": [np.nan] * 3,
            "neu_ratio": [np.nan] * 3,
            "weighted_score": [np.nan] * 3,
            "likes": [np.nan] * 3,
            "reposts": [np.nan] * 3,
            "views": [np.nan] * 3,
            "avg_followers": [np.nan] * 3,
        },
        index=index,
    )
    db.upsert_sentiment_bars(bars, "BTC-USD", "1d", "apify")
    loaded = db.load_sentiment_bars("BTC-USD", "1d", "apify")
    for column in ("mean_score", "likes", "views", "avg_followers"):
        assert loaded[column].dtype == "float64", column
    assert loaded["mean_score"].diff().isna().all()


def test_posts_with_null_engagement_load_back_as_float(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    posts = pd.DataFrame(
        {
            "post_id": ["p1"],
            "symbol": ["BTC-USD"],
            "search_query": ["$BTC"],
            "post_text": ["hello"],
            "posted_at": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "bar_ts": pd.to_datetime(["2024-01-01"], utc=True),
            "likes": [np.nan],
            "reposts": [np.nan],
            "views": [np.nan],
            "author_handle": ["someone"],
            "author_followers": [np.nan],
            "sentiment_score": [np.nan],
            "sentiment_label": [None],
            "data_source": ["apify"],
        }
    )
    db.upsert_posts(posts)
    loaded = db.load_posts("BTC-USD", "apify")
    assert loaded["likes"].dtype == "float64"
    assert loaded["sentiment_score"].dtype == "float64"
