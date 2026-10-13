"""Pytest configuration: make the project root importable and share fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def bar_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")


@pytest.fixture()
def prices(bar_index) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=len(bar_index)))
    return pd.DataFrame(
        {
            "open": close * (1 - rng.normal(0, 0.002, len(bar_index))),
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            "adj_close": close,
            "volume": rng.lognormal(10, 0.3, len(bar_index)),
        },
        index=bar_index,
    )


@pytest.fixture()
def posts(bar_index) -> pd.DataFrame:
    rows = []
    for position, bar in enumerate(bar_index):
        for index in range(3):
            tone = "great move up" if position % 2 == 0 else "terrible selloff"
            rows.append(
                {
                    "post_id": f"post-{position}-{index}",
                    "symbol": "BTC-USD",
                    "search_query": "$BTC",
                    "post_text": f"BTC {tone}, position {position}",
                    "posted_at": bar + pd.Timedelta(hours=index + 1),
                    "likes": 10 + index,
                    "reposts": 2 + index,
                    "views": 500 + 100 * index,
                    "author_handle": f"user{index}",
                    "author_followers": 1000 * (index + 1),
                    "data_source": "apify",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def scored_posts(posts) -> pd.DataFrame:
    from src.sentiment.aggregator import score_posts

    return score_posts(posts, "vader")


@pytest.fixture()
def settings(tmp_path):
    from config.settings import AppSettings

    return AppSettings(
        asset_symbol="BTC-USD",
        period_days=180,
        bar_interval="1d",
        demo_mode=True,
        models=("naive", "random_forest"),
        horizons=(1, 2, 3),
        min_train_samples=20,
        refit_every=3,
        random_seed=42,
        db_path=tmp_path / "test.sqlite3",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        forecast_dir=tmp_path / "forecasts",
        chart_dir=tmp_path / "charts",
        report_dir=tmp_path / "reports",
    )
