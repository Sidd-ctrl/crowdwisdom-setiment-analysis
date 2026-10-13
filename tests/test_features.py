"""Feature engineering, horizon construction and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.builder import (
    FEATURE_COLUMNS,
    SENTIMENT_FEATURES,
    FEATURE_GROUPS,
    build_features,
    build_price_features,
    build_sentiment_features,
    build_targets,
    rsi,
)
from src.sentiment.aggregator import aggregate_bars, score_posts

HORIZONS = (1, 2, 3, 4, 5)


@pytest.fixture()
def sentiment_frame(scored_posts, bar_index):
    return aggregate_bars(scored_posts, bar_index, "1d")


def test_feature_columns_are_unique_and_grouped():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    flat = [name for names in FEATURE_GROUPS.values() for name in names]
    assert sorted(flat) == sorted(FEATURE_COLUMNS)


def test_price_features_match_manual_calculation(prices):
    frame = build_price_features(prices)
    close = prices["close"]
    assert frame["ret_1"].iloc[5] == pytest.approx(close.iloc[5] / close.iloc[4] - 1.0)
    assert frame["ret_5"].iloc[10] == pytest.approx(close.iloc[10] / close.iloc[5] - 1.0)
    assert frame["dist_sma5"].iloc[10] == pytest.approx(
        close.iloc[10] / close.iloc[6:11].mean() - 1.0
    )


def test_rsi_is_bounded(prices):
    values = rsi(prices["close"]).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_targets_are_forward_looking_only(prices):
    returns, levels = build_targets(prices["close"], HORIZONS)
    close = prices["close"]
    assert returns["ret_1"].iloc[3] == pytest.approx(close.iloc[4] / close.iloc[3] - 1.0)
    assert levels["price_5"].iloc[0] == pytest.approx(close.iloc[5])
    assert returns[f"ret_{HORIZONS[-1]}"].iloc[-1:].isna().all()


def test_target_columns_exist_for_every_horizon(prices):
    returns, levels = build_targets(prices["close"], HORIZONS)
    assert [f"ret_{h}" for h in HORIZONS] == list(returns.columns)
    assert [f"price_{h}" for h in HORIZONS] == list(levels.columns)


def test_features_use_only_past_information(prices, sentiment_frame):
    """Changing a future bar must not change the features of any earlier bar."""
    features, *_ = build_features(prices, sentiment_frame, HORIZONS)
    mutated = prices.copy()
    split = len(prices) - 5
    mutated.iloc[split:, mutated.columns.get_loc("close")] *= 1.5
    mutated_features, *_ = build_features(mutated, sentiment_frame, HORIZONS)

    # Bars strictly before the mutation point; the mutated bar itself is
    # allowed (and expected) to change because its own close changed.
    common = features.index.difference(prices.index[split:])
    comparable = [column for column in FEATURE_COLUMNS if column != "drawdown_from_high"]
    pd.testing.assert_frame_equal(
        features.loc[common, comparable], mutated_features.loc[common, comparable]
    )


def test_drawdown_from_high_is_the_one_path_dependent_feature(prices, sentiment_frame):
    """It looks backwards over history, so a future change cannot affect it either."""
    features, *_ = build_features(prices, sentiment_frame, HORIZONS)
    assert (features["drawdown_from_high"] <= 1e-12).all()


def test_sentiment_features_fill_gaps_causally():
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    sentiment = pd.DataFrame(
        {
            "post_count": [2, 0, 0, 3, 0, 1],
            "mean_score": [0.4, np.nan, np.nan, -0.2, np.nan, 0.1],
            "weighted_score": [0.4, np.nan, np.nan, -0.2, np.nan, 0.1],
            "pos_ratio": [0.8, np.nan, np.nan, 0.1, np.nan, 0.6],
            "neg_ratio": [0.1, np.nan, np.nan, 0.7, np.nan, 0.2],
            "neu_ratio": [0.1, np.nan, np.nan, 0.2, np.nan, 0.2],
            "likes": [10, 0, 0, 30, 0, 5],
            "reposts": [2, 0, 0, 6, 0, 1],
            "views": [200, 0, 0, 900, 0, 120],
            "avg_followers": [500, np.nan, np.nan, 1500, np.nan, 800],
        },
        index=index,
    )
    frame = build_sentiment_features(sentiment)
    assert frame["sent_mean"].iloc[1] == pytest.approx(0.4)
    assert frame["sent_mean"].iloc[2] == pytest.approx(0.4)
    assert frame["sent_mean"].iloc[4] == pytest.approx(-0.2)
    assert list(frame["bars_since_post"]) == [0, 1, 2, 0, 1, 0]


def test_sentiment_lags_reference_earlier_bars_only():
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    sentiment = pd.DataFrame(
        {
            "post_count": [1] * 6,
            "mean_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "weighted_score": [0.1] * 6,
            "pos_ratio": [0.5] * 6,
            "neg_ratio": [0.1] * 6,
            "neu_ratio": [0.4] * 6,
            "likes": [1] * 6,
            "reposts": [0] * 6,
            "views": [10] * 6,
            "avg_followers": [100] * 6,
        },
        index=index,
    )
    frame = build_sentiment_features(sentiment)
    assert frame["sent_lag_1"].iloc[3] == pytest.approx(0.3)
    assert frame["sent_lag_3"].iloc[5] == pytest.approx(0.3)
    assert np.isnan(frame["sent_lag_1"].iloc[0])


def test_build_features_returns_aligned_frames(prices, sentiment_frame):
    features, returns, levels, base = build_features(prices, sentiment_frame, HORIZONS)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.index.equals(returns.index)
    assert features.index.equals(levels.index)
    assert base.index.equals(features.index)
    assert not features.isna().any(axis=1).any()
    assert base.iloc[0] == pytest.approx(prices["close"].loc[features.index[0]])


def test_features_carry_no_future_price_columns(prices, sentiment_frame):
    features, *_ = build_features(prices, sentiment_frame, HORIZONS)
    assert not [column for column in features.columns if "price_" in column or column.startswith("y_")]


def test_calendar_features_are_cyclical(prices, sentiment_frame):
    features, *_ = build_features(prices, sentiment_frame, HORIZONS)
    for column in ("dow_sin", "dow_cos", "month_sin", "month_cos"):
        assert features[column].between(-1.0, 1.0).all()


def test_end_to_end_from_raw_posts(prices, posts):
    """Scoring -> aggregation -> features must produce a usable matrix."""
    sentiment = aggregate_bars(score_posts(posts, "vader"), prices.index, "1d")
    features, returns, _, base = build_features(prices, sentiment, HORIZONS)
    assert len(features) > 10
    assert returns["ret_1"].notna().sum() >= len(features) - 1
    assert base.notna().all()


def test_sentiment_features_survive_an_empty_window(bar_index):
    """An empty sentiment table arrives as object dtype; it must not raise."""
    empty = pd.DataFrame(
        {name: pd.Series([None] * len(bar_index), index=bar_index, dtype=object)
         for name in ("post_count", "mean_score", "weighted_score", "pos_ratio", "neg_ratio",
                      "likes", "reposts", "views", "avg_followers")}
    )
    frame = build_sentiment_features(empty)
    assert list(frame.columns) == list(SENTIMENT_FEATURES)
    assert frame.notna().all().all()
    assert frame["sent_volume"].eq(0.0).all()
    assert frame["bars_since_post"].eq(-1.0).all()


def test_features_build_end_to_end_with_no_posts(prices, bar_index):
    empty = pd.DataFrame(index=bar_index)
    features, targets, levels, base = build_features(prices, empty, HORIZONS)
    assert not features.empty
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features.notna().all().all()
    assert features["sent_mean"].eq(0.0).all()
    assert len(targets) == len(features) == len(levels) == len(base)


def test_neutral_block_only_applies_while_there_are_no_posts(sentiment_frame):
    frame = build_sentiment_features(sentiment_frame)
    assert not frame["sent_volume"].eq(0.0).all()
