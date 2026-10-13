"""Synthetic demo data: determinism, shape and explicit labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.synthetic import generate_synthetic_posts, generate_synthetic_prices


def test_synthetic_prices_are_deterministic():
    first = generate_synthetic_prices("BTC-USD", 90, "1d", seed=11)
    second = generate_synthetic_prices("BTC-USD", 90, "1d", seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_synthetic_prices_have_the_expected_shape():
    frame = generate_synthetic_prices("BTC-USD", 90, "1d", seed=3)
    assert len(frame) == 90
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]
    assert frame["close"].notna().all()
    assert (frame["close"] > 0).all()
    assert (frame["high"] >= frame["low"]).all()
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["timestamp"].dt.tz is not None


def test_different_seeds_produce_different_paths():
    a = generate_synthetic_prices("BTC-USD", 60, "1d", seed=1)
    b = generate_synthetic_prices("BTC-USD", 60, "1d", seed=2)
    assert not np.allclose(a["close"], b["close"])


def test_synthetic_posts_are_labelled_synthetic():
    prices = generate_synthetic_prices("BTC-USD", 30, "1d", seed=5)
    posts = generate_synthetic_posts(prices, "BTC-USD", "1d", seed=5)
    assert not posts.empty
    assert set(posts["data_source"]) == {"synthetic"}
    assert posts["search_query"].str.contains("synthetic").all()
    assert posts["post_id"].str.startswith("syn-").all()
    assert posts["post_id"].is_unique


def test_synthetic_posts_stay_inside_their_bar():
    prices = generate_synthetic_prices("BTC-USD", 20, "1d", seed=9)
    posts = generate_synthetic_posts(prices, "BTC-USD", "1d", seed=9)
    assert (posts["posted_at"] >= posts["bar_ts"]).all()
    assert (posts["posted_at"] < posts["bar_ts"] + pd.Timedelta(days=1)).all()
    assert set(posts["bar_ts"]).issubset(set(prices["timestamp"]))


def test_synthetic_posts_are_deterministic_and_engagement_is_sane():
    prices = generate_synthetic_prices("BTC-USD", 30, "1d", seed=4)
    first = generate_synthetic_posts(prices, "BTC-USD", "1d", seed=4)
    second = generate_synthetic_posts(prices, "BTC-USD", "1d", seed=4)
    pd.testing.assert_frame_equal(first, second)
    assert (first["likes"] >= 0).all()
    assert (first["views"] >= first["likes"]).all()
    assert (first["author_followers"] > 0).all()


def test_synthetic_posts_react_to_past_returns_only():
    """Sentiment must not encode a future return, or demo metrics would be fake."""
    prices = generate_synthetic_prices("BTC-USD", 40, "1d", seed=6)
    posts = generate_synthetic_posts(prices, "BTC-USD", "1d", seed=6)
    tone = posts.groupby("bar_ts")["post_text"].apply(
        lambda texts: texts.str.contains("breaking out|pumping|strong").mean()
    )
    close = prices.set_index("timestamp")["close"]
    past_return = close.pct_change().shift(1)
    future_return = close.pct_change().shift(-1)
    aligned_tone = tone.reindex(close.index)
    past_corr = aligned_tone.corr(past_return)
    future_corr = aligned_tone.corr(future_return)
    assert past_corr > future_corr
    assert future_corr < 0.5


def test_synthetic_pipeline_produces_a_usable_matrix():
    from src.features.builder import build_features
    from src.sentiment.aggregator import aggregate_bars, score_posts

    prices = generate_synthetic_prices("BTC-USD", 60, "1d", seed=8)
    posts = generate_synthetic_posts(prices, "BTC-USD", "1d", seed=8)
    index = pd.DatetimeIndex(prices["timestamp"])
    sentiment = aggregate_bars(score_posts(posts, "vader"), index, "1d")
    features, returns, _, base = build_features(
        prices.set_index("timestamp"), sentiment, (1, 2, 3)
    )
    assert len(features) > 20
    assert features.notna().all().all()
    assert base.notna().all()
    assert returns["ret_1"].notna().any()
