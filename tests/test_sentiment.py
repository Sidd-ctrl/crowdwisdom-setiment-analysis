"""Sentiment scoring and bar aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sentiment.aggregator import aggregate_bars, engagement_weight, score_posts
from src.sentiment.scoring import (
    FinBertScorer,
    VaderScorer,
    finance_adjustment,
    get_scorer,
    label_from_score,
)


def test_vader_scores_positive_and_negative_text():
    scorer = VaderScorer()
    positive = scorer.score("BTC is breaking out strongly, great momentum, buyers in control")
    negative = scorer.score("BTC is crashing, terrible selloff, complete disaster")
    assert positive["sentiment_score"] > 0.2
    assert negative["sentiment_score"] < -0.2
    assert positive["sentiment_label"] == "positive"
    assert negative["sentiment_label"] == "negative"


def test_scores_are_bounded():
    scorer = VaderScorer()
    for text in ("", "bullish moon pump rally", "crash dump rug liquidation"):
        score = scorer.score(text)["sentiment_score"]
        assert -1.0 <= score <= 1.0


def test_finance_adjustment_is_capped():
    text = "bullish moon pump rally breakout ath accumulating"
    adjustment = finance_adjustment(text)
    assert 0 < adjustment <= 0.25


def test_label_thresholds():
    assert label_from_score(0.05) == "positive"
    assert label_from_score(-0.05) == "negative"
    assert label_from_score(0.0) == "neutral"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_scorer("magic-model")


def test_finbert_backend_reports_missing_dependency(monkeypatch):
    """FinBERT must fail loudly with install instructions, not silently."""
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == "transformers":
            raise ImportError("transformers is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="transformers"):
        FinBertScorer()


def test_score_posts_adds_columns(scored_posts):
    assert {"sentiment_score", "sentiment_label"} <= set(scored_posts.columns)
    assert scored_posts["sentiment_score"].notna().all()
    assert len(scored_posts) == len(scored_posts.drop_duplicates("post_id"))


def test_aggregate_bars_counts_and_scores(scored_posts, bar_index):
    frame = aggregate_bars(scored_posts, bar_index, "1d")
    assert len(frame) == len(bar_index)
    assert (frame["post_count"] == 3).all()
    assert frame["mean_score"].notna().all()
    assert frame[["pos_ratio", "neg_ratio", "neu_ratio"]].sum(axis=1).sub(1.0).abs().max() < 1e-9


def test_aggregate_bars_leaves_empty_bars_nan(bar_index):
    subset = pd.DataFrame(
        [
            {
                "post_id": "only",
                "posted_at": bar_index[0] + pd.Timedelta(hours=2),
                "post_text": "great",
                "sentiment_score": 0.5,
                "sentiment_label": "positive",
                "likes": 5,
                "reposts": 1,
                "views": 100,
                "author_followers": 200,
            }
        ]
    )
    frame = aggregate_bars(subset, bar_index, "1d")
    assert int(frame["post_count"].iloc[0]) == 1
    assert frame["mean_score"].iloc[0] == pytest.approx(0.5)
    assert frame["mean_score"].iloc[1:].isna().all()
    assert (frame["post_count"].iloc[1:] == 0).all()


def test_aggregate_bars_excludes_posts_after_final_bar_close(bar_index):
    late = bar_index[-1] + pd.Timedelta(days=1, hours=3)
    subset = pd.DataFrame(
        [
            {
                "post_id": "late",
                "posted_at": late,
                "post_text": "too late",
                "sentiment_score": 0.9,
                "sentiment_label": "positive",
                "likes": 1,
                "reposts": 0,
                "views": 10,
                "author_followers": 10,
            }
        ]
    )
    frame = aggregate_bars(subset, bar_index, "1d")
    assert int(frame["post_count"].sum()) == 0


def test_aggregate_bars_bins_against_price_bar_starts(bar_index):
    """Irregular session opens must not shift posts into the wrong bar."""
    odd_index = pd.DatetimeIndex(
        [bar + pd.Timedelta(hours=9, minutes=30) for bar in bar_index[:5]]
    )
    subset = pd.DataFrame(
        [
            {
                "post_id": "p",
                "posted_at": odd_index[1] + pd.Timedelta(minutes=20),
                "post_text": "hi",
                "sentiment_score": 0.3,
                "sentiment_label": "positive",
                "likes": 1,
                "reposts": 0,
                "views": 1,
                "author_followers": 1,
            }
        ]
    )
    frame = aggregate_bars(subset, odd_index, "1d")
    assert int(frame["post_count"].iloc[1]) == 1
    assert int(frame["post_count"].sum()) == 1


def test_aggregate_bars_handles_empty_posts(bar_index):
    frame = aggregate_bars(pd.DataFrame(), bar_index, "1d")
    assert (frame["post_count"] == 0).all()
    assert frame["mean_score"].isna().all()


def test_engagement_weight_is_non_negative_and_monotonic():
    small = engagement_weight(pd.Series({"likes": 1, "reposts": 0, "views": 10, "author_followers": 10}))
    large = engagement_weight(pd.Series({"likes": 1000, "reposts": 400, "views": 90000, "author_followers": 50000}))
    assert small >= 0
    assert large > small
    assert engagement_weight(pd.Series({"likes": np.nan, "reposts": None, "views": None, "author_followers": None})) == 0.0
