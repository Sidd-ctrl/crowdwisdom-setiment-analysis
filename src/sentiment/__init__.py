"""Sentiment package: post scoring and bar-level aggregation."""

from src.sentiment.aggregator import SENTIMENT_COLUMNS, aggregate_bars, engagement_weight, score_posts
from src.sentiment.scoring import (
    FinBertScorer,
    VaderScorer,
    finance_adjustment,
    get_scorer,
    label_from_score,
)

__all__ = [
    "SENTIMENT_COLUMNS",
    "aggregate_bars",
    "engagement_weight",
    "score_posts",
    "FinBertScorer",
    "VaderScorer",
    "finance_adjustment",
    "get_scorer",
    "label_from_score",
]
