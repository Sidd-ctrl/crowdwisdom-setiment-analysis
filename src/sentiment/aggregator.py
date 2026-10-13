"""Scoring posts and aggregating them onto the price bar grid.

Leakage rule enforced here: a bar's aggregate only contains posts that fall in
that bar's own window, i.e. posts published before the bar closes. Posts are
binned against the *actual* price bar starts (searchsorted) rather than a
calendar floor, so irregular session opens do not drop or shift records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from src.ingestion.timeutils import interval_timedelta
from src.sentiment.scoring import get_scorer

log = get_logger(__name__)

SENTIMENT_COLUMNS = [
    "post_count",
    "mean_score",
    "weighted_score",
    "pos_ratio",
    "neg_ratio",
    "neu_ratio",
    "likes",
    "reposts",
    "views",
    "avg_followers",
]


def score_posts(posts: pd.DataFrame, backend: str = "vader") -> pd.DataFrame:
    """Add sentiment_score / sentiment_label to every post."""
    if posts.empty:
        scored = posts.copy()
        scored["sentiment_score"] = pd.Series(dtype="float64")
        scored["sentiment_label"] = pd.Series(dtype="object")
        return scored

    scorer = get_scorer(backend)
    results = [scorer.score(text) for text in posts["post_text"].fillna("").tolist()]
    scored = posts.copy()
    scored["sentiment_score"] = [item["sentiment_score"] for item in results]
    scored["sentiment_label"] = [item["sentiment_label"] for item in results]
    log.info(
        "scored %d post(s) with %s: mean %.3f",
        len(scored),
        backend,
        float(scored["sentiment_score"].mean()),
    )
    return scored


def engagement_weight(row: pd.Series) -> float:
    """Reach-weighted importance of a post (log-compressed, always >= 0)."""
    likes = _zero(row.get("likes"))
    reposts = _zero(row.get("reposts"))
    views = _zero(row.get("views"))
    followers = _zero(row.get("author_followers"))
    return float(np.log1p(likes + 2.0 * reposts + views / 100.0 + followers / 1000.0))


def aggregate_bars(
    posts: pd.DataFrame, bar_index: pd.DatetimeIndex, interval: str
) -> pd.DataFrame:
    """Aggregate posts onto `bar_index` (price bar starts), one row per bar.

    Bars without posts get post_count = 0 and NaN scores, which the feature
    builder fills causally (forward fill from the last bar that had posts).
    """
    frame = pd.DataFrame(index=pd.DatetimeIndex(bar_index))
    frame.index.name = "bar_ts"
    if posts.empty or len(bar_index) == 0:
        for column in SENTIMENT_COLUMNS:
            frame[column] = np.nan
        frame["post_count"] = 0
        return frame

    work = posts.copy()
    work["posted_at"] = pd.to_datetime(work["posted_at"], utc=True, errors="coerce")
    work = work.dropna(subset=["posted_at"]).sort_values("posted_at")
    if work.empty:
        for column in SENTIMENT_COLUMNS:
            frame[column] = np.nan
        frame["post_count"] = 0
        return frame

    # Bar containing each post = last bar start that is not after the post time.
    positions = np.searchsorted(bar_index.values, work["posted_at"].values, side="right") - 1
    work = work[(positions >= 0)].copy()
    work["_bar"] = bar_index[positions[positions >= 0]]

    # Posts arriving after the final bar close belong to no bar yet.
    last_close = bar_index[-1] + interval_timedelta(interval)
    work = work[work["posted_at"] < last_close]
    dropped = len(posts) - len(work)
    if dropped > 0:
        log.warning("excluded %d post(s) outside the price bar grid", dropped)

    work["_weight"] = work.apply(engagement_weight, axis=1)
    work["_weighted"] = work["sentiment_score"] * work["_weight"]
    work["_label"] = work["sentiment_label"].fillna("neutral")

    grouped = work.groupby("_bar")
    counts = grouped.size()
    totals = grouped[["likes", "reposts", "views"]].sum(min_count=1)
    followers = grouped["author_followers"].mean()
    mean_score = grouped["sentiment_score"].mean()
    weighted_sum = grouped["_weighted"].sum()
    weight_total = grouped["_weight"].sum()
    labels = grouped["_label"].value_counts().unstack(fill_value=0)

    frame["post_count"] = counts.reindex(bar_index).fillna(0).astype(int)
    frame["mean_score"] = mean_score.reindex(bar_index)
    frame["weighted_score"] = (weighted_sum / weight_total.replace(0, np.nan)).reindex(bar_index)
    for label, column in (("positive", "pos_ratio"), ("negative", "neg_ratio"), ("neutral", "neu_ratio")):
        series = labels[label] if label in labels.columns else pd.Series(0, index=labels.index)
        frame[column] = (series / counts).reindex(bar_index)
    frame["likes"] = totals.get("likes").reindex(bar_index)
    frame["reposts"] = totals.get("reposts").reindex(bar_index)
    frame["views"] = totals.get("views").reindex(bar_index)
    frame["avg_followers"] = followers.reindex(bar_index)

    covered = int((frame["post_count"] > 0).sum())
    log.info(
        "aggregated %d post(s) onto %d bar(s); %d bar(s) have at least one post",
        int(frame["post_count"].sum()),
        len(frame),
        covered,
    )
    return frame[SENTIMENT_COLUMNS]


def _zero(value) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(number) else number
