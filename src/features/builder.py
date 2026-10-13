"""Feature engineering and forecast-target construction.

Point-in-time rule: the feature row for bar t may only use information that was
known at the close of bar t -- its own OHLCV and the sentiment of posts
published inside that bar. Forward-looking columns appear only in the target
frame (`close.shift(-k)`), never in the feature frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logging_config import get_logger

log = get_logger(__name__)

PRICE_FEATURES = (
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "log_ret_1",
    "vol_5",
    "vol_10",
    "vol_20",
    "dist_sma5",
    "dist_sma20",
    "sma_spread",
    "range_ratio",
    "volume_ratio",
    "rsi_14",
    "drawdown_from_high",
)

SENTIMENT_FEATURES = (
    "sent_mean",
    "sent_weighted",
    "sent_net",
    "sent_pos_ratio",
    "sent_neg_ratio",
    "sent_volume",
    "sent_engagement",
    "sent_followers",
    "sent_change_1",
    "sent_change_3",
    "sent_momentum",
    "sent_lag_1",
    "sent_lag_2",
    "sent_lag_3",
    "bars_since_post",
)

CALENDAR_FEATURES = ("dow_sin", "dow_cos", "month_sin", "month_cos")

FEATURE_COLUMNS = list(PRICE_FEATURES) + list(SENTIMENT_FEATURES) + list(CALENDAR_FEATURES)

FEATURE_GROUPS = {
    "price": list(PRICE_FEATURES),
    "sentiment": list(SENTIMENT_FEATURES),
    "calendar": list(CALENDAR_FEATURES),
}

# Definitions are rendered into the evaluation report and mirrored in the README.
FEATURE_DEFINITIONS = {
    "ret_1": "1-bar simple return of the close",
    "ret_2": "2-bar simple return of the close",
    "ret_3": "3-bar simple return of the close",
    "ret_5": "5-bar simple return of the close",
    "log_ret_1": "1-bar log return",
    "vol_5": "rolling 5-bar standard deviation of ret_1",
    "vol_10": "rolling 10-bar standard deviation of ret_1",
    "vol_20": "rolling 20-bar standard deviation of ret_1",
    "dist_sma5": "close / 5-bar SMA − 1",
    "dist_sma20": "close / 20-bar SMA − 1",
    "sma_spread": "5-bar SMA / 20-bar SMA − 1",
    "range_ratio": "(high − low) / close for the bar",
    "volume_ratio": "bar volume / 20-bar mean volume",
    "rsi_14": "Wilder RSI(14) divided by 100",
    "drawdown_from_high": "close / running maximum close − 1",
    "sent_mean": "mean VADER compound score of posts inside the bar (forward-filled when a bar has no posts)",
    "sent_weighted": "engagement-weighted mean sentiment: Σ(score·w)/Σw with w = log1p(likes + 2·reposts + views/100 + followers/1000)",
    "sent_net": "pos_ratio − neg_ratio for the bar",
    "sent_pos_ratio": "share of posts with compound ≥ 0.05",
    "sent_neg_ratio": "share of posts with compound ≤ −0.05",
    "sent_volume": "log1p(post_count) for the bar",
    "sent_engagement": "log1p(likes + 2·reposts + views/100) summed over the bar",
    "sent_followers": "log1p(mean author followers of the bar's posts)",
    "sent_change_1": "sent_mean − sent_mean[1 bar ago]",
    "sent_change_3": "sent_mean − sent_mean[3 bars ago]",
    "sent_momentum": "sent_mean − 7-bar rolling mean of sent_mean",
    "sent_lag_1": "sent_mean lagged 1 bar",
    "sent_lag_2": "sent_mean lagged 2 bars",
    "sent_lag_3": "sent_mean lagged 3 bars",
    "bars_since_post": "bars elapsed since the last bar that contained a post",
    "dow_sin": "sin(2π · day-of-week / 7)",
    "dow_cos": "cos(2π · day-of-week / 7)",
    "month_sin": "sin(2π · (month − 1) / 12)",
    "month_cos": "cos(2π · (month − 1) / 12)",
}


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    ratio = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=prices.index)
    close = prices["close"]
    frame["ret_1"] = close.pct_change(1)
    frame["ret_2"] = close.pct_change(2)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_5"] = close.pct_change(5)
    frame["log_ret_1"] = np.log(close / close.shift(1))
    frame["vol_5"] = frame["ret_1"].rolling(5).std()
    frame["vol_10"] = frame["ret_1"].rolling(10).std()
    frame["vol_20"] = frame["ret_1"].rolling(20).std()
    frame["dist_sma5"] = close / close.rolling(5).mean() - 1.0
    frame["dist_sma20"] = close / close.rolling(20).mean() - 1.0
    frame["sma_spread"] = close.rolling(5).mean() / close.rolling(20).mean() - 1.0
    span = (prices["high"] - prices["low"]) if {"high", "low"}.issubset(prices.columns) else pd.Series(0.0, index=prices.index)
    frame["range_ratio"] = span / close
    volume = prices["volume"] if "volume" in prices.columns else pd.Series(np.nan, index=prices.index)
    frame["volume_ratio"] = volume / volume.rolling(20).mean()
    frame["rsi_14"] = rsi(close) / 100.0
    frame["drawdown_from_high"] = close / close.cummax() - 1.0
    return frame[list(PRICE_FEATURES)]


def _as_float(column: pd.Series | None, index: pd.Index) -> pd.Series:
    """Coerce a stored sentiment column to float, mapping absent values to NaN.

    SQLite and pandas hand back object-dtype columns when a table is empty, which
    makes arithmetic such as .diff() raise instead of returning NaN.
    """
    if column is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(column, errors="coerce").astype("float64")


def build_sentiment_features(sentiment: pd.DataFrame) -> pd.DataFrame:
    index = sentiment.index

    def column(name: str) -> pd.Series:
        return _as_float(sentiment[name] if name in sentiment.columns else None, index)

    frame = pd.DataFrame(index=index)
    counts = column("post_count").fillna(0.0)
    has_posts = counts > 0.0

    # Bars with no posts carry the last observed score forward. Forward fill is
    # causal (it only ever looks backwards), unlike any centred or future fill.
    score = column("mean_score").ffill()
    weighted = column("weighted_score").ffill()
    pos = column("pos_ratio").ffill()
    neg = column("neg_ratio").ffill()

    frame["sent_mean"] = score
    frame["sent_weighted"] = weighted
    frame["sent_pos_ratio"] = pos
    frame["sent_neg_ratio"] = neg
    frame["sent_net"] = pos - neg
    frame["sent_volume"] = np.log1p(counts)
    engagement = (
        column("likes").fillna(0.0)
        + 2.0 * column("reposts").fillna(0.0)
        + column("views").fillna(0.0) / 100.0
    )
    frame["sent_engagement"] = np.log1p(engagement)
    frame["sent_followers"] = np.log1p(column("avg_followers").ffill().fillna(0.0))
    frame["sent_change_1"] = score.diff(1)
    frame["sent_change_3"] = score.diff(3)
    frame["sent_momentum"] = score - score.rolling(7, min_periods=1).mean()
    frame["sent_lag_1"] = score.shift(1)
    frame["sent_lag_2"] = score.shift(2)
    frame["sent_lag_3"] = score.shift(3)

    last_post_bar = np.where(has_posts.to_numpy(), np.arange(len(has_posts)), np.nan)
    last_seen = pd.Series(last_post_bar, index=has_posts.index).ffill()
    frame["bars_since_post"] = pd.Series(np.arange(len(has_posts)), index=has_posts.index) - last_seen

    if not has_posts.any():
        # No posts anywhere in the window: the sentiment block cannot say anything, so it is
        # pinned to a documented neutral value rather than left as NaN. NaN here would drop
        # every training row and turn "missing sentiment" into an empty model. `-1` marks
        # "no post observed in this window" for the recency feature.
        log.warning(
            "no posts in the window: sentiment features are pinned to neutral (0.0); "
            "any forecasts from this run are price-only"
        )
        for name in SENTIMENT_FEATURES:
            frame[name] = 0.0
        frame["bars_since_post"] = -1.0

    return frame[list(SENTIMENT_FEATURES)]


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    dow = index.dayofweek.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    frame["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    frame["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    return frame[list(CALENDAR_FEATURES)]


def build_targets(close: pd.Series, horizons: tuple[int, ...] | list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward returns and forward price levels, one column per horizon.

    `n+0` is not a forecast: the base price series (`close`) is the latest
    observed price and is reported alongside the model output for n+1..n+k.
    """
    returns = pd.DataFrame(index=close.index)
    levels = pd.DataFrame(index=close.index)
    for horizon in horizons:
        future = close.shift(-horizon)
        returns[f"ret_{horizon}"] = future / close - 1.0
        levels[f"price_{horizon}"] = future
    return returns, levels


def build_features(
    prices: pd.DataFrame, sentiment: pd.DataFrame, horizons: tuple[int, ...] | list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (features, target returns, target price levels, base prices)."""
    prices = prices.sort_index()
    features = build_price_features(prices)

    aligned = sentiment.reindex(prices.index)
    has_sentiment = "post_count" in aligned.columns and aligned["post_count"].notna().any()
    if not has_sentiment:
        log.warning("no sentiment bars aligned to the price grid; sentiment features will be empty")
    features = features.join(build_sentiment_features(aligned)).join(build_calendar_features(prices.index))

    targets, levels = build_targets(prices["close"], horizons)
    base = prices["close"].rename("base_price")

    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"feature builder did not produce: {missing}")

    warmup_rows = int(features[list(PRICE_FEATURES)].isna().any(axis=1).sum())
    usable = features.dropna(subset=["ret_1", "vol_20", "dist_sma20", "sma_spread", "rsi_14", "sent_mean"])
    log.info(
        "built %d feature(s) over %d bar(s); %d bar(s) dropped as warm-up/incomplete",
        len(FEATURE_COLUMNS),
        len(features),
        len(features) - len(usable),
    )
    if warmup_rows:
        log.debug("%d row(s) still contain NaN price features before the dropna step", warmup_rows)

    features = usable[FEATURE_COLUMNS]
    return features, targets.loc[features.index], levels.loc[features.index], base.loc[features.index]
