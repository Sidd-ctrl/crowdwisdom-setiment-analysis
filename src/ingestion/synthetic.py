"""Explicitly labelled synthetic data for DEMO_MODE.

Demo mode exists so the pipeline can be exercised end to end without API
credentials. The data is generated here from a fixed seed, and every record is
tagged `data_source = 'synthetic'` so it can never be mixed into, or mistaken
for, a live result. Post text is synthetic but sentiment scores are still
computed by the real VADER scorer in the sentiment stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from src.ingestion.timeutils import interval_timedelta

log = get_logger(__name__)

POSITIVE_TEXTS = (
    "{ticker} breaking out here, volume confirms the move, this looks strong",
    "long {ticker} into the close, momentum and order flow both improving",
    "great setup on {ticker}, holding support and reclaiming the moving average",
    "{ticker} pumping hard, buyers are in full control today",
    "impressive strength in {ticker}, best risk reward I have seen this month",
    "accumulating {ticker}, the trend is clearly working",
)

NEGATIVE_TEXTS = (
    "{ticker} is breaking down, this looks weak and sellers are aggressive",
    "stopped out of {ticker}, terrible price action and heavy selling",
    "{ticker} keeps failing at resistance, ugly chart and awful momentum",
    "dumping my {ticker} position, downside risk is far too high right now",
    "{ticker} looks broken, support failed and volume is drying up",
    "worst trade of the week was {ticker}, complete disaster",
)

NEUTRAL_TEXTS = (
    "watching {ticker} around this level, nothing decided yet",
    "{ticker} chopping sideways, waiting for a clearer range",
    "checking the {ticker} chart before the open, no position for now",
    "{ticker} volume is average today, still consolidating",
    "anyone else tracking {ticker}? just observing the structure",
    "{ticker} closed flat, nothing to report",
)


def generate_synthetic_prices(
    symbol: str, period_days: int, interval: str, seed: int = 42, start_price: float | None = None
) -> pd.DataFrame:
    """Geometric random walk with regime shifts and occasional jumps."""
    rng = np.random.default_rng(seed)
    bars_per_day = max(1, int(round(pd.Timedelta(days=1) / interval_timedelta(interval))))
    n_bars = int(period_days * bars_per_day)
    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - interval_timedelta(interval) * (n_bars - 1)
    stamps = pd.date_range(start, end, periods=n_bars, tz="UTC")

    price = start_price or (42_000.0 if symbol.upper().startswith("BTC") else 250.0)
    drift = rng.normal(0.0004, 0.0006, size=n_bars)
    vol = np.full(n_bars, 0.018)
    for change_at in rng.choice(n_bars, size=max(1, n_bars // 45), replace=False):
        length = min(n_bars - int(change_at), int(rng.integers(8, 30)))
        vol[int(change_at) : int(change_at) + length] *= float(rng.uniform(1.4, 2.4))
    shocks = rng.normal(0.0, 1.0, size=n_bars) * vol
    jumps = np.where(rng.random(n_bars) < 0.02, rng.normal(0.0, 0.045, size=n_bars), 0.0)
    returns = drift + shocks + jumps

    close = price * np.cumprod(1.0 + returns)
    intrabar = np.abs(rng.normal(0.0, vol, size=n_bars))
    open_ = close / (1.0 + returns)
    high = np.maximum(open_, close) * (1.0 + intrabar)
    low = np.minimum(open_, close) * (1.0 - intrabar)
    volume = rng.lognormal(mean=16.0, sigma=0.45, size=n_bars) * (1.0 + 6.0 * np.abs(returns))

    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume.round(2),
        }
    )
    log.warning(
        "generated %d SYNTHETIC bars for %s (demo mode only, not market data)",
        len(frame),
        symbol,
    )
    return frame


def generate_synthetic_posts(
    prices: pd.DataFrame, symbol: str, interval: str, seed: int = 42
) -> pd.DataFrame:
    """Posts per bar whose tone reacts to the *previous* bar return.

    Sentiment is driven by past returns plus idiosyncratic noise, which mirrors
    how crowd commentary actually behaves and keeps the demo free of look-ahead:
    no post references a return that has not happened yet.
    """
    rng = np.random.default_rng(seed + 7)
    ticker = symbol.split("-")[0]
    span = interval_timedelta(interval)
    rows: list[dict] = []

    closes = prices["close"].to_numpy()
    stamps = pd.DatetimeIndex(prices["timestamp"])
    prev_returns = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])

    for position, bar_start in enumerate(stamps):
        signal = float(np.clip(prev_returns[position] * 22.0, -1.6, 1.6))
        rate = float(np.clip(14.0 + 55.0 * abs(prev_returns[position]) ** 0.5, 4.0, 90.0))
        count = int(rng.poisson(rate))
        for index in range(count):
            draw = float(np.clip(rng.normal(signal, 0.85), -2.0, 2.0))
            if draw > 0.25:
                template, label = rng.choice(POSITIVE_TEXTS), "positive"
            elif draw < -0.25:
                template, label = rng.choice(NEGATIVE_TEXTS), "negative"
            else:
                template, label = rng.choice(NEUTRAL_TEXTS), "neutral"

            text = str(template).format(ticker=ticker)
            offset = span * float(rng.uniform(0.0, 0.98))
            followers = int(np.clip(rng.lognormal(mean=6.4, sigma=1.5), 12, 4_000_000))
            reach = float(np.clip(rng.lognormal(mean=3.1, sigma=1.2), 0.0, 40.0))
            likes = int(followers * reach * float(rng.uniform(0.0008, 0.012)))
            rows.append(
                {
                    "post_id": f"syn-{ticker}-{bar_start:%Y%m%d%H%M}-{index:04d}",
                    "symbol": symbol,
                    "search_query": f"${ticker} (synthetic)",
                    "post_text": text,
                    "posted_at": bar_start + offset,
                    "bar_ts": bar_start,
                    "likes": likes,
                    "reposts": int(likes * float(rng.uniform(0.05, 0.45))),
                    "views": int(max(likes * float(rng.uniform(8.0, 60.0)), likes)),
                    "author_handle": f"{ticker.lower()}_watcher_{rng.integers(100, 999)}",
                    "author_followers": followers,
                    "data_source": "synthetic",
                }
            )

    frame = pd.DataFrame(rows)
    log.warning(
        "generated %d SYNTHETIC posts for %s (demo mode only, not real X content)",
        len(frame),
        symbol,
    )
    return frame
