"""Historical OHLCV ingestion through yfinance."""

from __future__ import annotations

import pandas as pd

from config.logging_config import get_logger
from src.storage.database import SOURCE_YFINANCE

log = get_logger(__name__)

# yfinance only serves limited intraday history; requesting more returns an
# empty frame, so the window is capped and the reduction is reported.
INTRADAY_CAPS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "90m": 60, "1h": 730}

COLUMN_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "volume": "volume",
}


class PriceIngestionError(RuntimeError):
    """Raised when no usable price history could be downloaded."""


def fetch_prices(symbol: str, period_days: int, interval: str) -> pd.DataFrame:
    """Download and normalise OHLCV history for `symbol`."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PriceIngestionError("yfinance is not installed; run pip install -r requirements.txt") from exc

    effective_days = period_days
    if interval in INTRADAY_CAPS:
        cap = INTRADAY_CAPS[interval]
        if period_days > cap:
            log.warning(
                "yfinance only serves %d days of %s history; requesting %d instead of %d.",
                cap,
                interval,
                cap,
                period_days,
            )
            effective_days = cap

    log.info("yfinance: downloading %s (%s, %d days)", symbol, interval, effective_days)
    try:
        raw = yf.download(
            tickers=symbol,
            period=f"{effective_days + 10}d",
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # yfinance raises a variety of network/parse errors
        raise PriceIngestionError(f"yfinance download failed for {symbol}: {exc}") from exc

    if raw is None or raw.empty:
        raise PriceIngestionError(
            f"yfinance returned no rows for '{symbol}' at interval '{interval}'. "
            "Check the ticker symbol, or the network connection."
        )

    frame = _flatten_columns(raw)
    frame.index = _to_utc_index(frame.index)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    if "close" not in frame.columns:
        raise PriceIngestionError(f"yfinance response for {symbol} has no close column: {list(frame.columns)}")

    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    dropped = int(frame["close"].isna().sum())
    if dropped:
        log.warning("dropping %d bar(s) with a missing close for %s", dropped, symbol)
        frame = frame.dropna(subset=["close"])
    if frame.empty:
        raise PriceIngestionError(f"all {symbol} rows had a missing close.")

    for column in ("open", "high", "low", "volume", "adj_close"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    frame.index.name = "timestamp"
    log.info(
        "yfinance: %d bar(s) for %s from %s to %s",
        len(frame),
        symbol,
        frame.index.min(),
        frame.index.max(),
    )
    return frame[["open", "high", "low", "close", "adj_close", "volume"]].reset_index()


def _flatten_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance column naming (single- and multi-index variants)."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.rename(columns=COLUMN_MAP)
    keep = [column for column in ("open", "high", "low", "close", "adj_close", "volume") if column in frame.columns]
    return frame[keep]


def _to_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    stamp = pd.to_datetime(index)
    if stamp.tz is None:
        # Daily bars are exchange-local calendar dates with no intraday meaning;
        # they are treated as UTC day boundaries for consistent joins with posts.
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def price_source(demo_mode: bool) -> str:
    return "synthetic" if demo_mode else SOURCE_YFINANCE
