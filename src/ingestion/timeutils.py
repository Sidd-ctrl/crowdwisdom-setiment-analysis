"""Timestamp normalisation and price-bar alignment.

X posts arrive with heterogeneous timestamp formats (ISO-8601, RFC-822, unix
seconds or milliseconds). Everything is converted to tz-aware UTC before it is
stored, and posts are snapped to the bar that *contains* them so that a bar's
sentiment aggregate only ever uses posts published before that bar closes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

# yfinance interval strings -> pandas offsets. Fixed-frequency intervals use
# floor(); calendar intervals (month) and weekly bars are handled explicitly.
PANDAS_OFFSETS = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "90m": "90min",
    "1h": "1h",
    "1d": "1D",
    "5d": "5D",
    "1wk": "1W",
    "1mo": "1MS",
}


def interval_offset(interval: str):
    if interval not in PANDAS_OFFSETS:
        raise ValueError(f"unsupported bar interval '{interval}'. Choose from {sorted(PANDAS_OFFSETS)}.")
    return pd.tseries.frequencies.to_offset(PANDAS_OFFSETS[interval])


def interval_timedelta(interval: str) -> pd.Timedelta:
    """Bar length as a Timedelta (calendar intervals approximated by their span)."""
    if interval == "1mo":
        return pd.Timedelta(days=30)
    return pd.Timedelta(interval_offset(interval))


def parse_utc_timestamp(value) -> pd.Timestamp | None:
    """Convert almost any timestamp representation into tz-aware UTC.

    Returns None when the value cannot be interpreted, so callers can drop or
    flag the record instead of silently coercing it to NaT.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        stamp = value
    elif isinstance(value, datetime):
        stamp = pd.Timestamp(value)
    elif isinstance(value, (int, float)):
        stamp = _from_epoch(float(value))
        if stamp is None:
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit() or _is_numeric_string(text):
            stamp = _from_epoch(float(text))
            if stamp is None:
                return None
        else:
            try:
                stamp = pd.to_datetime(text, utc=True, errors="raise")
            except (ValueError, TypeError):
                return None
    else:
        return None

    if stamp is None or pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _is_numeric_string(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _from_epoch(seconds: float) -> pd.Timestamp | None:
    """Interpret an epoch value as seconds or milliseconds."""
    if seconds <= 0:
        return None
    # Values above ~1e12 are millisecond epochs (year 33658 in seconds).
    if seconds > 1e11:
        seconds = seconds / 1000.0
    try:
        return pd.Timestamp(datetime.fromtimestamp(seconds, tz=timezone.utc))
    except (OverflowError, OSError, ValueError):
        return None


def bar_floor(timestamp: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Start of the bar that contains `timestamp` (UTC preserved)."""
    stamp = parse_utc_timestamp(timestamp)
    if stamp is None:
        raise ValueError("cannot align an unparseable timestamp to a bar")
    if interval == "1mo":
        return stamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if interval == "1wk":
        return (stamp - pd.Timedelta(days=int(stamp.weekday()))).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    return stamp.floor(interval_offset(interval))


def bar_close(timestamp: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Close of the bar that contains `timestamp`."""
    start = bar_floor(timestamp, interval)
    if interval == "1mo":
        return (start + pd.DateOffset(months=1)) - pd.Timedelta(seconds=1)
    return start + interval_timedelta(interval) - pd.Timedelta(seconds=1)


def build_bar_index(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    """Continuous bar-start index covering [start, end]."""
    first = bar_floor(start, interval)
    last = bar_floor(end, interval)
    if interval == "1mo":
        stamps = pd.date_range(first, last, freq="MS", tz="UTC")
    elif interval == "1wk":
        stamps = pd.date_range(first, last, freq="W-MON", tz="UTC")
    else:
        stamps = pd.date_range(first, last, freq=interval_offset(interval), tz="UTC")
    return pd.DatetimeIndex(stamps)


def periods_per_year(interval: str) -> int:
    """Trading periods per year, used only as a scaling reference in reports."""
    total_seconds = interval_timedelta(interval).total_seconds()
    return max(1, int(round(365 * 24 * 3600 / total_seconds)))
