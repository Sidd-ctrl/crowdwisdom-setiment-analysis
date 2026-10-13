"""Timestamp normalisation and bar alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.timeutils import (
    bar_close,
    bar_floor,
    build_bar_index,
    interval_timedelta,
    parse_utc_timestamp,
    periods_per_year,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024-03-10T14:22:31Z", "2024-03-10 14:22:31+00:00"),
        ("2024-03-10T14:22:31+02:00", "2024-03-10 12:22:31+00:00"),
        ("2024-03-10 14:22:31", "2024-03-10 14:22:31+00:00"),
        (1710080551, "2024-03-10 14:22:31+00:00"),
        (1710080551000, "2024-03-10 14:22:31+00:00"),
        ("1710080551", "2024-03-10 14:22:31+00:00"),
    ],
)
def test_parse_utc_timestamp_formats(value, expected):
    assert str(parse_utc_timestamp(value)) == expected


def test_parse_utc_timestamp_is_always_tz_aware():
    stamp = parse_utc_timestamp("2024-01-01T00:00:00")
    assert stamp.tzinfo is not None
    assert stamp.utcoffset().total_seconds() == 0


@pytest.mark.parametrize("value", [None, "", "not-a-date", float("nan"), 0, -5, [], {}])
def test_parse_utc_timestamp_rejects_garbage(value):
    assert parse_utc_timestamp(value) is None


def test_bar_floor_daily_snaps_to_midnight_utc():
    stamp = parse_utc_timestamp("2024-03-10T23:59:59Z")
    assert bar_floor(stamp, "1d") == pd.Timestamp("2024-03-10", tz="UTC")


def test_bar_floor_hourly_and_monthly():
    stamp = parse_utc_timestamp("2024-03-10T14:37:00Z")
    assert bar_floor(stamp, "1h") == pd.Timestamp("2024-03-10 14:00:00", tz="UTC")
    assert bar_floor(stamp, "1mo") == pd.Timestamp("2024-03-01", tz="UTC")


def test_bar_close_is_end_of_containing_bar():
    stamp = parse_utc_timestamp("2024-03-10T09:15:00Z")
    assert bar_close(stamp, "1h") == pd.Timestamp("2024-03-10 09:59:59", tz="UTC")


def test_build_bar_index_is_continuous_and_inclusive():
    index = build_bar_index(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-10", tz="UTC"), "1d"
    )
    assert len(index) == 10
    assert (index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_interval_timedelta_and_periods_per_year():
    assert interval_timedelta("1h") == pd.Timedelta(hours=1)
    assert periods_per_year("1d") == 365


def test_bar_floor_rejects_unparseable():
    with pytest.raises(ValueError):
        bar_floor("garbage", "1d")


def test_interval_offset_rejects_unknown_interval():
    with pytest.raises(ValueError):
        interval_timedelta("fortnight")
