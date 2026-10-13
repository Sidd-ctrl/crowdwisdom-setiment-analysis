"""SQLite persistence layer.

The database is the single source of truth between pipeline stages, which keeps
ingestion repeatable: re-running a stage upserts on natural keys instead of
appending duplicates. Every row carries a `data_source` tag so synthetic demo
data can never be mixed into a live run.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from config.logging_config import get_logger

log = get_logger(__name__)

# Columns read back from SQLite. A column holding only NULLs comes back as
# object dtype full of None, which breaks arithmetic further down the pipeline,
# so every numeric column is coerced to float at the read boundary.
SENTIMENT_METRIC_COLUMNS = (
    "post_count", "mean_score", "pos_ratio", "neg_ratio", "neu_ratio",
    "weighted_score", "likes", "reposts", "views", "avg_followers",
)
POST_METRIC_COLUMNS = ("likes", "reposts", "views", "author_followers", "sentiment_score")

SOURCE_YFINANCE = "yfinance"
SOURCE_APIFY = "apify"
SOURCE_TAVILY = "tavily"
SOURCE_EXA = "exa"
SOURCE_SYNTHETIC = "synthetic"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol       TEXT NOT NULL,
    interval     TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    adj_close    REAL,
    volume       REAL,
    data_source  TEXT NOT NULL,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (symbol, interval, timestamp, data_source)
);

CREATE TABLE IF NOT EXISTS x_posts (
    post_id          TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    search_query     TEXT,
    post_text        TEXT,
    posted_at        TEXT NOT NULL,
    bar_ts           TEXT,
    likes            INTEGER,
    reposts          INTEGER,
    views            INTEGER,
    author_handle    TEXT,
    author_followers INTEGER,
    sentiment_score  REAL,
    sentiment_label  TEXT,
    data_source      TEXT NOT NULL,
    ingested_at      TEXT NOT NULL,
    PRIMARY KEY (post_id, symbol, data_source)
);

CREATE INDEX IF NOT EXISTS idx_posts_bar ON x_posts (symbol, bar_ts);

CREATE TABLE IF NOT EXISTS sentiment_bars (
    symbol              TEXT NOT NULL,
    interval            TEXT NOT NULL,
    bar_ts              TEXT NOT NULL,
    post_count          INTEGER,
    mean_score          REAL,
    pos_ratio           REAL,
    neg_ratio           REAL,
    neu_ratio           REAL,
    weighted_score      REAL,
    likes               INTEGER,
    reposts             INTEGER,
    views               INTEGER,
    avg_followers       REAL,
    data_source         TEXT NOT NULL,
    PRIMARY KEY (symbol, interval, bar_ts, data_source)
);

CREATE TABLE IF NOT EXISTS news_items (
    news_id      TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    source       TEXT NOT NULL,
    title        TEXT,
    url          TEXT,
    published_at TEXT,
    summary      TEXT,
    relevance    REAL,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (news_id, source)
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id           TEXT NOT NULL,
    mode             TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    interval         TEXT NOT NULL,
    model            TEXT NOT NULL,
    horizon          INTEGER NOT NULL,
    origin_ts        TEXT NOT NULL,
    target_ts        TEXT,
    base_price       REAL,
    predicted_return REAL,
    predicted_price  REAL,
    actual_return    REAL,
    actual_price     REAL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, model, horizon, origin_ts)
);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    run_id       TEXT NOT NULL,
    mode         TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    model        TEXT NOT NULL,
    horizon      INTEGER NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value REAL,
    n_samples    INTEGER,
    notes        TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, model, horizon, metric_name)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    """Thin wrapper around sqlite3 with idempotent upserts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ prices
    def upsert_prices(
        self, frame: pd.DataFrame, symbol: str, interval: str, source: str
    ) -> int:
        required = {"timestamp", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"price frame is missing columns: {sorted(missing)}")
        stamp = utc_now()
        rows = []
        for record in frame.to_dict("records"):
            rows.append(
                (
                    symbol,
                    interval,
                    str(record["timestamp"]),
                    _num(record.get("open")),
                    _num(record.get("high")),
                    _num(record.get("low")),
                    _num(record.get("close")),
                    _num(record.get("adj_close")),
                    _num(record.get("volume")),
                    source,
                    stamp,
                )
            )
        return self._insert_many(
            "prices",
            "symbol, interval, timestamp, open, high, low, close, adj_close, volume, "
            "data_source, ingested_at",
            rows,
        )

    def load_prices(self, symbol: str, interval: str, source: str) -> pd.DataFrame:
        query = (
            "SELECT timestamp, open, high, low, close, adj_close, volume "
            "FROM prices WHERE symbol = ? AND interval = ? AND data_source = ? "
            "ORDER BY timestamp"
        )
        frame = pd.read_sql_query(query, sqlite3.connect(self.path), params=(symbol, interval, source))
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.set_index("timestamp")

    # ------------------------------------------------------------------- posts
    def upsert_posts(self, posts: pd.DataFrame) -> int:
        if posts.empty:
            return 0
        stamp = utc_now()
        rows = []
        for record in posts.to_dict("records"):
            rows.append(
                (
                    str(record["post_id"]),
                    record.get("symbol", ""),
                    record.get("search_query"),
                    record.get("post_text"),
                    str(record["posted_at"]),
                    str(record.get("bar_ts")) if record.get("bar_ts") is not None else None,
                    _int(record.get("likes")),
                    _int(record.get("reposts")),
                    _int(record.get("views")),
                    record.get("author_handle"),
                    _int(record.get("author_followers")),
                    _num(record.get("sentiment_score")),
                    record.get("sentiment_label"),
                    record.get("data_source", SOURCE_SYNTHETIC),
                    stamp,
                )
            )
        return self._insert_many(
            "x_posts",
            "post_id, symbol, search_query, post_text, posted_at, bar_ts, likes, reposts, "
            "views, author_handle, author_followers, sentiment_score, sentiment_label, "
            "data_source, ingested_at",
            rows,
        )

    def load_posts(self, symbol: str, source: str) -> pd.DataFrame:
        query = (
            "SELECT post_id, symbol, search_query, post_text, posted_at, bar_ts, likes, "
            "reposts, views, author_handle, author_followers, sentiment_score, "
            "sentiment_label, data_source FROM x_posts WHERE symbol = ? AND data_source = ? "
            "ORDER BY posted_at"
        )
        frame = pd.read_sql_query(query, sqlite3.connect(self.path), params=(symbol, source))
        for column in POST_METRIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        if frame.empty:
            return frame
        frame["posted_at"] = pd.to_datetime(frame["posted_at"], utc=True, errors="coerce")
        frame["bar_ts"] = pd.to_datetime(frame["bar_ts"], utc=True, errors="coerce")
        return frame

    # -------------------------------------------------------- sentiment bars
    def upsert_sentiment_bars(
        self, frame: pd.DataFrame, symbol: str, interval: str, source: str
    ) -> int:
        if frame.empty:
            return 0
        rows = []
        for bar_ts, record in frame.iterrows():
            rows.append(
                (
                    symbol,
                    interval,
                    str(bar_ts),
                    _int(record.get("post_count")),
                    _num(record.get("mean_score")),
                    _num(record.get("pos_ratio")),
                    _num(record.get("neg_ratio")),
                    _num(record.get("neu_ratio")),
                    _num(record.get("weighted_score")),
                    _int(record.get("likes")),
                    _int(record.get("reposts")),
                    _int(record.get("views")),
                    _num(record.get("avg_followers")),
                    source,
                )
            )
        return self._insert_many(
            "sentiment_bars",
            "symbol, interval, bar_ts, post_count, mean_score, pos_ratio, neg_ratio, "
            "neu_ratio, weighted_score, likes, reposts, views, avg_followers, data_source",
            rows,
        )

    def load_sentiment_bars(self, symbol: str, interval: str, source: str) -> pd.DataFrame:
        query = (
            "SELECT bar_ts, post_count, mean_score, pos_ratio, neg_ratio, neu_ratio, "
            "weighted_score, likes, reposts, views, avg_followers FROM sentiment_bars "
            "WHERE symbol = ? AND interval = ? AND data_source = ? ORDER BY bar_ts"
        )
        frame = pd.read_sql_query(
            query, sqlite3.connect(self.path), params=(symbol, interval, source)
        )
        for column in SENTIMENT_METRIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        if frame.empty:
            return frame
        frame["bar_ts"] = pd.to_datetime(frame["bar_ts"], utc=True)
        return frame.set_index("bar_ts")

    # -------------------------------------------------------------------- news
    def upsert_news(self, news: pd.DataFrame) -> int:
        if news.empty:
            return 0
        stamp = utc_now()
        rows = []
        for record in news.to_dict("records"):
            rows.append(
                (
                    str(record["news_id"]),
                    record.get("symbol", ""),
                    str(record.get("source", "unknown")),
                    record.get("title"),
                    record.get("url"),
                    record.get("published_at"),
                    record.get("summary"),
                    _num(record.get("relevance")),
                    stamp,
                )
            )
        return self._insert_many(
            "news_items",
            "news_id, symbol, source, title, url, published_at, summary, relevance, ingested_at",
            rows,
        )

    def load_news(self, symbol: str | None = None) -> pd.DataFrame:
        query = "SELECT news_id, symbol, source, title, url, published_at, summary, relevance FROM news_items"
        params: tuple = ()
        if symbol:
            query += " WHERE symbol = ?"
            params = (symbol,)
        query += " ORDER BY published_at DESC"
        return pd.read_sql_query(query, sqlite3.connect(self.path), params=params)

    # --------------------------------------------------------------- forecasts
    def save_forecasts(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        stamp = utc_now()
        rows = []
        for record in frame.to_dict("records"):
            rows.append(
                (
                    record["run_id"],
                    record["mode"],
                    record["symbol"],
                    record["interval"],
                    record["model"],
                    int(record["horizon"]),
                    str(record["origin_ts"]),
                    str(record["target_ts"]) if record.get("target_ts") is not None else None,
                    _num(record.get("base_price")),
                    _num(record.get("predicted_return")),
                    _num(record.get("predicted_price")),
                    _num(record.get("actual_return")),
                    _num(record.get("actual_price")),
                    stamp,
                )
            )
        return self._insert_many(
            "forecasts",
            "run_id, mode, symbol, interval, model, horizon, origin_ts, target_ts, base_price, "
            "predicted_return, predicted_price, actual_return, actual_price, created_at",
            rows,
        )

    def load_forecasts(self, run_id: str | None = None, symbol: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM forecasts"
        clauses, params = [], []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY model, horizon, origin_ts"
        frame = pd.read_sql_query(query, sqlite3.connect(self.path), params=tuple(params))
        if not frame.empty:
            frame["origin_ts"] = pd.to_datetime(frame["origin_ts"], utc=True)
        return frame

    def latest_run_id(self, symbol: str | None = None, mode: str | None = None) -> str | None:
        query = "SELECT run_id FROM forecasts"
        clauses, params = [], []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        return row["run_id"] if row else None

    def save_metrics(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        stamp = utc_now()
        rows = [
            (
                record["run_id"],
                record["mode"],
                record["symbol"],
                record["model"],
                int(record["horizon"]),
                record["metric_name"],
                _num(record.get("metric_value")),
                _int(record.get("n_samples")),
                record.get("notes"),
                stamp,
            )
            for record in frame.to_dict("records")
        ]
        return self._insert_many(
            "evaluation_metrics",
            "run_id, mode, symbol, model, horizon, metric_name, metric_value, n_samples, "
            "notes, created_at",
            rows,
        )

    def load_metrics(self, run_id: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM evaluation_metrics"
        params: tuple = ()
        if run_id:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " ORDER BY model, horizon, metric_name"
        return pd.read_sql_query(query, sqlite3.connect(self.path), params=params)

    # ------------------------------------------------------------------ writer
    def _insert_many(self, table: str, columns: str, rows: list[tuple]) -> int:
        """INSERT OR IGNORE so re-ingesting identical records is a no-op."""
        if not rows:
            return 0
        placeholders = ", ".join(["?"] * len(rows[0]))
        statement = f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})"
        with self.connect() as conn:
            before = conn.total_changes
            conn.executemany(statement, rows)
            inserted = conn.total_changes - before
        if inserted:
            log.debug("inserted %d new row(s) into %s", inserted, table)
        return inserted

    # ------------------------------------------------------------------ counts
    def table_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self.connect() as conn:
            for table in (
                "prices",
                "x_posts",
                "sentiment_bars",
                "news_items",
                "forecasts",
                "evaluation_metrics",
            ):
                counts[table] = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        return counts


def _num(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def _int(value) -> int | None:
    number = _num(value)
    return None if number is None else int(number)
