"""News context from Tavily and Exa.

News is research context only. Nothing produced here enters the feature matrix;
the modelling stage never reads this module's output.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from config.logging_config import get_logger
from src.ingestion.timeutils import parse_utc_timestamp
from src.storage.database import SOURCE_EXA, SOURCE_TAVILY

log = get_logger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
EXA_URL = "https://api.exa.ai/search"


class NewsIngestionError(RuntimeError):
    """Raised when a news provider cannot be queried."""


def fetch_tavily(
    query: str, api_key: str, max_results: int, lookback_days: int, session: requests.Session | None = None
) -> list[dict]:
    if not api_key:
        raise NewsIngestionError("TAVILY_API_KEY is not set; skipping Tavily news.")
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "topic": "news",
        "days": lookback_days,
        "include_answer": False,
    }
    client = session or requests.Session()
    try:
        response = client.post(TAVILY_URL, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise NewsIngestionError(f"Tavily request failed: {exc}") from exc
    if response.status_code in (401, 403):
        raise NewsIngestionError(f"Tavily rejected the API key (HTTP {response.status_code}).")
    if response.status_code >= 400:
        raise NewsIngestionError(f"Tavily request failed (HTTP {response.status_code}): {response.text[:200]}")
    results = response.json().get("results") or []
    log.info("Tavily returned %d result(s) for '%s'", len(results), query)
    return results


def fetch_exa(
    query: str, api_key: str, max_results: int, lookback_days: int, session: requests.Session | None = None
) -> list[dict]:
    if not api_key:
        raise NewsIngestionError("EXA_API_KEY is not set; skipping Exa search.")
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    payload = {
        "query": query,
        "numResults": max_results,
        "type": "neural",
        "startPublishedDate": start,
        "contents": {"text": {"maxCharacters": 600}},
    }
    client = session or requests.Session()
    try:
        response = client.post(
            EXA_URL, json=payload, headers={"x-api-key": api_key, "Content-Type": "application/json"}, timeout=60
        )
    except requests.RequestException as exc:
        raise NewsIngestionError(f"Exa request failed: {exc}") from exc
    if response.status_code in (401, 403):
        raise NewsIngestionError(f"Exa rejected the API key (HTTP {response.status_code}).")
    if response.status_code >= 400:
        raise NewsIngestionError(f"Exa request failed (HTTP {response.status_code}): {response.text[:200]}")
    results = response.json().get("results") or []
    log.info("Exa returned %d result(s) for '%s'", len(results), query)
    return results


def normalize_news(raw_results: list[dict], source: str, symbol: str) -> pd.DataFrame:
    """Map provider payloads onto the canonical news schema."""
    rows = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        if source == SOURCE_TAVILY:
            title = item.get("title")
            url = item.get("url")
            summary = item.get("content") or item.get("snippet")
            published = parse_utc_timestamp(item.get("published_date") or item.get("publishedDate"))
            relevance = _as_float(item.get("score"))
        else:
            title = item.get("title")
            url = item.get("url") or item.get("id")
            body = item.get("text") or {}
            summary = body.get("text") if isinstance(body, dict) else item.get("summary")
            published = parse_utc_timestamp(item.get("publishedDate") or item.get("published_date"))
            relevance = _as_float(item.get("score"))
        if not title and not summary:
            continue
        rows.append(
            {
                "news_id": news_id(url, title),
                "symbol": symbol,
                "source": source,
                "title": _clean(title) or "(untitled)",
                "url": _clean(url),
                "published_at": published.isoformat() if published is not None else None,
                "summary": _truncate(_clean(summary), 600),
                "relevance": relevance,
            }
        )
    return pd.DataFrame(rows)


def deduplicate_news(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop repeats across providers using the canonical id, keeping the richest row."""
    if frame.empty:
        return frame
    enriched = frame.copy()
    enriched["_richness"] = enriched["summary"].fillna("").str.len() + enriched["title"].fillna("").str.len()
    enriched = enriched.sort_values("_richness", ascending=False)
    enriched = enriched.drop_duplicates(subset=["news_id"], keep="first")
    return enriched.drop(columns="_richness").sort_values("published_at", ascending=False).reset_index(drop=True)


def collect_news(
    queries: list[str],
    symbol: str,
    tavily_key: str = "",
    exa_key: str = "",
    max_results: int = 15,
    lookback_days: int = 30,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Query both providers and return (deduplicated news, list of problems)."""
    frames: list[pd.DataFrame] = []
    problems: list[str] = []

    if not tavily_key and not exa_key:
        problems.append("No news credentials configured (TAVILY_API_KEY / EXA_API_KEY both empty).")
        return pd.DataFrame(), problems

    for query in queries:
        if tavily_key:
            try:
                frames.append(normalize_news(fetch_tavily(query, tavily_key, max_results, lookback_days, session), SOURCE_TAVILY, symbol))
            except NewsIngestionError as exc:
                problems.append(str(exc))
        if exa_key:
            try:
                frames.append(normalize_news(fetch_exa(query, exa_key, max_results, lookback_days, session), SOURCE_EXA, symbol))
            except NewsIngestionError as exc:
                problems.append(str(exc))

    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        problems.append("News providers returned no usable items.")
        return pd.DataFrame(), problems
    return deduplicate_news(pd.concat(usable, ignore_index=True)), problems


def news_id(url: str | None, title: str | None) -> str:
    """Stable id so the same article is stored once across providers and runs."""
    basis = _normalize_url(url) if url else _normalize_title(title)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _normalize_url(url: str) -> str:
    cleaned = url.strip().lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = re.sub(r"^www\.", "", cleaned)
    cleaned = re.sub(r"[?#].*$", "", cleaned)
    return cleaned.rstrip("/")


def _normalize_title(title: str | None) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _clean(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _as_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
