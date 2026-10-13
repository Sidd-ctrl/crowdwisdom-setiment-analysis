"""News ingestion: normalisation, deduplication and credential handling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.news import (
    NewsIngestionError,
    collect_news,
    deduplicate_news,
    fetch_exa,
    fetch_tavily,
    news_id,
    normalize_news,
)
from src.storage.database import SOURCE_EXA, SOURCE_TAVILY


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        return self.responses.pop(0)


def test_missing_credentials_raise_a_clear_error():
    with pytest.raises(NewsIngestionError, match="TAVILY_API_KEY"):
        fetch_tavily("btc news", "", 5, 30)
    with pytest.raises(NewsIngestionError, match="EXA_API_KEY"):
        fetch_exa("btc news", "", 5, 30)


def test_tavily_rejects_a_bad_key_without_echoing_it():
    session = _Session([_Response(401, {"detail": "unauthorized"})])
    with pytest.raises(NewsIngestionError) as excinfo:
        fetch_tavily("btc", "secret-key", 5, 30, session)
    assert "secret-key" not in str(excinfo.value)


def test_tavily_results_are_normalised():
    payload = {
        "results": [
            {
                "title": "Bitcoin rallies",
                "url": "https://example.com/a?utm=1",
                "content": "Prices climbed on ETF inflows.",
                "published_date": "2024-03-10T12:00:00Z",
                "score": 0.91,
            }
        ]
    }
    session = _Session([_Response(200, payload)])
    raw = fetch_tavily("bitcoin", "key", 5, 30, session)
    frame = normalize_news(raw, SOURCE_TAVILY, "BTC-USD")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["title"] == "Bitcoin rallies"
    assert row["source"] == SOURCE_TAVILY
    assert row["published_at"].startswith("2024-03-10T12:00")
    assert row["relevance"] == pytest.approx(0.91)


def test_exa_results_are_normalised():
    payload = {
        "results": [
            {
                "title": "Nvidia guidance",
                "url": "https://news.example.org/n",
                "text": {"text": "Chips demand stays firm."},
                "publishedDate": "2024-03-11T08:30:00Z",
                "score": 0.7,
            }
        ]
    }
    session = _Session([_Response(200, payload)])
    raw = fetch_exa("nvidia", "key", 5, 30, session)
    frame = normalize_news(raw, SOURCE_EXA, "NVDA")
    assert frame.iloc[0]["summary"] == "Chips demand stays firm."
    assert frame.iloc[0]["source"] == SOURCE_EXA


def test_news_id_is_stable_across_url_formatting():
    assert news_id("https://example.com/story/", "t") == news_id("http://www.example.com/story", "t")
    assert news_id("https://example.com/story?utm_source=x", "t") == news_id("https://example.com/story", "t")
    assert news_id(None, "Same Headline") == news_id(None, "  same   headline ")
    assert news_id(None, "Other") != news_id(None, "Same Headline")


def test_deduplication_keeps_the_richer_record():
    frame = pd.DataFrame(
        [
            {"news_id": "x", "symbol": "BTC-USD", "source": SOURCE_TAVILY, "title": "T", "url": "u", "published_at": "2024-01-01", "summary": "short", "relevance": 0.5},
            {"news_id": "x", "symbol": "BTC-USD", "source": SOURCE_EXA, "title": "T", "url": "u", "published_at": "2024-01-01", "summary": "a much longer summary of the same story", "relevance": 0.4},
        ]
    )
    deduped = deduplicate_news(frame)
    assert len(deduped) == 1
    assert deduped.iloc[0]["summary"].startswith("a much longer")


def test_normalize_news_skips_items_without_content():
    frame = normalize_news([{"score": 0.5}, "junk", {"title": "kept"}], SOURCE_TAVILY, "BTC-USD")
    assert len(frame) == 1
    assert frame.iloc[0]["title"] == "kept"


def test_collect_news_reports_missing_credentials_instead_of_failing():
    frame, problems = collect_news(["btc"], "BTC-USD", tavily_key="", exa_key="")
    assert frame.empty
    assert any("credentials" in problem.lower() for problem in problems)


def test_collect_news_survives_one_provider_erroring():
    session = _Session(
        [
            _Response(401, {}),
            _Response(200, {"results": [{"title": "kept", "url": "https://e.example/x", "content": "body", "score": 0.5}]}),
        ]
    )
    frame, problems = collect_news(["btc"], "BTC-USD", tavily_key="bad", exa_key="good", session=session)
    assert len(frame) == 1
    assert len(problems) == 1


def test_collect_news_handles_empty_provider_responses():
    session = _Session([_Response(200, {"results": []}), _Response(200, {"results": []})])
    frame, problems = collect_news(["btc"], "BTC-USD", tavily_key="a", exa_key="b", session=session)
    assert frame.empty
    assert any("no usable items" in problem for problem in problems)
