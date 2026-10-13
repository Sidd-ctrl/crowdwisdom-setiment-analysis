"""Apify post normalisation across actor output schemas."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.apify_x import (
    ApifyIngestionError,
    ApifyXClient,
    api_actor_id,
    normalize_post,
    normalize_posts,
)


def test_normalize_post_apidojo_style():
    raw = {
        "id": "1789",
        "text": "BTC is pumping today",
        "createdAt": "2024-03-10T14:22:31.000Z",
        "likeCount": 42,
        "retweetCount": 7,
        "viewCount": 12000,
        "author": {"userName": "trader_jane", "followers": 5300},
    }
    post = normalize_post(raw, "BTC-USD", "$BTC", "1d")
    assert post["post_id"] == "1789"
    assert post["likes"] == 42
    assert post["reposts"] == 7
    assert post["views"] == 12000
    assert post["author_handle"] == "trader_jane"
    assert post["author_followers"] == 5300
    assert post["bar_ts"] == pd.Timestamp("2024-03-10", tz="UTC")
    assert str(post["posted_at"]) == "2024-03-10 14:22:31+00:00"


def test_normalize_post_twitter_legacy_style():
    raw = {
        "id_str": "55",
        "full_text": "terrible day for $BTC",
        "created_at": "Sun Mar 10 14:22:31 +0000 2024",
        "favorite_count": "1,200",
        "retweet_count": 30,
        "user": {"screen_name": "@bearish_bob", "followers_count": 900},
    }
    post = normalize_post(raw, "BTC-USD", "bitcoin", "1d")
    assert post["post_id"] == "55"
    assert post["likes"] == 1200
    assert post["author_handle"] == "bearish_bob"
    assert post["author_followers"] == 900


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"id": "1"},
        {"id": "1", "text": "hello"},
        {"text": "no id here", "createdAt": "2024-01-01T00:00:00Z"},
        {"id": "1", "text": "   ", "createdAt": "2024-01-01T00:00:00Z"},
        {"id": "1", "text": "hi", "createdAt": "not-a-timestamp"},
        "not-a-dict",
    ],
)
def test_normalize_post_drops_unusable_records(raw):
    assert normalize_post(raw, "BTC-USD", "$BTC", "1d") is None


def test_normalize_posts_deduplicates_and_reports_columns():
    raw = [
        {"id": "1", "text": "a", "createdAt": "2024-01-01T00:00:00Z"},
        {"id": "1", "text": "a", "createdAt": "2024-01-01T00:00:00Z"},
        {"id": "2", "text": "b", "createdAt": "2024-01-02T00:00:00Z"},
        {"junk": True},
    ]
    frame = normalize_posts(raw, "BTC-USD", "$BTC", "1d")
    assert len(frame) == 2
    assert list(frame["post_id"]) == ["1", "2"]
    assert set(frame["data_source"]) == {"apify"}


def test_normalize_posts_empty_input_has_schema():
    frame = normalize_posts([], "BTC-USD", "$BTC", "1d")
    assert frame.empty
    assert "post_text" in frame.columns


def test_client_requires_credentials():
    with pytest.raises(ApifyIngestionError):
        ApifyXClient(token="", actor_id="someone/actor")
    with pytest.raises(ApifyIngestionError):
        ApifyXClient(token="token", actor_id="")


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Returns queued responses and records what was called."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append(("post", url, params, json))
        return self.responses.pop(0)

    def get(self, url, params=None, timeout=None):
        self.calls.append(("get", url, params, None))
        return self.responses.pop(0)


def test_client_reports_bad_token_without_leaking_it():
    session = _FakeSession([_FakeResponse(401, {"error": "invalid token"})])
    client = ApifyXClient(token="secret-token", actor_id="someone/actor", session=session)
    with pytest.raises(ApifyIngestionError) as excinfo:
        client._start_run({})
    message = str(excinfo.value)
    assert "401" in message
    assert "secret-token" not in message


def test_client_reports_unknown_actor():
    session = _FakeSession([_FakeResponse(404, {"error": "not found"})])
    client = ApifyXClient(token="t", actor_id="nobody/nothing", session=session)
    with pytest.raises(ApifyIngestionError, match="not found|404"):
        client._start_run({})


def test_client_paginates_dataset_items():
    page_one = [{"id": str(index), "text": "x", "createdAt": "2024-01-01T00:00:00Z"} for index in range(1000)]
    page_two = [{"id": "last", "text": "y", "createdAt": "2024-01-02T00:00:00Z"}]
    session = _FakeSession([_FakeResponse(200, page_one), _FakeResponse(200, page_two)])
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    items = client._fetch_items("dataset-1")
    assert len(items) == 1001
    offsets = [params["offset"] for _, _, params, _ in session.calls]
    assert offsets == [0, 1000]


def test_client_collect_swallows_query_failures_but_keeps_working_queries():
    session = _FakeSession(
        [
            _FakeResponse(401, {"error": "nope"}),
            _FakeResponse(200, {"data": {"id": "run-1", "status": "SUCCEEDED", "defaultDatasetId": "ds-1"}}),
            _FakeResponse(200, [{"id": "9", "text": "ok", "createdAt": "2024-01-01T00:00:00Z"}]),
        ]
    )
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    frame = client.collect(["$BTC", "bitcoin"], "BTC-USD", "1d")
    assert len(frame) == 1
    assert frame["post_id"].iloc[0] == "9"


def test_client_collect_returns_empty_frame_when_every_query_fails():
    session = _FakeSession([_FakeResponse(403, {}), _FakeResponse(403, {})])
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    frame = client.collect(["$BTC", "bitcoin"], "BTC-USD", "1d")
    assert frame.empty


def test_search_key_is_configurable():
    session = _FakeSession([_FakeResponse(200, {"data": {"id": "r", "status": "SUCCEEDED", "defaultDatasetId": "d"}})])
    client = ApifyXClient(token="t", actor_id="a/b", search_key="search_terms", session=session)
    client._resolve_dataset("bitcoin")
    _, _, _, payload = session.calls[0]
    assert payload == {"search_terms": ["bitcoin"]}


def test_store_style_actor_id_is_converted_for_the_api_route():
    session = _FakeSession([_FakeResponse(200, {"data": {"id": "r", "status": "SUCCEEDED", "defaultDatasetId": "d"}})])
    client = ApifyXClient(token="t", actor_id="apidojo/tweet-scraper", session=session)
    client._start_run({})
    _, url, _, _ = session.calls[0]
    assert "/acts/apidojo~tweet-scraper/runs" in url
    assert "/acts/apidojo/tweet-scraper/runs" not in url


def test_actor_id_variants_normalise_to_one_route():
    assert api_actor_id("apidojo/tweet-scraper") == "apidojo~tweet-scraper"
    assert api_actor_id("apidojo~tweet-scraper") == "apidojo~tweet-scraper"
    assert api_actor_id("https://apify.com/apidojo/tweet-scraper") == "apidojo~tweet-scraper"
    assert api_actor_id("https://apify.com/apidojo/tweet-scraper/input") == "apidojo~tweet-scraper"
    assert api_actor_id("61RPP7dywgiy0JPD0") == "61RPP7dywgiy0JPD0"


def test_client_reports_billing_refusal_clearly():
    session = _FakeSession([_FakeResponse(402, {"error": "payment required"})])
    client = ApifyXClient(token="t", actor_id="apidojo/tweet-scraper", session=session)
    with pytest.raises(ApifyIngestionError) as excinfo:
        client._start_run({})
    message = str(excinfo.value).lower()
    assert "402" in message and "billing" in message


def test_dataset_read_uses_lowercase_clean_flag():
    """Apify rejects "True"; requests would render a Python bool that way (HTTP 400)."""
    session = _FakeSession([_FakeResponse(200, [])])
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    client._fetch_items("dataset-1")
    _, _, params, _ = session.calls[0]
    assert params["clean"] == "true"
    assert params["clean"] is not True


def test_placeholder_no_results_rows_are_not_treated_as_posts():
    session = _LogSession(
        [
            _FakeResponse(200, {"data": {"id": "run-1", "status": "SUCCEEDED", "defaultDatasetId": "ds-1"}}),
            _FakeResponse(200, [{"noResults": True}, {"noResults": True}]),
        ],
        log_text="2026-10-13T08:40:40.626Z Collected nothing this time.\n",
    )
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    frame = client.collect(["$BTC"], "BTC-USD", "1d")
    assert frame.empty


def test_placeholders_are_skipped_but_real_posts_are_kept():
    session = _FakeSession(
        [
            _FakeResponse(200, {"data": {"id": "run-1", "status": "SUCCEEDED", "defaultDatasetId": "ds-1"}}),
            _FakeResponse(
                200,
                [
                    {"noResults": True},
                    {"id": "5", "text": "real post", "createdAt": "Tue Oct 13 08:29:22 +0000 2026"},
                ],
            ),
        ]
    )
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    frame = client.collect(["$BTC"], "BTC-USD", "1d")
    assert len(frame) == 1
    assert frame["post_id"].iloc[0] == "5"


def test_is_placeholder_recognises_the_shapes_actors_emit():
    from src.ingestion.apify_x import _is_placeholder

    assert _is_placeholder({"noResults": True})
    assert _is_placeholder({"noResults": "true"})
    assert _is_placeholder({"error": "nope"})
    assert not _is_placeholder({"id": "1", "text": "hi", "createdAt": "2024-01-01T00:00:00Z"})
    assert not _is_placeholder({"noResults": True, "text": "x"})


class _LogSession(_FakeSession):
    """Fake session that also serves the run log endpoint."""

    def __init__(self, responses, log_text="", log_status=200):
        super().__init__(responses)
        self.log_text = log_text
        self.log_status = log_status

    def get(self, url, params=None, timeout=None):
        if "/logs/" in url:
            self.calls.append(("get", url, params, None))
            response = _FakeResponse(self.log_status, {})
            response.text = self.log_text
            return response
        return super().get(url, params, timeout)


def test_run_log_hint_quotes_the_actor_error():
    session = _LogSession(
        [_FakeResponse(200, {"data": {"id": "run-1", "status": "SUCCEEDED", "defaultDatasetId": "d"}})],
        log_text=(
            # 'LIMITED_PERMISSIONS' must not be mistaken for the real cause
            '2026-10-13T08:42:59.009Z ACTOR: Running under "LIMITED_PERMISSIONS".\n'
            "2026-10-13T08:43:08.981Z \n"
            "2026-10-13T08:43:08.981Z Monthly run limit exceeded per user.\n"
            "2026-10-13T08:43:08.982Z Please subscribe to a paid plan on Apify.\n"
        ),
    )
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    client._start_run({})
    hint = client._run_log_hint()
    assert hint == "Monthly run limit exceeded per user."


def test_run_log_hint_is_none_without_a_run_or_a_useful_line():
    session = _LogSession([], log_text="2026-10-13T08:40:40.626Z Scraping tweets now\n")
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    assert client._run_log_hint() is None
    client.last_run_id = "run-9"
    assert client._run_log_hint() is None


def test_collect_reports_platform_cause_when_every_query_is_empty(caplog):
    import logging

    session = _LogSession(
        [
            _FakeResponse(200, {"data": {"id": "run-1", "status": "SUCCEEDED", "defaultDatasetId": "d"}}),
            _FakeResponse(200, [{"noResults": True}]),
        ],
        log_text="2026-10-13T08:40:40.626Z Monthly run limit exceeded per user.\n",
    )
    client = ApifyXClient(token="t", actor_id="a/b", session=session)
    with caplog.at_level(logging.WARNING):
        frame = client.collect(["$BTC"], "BTC-USD", "1d")
    assert frame.empty
    assert any("Monthly run limit exceeded" in record.message for record in caplog.records)
