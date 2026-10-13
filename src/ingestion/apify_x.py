"""Apify-based X (Twitter) ingestion.

The actor id and its JSON input are configurable because Apify hosts many X
scrapers with different input/output schemas. `normalize_post` maps the common
field spellings onto the canonical schema; anything unrecognised is dropped
rather than guessed, and failures surface as `ApifyIngestionError` instead of
empty-but-successful results.
"""

from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
import requests

from config.logging_config import get_logger
from src.ingestion.timeutils import bar_floor, parse_utc_timestamp
from src.storage.database import SOURCE_APIFY

log = get_logger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
RUN_PAGE_LIMIT = 1000
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}

POST_ID_KEYS = ("id", "tweet_id", "tweetId", "id_str", "idStr", "postId")
TEXT_KEYS = ("text", "full_text", "fullText", "tweetText", "content")
TIMESTAMP_KEYS = (
    "createdAt",
    "created_at",
    "created",
    "timestamp",
    "date",
    "publishedAt",
)
LIKE_KEYS = ("likeCount", "likes", "favorite_count", "favoriteCount", "likesCount")
REPOST_KEYS = ("retweetCount", "retweets", "repostCount", "repost_count", "reposts")
VIEW_KEYS = ("viewCount", "views", "view_count", "impressions", "viewsCount")
HANDLE_PATHS = (
    "author.userName",
    "author.screen_name",
    "author.handle",
    "user.screen_name",
    "userName",
    "authorHandle",
)
FOLLOWER_PATHS = (
    "author.followers",
    "author.followersCount",
    "author.followers_count",
    "user.followers_count",
    "followersCount",
    "authorFollowers",
)


class ApifyIngestionError(RuntimeError):
    """Raised when the Apify run cannot be started, completed or read."""


def api_actor_id(actor_id: str) -> str:
    """Convert a store-style actor id into the form the REST route expects.

    The Apify console and store show actors as ``username/actor-name``, but the
    ``/v2/acts/...`` routes only resolve ``username~actor-name``. Pasted store
    URLs ("https://apify.com/user/actor") are accepted as well.
    """
    cleaned = actor_id.strip().rstrip("/")
    if cleaned.startswith(("http://", "https://")):
        cleaned = cleaned.split("//", 1)[1]
        cleaned = cleaned.split("/", 1)[1] if "/" in cleaned else ""
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[0]}~{parts[1]}"
    return cleaned.replace("/", "~")


def _is_placeholder(item: Any) -> bool:
    """True for the empty-search markers some actors write into their dataset.

    A record that carries any usable post field is never a placeholder, so real
    rows are kept even if they also contain a marker key.
    """
    if not isinstance(item, dict):
        return False
    usable = ("id", "postId", "tweet_id", "text", "full_text", "fullText", "createdAt")
    if any(item.get(key) not in (None, "") for key in usable):
        return False
    if item.get("noResults") in (True, "true", 1, "1"):
        return True
    return set(item) <= {"noResults", "error", "errorMessage", "url"}


def _pick(record: dict, keys: tuple[str, ...]) -> Any:
    """Return the first present value, supporting dotted nested paths."""
    for key in keys:
        node: Any = record
        found = True
        for part in key.split("."):
            if isinstance(node, dict) and part in node and node[part] is not None:
                node = node[part]
            else:
                found = False
                break
        if found:
            return node
    return None


def normalize_post(
    raw: dict, symbol: str, search_query: str, interval: str, source: str = SOURCE_APIFY
) -> dict | None:
    """Map one raw actor item onto the canonical post schema.

    Returns None when the record has no usable id/text/timestamp, so garbage
    rows are dropped at the boundary rather than propagating downstream.
    """
    if not isinstance(raw, dict):
        return None

    post_id = _pick(raw, POST_ID_KEYS)
    text = _pick(raw, TEXT_KEYS)
    posted_at = parse_utc_timestamp(_pick(raw, TIMESTAMP_KEYS))
    if post_id is None or text is None or posted_at is None:
        return None

    text = str(text).strip()
    if not text:
        return None

    return {
        "post_id": str(post_id),
        "symbol": symbol,
        "search_query": search_query,
        "post_text": text,
        "posted_at": posted_at,
        "bar_ts": bar_floor(posted_at, interval),
        "likes": _to_int(_pick(raw, LIKE_KEYS)),
        "reposts": _to_int(_pick(raw, REPOST_KEYS)),
        "views": _to_int(_pick(raw, VIEW_KEYS)),
        "author_handle": _clean_handle(_pick(raw, HANDLE_PATHS)),
        "author_followers": _to_int(_pick(raw, FOLLOWER_PATHS)),
        "data_source": source,
    }


def normalize_posts(
    items: list[dict], symbol: str, search_query: str, interval: str, source: str = SOURCE_APIFY
) -> pd.DataFrame:
    records = [
        record
        for record in (
            normalize_post(item, symbol, search_query, interval, source) for item in items
        )
        if record
    ]
    if not records:
        return pd.DataFrame(
            columns=[
                "post_id",
                "symbol",
                "search_query",
                "post_text",
                "posted_at",
                "bar_ts",
                "likes",
                "reposts",
                "views",
                "author_handle",
                "author_followers",
                "data_source",
            ]
        )
    frame = pd.DataFrame(records).drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    dropped = len(items) - len(frame)
    if dropped:
        log.warning("dropped %d unparseable item(s) from query '%s'", dropped, search_query)
    return frame


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace(" ", "").strip()
        multiplier = 1
        if cleaned.endswith(("K", "k")):
            cleaned, multiplier = cleaned[:-1], 1_000
        elif cleaned.endswith(("M", "m")):
            cleaned, multiplier = cleaned[:-1], 1_000_000
        try:
            value = float(cleaned) * multiplier
        except ValueError:
            return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean_handle(value: Any) -> str | None:
    if value is None:
        return None
    handle = str(value).strip().lstrip("@")
    return handle or None


class ApifyXClient:
    """Start (or reuse) an Apify actor run and page through its dataset."""

    def __init__(
        self,
        token: str,
        actor_id: str = "",
        input_json: dict | None = None,
        dataset_id: str = "",
        search_key: str = "searchTerms",
        max_items: int = 5000,
        timeout_s: int = 600,
        session: requests.Session | None = None,
    ):
        if not token:
            raise ApifyIngestionError("APIFY_TOKEN is not set; cannot call the Apify API.")
        if not actor_id and not dataset_id:
            raise ApifyIngestionError(
                "Neither APIFY_ACTOR_ID nor APIFY_DATASET_ID is set; nothing to collect."
            )
        self.token = token
        self.actor_id = actor_id
        self.input_json = dict(input_json or {})
        self.dataset_id = dataset_id
        self.search_key = search_key
        self.max_items = max_items
        self.timeout_s = timeout_s
        self.last_run_id: str | None = None
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ public
    def collect(self, queries: list[str], symbol: str, interval: str) -> pd.DataFrame:
        """Run the actor once per search query and return normalised posts."""
        frames = []
        for query in queries:
            log.info("Apify: collecting posts for query '%s'", query)
            try:
                dataset_id = self._resolve_dataset(query)
                items = self._fetch_items(dataset_id)
            except ApifyIngestionError as exc:
                log.error("Apify query '%s' failed: %s", query, exc)
                continue
            placeholders = [item for item in items if _is_placeholder(item)]
            usable = [item for item in items if not _is_placeholder(item)]
            if placeholders:
                log.warning(
                    "Apify: '%s' returned %d placeholder item(s) and no posts "
                    "(the actor reports empty searches as 'noResults' rows)",
                    query,
                    len(placeholders),
                )
            log.info("Apify: %d post item(s) for '%s'", len(usable), query)
            if placeholders and not usable:
                hint = self._run_log_hint()
                if hint:
                    log.warning("Apify run reported: %s", hint)
            frames.append(normalize_posts(usable, symbol, query, interval))

        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
        log.info("Apify: %d unique post(s) across %d query(ies)", len(combined), len(queries))
        return combined

    # ----------------------------------------------------------------- internals
    def _resolve_dataset(self, query: str) -> str:
        if self.dataset_id:
            return self.dataset_id
        payload = dict(self.input_json)
        if self.search_key:
            payload[self.search_key] = [query]
        run = self._start_run(payload)
        return self._wait_for_dataset(run)

    def _start_run(self, payload: dict) -> dict:
        url = f"{APIFY_BASE}/acts/{api_actor_id(self.actor_id)}/runs"
        params = {"token": self.token, "waitForFinish": 120}
        try:
            response = self.session.post(url, params=params, json=payload, timeout=120)
        except requests.RequestException as exc:
            raise ApifyIngestionError(f"could not reach the Apify API: {exc}") from exc
        if response.status_code in (401, 403):
            raise ApifyIngestionError(
                f"Apify rejected the request (HTTP {response.status_code}): the token is invalid "
                "or lacks access to this actor."
            )
        if response.status_code == 404:
            raise ApifyIngestionError(
                f"Apify actor '{api_actor_id(self.actor_id)}' was not found (HTTP 404). "
                "Check that APIFY_ACTOR_ID is a public actor, e.g. apidojo/tweet-scraper."
            )
        if response.status_code == 402:
            raise ApifyIngestionError(
                "Apify refused the run for billing reasons (HTTP 402). This actor is "
                "pay-per-result, so the account needs prepaid usage or a plan that allows it. "
                "Add funds under Apify -> Billing, lower maxItems in APIFY_ACTOR_INPUT_JSON, "
                "or set APIFY_DATASET_ID to reuse a dataset you already scraped."
            )
        if response.status_code >= 400:
            raise ApifyIngestionError(
                f"Apify run start failed (HTTP {response.status_code}): {_short(response.text)}"
            )
        data = response.json().get("data") or {}
        if not data.get("id"):
            raise ApifyIngestionError(f"Apify returned no run id: {_short(response.text)}")
        self.last_run_id = data["id"]
        log.debug("Apify run %s started with status %s", data["id"], data.get("status"))
        return data

    def _wait_for_dataset(self, run: dict) -> str:
        status = run.get("status")
        if status == "SUCCEEDED" and run.get("defaultDatasetId"):
            return run["defaultDatasetId"]
        if status in TERMINAL_STATES and status != "SUCCEEDED":
            raise ApifyIngestionError(f"Apify run ended with status {status}.")

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            time.sleep(5)
            current = self._run_status(run["id"])
            status = current.get("status")
            if status == "SUCCEEDED":
                return current["defaultDatasetId"]
            if status in TERMINAL_STATES:
                raise ApifyIngestionError(f"Apify run ended with status {status}.")
            log.debug("Apify run %s still %s", run["id"], status)
        raise ApifyIngestionError(f"Apify run did not finish within {self.timeout_s}s.")

    def _run_log_hint(self) -> str | None:
        """Return the actor's own error line when a finished run collected nothing.

        Some platform conditions are invisible in the API response: a free-plan
        account that has used up its monthly actor runs still reports the run as
        SUCCEEDED and writes only empty-search markers. The run log is the only
        place that says why, so it is quoted here.
        """
        if not self.last_run_id:
            return None
        try:
            response = self.session.get(
                f"{APIFY_BASE}/logs/{self.last_run_id}", params={"token": self.token}, timeout=30
            )
        except requests.RequestException:
            return None
        if response.status_code >= 400:
            return None

        # Ordered most-specific first: the log also contains harmless lines such as
        # 'Running under "LIMITED_PERMISSIONS"', which a loose keyword match would quote.
        patterns = (
            r"monthly run limit",
            r"run limit",
            r"quota exceeded",
            r"rate limit",
            r"exceeded",
            r"not enough",
            r"subscribe to a paid plan",
            r"unauthorized",
            r"forbidden",
            r"blocked by",
        )
        lines = [
            re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*", "", line.strip())
            for line in response.text.splitlines()
        ]
        lines = [line for line in lines if line]
        for pattern in patterns:
            for line in lines:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    return line[:200]
        return None

    def _run_status(self, run_id: str) -> dict:
        url = f"{APIFY_BASE}/actor-runs/{run_id}"
        response = self.session.get(url, params={"token": self.token}, timeout=60)
        if response.status_code >= 400:
            raise ApifyIngestionError(
                f"could not read Apify run status (HTTP {response.status_code})."
            )
        return response.json().get("data") or {}

    def _fetch_items(self, dataset_id: str) -> list[dict]:
        items: list[dict] = []
        offset = 0
        while offset < self.max_items:
            limit = min(RUN_PAGE_LIMIT, self.max_items - offset)
            url = f"{APIFY_BASE}/datasets/{dataset_id}/items"
            # Apify accepts only true/false/0/1 here; requests would render a Python
            # bool as "True", which the API rejects with HTTP 400.
            params = {"token": self.token, "offset": offset, "limit": limit, "clean": "true"}
            try:
                response = self.session.get(url, params=params, timeout=120)
            except requests.RequestException as exc:
                raise ApifyIngestionError(f"could not read Apify dataset: {exc}") from exc
            if response.status_code >= 400:
                raise ApifyIngestionError(
                    f"could not read Apify dataset {dataset_id} (HTTP {response.status_code})."
                )
            page = response.json()
            if not isinstance(page, list):
                raise ApifyIngestionError("Apify dataset response was not a JSON list.")
            items.extend(page)
            if len(page) < limit:
                break
            offset += limit
        return items


def _short(text: str, limit: int = 300) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit]
