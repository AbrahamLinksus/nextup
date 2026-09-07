"""Gmail connector: push-driven, untrusted, and metadata-poor by nature.

Three properties of this source shape the whole implementation:

**Untrusted.** Notion is what the user wrote; Gmail is what anyone sent them.
Every item produced here carries `TrustLevel.UNTRUSTED`, which makes the
pipeline delimit its content in prompts and run an injection check before
classification.

**Push, not poll.** Inbound email deadlines do not tolerate a twelve-hour
staleness window the way self-authored notes do. `watch()` plus Pub/Sub gives
real-time delivery -- at the cost of two expiries to manage, below.

**Almost no structured metadata.** `deadline` and `status` are essentially
always `None` here. That is not a degraded mode to apologise for; it is the case
the normalized vocabulary was designed to survive, and the classifier reads
content alone.

Two expiries, and they compound. The `watch()` subscription lapses after ~7
days, and the `historyId` it hands back is itself only valid for ~7 days. If a
subscription lapses unrenewed past that window, `history.list` cannot diff at
all -- so an expired cursor falls back to a date-bounded search rather than
raising. Losing incremental precision is recoverable; losing a week of mail is
not.
"""

from __future__ import annotations

import base64
import binascii
import html
import json
import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from assistant.config import get_settings
from assistant.connectors.base import SourceConnector
from assistant.models import Item, TrustLevel

log = structlog.get_logger(__name__)

API_BASE = "https://gmail.googleapis.com/gmail/v1"
TOKEN_URL = "https://oauth2.googleapis.com/token"


class GmailNotConfigured(RuntimeError):
    pass


class GmailConnector(SourceConnector):
    source_type = "gmail"
    supports_push = True
    trust_level = TrustLevel.UNTRUSTED

    def __init__(self, token_file: str | None = None, *, timeout: float = 60.0) -> None:
        settings = get_settings()
        self.token_file = Path(token_file or settings.gmail_token_file)
        self.query = settings.gmail_query
        self.topic = settings.gmail_topic
        self._cursor: str | None = None        # historyId
        self._watch_expires_at: datetime | None = None
        self._access_token: str | None = None
        self._token_expiry: datetime | None = None
        self._client = httpx.AsyncClient(base_url=API_BASE, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def load_state(self, state: dict) -> None:
        self._cursor = state.get("cursor")
        self._watch_expires_at = state.get("watch_expires_at")

    def dump_state(self) -> dict:
        return {"cursor": self._cursor, "watch_expires_at": self._watch_expires_at}

    # -- auth ------------------------------------------------------------

    async def _token(self) -> str:
        """Return a live access token, refreshing when it is about to lapse.

        Refreshed 60s early rather than on expiry: a token that expires mid-crawl
        turns a partial fetch into a 401 the retry decorator cannot fix.
        """
        now = datetime.now(UTC)
        if self._access_token and self._token_expiry and now < self._token_expiry:
            return self._access_token

        if not self.token_file.exists():
            raise GmailNotConfigured(
                f"{self.token_file} not found -- run `assistant gmail-auth` first"
            )
        stored = json.loads(self.token_file.read_text())

        refresh_token = stored.get("refresh_token")
        if not refresh_token:
            raise GmailNotConfigured(f"{self.token_file} has no refresh_token")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TOKEN_URL,
                data={
                    "client_id": stored["client_id"],
                    "client_secret": stored["client_secret"],
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            payload = response.json()

        self._access_token = payload["access_token"]
        self._token_expiry = now + timedelta(seconds=int(payload.get("expires_in", 3600)) - 60)
        return self._access_token

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=10),
        reraise=True,
    )
    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        token = await self._token()
        return await self._client.request(
            method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs
        )

    async def _get_json(self, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._request("GET", path, **kwargs)
        response.raise_for_status()
        return response.json()

    # -- SourceConnector -------------------------------------------------

    async def list_items(self, since: datetime | None = None, *, limit: int = 100) -> list[Item]:
        message_ids = await self._changed_message_ids(since, limit=limit)
        items: list[Item] = []
        for message_id in message_ids[:limit]:
            try:
                items.append(await self.fetch_item(message_id))
            except httpx.HTTPStatusError as exc:
                # A message deleted between listing and fetching is normal, not
                # an error worth aborting a whole poll over.
                if exc.response.status_code == 404:
                    log.info("gmail.message_vanished", message_id=message_id)
                    continue
                raise
        log.info("gmail.listed", count=len(items))
        return items

    async def _changed_message_ids(self, since: datetime | None, *, limit: int) -> list[str]:
        if self._cursor:
            try:
                return await self._history_ids(self._cursor, limit=limit)
            except HistoryExpired:
                log.warning("gmail.history_expired", cursor=self._cursor)
                self._cursor = None

        return await self._search_ids(since, limit=limit)

    async def _history_ids(self, start_history_id: str, *, limit: int) -> list[str]:
        ids: list[str] = []
        page_token: str | None = None

        while len(ids) < limit:
            params: dict[str, Any] = {
                "startHistoryId": start_history_id,
                "historyTypes": ["messageAdded"],
                "maxResults": 100,
            }
            if page_token:
                params["pageToken"] = page_token

            response = await self._request("GET", "/users/me/history", params=params)
            if response.status_code == 404:
                raise HistoryExpired(start_history_id)
            response.raise_for_status()
            payload = response.json()

            for record in payload.get("history", []):
                for added in record.get("messagesAdded", []):
                    message_id = added.get("message", {}).get("id")
                    if message_id:
                        ids.append(message_id)

            # Advance the cursor even when nothing matched: the point is to not
            # re-walk this stretch of history next time.
            if payload.get("historyId"):
                self._cursor = str(payload["historyId"])
            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return list(dict.fromkeys(ids))

    async def _search_ids(self, since: datetime | None, *, limit: int) -> list[str]:
        """Date-bounded fallback, also the first-run path.

        `after:` takes a Unix timestamp and is inclusive to the day, so this can
        return a few messages already seen. That is harmless: the ingest CAS on
        `last_edited_at` drops them before any model call happens.
        """
        cutoff = since or datetime.now(UTC) - timedelta(days=7)
        query = f"{self.query} after:{int(cutoff.timestamp())}".strip()

        payload = await self._get_json(
            "/users/me/messages", params={"q": query, "maxResults": min(limit, 100)}
        )
        ids = [message["id"] for message in payload.get("messages", [])]

        # Re-anchor the incremental cursor so the next run can use history again.
        profile = await self._get_json("/users/me/profile")
        if profile.get("historyId"):
            self._cursor = str(profile["historyId"])
        return ids

    async def fetch_item(self, item_id: str) -> Item:
        message = await self._get_json(
            f"/users/me/messages/{item_id}", params={"format": "full"}
        )
        return message_to_item(message, trust_level=self.trust_level)

    async def register_watch(self) -> dict:
        if not self.topic:
            raise GmailNotConfigured("GMAIL_TOPIC is not set; cannot call watch()")

        response = await self._request(
            "POST",
            "/users/me/watch",
            json={"topicName": self.topic, "labelIds": ["INBOX"]},
        )
        response.raise_for_status()
        payload = response.json()

        self._cursor = str(payload.get("historyId", "")) or self._cursor
        expiration = payload.get("expiration")
        if expiration:
            self._watch_expires_at = datetime.fromtimestamp(int(expiration) / 1000, tz=UTC)
        log.info("gmail.watch_registered", expires_at=self._watch_expires_at)
        return payload

    @property
    def watch_needs_renewal(self) -> bool:
        """True inside the last day of the subscription.

        Renewed early because letting it lapse does not merely pause delivery --
        it starts the clock on the history window closing too.
        """
        if self._watch_expires_at is None:
            return True
        return datetime.now(UTC) > self._watch_expires_at - timedelta(days=1)

    async def handle_change_event(self, payload: dict) -> list[str]:
        """Decode a Pub/Sub push body into changed message ids.

        The notification carries a historyId and nothing else -- it says that
        something changed, never what it now says -- so the diff is done here
        and the content is fetched separately.
        """
        encoded = (payload.get("message") or {}).get("data", "")
        if not encoded:
            log.warning("gmail.push_without_data", payload=payload)
            return []
        try:
            decoded = json.loads(base64.urlsafe_b64decode(encoded + "==").decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            log.warning("gmail.push_undecodable", error=str(exc))
            return []

        start = self._cursor or str(decoded.get("historyId", ""))
        if not start:
            return []
        try:
            return await self._history_ids(start, limit=100)
        except HistoryExpired:
            log.warning("gmail.history_expired_on_push")
            self._cursor = None
            return await self._search_ids(None, limit=100)


class HistoryExpired(RuntimeError):
    """The stored historyId is outside Gmail's ~7-day window."""


# ---------------------------------------------------------------------------
# Message -> Item
# ---------------------------------------------------------------------------

_HEADER_URGENCY = {"1": "high", "2": "high", "4": "low", "5": "low"}


def message_to_item(message: dict[str, Any], *, trust_level: TrustLevel) -> Item:
    payload = message.get("payload", {})
    headers = {
        header.get("name", "").lower(): header.get("value", "")
        for header in payload.get("headers", [])
    }

    received = _received_at(message, headers)
    body = extract_body(payload) or message.get("snippet", "")

    # The sender line matters to triage -- "your exam is on Friday" from the
    # university and from a stranger are not the same item -- so it is part of
    # the content rather than metadata the classifier has to be told to consult.
    content = "\n\n".join(
        part
        for part in (
            f"**From:** {headers.get('from', '(unknown sender)')}",
            f"**Subject:** {headers.get('subject', '(no subject)')}",
            body.strip(),
        )
        if part
    )

    return Item(
        source_type="gmail",
        source_id=message["id"],
        url=f"https://mail.google.com/mail/u/0/#inbox/{message['id']}",
        title=headers.get("subject") or "(no subject)",
        content=content,
        created_at=received,
        last_edited_at=received,
        # Gmail carries no deadline or status field. Left None deliberately.
        deadline=None,
        status=None,
        urgency_hint=_urgency(headers),
        raw_properties={
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "cc": headers.get("cc", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "list_id": headers.get("list-id", ""),
            "labels": message.get("labelIds", []),
            "thread_id": message.get("threadId", ""),
            "snippet": message.get("snippet", ""),
        },
        trust_level=trust_level,
    )


def _received_at(message: dict[str, Any], headers: dict[str, str]) -> datetime:
    """Prefer Gmail's own receipt timestamp over the sender's Date: header.

    `internalDate` is when *this mailbox* received it; `Date:` is whatever the
    sending client claimed, which can be wrong by hours or spoofed outright. The
    receipt time is what the relative-date anchor should be.
    """
    internal = message.get("internalDate")
    if internal:
        return datetime.fromtimestamp(int(internal) / 1000, tz=UTC)
    raw_date = headers.get("date")
    if raw_date:
        try:
            parsed = parsedate_to_datetime(raw_date)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    return datetime.now(UTC)


def _urgency(headers: dict[str, str]) -> str | None:
    priority = headers.get("x-priority", "").strip()[:1]
    if priority in _HEADER_URGENCY:
        return _HEADER_URGENCY[priority]
    importance = headers.get("importance", "").strip().lower()
    return importance if importance in {"high", "low", "normal"} else None


def extract_body(payload: dict[str, Any]) -> str:
    """Pull the best available body out of a MIME tree, as markdown-ish text.

    text/plain wins when present -- it is what the sender's client produced from
    their own markup, and converting HTML is always lossy. HTML is converted only
    when there is nothing else.
    """
    plain = _find_part(payload, "text/plain")
    if plain:
        return _decode(plain)
    markup = _find_part(payload, "text/html")
    if markup:
        return html_to_text(_decode(markup))
    return ""


def _find_part(payload: dict[str, Any], mime_type: str) -> dict[str, Any] | None:
    if payload.get("mimeType") == mime_type and payload.get("body", {}).get("data"):
        return payload
    for part in payload.get("parts", []) or []:
        found = _find_part(part, mime_type)
        if found:
            return found
    return None


def _decode(part: dict[str, Any]) -> str:
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    try:
        raw = base64.urlsafe_b64decode(data + "==")
    except binascii.Error:
        return ""
    return raw.decode("utf-8", errors="replace")


_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_LINK_RE = re.compile(
    r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
)
_BREAK_RE = re.compile(r"<(br|/p|/div|/tr|/li|/ul|/ol|/table|/h[1-6])\s*/?>", re.IGNORECASE)
_LIST_RE = re.compile(r"<li\b[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(markup: str) -> str:
    """Convert HTML email into plain-ish markdown.

    Links keep their URLs, because "click here to confirm your slot" with the
    URL stripped loses the only actionable thing in the message. List markers
    are preserved for the same reason paragraph breaks are: they are how a
    sender separated several deadlines from each other.
    """
    text = _SCRIPT_RE.sub(" ", markup)
    text = _LINK_RE.sub(lambda m: f"[{_TAG_RE.sub('', m.group(2)).strip()}]({m.group(1)})", text)
    text = _BREAK_RE.sub("\n", text)
    text = _LIST_RE.sub("\n- ", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()
