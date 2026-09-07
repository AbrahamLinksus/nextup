"""The webhook receiver.

Two rules worth testing here. The first is the push contract: a 200 means
"notification received", never "email processed". Pub/Sub retries anything it does not get a
prompt 2xx for, and processing an email is several model calls -- holding the
connection open for that turns one notification into a storm of duplicates, each
costing a fetch before the ingest guard absorbs it.

The second is that a delivery has to prove where it came from before any of
that applies. Verification is configured for these tests rather than disabled,
so the push tests exercise the same code path production does -- a suite that
sets WEBHOOK_ALLOW_UNVERIFIED would be testing the escape hatch.

The app's real lifespan opens a connection pool and builds live services, so
these tests drive the routes with `app.state` set by hand instead. The routes
are thin on purpose; what is being checked is routing and ordering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from assistant.api import main as api
from assistant.config import get_settings

PUSH_TOKEN = "push-token-for-tests"
NOTION_SECRET = "notion-shared-secret"


def gmail_push(client, payload):
    """A Pub/Sub delivery carrying the configured URL token."""
    return client.post(f"/webhooks/gmail?token={PUSH_TOKEN}", json=payload)


def notion_push(client, payload):
    """A Notion delivery signed over exactly the bytes being sent."""
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(NOTION_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return client.post(
        "/webhooks/notion",
        content=body,
        headers={"Content-Type": "application/json", "X-Notion-Signature": signature},
    )


class RecordingConnector:
    def __init__(self, ids=("msg-1",), error: Exception | None = None):
        self.ids = list(ids)
        self.error = error
        self.payloads: list[dict] = []

    async def load_state(self, state):
        return None

    async def handle_change_event(self, payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return self.ids


class FakePool:
    """A pool whose connections do nothing, for routes that only pass one along."""

    def connection(self):
        class _Ctx:
            async def __aenter__(self_inner):
                return SimpleNamespace(execute=_noop)

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


async def _noop(*_args, **_kwargs):
    return None


@pytest.fixture
def client(monkeypatch):
    """A client over the app with the lifespan's collaborators stubbed in."""
    # Settings are cached for the process, so an override has to invalidate the
    # cache on the way in and on the way out -- otherwise the first test to
    # touch settings decides what every later one sees.
    monkeypatch.setenv("WEBHOOK_TOKEN", PUSH_TOKEN)
    monkeypatch.setenv("NOTION_WEBHOOK_SECRET", NOTION_SECRET)
    monkeypatch.setenv("API_TOKEN", "")
    get_settings.cache_clear()

    ingested: list[tuple[str, list[str]]] = []

    async def fake_ingest(_conn, _connectors, source_type, item_ids, **_kwargs):
        ingested.append((source_type, list(item_ids)))
        return {"source": source_type, "fetched": len(item_ids)}

    async def fake_state(_conn, _source_type):
        return {}

    monkeypatch.setattr(api, "ingest_changed_ids", fake_ingest)
    monkeypatch.setattr(api.repo, "get_source_state", fake_state)

    connector = RecordingConnector()
    api.app.state.pool = FakePool()
    api.app.state.services = SimpleNamespace(
        provider=SimpleNamespace(name="fake"),
        connectors=SimpleNamespace(
            get=lambda _source: connector,
            all=lambda: [
                SimpleNamespace(
                    connector=SimpleNamespace(source_type="gmail", trust_level="untrusted"),
                    poll_interval_hours=None,
                )
            ],
        ),
        classifier=None,
        embedder=None,
        executors=None,
    )

    # Deliberately not entered as a context manager: the app's real lifespan
    # opens a connection pool and builds live services, which would replace the
    # stubs above with the very things these tests exist to avoid.
    http = TestClient(api.app)
    http.ingested = ingested
    http.connector = connector
    yield http
    get_settings.cache_clear()


def test_health_reports_what_is_wired_up(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["sources"] == ["gmail"]


def test_a_gmail_push_is_acknowledged_before_the_work_happens(client):
    """The 200 answers 'stop retrying', not 'the calendar has been updated'."""
    response = gmail_push(client, {"message": {"data": "eyJ9"}})

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


def test_the_changed_ids_from_a_push_go_through_the_normal_pipeline(client):
    gmail_push(client, {"message": {"data": "eyJ9"}})

    assert client.ingested == [("gmail", ["msg-1"])]


def test_a_push_carrying_nothing_new_does_not_start_an_ingest(client):
    client.connector.ids = []
    gmail_push(client, {"message": {"data": "eyJ9"}})

    assert client.ingested == []


def test_a_malformed_push_is_logged_rather_than_killing_the_server(client):
    """The background task has no caller left to raise into."""
    client.connector.error = ValueError("no historyId in payload")
    response = gmail_push(client, {"nonsense": True})

    assert response.status_code == 200
    assert client.ingested == []


# ---------------------------------------------------------------------------
# Proving where a delivery came from
# ---------------------------------------------------------------------------

def test_an_unsigned_gmail_push_is_rejected_before_any_work_starts(client):
    """Costly to process and reachable by anyone who learns the URL."""
    response = client.post("/webhooks/gmail", json={"message": {"data": "eyJ9"}})

    assert response.status_code == 401
    assert client.ingested == []


def test_a_gmail_push_with_the_wrong_token_is_rejected(client):
    response = client.post("/webhooks/gmail?token=not-the-token", json={"message": {}})

    assert response.status_code == 401
    assert client.ingested == []


def test_an_unsigned_notion_push_is_rejected(client):
    response = client.post("/webhooks/notion", json={"entity": {"id": "page-9"}})

    assert response.status_code == 401
    assert client.ingested == []


def test_a_notion_push_signed_over_different_bytes_is_rejected(client):
    """The signature covers the body, so tampering with the page id invalidates it."""
    body = json.dumps({"entity": {"id": "page-9"}}).encode()
    signature = "sha256=" + hmac.new(NOTION_SECRET.encode(), body, hashlib.sha256).hexdigest()
    response = client.post(
        "/webhooks/notion",
        content=json.dumps({"entity": {"id": "page-EVIL"}}).encode(),
        headers={"Content-Type": "application/json", "X-Notion-Signature": signature},
    )

    assert response.status_code == 401
    assert client.ingested == []


def test_a_webhook_with_no_verification_configured_refuses_deliveries(client, monkeypatch):
    """Fail closed: an endpoint nobody configured is not an open endpoint."""
    monkeypatch.setenv("WEBHOOK_TOKEN", "")
    monkeypatch.setenv("NOTION_WEBHOOK_SECRET", "")
    get_settings.cache_clear()

    response = client.post("/webhooks/gmail", json={"message": {"data": "eyJ9"}})

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    assert client.ingested == []


def test_the_notion_subscription_handshake_is_echoed_back(client):
    """Notion posts this once to prove the endpoint is ours; nothing else happens."""
    response = client.post("/webhooks/notion", json={"verification_token": "tok-123"})

    assert response.json() == {"verification_token": "tok-123"}
    assert client.ingested == []


def test_a_notion_push_ingests_the_page_it_names(client):
    notion_push(client, {"entity": {"id": "page-9"}})

    assert client.ingested == [("notion", ["page-9"])]


def test_a_notion_push_with_no_page_id_is_rejected(client):
    """Accepting it would mean acknowledging a delivery nothing can act on."""
    assert notion_push(client, {"entity": {}}).status_code == 400


# ---------------------------------------------------------------------------
# The dashboard's read surface
# ---------------------------------------------------------------------------

def test_the_dashboard_page_is_served_at_the_root(client):
    """Served from disk per request, so editing the page needs no restart."""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Jake's assistant" in response.text


def test_the_overview_says_whether_actions_are_real(client, monkeypatch):
    """The single most important thing to see on the page: nothing is being written."""
    async def fake_counts(_conn):
        return {"items": 3, "chunks": 9, "embedded": 9, "pending": 1, "actions": 2}

    async def fake_state(_conn, _source):
        return {}

    monkeypatch.setattr(api.repo, "counts", fake_counts)
    monkeypatch.setattr(api.repo, "get_source_state", fake_state)

    body = client.get("/overview").json()

    assert body["counts"]["items"] == 3
    assert body["dry_run"] is True
    assert body["sources"][0]["source_type"] == "gmail"


def test_an_ingest_of_an_unregistered_source_is_a_404_not_a_500(client, monkeypatch):
    async def explode(*_args, **_kwargs):
        raise ValueError("no connector registered for source_type='slack'")

    monkeypatch.setattr(api, "ingest_source", explode)
    assert client.post("/ingest/slack").status_code == 404


def test_an_unreachable_source_reports_the_failure_rather_than_crashing(client, monkeypatch):
    """A missing API key is a normal thing to see on the page, not a stack trace."""
    async def explode(*_args, **_kwargs):
        raise RuntimeError("NOTION_API_KEY is not set")

    monkeypatch.setattr(api, "ingest_source", explode)
    response = client.post("/ingest/notion")

    assert response.status_code == 502
    assert "NOTION_API_KEY" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_the_metrics_endpoint_reports_live_and_stored_numbers(client, monkeypatch):
    """Two different questions, kept apart: this process, and this fortnight."""
    from assistant import metrics

    async def fake_summary(_conn, *, since):
        return {"since": since.isoformat(), "totals": {"items": 12}, "by_outcome": [],
                "by_source": [], "stages": []}

    monkeypatch.setattr(api.repo, "metrics_summary", fake_summary)
    metrics.REGISTRY.reset()
    metrics.record_llm_call(
        provider="fake", model="m", operation="structured", milliseconds=90, input_tokens=7
    )

    body = client.get("/metrics?days=14").json()

    assert body["window"]["totals"]["items"] == 12
    assert body["live"]["counters"]["llm_calls"]
    metrics.REGISTRY.reset()


def test_prometheus_exposition_is_served_as_text(client):
    from assistant import metrics

    metrics.REGISTRY.reset()
    metrics.record_gate_decision(allowed=False, reason="low_confidence")

    response = client.get("/metrics.prom")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "assistant_gate_decisions_total" in response.text
    metrics.REGISTRY.reset()


# ---------------------------------------------------------------------------
# Shared-secret auth, through the real routing stack
# ---------------------------------------------------------------------------

def test_a_configured_token_is_required_by_every_data_route(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    try:
        assert client.get("/queue").status_code == 401
        assert client.post("/ingest/notion").status_code == 401
        assert client.get("/metrics.prom").status_code == 401
        # The shell and the probe stay open; the shell fetches its data over
        # authenticated XHR.
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200
    finally:
        get_settings.cache_clear()


def test_the_token_lets_the_request_through(client, monkeypatch, ):
    async def fake_audit(_conn, *, item_key=None, limit=50):
        return []

    monkeypatch.setattr(api.repo, "list_audit", fake_audit)
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    try:
        response = client.get("/audit", headers={"Authorization": "Bearer s3cret"})
        assert response.status_code == 200
    finally:
        get_settings.cache_clear()


def test_a_webhook_does_not_need_the_api_token(client, monkeypatch):
    """It could not have it -- Pub/Sub is not configured with this deployment's
    secret -- so it proves itself with its own signature instead."""
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    try:
        assert gmail_push(client, {"message": {"data": "eyJ9"}}).status_code == 200
    finally:
        get_settings.cache_clear()


def test_search_needs_a_query(client):
    assert client.post("/search", json={"query": "  "}).status_code == 400


def test_ask_needs_a_message(client):
    assert client.post("/ask", json={}).status_code == 400
