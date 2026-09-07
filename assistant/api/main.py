"""Webhook receiver and a small read API.

The only thing here that must exist is the Gmail push endpoint -- real-time
delivery needs a public URL to deliver to. Everything else is a convenience for
looking at what the system decided without opening psql.

Every route here is authenticated, in one of two ways. The read/write API takes
a shared secret (`assistant.api.security`), because `/ask` and `/ingest` spend
model time and `/queue/{id}/promote` writes to a real calendar. The webhooks
cannot use that secret -- Google has no way to learn it -- so each is verified
the way its source signs deliveries: an OIDC token for Pub/Sub, an HMAC over the
raw body for Notion.

Push handling has one non-obvious rule: **acknowledge immediately, process
after**. Pub/Sub retries any delivery it does not get a prompt 2xx for, and
processing an email involves several model calls. Holding the connection open
for that long turns one notification into a retry storm of duplicates. They
would all be absorbed by the ingest guard, but each one would still cost the
fetch that precedes it.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from assistant import metrics
from assistant import repository as repo
from assistant.api.security import (
    WebhookVerificationError,
    build_google_verifier,
    describe_posture,
    enforce_bind,
    guard_unverified,
    is_notion_handshake,
    require_api_token,
    verify_gmail_push,
    verify_notion_signature,
)
from assistant.config import get_settings
from assistant.db import make_pool
from assistant.models import QueueStatus
from assistant.runner import build_services, ingest_changed_ids, ingest_source

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Before anything is listening: an unauthenticated API on a reachable
    # interface is a configuration mistake, and it should stop startup rather
    # than be discovered in an access log.
    enforce_bind(settings)

    pool = make_pool()
    await pool.open()

    async with build_services() as services:
        app.state.pool = pool
        app.state.services = services
        app.state.google_verifier = build_google_verifier(settings)
        log.info(
            "api.started",
            provider=services.provider.name,
            timezone=settings.timezone,
            **describe_posture(settings),
        )
        try:
            yield
        finally:
            await pool.close()


# The shared-secret check is declared once on the app rather than route by
# route, so a route added later is protected by default. Public paths and the
# self-verifying webhooks are exempted inside the dependency itself -- the list
# of what is *not* authenticated is worth having in one readable place.
app = FastAPI(
    title="Jake's assistant",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_token)],
)

DASHBOARD = Path(__file__).with_name("dashboard.html")


def _toolbox(conn):
    """A ToolBox over one request's connection.

    Built per request rather than held on app.state because it closes over a
    connection, and a pooled connection outliving the request that borrowed it
    is how a web app ends up with two requests writing down one socket.
    """
    from assistant.tools import ToolBox

    services = app.state.services
    return ToolBox(
        conn,
        connectors=services.connectors,
        embedder=services.embedder,
        provider=services.provider,
        classifier=services.classifier,
        executors=services.executors,
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Read from disk per request so editing the page needs no restart."""
    return DASHBOARD.read_text()


@app.get("/health")
async def health() -> dict[str, Any]:
    async with app.state.pool.connection() as conn:
        await conn.execute("SELECT 1")
    return {
        "status": "ok",
        "llm_provider": app.state.services.provider.name,
        "sources": [c.connector.source_type for c in app.state.services.connectors.all()],
        # What is actually enforced, not what was intended: "I thought the token
        # was set" is the ordinary way a secret ends up unchecked.
        "security": describe_posture(),
    }


@app.post("/webhooks/gmail")
async def gmail_webhook(request: Request, background: BackgroundTasks) -> dict[str, str]:
    """Pub/Sub push target. Returns immediately; the work happens after.

    A 200 here means "notification received", never "email processed". That is
    the correct contract: Pub/Sub is asking whether to stop retrying, not whether
    the calendar has been updated.

    Verification runs *before* the acknowledgement, which is the one place the
    acknowledge-first rule bends. A delivery that cannot prove where it came
    from should not be queued for processing at all, and 401 is also the honest
    answer to Pub/Sub: retrying will not help.
    """
    settings = get_settings()
    guard_unverified("gmail", settings.gmail_webhook_verifiers)
    try:
        passed = await verify_gmail_push(request, getattr(app.state, "google_verifier", None))
    except WebhookVerificationError as exc:
        log.warning("api.gmail_push_unverified", error=str(exc))
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    log.info("api.gmail_push_verified", checks=passed)
    payload = await request.json()
    background.add_task(_process_gmail_push, payload)
    return {"status": "accepted"}


async def _process_gmail_push(payload: dict[str, Any]) -> None:
    services = app.state.services
    connector = services.connectors.get("gmail")

    async with app.state.pool.connection() as conn:
        state = await repo.get_source_state(conn, "gmail")
        await connector.load_state(state)

        try:
            message_ids = await connector.handle_change_event(payload)
        except Exception as exc:  # noqa: BLE001 - a bad push must not kill the server
            log.exception("api.gmail_push_failed", error=str(exc))
            return

        if not message_ids:
            log.info("api.gmail_push_empty")
            return

        await ingest_changed_ids(
            conn,
            services.connectors,
            "gmail",
            message_ids,
            classifier=services.classifier,
            provider=services.provider,
            embedder=services.embedder,
            executors=services.executors,
        )


@app.post("/webhooks/notion")
async def notion_webhook(request: Request, background: BackgroundTasks) -> dict[str, Any]:
    """Optional. Notion is polled by default; this exists if push is ever enabled.

    Notion's subscription handshake posts a `verification_token` once, which must
    be echoed. After that, payloads are notification-only like Gmail's -- the
    page id is all that arrives, and the content is fetched separately.

    The body is read as bytes because that is what Notion signed. Hashing a
    re-serialized copy of the parsed JSON would compare a different string for
    any payload whose spacing or key order differs from what was sent, which is
    most of them.
    """
    raw = await request.body()
    payload = json.loads(raw or b"{}")

    if is_notion_handshake(payload):
        # Necessarily unsigned: the token it carries *is* the HMAC key, so this
        # delivery arrives before the secret exists. Safe to exempt because it
        # starts no work -- it names no page and the handler only echoes.
        log.info("api.notion_verification", token=payload["verification_token"])
        return {"verification_token": payload["verification_token"]}

    settings = get_settings()
    guard_unverified("notion", settings.notion_webhook_verifiers)
    if settings.notion_webhook_secret:
        try:
            verify_notion_signature(
                raw, request.headers.get("x-notion-signature"), settings.notion_webhook_secret
            )
        except WebhookVerificationError as exc:
            log.warning("api.notion_push_unverified", error=str(exc))
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    entity = payload.get("entity", {})
    page_id = entity.get("id") or payload.get("page_id")
    if not page_id:
        raise HTTPException(status_code=400, detail="no page id in payload")

    background.add_task(_process_notion_push, page_id)
    return {"status": "accepted"}


async def _process_notion_push(page_id: str) -> None:
    services = app.state.services
    async with app.state.pool.connection() as conn:
        await ingest_changed_ids(
            conn,
            services.connectors,
            "notion",
            [page_id],
            classifier=services.classifier,
            provider=services.provider,
            embedder=services.embedder,
            executors=services.executors,
        )


@app.get("/queue")
async def queue(limit: int = 20, reason: str | None = None) -> list[dict[str, Any]]:
    async with app.state.pool.connection() as conn:
        entries = await repo.list_pending(conn, limit=limit, reason=reason)
    return [
        {
            "entry_id": str(entry.id),
            "item": entry.item_key,
            "chunk": entry.chunk_key,
            "label": str(entry.classification_label),
            "confidence": entry.classification_confidence,
            "queued_because": str(entry.queue_reason),
            "drafted_action": entry.action_spec,
            "created_at": entry.created_at.isoformat(),
        }
        for entry in entries
    ]


@app.get("/audit")
async def audit(item_key: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Why the system did what it did, newest first."""
    async with app.state.pool.connection() as conn:
        return await repo.list_audit(conn, item_key=item_key, limit=limit)


@app.post("/queue/{entry_id}/dismiss")
async def dismiss(entry_id: str, note: str = "") -> dict[str, str]:
    async with app.state.pool.connection() as conn:
        await repo.resolve_queue_entry(
            conn,
            UUID(entry_id),
            status=QueueStatus.DISMISSED,
            resolution=note or "dismissed via API",
        )
    return {"status": "dismissed"}


@app.get("/overview")
async def overview() -> dict[str, Any]:
    settings = get_settings()
    async with app.state.pool.connection() as conn:
        totals = await repo.counts(conn)
        sources = []
        for config in app.state.services.connectors.all():
            source_type = config.connector.source_type
            state = await repo.get_source_state(conn, source_type)
            sources.append(
                {
                    "source_type": source_type,
                    "trust_level": str(config.connector.trust_level),
                    "poll_interval_hours": config.poll_interval_hours,
                    "last_fetch_at": state.get("last_fetch_at"),
                }
            )
    return {
        "counts": totals,
        "sources": sources,
        "provider": app.state.services.provider.name,
        "timezone": settings.timezone,
        "dry_run": not settings.calendar_mcp_command.strip(),
    }


@app.get("/metrics")
async def metrics_json(days: int = 7) -> dict[str, Any]:
    """Both halves of the cost picture in one response.

    `live` is this process since it started; `window` is every item stored in
    the last `days`, across restarts. Neither is a substitute for the other, and
    quoting one while meaning the other is the mistake this shape is trying to
    prevent.
    """
    since = datetime.now(get_settings().tz) - timedelta(days=days)
    async with app.state.pool.connection() as conn:
        window = await repo.metrics_summary(conn, since=since)
    return {"live": metrics.REGISTRY.snapshot(), "window": window}


@app.get("/metrics.prom", response_class=PlainTextResponse)
async def metrics_prometheus() -> str:
    """Text exposition, for a scraper that already exists.

    Authenticated like everything else -- a scrape reveals volumes, model
    choice, and how often the gate declines to act, which is not public
    information about someone's mail.
    """
    return metrics.REGISTRY.prometheus()


@app.get("/items")
async def items(limit: int = 50, source_type: str | None = None) -> list[dict[str, Any]]:
    async with app.state.pool.connection() as conn:
        return await repo.list_recent_items(conn, limit=limit, source_type=source_type)


@app.get("/upcoming")
async def upcoming(days: int = 60) -> list[dict[str, Any]]:
    settings = get_settings()
    start = datetime.now(settings.tz) - timedelta(days=1)
    async with app.state.pool.connection() as conn:
        rows = await repo.list_upcoming_actions(conn, start=start, end=start + timedelta(days=days))
    return [
        {
            "when": row["last_extracted_date"],
            "action_type": row["action_type"],
            "action_id": row["action_id"],
            "item_key": row["item_key"],
            "chunk_key": row["chunk_key"],
            "title": row.get("title"),
            "url": row.get("url"),
        }
        for row in rows
    ]


@app.post("/search")
async def search(body: dict[str, Any]) -> dict[str, Any]:
    from assistant.retrieval import search_content

    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    async with app.state.pool.connection() as conn:
        hits = await search_content(
            conn, app.state.services.embedder, query, top_k=body.get("top_k")
        )
    return {
        "query": query,
        "hits": [
            {
                "chunk_key": hit.chunk_key,
                "content": hit.content,
                "similarity": hit.similarity,
                "title": hit.title,
                "url": hit.url,
                "source_type": hit.source_type,
            }
            for hit in hits
        ],
    }


@app.post("/ask")
async def ask(body: dict[str, Any]) -> dict[str, Any]:
    """One turn with the conversational agent, tools and all.

    Stateless: each request is its own conversation. The dashboard is a window
    onto what the system decided, not a chat client -- `assistant chat` is where
    a thread with history lives.
    """
    from assistant.agent import Conversation

    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    async with app.state.pool.connection() as conn:
        box = _toolbox(conn)
        conversation = Conversation(provider=app.state.services.provider, tools=box)
        answer = await conversation.send(message)
    return {
        "answer": answer,
        # Which tools it reached for is the interesting half of the answer: it
        # is the difference between a reply that was looked up and one that was
        # not, which is exactly what the grounding rule is about.
        "tools_used": [
            entry["name"] for entry in conversation.messages if entry["role"] == "tool"
        ],
    }


@app.post("/queue/{entry_id}/promote")
async def promote(entry_id: str, body: dict[str, Any] | None = None) -> dict[str, str]:
    """Approve a queued draft. The same path the agent's tool takes."""
    async with app.state.pool.connection() as conn:
        result = await _toolbox(conn).call(
            "promote_queue_entry",
            {"entry_id": entry_id, "note": (body or {}).get("note", "approved from the dashboard")},
        )
    return {"result": result}


@app.post("/ingest/{source_type}")
async def ingest(source_type: str) -> dict[str, Any]:
    """Fetch a source on demand -- the same pipeline the poll and the agent run."""
    services = app.state.services
    async with app.state.pool.connection() as conn:
        try:
            return await ingest_source(
                conn,
                services.connectors,
                source_type,
                classifier=services.classifier,
                provider=services.provider,
                embedder=services.embedder,
                executors=services.executors,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - a source being unreachable is a result
            log.warning("api.ingest_failed", source=source_type, error=str(exc))
            raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    settings = get_settings()
    enforce_bind(settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":  # pragma: no cover
    main()
