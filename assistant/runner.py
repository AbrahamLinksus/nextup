"""Wiring: build the pipeline's collaborators once, run sources through it.

Everything here is composition. The reason it is a module rather than scattered
across the CLI, the webhook handler, and the agent's fetch tool is that all
three must run *the same* pipeline -- if on-demand fetch had its own assembly,
it would drift from the scheduled path, and the guarantee that the trigger type
never changes downstream processing would quietly stop being true.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from psycopg import AsyncConnection

from assistant import repository as repo
from assistant.actions import ExecutorPool
from assistant.classify import Classifier, SingleCallClassifier
from assistant.connectors import ConnectorRegistry, default_registry
from assistant.embeddings import Embedder
from assistant.llm import LLMProvider, make_provider
from assistant.models import Item
from assistant.pipeline import Pipeline, PipelineOutcome

log = structlog.get_logger(__name__)


@dataclass
class Services:
    """The long-lived collaborators, built once per process."""

    provider: LLMProvider
    embedder: Embedder
    classifier: Classifier
    connectors: ConnectorRegistry
    executors: ExecutorPool

    async def close(self) -> None:
        await self.executors.close()
        await self.connectors.close()
        await self.embedder.close()
        await self.provider.close()


@asynccontextmanager
async def build_services(connectors: ConnectorRegistry | None = None):
    provider = make_provider()
    embedder = Embedder()
    services = Services(
        provider=provider,
        embedder=embedder,
        classifier=SingleCallClassifier(provider),
        connectors=connectors or default_registry(),
        executors=ExecutorPool(),
    )
    try:
        yield services
    finally:
        await services.close()


def make_pipeline(conn: AsyncConnection, services: Services) -> Pipeline:
    return Pipeline(
        conn,
        classifier=services.classifier,
        provider=services.provider,
        embedder=services.embedder,
        executors=services.executors,
    )


async def process_items(
    conn: AsyncConnection,
    items: list[Item],
    *,
    classifier: Classifier,
    provider: LLMProvider,
    embedder: Embedder,
    executors: ExecutorPool | None = None,
) -> list[PipelineOutcome]:
    """Run a batch through the pipeline, one transaction per item.

    Per item rather than per batch: one page that fails to classify should not
    roll back the twelve that already succeeded, and each item's own writes are
    the unit that has to be all-or-nothing.
    """
    pipeline = Pipeline(
        conn,
        classifier=classifier,
        provider=provider,
        embedder=embedder,
        executors=executors or ExecutorPool(),
    )

    outcomes: list[PipelineOutcome] = []
    for item in items:
        try:
            outcome = await pipeline.process(item)
            await conn.commit()
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop a poll
            await conn.rollback()
            log.exception("pipeline.item_failed", item=item.key, error=str(exc))
            outcomes.append(
                PipelineOutcome(item_key=item.key, skipped=True, skip_reason=f"error: {exc}")
            )
            continue
        outcomes.append(outcome)
        log.info(
            "pipeline.processed",
            item=outcome.item_key,
            label=str(outcome.label) if outcome.label else None,
            skipped=outcome.skipped,
            actions=len(outcome.actions_taken),
            queued=len(outcome.queued),
        )
    return outcomes


async def ingest_source(
    conn: AsyncConnection,
    connectors: ConnectorRegistry,
    source_type: str,
    *,
    classifier: Classifier,
    provider: LLMProvider,
    embedder: Embedder,
    executors: ExecutorPool | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Fetch everything new from one source and run it through the pipeline.

    The cursor is only advanced after the items it covers have been committed.
    Advancing first would, on a crash, skip precisely the items that failed --
    and Gmail's history window closes after about a week, so those do not come
    back.
    """
    connector = connectors.get(source_type)
    state = await repo.get_source_state(conn, source_type)
    await connector.load_state(state)

    items = await connector.list_items(since=state.get("last_fetch_at"), limit=limit)
    outcomes = await process_items(
        conn,
        items,
        classifier=classifier,
        provider=provider,
        embedder=embedder,
        executors=executors,
    )

    cursor_state = connector.dump_state()
    await repo.save_source_state(
        conn,
        source_type,
        last_fetch_at=datetime.now(UTC),
        cursor=cursor_state.get("cursor"),
        watch_expires_at=cursor_state.get("watch_expires_at"),
    )
    await conn.commit()

    return summarize(source_type, outcomes)


async def ingest_changed_ids(
    conn: AsyncConnection,
    connectors: ConnectorRegistry,
    source_type: str,
    item_ids: list[str],
    *,
    classifier: Classifier,
    provider: LLMProvider,
    embedder: Embedder,
    executors: ExecutorPool | None = None,
) -> dict[str, Any]:
    """Push path: fetch the named items and run the identical pipeline.

    Push payloads are notification-only for both sources -- they say that
    something changed, never what it now says -- so content is always fetched
    here rather than read out of the notification.
    """
    connector = connectors.get(source_type)
    items = [await connector.fetch_item(item_id) for item_id in item_ids]
    outcomes = await process_items(
        conn,
        items,
        classifier=classifier,
        provider=provider,
        embedder=embedder,
        executors=executors,
    )
    cursor_state = connector.dump_state()
    await repo.save_source_state(
        conn,
        source_type,
        cursor=cursor_state.get("cursor"),
        watch_expires_at=cursor_state.get("watch_expires_at"),
    )
    await conn.commit()
    return summarize(source_type, outcomes)


def summarize(source_type: str, outcomes: list[PipelineOutcome]) -> dict[str, Any]:
    return {
        "source": source_type,
        "fetched": len(outcomes),
        "processed": sum(1 for o in outcomes if not o.skipped),
        "skipped": sum(1 for o in outcomes if o.skipped),
        "actions_taken": [action for o in outcomes for action in o.actions_taken],
        "queued": [entry for o in outcomes for entry in o.queued],
        "reconciled": [key for o in outcomes for key in o.reconciled],
    }
