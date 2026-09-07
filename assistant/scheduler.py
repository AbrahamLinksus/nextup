"""The poll loop and the maintenance jobs that keep push delivery alive.

Three recurring jobs, and only the first is the obvious one:

* **Per-connector polling.** Notion every 12 hours. Gmail declares no interval
  because push covers it.
* **Watch renewal.** Gmail's `watch()` subscription expires after ~7 days. This
  renews inside the last day, because letting it lapse does not merely pause
  delivery -- it also starts the clock on the `historyId` window closing, and
  once that is gone incremental sync cannot resume at all.
* **Backfill crawl.** A daily full-window pass over Notion. This is the safety
  net for the fact that Notion's integration access is opt-in per page: a page
  shared with the integration after the last poll has a `last_edited_time` older
  than the cursor and would never appear in an incremental fetch. The crawl
  costs almost nothing, because everything it re-fetches is rejected by the
  ingest guard before a single model call.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from assistant import repository as repo
from assistant.db import make_pool
from assistant.runner import Services, build_services, ingest_source, process_items

log = structlog.get_logger(__name__)

BACKFILL_INTERVAL_HOURS = 24
BACKFILL_WINDOW_DAYS = 30
WATCH_CHECK_INTERVAL_HOURS = 6


async def poll_source(pool, services: Services, source_type: str) -> None:
    async with pool.connection() as conn:
        summary = await ingest_source(
            conn,
            services.connectors,
            source_type,
            classifier=services.classifier,
            provider=services.provider,
            embedder=services.embedder,
            executors=services.executors,
        )
    log.info("scheduler.polled", **summary)


async def backfill(pool, services: Services, source_type: str) -> None:
    """Re-crawl a wide window, ignoring the incremental cursor.

    Deliberately does not advance `last_fetch_at`: this is a safety net running
    alongside the incremental path, not a replacement for it.
    """
    connector = services.connectors.get(source_type)
    since = datetime.now(UTC) - timedelta(days=BACKFILL_WINDOW_DAYS)

    async with pool.connection() as conn:
        state = await repo.get_source_state(conn, source_type)
        await connector.load_state(state)
        items = await connector.list_items(since=since, limit=500)
        outcomes = await process_items(
            conn,
            items,
            classifier=services.classifier,
            provider=services.provider,
            embedder=services.embedder,
            executors=services.executors,
        )

    fresh = [o for o in outcomes if not o.skipped]
    log.info(
        "scheduler.backfilled",
        source=source_type,
        crawled=len(outcomes),
        new_or_changed=len(fresh),
    )


async def renew_watches(pool, services: Services) -> None:
    for config in services.connectors.all():
        connector = config.connector
        if not connector.supports_push:
            continue

        async with pool.connection() as conn:
            state = await repo.get_source_state(conn, connector.source_type)
            await connector.load_state(state)

            if not getattr(connector, "watch_needs_renewal", False):
                continue
            try:
                await connector.register_watch()
            except Exception as exc:  # noqa: BLE001 - one source must not stop the rest
                log.warning(
                    "scheduler.watch_renewal_failed",
                    source=connector.source_type,
                    error=str(exc),
                )
                continue

            dumped = connector.dump_state()
            await repo.save_source_state(
                conn,
                connector.source_type,
                cursor=dumped.get("cursor"),
                watch_expires_at=dumped.get("watch_expires_at"),
            )
            await conn.commit()
            log.info("scheduler.watch_renewed", source=connector.source_type)


async def run() -> None:  # pragma: no cover - long-running process
    pool = make_pool()
    await pool.open()

    async with build_services() as services:
        scheduler = AsyncIOScheduler()

        for config in services.connectors.polling():
            source_type = config.connector.source_type
            scheduler.add_job(
                poll_source,
                IntervalTrigger(hours=config.poll_interval_hours),
                args=[pool, services, source_type],
                id=f"poll-{source_type}",
                max_instances=1,
                coalesce=True,
            )
            scheduler.add_job(
                backfill,
                IntervalTrigger(hours=BACKFILL_INTERVAL_HOURS),
                args=[pool, services, source_type],
                id=f"backfill-{source_type}",
                max_instances=1,
                coalesce=True,
            )
            log.info(
                "scheduler.registered",
                source=source_type,
                every_hours=config.poll_interval_hours,
            )

        if any(c.connector.supports_push for c in services.connectors.all()):
            scheduler.add_job(
                renew_watches,
                IntervalTrigger(hours=WATCH_CHECK_INTERVAL_HOURS),
                args=[pool, services],
                id="renew-watches",
                max_instances=1,
            )

        scheduler.start()
        log.info("scheduler.started")
        try:
            await asyncio.Event().wait()
        finally:
            scheduler.shutdown(wait=False)
            await pool.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run())
