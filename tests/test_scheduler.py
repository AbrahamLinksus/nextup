"""The recurring jobs.

Polling is the obvious one and is mostly the runner's behaviour, already tested.
The two worth their own tests are the ones that exist to prevent silent data
loss: the backfill crawl, which must not advance the incremental cursor it runs
alongside, and watch renewal, where letting a Gmail subscription lapse does not
merely pause delivery -- it starts the clock on the history window closing,
after which incremental sync cannot resume at all.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from assistant import repository as repo
from assistant.actions import ExecutorPool
from assistant.classify import SingleCallClassifier
from assistant.connectors import ConnectorRegistry
from assistant.connectors.base import SourceConnector
from assistant.models import TrustLevel
from assistant.runner import Services
from assistant.scheduler import BACKFILL_WINDOW_DAYS, backfill, poll_source, renew_watches
from tests.conftest import FakeEmbedder, FakeProvider, make_item

pytestmark = pytest.mark.db

NOW = datetime(2026, 8, 22, 9, 0, tzinfo=UTC)
INFORMATIONAL = {"label": "informational", "confidence": 0.95, "rationale": "notes"}


class SinglePool:
    """Hands every job the one test connection, which the db fixture tears down."""

    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self.conn


class PollingConnector(SourceConnector):
    source_type = "stub"

    def __init__(self, items=None):
        self.items = list(items or [])
        self.since: list[datetime | None] = []

    async def list_items(self, since=None, *, limit=100):
        self.since.append(since)
        return self.items

    async def fetch_item(self, item_id):  # pragma: no cover - unused here
        raise AssertionError("not needed")

    def dump_state(self):
        return {"cursor": "cursor-2"}


class PushConnector(SourceConnector):
    """A push source whose watch may or may not be near expiry."""

    supports_push = True
    trust_level = TrustLevel.UNTRUSTED

    def __init__(self, source_type="pushy", *, needs_renewal=True, fails=False):
        self.source_type = source_type
        self.watch_needs_renewal = needs_renewal
        self.fails = fails
        self.registrations = 0

    async def list_items(self, since=None, *, limit=100):
        return []

    async def fetch_item(self, item_id):  # pragma: no cover - unused here
        raise AssertionError("not needed")

    async def register_watch(self):
        self.registrations += 1
        if self.fails:
            raise RuntimeError("gmail said no")
        self.watch_needs_renewal = False
        return {}

    def dump_state(self):
        return {"cursor": "history-9", "watch_expires_at": NOW + timedelta(days=7)}


def services_for(*connectors, responses=None) -> Services:
    provider = FakeProvider(responses or [INFORMATIONAL] * 10)
    registry = ConnectorRegistry()
    for connector in connectors:
        registry.register(connector, poll_interval_hours=12)
    return Services(
        provider=provider,
        embedder=FakeEmbedder(),
        classifier=SingleCallClassifier(provider),
        connectors=registry,
        executors=ExecutorPool(),
    )


# --- polling ---------------------------------------------------------------

async def test_a_poll_runs_the_same_ingest_as_every_other_trigger(db):
    connector = PollingConnector([make_item(source_type="stub", source_id="page-1")])
    await poll_source(SinglePool(db), services_for(connector), "stub")

    assert await repo.get_item(db, "stub:page-1") is not None
    assert (await repo.get_source_state(db, "stub"))["cursor"] == "cursor-2"


# --- backfill --------------------------------------------------------------

async def test_the_backfill_crawls_a_wide_window_regardless_of_the_cursor(db):
    """Notion access is opt-in per page: a newly shared page predates the cursor.

    The window is measured from the real clock, not from the stored cursor --
    which is the behaviour under test, and also why this asserts against
    `datetime.now` rather than the fixed NOW the state is seeded with. An
    assertion pinned to a hardcoded date passes on the day it is written and
    fails quietly a week later, which is worse than no test at all.
    """
    connector = PollingConnector([make_item(source_type="stub", source_id="page-1")])
    await repo.save_source_state(db, "stub", cursor="cursor-1", last_fetch_at=NOW)
    await db.commit()

    await backfill(SinglePool(db), services_for(connector), "stub")

    asked_from = connector.since[0]
    now = datetime.now(UTC)
    assert asked_from < now - timedelta(days=BACKFILL_WINDOW_DAYS - 1)
    assert asked_from > now - timedelta(days=BACKFILL_WINDOW_DAYS + 1)
    assert asked_from != NOW  # not the cursor it was told about


async def test_the_backfill_does_not_advance_the_incremental_cursor(db):
    """It is a safety net running alongside the incremental path, not a replacement."""
    connector = PollingConnector([make_item(source_type="stub", source_id="page-1")])
    await repo.save_source_state(db, "stub", cursor="cursor-1", last_fetch_at=NOW)
    await db.commit()

    await backfill(SinglePool(db), services_for(connector), "stub")

    state = await repo.get_source_state(db, "stub")
    assert state["cursor"] == "cursor-1"
    assert state["last_fetch_at"] == NOW


async def test_a_backfill_over_already_seen_pages_costs_no_model_calls(db):
    """The crawl is affordable only because the ingest guard rejects before classify."""
    item = make_item(source_type="stub", source_id="page-1")
    services = services_for(PollingConnector([item]))
    await poll_source(SinglePool(db), services, "stub")
    calls_after_first_pass = len(services.provider.prompts)

    await backfill(SinglePool(db), services, "stub")

    assert len(services.provider.prompts) == calls_after_first_pass


# --- watch renewal ---------------------------------------------------------

async def test_a_watch_near_expiry_is_renewed_and_its_new_cursor_stored(db):
    connector = PushConnector()
    await renew_watches(SinglePool(db), services_for(connector))

    assert connector.registrations == 1
    state = await repo.get_source_state(db, "pushy")
    assert state["cursor"] == "history-9"
    assert state["watch_expires_at"] is not None


async def test_a_watch_with_time_left_is_not_renewed(db):
    """Renewing every six hours would be pointless churn against the source's API."""
    connector = PushConnector(needs_renewal=False)
    await renew_watches(SinglePool(db), services_for(connector))

    assert connector.registrations == 0


async def test_poll_only_sources_are_left_alone(db):
    connector = PollingConnector()
    await renew_watches(SinglePool(db), services_for(connector))

    assert await repo.get_source_state(db, "stub") == {}


async def test_one_source_failing_to_renew_does_not_stop_the_others(db):
    failing = PushConnector("broken", fails=True)
    healthy = PushConnector("pushy")
    await renew_watches(SinglePool(db), services_for(failing, healthy))

    assert healthy.registrations == 1
    assert (await repo.get_source_state(db, "pushy"))["cursor"] == "history-9"


async def test_a_failed_renewal_does_not_record_a_cursor_it_never_got(db):
    """Storing one would make the next run believe delivery is healthy."""
    failing = PushConnector("broken", fails=True)
    await renew_watches(SinglePool(db), services_for(failing))

    assert await repo.get_source_state(db, "broken") == {}
