"""Composition: batching, transaction boundaries, and when the cursor moves.

There is no logic here to speak of, which is the point -- the scheduled poll,
the webhook, and the agent's fetch tool all run the *same* pipeline through this
module. What is worth testing is the two things that would quietly lose data if
they were wrong: one item's failure rolling back twelve that succeeded, and a
cursor advancing past items that never committed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from assistant import repository as repo
from assistant.actions import ExecutorPool
from assistant.actions.calendar import DryRunCalendarExecutor
from assistant.classify import SingleCallClassifier
from assistant.connectors import ConnectorRegistry
from assistant.connectors.base import SourceConnector
from assistant.pipeline import PipelineOutcome
from assistant.runner import ingest_changed_ids, ingest_source, process_items, summarize
from tests.conftest import FakeEmbedder, FakeProvider, make_item

pytestmark = pytest.mark.db

NOW = datetime(2026, 8, 22, 9, 0, tzinfo=UTC)
INFORMATIONAL = {"label": "informational", "confidence": 0.95, "rationale": "notes"}


class StubConnector(SourceConnector):
    """Records what it was asked for and what cursor it was handed."""

    source_type = "stub"

    def __init__(self, items=None, *, cursor="cursor-2"):
        self.items = list(items or [])
        self.cursor = cursor
        self.loaded: list[dict] = []
        self.since: list[datetime | None] = []
        self.fetched: list[str] = []

    async def list_items(self, since=None, *, limit=100):
        self.since.append(since)
        return self.items

    async def fetch_item(self, item_id):
        self.fetched.append(item_id)
        return make_item(source_type="stub", source_id=item_id)

    async def load_state(self, state):
        self.loaded.append(dict(state))

    def dump_state(self):
        return {"cursor": self.cursor}


class ExplodingProvider(FakeProvider):
    """Fails to classify one specific item, succeeds on the rest."""

    def __init__(self, poison: str, responses=None):
        super().__init__(responses)
        self.poison = poison

    async def structured(self, **kwargs):
        if self.poison in kwargs["prompt"]:
            raise RuntimeError("the model fell over")
        return await super().structured(**kwargs)


def collaborators(provider=None):
    provider = provider or FakeProvider([INFORMATIONAL] * 10)
    pool = ExecutorPool()
    pool._executors["create_calendar_event"] = DryRunCalendarExecutor()  # noqa: SLF001
    return {
        "classifier": SingleCallClassifier(provider),
        "provider": provider,
        "embedder": FakeEmbedder(),
        "executors": pool,
    }


def registry_with(connector) -> ConnectorRegistry:
    registry = ConnectorRegistry()
    registry.register(connector, poll_interval_hours=1)
    return registry


# --- batching --------------------------------------------------------------

async def test_each_item_is_committed_on_its_own(db):
    items = [make_item(source_id=f"page-{n}") for n in range(3)]
    outcomes = await process_items(db, items, **collaborators())

    assert len(outcomes) == 3
    for item in items:
        assert await repo.get_item(db, item.key) is not None


async def test_one_failing_item_does_not_roll_back_the_ones_before_it(db):
    """A poll of twelve pages should not be lost because the seventh crashed."""
    good = make_item("Notes on paging and TLB behaviour, for reference.", source_id="good")
    bad = make_item("Notes that make the model fall over: poison.", source_id="bad")
    later = make_item("More notes, written after the failure.", source_id="later")

    outcomes = await process_items(
        db, [good, bad, later], **collaborators(ExplodingProvider("poison"))
    )

    assert await repo.get_item(db, good.key) is not None
    assert await repo.get_item(db, later.key) is not None
    assert [o.skipped for o in outcomes] == [False, True, False]


async def test_a_failed_item_leaves_nothing_half_written(db):
    """Its row is inserted before classification runs, so the rollback has to remove it."""
    bad = make_item("Notes that make the model fall over: poison.", source_id="bad")
    await process_items(db, [bad], **collaborators(ExplodingProvider("poison")))

    assert await repo.get_item(db, bad.key) is None


async def test_the_failure_reason_travels_with_the_outcome(db):
    bad = make_item("Notes that make the model fall over: poison.", source_id="bad")
    outcomes = await process_items(db, [bad], **collaborators(ExplodingProvider("poison")))

    assert "the model fell over" in outcomes[0].skip_reason


# --- source ingestion ------------------------------------------------------

async def test_the_stored_cursor_is_handed_to_the_connector_before_fetching(db):
    connector = StubConnector()
    await repo.save_source_state(db, "stub", cursor="cursor-1", last_fetch_at=NOW)
    await db.commit()

    await ingest_source(db, registry_with(connector), "stub", **collaborators())

    assert connector.loaded[0]["cursor"] == "cursor-1"
    assert connector.since[0] == NOW


async def test_the_cursor_advances_only_after_the_items_it_covers_are_stored(db):
    """Advancing first would, on a crash, skip precisely the items that failed."""
    connector = StubConnector([make_item(source_type="stub", source_id="page-1")])
    await ingest_source(db, registry_with(connector), "stub", **collaborators())

    state = await repo.get_source_state(db, "stub")
    assert state["cursor"] == "cursor-2"
    assert state["last_fetch_at"] is not None
    assert await repo.get_item(db, "stub:page-1") is not None


async def test_an_empty_fetch_still_advances_the_clock(db):
    """Nothing new is a successful poll, and re-asking from the old point wastes the next one."""
    await ingest_source(db, registry_with(StubConnector([])), "stub", **collaborators())

    state = await repo.get_source_state(db, "stub")
    assert state["last_fetch_at"] is not None


async def test_an_unregistered_source_is_an_error_not_a_silent_no_op(db):
    with pytest.raises(ValueError, match="no connector registered"):
        await ingest_source(db, ConnectorRegistry(), "stub", **collaborators())


# --- the push path ---------------------------------------------------------

async def test_push_fetches_the_content_rather_than_reading_it_from_the_payload(db):
    """Both sources' payloads are notification-only: they say something changed, never what."""
    connector = StubConnector()
    summary = await ingest_changed_ids(
        db, registry_with(connector), "stub", ["msg-1", "msg-2"], **collaborators()
    )

    assert connector.fetched == ["msg-1", "msg-2"]
    assert summary["fetched"] == 2
    assert await repo.get_item(db, "stub:msg-1") is not None


async def test_the_push_path_does_not_move_the_poll_clock(db):
    """`last_fetch_at` means 'everything changed before this was seen', which a push does not."""
    connector = StubConnector()
    await ingest_changed_ids(db, registry_with(connector), "stub", ["msg-1"], **collaborators())

    state = await repo.get_source_state(db, "stub")
    assert state["cursor"] == "cursor-2"
    assert state["last_fetch_at"] is None


# --- reporting -------------------------------------------------------------

def test_the_summary_separates_processed_from_skipped():
    outcomes = [
        PipelineOutcome(item_key="a", actions_taken=["created x"]),
        PipelineOutcome(item_key="b", queued=["queued y"]),
        PipelineOutcome(item_key="c", skipped=True, skip_reason="duplicate"),
    ]
    summary = summarize("stub", outcomes)

    assert summary == {
        "source": "stub",
        "fetched": 3,
        "processed": 2,
        "skipped": 1,
        "actions_taken": ["created x"],
        "queued": ["queued y"],
        "reconciled": [],
    }
