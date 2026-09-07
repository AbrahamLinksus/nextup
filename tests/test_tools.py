"""The agent's tool surface.

Two things matter here beyond "the handler works". The surface is *derived* --
registering a source has to make it callable with no edit in tools.py -- and a
tool failure has to come back as a string the model can read, because a raised
exception inside a tool loop ends the user's turn with nothing.

The promotion path gets the most attention: it is the one place where a
user-approved action has to end up under the same reconciliation rules as an
automatic one, and the only thing that makes that true is the lifecycle row it
writes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from assistant import repository as repo
from assistant.actions import ExecutorPool
from assistant.actions.calendar import DryRunCalendarExecutor
from assistant.classify import SingleCallClassifier
from assistant.connectors import ConnectorRegistry
from assistant.connectors.base import SourceConnector
from assistant.models import (
    ActionSpec,
    ClassificationResult,
    FieldValue,
    Label,
    LifecycleRow,
    QueueReason,
    QueueStatus,
)
from assistant.tools import ToolBox
from tests.conftest import FakeEmbedder, FakeProvider, make_item

pytestmark = pytest.mark.db

NOW = datetime(2026, 8, 22, 9, 0, tzinfo=UTC)


class StubConnector(SourceConnector):
    """A source with one item, so the derived fetch tool has something to do."""

    source_type = "stub"

    def __init__(self, items=None):
        self.items = items if items is not None else [make_item(source_type="stub")]
        self.since: list[datetime | None] = []

    async def list_items(self, since=None, *, limit=100):
        self.since.append(since)
        return self.items

    async def fetch_item(self, item_id):  # pragma: no cover - unused here
        raise AssertionError("not needed")


def draft(start="2026-08-28", start_confidence=0.4) -> ActionSpec:
    return ActionSpec(
        action_type="create_calendar_event",
        fields={
            "title": FieldValue("DBMS Assessment 2", 0.95),
            "start": FieldValue(start, start_confidence),
        },
        source_item_key="notion:page-1",
    )


def classification(confidence=0.9, label=Label.ACTIONABLE_MANDATORY) -> ClassificationResult:
    return ClassificationResult(label=label, confidence=confidence, rationale="test")


class Box:
    """A ToolBox with a scripted model and an inspectable executor."""

    def __init__(self, db, *, connectors=None, structured=None, tool_calls=None):
        self.executor = DryRunCalendarExecutor()
        pool = ExecutorPool()
        pool._executors["create_calendar_event"] = self.executor  # noqa: SLF001
        self.provider = FakeProvider(structured, tool_calls=tool_calls)
        self.connectors = connectors or ConnectorRegistry()
        self.tools = ToolBox(
            db,
            connectors=self.connectors,
            embedder=FakeEmbedder(),
            provider=self.provider,
            classifier=SingleCallClassifier(self.provider),
            executors=pool,
        )

    async def call(self, name, **arguments):
        return await self.tools.call(name, arguments)


async def queued(db, *, spec=..., reason=QueueReason.LOW_CONFIDENCE, chunk_key="A#0") -> UUID:
    spec = draft() if spec is ... else spec
    entry_id = await repo.upsert_queue_entry(
        db,
        item_key="notion:page-1",
        chunk_key=chunk_key,
        classification=classification(),
        spec=spec,
        reason=reason,
    )
    await db.commit()
    return entry_id


# --- the surface -----------------------------------------------------------

async def test_a_registered_source_becomes_a_fetch_tool_with_no_edit_here(db):
    """One tool per connector is what lets 'check my mail' resolve to a single call."""
    registry = ConnectorRegistry()
    registry.register(StubConnector(), poll_interval_hours=1)
    box = Box(db, connectors=registry)

    assert "fetch_stub" in {schema.name for schema in box.tools.schemas()}


async def test_an_unknown_tool_name_answers_rather_than_raising(db):
    """A hallucinated tool name inside the loop must not end the user's turn."""
    result = await Box(db).call("fetch_the_moon")

    assert "No such tool" in result
    assert "search_content" in result


async def test_a_failing_tool_returns_the_failure_as_a_result(db):
    box = Box(db)
    result = await box.call("dismiss_queue_entry", entry_id="not-a-uuid")

    assert result.startswith("Tool 'dismiss_queue_entry' failed")


async def test_the_fetch_tool_runs_the_full_pipeline_not_a_bare_fetch(db):
    """The trigger type never changes downstream processing -- on demand included."""
    registry = ConnectorRegistry()
    connector = StubConnector()
    registry.register(connector, poll_interval_hours=1)
    box = Box(db, connectors=registry, structured=[{"label": "informational", "confidence": 0.9}])

    summary = json.loads(await box.call("fetch_stub"))

    assert summary["source"] == "stub"
    assert summary["processed"] == 1
    assert await repo.get_item(db, connector.items[0].key) is not None


# --- reading state ---------------------------------------------------------

async def test_an_empty_queue_says_so_plainly(db):
    assert "empty" in await Box(db).call("list_pending_queue")


async def test_a_queued_entry_is_returned_with_the_action_that_was_drafted(db):
    """The user is approving a specific event, so the draft has to be visible."""
    entry_id = await queued(db)
    entries = json.loads(await Box(db).call("list_pending_queue"))

    assert entries[0]["entry_id"] == str(entry_id)
    assert entries[0]["queued_because"] == "low_confidence"
    assert entries[0]["drafted_action"]["fields"]["start"]["value"] == "2026-08-28"


async def test_the_queue_can_be_filtered_by_why_something_was_queued(db):
    await queued(db, reason=QueueReason.OPTIONAL, chunk_key="A#0")
    await queued(db, reason=QueueReason.LOW_CONFIDENCE, chunk_key="A#1")

    filtered = json.loads(await Box(db).call("list_pending_queue", reason="optional"))
    assert [entry["queued_because"] for entry in filtered] == ["optional"]


async def test_upcoming_actions_are_read_from_scheduled_state_not_searched_for(db):
    await repo.upsert_item(db, make_item(source_id="page-1"))
    await repo.upsert_lifecycle(
        db,
        LifecycleRow(
            item_key="notion:page-1",
            chunk_key="A#0",
            action_type="create_calendar_event",
            last_classification=Label.ACTIONABLE_MANDATORY,
            last_extracted_date=datetime(2026, 8, 28, 9, 0, tzinfo=UTC),
            action_id="evt-1",
            updated_at=NOW,
        ),
    )
    await db.commit()

    rows = json.loads(
        await Box(db).call("list_upcoming_actions", start_date="2026-08-25", end_date="2026-09-01")
    )
    assert rows[0]["action_id"] == "evt-1"


async def test_an_empty_range_distinguishes_nothing_scheduled_from_a_failure(db):
    """'Nothing is scheduled' and 'I could not find out' are different answers."""
    answer = await Box(db).call(
        "list_upcoming_actions", start_date="2026-08-25", end_date="2026-09-01"
    )
    assert "Nothing scheduled" in answer


async def test_a_range_with_no_dates_asks_for_them_rather_than_guessing(db):
    assert "required" in await Box(db).call("list_upcoming_actions", start_date="2026-08-25")


async def test_search_with_no_matching_content_returns_the_grounding_instruction(db):
    answer = await Box(db).call("search_content", query="anything at all")
    assert "do not answer from" in answer.lower()


async def test_dismissals_are_readable_back_for_calibration(db):
    entry_id = await queued(db)
    await Box(db).call("dismiss_queue_entry", entry_id=str(entry_id), note="already done")

    dismissed = json.loads(await Box(db).call("list_dismissed", since_days=1))
    assert dismissed[0]["resolution"] == "already done"


# --- acting ----------------------------------------------------------------

async def test_promoting_executes_the_drafted_action(db):
    entry_id = await queued(db)
    box = Box(db)
    answer = await box.call("promote_queue_entry", entry_id=str(entry_id), note="go ahead")

    assert [operation for operation, _, _ in box.executor.performed] == ["create"]
    assert "DBMS Assessment 2" in answer


async def test_a_promoted_action_lands_under_the_same_reconciliation_rules(db):
    """Without the lifecycle row, a later edit to the source could never update it."""
    entry_id = await queued(db)
    box = Box(db)
    await box.call("promote_queue_entry", entry_id=str(entry_id))

    row = await repo.get_lifecycle(db, "notion:page-1", "A#0")
    assert row is not None
    assert row.action_id == box.executor.performed[0][1]
    from assistant.dates import to_local

    assert to_local(row.last_extracted_date).date().isoformat() == "2026-08-28"


async def test_promoting_resolves_the_entry_so_it_stops_being_pending(db):
    entry_id = await queued(db)
    await Box(db).call("promote_queue_entry", entry_id=str(entry_id))

    entry = await repo.get_queue_entry(db, entry_id)
    assert entry.status is QueueStatus.PROMOTED
    assert await repo.list_pending(db) == []


async def test_promoting_twice_does_not_create_a_second_event(db):
    """The second attempt is a mistake -- a duplicate calendar entry is the cost."""
    entry_id = await queued(db)
    box = Box(db)
    await box.call("promote_queue_entry", entry_id=str(entry_id))
    second = await box.call("promote_queue_entry", entry_id=str(entry_id))

    assert "already" in second
    assert len(box.executor.performed) == 1


async def test_promoting_an_entry_with_no_draft_explains_rather_than_inventing_one(db):
    """Queued with no tool available means there is nothing to approve."""
    entry_id = await queued(db, spec=None, reason=QueueReason.NO_TOOL_AVAILABLE)
    box = Box(db)
    answer = await box.call("promote_queue_entry", entry_id=str(entry_id))

    assert "no drafted action" in answer
    assert box.executor.performed == []


async def test_promoting_something_that_does_not_exist_says_so(db):
    assert "No queue entry" in await Box(db).call("promote_queue_entry", entry_id=str(uuid4()))


async def test_a_direct_instruction_schedules_without_passing_the_confidence_gate(db):
    """The gate substitutes for a human. When the user asks directly, the human is present."""
    box = Box(db)
    answer = await box.call(
        "schedule_event", title="Dentist", start="2026-09-02T11:00:00", location="Clinic"
    )

    assert "Dentist" in answer
    _, _, event = box.executor.performed[0]
    assert event["location"] == "Clinic"


async def test_a_directly_scheduled_event_is_still_recorded_in_the_audit_log(db):
    box = Box(db)
    await box.call("schedule_event", title="Dentist", start="2026-09-02T11:00:00")

    audit = await repo.list_audit(db, item_key="conversation")
    assert audit and audit[0]["event_type"] == "action_taken"


async def test_remembering_stores_and_indexes_without_creating_an_action(db):
    """Direct input is declared memory, not a fetched item to judge for action."""
    box = Box(db)
    answer = await box.call(
        "remember", content="Prof said the viva is oral only.", title="Viva format"
    )

    assert "Viva format" in answer
    assert box.executor.performed == []
    assert box.provider.prompts == []  # nothing was classified


async def test_remembering_the_same_note_twice_updates_one_row(db):
    """Near-duplicates would both surface in retrieval, which is worse than either alone."""
    box = Box(db)
    note = "Prof said the viva is oral only."
    await box.call("remember", content=note)
    second = await box.call("remember", content=note)

    assert "Already stored" in second and "nothing changed" in second


async def test_remembered_content_is_findable_afterwards(db):
    box = Box(db)
    await box.call("remember", content="The viva is oral only.", title="Viva format")

    hits = await box.call("search_content", query="The viva is oral only.")
    assert "Viva format" in hits
