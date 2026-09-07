"""Database behaviour: the CAS guard, the chunk diff, the queue index, the floor.

These run against a real Postgres because all four are behaviours of the
database rather than of Python. Mocking them would assert that the mock works.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from assistant import repository as repo
from assistant.chunking import chunk_markdown
from assistant.models import (
    ActionSpec,
    ClassificationResult,
    FieldValue,
    Label,
    LifecycleRow,
    QueueReason,
    QueueStatus,
)
from tests.conftest import FakeEmbedder, make_item

pytestmark = pytest.mark.db

T0 = datetime(2026, 8, 22, 9, 0, tzinfo=UTC)


def classification(confidence=0.6, label=Label.ACTIONABLE_MANDATORY):
    return ClassificationResult(label=label, confidence=confidence, rationale="r")


# ---------------------------------------------------------------------------
# The ingest guard
# ---------------------------------------------------------------------------

async def test_a_new_item_is_accepted(db):
    outcome = await repo.upsert_item(db, make_item(edited=T0))
    assert outcome.accepted and outcome.is_new and outcome.content_changed


async def test_a_redelivered_event_is_a_non_event(db):
    """No dedup table. last_edited_at does the job directly."""
    item = make_item(edited=T0)
    await repo.upsert_item(db, item)
    assert not (await repo.upsert_item(db, item)).accepted


async def test_a_late_out_of_order_event_cannot_revert_content(db):
    """Falls out of the CAS for free -- only strictly newer edits are accepted."""
    await repo.upsert_item(db, make_item("# New\n\ncurrent content here", edited=T0))
    stale = make_item("# Old\n\nsuperseded content", edited=T0 - timedelta(hours=3))

    assert not (await repo.upsert_item(db, stale)).accepted
    row = await repo.get_item(db, "notion:page-1")
    assert "current content" in row["content"]


async def test_a_genuine_edit_is_accepted_and_flagged_as_changed(db):
    await repo.upsert_item(db, make_item("# A\n\noriginal text goes here", edited=T0))
    outcome = await repo.upsert_item(
        db, make_item("# A\n\nrewritten text goes here", edited=T0 + timedelta(hours=1))
    )
    assert outcome.accepted and outcome.content_changed and not outcome.is_new


async def test_a_touch_with_no_content_change_is_detected(db):
    """Distinguishes "edited" from "re-saved", so a touch costs no model call."""
    content = "# A\n\nunchanged body text for this test"
    await repo.upsert_item(db, make_item(content, edited=T0))
    outcome = await repo.upsert_item(db, make_item(content, edited=T0 + timedelta(hours=1)))
    assert outcome.accepted and not outcome.content_changed


async def test_normalized_properties_round_trip(db):
    item = make_item(
        deadline=datetime(2026, 8, 28, tzinfo=UTC), status="Not started", urgency_hint="High"
    )
    await repo.upsert_item(db, item)
    row = await repo.get_item(db, item.key)
    assert row["status"] == "Not started"
    assert row["deadline"].date() == datetime(2026, 8, 28).date()


# ---------------------------------------------------------------------------
# Chunk sync
# ---------------------------------------------------------------------------

async def test_only_changed_chunks_are_re_embedded(db):
    embedder = FakeEmbedder()
    doc = "# A\n\nfirst section body\n\n## B\n\nsecond section body"
    outcome = await repo.upsert_item(db, make_item(doc, edited=T0))

    first = await repo.sync_chunks(db, outcome.item_id, chunk_markdown(doc), embedder.embed)
    assert first == 2

    again = await repo.sync_chunks(db, outcome.item_id, chunk_markdown(doc), embedder.embed)
    assert again == 0, "unchanged content must not be re-embedded"

    edited = doc.replace("second section body", "rewritten second section")
    changed = await repo.sync_chunks(db, outcome.item_id, chunk_markdown(edited), embedder.embed)
    assert changed == 1, "only the edited chunk should be re-embedded"


async def test_deleted_chunks_are_removed(db):
    embedder = FakeEmbedder()
    doc = "# A\n\nbody one here\n\n## B\n\nbody two here"
    outcome = await repo.upsert_item(db, make_item(doc, edited=T0))
    await repo.sync_chunks(db, outcome.item_id, chunk_markdown(doc), embedder.embed)

    shrunk = "# A\n\nbody one here"
    await repo.sync_chunks(db, outcome.item_id, chunk_markdown(shrunk), embedder.embed)

    rows = await (
        await db.execute("SELECT chunk_key FROM chunks WHERE item_id = %s", (outcome.item_id,))
    ).fetchall()
    assert [row["chunk_key"] for row in rows] == ["A#0"]


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

async def test_one_pending_entry_per_item_and_chunk(db):
    """A re-edit while unreviewed updates the entry rather than stacking a copy."""
    first = await repo.upsert_queue_entry(
        db, item_key="notion:p", chunk_key="A#0",
        classification=classification(0.5), spec=None, reason=QueueReason.LOW_CONFIDENCE,
    )
    second = await repo.upsert_queue_entry(
        db, item_key="notion:p", chunk_key="A#0",
        classification=classification(0.7), spec=None, reason=QueueReason.LOW_CONFIDENCE,
    )
    assert first == second

    entries = await repo.list_pending(db)
    assert len(entries) == 1
    assert entries[0].classification_confidence == pytest.approx(0.7)


async def test_different_chunks_of_one_item_queue_separately(db):
    """Multiple deadlines in one page are multiple decisions, not one."""
    for chunk_key in ("A#0", "A > B#0"):
        await repo.upsert_queue_entry(
            db, item_key="notion:p", chunk_key=chunk_key,
            classification=classification(), spec=None, reason=QueueReason.LOW_CONFIDENCE,
        )
    assert len(await repo.list_pending(db)) == 2


async def test_resolved_entries_free_the_slot(db):
    entry_id = await repo.upsert_queue_entry(
        db, item_key="notion:p", chunk_key=None,
        classification=classification(), spec=None, reason=QueueReason.OPTIONAL,
    )
    await repo.resolve_queue_entry(
        db, entry_id, status=QueueStatus.DISMISSED, resolution="not interested"
    )
    reopened = await repo.upsert_queue_entry(
        db, item_key="notion:p", chunk_key=None,
        classification=classification(), spec=None, reason=QueueReason.OPTIONAL,
    )
    assert reopened != entry_id
    assert len(await repo.list_pending(db)) == 1


async def test_the_drafted_spec_survives_the_round_trip(db):
    """A queued item is a pre-filled draft to accept, not a blank to redo."""
    spec = ActionSpec(
        action_type="create_calendar_event",
        fields={"title": FieldValue("Exam", 0.9), "start": FieldValue("2026-08-28", 0.4)},
    )
    await repo.upsert_queue_entry(
        db, item_key="notion:p", chunk_key="A#0",
        classification=classification(), spec=spec, reason=QueueReason.LOW_CONFIDENCE,
    )
    restored = ActionSpec.from_payload((await repo.list_pending(db))[0].action_spec)
    assert restored.value("start") == "2026-08-28"
    assert restored.confidence("start") == pytest.approx(0.4)


async def test_dismissals_are_retrievable_for_calibration(db):
    entry_id = await repo.upsert_queue_entry(
        db, item_key="notion:p", chunk_key=None,
        classification=classification(0.55), spec=None, reason=QueueReason.LOW_CONFIDENCE,
    )
    await repo.resolve_queue_entry(
        db, entry_id, status=QueueStatus.DISMISSED, resolution="wrong, this is not an exam"
    )
    dismissed = await repo.list_dismissed(db, since=T0 - timedelta(days=1))
    assert len(dismissed) == 1
    assert dismissed[0].classification_confidence == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Lifecycle and retrieval
# ---------------------------------------------------------------------------

async def test_lifecycle_is_keyed_by_item_and_chunk(db):
    for chunk_key in ("A#0", "A > B#0"):
        await repo.upsert_lifecycle(
            db,
            LifecycleRow(
                item_key="notion:p", chunk_key=chunk_key,
                action_type="create_calendar_event",
                last_classification=Label.ACTIONABLE_MANDATORY,
                last_extracted_date=T0, action_id=f"evt-{chunk_key}", updated_at=T0,
            ),
        )
    rows = await repo.list_lifecycle_for_item(db, "notion:p")
    assert len(rows) == 2


async def test_upcoming_actions_are_read_from_lifecycle_state(db):
    await repo.upsert_lifecycle(
        db,
        LifecycleRow(
            item_key="notion:page-1", chunk_key="A#0",
            action_type="create_calendar_event",
            last_classification=Label.ACTIONABLE_MANDATORY,
            last_extracted_date=T0 + timedelta(days=3), action_id="evt-1", updated_at=T0,
        ),
    )
    inside = await repo.list_upcoming_actions(db, start=T0, end=T0 + timedelta(days=7))
    outside = await repo.list_upcoming_actions(db, start=T0, end=T0 + timedelta(days=1))
    assert len(inside) == 1 and outside == []


async def test_the_similarity_floor_returns_nothing_rather_than_noise(db):
    """There is always a nearest neighbour. That is what the floor is for."""
    embedder = FakeEmbedder()
    doc = "# A\n\nsome indexed content about databases"
    outcome = await repo.upsert_item(db, make_item(doc, edited=T0))
    await repo.sync_chunks(db, outcome.item_id, chunk_markdown(doc), embedder.embed)

    query = await embedder.embed_one("something entirely unrelated")
    assert await repo.search_chunks(db, query, top_k=5, min_similarity=0.999) == []
    assert await repo.search_chunks(db, query, top_k=5, min_similarity=0.0) != []


async def test_search_results_carry_their_source_for_attribution(db):
    embedder = FakeEmbedder()
    doc = "# A\n\ncontent worth citing later"
    outcome = await repo.upsert_item(db, make_item(doc, edited=T0, title="Course page"))
    await repo.sync_chunks(db, outcome.item_id, chunk_markdown(doc), embedder.embed)

    hits = await repo.search_chunks(
        db, await embedder.embed_one("content"), top_k=1, min_similarity=0.0
    )
    assert hits[0]["title"] == "Course page"
    assert hits[0]["url"]


async def test_audit_rows_are_append_only_history(db):
    await repo.record_audit(db, event_type="classification", item_key="notion:p", payload={"c": 1})
    await repo.record_audit(db, event_type="queued", item_key="notion:p", payload={"c": 2})
    rows = await repo.list_audit(db, item_key="notion:p")
    assert [row["event_type"] for row in rows] == ["queued", "classification"]


async def test_source_state_advances_without_clobbering_other_fields(db):
    await repo.save_source_state(db, "gmail", cursor="12345")
    await repo.save_source_state(db, "gmail", last_fetch_at=T0)

    state = await repo.get_source_state(db, "gmail")
    assert state["cursor"] == "12345", "advancing one field must not blank the others"
    assert state["last_fetch_at"] == T0


async def test_timestamps_come_back_in_the_configured_timezone(db):
    """A timestamptz is an instant either way, but what it *renders as* gets read
    by people and quoted by the model -- and 15 September 00:00 IST rendered in
    UTC reports as the 14th, which is the wrong day."""
    from assistant.config import get_settings

    await repo.upsert_lifecycle(
        db,
        LifecycleRow(
            item_key="notion:fees",
            chunk_key="A#0",
            action_type="create_calendar_event",
            last_classification=Label.ACTIONABLE_MANDATORY,
            last_extracted_date=datetime(2026, 9, 15, 0, 0, tzinfo=get_settings().tz),
            action_id="evt-1",
            updated_at=datetime.now(UTC),
        ),
    )
    row = await repo.get_lifecycle(db, "notion:fees", "A#0")

    assert row.last_extracted_date.date().isoformat() == "2026-09-15"
    assert row.last_extracted_date.hour == 0
