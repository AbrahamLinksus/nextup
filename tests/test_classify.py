"""Classification: thresholds, the pre-filter, aggregation, and bad model output."""

from __future__ import annotations

from assistant.chunking import chunk_markdown
from assistant.classify import (
    SelfConsistencyClassifier,
    SingleCallClassifier,
    classify_item,
    should_skip_classification,
)
from assistant.models import Label
from tests.conftest import FakeProvider, make_item

MANDATORY = {"label": "actionable-mandatory", "confidence": 0.92, "rationale": "explicit date"}
INFORMATIONAL = {"label": "informational", "confidence": 0.9, "rationale": "notes"}
OPTIONAL = {"label": "actionable-optional", "confidence": 0.6, "rationale": "maybe"}


async def test_a_single_call_produces_a_label_and_confidence():
    result = await SingleCallClassifier(FakeProvider([MANDATORY])).classify(make_item())
    assert result.label is Label.ACTIONABLE_MANDATORY
    assert result.confidence == 0.92


async def test_an_unrecognized_label_becomes_informational_at_zero_confidence():
    """Never guess. Zero confidence routes to the queue under every rule."""
    provider = FakeProvider([{"label": "URGENT!!", "confidence": 0.99, "rationale": "x"}])
    result = await SingleCallClassifier(provider).classify(make_item())
    assert result.label is Label.INFORMATIONAL
    assert result.confidence == 0.0


async def test_confidence_is_clamped_to_the_unit_interval():
    provider = FakeProvider([{"label": "informational", "confidence": 4.2, "rationale": "x"}])
    assert (await SingleCallClassifier(provider).classify(make_item())).confidence == 1.0


def test_trivial_content_skips_classification_entirely():
    assert should_skip_classification(make_item("ok"))


def test_short_content_with_a_property_signal_is_still_classified():
    """A one-line block with a due date is exactly what must not be skipped."""
    from datetime import UTC, datetime

    item = make_item("Exam", deadline=datetime(2026, 8, 28, tzinfo=UTC))
    assert not should_skip_classification(item)


async def test_skipping_is_recorded_as_informational_not_as_an_error():
    result, per_chunk = await classify_item(
        make_item("ok"), [], SingleCallClassifier(FakeProvider())
    )
    assert result.label is Label.INFORMATIONAL
    assert per_chunk == {}


async def test_the_item_takes_the_highest_consequence_chunk_label():
    doc = "# A\n\nSome general notes about the course.\n\n## B\n\nThe exam is on 28 August 2026."
    chunks = chunk_markdown(doc)
    provider = FakeProvider([INFORMATIONAL, MANDATORY])

    result, per_chunk = await classify_item(
        make_item(doc), chunks, SingleCallClassifier(provider)
    )
    assert result.label is Label.ACTIONABLE_MANDATORY
    assert result.triggering_chunk_key == chunks[1].chunk_key
    assert len(per_chunk) == 2


async def test_the_triggering_chunk_is_identified_for_extraction():
    """Extraction runs against the chunk that triggered, not the whole page."""
    doc = "# A\n\nnotes here that are long enough\n\n## B\n\nexam on 28 August 2026"
    chunks = chunk_markdown(doc)
    result, _ = await classify_item(
        make_item(doc), chunks, SingleCallClassifier(FakeProvider([INFORMATIONAL, MANDATORY]))
    )
    assert result.triggering_chunk_key == "A > B#0"


async def test_the_prefilter_skips_obviously_irrelevant_chunks_on_large_items():
    doc = "\n\n".join(
        f"## Section {i}\n\nGeneral prose about topic {i} with no obligation in it."
        for i in range(8)
    ) + "\n\n## Exam\n\nThe exam is due on 28 August 2026."
    chunks = chunk_markdown(doc)
    provider = FakeProvider([MANDATORY])

    result, per_chunk = await classify_item(
        make_item(doc), chunks, SingleCallClassifier(provider)
    )
    assert len(per_chunk) == 1, "only the chunk with a deadline signal should cost a call"
    assert result.label is Label.ACTIONABLE_MANDATORY


async def test_small_items_are_never_prefiltered():
    """Below the chunk-count threshold, everything is classified.

    The pre-filter exists to stop a 60-chunk page costing 60 calls, not to save
    two -- and a keyword filter deciding labels would defeat the classifier.
    """
    doc = "# A\n\nsomething with no keywords at all here\n\n## B\n\nmore of the same"
    chunks = chunk_markdown(doc)
    _, per_chunk = await classify_item(
        make_item(doc), chunks, SingleCallClassifier(FakeProvider([INFORMATIONAL, INFORMATIONAL]))
    )
    assert len(per_chunk) == len(chunks)


async def test_self_consistency_reports_agreement_not_self_report():
    """Confidence becomes something the model did, not something it claimed."""
    provider = FakeProvider([MANDATORY, MANDATORY, OPTIONAL])
    result = await SelfConsistencyClassifier(provider, runs=3).classify(make_item())

    assert result.label is Label.ACTIONABLE_MANDATORY
    assert result.confidence == 2 / 3
    assert result.raw_runs is not None and len(result.raw_runs) == 3


async def test_v1_leaves_raw_runs_empty_so_the_audit_schema_never_changes():
    result = await SingleCallClassifier(FakeProvider([MANDATORY])).classify(make_item())
    assert result.raw_runs is None
