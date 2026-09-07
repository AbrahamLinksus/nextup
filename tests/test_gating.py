"""The gate. Every rule here corresponds to a specific way auto-action goes wrong."""

from __future__ import annotations

from datetime import UTC, datetime

from assistant import gating
from assistant.actions import get_definition
from assistant.models import (
    ActionSpec,
    ClassificationResult,
    FieldValue,
    Label,
    QueueReason,
)

NOW = datetime(2026, 8, 22, 9, 0, tzinfo=UTC)
DEFINITION = get_definition("create_calendar_event")


def classification(label=Label.ACTIONABLE_MANDATORY, confidence=0.95):
    return ClassificationResult(label=label, confidence=confidence, rationale="because")


def spec(title=0.95, start=0.95, start_value="2026-08-28"):
    return ActionSpec(
        action_type="create_calendar_event",
        fields={
            "title": FieldValue("DBMS exam", title),
            "start": FieldValue(start_value, start),
        },
    )


def test_a_confident_mandatory_proposal_passes():
    assert gating.evaluate(classification(), spec(), DEFINITION, now=NOW).allowed


def test_optional_never_auto_acts():
    decision = gating.evaluate(
        classification(label=Label.ACTIONABLE_OPTIONAL), spec(), DEFINITION, now=NOW
    )
    assert not decision.allowed
    assert decision.reason is QueueReason.OPTIONAL


def test_low_classification_confidence_queues():
    decision = gating.evaluate(classification(confidence=0.4), spec(), DEFINITION, now=NOW)
    assert decision.reason is QueueReason.LOW_CONFIDENCE


def test_a_confident_label_with_a_guessed_date_still_queues():
    """The failure whole-proposal confidence hides.

    Sure the exam needs scheduling, guessing when. Without per-field confidence
    this proposal would sail through and produce a wrong calendar entry.
    """
    decision = gating.evaluate(
        classification(confidence=0.98), spec(title=0.99, start=0.3), DEFINITION, now=NOW
    )
    assert not decision.allowed
    assert decision.reason is QueueReason.LOW_CONFIDENCE
    assert "'start'" in decision.detail


def test_past_dates_never_auto_act():
    decision = gating.evaluate(
        classification(), spec(start_value="2026-01-05"), DEFINITION, now=NOW
    )
    assert decision.reason is QueueReason.PAST_DATE


def test_no_registered_tool_queues_rather_than_forcing_a_shape():
    decision = gating.evaluate(classification(), None, None, now=NOW)
    assert decision.reason is QueueReason.NO_TOOL_AVAILABLE


def test_injection_overrides_a_perfect_score():
    """A manipulated confidence is never consulted.

    The item the injection test drives got the classifier to 1.00. The override
    runs before any confidence is read, which is the only ordering that works.
    """
    decision = gating.evaluate(
        classification(confidence=1.0),
        spec(title=1.0, start=1.0),
        DEFINITION,
        now=NOW,
        injection_flagged=True,
        injection_reason="matched injection pattern",
    )
    assert not decision.allowed
    assert decision.reason is QueueReason.INJECTION_SUSPECTED


def test_a_missing_required_field_fails_rather_than_defaulting():
    bare = ActionSpec(
        action_type="create_calendar_event", fields={"title": FieldValue("x", 0.99)}
    )
    decision = gating.evaluate(classification(), bare, DEFINITION, now=NOW)
    assert not decision.allowed


def test_an_unresolvable_date_queues():
    decision = gating.evaluate(
        classification(), spec(start_value="whenever it happens"), DEFINITION, now=NOW
    )
    assert not decision.allowed
    assert "did not resolve" in decision.detail
