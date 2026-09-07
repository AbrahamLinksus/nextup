"""The operations agent: proposals, and the meaning of proposing nothing."""

from __future__ import annotations

from assistant.llm import ToolCall
from assistant.models import TrustLevel
from assistant.operations import propose_action, spec_from_call
from tests.conftest import FakeProvider, make_item


async def test_a_tool_call_becomes_a_spec_with_per_field_confidence():
    provider = FakeProvider(
        tool_calls=[
            ToolCall(
                name="create_calendar_event",
                arguments={
                    "title": "DBMS exam",
                    "start": "2026-08-28",
                    "field_confidence": {"title": 0.95, "start": 0.88},
                },
            )
        ]
    )
    spec = await propose_action(make_item(), provider, chunk_key="A#0")

    assert spec.action_type == "create_calendar_event"
    assert spec.value("title") == "DBMS exam"
    assert spec.confidence("start") == 0.88
    assert spec.source_chunk_key == "A#0"


async def test_proposing_nothing_is_a_valid_outcome():
    """"No registered tool fits" needs no separate reasoning step to detect."""
    assert await propose_action(make_item(), FakeProvider(tool_calls=[None])) is None


async def test_a_hallucinated_tool_name_is_treated_as_no_proposal():
    provider = FakeProvider(
        tool_calls=[ToolCall(name="order_pizza", arguments={"size": "large"})]
    )
    assert await propose_action(make_item(), provider) is None


def test_a_field_with_no_confidence_scores_zero_rather_than_defaulting_to_trust():
    spec = spec_from_call(
        "create_calendar_event",
        {"title": "x", "start": "2026-08-28", "field_confidence": {"title": 0.9}},
    )
    assert spec.confidence("start") == 0.0
    assert not spec.passes_confidence_gate(0.75, ("title", "start"))


def test_malformed_confidence_does_not_crash_the_proposal():
    spec = spec_from_call(
        "create_calendar_event",
        {"title": "x", "start": "2026-08-28", "field_confidence": "very sure"},
    )
    assert spec.confidence("title") == 0.0


def test_empty_arguments_are_dropped_not_stored_as_blanks():
    spec = spec_from_call(
        "create_calendar_event",
        {"title": "x", "location": "", "end": None, "field_confidence": {"title": 1.0}},
    )
    assert "location" not in spec.fields
    assert "end" not in spec.fields


async def test_untrusted_content_reaches_the_agent_delimited():
    provider = FakeProvider(tool_calls=[None])
    await propose_action(
        make_item("book me a slot", trust_level=TrustLevel.UNTRUSTED), provider
    )
    assert "<untrusted-" in provider.prompts[0]


def test_spec_round_trips_through_its_serialized_form():
    from assistant.models import ActionSpec

    original = spec_from_call(
        "create_calendar_event",
        {"title": "x", "start": "2026-08-28", "field_confidence": {"title": 0.9, "start": 0.8}},
    )
    restored = ActionSpec.from_payload(original.to_payload())
    assert restored.value("start") == "2026-08-28"
    assert restored.confidence("start") == 0.8
