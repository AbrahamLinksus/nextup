"""The execution layer: payload mapping, the dry-run default, and MCP resolution.

No database and no server. What is worth testing here is the mapping between an
`ActionSpec` and a destination's payload -- the point where an internally
resolved date becomes something a calendar API will store on a specific day --
plus the two decisions that keep an unconfigured clone safe: dry-run by default,
and a resolution failure that names what the server actually offers.
"""

from __future__ import annotations

import pytest

from assistant.actions import ExecutorPool, get_definition, tool_schemas
from assistant.actions.calendar import (
    ACTION_TYPE,
    CalendarMCPActionExecutor,
    DryRunCalendarExecutor,
    build_event,
    summarize,
)
from assistant.actions.mcp_client import MCPClient, MCPToolUnavailable
from assistant.config import get_settings
from assistant.models import ActionSpec, FieldValue


def spec(**values) -> ActionSpec:
    fields = {
        "title": FieldValue("DBMS Assessment 2", 0.95),
        "start": FieldValue("2026-08-28", 0.9),
    }
    fields.update({name: FieldValue(value, 0.9) for name, value in values.items()})
    return ActionSpec(action_type=ACTION_TYPE, fields=fields, source_item_key="notion:page-1")


class FakeMCP:
    """A server that exposes whatever it is told to and records what it was called with."""

    def __init__(self, tools=("create_event", "update_event", "delete_event"), reply=None):
        self._tools = list(tools)
        self.reply = reply if reply is not None else {"id": "evt-1"}
        self.calls: list[tuple[str, dict]] = []

    def resolve(self, candidates):
        return _stub_client(self._tools).resolve(candidates)

    async def call(self, tool, arguments):
        self.calls.append((tool, arguments))
        return self.reply


def _stub_client(tool_names):
    client = MCPClient.__new__(MCPClient)
    client.command = "fake-server"
    client._tool_names = list(tool_names)  # noqa: SLF001
    return client


# --- payload mapping -------------------------------------------------------

def test_a_bare_date_becomes_local_midnight_not_utc_midnight():
    """00:00 UTC is the previous evening in IST -- the event would land a day early."""
    event = build_event(spec())
    assert event["start"]["dateTime"].startswith("2026-08-28T00:00:00")
    assert event["start"]["timeZone"] == get_settings().timezone


def test_a_missing_end_defaults_to_an_hour_after_start():
    event = build_event(spec(start="2026-08-28T14:00:00"))
    assert event["start"]["dateTime"].startswith("2026-08-28T14:00")
    assert event["end"]["dateTime"].startswith("2026-08-28T15:00")


def test_an_all_day_deadline_uses_dates_not_timestamps():
    """A deadline with no stated time is a date, and sending 00:00 makes it look like one."""
    event = build_event(spec(all_day=True))
    assert event["start"] == {"date": "2026-08-28"}
    assert "dateTime" not in event["start"]


def test_optional_details_are_omitted_rather_than_sent_empty():
    event = build_event(spec())
    assert "location" not in event and "description" not in event

    with_detail = build_event(spec(location="Lab 3", description="Chapters 4-7"))
    assert with_detail["location"] == "Lab 3"


def test_an_unresolvable_start_is_an_error_not_a_guessed_date():
    """Nothing downstream can recover from this, and a guessed date is worse than a failure."""
    with pytest.raises(ValueError):
        build_event(ActionSpec(action_type=ACTION_TYPE, fields={"title": FieldValue("x", 1.0)}))


def test_the_summary_names_the_event_and_when():
    assert "DBMS Assessment 2" in summarize(spec())
    assert "2026-08-28" in summarize(spec())


# --- dry run, the default --------------------------------------------------

async def test_the_dry_run_id_is_marked_so_a_later_update_is_visible_in_the_data():
    """A `dryrun:` id flows into lifecycle state; reconciling against one should be obvious."""
    executor = DryRunCalendarExecutor()
    result = await executor.create(spec())

    assert result.action_id.startswith("dryrun:")
    assert result.dry_run is True


async def test_a_dry_run_records_what_it_would_have_done():
    executor = DryRunCalendarExecutor()
    created = await executor.create(spec())
    await executor.update(created.action_id, spec(start="2026-08-29"))
    await executor.delete(created.action_id)

    assert [operation for operation, _, _ in executor.performed] == ["create", "update", "delete"]
    _, _, updated = executor.performed[1]
    assert updated["start"]["dateTime"].startswith("2026-08-29")


async def test_an_unconfigured_pool_hands_back_the_dry_run_executor():
    """The safe default is structural: no MCP command configured means nothing is written."""
    pool = ExecutorPool()
    executor = await pool.get(ACTION_TYPE)
    assert isinstance(executor, DryRunCalendarExecutor)


async def test_executors_are_cached_rather_than_rebuilt_per_action():
    pool = ExecutorPool()
    assert await pool.get(ACTION_TYPE) is await pool.get(ACTION_TYPE)


async def test_an_unregistered_action_type_raises_rather_than_falling_back():
    pool = ExecutorPool()
    with pytest.raises(KeyError):
        await pool.get("send_carrier_pigeon")


# --- MCP execution ---------------------------------------------------------

async def test_creating_through_mcp_returns_the_servers_event_id():
    """Reconciliation is impossible without it: no id means no later update or delete."""
    client = FakeMCP()
    result = await CalendarMCPActionExecutor(client).create(spec())

    assert result.action_id == "evt-1"
    assert result.dry_run is False
    tool, arguments = client.calls[0]
    assert tool == "create_event"
    assert arguments["summary"] == "DBMS Assessment 2"


async def test_an_update_targets_the_stored_event_rather_than_creating_a_second_one():
    client = FakeMCP()
    await CalendarMCPActionExecutor(client).update("evt-1", spec(start="2026-08-29"))

    tool, arguments = client.calls[0]
    assert tool == "update_event"
    assert arguments["eventId"] == "evt-1"


async def test_a_delete_needs_only_the_id():
    client = FakeMCP()
    await CalendarMCPActionExecutor(client).delete("evt-1")

    tool, arguments = client.calls[0]
    assert tool == "delete_event"
    assert arguments["eventId"] == "evt-1"


async def test_a_nested_event_id_is_still_found():
    client = FakeMCP(reply={"event": {"id": "evt-nested"}})
    result = await CalendarMCPActionExecutor(client).create(spec())
    assert result.action_id == "evt-nested"


async def test_a_response_with_no_id_fails_loudly():
    """Silently accepting it would leave a real event nothing can ever update or delete."""
    client = FakeMCP(reply={"text": "Event created!"})
    with pytest.raises(RuntimeError, match="no event id"):
        await CalendarMCPActionExecutor(client).create(spec())


# --- tool-name resolution --------------------------------------------------

@pytest.mark.parametrize("exposed", ["create_event", "create-event", "Create_Event"])
def test_the_same_operation_is_matched_across_naming_conventions(exposed):
    """Being strict about punctuation makes an otherwise-working server unusable."""
    client = _stub_client([exposed])
    assert client.resolve(("create_event", "create-event")) == exposed


def test_candidates_are_tried_in_order():
    client = _stub_client(["add_event", "create_event"])
    assert client.resolve(("create_event", "add_event")) == "create_event"


def test_an_unmatched_operation_names_what_the_server_does_offer():
    """That list is the information needed to fix the config, so it belongs in the error."""
    client = _stub_client(["list_events", "get_freebusy"])
    with pytest.raises(MCPToolUnavailable) as caught:
        client.resolve(("create_event",))

    assert "list_events" in str(caught.value) and "get_freebusy" in str(caught.value)


# --- the registry ----------------------------------------------------------

def test_the_agents_tool_list_is_derived_from_the_registry():
    names = [schema.name for schema in tool_schemas()]
    assert names == list(dict.fromkeys(names))
    assert ACTION_TYPE in names


def test_every_required_field_carries_its_own_confidence_slot():
    """The gate reads per-field confidence; a required field with nowhere to report it
    could never pass."""
    definition = get_definition(ACTION_TYPE)
    properties = definition.parameters["properties"]
    assert "field_confidence" in properties
    for name in definition.required_fields:
        assert name in properties


def test_an_unregistered_action_type_has_no_definition():
    assert get_definition("send_carrier_pigeon") is None
