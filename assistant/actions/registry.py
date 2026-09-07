"""ACTION_REGISTRY -- the set of things the system knows how to do.

Registry-based rather than a fixed calendar-shaped extraction schema, because
binding the pipeline to calendar events would mean every future action type
(draft a reply, set a reminder, file a document) reopens classify and
reconciliation. Here, a new action type is one `ActionDefinition` plus an
`ActionExecutor`; nothing upstream changes.

The unregistered case is handled by construction. The operations agent is shown
exactly these tools and nothing else, so "no registered tool fits" is simply the
absence of a call -- an item that needs something we cannot do queues with its
reason recorded, instead of being forced into the nearest available shape or
dropped.
"""

from __future__ import annotations

from typing import Any

from assistant.actions.base import ActionDefinition, ActionExecutor
from assistant.actions.calendar import (
    ACTION_TYPE as CALENDAR_ACTION_TYPE,
)
from assistant.actions.calendar import (
    CalendarMCPActionExecutor,
    DryRunCalendarExecutor,
)
from assistant.actions.mcp_client import MCPClient
from assistant.llm import ToolSchema

# Per-field confidence rides alongside the arguments rather than wrapping each
# one, so the action's own schema stays readable as the thing it describes. The
# gate reads this map; a field absent from it scores zero and fails.
_FIELD_CONFIDENCE_SCHEMA = {
    "type": "object",
    "description": (
        "Your confidence, 0.0-1.0, in each argument you filled -- keyed by "
        "argument name. Be honest and specific: being certain an exam needs "
        "scheduling while guessing its date should show as a high confidence "
        "for 'title' and a low one for 'start'. A low score here routes the "
        "item to the user for review, which is the correct outcome when you "
        "are unsure; it is not a failure."
    ),
    "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0},
}

CALENDAR_EVENT = ActionDefinition(
    action_type=CALENDAR_ACTION_TYPE,
    description=(
        "Create a calendar event for an obligation that happens at a specific "
        "date, and optionally a specific time. Use this for exams, submission "
        "deadlines, appointments, and scheduled sessions. Do not use it for "
        "vague future intentions with no settled date."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short event title, e.g. 'DBMS Assessment 2'.",
            },
            "start": {
                "type": "string",
                "description": (
                    "ISO-8601 start. Use a full timestamp when a time is stated "
                    "('2026-08-28T14:00:00'), a bare date otherwise "
                    "('2026-08-28'). Prefer a date already resolved in the "
                    "content in square brackets over doing your own arithmetic."
                ),
            },
            "end": {
                "type": "string",
                "description": "ISO-8601 end. Omit to default to one hour after start.",
            },
            "all_day": {
                "type": "boolean",
                "description": "True when a deadline has a date but no time of day.",
            },
            "location": {"type": "string", "description": "Where, if the content says."},
            "description": {
                "type": "string",
                "description": "Details worth carrying onto the event, e.g. syllabus scope.",
            },
            "field_confidence": _FIELD_CONFIDENCE_SCHEMA,
        },
        "required": ["title", "start", "field_confidence"],
        "additionalProperties": False,
    },
    required_fields=("title", "start"),
    temporal_fields=("start",),
)


ACTION_REGISTRY: dict[str, ActionDefinition] = {
    CALENDAR_EVENT.action_type: CALENDAR_EVENT,
}


def tool_schemas() -> list[ToolSchema]:
    """What the operations agent is allowed to propose."""
    return [definition.as_tool_schema() for definition in ACTION_REGISTRY.values()]


def get_definition(action_type: str) -> ActionDefinition | None:
    return ACTION_REGISTRY.get(action_type)


def register(definition: ActionDefinition) -> None:
    """Add an action type. One call is the whole cost of a new destination."""
    ACTION_REGISTRY[definition.action_type] = definition


class ExecutorPool:
    """Lazily builds and caches one executor per action type.

    Lazy because connecting is a subprocess launch: importing the registry to
    read a tool schema should not start an MCP server, and the common pipeline
    run -- everything classified informational -- never needs an executor at all.
    """

    def __init__(self, mcp_client: Any | None = None) -> None:
        self._mcp = mcp_client
        self._mcp_connected = False
        self._executors: dict[str, ActionExecutor] = {}

    async def get(self, action_type: str) -> ActionExecutor:
        if action_type in self._executors:
            return self._executors[action_type]
        if action_type not in ACTION_REGISTRY:
            raise KeyError(f"no registered action type {action_type!r}")

        executor = await self._build(action_type)
        self._executors[action_type] = executor
        return executor

    async def _build(self, action_type: str) -> ActionExecutor:
        if action_type != CALENDAR_ACTION_TYPE:
            raise KeyError(f"no executor wired for action type {action_type!r}")

        if self._mcp is None:
            self._mcp = MCPClient.from_settings()
        if self._mcp is None:
            return DryRunCalendarExecutor()

        if not self._mcp_connected:
            await self._mcp.connect()
            self._mcp_connected = True
        return CalendarMCPActionExecutor(self._mcp)

    async def close(self) -> None:
        for executor in self._executors.values():
            await executor.close()
        if self._mcp is not None and self._mcp_connected:
            await self._mcp.close()
            self._mcp_connected = False
        self._executors.clear()
