"""Action proposal targets and their executors."""

from assistant.actions.base import ActionDefinition, ActionExecutor, ActionResult
from assistant.actions.calendar import (
    CalendarMCPActionExecutor,
    DryRunCalendarExecutor,
)
from assistant.actions.registry import (
    ACTION_REGISTRY,
    ExecutorPool,
    get_definition,
    register,
    tool_schemas,
)

__all__ = [
    "ACTION_REGISTRY",
    "ActionDefinition",
    "ActionExecutor",
    "ActionResult",
    "CalendarMCPActionExecutor",
    "DryRunCalendarExecutor",
    "ExecutorPool",
    "get_definition",
    "register",
    "tool_schemas",
]
