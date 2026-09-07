"""The executor interface, symmetric to SourceConnector.

Reconciliation only ever calls `create` / `update` / `delete`. It never reaches
a destination's API or MCP tool directly, which is what lets a second action
type -- an email draft, a Slack reminder -- be added without touching the
lifecycle logic that decides *which* of those three to call.

MCP answers "how do I talk to this destination". It does not replace this
interface: something still has to decide which tool to call and map internal
state into that tool's arguments, and that is what an executor is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from assistant.llm import ToolSchema
from assistant.models import ActionSpec


@dataclass(frozen=True)
class ActionDefinition:
    """One registered action type: what it is, what it needs, how to run it."""

    action_type: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema shown to the operations agent."""

    required_fields: tuple[str, ...]
    """Fields that must each clear the confidence gate on their own. Whole-
    proposal confidence is not enough: an agent can be sure an exam needs
    scheduling and be guessing at the date."""

    temporal_fields: tuple[str, ...] = ()
    """Fields carrying a date/time, checked against the past-date rule. A
    past-dated auto-created event is never correct -- it means stale content or
    a misresolved relative phrase, both of which want a human."""

    def as_tool_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.action_type,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass
class ActionResult:
    action_id: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False


class ActionExecutor(ABC):
    action_type: str

    @abstractmethod
    async def create(self, spec: ActionSpec) -> ActionResult: ...

    @abstractmethod
    async def update(self, action_id: str, spec: ActionSpec) -> ActionResult: ...

    @abstractmethod
    async def delete(self, action_id: str) -> None: ...

    async def close(self) -> None:  # pragma: no cover - trivial
        return None
