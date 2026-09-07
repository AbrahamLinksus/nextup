"""The source-connector interface and the trigger layer.

Nothing downstream of `list_items` / `fetch_item` branches on `source_type`.
That is the whole point of the abstraction, and it is the reason the pipeline,
the storage layer, and the retrieval layer never learn that Gmail exists.

Two things live here that the original prototype did not have, both of which
later decisions made necessary:

* `trust_level` -- Notion is the user's own workspace, Gmail is whatever the
  internet sent. Both arrive as `Item`s, so the distinction has to travel with
  the connector or it is lost exactly where it matters.
* async -- every layer above this awaits (Postgres, Ollama, MCP). A synchronous
  connector would block the loop during a multi-page Notion crawl.

Trigger strategy is per-connector rather than global, because urgency lives
asymmetrically: self-authored Notion content tolerates a staleness window that
an inbound email deadline does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import structlog

from assistant.llm import ToolSchema
from assistant.models import Item, TrustLevel

log = structlog.get_logger(__name__)


class SourceConnector(ABC):
    """Every source implements this. Nothing above it talks to a native API."""

    source_type: str
    supports_push: bool = False
    trust_level: TrustLevel = TrustLevel.TRUSTED

    @abstractmethod
    async def list_items(self, since: datetime | None = None, *, limit: int = 100) -> list[Item]:
        """Items changed since `since` (or a recent window, if None).

        Three callers, one method: the backfill crawler, the scheduled poll, and
        on-demand fetch. The method does not know which one called it -- the
        trigger type never changes what happens downstream.
        """

    @abstractmethod
    async def fetch_item(self, item_id: str) -> Item:
        """Fetch one item by its source-native id, e.g. from a webhook payload."""

    async def register_watch(self) -> dict:
        """Set up push notifications. Only called when supports_push."""
        raise NotImplementedError(f"{self.source_type} does not support push")

    async def handle_change_event(self, payload: dict) -> list[str]:
        """Turn a source-native push payload into changed item ids.

        Payloads are notification-only for both sources we support: they say
        that something changed, never what it now says. Content is always
        fetched separately.
        """
        raise NotImplementedError(f"{self.source_type} does not support push")

    async def load_state(self, state: dict) -> None:
        """Adopt the cursor persisted from the last run.

        Cursors live in the database rather than on the connector because they
        must advance in the same transaction as the items they describe. A
        cursor saved after a failed commit skips those messages permanently --
        and Gmail's history window closes after about seven days, so "skipped"
        there means "gone".
        """
        return None

    def dump_state(self) -> dict:
        """Cursor to persist after a successful fetch. Empty when there is none."""
        return {}

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    def as_tool_schema(self) -> ToolSchema:
        """Expose on-demand fetch as one LLM-callable tool per connector.

        One tool per connector rather than one generic tool with a source
        parameter: "check my mail" then resolves to a single call and "update
        everything" to every tool in the same turn, with no fan-out logic of our
        own to write or keep correct.
        """
        return ToolSchema(
            name=f"fetch_{self.source_type}",
            description=(
                f"Check {self.source_type} for items changed since the last fetch, "
                f"then run anything new through the normal triage pipeline. Use "
                f"this when the user asks to check, refresh, or sync "
                f"{self.source_type}."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )


@dataclass
class TriggerConfig:
    connector: SourceConnector
    poll_interval_hours: float | None = None
    """None means push-driven or on-demand only."""


class ConnectorRegistry:
    """The set of live sources. Adding a third is one `register` call."""

    def __init__(self) -> None:
        self._configs: dict[str, TriggerConfig] = {}

    def register(self, connector: SourceConnector, poll_interval_hours: float | None) -> None:
        self._configs[connector.source_type] = TriggerConfig(connector, poll_interval_hours)

    def get(self, source_type: str) -> SourceConnector:
        config = self._configs.get(source_type)
        if config is None:
            raise ValueError(f"no connector registered for source_type={source_type!r}")
        return config.connector

    def all(self) -> list[TriggerConfig]:
        return list(self._configs.values())

    def polling(self) -> list[TriggerConfig]:
        return [c for c in self._configs.values() if c.poll_interval_hours is not None]

    def tool_schemas(self) -> list[ToolSchema]:
        """Fetch tools for the conversational agent -- one per connector.

        This list is derived, never hand-maintained: registering a source makes
        it callable by the agent with no second edit anywhere.
        """
        return [config.connector.as_tool_schema() for config in self._configs.values()]

    async def close(self) -> None:
        for config in self._configs.values():
            await config.connector.close()
