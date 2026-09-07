"""
Jake's assistant — connector interface prototype.

Validates the fetch + trigger layer design before this moves into the real
repo (Claude Code). Nothing below the SourceConnector interface should ever
branch on source_type — that's the whole point of the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Canonical item model
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """A single unit of content from any source, normalized to one shape.

    `properties` is deliberately loose — Notion gives rich structured fields
    (due dates, status), Gmail gives almost none. Forcing a rigid schema here
    would either lose Notion's signal or invent fake fields for Gmail.
    Downstream consumers (the triage layer) must treat every key as optional.
    """
    source_type: str          # "notion", "gmail", ...
    source_id: str            # source-native id, e.g. Notion page_id, Gmail message id
    url: str
    title: str
    content: str               # flattened plain text / markdown — never raw blocks/MIME
    created_at: datetime
    last_edited_at: datetime
    properties: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Connector interface
# ---------------------------------------------------------------------------

class SourceConnector(ABC):
    """Every source implements this. The trigger layer, triage layer, and
    storage layer only ever talk to this interface — never to a source's
    native API directly.
    """

    source_type: str
    supports_push: bool = False

    @abstractmethod
    def list_items(self, since: datetime | None = None) -> list[Item]:
        """Items changed since `since` (or everything, if None).

        Used for three different callers: the backfill crawler, the
        scheduled poll, and on-demand fetch. One method, three triggers —
        the method itself doesn't know or care which one called it.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_item(self, item_id: str) -> Item:
        """Fetch a single item by its source-native id (e.g. from a webhook payload)."""
        raise NotImplementedError

    def register_watch(self) -> None:
        """Set up push-based change notifications. Only called if supports_push."""
        raise NotImplementedError(f"{self.source_type} does not support push")

    def handle_change_event(self, raw_payload: dict) -> list[str]:
        """Parse a source-native webhook/push payload into changed item ids."""
        raise NotImplementedError(f"{self.source_type} does not support push")

    def as_tool_schema(self) -> dict:
        """Expose this connector's on-demand fetch as an LLM-callable tool.

        This is the piece that lets 'check my mail' resolve to fetch_gmail()
        and 'update all' resolve to calling every registered tool — the LLM
        picks from this list, no special-casing needed on our end.
        """
        return {
            "name": f"fetch_{self.source_type}",
            "description": (
                f"Check {self.source_type} for items changed since the last "
                f"fetch, or fetch everything if this source has never been fetched."
            ),
            "parameters": {"type": "object", "properties": {}},
        }


# ---------------------------------------------------------------------------
# Stub implementations — real API calls TODO, shape is what matters here
# ---------------------------------------------------------------------------

class NotionConnector(SourceConnector):
    source_type = "notion"
    supports_push = False  # decision: poll twice daily, no webhook wired up (yet)

    def list_items(self, since: datetime | None = None) -> list[Item]:
        # TODO: data_source query (2026-03-01+ API), filter by last_edited_time > since
        # TODO: flatten blocks -> markdown, flatten page properties -> dict
        raise NotImplementedError

    def fetch_item(self, item_id: str) -> Item:
        # TODO: pages.retrieve + blocks.children.list (recursive) -> flatten
        raise NotImplementedError


class GmailConnector(SourceConnector):
    source_type = "gmail"
    supports_push = True  # decision: real-time push via watch() + Pub/Sub

    def list_items(self, since: datetime | None = None) -> list[Item]:
        # TODO: users.history.list from last stored historyId (or a date-bounded
        # search if since is None / no historyId stored yet — history expires ~7 days)
        raise NotImplementedError

    def fetch_item(self, item_id: str) -> Item:
        # TODO: users.messages.get -> flatten headers + body -> Item
        raise NotImplementedError

    def register_watch(self) -> None:
        # TODO: users.watch() -> store returned historyId + expiration, renew before expiry
        pass

    def handle_change_event(self, raw_payload: dict) -> list[str]:
        # TODO: decode Pub/Sub push payload -> extract historyId -> history.list diff
        # -> return list of changed message ids
        return []


# ---------------------------------------------------------------------------
# Trigger layer — decides *when* list_items() gets called, per connector
# ---------------------------------------------------------------------------

@dataclass
class TriggerConfig:
    connector: SourceConnector
    poll_interval_hours: float | None = None  # None = push-only or on-demand-only


TRIGGER_REGISTRY: list[TriggerConfig] = [
    TriggerConfig(connector=NotionConnector(), poll_interval_hours=12),
    TriggerConfig(connector=GmailConnector(), poll_interval_hours=None),  # push handles it
]


def registered_tools() -> list[dict]:
    """Tool schemas to hand to the LLM for on-demand fetch — one per connector.
    Adding a third source later means adding one TriggerConfig; this list
    (and therefore what the LLM can call) updates automatically.
    """
    return [cfg.connector.as_tool_schema() for cfg in TRIGGER_REGISTRY]


def run_scheduled_polls(last_fetch_times: dict[str, datetime]) -> dict[str, list[Item]]:
    """Called by cron. Only touches connectors that declare a poll interval —
    Gmail is skipped here because it's push-driven.
    """
    results: dict[str, list[Item]] = {}
    for cfg in TRIGGER_REGISTRY:
        if cfg.poll_interval_hours is None:
            continue
        since = last_fetch_times.get(cfg.connector.source_type)
        results[cfg.connector.source_type] = cfg.connector.list_items(since=since)
    return results


def on_demand_fetch(source_type: str, since: datetime | None = None) -> list[Item]:
    """Called when the LLM invokes a fetch_<source> tool. 'update all' means
    the LLM calls this once per connector in the same turn — no fan-out
    logic needed here, that decision lives entirely with the LLM.
    """
    for cfg in TRIGGER_REGISTRY:
        if cfg.connector.source_type == source_type:
            return cfg.connector.list_items(since=since)
    raise ValueError(f"No connector registered for source_type={source_type!r}")


if __name__ == "__main__":
    # Sanity check: the abstraction should let us enumerate tools and route
    # triggers without knowing anything about Notion or Gmail internals.
    print("Registered tools for LLM:")
    for tool in registered_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    print("\nPoll schedule:")
    for cfg in TRIGGER_REGISTRY:
        mode = (
            f"every {cfg.poll_interval_hours}h"
            if cfg.poll_interval_hours
            else "push/on-demand only"
        )
        print(f"  - {cfg.connector.source_type}: {mode}")
