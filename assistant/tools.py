"""The tool surface the conversational agent is allowed to call.

No separate router component. Deciding "is this a structured lookup or a
semantic search" is the same tool-selection problem already solved on the fetch
side, so it is solved the same way: everything is a registered tool and the
model picks. A question that needs both -- "what's due this week and what is it
about" -- is two calls in one turn, which is behaviour the model already has.

Structured lookups are **fixed, named functions**, never generated SQL. The
realistic query surface is small, and handing a model arbitrary SQL against a
database holding the user's mail is a real risk for a marginal gain.

Actions taken here skip the confidence gate, and that is not an inconsistency.
Gating exists to substitute for a human when the input is an inferred
classification. When the user types "schedule this for tomorrow", the human is
right there and the instruction *is* the confidence signal.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from psycopg import AsyncConnection

from assistant import repository as repo
from assistant.actions import ExecutorPool, get_definition
from assistant.classify import Classifier
from assistant.config import get_settings
from assistant.connectors import ConnectorRegistry
from assistant.embeddings import Embedder
from assistant.llm import LLMProvider, ToolSchema
from assistant.models import ActionSpec, FieldValue, LifecycleRow, QueueStatus
from assistant.retrieval import render_hits, search_content

log = structlog.get_logger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    schema: ToolSchema
    handler: Handler


class ToolBox:
    """Everything the agent can do, assembled from the live registries.

    Fetch tools are derived from the connector registry rather than listed, so
    registering a third source makes it callable with no edit here.
    """

    def __init__(
        self,
        conn: AsyncConnection,
        *,
        connectors: ConnectorRegistry,
        embedder: Embedder,
        provider: LLMProvider,
        classifier: Classifier,
        executors: ExecutorPool | None = None,
    ) -> None:
        self.conn = conn
        self.connectors = connectors
        self.embedder = embedder
        self.provider = provider
        self.classifier = classifier
        self.executors = executors or ExecutorPool()
        self._tools: dict[str, Tool] = {}
        self._build()

    # -- assembly --------------------------------------------------------

    def _build(self) -> None:
        for config in self.connectors.all():
            source_type = config.connector.source_type
            self._register(
                config.connector.as_tool_schema(),
                self._make_fetch_handler(source_type),
            )

        self._register(
            ToolSchema(
                name="list_pending_queue",
                description=(
                    "List items awaiting the user's review: things that were not "
                    "confident enough to act on automatically, were optional, or "
                    "had no matching action. Use this for 'what's in my queue', "
                    "'anything waiting for me', 'what did you not do'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Max entries, default 20."},
                        "reason": {
                            "type": "string",
                            "enum": [
                                "optional", "low_confidence", "no_tool_available",
                                "past_date", "date_conflict", "injection_suspected",
                            ],
                            "description": "Optional filter by why it was queued.",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            self._list_pending,
        )

        self._register(
            ToolSchema(
                name="list_upcoming_actions",
                description=(
                    "List actions the system has already scheduled in a date "
                    "range -- what is actually on the calendar. Use this for "
                    "'what's due this week', 'what's coming up', 'am I free "
                    "Thursday'. This reads scheduled state directly; it is not "
                    "a search over content."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "ISO date, inclusive."},
                        "end_date": {"type": "string", "description": "ISO date, exclusive."},
                    },
                    "required": ["start_date", "end_date"],
                    "additionalProperties": False,
                },
            ),
            self._list_upcoming,
        )

        self._register(
            ToolSchema(
                name="list_dismissed",
                description=(
                    "List queue entries the user dismissed recently. Useful for "
                    "'what did I skip' or 'did I say no to something about X'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "since_days": {"type": "integer", "description": "Look back N days."}
                    },
                    "additionalProperties": False,
                },
            ),
            self._list_dismissed,
        )

        self._register(
            ToolSchema(
                name="search_content",
                description=(
                    "Semantic search across everything ingested from the user's "
                    "sources. Use this for questions about what something says "
                    "or covers -- 'what's on the DBMS syllabus', 'what did the "
                    "email about fees say'. Returns nothing when the corpus has "
                    "no relevant content, which is a real answer."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language query."},
                        "top_k": {"type": "integer", "description": "Max results, default 10."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._search,
        )

        self._register(
            ToolSchema(
                name="promote_queue_entry",
                description=(
                    "Execute the action drafted on a queued entry, after the user "
                    "has approved it. Use only when the user clearly says to go "
                    "ahead with a specific queued item."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string", "description": "Queue entry UUID."},
                        "note": {"type": "string", "description": "What the user said."},
                    },
                    "required": ["entry_id"],
                    "additionalProperties": False,
                },
            ),
            self._promote,
        )

        self._register(
            ToolSchema(
                name="dismiss_queue_entry",
                description="Dismiss a queued item the user does not want acted on.",
                parameters={
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string", "description": "Queue entry UUID."},
                        "note": {"type": "string", "description": "Why, in the user's words."},
                    },
                    "required": ["entry_id"],
                    "additionalProperties": False,
                },
            ),
            self._dismiss,
        )

        self._register(
            ToolSchema(
                name="schedule_event",
                description=(
                    "Create a calendar event because the user directly asked for "
                    "one in this conversation. Do not use this to act on your own "
                    "judgment about an item -- only on an explicit instruction."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "start": {"type": "string", "description": "ISO-8601 date or datetime."},
                        "end": {"type": "string", "description": "ISO-8601; optional."},
                        "all_day": {"type": "boolean"},
                        "location": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["title", "start"],
                    "additionalProperties": False,
                },
            ),
            self._schedule,
        )

        self._register(
            ToolSchema(
                name="remember",
                description=(
                    "Store something the user tells you directly into their "
                    "searchable memory. Use when they say 'remember that...', "
                    "'note this down', or paste content to keep. This stores and "
                    "indexes it; it does not create any action."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Text to store."},
                        "title": {"type": "string", "description": "Short label."},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            ),
            self._remember,
        )

    def _register(self, schema: ToolSchema, handler: Handler) -> None:
        self._tools[schema.name] = Tool(schema=schema, handler=handler)

    # -- access ----------------------------------------------------------

    def schemas(self) -> list[ToolSchema]:
        return [tool.schema for tool in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"No such tool: {name!r}. Available: {sorted(self._tools)}"
        try:
            return await tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - a tool failure is a result, not a crash
            log.warning("tool.failed", tool=name, error=str(exc))
            return f"Tool {name!r} failed: {exc}"

    # -- handlers --------------------------------------------------------

    def _make_fetch_handler(self, source_type: str) -> Handler:
        async def handler(_arguments: dict[str, Any]) -> str:
            """On-demand fetch runs the *full* pipeline, same as a poll.

            The trigger type never changes downstream processing. Only an
            explicit instruction typed by the user bypasses the gate; content
            that was fetched never does, however it came to be fetched.
            """
            from assistant.runner import ingest_source

            summary = await ingest_source(
                self.conn,
                self.connectors,
                source_type,
                classifier=self.classifier,
                provider=self.provider,
                embedder=self.embedder,
                executors=self.executors,
            )
            return json.dumps(summary, default=str)

        return handler

    async def _list_pending(self, arguments: dict[str, Any]) -> str:
        entries = await repo.list_pending(
            self.conn,
            limit=int(arguments.get("limit", 20)),
            reason=arguments.get("reason"),
        )
        if not entries:
            return "The review queue is empty."
        return json.dumps(
            [
                {
                    "entry_id": str(entry.id),
                    "item": entry.item_key,
                    "chunk": entry.chunk_key,
                    "label": str(entry.classification_label),
                    "confidence": round(entry.classification_confidence, 2),
                    "queued_because": str(entry.queue_reason),
                    "drafted_action": entry.action_spec,
                    "created_at": entry.created_at,
                }
                for entry in entries
            ],
            default=str,
        )

    async def _list_upcoming(self, arguments: dict[str, Any]) -> str:
        start = _as_datetime(arguments.get("start_date"))
        end = _as_datetime(arguments.get("end_date"))
        if start is None or end is None:
            return "Both start_date and end_date are required, as ISO dates."

        rows = await repo.list_upcoming_actions(self.conn, start=start, end=end)
        if not rows:
            return f"Nothing scheduled between {start.date()} and {end.date()}."
        return json.dumps(
            [
                {
                    "when": row["last_extracted_date"],
                    "action_type": row["action_type"],
                    "item": row["item_key"],
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "action_id": row["action_id"],
                }
                for row in rows
            ],
            default=str,
        )

    async def _list_dismissed(self, arguments: dict[str, Any]) -> str:
        since = datetime.now(UTC) - timedelta(days=int(arguments.get("since_days", 30)))
        entries = await repo.list_dismissed(self.conn, since=since)
        if not entries:
            return "Nothing was dismissed in that period."
        return json.dumps(
            [
                {
                    "item": entry.item_key,
                    "label": str(entry.classification_label),
                    "resolution": entry.resolution,
                    "resolved_at": entry.resolved_at,
                }
                for entry in entries
            ],
            default=str,
        )

    async def _search(self, arguments: dict[str, Any]) -> str:
        hits = await search_content(
            self.conn,
            self.embedder,
            arguments["query"],
            top_k=arguments.get("top_k"),
        )
        return render_hits(hits)

    async def _promote(self, arguments: dict[str, Any]) -> str:
        """Promotion is the bridge between the queue and the lifecycle table.

        It executes the drafted spec, then writes the lifecycle row -- which is
        what puts a user-approved action under the same reconciliation rules as
        an automatic one. Without that row, a later edit to the source item
        could not update or remove what the user approved.
        """
        entry = await repo.get_queue_entry(self.conn, UUID(arguments["entry_id"]))
        if entry is None:
            return f"No queue entry {arguments['entry_id']!r}."
        if entry.status is not QueueStatus.PENDING:
            return f"That entry is already {entry.status}."
        if not entry.action_spec:
            return (
                "That entry has no drafted action to execute -- it was queued "
                "because nothing in the registry fitted it."
            )

        spec = ActionSpec.from_payload(entry.action_spec)
        definition = get_definition(spec.action_type)
        if definition is None:
            return f"Action type {spec.action_type!r} is no longer registered."

        executor = await self.executors.get(spec.action_type)
        result = await executor.create(spec)

        from assistant.dates import parse_iso_local

        await repo.upsert_lifecycle(
            self.conn,
            LifecycleRow(
                item_key=entry.item_key,
                chunk_key=entry.chunk_key or "(item)",
                action_type=spec.action_type,
                last_classification=entry.classification_label,
                last_extracted_date=(
                    parse_iso_local(spec.value(definition.temporal_fields[0]))
                    if definition.temporal_fields
                    else None
                ),
                action_id=result.action_id,
                updated_at=datetime.now(get_settings().tz),
            ),
        )
        await repo.resolve_queue_entry(
            self.conn,
            entry.id,
            status=QueueStatus.PROMOTED,
            resolution=arguments.get("note") or "approved by the user in conversation",
        )
        await repo.record_outcome(
            self.conn,
            item_key=entry.item_key,
            chunk_key=entry.chunk_key,
            action_type=spec.action_type,
            operation="create",
            action_id=result.action_id,
            summary=result.summary,
            payload=result.detail,
        )
        await repo.record_audit(
            self.conn,
            event_type="action_taken",
            item_key=entry.item_key,
            chunk_key=entry.chunk_key,
            payload={"via": "queue_promotion", "summary": result.summary},
        )
        await self.conn.commit()
        return f"Done: {result.summary}" + (" (dry run)" if result.dry_run else "")

    async def _dismiss(self, arguments: dict[str, Any]) -> str:
        entry_id = UUID(arguments["entry_id"])
        await repo.resolve_queue_entry(
            self.conn,
            entry_id,
            status=QueueStatus.DISMISSED,
            resolution=arguments.get("note") or "dismissed by the user in conversation",
        )
        await self.conn.commit()
        return "Dismissed. Recorded for threshold calibration."

    async def _schedule(self, arguments: dict[str, Any]) -> str:
        # Confidence 1.0 across the board: the user said so. This reuses the
        # same executor as the background pipeline -- only the gate is absent.
        spec = ActionSpec(
            action_type="create_calendar_event",
            fields={
                name: FieldValue(value=value, confidence=1.0)
                for name, value in arguments.items()
                if value not in (None, "")
            },
            source_item_key="conversation",
        )
        executor = await self.executors.get("create_calendar_event")
        result = await executor.create(spec)
        await repo.record_outcome(
            self.conn,
            item_key="conversation",
            chunk_key=None,
            action_type=spec.action_type,
            operation="create",
            action_id=result.action_id,
            summary=result.summary,
            payload=result.detail,
        )
        await repo.record_audit(
            self.conn,
            event_type="action_taken",
            item_key="conversation",
            payload={"via": "direct_instruction", "summary": result.summary},
        )
        await self.conn.commit()
        return f"Created {result.summary}" + (" (dry run)" if result.dry_run else "")

    async def _remember(self, arguments: dict[str, Any]) -> str:
        from assistant.chunking import chunk_markdown
        from assistant.connectors import make_item

        item = make_item(arguments["content"], title=arguments.get("title", ""))
        outcome = await repo.upsert_item(self.conn, item)
        if not outcome.accepted:
            return "Already stored; nothing changed."

        chunks = chunk_markdown(item.content)
        embedded = await repo.sync_chunks(
            self.conn,
            outcome.item_id,
            chunks,
            lambda texts: self.embedder.embed(texts, kind="document"),
        )

        if not outcome.content_changed and not outcome.is_new:
            # A re-paste carries a fresh timestamp, so the ingest CAS accepts it
            # even though the text is identical -- the id is the content hash, so
            # it updated the one row rather than adding a near-duplicate. Nothing
            # was re-embedded, and reading back "stored and indexed (0 chunks)"
            # would suggest otherwise.
            await self.conn.commit()
            return f"Already stored as {item.title!r}; nothing changed."

        await repo.record_audit(
            self.conn,
            event_type="skipped",
            item_key=item.key,
            payload={"reason": "direct input is declared memory, not triaged for action"},
        )
        await self.conn.commit()
        return f"Stored and indexed as {item.title!r} ({embedded} chunks embedded)."


def _as_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    from assistant.dates import parse_iso_local

    return parse_iso_local(str(value))
