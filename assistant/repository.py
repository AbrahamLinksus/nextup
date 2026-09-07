"""Every SQL statement in the system.

Nothing else in the codebase writes SQL. That is worth the indirection here
because two of these queries are load-bearing in ways that are invisible at the
call site:

* `upsert_item` is the idempotency guard *and* the change detector, in one
  statement. There is no dedup table -- `last_edited_at` does that job directly,
  and because only strictly newer edits are ever accepted, redelivery and
  out-of-order delivery resolve to the same non-event.
* `sync_chunks` diffs on `chunk_key`, not position, so re-embedding costs are
  proportional to what actually changed rather than to where it changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from pgvector import Vector
from psycopg import AsyncConnection

from assistant.chunking import content_hash
from assistant.models import (
    ActionSpec,
    Chunk,
    ClassificationResult,
    Item,
    Label,
    LifecycleRow,
    QueueEntry,
    QueueReason,
    QueueStatus,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestOutcome:
    """What the ingest guard decided about one incoming item."""

    item_id: UUID | None
    content_changed: bool
    is_new: bool

    @property
    def accepted(self) -> bool:
        """False means a duplicate or a stale event -- skip the whole pipeline."""
        return self.item_id is not None


async def upsert_item(conn: AsyncConnection, item: Item) -> IngestOutcome:
    """Compare-and-swap on `last_edited_at`. Zero rows means do nothing.

    Two distinct cases collapse into that single "zero rows" outcome, and both
    are correct: a redelivered webhook for an edit already processed, and a late
    event that is older than what we hold. Only strictly-newer edits are
    accepted, so a stale event arriving out of order cannot revert content that
    has already been acted on.

    This runs before embedding, classification, and the operations agent, so a
    known duplicate costs one statement rather than several model calls.
    """
    incoming_hash = content_hash(item.content)

    row = await (
        await conn.execute(
            """
            WITH previous AS (
                SELECT content_hash
                FROM items
                WHERE source_type = %(source_type)s AND source_id = %(source_id)s
            ),
            upserted AS (
                INSERT INTO items (
                    source_type, source_id, url, title, content, content_hash,
                    created_at, last_edited_at, deadline, status, urgency_hint,
                    raw_properties, trust_level
                )
                VALUES (
                    %(source_type)s, %(source_id)s, %(url)s, %(title)s, %(content)s,
                    %(content_hash)s, %(created_at)s, %(last_edited_at)s, %(deadline)s,
                    %(status)s, %(urgency_hint)s, %(raw_properties)s, %(trust_level)s
                )
                ON CONFLICT (source_type, source_id) DO UPDATE SET
                    url            = EXCLUDED.url,
                    title          = EXCLUDED.title,
                    content        = EXCLUDED.content,
                    content_hash   = EXCLUDED.content_hash,
                    last_edited_at = EXCLUDED.last_edited_at,
                    deadline       = EXCLUDED.deadline,
                    status         = EXCLUDED.status,
                    urgency_hint   = EXCLUDED.urgency_hint,
                    raw_properties = EXCLUDED.raw_properties,
                    trust_level    = EXCLUDED.trust_level
                WHERE items.last_edited_at < EXCLUDED.last_edited_at
                RETURNING id
            )
            SELECT upserted.id, previous.content_hash AS previous_hash
            FROM upserted LEFT JOIN previous ON TRUE
            """,
            {
                "source_type": item.source_type,
                "source_id": item.source_id,
                "url": item.url,
                "title": item.title,
                "content": item.content,
                "content_hash": incoming_hash,
                "created_at": item.created_at,
                "last_edited_at": item.last_edited_at,
                "deadline": item.deadline,
                "status": item.status,
                "urgency_hint": item.urgency_hint,
                "raw_properties": json.dumps(item.raw_properties, default=str),
                "trust_level": str(item.trust_level),
            },
        )
    ).fetchone()

    if row is None:
        log.info("ingest.skipped_not_newer", item=item.key)
        return IngestOutcome(item_id=None, content_changed=False, is_new=False)

    previous_hash = row["previous_hash"]
    return IngestOutcome(
        item_id=row["id"],
        content_changed=previous_hash != incoming_hash,
        is_new=previous_hash is None,
    )


async def get_item(conn: AsyncConnection, item_key: str) -> dict[str, Any] | None:
    source_type, _, source_id = item_key.partition(":")
    return await (
        await conn.execute(
            "SELECT * FROM items WHERE source_type = %s AND source_id = %s",
            (source_type, source_id),
        )
    ).fetchone()


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

async def sync_chunks(
    conn: AsyncConnection,
    item_id: UUID,
    chunks: list[Chunk],
    embed: Any,
) -> int:
    """Reconcile an item's chunks, embedding only what actually changed.

    The diff joins on `chunk_key`. Because keys follow document structure rather
    than position, inserting a paragraph at the top of a page re-embeds one chunk
    instead of all of them -- and, more importantly, leaves every other chunk's
    key pointing at the same content, which is what the lifecycle table depends
    on to update the right calendar event.

    Returns how many chunks were embedded, which is the number worth logging:
    "20 chunks, 1 embedded" is the design working.
    """
    existing = {
        row["chunk_key"]: row
        for row in await (
            await conn.execute(
                "SELECT chunk_key, content_hash FROM chunks WHERE item_id = %s", (item_id,)
            )
        ).fetchall()
    }

    stale = [
        chunk
        for chunk in chunks
        if chunk.chunk_key not in existing
        or existing[chunk.chunk_key]["content_hash"] != chunk.content_hash
    ]
    vectors = await embed([chunk.content for chunk in stale]) if stale else []

    for chunk, vector in zip(stale, vectors, strict=True):
        await conn.execute(
            """
            INSERT INTO chunks (
                item_id, chunk_key, chunk_index, heading_path, chunk_type,
                content, content_hash, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_id, chunk_key) DO UPDATE SET
                chunk_index  = EXCLUDED.chunk_index,
                heading_path = EXCLUDED.heading_path,
                chunk_type   = EXCLUDED.chunk_type,
                content      = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                embedding    = EXCLUDED.embedding
            """,
            (
                item_id,
                chunk.chunk_key,
                chunk.chunk_index,
                " > ".join(chunk.heading_path),
                str(chunk.chunk_type),
                chunk.content,
                chunk.content_hash,
                Vector(vector),
            ),
        )

    # Unchanged chunks may still have moved. Position is not identity, but it is
    # what orders a citation, so it is kept current.
    for chunk in chunks:
        if chunk not in stale:
            await conn.execute(
                "UPDATE chunks SET chunk_index = %s WHERE item_id = %s AND chunk_key = %s",
                (chunk.chunk_index, item_id, chunk.chunk_key),
            )

    live_keys = [chunk.chunk_key for chunk in chunks]
    if live_keys:
        await conn.execute(
            "DELETE FROM chunks WHERE item_id = %s AND chunk_key <> ALL(%s)",
            (item_id, live_keys),
        )
    else:
        await conn.execute("DELETE FROM chunks WHERE item_id = %s", (item_id,))

    return len(stale)


async def search_chunks(
    conn: AsyncConnection,
    embedding: list[float],
    *,
    top_k: int,
    min_similarity: float,
) -> list[dict[str, Any]]:
    """Cosine top-k above a similarity floor, with the parent item's context.

    The floor is the point. Without one, an unrelated query still returns the
    closest available chunks -- there is always a nearest neighbour -- and the
    agent answers confidently from whatever those happened to be. Below the
    floor the honest result is nothing.

    The parent item's title and url ride along because a chunk alone is often
    unciteable, and because answers are supposed to say where they came from.
    """
    rows = await (
        await conn.execute(
            """
            SELECT
                c.chunk_key, c.content, c.chunk_type, c.chunk_index,
                i.title, i.url, i.source_type, i.source_id, i.last_edited_at,
                1 - (c.embedding <=> %(embedding)s) AS similarity
            FROM chunks c
            JOIN items i ON i.id = c.item_id
            WHERE c.embedding IS NOT NULL
              AND 1 - (c.embedding <=> %(embedding)s) >= %(min_similarity)s
            ORDER BY c.embedding <=> %(embedding)s
            LIMIT %(top_k)s
            """,
            {
                # Wrapped rather than passed as a bare list: psycopg adapts a
                # Python list to float8[], and `vector <=> float8[]` has no
                # operator -- the query fails at runtime, not at write time.
                "embedding": Vector(embedding),
                "top_k": top_k,
                "min_similarity": min_similarity,
            },
        )
    ).fetchall()
    return list(rows)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def get_lifecycle(
    conn: AsyncConnection, item_key: str, chunk_key: str
) -> LifecycleRow | None:
    row = await (
        await conn.execute(
            "SELECT * FROM item_lifecycle WHERE item_key = %s AND chunk_key = %s",
            (item_key, chunk_key),
        )
    ).fetchone()
    return _to_lifecycle(row) if row else None


async def list_lifecycle_for_item(conn: AsyncConnection, item_key: str) -> list[LifecycleRow]:
    rows = await (
        await conn.execute(
            "SELECT * FROM item_lifecycle WHERE item_key = %s", (item_key,)
        )
    ).fetchall()
    return [_to_lifecycle(row) for row in rows]


async def upsert_lifecycle(conn: AsyncConnection, row: LifecycleRow) -> None:
    await conn.execute(
        """
        INSERT INTO item_lifecycle (
            item_key, chunk_key, action_type, last_classification,
            last_extracted_date, action_id
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (item_key, chunk_key) DO UPDATE SET
            action_type         = EXCLUDED.action_type,
            last_classification = EXCLUDED.last_classification,
            last_extracted_date = EXCLUDED.last_extracted_date,
            action_id           = EXCLUDED.action_id
        """,
        (
            row.item_key,
            row.chunk_key,
            row.action_type,
            str(row.last_classification),
            row.last_extracted_date,
            row.action_id,
        ),
    )


async def delete_lifecycle(conn: AsyncConnection, item_key: str, chunk_key: str) -> None:
    await conn.execute(
        "DELETE FROM item_lifecycle WHERE item_key = %s AND chunk_key = %s",
        (item_key, chunk_key),
    )


def _to_lifecycle(row: dict[str, Any]) -> LifecycleRow:
    return LifecycleRow(
        item_key=row["item_key"],
        chunk_key=row["chunk_key"],
        action_type=row["action_type"],
        last_classification=Label(row["last_classification"]),
        last_extracted_date=row["last_extracted_date"],
        action_id=row["action_id"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

async def upsert_queue_entry(
    conn: AsyncConnection,
    *,
    item_key: str,
    chunk_key: str | None,
    classification: ClassificationResult,
    spec: ActionSpec | None,
    reason: QueueReason,
) -> UUID:
    """Create or refresh the single pending entry for this (item, chunk).

    A re-edit while an entry is still unreviewed updates that entry. Stacking a
    second one would show the user the same decision twice and make the
    (confidence, resolution) pair that feeds threshold tuning ambiguous about
    which version they actually judged.
    """
    row = await (
        await conn.execute(
            """
            INSERT INTO queue_entries (
                item_key, chunk_key, classification_label, classification_confidence,
                action_type, action_spec, queue_reason, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
            ON CONFLICT (item_key, COALESCE(chunk_key, '')) WHERE status = 'pending'
            DO UPDATE SET
                classification_label      = EXCLUDED.classification_label,
                classification_confidence = EXCLUDED.classification_confidence,
                action_type               = EXCLUDED.action_type,
                action_spec               = EXCLUDED.action_spec,
                queue_reason              = EXCLUDED.queue_reason,
                created_at                = now()
            RETURNING id
            """,
            (
                item_key,
                chunk_key,
                str(classification.label),
                classification.confidence,
                spec.action_type if spec else None,
                json.dumps(spec.to_payload()) if spec else None,
                str(reason),
            ),
        )
    ).fetchone()
    return row["id"]


async def list_pending(
    conn: AsyncConnection, *, limit: int = 50, reason: str | None = None
) -> list[QueueEntry]:
    sql = "SELECT * FROM queue_entries WHERE status = 'pending'"
    params: list[Any] = []
    if reason:
        sql += " AND queue_reason = %s"
        params.append(reason)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    rows = await (await conn.execute(sql, params)).fetchall()
    return [_to_queue_entry(row) for row in rows]


async def get_queue_entry(conn: AsyncConnection, entry_id: UUID) -> QueueEntry | None:
    row = await (
        await conn.execute("SELECT * FROM queue_entries WHERE id = %s", (entry_id,))
    ).fetchone()
    return _to_queue_entry(row) if row else None


async def resolve_queue_entry(
    conn: AsyncConnection, entry_id: UUID, *, status: QueueStatus, resolution: str
) -> None:
    """Close a queue entry, recording what the user actually decided.

    The resolution text is not decoration: every promote/dismiss yields a
    (confidence, outcome) pair, and those pairs are the entire threshold-tuning
    plan -- calibration from real behaviour instead of a hand-labelled set
    nobody would sit down and build.
    """
    await conn.execute(
        """
        UPDATE queue_entries
        SET status = %s, resolution = %s, resolved_at = now()
        WHERE id = %s
        """,
        (str(status), resolution, entry_id),
    )


async def clear_pending_for(conn: AsyncConnection, item_key: str, chunk_key: str | None) -> None:
    """Withdraw a pending entry that reality has overtaken.

    Used when a re-edit turns a queued item into one the gate now accepts, or
    into one with no action at all: asking the user to review a decision the
    system already made differently is worse than not asking.
    """
    await conn.execute(
        """
        UPDATE queue_entries
        SET status = 'dismissed', resolution = 'superseded by a later edit',
            resolved_at = now()
        WHERE status = 'pending' AND item_key = %s AND COALESCE(chunk_key, '') = %s
        """,
        (item_key, chunk_key or ""),
    )


def _to_queue_entry(row: dict[str, Any]) -> QueueEntry:
    return QueueEntry(
        id=row["id"],
        item_key=row["item_key"],
        chunk_key=row["chunk_key"],
        classification_label=Label(row["classification_label"]),
        classification_confidence=row["classification_confidence"],
        action_type=row["action_type"],
        action_spec=row["action_spec"],
        queue_reason=QueueReason(row["queue_reason"]),
        status=QueueStatus(row["status"]),
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
        resolution=row["resolution"],
    )


# ---------------------------------------------------------------------------
# Structured query tools (fixed functions, never generated SQL)
# ---------------------------------------------------------------------------

async def list_upcoming_actions(
    conn: AsyncConnection, *, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    """What is currently scheduled in a window, from lifecycle state.

    Answered from the lifecycle table rather than by searching the vector index.
    "What's due this week" is a structured question about rows we own; making
    retrieval rediscover it from prose would be both slower and less reliable.
    """
    rows = await (
        await conn.execute(
            """
            SELECT l.*, i.title, i.url, i.source_type
            FROM item_lifecycle l
            LEFT JOIN items i
              ON i.source_type = split_part(l.item_key, ':', 1)
             AND i.source_id   = substr(l.item_key, strpos(l.item_key, ':') + 1)
            WHERE l.last_extracted_date >= %s AND l.last_extracted_date < %s
            ORDER BY l.last_extracted_date
            """,
            (start, end),
        )
    ).fetchall()
    return list(rows)


async def list_recent_items(
    conn: AsyncConnection, *, limit: int = 50, source_type: str | None = None
) -> list[dict[str, Any]]:
    """Recently ingested items with what triage concluded about each.

    The label is read back from the audit log rather than stored on the item:
    classification is an event that happened to an item, not an attribute of it,
    and an item re-edited into a different label has both facts in its history.
    """
    sql = """
        SELECT
            i.source_type, i.source_id, i.title, i.url, i.last_edited_at,
            i.trust_level, i.deadline, i.status, i.urgency_hint,
            (SELECT count(*) FROM chunks c WHERE c.item_id = i.id) AS chunks,
            (
                SELECT a.payload->>'label'
                FROM audit_log a
                WHERE a.item_key = i.source_type || ':' || i.source_id
                  AND a.event_type = 'classification'
                ORDER BY a.created_at DESC, a.id DESC
                LIMIT 1
            ) AS label,
            (
                SELECT count(*) FROM item_lifecycle l
                WHERE l.item_key = i.source_type || ':' || i.source_id
            ) AS actions
        FROM items i
    """
    params: list[Any] = []
    if source_type:
        sql += " WHERE i.source_type = %s"
        params.append(source_type)
    sql += " ORDER BY i.last_edited_at DESC LIMIT %s"
    params.append(limit)
    return list(await (await conn.execute(sql, params)).fetchall())


async def counts(conn: AsyncConnection) -> dict[str, int]:
    """One round trip for the dashboard's header numbers."""
    row = await (
        await conn.execute(
            """
            SELECT
                (SELECT count(*) FROM items)                                   AS items,
                (SELECT count(*) FROM chunks)                                  AS chunks,
                (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL)      AS embedded,
                (SELECT count(*) FROM queue_entries WHERE status = 'pending')  AS pending,
                (SELECT count(*) FROM item_lifecycle WHERE action_id IS NOT NULL) AS actions
            """
        )
    ).fetchone()
    return {key: int(value) for key, value in row.items()}


async def list_dismissed(
    conn: AsyncConnection, *, since: datetime, limit: int = 50
) -> list[QueueEntry]:
    rows = await (
        await conn.execute(
            """
            SELECT * FROM queue_entries
            WHERE status = 'dismissed' AND resolved_at >= %s
            ORDER BY resolved_at DESC LIMIT %s
            """,
            (since, limit),
        )
    ).fetchall()
    return [_to_queue_entry(row) for row in rows]


# ---------------------------------------------------------------------------
# Outcomes and audit
# ---------------------------------------------------------------------------

async def record_outcome(
    conn: AsyncConnection,
    *,
    item_key: str,
    chunk_key: str | None,
    action_type: str,
    operation: str,
    action_id: str | None,
    summary: str,
    payload: dict[str, Any],
    embedding: list[float] | None = None,
) -> None:
    """Append an immutable record of something that actually happened.

    Append-only, in its own table with its own vector space. A record whose
    meaning changes over time -- scheduled, then completed, then scored -- is not
    one embedding to mutate; the later facts arrive as their own items through
    the normal ingestion path.
    """
    await conn.execute(
        """
        INSERT INTO action_outcomes (
            item_key, chunk_key, action_type, operation, action_id,
            summary, payload, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            item_key,
            chunk_key,
            action_type,
            operation,
            action_id,
            summary,
            json.dumps(payload, default=str),
            Vector(embedding) if embedding is not None else None,
        ),
    )


async def record_audit(
    conn: AsyncConnection,
    *,
    event_type: str,
    item_key: str | None,
    chunk_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """The answer to "why did it do that", written at the moment it did it."""
    await conn.execute(
        "INSERT INTO audit_log (event_type, item_key, chunk_key, payload) VALUES (%s,%s,%s,%s)",
        (event_type, item_key, chunk_key, json.dumps(payload or {}, default=str)),
    )


async def list_audit(
    conn: AsyncConnection, *, item_key: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    # Ordered by id as well as timestamp: `now()` is transaction-scoped, so
    # every audit row written while processing one item shares a `created_at`.
    # On timestamp alone the sequence within an item -- classified, then
    # extracted, then queued -- comes back in arbitrary order, which defeats the
    # point of keeping the log.
    sql = "SELECT * FROM audit_log"
    params: list[Any] = []
    if item_key:
        sql += " WHERE item_key = %s"
        params.append(item_key)
    sql += " ORDER BY created_at DESC, id DESC LIMIT %s"
    params.append(limit)
    return list(await (await conn.execute(sql, params)).fetchall())


# ---------------------------------------------------------------------------
# Source state
# ---------------------------------------------------------------------------

async def get_source_state(conn: AsyncConnection, source_type: str) -> dict[str, Any]:
    row = await (
        await conn.execute(
            "SELECT * FROM source_state WHERE source_type = %s", (source_type,)
        )
    ).fetchone()
    return dict(row) if row else {}


async def save_source_state(
    conn: AsyncConnection,
    source_type: str,
    *,
    last_fetch_at: datetime | None = None,
    cursor: str | None = None,
    watch_expires_at: datetime | None = None,
) -> None:
    """Advance a connector's cursor.

    Called inside the same transaction as the items it describes. A cursor
    committed independently of them would, on a failed write, skip exactly the
    messages that failed -- and Gmail's history window closes after about a week,
    so those would not come back.
    """
    await conn.execute(
        """
        INSERT INTO source_state (source_type, last_fetch_at, cursor, watch_expires_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_type) DO UPDATE SET
            last_fetch_at    = COALESCE(EXCLUDED.last_fetch_at, source_state.last_fetch_at),
            cursor           = COALESCE(EXCLUDED.cursor, source_state.cursor),
            watch_expires_at = COALESCE(
                EXCLUDED.watch_expires_at, source_state.watch_expires_at
            ),
            updated_at       = now()
        """,
        (source_type, last_fetch_at, cursor, watch_expires_at),
    )


# ---------------------------------------------------------------------------
# pipeline_metrics
# ---------------------------------------------------------------------------

async def record_pipeline_metrics(
    conn: AsyncConnection,
    *,
    item_key: str,
    source_type: str,
    outcome: str,
    label: str | None,
    duration_ms: int,
    detail: dict[str, Any],
) -> None:
    """One row per processed item, in that item's own transaction.

    `detail` is whatever `metrics.ItemMetrics.as_row()` produced; the columns
    are pulled out of it here so the caller does not have to know the schema and
    the schema does not have to know about ContextVars.
    """
    await conn.execute(
        """
        INSERT INTO pipeline_metrics (
            item_key, source_type, outcome, label, duration_ms,
            llm_calls, llm_ms, input_tokens, output_tokens,
            embed_calls, embed_ms, embedded_texts, stages
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            item_key,
            source_type,
            outcome,
            label,
            duration_ms,
            detail.get("llm_calls", 0),
            detail.get("llm_ms", 0),
            detail.get("input_tokens", 0),
            detail.get("output_tokens", 0),
            detail.get("embed_calls", 0),
            detail.get("embed_ms", 0),
            detail.get("embedded_texts", 0),
            json.dumps(detail.get("stages", {})),
        ),
    )


async def metrics_summary(conn: AsyncConnection, *, since: datetime) -> dict[str, Any]:
    """Aggregate cost over a window.

    Percentiles come from `percentile_cont` over the real rows rather than from
    the in-process histogram's buckets: this is the readout that gets quoted, so
    it should be the exact one. The bucketed version exists for the live
    endpoint, where bounded memory matters more than the second decimal place.
    """
    row = await (
        await conn.execute(
            """
            SELECT
                count(*)                                                  AS items,
                coalesce(sum(llm_calls), 0)                               AS llm_calls,
                coalesce(sum(input_tokens), 0)                            AS input_tokens,
                coalesce(sum(output_tokens), 0)                           AS output_tokens,
                coalesce(sum(embedded_texts), 0)                          AS embedded_texts,
                coalesce(round(avg(duration_ms)), 0)                      AS mean_ms,
                coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms), 0)  AS p50_ms,
                coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms), 0) AS p95_ms,
                coalesce(max(duration_ms), 0)                             AS max_ms,
                coalesce(sum(llm_ms), 0)                                  AS llm_ms,
                coalesce(sum(embed_ms), 0)                                AS embed_ms,
                -- The share of each item's wall clock spent waiting on the
                -- reasoning model. Reasoning time only: embedding runs
                -- *concurrently* with classification, so adding the two
                -- produces percentages above 100 -- a number that is not merely
                -- ugly but wrong, since the overlapped second gets counted
                -- twice. Model calls within an item are sequential, so this one
                -- is a real fraction.
                coalesce(round(avg(
                    CASE WHEN duration_ms > 0
                         THEN 100.0 * llm_ms / duration_ms END
                ), 1), 0)                                                 AS model_wait_pct
            FROM pipeline_metrics
            WHERE created_at >= %s
            """,
            (since,),
        )
    ).fetchone()

    by_outcome = await (
        await conn.execute(
            """
            SELECT outcome, count(*) AS items,
                   coalesce(round(avg(duration_ms)), 0) AS mean_ms
            FROM pipeline_metrics
            WHERE created_at >= %s
            GROUP BY outcome ORDER BY items DESC
            """,
            (since,),
        )
    ).fetchall()

    by_source = await (
        await conn.execute(
            """
            SELECT source_type, count(*) AS items,
                   coalesce(round(avg(duration_ms)), 0) AS mean_ms
            FROM pipeline_metrics
            WHERE created_at >= %s
            GROUP BY source_type ORDER BY items DESC
            """,
            (since,),
        )
    ).fetchall()

    stages = await (
        await conn.execute(
            """
            SELECT stage, round(avg(ms)::numeric, 1) AS mean_ms, count(*) AS samples
            FROM pipeline_metrics,
                 LATERAL jsonb_each_text(stages) AS s(stage, ms_text),
                 LATERAL (SELECT ms_text::numeric AS ms) AS parsed
            WHERE created_at >= %s
            GROUP BY stage ORDER BY mean_ms DESC
            """,
            (since,),
        )
    ).fetchall()

    return {
        "since": since.isoformat(),
        "totals": {key: _number(value) for key, value in row.items()},
        "by_outcome": [{k: _number(v) for k, v in r.items()} for r in by_outcome],
        "by_source": [{k: _number(v) for k, v in r.items()} for r in by_source],
        "stages": [{k: _number(v) for k, v in r.items()} for r in stages],
    }


def _number(value: Any) -> Any:
    """Decimals out of aggregate functions are not JSON-serializable."""
    from decimal import Decimal

    if isinstance(value, Decimal):
        return float(value)
    return value
