"""Command line entry point.

The commands mirror the two execution models rather than the module layout:
`ingest` and `poll` drive the deterministic pipeline, `chat` drives the agentic
one, and `queue` is where the two meet -- the pull-based list of everything the
pipeline declined to decide on its own.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from uuid import UUID

import structlog

from assistant import repository as repo
from assistant.config import get_settings
from assistant.db import connect, migrate


def _configure_logging(level: str) -> None:
    import logging

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level.upper())
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        )
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_migrate(_args: argparse.Namespace) -> int:
    applied = await migrate()
    print("Applied:", ", ".join(applied) if applied else "nothing (already up to date)")
    return 0


async def cmd_status(_args: argparse.Namespace) -> int:
    settings = get_settings()
    conn = await connect()
    try:
        counts = await (
            await conn.execute(
                """
                SELECT
                    (SELECT count(*) FROM items)          AS items,
                    (SELECT count(*) FROM chunks)          AS chunks,
                    (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL) AS embedded,
                    (SELECT count(*) FROM queue_entries WHERE status='pending') AS pending,
                    (SELECT count(*) FROM item_lifecycle)  AS live_actions
                """
            )
        ).fetchone()
        sources = await (
            await conn.execute(
                "SELECT source_type, last_fetch_at, watch_expires_at FROM source_state"
            )
        ).fetchall()
    finally:
        await conn.close()

    print(f"provider     : {settings.llm_provider}")
    print(f"embeddings   : {settings.embedding_model} ({settings.embedding_dim}d)")
    print(f"timezone     : {settings.timezone}")
    print(
        "execution    : "
        + (
            "MCP calendar"
            if settings.calendar_mcp_command
            else "dry run (no CALENDAR_MCP_COMMAND set -- nothing is written to a real calendar)"
        )
    )
    print()
    print(f"items        : {counts['items']}")
    print(f"chunks       : {counts['chunks']} ({counts['embedded']} embedded)")
    print(f"queue pending: {counts['pending']}")
    print(f"live actions : {counts['live_actions']}")
    for row in sources:
        print(
            f"  {row['source_type']:<8} last fetch {row['last_fetch_at']}"
            f"  watch expires {row['watch_expires_at']}"
        )
    return 0


async def cmd_ingest(args: argparse.Namespace) -> int:
    from assistant.runner import build_services, ingest_source

    conn = await connect()
    try:
        async with build_services() as services:
            sources = (
                [args.source]
                if args.source != "all"
                else [c.connector.source_type for c in services.connectors.all()]
            )
            for source in sources:
                summary = await ingest_source(
                    conn,
                    services.connectors,
                    source,
                    classifier=services.classifier,
                    provider=services.provider,
                    embedder=services.embedder,
                    executors=services.executors,
                    limit=args.limit,
                )
                print(json.dumps(summary, indent=2, default=str))
    finally:
        await conn.close()
    return 0


async def cmd_queue_list(args: argparse.Namespace) -> int:
    conn = await connect()
    try:
        entries = await repo.list_pending(conn, limit=args.limit, reason=args.reason)
    finally:
        await conn.close()

    if not entries:
        print("Queue is empty.")
        return 0

    for entry in entries:
        print(f"\n{entry.id}")
        print(f"  item       : {entry.item_key}  chunk={entry.chunk_key}")
        print(
            f"  label      : {entry.classification_label} "
            f"({entry.classification_confidence:.2f})"
        )
        print(f"  queued     : {entry.queue_reason}")
        if entry.action_spec:
            fields = entry.action_spec.get("fields", {})
            drafted = ", ".join(
                f"{name}={body['value']!r}@{body['confidence']:.2f}"
                for name, body in fields.items()
            )
            print(f"  draft      : {entry.action_spec['action_type']}({drafted})")
    return 0


async def cmd_queue_resolve(args: argparse.Namespace) -> int:
    from assistant.runner import build_services
    from assistant.tools import ToolBox

    conn = await connect()
    try:
        async with build_services() as services:
            box = ToolBox(
                conn,
                connectors=services.connectors,
                embedder=services.embedder,
                provider=services.provider,
                classifier=services.classifier,
                executors=services.executors,
            )
            handler = box._promote if args.action == "promote" else box._dismiss  # noqa: SLF001
            print(await handler({"entry_id": str(UUID(args.entry_id)), "note": args.note}))
    finally:
        await conn.close()
    return 0


async def cmd_remember(args: argparse.Namespace) -> int:
    from assistant.chunking import chunk_markdown
    from assistant.connectors import item_from_file, make_item
    from assistant.embeddings import Embedder

    item = (
        item_from_file(args.file)
        if args.file
        else make_item(args.text or sys.stdin.read(), title=args.title or "")
    )

    conn = await connect()
    try:
        async with Embedder() as embedder:
            outcome = await repo.upsert_item(conn, item)
            if not outcome.accepted:
                print("Already stored; nothing changed.")
                return 0
            chunks = chunk_markdown(item.content)
            embedded = await repo.sync_chunks(
                conn,
                outcome.item_id,
                chunks,
                lambda texts: embedder.embed(texts, kind="document"),
            )
            await conn.commit()
    finally:
        await conn.close()

    print(f"Stored {item.title!r} as {item.key} ({embedded} chunks embedded).")
    return 0


async def cmd_search(args: argparse.Namespace) -> int:
    from assistant.embeddings import Embedder
    from assistant.retrieval import search_content

    conn = await connect()
    try:
        async with Embedder() as embedder:
            hits = await search_content(conn, embedder, args.query, top_k=args.top_k)
    finally:
        await conn.close()

    if not hits:
        print("Nothing indexed matched that above the relevance threshold.")
        return 0
    for hit in hits:
        print(f"\n{hit.similarity:.3f}  {hit.cite()}")
        print("  " + hit.content.replace("\n", "\n  ")[:400])
    return 0


async def cmd_chat(args: argparse.Namespace) -> int:
    from assistant.agent import Conversation
    from assistant.runner import build_services
    from assistant.tools import ToolBox

    conn = await connect()
    try:
        async with build_services() as services:
            box = ToolBox(
                conn,
                connectors=services.connectors,
                embedder=services.embedder,
                provider=services.provider,
                classifier=services.classifier,
                executors=services.executors,
            )
            conversation = Conversation(provider=services.provider, tools=box)

            if args.message:
                print(await conversation.send(args.message))
                return 0

            print("Chat with your assistant. Ctrl-D to exit.\n")
            while True:
                try:
                    line = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not line:
                    continue
                print(f"\n{await conversation.send(line)}\n")
    finally:
        await conn.close()


async def cmd_audit(args: argparse.Namespace) -> int:
    conn = await connect()
    try:
        rows = await repo.list_audit(conn, item_key=args.item, limit=args.limit)
    finally:
        await conn.close()

    for row in reversed(rows):  # query is newest-first; read the story forwards
        when = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"{when}  {row['event_type']:<18} {row['item_key']}  {row['chunk_key'] or ''}")
        print(f"    {json.dumps(row['payload'], default=str)[:300]}")
    return 0


async def cmd_upcoming(args: argparse.Namespace) -> int:
    settings = get_settings()
    start = datetime.now(settings.tz)
    end = start + timedelta(days=args.days)

    conn = await connect()
    try:
        rows = await repo.list_upcoming_actions(conn, start=start, end=end)
    finally:
        await conn.close()

    if not rows:
        print(f"Nothing scheduled in the next {args.days} days.")
        return 0
    for row in rows:
        print(
            f"{row['last_extracted_date']}  {row['action_type']}  "
            f"{row.get('title') or row['item_key']}"
        )
    return 0


async def cmd_metrics(args: argparse.Namespace) -> int:
    """What the pipeline has actually cost over a window.

    Reads the stored per-item rows rather than the live counters: a CLI
    invocation is a fresh process, so its in-memory counters are empty by
    construction and would report a system that does nothing.
    """
    settings = get_settings()
    since = datetime.now(settings.tz) - timedelta(days=args.days)

    conn = await connect()
    try:
        summary = await repo.metrics_summary(conn, since=since)
    finally:
        await conn.close()

    totals = summary["totals"]
    if not totals["items"]:
        print(f"No items processed in the last {args.days} days.")
        return 0

    items = totals["items"]
    print(f"Last {args.days} days -- {items} items processed\n")
    print(f"  per item     mean {totals['mean_ms'] / 1000:.1f}s   "
          f"p50 {totals['p50_ms'] / 1000:.1f}s   "
          f"p95 {totals['p95_ms'] / 1000:.1f}s   "
          f"max {totals['max_ms'] / 1000:.1f}s")
    print(f"  model calls  {totals['llm_calls']} total, "
          f"{totals['llm_calls'] / items:.2f} per item")
    print(f"  tokens       {totals['input_tokens']:,} in / {totals['output_tokens']:,} out")
    print(f"  embedded     {totals['embedded_texts']:,} chunks "
          f"in {totals['embed_ms'] / 1000:.1f}s (overlapped with classification)")
    print(f"  model wait   {totals['model_wait_pct']}% of wall clock")

    print("\n  by outcome")
    for row in summary["by_outcome"]:
        print(f"    {row['outcome']:<10} {row['items']:>5}   mean {row['mean_ms'] / 1000:.1f}s")

    print("\n  by source")
    for row in summary["by_source"]:
        print(f"    {row['source_type']:<10} {row['items']:>5}   mean {row['mean_ms'] / 1000:.1f}s")

    print("\n  slowest stages (mean)")
    for row in summary["stages"][:6]:
        print(f"    {row['stage']:<18} {row['mean_ms'] / 1000:>6.2f}s   n={row['samples']}")
    return 0


async def cmd_watch(_args: argparse.Namespace) -> int:
    from assistant.runner import build_services

    conn = await connect()
    try:
        async with build_services() as services:
            for config in services.connectors.all():
                connector = config.connector
                if not connector.supports_push:
                    continue
                payload = await connector.register_watch()
                dumped = connector.dump_state()
                await repo.save_source_state(
                    conn,
                    connector.source_type,
                    cursor=dumped.get("cursor"),
                    watch_expires_at=dumped.get("watch_expires_at"),
                )
                await conn.commit()
                print(f"{connector.source_type}: watch registered -> {payload}")
    finally:
        await conn.close()
    return 0


def cmd_serve(_args: argparse.Namespace) -> int:
    from assistant.api.main import main as serve

    serve()
    return 0


def cmd_poll(_args: argparse.Namespace) -> int:
    from assistant.scheduler import run

    asyncio.run(run())
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assistant", description=__doc__)
    parser.add_argument("--log-level", default=get_settings().log_level)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="create or update the database schema")
    sub.add_parser("status", help="what is stored, scheduled, and pending")

    ingest = sub.add_parser("ingest", help="fetch a source and run the triage pipeline")
    ingest.add_argument("source", nargs="?", default="all", help="notion | gmail | all")
    ingest.add_argument("--limit", type=int, default=50)

    queue = sub.add_parser("queue", help="the pull-based review queue")
    queue_sub = queue.add_subparsers(dest="queue_command", required=True)
    queue_list = queue_sub.add_parser("list")
    queue_list.add_argument("--limit", type=int, default=20)
    queue_list.add_argument("--reason", default=None)
    for action in ("promote", "dismiss"):
        resolve = queue_sub.add_parser(action)
        resolve.add_argument("entry_id")
        resolve.add_argument("--note", default="")

    remember = sub.add_parser("remember", help="store something directly into memory")
    remember.add_argument("text", nargs="?", default=None, help="text (or pipe on stdin)")
    remember.add_argument("--file", default=None, help="read from a file instead")
    remember.add_argument("--title", default=None)

    search = sub.add_parser("search", help="semantic search over stored content")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=10)

    chat = sub.add_parser("chat", help="talk to the assistant")
    chat.add_argument("message", nargs="?", default=None)

    audit = sub.add_parser("audit", help="why the system did what it did")
    audit.add_argument("--item", default=None)
    audit.add_argument("--limit", type=int, default=30)

    upcoming = sub.add_parser("upcoming", help="what is currently scheduled")
    upcoming.add_argument("--days", type=int, default=7)

    metrics_cmd = sub.add_parser("metrics", help="what the pipeline has cost, measured")
    metrics_cmd.add_argument("--days", type=int, default=7)

    sub.add_parser("watch", help="register push subscriptions for push-capable sources")
    sub.add_parser("serve", help="run the webhook API")
    sub.add_parser("poll", help="run the scheduler (polling, backfill, watch renewal)")
    return parser


_ASYNC_COMMANDS = {
    "migrate": cmd_migrate,
    "status": cmd_status,
    "ingest": cmd_ingest,
    "remember": cmd_remember,
    "search": cmd_search,
    "chat": cmd_chat,
    "audit": cmd_audit,
    "upcoming": cmd_upcoming,
    "metrics": cmd_metrics,
    "watch": cmd_watch,
}

_SYNC_COMMANDS = {"serve": cmd_serve, "poll": cmd_poll}


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging(args.log_level)

    if args.command in _SYNC_COMMANDS:
        return _SYNC_COMMANDS[args.command](args)

    if args.command == "queue":
        if args.queue_command == "list":
            return asyncio.run(cmd_queue_list(args))
        args.action = args.queue_command
        return asyncio.run(cmd_queue_resolve(args))

    return asyncio.run(_ASYNC_COMMANDS[args.command](args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
