"""Async Postgres connectivity.

Transaction discipline: `async with pool.connection() as conn:` commits when the
block exits cleanly and rolls back on exception. The pipeline wraps one item's
entire processing in a single such block, so "the item was recorded as ingested"
and "its chunks, queue entry, and audit rows exist" are one atomic fact. A
partial commit there would leave an item marked processed with nothing to show
for it, and the CAS guard would refuse to reprocess it.
"""

from __future__ import annotations

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from assistant.config import get_settings


async def _configure(conn: AsyncConnection) -> None:
    # Registers the pgvector adapters on every pooled connection, and makes
    # every query return dicts rather than positional tuples.
    from pgvector.psycopg import register_vector_async

    await register_vector_async(conn)
    conn.row_factory = dict_row
    await _set_session_timezone(conn)


async def _set_session_timezone(conn: AsyncConnection) -> None:
    """Read every timestamptz back in the timezone this system runs in.

    A `timestamptz` is an instant, so this changes no stored value and no
    comparison -- but it does change what a value *renders as*, and those
    renderings are read by people and quoted by the model. Left at the server
    default, "the fee deadline" comes back as 14 September 18:30Z and gets
    reported as the 14th, which is the wrong day. The timezone is fixed for v1
    (design log, Timezone), so pinning the session to it is the same decision
    applied one layer lower.
    """
    await conn.execute(f"SET TIME ZONE '{get_settings().timezone}'")
    await conn.commit()


def make_pool(
    conninfo: str | None = None,
    *,
    min_size: int = 1,
    max_size: int = 5,
) -> AsyncConnectionPool:
    """Build a (not yet opened) pool. Call `await pool.open()` to start it."""
    return AsyncConnectionPool(
        conninfo or get_settings().database_url,
        configure=_configure,
        min_size=min_size,
        max_size=max_size,
        open=False,
    )


async def connect(
    conninfo: str | None = None, *, register_vector: bool = True
) -> AsyncConnection:
    """One-off connection, for scripts and tests.

    `register_vector=False` exists for exactly one caller: the migration runner.
    Registering pgvector's adapters requires the `vector` type to already exist,
    and on a fresh database the statement that creates it is the first line of
    the first migration -- so bootstrapping has to happen on a connection that
    has not tried to register it yet.
    """
    conn = await AsyncConnection.connect(conninfo or get_settings().database_url)
    if register_vector:
        await _configure(conn)
    else:
        conn.row_factory = dict_row
        await _set_session_timezone(conn)
    return conn


async def migrate(conninfo: str | None = None, directory: str = "migrations") -> list[str]:
    """Bring a database up to date and report what it applied."""
    conn = await connect(conninfo, register_vector=False)
    try:
        return await apply_migrations(conn, directory)
    finally:
        await conn.close()


async def apply_migrations(conn: AsyncConnection, directory: str = "migrations") -> list[str]:
    """Run every .sql file in order, skipping ones already applied.

    Deliberately not Alembic: the schema is a handful of tables written as SQL
    with the reasoning in the comments, and a migration runner that can be read
    in one screen is worth more here than autogenerate.
    """
    import pathlib

    await conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  filename TEXT PRIMARY KEY,"
        "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    await conn.commit()

    applied: list[str] = []
    rows = await (await conn.execute("SELECT filename FROM schema_migrations")).fetchall()
    already = {row["filename"] for row in rows}

    for path in sorted(pathlib.Path(directory).glob("*.sql")):
        if path.name in already:
            continue
        await conn.execute(path.read_text())
        await conn.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
        )
        await conn.commit()
        applied.append(path.name)

    return applied
