"""Shared fixtures.

Database tests run against a real Postgres with pgvector rather than a mock,
because the three queries worth testing -- the ingest CAS, the partial unique
index on the queue, and the vector similarity floor -- are behaviours of
PostgreSQL, not of Python. A mock would assert that my mock works.

They skip rather than fail when no database is reachable, so `pytest` is useful
on a laptop with nothing running.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from assistant.llm import LLMProvider, ToolCall
from assistant.models import Item, TrustLevel

# Both URLs come from the environment so the same suite runs against the local
# compose stack (port 5434, chosen to dodge two other Postgres instances on this
# machine) and against CI's service container, which has no such conflict and
# uses the default port.
ADMIN_DB_URL = os.environ.get(
    "TEST_ADMIN_DATABASE_URL", "postgresql://assistant:assistant@localhost:5434/assistant"
)
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://assistant:assistant@localhost:5434/assistant_test"
)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def db():
    """A migrated, empty database, torn down between tests.

    Truncating rather than recreating: schema creation is the slow part, and the
    tests care about row state, not about DDL.
    """
    import psycopg

    from assistant.db import connect, migrate

    try:
        admin = await psycopg.AsyncConnection.connect(ADMIN_DB_URL, autocommit=True)
    except psycopg.OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no Postgres reachable for tests ({exc})")

    await admin.execute("DROP DATABASE IF EXISTS assistant_test")
    await admin.execute("CREATE DATABASE assistant_test")
    await admin.close()

    await migrate(TEST_DB_URL)
    conn = await connect(TEST_DB_URL)
    try:
        yield conn
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeProvider(LLMProvider):
    """A scripted model. Every pipeline test needs determinism, not a model.

    Records what it was asked so tests can assert on prompt contents -- which is
    how the "untrusted content is delimited" guarantee gets checked at all.
    """

    name = "fake"

    def __init__(
        self,
        structured_responses: list[dict] | None = None,
        *,
        tool_calls: list[ToolCall | None] | None = None,
    ) -> None:
        self.structured_responses = list(structured_responses or [])
        self.tool_calls = list(tool_calls or [])
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def structured(self, *, system, prompt, schema, max_tokens=1024, temperature=0.0):
        self.systems.append(system)
        self.prompts.append(prompt)
        if not self.structured_responses:
            return {"label": "informational", "confidence": 0.9, "rationale": "default"}
        return self.structured_responses.pop(0)

    async def choose_tool(self, *, system, prompt, tools, max_tokens=2048):
        self.systems.append(system)
        self.prompts.append(prompt)
        return self.tool_calls.pop(0) if self.tool_calls else None

    async def converse(self, *, system, messages, tools, max_tokens=4096):
        from assistant.llm import AssistantTurn

        return AssistantTurn(text="ok")


class FakeEmbedder:
    """Deterministic pseudo-embeddings. Dimension is what matters downstream."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.calls: list[tuple[str, int]] = []

    async def embed(self, texts, *, kind="document"):
        self.calls.append((kind, len(texts)))
        return [self._vector(text) for text in texts]

    async def embed_one(self, text, *, kind="query"):
        return (await self.embed([text], kind=kind))[0]

    async def close(self) -> None:
        return None

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(char) for char in text) or 1
        return [((seed * (i + 1)) % 97) / 97.0 for i in range(self.dim)]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_item(
    content: str = "# Title\n\nSome content that is definitely long enough to classify.",
    *,
    source_type: str = "notion",
    source_id: str = "page-1",
    edited: datetime | None = None,
    trust_level: TrustLevel = TrustLevel.TRUSTED,
    **kwargs,
) -> Item:
    when = edited or datetime(2026, 8, 22, 9, 0, tzinfo=UTC)
    return Item(
        source_type=source_type,
        source_id=source_id,
        url=f"https://example.test/{source_id}",
        title=kwargs.pop("title", "Test item"),
        content=content,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        last_edited_at=when,
        trust_level=trust_level,
        **kwargs,
    )
