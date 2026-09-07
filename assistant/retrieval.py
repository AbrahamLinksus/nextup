"""Semantic search over stored chunks.

Two rules here are the query-side counterpart to the classifier's boundary bias,
and both exist because the failure mode is a *confident wrong answer* rather than
an error:

**A similarity floor.** Vector search always returns something -- there is
always a nearest neighbour. Without a floor, "what did the professor say about
the Rust assignment" against a corpus containing no Rust returns the closest
available chunks and the agent answers from them. Below the floor, the result is
empty, and empty is a real answer.

**Attribution.** Every hit carries its parent item's title and url, so an answer
can say where it came from. This is the same instinct as the audit log: a claim
whose provenance cannot be checked is worth less than one that can.

Deferred, and honestly so: a single chunk can lack the context around it (a
paragraph separated from its heading). The parent title travels with every hit,
which covers the common case. Full adjacent-chunk stitching is a real refinement
that is not worth building before real usage shows it is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from psycopg import AsyncConnection

from assistant import repository as repo
from assistant.config import get_settings
from assistant.embeddings import Embedder

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SearchHit:
    chunk_key: str
    content: str
    similarity: float
    title: str
    url: str
    source_type: str
    source_id: str

    def cite(self) -> str:
        where = f"{self.title} ({self.url})" if self.url else self.title
        return f"[{where}]"

    def render(self) -> str:
        return f"{self.cite()} (similarity {self.similarity:.2f})\n{self.content}"


async def search_content(
    conn: AsyncConnection,
    embedder: Embedder,
    query: str,
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[SearchHit]:
    """Embed the query and return chunks above the similarity floor.

    The query is embedded with the same model and the query-side task prefix.
    Both halves matter: vectors from different models do not compare at all, and
    a document-prefixed query compares measurably worse than a query-prefixed
    one against the same corpus.
    """
    settings = get_settings()
    vector = await embedder.embed_one(query, kind="query")

    rows = await repo.search_chunks(
        conn,
        vector,
        top_k=top_k or settings.retrieval_top_k,
        min_similarity=(
            settings.retrieval_min_similarity if min_similarity is None else min_similarity
        ),
    )
    hits = [
        SearchHit(
            chunk_key=row["chunk_key"],
            content=row["content"],
            similarity=float(row["similarity"]),
            title=row["title"],
            url=row["url"],
            source_type=row["source_type"],
            source_id=row["source_id"],
        )
        for row in rows
    ]
    log.info("retrieval.searched", query=query[:80], hits=len(hits))
    return hits


def render_hits(hits: list[SearchHit]) -> str:
    """Format results for a model, saying plainly when there are none.

    The empty case returns a sentence, not an empty string. A tool that returns
    nothing invites the model to fill the silence from general knowledge, which
    is exactly what the grounding rule forbids.
    """
    if not hits:
        return (
            "No indexed content matched this query above the relevance threshold. "
            "Nothing in the user's sources covers it. Say so; do not answer from "
            "general knowledge."
        )
    return "\n\n---\n\n".join(hit.render() for hit in hits)
