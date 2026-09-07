"""Embedding client. Always Ollama, always local, always nomic-embed-text.

Unlike the reasoning model, this is deliberately *not* swappable at runtime.
Changing embedding models is a full corpus re-embed, not a config change --
different models produce different vector spaces, not merely different widths,
so existing vectors become meaningless rather than merely stale. A setting that
looks flippable would invite exactly the change that silently breaks retrieval.

The dimension check in `embed` is not defensive padding: `chunks.embedding` is
declared `VECTOR(768)`, and pointing EMBEDDING_MODEL at a different-width model
would otherwise surface as an opaque insert error deep inside the pipeline.
"""

from __future__ import annotations

from typing import Literal, Self

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from assistant import metrics
from assistant.config import get_settings


class EmbeddingDimensionError(RuntimeError):
    pass


# nomic-embed-text is trained with task prefixes, and it is not optional
# decoration: the model uses them to place queries and documents in compatible
# regions of the space. Without them an unrelated query still scores ~0.44
# against arbitrary content, which leaves almost no gap between "relevant" and
# "nearest thing available" for the similarity floor to sit in.
_PREFIXES = {
    "document": "search_document: ",
    "query": "search_query: ",
}

EmbeddingKind = Literal["document", "query"]


class Embedder:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    async def embed(
        self, texts: list[str], *, kind: EmbeddingKind = "document"
    ) -> list[list[float]]:
        """Embed a batch in one call.

        Batching matters on a single consumer GPU: a 20-chunk Notion page costs
        one round trip rather than 20, and the model stays resident between them.

        `kind` selects the task prefix. Stored chunks are always "document" and
        search input is always "query" -- mixing them silently degrades every
        comparison, and the damage is only visible as slightly worse retrieval,
        never as an error.
        """
        if not texts:
            return []

        prefix = _PREFIXES[kind]
        with metrics.timed() as elapsed:
            response = await self._client.post(
                "/api/embed", json={"model": self.model, "input": [prefix + t for t in texts]}
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings") or []
        # Recorded per *call*, with the batch size alongside, because the thing
        # worth knowing is that batching works: 20 chunks in one call and 20
        # calls of one chunk are the same texts and very different latencies.
        metrics.record_embedding(model=self.model, milliseconds=elapsed[0], texts=len(texts))

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding count mismatch: asked for {len(texts)}, got {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingDimensionError(
                    f"{self.model} returned {len(vector)}-dim vectors, "
                    f"but the schema expects VECTOR({self.dim})"
                )
        return vectors

    async def embed_one(self, text: str, *, kind: EmbeddingKind = "query") -> list[float]:
        """Single embedding. Defaults to "query" -- the only single-text caller
        is search, and a wrong default there is the expensive mistake."""
        return (await self.embed([text], kind=kind))[0]
