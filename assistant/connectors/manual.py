"""Direct memory input: content the user hands over rather than a source pushes.

This is a `SourceConnector` in name only -- there is nothing to poll and nothing
to watch -- but it produces the same `Item`, into the same table, through the
same chunk-and-embed path. One schema, one embedding space, no branching. That
is the "conform the input to the schema" principle applied to the input layer
itself.

Non-text input (an image, a PDF, a document) is converted to text or markdown
*first* and then flows through the identical pipeline. There is no separate
image path; there is a conversion step in front of the only path.

**Stated rather than assumed:** directly-added content skips triage and goes
straight to embedding. It is declared memory, not a fetched item to judge for
action. Wanting something typed in to *also* become an action is a different
request, and it goes through the conversational agent's direct-action path,
where an explicit human instruction is the confidence signal that gating exists
to substitute for.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import structlog

from assistant.connectors.base import SourceConnector
from assistant.models import Item, TrustLevel

log = structlog.get_logger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".org", ".csv", ".json", ".yaml", ".yml"}


class ManualConnector(SourceConnector):
    """Not fetched from anywhere. Present so manual input is not a special case."""

    source_type = "manual"
    supports_push = False
    trust_level = TrustLevel.TRUSTED

    async def list_items(self, since: datetime | None = None, *, limit: int = 100) -> list[Item]:
        return []

    async def fetch_item(self, item_id: str) -> Item:
        raise NotImplementedError("manual items are supplied, never fetched")

    def as_tool_schema(self):  # noqa: ANN201 - matches the base signature
        """Not exposed to the agent: 'check my manual notes' is not a fetch."""
        raise NotImplementedError("the manual source has no on-demand fetch tool")


def make_item(
    content: str,
    *,
    title: str = "",
    url: str = "",
    source_id: str | None = None,
) -> Item:
    """Build a manual item.

    The id is derived from the content hash when not supplied, so pasting the
    same note twice updates one row instead of accumulating near-duplicates that
    both surface in retrieval.
    """
    now = datetime.now(UTC)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "note")

    return Item(
        source_type="manual",
        source_id=source_id or digest,
        url=url,
        title=title or first_line[:120],
        content=content,
        created_at=now,
        last_edited_at=now,
        raw_properties={"origin": "direct-input"},
        trust_level=TrustLevel.TRUSTED,
    )


def item_from_file(path: str | Path) -> Item:
    """Read a local file into an Item, converting non-text formats to markdown.

    Conversion happens here rather than downstream so that everything past this
    point sees text and only text.
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        content = _pdf_to_markdown(resolved)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}:
        content = _image_to_markdown(resolved)
    else:
        # Unknown binary formats are refused rather than read as mojibake: an
        # item whose content is decoded noise pollutes the embedding space and
        # is worse than an item that never existed.
        raise UnsupportedInput(f"no text conversion for {suffix!r} ({resolved.name})")

    return make_item(content, title=resolved.stem, url=resolved.as_uri())


class UnsupportedInput(RuntimeError):
    pass


def _pdf_to_markdown(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise UnsupportedInput(
            "PDF input needs `pypdf` installed (pip install pypdf)"
        ) from exc

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"## Page {number}\n\n{text}")
    if not pages:
        # A scanned PDF has no extractable text; captioning is the OCR path, and
        # returning an empty item would silently store nothing.
        raise UnsupportedInput(f"{path.name} has no extractable text (scanned?)")
    return "\n\n".join(pages)


def _image_to_markdown(path: Path) -> str:
    """Caption an image with a local vision model via Ollama.

    Deliberately synchronous-looking and best-effort: image input is a
    convenience path, and the design already accepts a placeholder for v1 rather
    than treating OCR as a blocking dependency.
    """
    import base64

    import httpx

    from assistant.config import get_settings

    settings = get_settings()
    encoded = base64.b64encode(path.read_bytes()).decode()
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_reasoning_model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Transcribe every piece of text in this image, then "
                            "describe what it shows in one sentence. Preserve "
                            "table structure as a markdown table if present."
                        ),
                        "images": [encoded],
                    }
                ],
                "stream": False,
            },
            timeout=180.0,
        )
        response.raise_for_status()
        described = response.json()["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 - conversion is best-effort by design
        log.warning("manual.image_caption_failed", path=str(path), error=str(exc))
        raise UnsupportedInput(
            f"could not caption {path.name}: {exc}. Set OLLAMA_REASONING_MODEL to a "
            f"vision-capable model, or add the text by hand."
        ) from exc

    return f"# {path.stem}\n\n[image: {path.name}]\n\n{described}"
