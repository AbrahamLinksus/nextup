"""Notion connector: pages and their block trees, flattened to markdown.

Markdown, not plain text. Joining a table's cell text loses which value belongs
to which row and column -- "Normalization 40% Transactions 60%" is not the table
it came from. Embedding models and LLMs are both trained to read markdown
structure back out, so pipes, fences, and headings survive the trip in a form
that still means what it meant.

Two things here are the resolved form of open questions in the design log:

* **Flatten depth.** Notion nesting is unbounded in principle (toggles inside
  toggles inside columns). Everything below `max_depth` is dropped with a
  visible marker rather than silently truncated, so a classifier reading the
  result can tell the difference between "nothing there" and "not shown".
* **Property matching by type, not label.** A due date might be called "Due
  Date", "Deadline", or "Target"; what does not vary is that it is a
  date-typed property. Names are used only to *rank* candidates when a page has
  several date properties, never to find them in the first place.

Trigger strategy is polling, not webhooks. Notion webhook coverage is opt-in per
page, so a subscription set is something to maintain and get wrong; a periodic
crawl over everything shared with the integration has no such failure mode. The
search endpoint returns exactly what the integration can see, which makes the
opt-in boundary explicit instead of silent.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from assistant.config import get_settings
from assistant.connectors.base import SourceConnector
from assistant.dates import parse_iso_local
from assistant.models import Item, TrustLevel

log = structlog.get_logger(__name__)


# Deeper than this and nested content stops being context and starts being
# noise for the classifier -- a to-do four toggles down inside an archive page
# is not what a page is "about". Configurable, but a default has to be chosen.
MAX_BLOCK_DEPTH = 4

# Ranking hints for picking *which* date property is the deadline when a page
# has several. Order matters; first match wins.
_DEADLINE_NAME_HINTS = (
    "deadline", "due", "due date", "submission", "submit by", "last date",
    "target", "exam", "test", "scheduled",
)
_STATUS_NAME_HINTS = ("status", "state", "progress", "stage")
_URGENCY_NAME_HINTS = ("priority", "urgency", "importance", "severity")


class NotionConnector(SourceConnector):
    source_type = "notion"
    supports_push = False
    trust_level = TrustLevel.TRUSTED

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_depth: int = MAX_BLOCK_DEPTH,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.notion_api_key
        self.max_depth = max_depth
        self._client = httpx.AsyncClient(
            base_url=settings.notion_api_base,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Notion-Version": settings.notion_version,
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    # -- HTTP ------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=10),
        reraise=True,
    )
    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not self.api_key:
            raise NotionNotConfigured("NOTION_API_KEY is not set")
        response = await self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    # -- SourceConnector -------------------------------------------------

    async def list_items(self, since: datetime | None = None, *, limit: int = 100) -> list[Item]:
        """Pages edited since `since`, newest first.

        The search endpoint sorts by last_edited_time descending, so the crawl
        stops at the first page older than the cursor rather than reading the
        whole workspace to find out nothing changed.
        """
        cutoff = since or datetime.now(UTC) - timedelta(days=30)
        items: list[Item] = []
        cursor: str | None = None

        while len(items) < limit:
            body: dict[str, Any] = {
                "filter": {"value": "page", "property": "object"},
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": min(100, limit - len(items)),
            }
            if cursor:
                body["start_cursor"] = cursor

            payload = await self._request("POST", "/v1/search", json=body)
            page_batch = payload.get("results", [])
            if not page_batch:
                break

            stop = False
            for page in page_batch:
                edited = parse_iso_local(page.get("last_edited_time"))
                if edited is not None and edited <= cutoff:
                    stop = True
                    break
                items.append(await self._page_to_item(page))

            if stop or not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")

        log.info("notion.listed", count=len(items), since=cutoff.isoformat())
        return items

    async def fetch_item(self, item_id: str) -> Item:
        page = await self._request("GET", f"/v1/pages/{item_id}")
        return await self._page_to_item(page)

    # -- Conversion ------------------------------------------------------

    async def _page_to_item(self, page: dict[str, Any]) -> Item:
        page_id = page["id"]
        properties = page.get("properties", {})
        blocks = await self._fetch_blocks(page_id, depth=0)

        title = _page_title(properties) or "(untitled)"
        content = blocks_to_markdown(blocks)
        deadline, status, urgency = normalize_properties(properties)

        return Item(
            source_type=self.source_type,
            source_id=page_id,
            url=page.get("url", ""),
            title=title,
            content=content,
            created_at=parse_iso_local(page.get("created_time")) or datetime.now(UTC),
            last_edited_at=parse_iso_local(page.get("last_edited_time")) or datetime.now(UTC),
            deadline=deadline,
            status=status,
            urgency_hint=urgency,
            raw_properties=flatten_properties(properties),
            trust_level=self.trust_level,
        )

    async def _fetch_blocks(self, block_id: str, *, depth: int) -> list[dict[str, Any]]:
        """Recursively pull a block subtree, annotating each block with its depth."""
        if depth > self.max_depth:
            return [{"type": "_truncated", "_depth": depth}]

        blocks: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            payload = await self._request(
                "GET", f"/v1/blocks/{block_id}/children", params=params
            )
            for block in payload.get("results", []):
                block["_depth"] = depth
                blocks.append(block)
                if block.get("has_children"):
                    block["_children"] = await self._fetch_blocks(
                        block["id"], depth=depth + 1
                    )
            if not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")
        return blocks


class NotionNotConfigured(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Block -> markdown
# ---------------------------------------------------------------------------

def rich_text_to_markdown(rich: list[dict[str, Any]]) -> str:
    """Render Notion rich text, preserving the annotations that carry meaning.

    Bold and italic are kept because emphasis is often exactly how a deadline is
    marked ("**due Friday**"); dropping it would flatten the strongest signal on
    the page into ordinary prose.
    """
    parts: list[str] = []
    for span in rich or []:
        text = span.get("plain_text", "")
        if not text:
            continue
        annotations = span.get("annotations", {})
        if annotations.get("code"):
            text = f"`{text}`"
        if annotations.get("bold"):
            text = f"**{text}**"
        if annotations.get("italic"):
            text = f"*{text}*"
        if annotations.get("strikethrough"):
            text = f"~~{text}~~"
        href = span.get("href")
        if href:
            text = f"[{text}]({href})"
        parts.append(text)
    return "".join(parts)


def _table_to_markdown(block: dict[str, Any]) -> str:
    """Render a Notion table as a pipe table, header separator included.

    The separator row is what makes downstream chunking able to recognise this
    as a table and keep it atomic. Without it a table is just lines with pipes.
    """
    rows = [
        child
        for child in block.get("_children", [])
        if child.get("type") == "table_row"
    ]
    if not rows:
        return ""

    rendered: list[list[str]] = []
    for row in rows:
        cells = row.get("table_row", {}).get("cells", [])
        rendered.append([rich_text_to_markdown(cell).replace("|", "\\|") for cell in cells])

    width = max((len(row) for row in rendered), default=0)
    if width == 0:
        return ""
    rendered = [row + [""] * (width - len(row)) for row in rendered]

    has_header = block.get("table", {}).get("has_column_header", False)
    header = rendered[0] if has_header else [""] * width
    body = rendered[1:] if has_header else rendered

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _block_to_markdown(block: dict[str, Any]) -> str:
    """One block, without its children. Returns '' for blocks with no text form."""
    kind = block.get("type", "")
    body = block.get(kind, {}) if isinstance(block.get(kind), dict) else {}
    text = rich_text_to_markdown(body.get("rich_text", []))

    match kind:
        case "paragraph":
            return text
        case "heading_1":
            return f"# {text}"
        case "heading_2":
            return f"## {text}"
        case "heading_3":
            return f"### {text}"
        case "bulleted_list_item":
            return f"- {text}"
        case "numbered_list_item":
            return f"1. {text}"
        case "to_do":
            # Checked state is a real status signal -- an unticked box with a
            # date is an open obligation, a ticked one is history.
            mark = "x" if body.get("checked") else " "
            return f"- [{mark}] {text}"
        case "toggle":
            return f"- {text}"
        case "quote":
            return f"> {text}"
        case "callout":
            icon = (body.get("icon") or {}).get("emoji", "")
            return f"> {icon} {text}".strip()
        case "code":
            language = body.get("language", "")
            return f"```{language}\n{text}\n```"
        case "divider":
            return "---"
        case "equation":
            return f"$$\n{body.get('expression', '')}\n$$"
        case "child_page":
            return f"### {body.get('title', '(subpage)')}"
        case "child_database":
            return f"### {body.get('title', '(database)')}"
        case "bookmark" | "embed" | "link_preview":
            url = body.get("url", "")
            return f"[{text or url}]({url})" if url else text
        case "table":
            return _table_to_markdown(block)
        case "table_row":
            return ""  # consumed by the parent table
        case "image" | "file" | "video" | "pdf" | "audio":
            # No markdown representation, and flattening cannot invent one.
            # A visible placeholder is honest; silently dropping the block would
            # make an exam timetable posted as a screenshot look like an empty
            # page. OCR/captioning is a separate subsystem, deferred for v1.
            caption = rich_text_to_markdown(body.get("caption", []))
            label = caption or f"{kind} attachment"
            return f"[unprocessed {kind}: {label}]"
        case "_truncated":
            return "[nested content below the flattening depth limit was not included]"
        case "column_list" | "column" | "synced_block" | "table_of_contents":
            return ""  # structural only; children carry the text
        case _:
            return text


def _render_blocks(blocks: list[dict[str, Any]], indent: int) -> list[str]:
    """Render a block list to lines, deciding where blank lines belong.

    Blank-line placement is not cosmetic here -- it is what the chunker splits
    on. Consecutive list items must stay together (one list is one thought), and
    a paragraph must be separated from its neighbours (two paragraphs are two).
    Getting this wrong either fuses a whole page into one chunk or shatters a
    bullet list into one chunk per bullet.
    """
    lines: list[str] = []
    pad = "  " * indent

    for position, block in enumerate(blocks):
        rendered = _block_to_markdown(block)
        if rendered:
            lines.extend(f"{pad}{line}" if line else "" for line in rendered.split("\n"))

        children = block.get("_children")
        if children and block.get("type") != "table":
            nested_indent = indent + 1 if _is_list_like(block) else indent
            lines.extend(_render_blocks(children, nested_indent))

        if not rendered:
            continue

        following = blocks[position + 1] if position + 1 < len(blocks) else None
        both_list_items = (
            _is_list_like(block) and following is not None and _is_list_like(following)
        )
        if not both_list_items:
            lines.append("")

    return lines


def blocks_to_markdown(blocks: list[dict[str, Any]], *, indent: int = 0) -> str:
    """Flatten a block tree into markdown, indenting nested list content."""
    # Collapse runs of blank lines; Notion emits a lot of empty paragraphs and
    # they would otherwise fragment a section into several chunks.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(_render_blocks(blocks, indent))).strip()


def _is_list_like(block: dict[str, Any]) -> bool:
    return block.get("type") in {
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "toggle",
    }


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def _page_title(properties: dict[str, Any]) -> str:
    for prop in properties.values():
        if prop.get("type") == "title":
            return rich_text_to_markdown(prop.get("title", []))
    return ""


def _rank(name: str, hints: tuple[str, ...]) -> int:
    """Lower is better; len(hints) means 'no hint matched'."""
    lowered = name.lower()
    for position, hint in enumerate(hints):
        if hint in lowered:
            return position
    return len(hints)


def normalize_properties(
    properties: dict[str, Any],
) -> tuple[datetime | None, str | None, str | None]:
    """Map Notion's native properties into the fixed normalized vocabulary.

    Matching is by *type* first: a date property is a date property whatever it
    is called. Names only break ties, because a page can carry several date
    properties ("Created", "Reviewed", "Due") and picking the wrong one is worse
    than picking none -- an auto-created event on the review date is a false
    entry, whereas no deadline just routes the item to the queue.

    Anything that does not confidently map is left `None`. That is the designed
    behaviour, not a gap: consumers already handle absent signals because Gmail
    has almost none.
    """
    date_candidates: list[tuple[int, datetime]] = []
    status: str | None = None
    status_rank = 99
    urgency: str | None = None
    urgency_rank = 99

    for name, prop in properties.items():
        kind = prop.get("type")

        if kind == "date":
            value = (prop.get("date") or {}).get("start")
            parsed = parse_iso_local(value) if value else None
            if parsed is not None:
                date_candidates.append((_rank(name, _DEADLINE_NAME_HINTS), parsed))

        elif kind == "status":
            value = (prop.get("status") or {}).get("name")
            rank = _rank(name, _STATUS_NAME_HINTS)
            if value and rank < status_rank:
                status, status_rank = value, rank

        elif kind == "select":
            value = (prop.get("select") or {}).get("name")
            if not value:
                continue
            status_hit = _rank(name, _STATUS_NAME_HINTS)
            urgency_hit = _rank(name, _URGENCY_NAME_HINTS)
            if status_hit < len(_STATUS_NAME_HINTS) and status_hit < status_rank:
                status, status_rank = value, status_hit
            elif urgency_hit < len(_URGENCY_NAME_HINTS) and urgency_hit < urgency_rank:
                urgency, urgency_rank = value, urgency_hit

        elif kind == "multi_select":
            values = [entry.get("name") for entry in prop.get("multi_select", [])]
            rank = _rank(name, _URGENCY_NAME_HINTS)
            if values and rank < len(_URGENCY_NAME_HINTS) and rank < urgency_rank:
                urgency, urgency_rank = ", ".join(filter(None, values)), rank

        elif kind == "checkbox" and _rank(name, ("done", "complete", "finished")) < 3:
            if status is None:
                status = "done" if prop.get("checkbox") else "not started"

    deadline = None
    if date_candidates:
        # Best name match first; among equals, the earliest date -- the nearest
        # obligation is the one that matters.
        date_candidates.sort(key=lambda pair: (pair[0], pair[1]))
        deadline = date_candidates[0][1]

    return deadline, status, urgency


def flatten_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """A readable passthrough of every property, for what the fixed set misses.

    Kept simple rather than faithful: the operations agent reads this to fill an
    argument the normalized trio does not cover (a "Location" for an in-person
    exam), and a nested Notion property object is worse input for that than a
    plain string.
    """
    out: dict[str, Any] = {}
    for name, prop in properties.items():
        kind = prop.get("type")
        match kind:
            case "title" | "rich_text":
                value = rich_text_to_markdown(prop.get(kind, []))
            case "select" | "status":
                value = (prop.get(kind) or {}).get("name")
            case "multi_select":
                value = [entry.get("name") for entry in prop.get(kind, [])]
            case "date":
                value = prop.get("date") or None
            case "checkbox" | "number" | "url" | "email" | "phone_number":
                value = prop.get(kind)
            case "people":
                value = [person.get("name") for person in prop.get("people", [])]
            case "files":
                value = [entry.get("name") for entry in prop.get("files", [])]
            case "formula":
                formula = prop.get("formula", {})
                value = formula.get(formula.get("type", ""), None)
            case _:
                continue
        if value not in (None, "", [], {}):
            out[name] = value
    return out
