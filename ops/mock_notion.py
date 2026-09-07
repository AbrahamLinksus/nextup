"""A fake Notion API, so the connector can be exercised without a real workspace.

This mocks the *HTTP API*, not the connector. That is the whole point: pointing
`NOTION_API_BASE` at this server runs the real `NotionConnector` -- its crawl,
its recursive block flattening, its property-matching-by-type, its markdown
tables -- against content that is checked in and editable. A stubbed connector
would skip exactly the code most worth seeing work.

Three endpoints is the entire surface the connector uses:

    POST /v1/search                    which pages exist, newest edit first
    GET  /v1/pages/{id}                one page and its properties
    GET  /v1/blocks/{id}/children      one level of the block tree

Fixtures are re-read on every request, so editing `ops/notion_fixtures.json`
takes effect immediately -- no restart. Each page's `last_edited_time` defaults
to the fixture file's own mtime, which makes an edit to the file look exactly
like an edit in Notion: the ingest guard accepts it as newer, pages whose text
did not change are rejected by the content hash before any model call, and a
changed date reconciles onto the same calendar event instead of creating a
second one.

Run it:

    .venv/bin/python ops/mock_notion.py          # serves on :8200

Then, in .env:

    NOTION_API_KEY=mock
    NOTION_API_BASE=http://localhost:8200
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException

FIXTURES = Path(__file__).with_name("notion_fixtures.json")
PORT = 8200

app = FastAPI(title="Mock Notion API", version="0.1.0")


# ---------------------------------------------------------------------------
# Rich text
# ---------------------------------------------------------------------------

_SPAN = re.compile(
    r"\[(?P<link_text>[^\]]+)\]\((?P<href>[^)]+)\)"
    r"|\*\*(?P<bold>[^*]+)\*\*"
    r"|(?<!\*)\*(?P<italic>[^*]+)\*(?!\*)"
    r"|`(?P<code>[^`]+)`"
)


def rich_text(text: str) -> list[dict[str, Any]]:
    """Parse inline markdown back into Notion's annotated spans.

    The fixtures are written as markdown because that is what a person can edit,
    but handing the connector pre-rendered markdown inside one plain span would
    mean its annotation handling never runs. Bold is how a deadline is usually
    marked on a real page, so it is worth carrying through properly.
    """
    spans: list[dict[str, Any]] = []
    position = 0

    def plain(content: str, **annotations: bool) -> dict[str, Any]:
        return {
            "type": "text",
            "text": {"content": content},
            "plain_text": content,
            "href": annotations.pop("href", None),
            "annotations": {
                "bold": annotations.get("bold", False),
                "italic": annotations.get("italic", False),
                "strikethrough": False,
                "underline": False,
                "code": annotations.get("code", False),
                "color": "default",
            },
        }

    for match in _SPAN.finditer(text):
        if match.start() > position:
            spans.append(plain(text[position : match.start()]))
        if match.group("link_text"):
            spans.append(plain(match.group("link_text"), href=match.group("href")))
        elif match.group("bold"):
            spans.append(plain(match.group("bold"), bold=True))
        elif match.group("italic"):
            spans.append(plain(match.group("italic"), italic=True))
        elif match.group("code"):
            spans.append(plain(match.group("code"), code=True))
        position = match.end()

    if position < len(text):
        spans.append(plain(text[position:]))
    return [span for span in spans if span["plain_text"]]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def build_properties(title: str, declared: dict[str, Any]) -> dict[str, Any]:
    """Expand the fixture shorthand into real Notion property objects."""
    properties: dict[str, Any] = {
        "Name": {"id": "title", "type": "title", "title": rich_text(title)}
    }
    for name, spec in declared.items():
        kind, value = spec["type"], spec.get("value")
        match kind:
            case "date":
                properties[name] = {"type": "date", "date": {"start": value, "end": None}}
            case "select" | "status":
                properties[name] = {"type": kind, kind: {"name": value, "color": "default"}}
            case "multi_select":
                properties[name] = {
                    "type": kind,
                    kind: [{"name": entry, "color": "default"} for entry in value],
                }
            case "rich_text":
                properties[name] = {"type": kind, kind: rich_text(value)}
            case _:
                properties[name] = {"type": kind, kind: value}
    return properties


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def _block(block_id: str, kind: str, body: dict[str, Any], children=None) -> dict[str, Any]:
    return {
        "object": "block",
        "id": block_id,
        "type": kind,
        kind: body,
        "has_children": bool(children),
        "_children": children or [],
    }


def build_blocks(lines: list[str], prefix: str) -> list[dict[str, Any]]:
    """Expand the body shorthand into a Notion block tree.

    Tables and toggles become parents with real children, because that is the
    shape that makes the connector recurse -- and recursion (with its depth
    limit and its atomic table handling) is the part worth exercising.
    """
    blocks: list[dict[str, Any]] = []
    index = 0
    position = 0

    def next_id() -> str:
        return f"{prefix}-b{index}"

    while position < len(lines):
        line = lines[position]
        stripped = line.strip()

        if not stripped:
            position += 1
            continue

        # Fenced code: everything up to the closing fence is one block.
        if stripped.startswith("```"):
            language = stripped[3:].strip() or "plain text"
            position += 1
            body: list[str] = []
            while position < len(lines) and not lines[position].strip().startswith("```"):
                body.append(lines[position])
                position += 1
            position += 1
            blocks.append(
                _block(
                    next_id(),
                    "code",
                    {"rich_text": rich_text("\n".join(body)), "language": language},
                )
            )
            index += 1
            continue

        # A run of pipe rows is one table. The separator row is dropped: Notion
        # has no such row, it carries the header as a flag on the table itself.
        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while position < len(lines) and lines[position].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[position].strip().strip("|").split("|")]
                if not all(set(cell) <= set("- :") and cell for cell in cells):
                    rows.append(cells)
                position += 1
            table_id = next_id()
            children = [
                _block(
                    f"{table_id}.{row_number}",
                    "table_row",
                    {"cells": [rich_text(cell) for cell in row]},
                )
                for row_number, row in enumerate(rows)
            ]
            blocks.append(
                _block(
                    table_id,
                    "table",
                    {
                        "table_width": max((len(row) for row in rows), default=0),
                        "has_column_header": True,
                        "has_row_header": False,
                    },
                    children,
                )
            )
            index += 1
            continue

        # A toggle owns the indented lines beneath it.
        if stripped.startswith("v "):
            toggle_id = next_id()
            position += 1
            nested: list[str] = []
            while position < len(lines) and (
                lines[position].startswith("  ") or not lines[position].strip()
            ):
                nested.append(lines[position][2:])
                position += 1
            while nested and not nested[-1].strip():
                nested.pop()
            blocks.append(
                _block(
                    toggle_id,
                    "toggle",
                    {"rich_text": rich_text(stripped[2:])},
                    build_blocks(nested, f"{toggle_id}.c"),
                )
            )
            index += 1
            continue

        kind, body = _leaf(stripped)
        blocks.append(_block(next_id(), kind, body))
        index += 1
        position += 1

    return blocks


def _leaf(line: str) -> tuple[str, dict[str, Any]]:
    if line.startswith("### "):
        return "heading_3", {"rich_text": rich_text(line[4:])}
    if line.startswith("## "):
        return "heading_2", {"rich_text": rich_text(line[3:])}
    if line.startswith("# "):
        return "heading_1", {"rich_text": rich_text(line[2:])}
    if line.startswith("- [x] ") or line.startswith("- [ ] "):
        return "to_do", {"rich_text": rich_text(line[6:]), "checked": line[3] == "x"}
    if line.startswith("- "):
        return "bulleted_list_item", {"rich_text": rich_text(line[2:])}
    if re.match(r"^\d+\.\s", line):
        return "numbered_list_item", {"rich_text": rich_text(re.sub(r"^\d+\.\s", "", line))}
    if line.startswith("!! "):
        return "callout", {"rich_text": rich_text(line[3:]), "icon": {"emoji": "⚠️"}}
    if line.startswith("> "):
        return "quote", {"rich_text": rich_text(line[2:])}
    if line == "---":
        return "divider", {}
    return "paragraph", {"rich_text": rich_text(line)}


# ---------------------------------------------------------------------------
# The workspace
# ---------------------------------------------------------------------------

def load_workspace() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read the fixtures fresh and expand them. Returns (pages, blocks-by-id)."""
    raw = json.loads(FIXTURES.read_text())
    edited_default = datetime.fromtimestamp(FIXTURES.stat().st_mtime, UTC).isoformat()

    pages: list[dict[str, Any]] = []
    blocks: dict[str, dict[str, Any]] = {}

    for fixture in raw["pages"]:
        page_id = fixture["id"]
        tree = build_blocks(fixture.get("body", []), page_id)
        blocks[page_id] = {"children": tree}
        _index(tree, blocks)

        pages.append(
            {
                "object": "page",
                "id": page_id,
                "created_time": fixture.get("created_time", edited_default),
                "last_edited_time": fixture.get("last_edited_time", edited_default),
                "url": f"https://notion.mock/{page_id}",
                "archived": False,
                "properties": build_properties(fixture["title"], fixture.get("properties", {})),
            }
        )

    pages.sort(key=lambda page: page["last_edited_time"], reverse=True)
    return pages, blocks


def _index(tree: list[dict[str, Any]], blocks: dict[str, dict[str, Any]]) -> None:
    for block in tree:
        blocks[block["id"]] = {"children": block["_children"]}
        _index(block["_children"], blocks)


def _without_children(block: dict[str, Any]) -> dict[str, Any]:
    """The API returns one level at a time; children are a separate request."""
    return {key: value for key, value in block.items() if key != "_children"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _authorize(authorization: str | None) -> None:
    """Reject an unauthenticated call, as the real API would.

    Worth keeping: it means `NOTION_API_KEY` still has to be set to *something*,
    so the configured path is the same one the real API takes.
    """
    if not authorization or not authorization.startswith("Bearer ") or not authorization[7:]:
        raise HTTPException(status_code=401, detail="API token is invalid.")


@app.post("/v1/search")
async def search(body: dict[str, Any], authorization: str | None = Header(default=None)):
    _authorize(authorization)
    pages, _ = load_workspace()

    page_size = int(body.get("page_size", 100))
    start = int(body.get("start_cursor") or 0)
    window = pages[start : start + page_size]
    has_more = start + page_size < len(pages)

    return {
        "object": "list",
        "results": window,
        "has_more": has_more,
        "next_cursor": str(start + page_size) if has_more else None,
    }


@app.get("/v1/pages/{page_id}")
async def get_page(page_id: str, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    pages, _ = load_workspace()
    for page in pages:
        if page["id"] == page_id:
            return page
    raise HTTPException(status_code=404, detail=f"Could not find page with ID: {page_id}")


@app.get("/v1/blocks/{block_id}/children")
async def get_children(block_id: str, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    _, blocks = load_workspace()
    entry = blocks.get(block_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Could not find block with ID: {block_id}")
    return {
        "object": "list",
        "results": [_without_children(block) for block in entry["children"]],
        "has_more": False,
        "next_cursor": None,
    }


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    print(f"Mock Notion workspace from {FIXTURES}")
    print(f"Point NOTION_API_BASE at http://localhost:{PORT} and set NOTION_API_KEY to anything.\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
