"""Structure-aware markdown chunking with atomic table and code blocks.

Three properties this module guarantees, all covered by tests:

1. **Boundary stability.** Chunk boundaries follow document structure, not
   character position, so editing one section leaves every other section's text
   byte-identical -- its hash matches and it is not re-embedded.

2. **Identity stability.** A chunk's identity is `chunk_key` (heading path plus
   an ordinal *within that section*), never `chunk_index`. Inserting a paragraph
   at the top of a page shifts every index but no key. This is what keeps a
   lifecycle row pointing at the same content across edits -- without it,
   reconciliation would update the calendar event belonging to a different
   deadline.

3. **Atomicity.** Tables and fenced code are never split mid-block. Half a table
   is not a smaller table: without its header row, nobody -- model or human --
   can say which value belongs to which column. The same goes for a function cut
   mid-body. These become their own chunks whatever their size, and only a
   pathologically large one is split at all (by rows, with the header repeated).

The hashing invariant: `content_hash` is the SHA-256 of exactly the string that
gets embedded, so equal hash implies equal embedding input.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from assistant.models import Chunk, ChunkType

# Roughly 400-450 tokens at ~4 chars/token -- comfortably inside
# nomic-embed-text's 8192-token window, and small enough that a retrieved chunk
# is targeted evidence rather than a whole section dump.
DEFAULT_MAX_CHARS = 1800

# Ceiling past which even an "atomic" block has to be broken up, because a
# single chunk larger than the embedding window is not embeddable at all.
# ~6000 tokens, leaving headroom under 8192 for the heading breadcrumb.
ATOMIC_MAX_CHARS = 24000

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_PREAMBLE_KEY = "(preamble)"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Section:
    heading_path: tuple[str, ...]
    body: str


@dataclass
class Block:
    """One structural unit within a section, before packing."""

    text: str
    kind: ChunkType = ChunkType.TEXT

    @property
    def atomic(self) -> bool:
        """Tables and code never merge with a neighbour or split down the middle."""
        return self.kind is not ChunkType.TEXT


@dataclass
class _Packed:
    kind: ChunkType
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)


# ---------------------------------------------------------------------------
# Lexing helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Strip insignificant whitespace so cosmetic edits do not force a re-embed."""
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _iter_lines_with_fence_state(text: str):
    """Yield (line, in_fence) so markdown structure inside code blocks is ignored.

    A `# comment` line inside a fenced block is not a heading, and a blank line
    inside one is not a paragraph break.
    """
    in_fence = False
    fence_char = ""
    for line in text.split("\n"):
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, marker
                yield line, True
                continue
            if marker == fence_char:
                in_fence, fence_char = False, ""
                yield line, True
                continue
        yield line, in_fence


def split_sections(text: str) -> list[Section]:
    """Split markdown into sections keyed by heading path.

    Handles skipped heading levels (an h1 followed directly by an h3) with a
    level-annotated stack rather than assuming depth equals list length.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    body_lines: list[str] = []

    def flush() -> None:
        body = _normalize("\n".join(body_lines))
        if body:
            sections.append(Section(tuple(title for _, title in stack), body))

    for line, in_fence in _iter_lines_with_fence_state(text):
        if not in_fence:
            heading = _ATX_HEADING_RE.match(line)
            if heading:
                flush()
                body_lines = []
                level = len(heading.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading.group(2).strip()))
                continue
        body_lines.append(line)

    flush()
    return sections


def split_blocks(body: str) -> list[Block]:
    """Split a section body into typed blocks.

    Three block kinds are recognised, and the two non-text kinds are what make
    the flattening decision (markdown, not plain text) pay off: a pipe table is
    recognisable *as* a table only because the connector preserved the pipes.
    """
    blocks: list[Block] = []
    current: list[str] = []
    current_kind = ChunkType.TEXT

    def flush() -> None:
        nonlocal current, current_kind
        joined = "\n".join(current).strip()
        if joined:
            blocks.append(Block(joined, current_kind))
        current = []
        current_kind = ChunkType.TEXT

    in_fence = False
    fence_char = ""

    for line in body.split("\n"):
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                flush()
                in_fence, fence_char = True, marker
                current_kind = ChunkType.CODE
                current.append(line)
                continue
            if marker == fence_char:
                current.append(line)
                in_fence, fence_char = False, ""
                flush()
                continue

        if in_fence:
            current.append(line)
            continue

        is_table_row = bool(_TABLE_ROW_RE.match(line))
        if is_table_row and current_kind is not ChunkType.TABLE:
            flush()
            current_kind = ChunkType.TABLE
        elif not is_table_row and current_kind is ChunkType.TABLE:
            flush()

        if not line.strip():
            if current_kind is not ChunkType.TABLE:
                flush()
            continue

        current.append(line)

    flush()
    return blocks


# ---------------------------------------------------------------------------
# Splitting oversized content
# ---------------------------------------------------------------------------

def _split_table(block: str, max_chars: int) -> list[str]:
    """Split a pathologically large table by rows, repeating the header.

    Repeating the header is the whole point: the reason tables are atomic is
    that a body row without its header is unreadable, so a split that drops the
    header would defeat the rule it is a fallback for.
    """
    lines = [line for line in block.split("\n") if line.strip()]
    header: list[str] = []
    if len(lines) >= 2 and set(lines[1].replace("|", "").strip()) <= set("-: "):
        header = lines[:2]
        lines = lines[2:]

    out: list[str] = []
    current = list(header)
    size = sum(len(line) + 1 for line in current)
    for line in lines:
        if current and len(current) > len(header) and size + len(line) + 1 > max_chars:
            out.append("\n".join(current))
            current = [*header, line]
            size = sum(len(part) + 1 for part in current)
        else:
            current.append(line)
            size += len(line) + 1
    if current and len(current) > len(header):
        out.append("\n".join(current))
    return out or [block]


def _split_code(block: str, max_chars: int) -> list[str]:
    """Split a pathologically large fenced block at line boundaries, re-fencing."""
    lines = block.split("\n")
    opener = lines[0] if _FENCE_RE.match(lines[0]) else "```"
    closer = opener.strip()[:3]
    body = lines[1:-1] if _FENCE_RE.match(lines[-1]) else lines[1:]

    out: list[str] = []
    current: list[str] = []
    size = 0
    for line in body:
        if current and size + len(line) + 1 > max_chars:
            out.append("\n".join([opener, *current, closer]))
            current, size = [line], len(line) + 1
        else:
            current.append(line)
            size += len(line) + 1
    if current:
        out.append("\n".join([opener, *current, closer]))
    return out or [block]


def _split_text(block: str, max_chars: int) -> list[str]:
    """Backstop for a single prose block larger than max_chars."""
    if len(block) <= max_chars:
        return [block]
    for pieces, joiner in ((_SENTENCE_BOUNDARY_RE.split(block), " "), (block.split("\n"), "\n")):
        if len(pieces) <= 1:
            continue
        packed = _pack(pieces, max_chars, joiner)
        if all(len(piece) <= max_chars for piece in packed):
            return packed
    return [block[i : i + max_chars] for i in range(0, len(block), max_chars)]


def _pack(pieces: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily pack pieces into groups no larger than max_chars."""
    out: list[str] = []
    current: list[str] = []
    size = 0
    for raw in pieces:
        piece = raw.strip()
        if not piece:
            continue
        addition = len(piece) + (len(joiner) if current else 0)
        if current and size + addition > max_chars:
            out.append(joiner.join(current))
            current, size = [piece], len(piece)
        else:
            current.append(piece)
            size += addition
    if current:
        out.append(joiner.join(current))
    return out


def _render(heading_path: tuple[str, ...], body: str) -> str:
    """Prepend the heading breadcrumb to the embedded text.

    A chunk reading "covers chapters 4-7" is ambiguous alone; "DBMS > Assessment
    2" in front of it is what lets retrieval tell two courses' assessments apart,
    and what gives the classifier the context to judge the fragment at all.
    """
    if not heading_path:
        return body
    return " > ".join(heading_path) + "\n\n" + body


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_markdown(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    atomic_max_chars: int = ATOMIC_MAX_CHARS,
) -> list[Chunk]:
    """Split a markdown item into structure-aware, stably-identified chunks.

    Deterministic: the same input always produces the same keys, contents, and
    hashes -- which is what makes reprocessing a redelivered event safe.
    """
    chunks: list[Chunk] = []
    seen_paths: dict[str, int] = {}
    index = 0

    for section in split_sections(text):
        path_str = " > ".join(section.heading_path) or _PREAMBLE_KEY
        occurrence = seen_paths.get(path_str, 0)
        seen_paths[path_str] = occurrence + 1
        section_key = path_str if occurrence == 0 else f"{path_str}~{occurrence}"

        for ordinal, packed in enumerate(_pack_blocks(section.body, max_chars, atomic_max_chars)):
            rendered = _render(section.heading_path, packed.text)
            chunks.append(
                Chunk(
                    chunk_key=f"{section_key}#{ordinal}",
                    heading_path=section.heading_path,
                    chunk_index=index,
                    chunk_type=packed.kind,
                    content=rendered,
                    body=packed.text,
                    content_hash=content_hash(rendered),
                )
            )
            index += 1

    return chunks


def _pack_blocks(body: str, max_chars: int, atomic_max_chars: int) -> list[_Packed]:
    """Group a section's blocks into chunks, honouring atomicity.

    Text blocks merge with their neighbours up to `max_chars`. Table and code
    blocks stand alone -- they neither absorb surrounding prose nor get absorbed
    by it, because a chunk that is half explanation and half table cannot be
    typed as either.
    """
    packed: list[_Packed] = []
    current: _Packed | None = None

    for block in split_blocks(body):
        if block.atomic:
            if current is not None:
                packed.append(current)
                current = None
            splitter = _split_table if block.kind is ChunkType.TABLE else _split_code
            pieces = (
                [block.text]
                if len(block.text) <= atomic_max_chars
                else splitter(block.text, atomic_max_chars)
            )
            packed.extend(_Packed(block.kind, [piece]) for piece in pieces)
            continue

        for piece in _split_text(block.text, max_chars):
            if current is None:
                current = _Packed(ChunkType.TEXT, [piece])
            elif len(current.text) + len(piece) + 2 <= max_chars:
                current.parts.append(piece)
            else:
                packed.append(current)
                current = _Packed(ChunkType.TEXT, [piece])

    if current is not None:
        packed.append(current)
    return packed
