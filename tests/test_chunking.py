"""Chunking guarantees: stability, identity, and atomicity."""

from __future__ import annotations

from assistant.chunking import chunk_markdown, split_blocks, split_sections
from assistant.models import ChunkType

DOC = """# Course

Intro paragraph.

## Assessment

The exam is on 28 August. It covers chapters 4-7.

| Topic | Weight |
|---|---|
| Normalization | 40% |
| Transactions | 60% |

Bring a calculator.

```sql
SELECT * FROM students WHERE gpa > 8;
```
"""


def test_tables_and_code_are_their_own_chunks():
    kinds = {chunk.chunk_type for chunk in chunk_markdown(DOC)}
    assert ChunkType.TABLE in kinds
    assert ChunkType.CODE in kinds


def test_a_table_is_never_split_from_its_header():
    table = next(c for c in chunk_markdown(DOC) if c.chunk_type is ChunkType.TABLE)
    assert "| Topic | Weight |" in table.body
    assert "| Normalization | 40% |" in table.body
    assert "| Transactions | 60% |" in table.body


def test_atomic_blocks_do_not_absorb_surrounding_prose():
    table = next(c for c in chunk_markdown(DOC) if c.chunk_type is ChunkType.TABLE)
    assert "Bring a calculator" not in table.body
    assert "covers chapters" not in table.body


def test_chunk_keys_survive_an_insertion_above_them():
    """The property the lifecycle table depends on.

    An edit at the top of a page shifts every chunk_index. If identity moved with
    it, reconciliation would update the calendar event belonging to a different
    deadline.
    """
    before = {c.chunk_key: c.content_hash for c in chunk_markdown(DOC)}
    edited = DOC.replace("Intro paragraph.", "A new sentence.\n\nIntro paragraph.")
    after = {c.chunk_key: c.content_hash for c in chunk_markdown(edited)}

    assert set(before) == set(after), "insertion changed which chunk keys exist"

    unchanged = [key for key in before if before[key] == after[key]]
    assert "Course > Assessment#0" in unchanged
    assert before["Course#0"] != after["Course#0"], "the edited chunk should differ"


def test_editing_one_section_leaves_others_byte_identical():
    before = {c.chunk_key: c.content for c in chunk_markdown(DOC)}
    edited = DOC.replace("Bring a calculator.", "Bring a calculator and a pen.")
    after = {c.chunk_key: c.content for c in chunk_markdown(edited)}

    untouched = [key for key in before if before[key] == after.get(key)]
    assert len(untouched) >= 3, "an unrelated edit re-wrote too many chunks"


def test_chunking_is_deterministic():
    first = [(c.chunk_key, c.content_hash) for c in chunk_markdown(DOC)]
    second = [(c.chunk_key, c.content_hash) for c in chunk_markdown(DOC)]
    assert first == second


def test_heading_breadcrumb_is_embedded_with_the_body():
    chunk = next(c for c in chunk_markdown(DOC) if c.chunk_key.startswith("Course > Assessment"))
    assert chunk.content.startswith("Course > Assessment")
    assert not chunk.body.startswith("Course > Assessment")


def test_repeated_heading_paths_get_distinct_keys():
    doc = "## Notes\n\nFirst.\n\n# Other\n\nx\n\n## Notes\n\nSecond."
    keys = [c.chunk_key for c in chunk_markdown(doc)]
    assert len(keys) == len(set(keys))


def test_fenced_code_hides_markdown_structure():
    doc = "# Real\n\ntext\n\n```py\n# not a heading\n\nstill code\n```\n"
    sections = split_sections(doc)
    assert [s.heading_path for s in sections] == [("Real",)]


def test_oversized_table_repeats_its_header():
    rows = "\n".join(f"| topic{i} | {i}% |" for i in range(400))
    doc = f"# T\n\n| Topic | Weight |\n|---|---|\n{rows}\n"
    chunks = chunk_markdown(doc, atomic_max_chars=1200)
    tables = [c for c in chunks if c.chunk_type is ChunkType.TABLE]
    assert len(tables) > 1
    assert all("| Topic | Weight |" in chunk.body for chunk in tables)


def test_blocks_are_typed():
    blocks = split_blocks("para\n\n| a | b |\n|---|---|\n\n```\ncode\n```")
    assert [block.kind for block in blocks] == [
        ChunkType.TEXT,
        ChunkType.TABLE,
        ChunkType.CODE,
    ]
