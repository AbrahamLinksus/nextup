"""The fixture-driven Notion mock.

Worth testing because its whole value is fidelity: if it emits shapes the real
API would not, then everything it "verifies" about the connector is verifying
the mock instead. The strongest check available is the round trip -- shorthand
in, real connector out, markdown that matches what was written.
"""

from __future__ import annotations

import pytest

from assistant.connectors.notion import blocks_to_markdown, normalize_properties
from ops.mock_notion import build_blocks, build_properties, load_workspace, rich_text


def flatten(lines: list[str]) -> str:
    """Expand shorthand, then hand it to the connector's own flattener.

    The children a real crawl would fetch separately are already attached as
    `_children`, which is the shape `_fetch_blocks` leaves behind.
    """
    return blocks_to_markdown(build_blocks(lines, "test"))


# --- rich text -------------------------------------------------------------

def test_bold_becomes_an_annotation_not_literal_asterisks():
    """Emphasis is often how a deadline is marked; the connector reads annotations."""
    spans = rich_text("due **Friday** sharp")

    assert [span["plain_text"] for span in spans] == ["due ", "Friday", " sharp"]
    assert spans[1]["annotations"]["bold"] is True


def test_a_link_keeps_its_target():
    spans = rich_text("see [the portal](https://portal.test)")
    assert spans[-1]["href"] == "https://portal.test"


def test_plain_text_is_one_span():
    assert len(rich_text("nothing special here")) == 1


# --- blocks ----------------------------------------------------------------

@pytest.mark.parametrize(
    ("shorthand", "expected"),
    [
        ("# Title", "heading_1"),
        ("## Section", "heading_2"),
        ("- point", "bulleted_list_item"),
        ("1. first", "numbered_list_item"),
        ("- [ ] open", "to_do"),
        ("> quoted", "quote"),
        ("!! warning", "callout"),
        ("plain prose", "paragraph"),
    ],
)
def test_shorthand_maps_onto_real_notion_block_types(shorthand, expected):
    assert build_blocks([shorthand], "p")[0]["type"] == expected


def test_a_ticked_box_carries_its_checked_state():
    """An unticked box with a date is an open obligation; a ticked one is history."""
    blocks = build_blocks(["- [x] done", "- [ ] not done"], "p")
    assert blocks[0]["to_do"]["checked"] is True
    assert blocks[1]["to_do"]["checked"] is False


def test_a_table_becomes_a_parent_with_row_children():
    """Notion has no separator row -- the header is a flag on the table itself."""
    blocks = build_blocks(["| A | B |", "| --- | --- |", "| 1 | 2 |"], "p")
    table = blocks[0]

    assert table["type"] == "table"
    assert table["has_children"] is True
    assert table["table"]["has_column_header"] is True
    assert [child["type"] for child in table["_children"]] == ["table_row", "table_row"]


def test_a_toggle_owns_the_lines_indented_under_it():
    blocks = build_blocks(["v Notes", "  inside the toggle", "  - and a bullet"], "p")

    assert blocks[0]["type"] == "toggle"
    assert [child["type"] for child in blocks[0]["_children"]] == [
        "paragraph",
        "bulleted_list_item",
    ]


def test_block_ids_are_unique_across_a_nested_tree():
    """The children endpoint is addressed by id; a collision would serve the wrong subtree."""
    blocks = build_blocks(["v One", "  a", "  b", "v Two", "  c"], "p")
    ids = []

    def walk(tree):
        for block in tree:
            ids.append(block["id"])
            walk(block["_children"])

    walk(blocks)
    assert len(ids) == len(set(ids))


# --- the round trip --------------------------------------------------------

def test_a_table_survives_the_round_trip_as_a_markdown_table():
    markdown = flatten(["| Unit | Weight |", "| --- | --- |", "| 4 | 30% |"])

    assert "| Unit | Weight |" in markdown
    assert "|---|---|" in markdown
    assert "| 4 | 30% |" in markdown


def test_prose_survives_the_round_trip_byte_for_byte():
    """If the mock and the connector disagree, this is where it shows."""
    written = ["# DBMS", "", "Assessment 2 is on **4 September** in Lab 3.", "", "- [x] revise"]
    markdown = flatten(written)

    assert "# DBMS" in markdown
    assert "Assessment 2 is on **4 September** in Lab 3." in markdown
    assert "- [x] revise" in markdown


def test_code_fences_survive_with_their_language():
    markdown = flatten(["```python", "x = 1", "```"])
    assert "```python\nx = 1\n```" in markdown


# --- properties ------------------------------------------------------------

def test_a_date_property_is_matched_by_type_whatever_it_is_called():
    properties = build_properties("Page", {"Whenever": {"type": "date", "value": "2026-09-04"}})
    deadline, _, _ = normalize_properties(properties)

    assert deadline is not None
    assert deadline.date().isoformat() == "2026-09-04"


def test_select_properties_land_in_the_normalized_vocabulary():
    properties = build_properties(
        "Page",
        {
            "Status": {"type": "select", "value": "Not started"},
            "Priority": {"type": "select", "value": "High"},
        },
    )
    _, status, urgency = normalize_properties(properties)

    assert (status, urgency) == ("Not started", "High")


def test_the_title_is_a_title_typed_property():
    assert build_properties("DBMS Assessment 2", {})["Name"]["type"] == "title"


# --- the workspace ---------------------------------------------------------

def test_the_checked_in_fixtures_load_and_expand():
    pages, blocks = load_workspace()

    assert pages, "the fixture file should describe at least one page"
    for page in pages:
        assert page["object"] == "page"
        assert page["id"] in blocks
        assert page["last_edited_time"]


def test_pages_come_back_newest_edit_first():
    """The connector's crawl stops at the first page older than its cursor."""
    pages, _ = load_workspace()
    times = [page["last_edited_time"] for page in pages]
    assert times == sorted(times, reverse=True)
