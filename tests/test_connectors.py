"""Connector conversion: what survives the trip from a source into an Item."""

from __future__ import annotations

import base64
from datetime import UTC

from assistant.chunking import chunk_markdown
from assistant.connectors.gmail import extract_body, html_to_text, message_to_item
from assistant.connectors.notion import (
    blocks_to_markdown,
    flatten_properties,
    normalize_properties,
    rich_text_to_markdown,
)
from assistant.models import ChunkType, TrustLevel


def rt(text, **annotations):
    return [{"plain_text": text, "annotations": annotations}]


# ---------------------------------------------------------------------------
# Notion
# ---------------------------------------------------------------------------

def test_a_notion_table_becomes_a_markdown_table():
    """Plain-text flattening would lose which value belongs to which column."""
    block = {
        "type": "table",
        "table": {"has_column_header": True},
        "_children": [
            {"type": "table_row", "table_row": {"cells": [rt("Topic"), rt("Weight")]}},
            {"type": "table_row", "table_row": {"cells": [rt("Normalization"), rt("40%")]}},
        ],
    }
    markdown = blocks_to_markdown([block])
    assert "| Topic | Weight |" in markdown
    assert "|---|---|" in markdown
    assert "| Normalization | 40% |" in markdown


def test_a_flattened_table_survives_chunking_as_one_atomic_chunk():
    """The two halves of the decision meeting: markdown output, atomic chunking."""
    block = {
        "type": "table",
        "table": {"has_column_header": True},
        "_children": [
            {"type": "table_row", "table_row": {"cells": [rt("Topic"), rt("Weight")]}},
            {"type": "table_row", "table_row": {"cells": [rt("Normalization"), rt("40%")]}},
        ],
    }
    chunks = chunk_markdown(blocks_to_markdown([block]))
    assert len(chunks) == 1
    assert chunks[0].chunk_type is ChunkType.TABLE


def test_emphasis_is_preserved_because_it_marks_deadlines():
    assert rich_text_to_markdown(rt("due Friday", bold=True)) == "**due Friday**"


def test_links_keep_their_urls():
    span = [{"plain_text": "portal", "annotations": {}, "href": "https://x.test"}]
    assert rich_text_to_markdown(span) == "[portal](https://x.test)"


def test_checkbox_state_survives_as_a_signal():
    blocks = [
        {"type": "to_do", "to_do": {"rich_text": rt("Revise"), "checked": False}},
        {"type": "to_do", "to_do": {"rich_text": rt("Practice"), "checked": True}},
    ]
    markdown = blocks_to_markdown(blocks)
    assert "- [ ] Revise" in markdown
    assert "- [x] Practice" in markdown


def test_nested_blocks_are_indented_not_flattened_away():
    blocks = [
        {
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": rt("Parent")},
            "_children": [
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rt("Child")}}
            ],
        }
    ]
    assert "  - Child" in blocks_to_markdown(blocks)


def test_paragraphs_are_separated_but_list_items_are_not():
    """Blank lines are what the chunker splits on, so this is load-bearing."""
    blocks = [
        {"type": "paragraph", "paragraph": {"rich_text": rt("One.")}},
        {"type": "paragraph", "paragraph": {"rich_text": rt("Two.")}},
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rt("a")}},
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rt("b")}},
    ]
    markdown = blocks_to_markdown(blocks)
    assert "One.\n\nTwo." in markdown
    assert "- a\n- b" in markdown


def test_images_leave_a_visible_placeholder_rather_than_vanishing():
    """A timetable posted as a screenshot must not look like an empty page."""
    blocks = [{"type": "image", "image": {"caption": rt("timetable.png")}}]
    assert "[unprocessed image: timetable.png]" in blocks_to_markdown(blocks)


def test_depth_truncation_is_announced():
    assert "depth limit" in blocks_to_markdown([{"type": "_truncated", "_depth": 9}])


def test_the_deadline_is_matched_by_type_then_ranked_by_name():
    """A due date may be called anything; what does not vary is that it is a date."""
    deadline, _, _ = normalize_properties(
        {
            "Created": {"type": "date", "date": {"start": "2026-07-01"}},
            "Target": {"type": "date", "date": {"start": "2026-08-28"}},
        }
    )
    assert deadline.date().isoformat() == "2026-08-28"


def test_status_and_urgency_map_into_the_fixed_vocabulary():
    _, status, urgency = normalize_properties(
        {
            "Status": {"type": "status", "status": {"name": "Not started"}},
            "Priority": {"type": "select", "select": {"name": "High"}},
        }
    )
    assert status == "Not started"
    assert urgency == "High"


def test_nothing_confidently_mappable_leaves_the_fields_none():
    """Absent is the designed state, not a gap to paper over."""
    assert normalize_properties({"Notes": {"type": "rich_text", "rich_text": rt("x")}}) == (
        None,
        None,
        None,
    )


def test_raw_properties_keep_what_the_fixed_vocabulary_misses():
    """The operations agent needs Location to fill a field the trio does not cover."""
    flat = flatten_properties({"Location": {"type": "rich_text", "rich_text": rt("Lab 3")}})
    assert flat["Location"] == "Lab 3"


# ---------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------

def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def message(**overrides):
    base = {
        "id": "m1",
        "threadId": "t1",
        "internalDate": "1755859200000",
        "labelIds": ["INBOX"],
        "snippet": "snippet",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "exams@univ.edu"},
                {"name": "Subject", "value": "DBMS end-sem"},
            ],
            "body": {"data": b64("The exam is on 28 August.")},
        },
    }
    base.update(overrides)
    return base


def test_email_is_always_untrusted():
    item = message_to_item(message(), trust_level=TrustLevel.UNTRUSTED)
    assert item.trust_level is TrustLevel.UNTRUSTED


def test_the_sender_is_part_of_the_content_not_hidden_metadata():
    """Who sent it is triage-relevant; the classifier should not have to be told."""
    item = message_to_item(message(), trust_level=TrustLevel.UNTRUSTED)
    assert "exams@univ.edu" in item.content
    assert "DBMS end-sem" in item.content


def test_gmail_leaves_the_normalized_trio_empty():
    item = message_to_item(message(), trust_level=TrustLevel.UNTRUSTED)
    assert item.deadline is None and item.status is None


def test_the_receipt_time_wins_over_a_spoofable_date_header():
    item = message_to_item(
        message(payload={**message()["payload"],
                         "headers": [{"name": "Date", "value": "Tue, 1 Jan 2019 00:00:00 +0000"}]}),
        trust_level=TrustLevel.UNTRUSTED,
    )
    assert item.last_edited_at.astimezone(UTC).year == 2025


def test_priority_headers_map_to_the_urgency_hint():
    payload = message()["payload"]
    payload["headers"] = [*payload["headers"], {"name": "X-Priority", "value": "1"}]
    item = message_to_item(message(payload=payload), trust_level=TrustLevel.UNTRUSTED)
    assert item.urgency_hint == "high"


def test_plain_text_is_preferred_over_html():
    payload = {
        "mimeType": "multipart/alternative",
        "headers": [],
        "parts": [
            {"mimeType": "text/html", "body": {"data": b64("<p>markup version</p>")}},
            {"mimeType": "text/plain", "body": {"data": b64("plain version")}},
        ],
    }
    assert extract_body(payload) == "plain version"


def test_html_conversion_keeps_links_and_list_structure():
    """A "click here to confirm" with the URL stripped loses the only action in it."""
    text = html_to_text(
        "<p>Hi</p><ul><li>Hall 3</li><li>Bring ID</li></ul>"
        "<a href='https://x.test/seat'>Check seat</a>"
    )
    assert "- Hall 3" in text
    assert "- Bring ID" in text
    assert "[Check seat](https://x.test/seat)" in text


def test_scripts_and_styles_are_stripped():
    assert "alert" not in html_to_text("<script>alert(1)</script><p>real text</p>")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_fetch_tools_are_derived_from_the_registry_not_hand_listed():
    from assistant.connectors import default_registry

    names = {schema.name for schema in default_registry().tool_schemas()}
    assert names == {"fetch_notion", "fetch_gmail"}


def test_only_connectors_declaring_an_interval_are_polled():
    """Notion polls; Gmail pushes. The asymmetry is the decision."""
    from assistant.connectors import default_registry

    polling = {config.connector.source_type for config in default_registry().polling()}
    assert polling == {"notion"}
