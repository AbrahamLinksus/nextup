"""The query side: the similarity floor, and what an empty result says.

Both rules exist for the same reason as the classifier's boundary bias -- the
failure mode is a confident wrong answer, not an error. Vector search always
returns a nearest neighbour, so "nothing matched" has to be manufactured, and it
has to reach the model as a sentence rather than as silence.
"""

from __future__ import annotations

import pytest

from assistant import retrieval
from assistant.config import get_settings
from assistant.retrieval import SearchHit, render_hits, search_content
from tests.conftest import FakeEmbedder


def hit(similarity=0.71, *, title="DBMS syllabus", url="https://notion.test/dbms") -> SearchHit:
    return SearchHit(
        chunk_key="notion:page-1#0",
        content="Assessment 2 covers chapters 4-7.",
        similarity=similarity,
        title=title,
        url=url,
        source_type="notion",
        source_id="page-1",
    )


class RecordingRepo:
    """Stands in for the SQL, which has its own tests against a real database."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls: list[dict] = []

    async def search_chunks(self, _conn, vector, *, top_k, min_similarity):
        self.calls.append({"vector": vector, "top_k": top_k, "min_similarity": min_similarity})
        return self.rows


@pytest.fixture
def repo_stub(monkeypatch):
    stub = RecordingRepo()
    monkeypatch.setattr(retrieval.repo, "search_chunks", stub.search_chunks)
    return stub


async def test_the_query_is_embedded_with_the_query_prefix_not_the_document_one(repo_stub):
    """A document-prefixed query compares measurably worse against the same corpus."""
    embedder = FakeEmbedder()
    await search_content(None, embedder, "what does the syllabus cover")

    assert embedder.calls == [("query", 1)]


async def test_the_configured_floor_applies_when_the_caller_names_none(repo_stub):
    await search_content(None, FakeEmbedder(), "anything")

    settings = get_settings()
    assert repo_stub.calls[0]["min_similarity"] == settings.retrieval_min_similarity
    assert repo_stub.calls[0]["top_k"] == settings.retrieval_top_k


async def test_a_caller_may_lower_the_floor_to_zero_without_it_reverting_to_the_default(repo_stub):
    """`or`-style defaulting would silently ignore 0.0, which is a legitimate ask."""
    await search_content(None, FakeEmbedder(), "anything", min_similarity=0.0)
    assert repo_stub.calls[0]["min_similarity"] == 0.0


async def test_rows_become_hits_that_carry_their_source(repo_stub):
    repo_stub.rows = [
        {
            "chunk_key": "notion:page-1#0",
            "content": "Assessment 2 covers chapters 4-7.",
            "similarity": 0.71,
            "title": "DBMS syllabus",
            "url": "https://notion.test/dbms",
            "source_type": "notion",
            "source_id": "page-1",
        }
    ]
    hits = await search_content(None, FakeEmbedder(), "syllabus")

    assert hits[0].title == "DBMS syllabus"
    assert hits[0].source_type == "notion"


def test_an_empty_result_is_a_sentence_not_an_empty_string():
    """A tool that returns nothing invites the model to fill the silence from general knowledge."""
    rendered = render_hits([])

    assert rendered.strip()
    assert "general knowledge" in rendered


def test_every_rendered_hit_names_where_it_came_from():
    rendered = render_hits([hit()])

    assert "DBMS syllabus" in rendered
    assert "https://notion.test/dbms" in rendered


def test_a_hit_with_no_url_still_cites_its_title():
    """Manually remembered content has no link, and unattributed text is worth less."""
    assert hit(url="").cite() == "[DBMS syllabus]"


def test_multiple_hits_are_separated_so_they_do_not_read_as_one_passage():
    rendered = render_hits([hit(), hit(0.65, title="OS notes", url="")])

    assert "---" in rendered
    assert "OS notes" in rendered and "DBMS syllabus" in rendered
