"""Date normalization: the resolutions, the annotation rule, and what it refuses."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from assistant.dates import normalize_dates, parse_iso_local, parse_relative, to_local

# Saturday, 22 August 2026.
ANCHOR = datetime(2026, 8, 22, 9, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("tomorrow", date(2026, 8, 23)),
        ("day after tomorrow", date(2026, 8, 24)),
        ("in 3 days", date(2026, 8, 25)),
        ("2 weeks from now", date(2026, 9, 5)),
        # dateparser returns None for every one of these, which is why the
        # weekday resolver exists at all.
        ("this Monday", date(2026, 8, 24)),
        ("coming Tuesday", date(2026, 8, 25)),
        ("next Friday", date(2026, 8, 28)),
        ("last Wednesday", date(2026, 8, 19)),
        ("end of this month", date(2026, 8, 31)),
        ("start of next week", date(2026, 8, 24)),
    ],
)
def test_relative_phrases_resolve(phrase, expected):
    assert parse_relative(phrase, ANCHOR) == expected


def test_annotation_preserves_the_original_phrasing():
    """Annotate, never overwrite.

    A wrong resolution stays visible and correctable instead of being silently
    baked into the stored corpus.
    """
    result = normalize_dates("The exam is next Friday.", ANCHOR)
    assert result.content == "The exam is next Friday [2026-08-28]."


def test_unresolvable_phrases_are_left_exactly_as_written():
    text = "Revision session sometime after the holidays."
    result = normalize_dates(text, ANCHOR)
    assert result.content == text


def test_normalization_is_idempotent():
    once = normalize_dates("Exam next Friday, report in 3 days.", ANCHOR)
    twice = normalize_dates(once.content, ANCHOR)
    assert twice.content == once.content


def test_the_anchor_is_the_edit_time_not_today():
    """Backfill correctness: a page written in March means March's Friday."""
    march = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)  # a Monday
    assert parse_relative("next Friday", march) == date(2026, 3, 13)
    assert parse_relative("next Friday", ANCHOR) == date(2026, 8, 28)


def test_unrelated_numbers_are_not_read_as_dates():
    text = "Covers chapters 4-7 and section 2.1 of the notes."
    assert normalize_dates(text, ANCHOR).content == text


def test_bare_dates_land_at_local_midnight_not_utc_midnight():
    """00:00 UTC is the previous evening in IST -- a calendar entry a day early."""
    parsed = parse_iso_local("2026-08-28")
    assert parsed.hour == 0
    assert parsed.date() == date(2026, 8, 28)
    assert parsed.tzinfo is not None


def test_naive_datetimes_are_treated_as_utc_then_localized():
    localized = to_local(datetime(2026, 8, 28, 12, 0))
    assert localized.utcoffset().total_seconds() == 5.5 * 3600
    assert localized.hour == 17 and localized.minute == 30


def test_unparseable_input_returns_none_rather_than_guessing():
    assert parse_iso_local("sometime soon-ish") is None
    assert parse_iso_local("") is None
    assert parse_iso_local(None) is None
