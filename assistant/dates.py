"""Relative-date resolution, applied at ingestion rather than at extraction.

Two reasons this runs before storage instead of inside the extract call:

1. Content stored and embedded with "the exam is next Friday" is ambiguous
   *forever*. Six months later, retrieval surfaces a chunk nobody -- model or
   human -- can date. Resolving at ingestion keeps what we stored unambiguous.
2. It makes the operations agent's job close to trivial in the common case: it
   reads an explicit date instead of doing calendar arithmetic, which is the
   part models get quietly wrong.

Three rules the implementation is built around:

**Annotate, never overwrite.** The output is ``the exam is next Friday
[2026-08-28]``, not a blind substitution. The original phrasing survives, so a
bad resolution is visible and correctable instead of silently baked into the
corpus.

**Deterministic first, model only for what the parser refuses.** ``dateparser``
with ``RELATIVE_BASE`` handles the common cases at zero marginal cost. Only
phrasing it cannot resolve escalates to an LLM, and if that is ambiguous too the
text is left exactly as written. Guessing is never the fallback.

**The anchor is ``item.last_edited_at``, not now.** This matters most on
backfill: a page written in March saying "next Friday" means the Friday after
March, not the Friday after whenever the crawler happened to reach it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import dateparser

from assistant.config import get_settings

# Phrases worth attempting. Deliberately a closed list rather than "run the
# parser over every noun phrase": dateparser will happily read a date out of
# "chapter 4" or "section 2.1", and a wrong annotation is worse than none.
_RELATIVE_PATTERNS = [
    r"\bday after tomorrow\b",
    r"\bday before yesterday\b",
    r"\b(?:today|tomorrow|yesterday|tonight)\b",
    r"\b(?:this|next|last|coming|following|past)\s+"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day\b",
    r"\b(?:this|next|last|coming|following)\s+(?:week|month|weekend|fortnight)\b",
    r"\bin\s+(?:a|an|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:day|week|month)s?\b",
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:day|week|month)s?\s+(?:from\s+now|later|ago|back|hence)\b",
    r"\b(?:end|start|beginning)\s+of\s+(?:this|next|the)\s+(?:week|month)\b",
    r"\bby\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day\b",
]

_RELATIVE_RE = re.compile("|".join(_RELATIVE_PATTERNS), re.IGNORECASE)

# An annotation this module already added. Matching it keeps re-normalization
# idempotent instead of producing "next Friday [2026-08-28] [2026-08-28]".
_ANNOTATION_RE = re.compile(r"\s*\[\d{4}-\d{2}-\d{2}\]")


@dataclass(frozen=True)
class DateResolution:
    phrase: str
    resolved: date | None
    method: str  # "parser" | "llm" | "unresolved"

    @property
    def ok(self) -> bool:
        return self.resolved is not None


@dataclass
class NormalizedContent:
    content: str
    resolutions: list[DateResolution] = field(default_factory=list)

    @property
    def unresolved(self) -> list[str]:
        return [r.phrase for r in self.resolutions if not r.ok]

    @property
    def resolved_dates(self) -> list[date]:
        return [r.resolved for r in self.resolutions if r.resolved is not None]


LLMFallback = Callable[[str, datetime], date | None]


# ---------------------------------------------------------------------------
# Weekday and period phrases -- resolved here, not by dateparser
# ---------------------------------------------------------------------------
#
# dateparser returns None for "next Friday", "this Monday", and "end of this
# month" -- which is most of how deadlines are actually written. Rather than
# escalate the commonest phrasing in the corpus to an LLM call on every ingest,
# these are resolved arithmetically.
#
# "next <weekday>" is genuinely contested in English: some readers mean the
# upcoming one, others the one in the following week. The convention chosen here
# is stated rather than assumed --
#
#   this/coming/coming-up/by <weekday>  -> the next occurrence strictly after
#                                          the anchor (within 7 days)
#   next <weekday>                      -> that weekday in the week *after* the
#                                          anchor's week (Monday-start)
#   last/past <weekday>                 -> the most recent occurrence before it
#
# -- and the original phrasing is preserved in the text either way, so a reader
# who meant the other one can see that and correct it. That visibility is the
# reason annotations never overwrite.

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_WEEKDAY_PHRASE_RE = re.compile(
    r"^(?:(this|next|last|coming|following|past|by)\s+)?"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",
    re.IGNORECASE,
)

_PERIOD_PHRASE_RE = re.compile(
    r"^(end|start|beginning)\s+of\s+(this|next|the)\s+(week|month)$",
    re.IGNORECASE,
)


def _next_occurrence(anchor_day: date, weekday: int) -> date:
    """The next time `weekday` comes around, strictly after `anchor_day`."""
    ahead = (weekday - anchor_day.weekday()) % 7 or 7
    return anchor_day + timedelta(days=ahead)


def _resolve_weekday(qualifier: str, weekday: int, anchor_day: date) -> date:
    if qualifier in ("last", "past"):
        behind = (anchor_day.weekday() - weekday) % 7 or 7
        return anchor_day - timedelta(days=behind)
    if qualifier in ("next", "following"):
        # Monday of the anchor's week, then a full week forward, then the weekday.
        week_start = anchor_day - timedelta(days=anchor_day.weekday())
        return week_start + timedelta(days=7 + weekday)
    return _next_occurrence(anchor_day, weekday)


def _resolve_period(edge: str, which: str, unit: str, anchor_day: date) -> date:
    if unit == "week":
        week_start = anchor_day - timedelta(days=anchor_day.weekday())
        if which == "next":
            week_start += timedelta(days=7)
        return week_start + timedelta(days=6) if edge == "end" else week_start

    first = anchor_day.replace(day=1)
    if which == "next":
        first = (first + timedelta(days=32)).replace(day=1)
    if edge != "end":
        return first
    return (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)


def resolve_structured(phrase: str, anchor: datetime) -> date | None:
    """Handle the weekday and period phrasings dateparser refuses.

    Returns None for anything it does not recognise, so the caller falls through
    to dateparser rather than this quietly becoming the only parser.
    """
    anchor_day = anchor.astimezone(get_settings().tz).date()
    text = " ".join(phrase.split()).lower()

    weekday_match = _WEEKDAY_PHRASE_RE.match(text)
    if weekday_match:
        qualifier = (weekday_match.group(1) or "this").lower()
        return _resolve_weekday(qualifier, _WEEKDAYS[weekday_match.group(2)], anchor_day)

    period_match = _PERIOD_PHRASE_RE.match(text)
    if period_match:
        edge, which, unit = (group.lower() for group in period_match.groups())
        return _resolve_period(edge, "next" if which == "next" else "this", unit, anchor_day)

    return None


def parse_relative(phrase: str, anchor: datetime) -> date | None:
    """Resolve one relative phrase against an anchor, in the configured timezone.

    Weekday and period phrases are resolved arithmetically first (dateparser
    returns None for "next Friday"); everything else -- "tomorrow", "in 3 days",
    "2 weeks from now" -- goes to dateparser with the anchor as its relative base.

    ``PREFER_DATES_FROM: future`` because these are overwhelmingly deadlines. A
    bare weekday in a task list means the next one, not the one that just passed.
    """
    structured = resolve_structured(phrase, anchor)
    if structured is not None:
        return structured

    settings = get_settings()
    parsed = dateparser.parse(
        phrase,
        settings={
            "RELATIVE_BASE": anchor.astimezone(settings.tz).replace(tzinfo=None),
            "TIMEZONE": settings.timezone,
            "TO_TIMEZONE": settings.timezone,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
    return None if parsed is None else parsed.astimezone(settings.tz).date()


def normalize_dates(
    content: str,
    anchor: datetime,
    *,
    llm_fallback: LLMFallback | None = None,
) -> NormalizedContent:
    """Annotate every resolvable relative date in `content` with an ISO date.

    Unresolvable phrases are recorded and left untouched in the text. They are
    not an error: an item saying "sometime after the holidays" genuinely has no
    date, and the downstream confidence gate is what turns that into a queued
    item rather than a wrong calendar entry.
    """
    resolutions: list[DateResolution] = []
    out: list[str] = []
    cursor = 0

    for match in _RELATIVE_RE.finditer(content):
        phrase = match.group(0)

        # Already annotated by a previous pass -- leave it exactly as is.
        if _ANNOTATION_RE.match(content[match.end() : match.end() + 14]):
            continue

        resolved = parse_relative(phrase, anchor)
        method = "parser"
        if resolved is None and llm_fallback is not None:
            resolved = llm_fallback(phrase, anchor)
            method = "llm"
        if resolved is None:
            resolutions.append(DateResolution(phrase, None, "unresolved"))
            continue

        resolutions.append(DateResolution(phrase, resolved, method))
        out.append(content[cursor : match.end()])
        out.append(f" [{resolved.isoformat()}]")
        cursor = match.end()

    out.append(content[cursor:])
    return NormalizedContent("".join(out), resolutions)


def to_local(value: datetime) -> datetime:
    """Move a datetime into the configured timezone, assuming UTC if naive.

    Source APIs hand back UTC; the user means local time. A resolved date with
    no timezone is still ambiguous by the time it reaches a calendar API, so
    every boundary crossing goes through here rather than trusting the caller.
    """
    from datetime import UTC

    settings = get_settings()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(settings.tz)


def parse_iso_local(value: str | datetime | None) -> datetime | None:
    """Parse an ISO-8601 string (as proposed by the operations agent) into local time.

    A bare date means "that day", which becomes 00:00 local rather than 00:00
    UTC -- the difference is a calendar entry on the wrong day for anywhere east
    of Greenwich, which includes the only timezone this system runs in.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_local(value)

    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        settings = get_settings()
        fallback = dateparser.parse(
            text, settings={"TIMEZONE": settings.timezone, "RETURN_AS_TIMEZONE_AWARE": True}
        )
        if fallback is None:
            return None
        parsed = fallback

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=get_settings().tz)
    return to_local(parsed)
