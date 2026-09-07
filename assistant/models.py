"""Canonical domain model.

Two boundaries are enforced here rather than by convention:

1. `Item` is the only shape anything downstream of a connector ever sees. No
   consumer branches on `source_type` -- that is the entire point of the
   connector abstraction.
2. `ActionSpec` is a *proposal*, never an executed action. It carries per-field
   confidence so the gate has something specific to check; execution lives in
   `assistant.actions` and only ever runs on a spec that cleared the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class Label(StrEnum):
    INFORMATIONAL = "informational"
    ACTIONABLE_OPTIONAL = "actionable-optional"
    ACTIONABLE_MANDATORY = "actionable-mandatory"


# Highest-consequence label wins when aggregating per-chunk results up to the
# item. Ordering lives next to the enum so the aggregation rule has one home.
LABEL_SEVERITY: dict[Label, int] = {
    Label.INFORMATIONAL: 0,
    Label.ACTIONABLE_OPTIONAL: 1,
    Label.ACTIONABLE_MANDATORY: 2,
}


class TrustLevel(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class QueueReason(StrEnum):
    OPTIONAL = "optional"
    LOW_CONFIDENCE = "low_confidence"
    NO_TOOL_AVAILABLE = "no_tool_available"
    PAST_DATE = "past_date"
    DATE_CONFLICT = "date_conflict"
    INJECTION_SUSPECTED = "injection_suspected"


class QueueStatus(StrEnum):
    PENDING = "pending"
    DISMISSED = "dismissed"
    PROMOTED = "promoted"


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    CODE = "code"


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """A single unit of content from any source, normalized to one shape.

    The normalized trio (`deadline`, `status`, `urgency_hint`) is populated by
    the *connector*, matching its own native fields into one fixed vocabulary,
    so both consumers -- the classifier and the operations agent -- reason from
    one consistent set of names instead of arbitrary per-source shapes.

    They are frequently `None` (Gmail has almost no structured metadata). That
    is expected, not an error: every consumer must degrade to content alone.
    `raw_properties` stays underneath as a full passthrough for anything the
    fixed vocabulary does not anticipate -- a "Location" property on an
    in-person exam, say, which the operations agent still needs to fill an
    ActionSpec field the normalized set does not cover.
    """

    source_type: str
    source_id: str
    url: str
    title: str
    content: str
    """MARKDOWN, never plain text. Flattening a Notion table by joining cell
    text loses which value belongs to which row; markdown keeps the structure
    in a form both embedding models and LLMs are trained to parse back out."""

    created_at: datetime
    last_edited_at: datetime

    deadline: datetime | None = None
    status: str | None = None
    urgency_hint: str | None = None
    raw_properties: dict[str, Any] = field(default_factory=dict)

    trust_level: TrustLevel = TrustLevel.TRUSTED
    """Set from the originating connector. Untrusted content is delimited in
    every prompt and is never allowed to talk itself past the gate."""

    @property
    def key(self) -> str:
        return f"{self.source_type}:{self.source_id}"


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit of an item.

    `chunk_key` -- not `chunk_index` -- is the identity the lifecycle and queue
    tables join on. Index is position, and position shifts whenever anything is
    inserted above it; a lifecycle row keyed on position would silently point at
    different content after the next edit, and reconciliation would update the
    wrong calendar event.
    """

    chunk_key: str
    heading_path: tuple[str, ...]
    chunk_index: int
    chunk_type: ChunkType
    content: str
    """Exactly what gets embedded: heading breadcrumb + body."""
    body: str
    content_hash: str

    @property
    def token_estimate(self) -> int:
        return len(self.content) // 4


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Deliberately narrow: the judgment call and nothing else.

    It carries no action type and no tool knowledge -- picking a tool and
    filling its arguments is the operations agent's single job, done in one
    motion. `raw_runs` is unused by v1 but present from day one so swapping
    SingleCallClassifier for a self-consistency strategy later changes no
    schema, no audit-log shape, and no caller.
    """

    label: Label
    confidence: float
    rationale: str
    raw_runs: list[dict[str, Any]] | None = None

    # Set when the result came from aggregating per-chunk classifications, so
    # extraction knows which chunk actually triggered the mandatory label.
    triggering_chunk_key: str | None = None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldValue:
    """One proposed argument plus how sure the model was about *that argument*.

    Whole-proposal confidence is not enough to gate on: an agent can be certain
    an exam needs scheduling and be guessing at the date. The gate checks the
    fields that matter individually.
    """

    value: Any
    confidence: float


@dataclass
class ActionSpec:
    """A proposed action. Nothing here has happened yet."""

    action_type: str
    fields: dict[str, FieldValue] = field(default_factory=dict)
    source_item_key: str | None = None
    source_chunk_key: str | None = None

    def value(self, name: str, default: Any = None) -> Any:
        found = self.fields.get(name)
        return default if found is None else found.value

    def confidence(self, name: str) -> float:
        found = self.fields.get(name)
        return 0.0 if found is None else found.confidence

    def passes_confidence_gate(self, threshold: float, required: tuple[str, ...]) -> bool:
        """Every field the action genuinely needs must clear the bar on its own."""
        return all(
            name in self.fields and self.fields[name].confidence >= threshold
            for name in required
        )

    def as_arguments(self) -> dict[str, Any]:
        """Flatten to the plain payload an executor consumes."""
        return {name: fv.value for name, fv in self.fields.items()}

    def to_payload(self) -> dict[str, Any]:
        """Serializable form for the queue's pre-filled draft and the audit log."""
        return {
            "action_type": self.action_type,
            "fields": {
                name: {"value": fv.value, "confidence": fv.confidence}
                for name, fv in self.fields.items()
            },
            "source_item_key": self.source_item_key,
            "source_chunk_key": self.source_chunk_key,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ActionSpec:
        return cls(
            action_type=payload["action_type"],
            fields={
                name: FieldValue(value=body["value"], confidence=body["confidence"])
                for name, body in payload.get("fields", {}).items()
            },
            source_item_key=payload.get("source_item_key"),
            source_chunk_key=payload.get("source_chunk_key"),
        )


@dataclass(frozen=True)
class GateDecision:
    """Why a proposal was or was not allowed to execute.

    The reason is recorded rather than derived later: "it was queued" is not an
    answer to "why did it do that", and the queue reason is exactly the signal
    threshold tuning reads back.
    """

    allowed: bool
    reason: QueueReason | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Queue and lifecycle
# ---------------------------------------------------------------------------

@dataclass
class QueueEntry:
    """Pre-execution state: what is awaiting a human decision.

    Kept separate from `item_lifecycle` (post-execution state: what action
    currently exists) on purpose. Promoting an entry is the bridge between them
    -- it executes, then writes a lifecycle row.
    """

    id: UUID
    item_key: str
    chunk_key: str | None
    classification_label: Label
    classification_confidence: float
    action_type: str | None
    action_spec: dict[str, Any] | None
    queue_reason: QueueReason
    status: QueueStatus
    created_at: datetime
    resolved_at: datetime | None = None
    resolution: str | None = None
    """What the user actually did. Every promote/dismiss produces a
    (confidence, outcome) pair, which is the whole threshold-tuning plan --
    no separate labelling effort."""


@dataclass
class LifecycleRow:
    """Post-execution state for one (item, chunk) pair.

    Without this, a re-edit of an already-acted-on item is unanswerable: the
    system cannot know whether to create, update, or delete.
    """

    item_key: str
    chunk_key: str
    action_type: str
    last_classification: Label
    last_extracted_date: datetime | None
    action_id: str | None
    updated_at: datetime
