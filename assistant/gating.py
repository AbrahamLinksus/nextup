"""The gate: the only place a proposal becomes permission to act.

Everything upstream is advisory. The classifier reports a judgment, the
operations agent reports a proposal, the injection detector reports a suspicion
-- none of them execute anything. This module is where those signals are
combined into a single yes or no, and it is intentionally small and dull, since
it is the one component whose failure has consequences outside the process.

Four rules, each traceable to a specific way auto-action goes wrong:

1. **Classification confidence.** An uncertain label should not produce a
   certain action.
2. **Per-field confidence.** Whole-proposal confidence hides the common failure:
   sure about the event, guessing at the date.
3. **Past dates.** A past-dated auto-created event is never correct. It means
   stale content or a misresolved relative phrase -- both want a human.
4. **Injection suspicion overrides everything.** A flagged item takes the
   conservative path regardless of reported confidence, because the confidence
   is precisely what a manipulative item would be trying to inflate.

Failing any rule is not a rejection. The item queues *with its partial spec
attached* -- a pre-filled draft to accept, not a blank to redo.
"""

from __future__ import annotations

from datetime import datetime

from assistant.actions import ActionDefinition
from assistant.config import get_settings
from assistant.dates import parse_iso_local, to_local
from assistant.models import (
    ActionSpec,
    ClassificationResult,
    GateDecision,
    Label,
    QueueReason,
)


def evaluate(
    classification: ClassificationResult,
    spec: ActionSpec | None,
    definition: ActionDefinition | None,
    *,
    now: datetime | None = None,
    injection_flagged: bool = False,
    injection_reason: str = "",
) -> GateDecision:
    """Decide whether this proposal may execute without a human.

    Order matters: the cheapest and most decisive checks run first, and the
    injection override runs before anything that reads a confidence value, so a
    manipulated confidence is never consulted at all.
    """
    settings = get_settings()

    if injection_flagged:
        return GateDecision(
            allowed=False,
            reason=QueueReason.INJECTION_SUSPECTED,
            detail=injection_reason or "content appeared to address the classifier",
        )

    if classification.label is not Label.ACTIONABLE_MANDATORY:
        return GateDecision(
            allowed=False,
            reason=QueueReason.OPTIONAL,
            detail=f"label is {classification.label}, which never auto-acts",
        )

    if spec is None or definition is None:
        return GateDecision(
            allowed=False,
            reason=QueueReason.NO_TOOL_AVAILABLE,
            detail="no registered action fits this item",
        )

    if classification.confidence < settings.classify_confidence_threshold:
        return GateDecision(
            allowed=False,
            reason=QueueReason.LOW_CONFIDENCE,
            detail=(
                f"classification confidence {classification.confidence:.2f} "
                f"< {settings.classify_confidence_threshold:.2f}"
            ),
        )

    if not spec.passes_confidence_gate(
        settings.field_confidence_threshold, definition.required_fields
    ):
        weakest = min(
            definition.required_fields,
            key=lambda name: spec.confidence(name),
            default="",
        )
        return GateDecision(
            allowed=False,
            reason=QueueReason.LOW_CONFIDENCE,
            detail=(
                f"field {weakest!r} confidence {spec.confidence(weakest):.2f} "
                f"< {settings.field_confidence_threshold:.2f}"
            ),
        )

    reference = to_local(now) if now else datetime.now(settings.tz)
    for name in definition.temporal_fields:
        resolved = parse_iso_local(spec.value(name))
        if resolved is None:
            return GateDecision(
                allowed=False,
                reason=QueueReason.LOW_CONFIDENCE,
                detail=f"field {name!r} did not resolve to a usable date",
            )
        if resolved < reference:
            return GateDecision(
                allowed=False,
                reason=QueueReason.PAST_DATE,
                detail=f"{name} resolves to {resolved.isoformat()}, already past",
            )

    return GateDecision(allowed=True)
