"""The background pipeline: deterministic, not agentic.

Every fetched item -- polled, pushed, or requested on demand -- runs this exact
sequence. The trigger type never changes what happens downstream:

    ingest guard (CAS)
      -> date normalization
      -> chunk
      -> embed  ||  classify (per chunk)          [in parallel]
      -> if actionable: injection check -> operations agent proposes
      -> GATE
      -> execute (and record lifecycle)  or  queue (with the draft attached)
      -> reconcile anything the edit invalidated

There are exactly two model calls in the actionable path -- the classifier's
structured label and the operations agent's single proposal -- and no free-
standing tool execution anywhere. That is deliberate. The confidence gating, the
boundary bias, and the guardrail layer all exist because an automatic action has
no human in front of it; an agent that chose freely and then acted immediately
would make all three ornamental.

The agentic half of the system lives in `assistant.agent`, where the user is
present and a direct instruction supplies the confidence that gating otherwise
has to infer.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from psycopg import AsyncConnection

from assistant import gating, metrics
from assistant import repository as repo
from assistant.actions import ExecutorPool, get_definition
from assistant.chunking import chunk_markdown
from assistant.classify import Classifier, classify_item
from assistant.config import get_settings
from assistant.dates import normalize_dates, parse_iso_local
from assistant.embeddings import Embedder
from assistant.guardrails import detect_injection
from assistant.llm import LLMProvider
from assistant.models import (
    ActionSpec,
    ClassificationResult,
    Item,
    Label,
    LifecycleRow,
    QueueReason,
)
from assistant.operations import propose_action

log = structlog.get_logger(__name__)


@dataclass
class PipelineOutcome:
    """What happened to one item. Returned so callers can report, not decide."""

    item_key: str
    skipped: bool = False
    skip_reason: str = ""
    label: Label | None = None
    confidence: float = 0.0
    chunks_embedded: int = 0
    actions_taken: list[str] = field(default_factory=list)
    queued: list[str] = field(default_factory=list)
    reconciled: list[str] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        return bool(self.actions_taken)


class Pipeline:
    def __init__(
        self,
        conn: AsyncConnection,
        *,
        classifier: Classifier,
        provider: LLMProvider,
        embedder: Embedder,
        executors: ExecutorPool | None = None,
    ) -> None:
        self.conn = conn
        self.classifier = classifier
        self.provider = provider
        self.embedder = embedder
        self.executors = executors or ExecutorPool()

    # -- entry point -----------------------------------------------------

    async def process(self, item: Item, *, now: datetime | None = None) -> PipelineOutcome:
        """Run one item through the whole sequence, and record what it cost.

        The measurement lives here rather than inside `_process` so it covers
        every path out, including the two cheap ones -- a duplicate that costs a
        single statement is exactly the case worth being able to prove is cheap.

        The row is written on the caller's connection, so it commits with the
        item it describes. Metrics for work that got rolled back would be a
        record of processing that, as far as everything else is concerned, never
        happened.
        """
        with metrics.collect_item() as measured, metrics.timed() as elapsed:
            outcome = await self._process(item, now=now)

        duration_ms = elapsed[0]
        metrics.record_item_processed(
            source_type=item.source_type,
            outcome=_outcome_kind(outcome),
            milliseconds=duration_ms,
        )
        await repo.record_pipeline_metrics(
            self.conn,
            item_key=item.key,
            source_type=item.source_type,
            outcome=_outcome_kind(outcome),
            label=str(outcome.label) if outcome.label else None,
            duration_ms=int(duration_ms),
            detail=measured.as_row(),
        )
        return outcome

    async def _process(self, item: Item, *, now: datetime | None = None) -> PipelineOutcome:
        """The sequence itself.

        The caller's transaction wraps this, so "the item was recorded as
        ingested" and "its chunks, queue entries, and audit rows exist" commit
        together. A partial commit would leave an item the CAS guard refuses to
        reprocess with nothing to show for it.
        """
        outcome = PipelineOutcome(item_key=item.key)
        settings = get_settings()
        reference = now or datetime.now(settings.tz)

        # 1. Date normalization, before storage -- so what we embed is
        #    unambiguous forever, not just at extraction time.
        with metrics.stage("normalize_dates"):
            normalized = normalize_dates(item.content, item.last_edited_at)
        item.content = normalized.content
        if normalized.unresolved:
            log.info(
                "pipeline.dates_unresolved", item=item.key, phrases=normalized.unresolved
            )

        # 2. The ingest guard. Runs first so a duplicate costs one statement,
        #    not an embedding pass and two model calls.
        with metrics.stage("ingest_guard"):
            ingest = await repo.upsert_item(self.conn, item)
        if not ingest.accepted:
            outcome.skipped = True
            outcome.skip_reason = "not newer than what is stored (duplicate or late event)"
            return outcome

        with metrics.stage("chunk"):
            chunks = chunk_markdown(item.content)

        if not ingest.content_changed and not ingest.is_new:
            # A touch with no content change: metadata may have moved, but
            # nothing worth re-judging did. Chunks are still synced (cheap, and
            # it keeps positions right); classification is not re-run.
            outcome.chunks_embedded = await repo.sync_chunks(
                self.conn, ingest.item_id, chunks, self._embed_documents
            )
            outcome.skipped = True
            outcome.skip_reason = "content unchanged since last processing"
            return outcome

        # 3. Embedding runs unconditionally and in parallel with classification.
        #    An actionable item still needs its own content searchable later --
        #    an exam notice's syllabus scope is worth retrieving whether or not
        #    a calendar event was created from it.
        # Safe to overlap on one connection because classification performs no
        # database work at all -- it is pure model I/O. The embed leg therefore
        # has the connection to itself for the whole window.
        # `create_task` copies the current context, so the embed leg reports its
        # timings into this item's collector rather than into nobody's.
        embed_task = asyncio.create_task(
            repo.sync_chunks(self.conn, ingest.item_id, chunks, self._embed_documents)
        )
        try:
            with metrics.stage("classify"):
                classification, per_chunk = await classify_item(item, chunks, self.classifier)
        except BaseException:
            # The embed leg is still mid-statement on this same connection. If it
            # is left running while the caller rolls back, its INSERT lands in the
            # *next* item's transaction, referencing an item row that no longer
            # exists -- and one failed item then poisons the rest of the poll,
            # which is precisely what per-item transactions exist to prevent.
            # Settling it here (its own error is already lost with the rollback)
            # leaves the connection quiescent for whoever cleans up.
            with suppress(Exception):
                await embed_task
            raise
        with metrics.stage("embed_wait"):
            outcome.chunks_embedded = await embed_task

        outcome.label = classification.label
        outcome.confidence = classification.confidence
        await repo.record_audit(
            self.conn,
            event_type="classification",
            item_key=item.key,
            chunk_key=classification.triggering_chunk_key,
            payload={
                "label": str(classification.label),
                "confidence": classification.confidence,
                "rationale": classification.rationale,
                "raw_runs": classification.raw_runs,
                "per_chunk": {
                    key: {"label": str(r.label), "confidence": r.confidence}
                    for key, r in per_chunk.items()
                },
            },
        )

        # 4. Actionable chunks each become their own proposal. A page with three
        #    deadlines produces three actions, each with its own lifecycle row --
        #    not one page collapsed to a single date.
        triggering = self._triggering_chunks(classification, per_chunk, chunks)
        # One obligation is routinely described by several chunks -- a heading,
        # a syllabus table, and a checklist can all name the same exam. Each is
        # genuinely mandatory, so each proposes; `identities` is what stops that
        # from becoming three copies of one calendar entry.
        acted: list[tuple[_Signature, str]] = []
        suppressed: set[str | None] = set()
        with metrics.stage("triage"):
            for chunk_key, chunk_result, chunk_content in triggering:
                owns = await self._handle_actionable(
                    item, chunk_key, chunk_result, chunk_content, outcome, reference, acted
                )
                if not owns:
                    suppressed.add(chunk_key)

        # 5. Anything previously acted on that this edit invalidated -- including
        #    a chunk that used to own an action and is now a duplicate of one.
        outcome.reconciled = await self._reconcile_orphans(
            item, {key for key, _, _ in triggering} - suppressed
        )
        return outcome

    # -- steps -----------------------------------------------------------

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embedder.embed(texts, kind="document")

    def _triggering_chunks(
        self,
        classification: ClassificationResult,
        per_chunk: dict[str, ClassificationResult],
        chunks: list,
    ) -> list[tuple[str | None, ClassificationResult, str | None]]:
        """Which chunks deserve an action proposal.

        Only mandatory chunks get one. Optional and informational chunks still
        reach the user -- optional ones queue, informational ones are simply
        indexed -- but neither is worth an operations-agent call.
        """
        if classification.label is Label.INFORMATIONAL:
            return []

        if not per_chunk:
            return [(None, classification, None)]

        by_key = {chunk.chunk_key: chunk for chunk in chunks}
        actionable = [
            (key, result, by_key[key].content)
            for key, result in per_chunk.items()
            if result.label is not Label.INFORMATIONAL and key in by_key
        ]
        return actionable or [(None, classification, None)]

    async def _handle_actionable(
        self,
        item: Item,
        chunk_key: str | None,
        classification: ClassificationResult,
        content: str | None,
        outcome: PipelineOutcome,
        reference: datetime,
        acted: list[tuple[_Signature, str]],
    ) -> bool:
        """Triage one actionable chunk. False means another chunk already owns
        this exact action, so this one must not hold a lifecycle row of its own."""
        # The injection check gates untrusted content only, and it runs before
        # anything reads a confidence value -- a manipulative item must not be
        # able to talk its way past the threshold it is trying to inflate.
        verdict = await detect_injection(item, self.provider, content=content)
        if verdict.flagged:
            await repo.record_audit(
                self.conn,
                event_type="injection_flagged",
                item_key=item.key,
                chunk_key=chunk_key,
                payload={"reason": verdict.reason, "method": verdict.method},
            )

        spec: ActionSpec | None = None
        if classification.label is Label.ACTIONABLE_MANDATORY:
            spec = await propose_action(
                item, self.provider, content=content, chunk_key=chunk_key
            )
            if spec is not None:
                await repo.record_audit(
                    self.conn,
                    event_type="extraction",
                    item_key=item.key,
                    chunk_key=chunk_key,
                    payload=spec.to_payload(),
                )

        definition = get_definition(spec.action_type) if spec else None

        signature = _signature_of(spec, definition)
        covered_by = next(
            (chunk for other, chunk in acted if signature and signature.same_obligation_as(other)),
            None,
        )
        if covered_by is not None:
            # Not a failure and not a near miss -- the obligation is already
            # handled, by a chunk named in the audit trail. Recorded rather than
            # dropped, so "why is there no event for this section" has an answer.
            await repo.record_audit(
                self.conn,
                event_type="duplicate_suppressed",
                item_key=item.key,
                chunk_key=chunk_key,
                payload={
                    "action_type": spec.action_type,
                    "already_covered_by": covered_by,
                    "spec": spec.to_payload(),
                },
            )
            log.info(
                "pipeline.duplicate_suppressed",
                item=item.key,
                chunk=chunk_key,
                covered_by=covered_by,
            )
            return False

        decision = gating.evaluate(
            classification,
            spec,
            definition,
            now=reference,
            injection_flagged=verdict.flagged,
            injection_reason=verdict.reason,
        )
        # Counted here rather than inside `gating.evaluate`, which is a pure
        # function and worth keeping that way. This is its only caller, so
        # nothing is missed by recording one layer out.
        metrics.record_gate_decision(
            allowed=decision.allowed,
            reason=str(decision.reason) if decision.reason else "allowed",
        )

        if signature is not None:
            acted.append((signature, chunk_key or "(item)"))

        if not decision.allowed:
            await self._queue(
                item, chunk_key, classification, spec, decision.reason, decision.detail
            )
            outcome.queued.append(f"{chunk_key or item.key}: {decision.reason}")
            return True

        await self._execute(item, chunk_key, classification, spec, definition, outcome)
        return True

    async def _execute(
        self,
        item: Item,
        chunk_key: str | None,
        classification: ClassificationResult,
        spec: ActionSpec,
        definition,
        outcome: PipelineOutcome,
    ) -> None:
        """The only place in the background path where anything real happens."""
        executor = await self.executors.get(spec.action_type)
        key = chunk_key or "(item)"
        existing = await repo.get_lifecycle(self.conn, item.key, key)
        resolved_date = parse_iso_local(spec.value(definition.temporal_fields[0])) if (
            definition.temporal_fields
        ) else None

        if existing is None or existing.action_id is None:
            result = await executor.create(spec)
            operation = "create"
        else:
            # Update the event we already made, never create a second one. This
            # is the entire reason the lifecycle table exists.
            result = await executor.update(existing.action_id, spec)
            operation = "update"

        await repo.upsert_lifecycle(
            self.conn,
            LifecycleRow(
                item_key=item.key,
                chunk_key=key,
                action_type=spec.action_type,
                last_classification=classification.label,
                last_extracted_date=resolved_date,
                action_id=result.action_id,
                updated_at=datetime.now(get_settings().tz),
            ),
        )
        await repo.record_outcome(
            self.conn,
            item_key=item.key,
            chunk_key=chunk_key,
            action_type=spec.action_type,
            operation=operation,
            action_id=result.action_id,
            summary=result.summary,
            payload=result.detail,
        )
        await repo.record_audit(
            self.conn,
            event_type="action_taken",
            item_key=item.key,
            chunk_key=chunk_key,
            payload={
                "operation": operation,
                "action_id": result.action_id,
                "summary": result.summary,
                "dry_run": result.dry_run,
                "spec": spec.to_payload(),
            },
        )
        # A pending review of something the system has now decided is stale.
        await repo.clear_pending_for(self.conn, item.key, chunk_key)
        outcome.actions_taken.append(result.summary)

    async def _queue(
        self,
        item: Item,
        chunk_key: str | None,
        classification: ClassificationResult,
        spec: ActionSpec | None,
        reason: QueueReason,
        detail: str,
    ) -> None:
        await repo.upsert_queue_entry(
            self.conn,
            item_key=item.key,
            chunk_key=chunk_key,
            classification=classification,
            spec=spec,
            reason=reason,
        )
        await repo.record_audit(
            self.conn,
            event_type="queued",
            item_key=item.key,
            chunk_key=chunk_key,
            payload={
                "reason": str(reason),
                "detail": detail,
                "label": str(classification.label),
                "confidence": classification.confidence,
                "spec": spec.to_payload() if spec else None,
            },
        )

    async def _reconcile_orphans(self, item: Item, still_actionable: set[str | None]) -> list[str]:
        """Delete actions whose reason for existing is gone.

        The third reconciliation case, and the one that needs the lifecycle
        table most: an item that *had* an event and is no longer mandatory. Its
        deadline was removed, the task was marked done, or the paragraph
        carrying it was deleted. Without this, the calendar quietly accumulates
        events for obligations that no longer exist.
        """
        live = {key or "(item)" for key in still_actionable}
        removed: list[str] = []

        for row in await repo.list_lifecycle_for_item(self.conn, item.key):
            if row.chunk_key in live:
                continue
            if row.action_id:
                executor = await self.executors.get(row.action_type)
                await executor.delete(row.action_id)
                await repo.record_outcome(
                    self.conn,
                    item_key=item.key,
                    chunk_key=row.chunk_key,
                    action_type=row.action_type,
                    operation="delete",
                    action_id=row.action_id,
                    summary=f"removed {row.action_type} for {row.chunk_key} (no longer actionable)",
                    payload={},
                )
            await repo.delete_lifecycle(self.conn, item.key, row.chunk_key)
            await repo.record_audit(
                self.conn,
                event_type="reconciled",
                item_key=item.key,
                chunk_key=row.chunk_key,
                payload={"action_id": row.action_id, "outcome": "deleted"},
            )
            removed.append(row.chunk_key)

        return removed


def _outcome_kind(outcome: PipelineOutcome) -> str:
    """One word for what happened, for grouping metrics by.

    Ordered by consequence: acting is the outcome with effects outside the
    process, so an item that both acted and queued (a page with two deadlines,
    one certain and one not) is counted as having acted.
    """
    if outcome.actions_taken:
        return "acted"
    if outcome.queued:
        return "queued"
    if outcome.skipped:
        return "skipped"
    return "indexed"


@dataclass(frozen=True)
class _Signature:
    """What one proposal claims, reduced to the parts that identify an obligation."""

    action_type: str
    instants: tuple[datetime, ...]
    texts: tuple[str, ...]

    def same_obligation_as(self, other: _Signature) -> bool:
        """Two proposals describing one thing.

        Same action type and the same resolved instants does most of the work --
        an obligation is identified far more reliably by *when* it falls due than
        by what the model chose to call it. Text fields then only have to be
        compatible rather than identical, because the same deadline read from a
        heading and from a checklist comes back as "Lab submission 3" and "Lab
        submission 3 deadline", which are the same thing named twice. Containment
        is the shape that failure takes; two genuinely different obligations that
        happen to fall at the same instant have names where neither contains the
        other.
        """
        if self.action_type != other.action_type or self.instants != other.instants:
            return False
        return all(
            mine == theirs or mine in theirs or theirs in mine
            for mine, theirs in zip(self.texts, other.texts, strict=True)
        )


def _signature_of(spec: ActionSpec | None, definition) -> _Signature | None:
    """Reduce a proposal to its identity, or None when it has none.

    The action's own `required_fields` are the material: they are, by definition,
    what the action cannot be described without. Deriving the signature from the
    registry rather than hardcoding "title and start" means a future action type
    gets de-duplication without touching this.

    Returns None when any required field is missing -- an incomplete proposal has
    nothing to match on, and collapsing two different incomplete ones would be
    worse than letting the gate queue them both.
    """
    if spec is None or definition is None or not definition.required_fields:
        return None

    instants: list[datetime] = []
    texts: list[str] = []
    for name in definition.required_fields:
        value = spec.value(name)
        if value in (None, ""):
            return None
        if name in definition.temporal_fields:
            # '2026-09-19T23:59' and '2026-09-19T23:59:00+05:30' are one moment
            # written twice.
            resolved = parse_iso_local(value)
            if resolved is None:
                return None
            instants.append(resolved)
        else:
            texts.append(" ".join(str(value).split()).casefold())

    return _Signature(spec.action_type, tuple(instants), tuple(texts))
