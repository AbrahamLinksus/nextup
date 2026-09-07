# Jake's assistant — Design Log

Living record of decisions, edge cases, and open tradeoffs. Updated as the design progresses — treat this as the source of truth over the chat history.

---

## Decisions made

**Scope & concept**
- New project, distinct from Study Copilot, QueuedUp, RAG knowledge base, and Crosscheck
- Core idea: continuously ingests personal sources, classifies each item as **informational** / **actionable-mandatory** / **actionable-optional**, auto-acts on high-confidence mandatory items, queues the rest
- Optional/ambiguous items go to a **pull-based queue** (checked when the user wants, not pushed)
- v1 source: **Notion only**; Gmail is the second source, designed for from the start rather than bolted on later

**Fetch / connector layer**
- Every source implements a common `SourceConnector` interface: `list_items(since)`, `fetch_item(id)`, `register_watch()`, `handle_change_event(payload)`, `as_tool_schema()`
- Nothing downstream of `fetch_item`/`list_items` branches on `source_type` — that's the abstraction boundary
- Canonical `Item` model: fixed core fields (`title`, `content`, `created_at`, `last_edited_at`, `url`, `source_id`) + a small connector-normalized vocabulary (`deadline`, `status`, `urgency_hint`) + `raw_properties: dict` passthrough for everything the normalized vocabulary doesn't cover (see Properties normalization decision)

**Trigger layer**
- Trigger strategy is **per-connector**, not global:
  - Notion: poll every 12h + on-demand
  - Gmail: push (`watch()` + Pub/Sub) + on-demand
- On-demand fetch exposed as **one LLM tool per connector** (`fetch_notion`, `fetch_gmail`, ...), not one generic tool with a source parameter — lets the LLM naturally resolve "check my mail" → one tool call, "update all" → all tools called in the same turn, with no fan-out logic to hardcode
- `list_items(since)` is reused for backfill, scheduled poll, and on-demand — one method, three callers

**Classification / triage**
- Classification and extraction are **two separate LLM calls**, not one combined call — costs latency, buys independent eval/debugging of each step
- Classifier input: block/message content **plus** the normalized `deadline`/`status`/`urgency_hint` fields (with `raw_properties` available as fallback context); must degrade gracefully when a source leaves these null
- Confidence threshold gates auto-action; **tuning approach**: start conservative, use the user's own accept/reject behavior on queued items as an implicit feedback signal over time, rather than hand-labeling an eval set upfront
- **Boundary-case bias**: when a classification is genuinely ambiguous, resolve toward the lower-consequence label (queue/informational over mandatory) — mild bias, applies only at the boundary, not to clearly-signaled mandatory items. Rationale: a missed queue item costs seconds later, a wrongly auto-scheduled item costs a false calendar entry with no review.
- **Confidence strategy is abstracted behind a `Classifier` interface** (`classify(item) -> ClassificationResult`) so v1 can ship with single-call self-reported confidence and later swap to self-consistency (N calls, agreement rate as confidence) without changing the triage layer. `ClassificationResult` carries a `raw_runs` field from day one (unused by v1) so the audit log schema doesn't change when the strategy swaps.
- Output must be structured (tool-call/JSON schema), never freeform text parsed after the fact

**Item lifecycle / state tracking**
- A persisted `(item_id, chunk_id) → last_classification, last_extracted_date, calendar_event_id` table is required for reconciliation on re-edit — without it, the system can't know what to update or delete when an item changes after it's already been acted on. Keyed by chunk, not just item, since one item can contain multiple distinct mandatory deadlines, each its own action. Reconciliation rule: no prior event + now mandatory → create; had an event + date changed → update the same event (never duplicate); had an event + no longer mandatory → delete via stored `calendar_event_id`.
- Conflicting/multiple dates within one item: solved via prompt instruction (prefer superseding language — "moved to," "rescheduled to" — over earlier mentions) plus the existing confidence-gating machinery; no new infrastructure needed. Genuinely ambiguous cases naturally drop `date.confidence` and route to queue.

**Guardrail / injection layer**
- `SourceConnector` gets a `trust_level: "trusted" | "untrusted"` field (Notion: trusted, Gmail: untrusted) — content from untrusted sources is wrapped in explicit delimiters in every prompt (classify/extract) with an instruction that delimited content is data to classify, never instructions to follow.
- Untrusted-source items additionally run a cheap injection-detection pass before classify; if flagged, the conservative path is forced regardless of reported confidence — a manipulative item should never be able to talk itself into auto-scheduling.

**Minimum content threshold**
- Items with stripped content under ~30 characters and no meaningful `properties` signal skip classification entirely, default to informational/pending. Re-processed normally on the next real edit — not a dead end.

**Oversized items**
- Reuses the structure-aware chunking already planned for storage, rather than a separate task/content-splitting system (which would just duplicate classify's own job with a cruder heuristic). Classify runs per-chunk (with a cheap keyword/property pre-filter to skip obviously irrelevant chunks), results aggregate at the item level by highest-consequence label (mandatory > optional > informational), and extraction runs against the specific chunk that triggered mandatory.
- **Multiple distinct mandatory deadlines within one item → multiple separate actions**, one per triggering chunk, each with its own lifecycle-state entry (keyed by item_id + chunk_id, not just item_id — see item lifecycle table).

**Timezone**
- Fixed to **Asia/Kolkata (IST)** for v1, not derived per-item. All normalized dates/times resolve into IST before being handed to the extract call or the calendar API.

**Action execution**
- `ActionExecutor` interface, symmetric to `SourceConnector`: `create(spec) -> action_id`, `update(action_id, spec)`, `delete(action_id)`. Reconciliation logic only ever calls this interface, never a destination's API/MCP tool directly.
- Calls go through **MCP servers** for each destination (Calendar MCP now, Gmail/Slack MCP later) rather than bespoke API clients — reuses existing standardized integrations/auth instead of writing one-off SDK wrappers per destination. `CalendarMCPActionExecutor` is the v1 implementation.
- MCP solves *how to call each destination*; it does not replace the `ActionExecutor` abstraction itself — something still has to decide which tool to call and map internal state into that tool's schema, which is what the interface is for.

**Storage**
- Postgres + pgvector (reusing the pattern from the RAG knowledge base project)
- Chunking is source-structure-aware (Notion blocks as boundaries), not fixed-size
- **Embedding is unconditional — runs on every item regardless of classification**, in parallel with the classify/action pipeline, not as its alternative. An actionable item still needs its original content searchable later (e.g. an exam notice's syllabus content), not just the fact that an action was taken. Optionally, a lightweight record of the action outcome ("created calendar event: X, date Y") could also be embedded — deferred as a nice-to-have, since the lifecycle/queue tables already hold this structurally and the query router (undesigned) should likely route "what's scheduled" questions there directly rather than relying on search to rediscover it.
- **If action-outcome records are embedded, they go in a separate namespace/collection from source content, and are immutable (append-only), never mutated in place.** Source content ("assessment covers chapters 4-7") and system-generated records ("calendar updated: assessment scheduled") sharing one vector space risks meta-text surfacing as if it were substantive content. A record that changes meaning over time (scheduled → completed → scored) is not a single mutable embedding — it's either a pure audit log (immutable, append-only) or, if genuinely new information arrives later (e.g. a score), that's a new `Item` from its own source, re-entering the normal ingestion pipeline rather than a bespoke in-place update mechanism.

**Orchestration — two distinct execution models, not one**
- **Background pipeline (deterministic, not agentic)**: runs on every poll/push/on-demand fetch result, always the same fixed sequence — normalize → (embed in parallel with) classify → if actionable, operations agent proposes a tool call → gate → execute/queue. The only model calls are the narrow ones already designed (classify's structured label/confidence, the operations agent's single tool-calling proposal); there is no free-standing tool execution outside the gate. This is deliberate — the confidence gating, boundary bias, and guardrail layer all exist because auto-actions need to be bounded and predictable, not the product of an agent choosing freely what to do and immediately doing it.
- **Conversational agent (genuinely agentic)**: the user-facing chat layer, with real tool-calling freedom — fetch tools (`fetch_notion`, `fetch_gmail`) and query tools (`list_pending_queue`, `search_content`, ...). This is the layer every "LLM picks which tool to call" decision in this design actually belongs to.
- **The conversational agent can also take actions directly** ("schedule this for me tomorrow") — reuses the same `ActionExecutor`/`ACTION_REGISTRY` as the background pipeline, but *without* confidence gating, since a direct human instruction is already the confidence signal the gating exists to substitute for when the input is only an inferred classification.
- **Items fetched on-demand still go through the full deterministic pipeline**, same as poll/push results — the trigger type never changes downstream processing. Only explicit direct action requests typed by the user bypass the pipeline; fetched content never does.
- No separate router component — reuses the same tool-calling pattern as the fetch/trigger layer. Structured lookups and semantic search are registered as additional LLM-callable tools alongside `fetch_notion`/`fetch_gmail`; the orchestrating LLM picks which to call based on the query, same as "check my mail" → `fetch_gmail`.
- Structured tools are **fixed, named query functions**, not free-form NL-to-SQL — `list_pending_queue()`, `list_upcoming_actions(date_range)`, `list_dismissed(since)`. Generating arbitrary SQL against the DB is a real risk for marginal benefit; a small set of predefined functions covers the realistic query surface.
- Semantic search is one more tool: `search_content(query) -> list[Chunk]`, embeds the query and does similarity search against `chunks`.
- **`search_content` mechanics**: embed query with the same model used for chunks (must match — mismatched embedding spaces don't compare meaningfully), cosine similarity top-k (k≈10 to start), **and a minimum similarity threshold** — without one, an unrelated query still returns *something* (the closest available chunks even if none are truly relevant), risking a confident answer built on noise. Below threshold → return empty, not weak matches.
- **Grounding rule (query-side counterpart to classify's boundary bias)**: the agent answers only from what its tools returned, never fills gaps from general knowledge. If `search_content` comes back empty, the correct answer is "nothing indexed about that," not a plausible-sounding fabrication. Stated explicitly, not assumed — models drift toward smooth answers over honest ones unless told otherwise.
- **Attribution**: answers drawing on retrieved chunks reference the source item's title/url, not just the content as free-floating fact — same transparency instinct as the audit log.
- Deferred, not solved: a single returned chunk can lack surrounding context (e.g. a paragraph without its heading). V1 default is including the parent item's title alongside every returned chunk; full adjacent-chunk stitching is a real refinement but not worth building before real usage shows it's actually needed.
- A query needing both ("what's due this week and what's it about") is handled by the LLM calling multiple tools in one turn — same fan-out behavior already designed for "update all" on the fetch side, not new logic.

**Properties normalization**
- Normalized at the connector, not left raw for the triage layer to interpret — each connector maps its own native fields (a Notion date-typed property, matched by type rather than by label since names vary — "Due Date," "Deadline," "Target") into a small fixed vocabulary: `deadline`, `status`, `urgency_hint`. This gives both consumers (classify and the operations agent) one consistent instruction to reason from, rather than each having to handle arbitrary per-source shapes.
- `raw_properties: dict` stays as a full passthrough underneath the normalized fields — the fixed vocabulary won't anticipate everything (e.g. a "Location" property for an in-person exam), so the operations agent still needs raw access when filling `ActionSpec` fields the normalized set doesn't cover.
- Connectors leave normalized fields `None` when nothing confidently maps — same graceful-degradation behavior already designed elsewhere, not a new failure mode.

**Direct memory input & embedding model**
- Embedding model: **`nomic-embed-text`, run locally** — same choice as the RAG knowledge-base project, for consistency and because 8192-token context suits embedding fuller documents, not just short chunks. Resolves the `vector(?)` placeholder in the storage schema to `vector(768)`.
- Directly-added content (user provides something other than a fetched item) gets a new `source_type: "manual"` — generated id, `last_edited_at` = time of insertion, same `items` table, no new table needed.
- Non-text direct input (image, PDF, document) is converted to text/markdown first (OCR/vision-model captioning for images, text extraction for documents) and then flows through the exact same `Item → chunk → embed` pipeline as everything else — one schema, one embedding space, no branching. This is the "conform input to schema" principle applied to the input layer itself.
- **Assumption stated, not silently made**: directly-added content skips classify/triage by default and goes straight to embedding — it's declared memory, not a fetched item to triage for action. Wanting something typed directly to also become an action goes through the conversational agent's direct-action tool path instead.
- Auth: not implemented for v1, given single-user scope. A scoping decision, not a permanent one — revisit if this ever serves more than one person.
- Query transformation: applied pragmatically, not built exhaustively upfront. Temporal resolution ("10 days back" → an actual date range, already designed) is the first concrete instance; further transformation (expansion, rewriting) is a lever to reach for once retrieval quality shows a need for it, not before.

**Idempotency**
- No separate dedup table — the `items` table's `last_edited_at` column serves as the guard directly, via a compare-and-swap update: `UPDATE items SET last_edited_at = :incoming WHERE source_id = :id AND (last_edited_at IS NULL OR last_edited_at < :incoming) RETURNING id`. Zero rows returned → skip (either a redelivered duplicate, or a late/out-of-order older event — both correctly resolve to "do nothing").
- This also solves out-of-order webhook delivery for free, not just redelivery — since only strictly newer events are ever accepted, a stale event arriving late can't revert already-processed content.
- Runs as the first gate, before embedding/classify/operations-agent — skips the entire expensive pipeline for a known duplicate rather than wasting LLM calls.
- Item-level granularity is sufficient, no chunk-level dedup needed — chunks only change when the item's content changes, which is exactly what `last_edited_at` tracks.

**Environment**
- Real repo will live in Claude Code (terminal, Arch) once the design is validated — this chat is for design + prototyping only

---

## Edge cases identified

- **Notion webhook coverage is opt-in per page/database** — pages not explicitly shared with the integration never fire events. Needs a periodic backfill crawler as a safety net, not just manual subscription upkeep.
- **Notion webhook payloads are notification-only** — they signal *that* something changed, not *what* changed; content must always be fetched separately.
- Notion aggregates rapid edits into a single `page.content_updated` event within a short window — reduces mid-edit triage noise (favorable, not a risk).
- **Webhook redelivery risk**: a redelivered event without an idempotency check could trigger a duplicate action (e.g. duplicate calendar event). Needs an idempotency key — `source_id` + `last_edited_time`.
- **Gmail `history.list` / `historyId` expires after ~7 days.** If a push subscription (`watch()`, itself expiring every 7 days) lapses past that window unrenewed, `list_items(since=...)` can't diff cleanly and needs a fallback to a date-bounded search.
- **Notion block flattening is recursive** (nested toggles, sub-pages, synced blocks) — how deep to flatten before it's noise for the classifier is an open design call, not just implementation detail.
- `deadline`/`status`/`urgency_hint` are frequently null for sources like Gmail that have little structured metadata — expected, not an error; consumers must never assume they're populated. `raw_properties` shape still varies by source and is only used as a fallback, never assumed to have a fixed structure.
- Ambiguous relative dates in extraction (e.g. "next Friday") — resolution anchor (page-edit time vs. fetch time vs. today) not yet decided; risk of silently wrong scheduled events if left unresolved.
- **Naive plain-text flattening destroys table structure** — joining cell text loses which value belongs to which row/column. `Item.content` must be markdown (pipe tables, fenced code, headings), not a plain string, since embedding models/LLMs are trained to parse markdown structure back out.
- **Tables and code blocks can't be split mid-block during chunking** without losing meaning (a table without its header row, a function split mid-body) — these need atomic chunk boundaries, stricter than the heading/paragraph-based structure-aware chunking already planned.
- **Images, embedded files, and other non-text blocks have no markdown representation** — flattening can't solve this; needs OCR/captioning as a separate subsystem, or a placeholder + drop for v1.
- **Timezone**: a resolved date without an explicit timezone is still ambiguous for a calendar API call. `last_edited_at` from source APIs is typically UTC, but the user's intended meaning is local time — normalization must resolve into local timezone explicitly, not just an ISO date.
- **Re-edits after an item has already been acted on** are not covered by the redelivery idempotency key alone — that only stops duplicate webhook firings, not a genuine second edit. Requires the item lifecycle state table (see Decisions).
- **Prompt injection via untrusted sources** (Gmail, not Notion) — inbound content could contain adversarial instructions aimed at manipulating the classifier into auto-acting. Addressed via the guardrail/trust_level layer (see Decisions).
- **One obligation is routinely described by several chunks.** The fan-out rule ("multiple distinct mandatory deadlines within one item → multiple separate actions") assumed distinctness and never enforced it. A real page — a heading naming an exam date, a syllabus table, and a revision checklist — has three mandatory chunks describing *one* exam, and produced three identical calendar events. Proposals are now de-duplicated within an item by **the action definition's own `required_fields`**: two proposals agreeing on every field the action cannot be described without are two descriptions of one obligation. Derived from the registry rather than hardcoded to title+start, so a new action type inherits it. Identity is the resolved **instant(s)** plus *compatible* text: an obligation is identified far more reliably by when it falls due than by what the model chose to call it, and the same deadline read from a heading and from a checklist comes back as "Lab submission 3" and "Lab submission 3 deadline". Text fields therefore match on containment, not equality — two genuinely different obligations that happen to share an instant have names where neither contains the other. A proposal missing any required field has no identity and is left to the gate. Suppressed chunks are recorded in the audit log with what covers them, and hold no lifecycle row — so a chunk that *becomes* a duplicate has its old event reconciled away.
- **The database session timezone is pinned to the configured timezone.** `timestamptz` is an instant regardless, so this changes no stored value and no comparison — but it changes what a value *renders as*, and those renderings are read by people and quoted verbatim by the model. Left at the server default, a 15 September 00:00 IST deadline comes back as 14 September 18:30Z and the agent reports the 14th: the wrong day, from correct data. Fixed at the connection layer rather than at each call site, since the timezone is fixed for v1 anyway.
- **A failed classification must not leave the parallel embed leg running.** Embedding and classification deliberately overlap on one connection (classification does no database work, so the embed leg has the connection to itself). If classification *raises*, the embed task is still mid-statement when the per-item transaction is rolled back — its INSERT then lands in the next item's transaction against an item row that no longer exists, and one bad item poisons the rest of the poll, which is exactly what per-item transactions exist to prevent. The pipeline now settles the embed task before letting the classification error propagate.
- **Switching embedding models later requires a full corpus re-embed, not a config change** — different models produce different vector spaces (not just different dimensions), so existing `embedding` column values become meaningless against a new model's output. Worth weighing before the still-open model choice is locked in.

---

## Tradeoffs discussed

| Decision point | Options weighed | Chosen | Why |
|---|---|---|---|
| Classify+extract | One combined LLM call vs. two separate calls | Two calls | Cleaner independent eval/debugging, acceptable latency at personal-use volume |
| Trigger mechanism | Webhook push (real-time, needs public endpoint) vs. poll (simpler, latency window) | Per-connector: Notion polls, Gmail pushes | Urgency lives asymmetrically — self-authored Notion content tolerates staleness, inbound email deadlines don't |
| Item schema | Rigid fixed schema vs. loose `properties` dict | Loose dict | Avoids losing Notion's structured signal or inventing fake fields for Gmail; cost is the classifier must handle missing fields |
| On-demand fetch exposure | One generic tool with source param vs. one tool per connector | One tool per connector | Matches natural LLM tool-selection ("check mail" vs "update all") without extra routing logic |
| Threshold tuning | Upfront hand-labeled eval set vs. implicit feedback from queue behavior | Implicit feedback | Less upfront work; system self-corrects with real usage |
| Change detection safety net | Trust webhook/manual subscription upkeep vs. periodic backfill crawl | Backfill crawl | Consistent with not trusting a single source of truth for change detection |
| Content flattening format | Plain concatenated text vs. markdown | Markdown | Preserves table/list/code structure in a format embedding models and LLMs are trained to parse; plain text loses row/column relationships entirely |
| Relative date handling | Resolve only at extraction time vs. normalize into content at ingestion | Normalize at ingestion (annotate, don't overwrite) | Fixes stored/embedded content for future retrieval too, not just extraction; also makes the extract call's job trivial for the common case |
| Extraction schema shape | Fixed calendar-only schema vs. registry-based pluggable action types | Registry-based, via a single operations-agent tool-calling step | Avoids binding the system to calendar events only; new action types register without touching classify/reconciliation logic; unregistered action types gracefully queue instead of forcing a shape or silently dropping |
| Action selection + execution | One combined agentic step (decide + execute) vs. propose-then-gate | Propose-then-gate — operations agent proposes, execution stays separate and gated | Untrusted content (Gmail) reaching the operations agent must never translate directly into a real action; existing confidence/sanity-check gates need something to check before anything actually happens |
| Properties handling | Raw per-source dict passed to consumers vs. connector-normalized fixed vocabulary | Connector-normalized (`deadline`/`status`/`urgency_hint`) + `raw_properties` fallback | Two consumers now read properties (classify, operations agent) — one consistent vocabulary avoids each having to handle arbitrary per-source shapes; raw stays available for anything the fixed set misses |

---

## Implementation spec (concrete, for Claude Code to build from)

This section is the actual build contract — kept in sync as each piece is finalized. Prose sections above explain *why*; this section is *what to build*.

### Item model

```python
@dataclass
class Item:
    source_type: str          # "notion", "gmail", ...
    source_id: str             # source-native id
    url: str
    title: str
    content: str                # MARKDOWN, not plain text — see edge cases
    created_at: datetime
    last_edited_at: datetime
    deadline: Optional[datetime] = None      # normalized, connector-populated when confidently mappable
    status: Optional[str] = None              # normalized
    urgency_hint: Optional[str] = None        # normalized
    raw_properties: dict = field(default_factory=dict)   # full source-native passthrough, fallback for anything the fixed vocabulary misses
```

### SourceConnector interface

See `connectors.py` (already prototyped and validated in this chat) for the working reference implementation — `SourceConnector` ABC, `NotionConnector`/`GmailConnector` stubs, `TriggerConfig` registry, `registered_tools()`, `run_scheduled_polls()`, `on_demand_fetch()`. Carry this file forward as-is into the real repo; it's the starting point, not a rewrite.

### Classifier interface

```python
@dataclass
class ClassificationResult:
    label: str            # "informational" | "actionable-mandatory" | "actionable-optional"
    confidence: float      # 0.0-1.0
    rationale: str
    raw_runs: list | None = None   # populated only by multi-call strategies (unused by v1)

class Classifier(ABC):
    @abstractmethod
    def classify(self, item: Item) -> ClassificationResult: ...

class SingleCallClassifier(Classifier):
    # v1 — one structured LLM call, self-reported confidence

class SelfConsistencyClassifier(Classifier):
    # future — N calls at temp>0, majority label, agreement rate as confidence
    # swap-in replacement, no changes needed to triage-layer callers
```

### Classifier system prompt (v1)

```
You are a triage classifier for a personal assistant that watches the user's
Notion workspace (and later, other sources) for items requiring action.

For each item, classify it into exactly one label:

- "actionable-mandatory": has a real deadline or obligation with a genuine
  consequence if missed (an exam, a submission, a scheduled appointment,
  a required task with a due date).
- "actionable-optional": suggests or invites an action but carries no real
  consequence if ignored (an optional session, a suggestion, a "you might
  want to" item).
- "informational": no action is implied at all (notes, reference material,
  completed items, general context).

Use both the item's content and its normalized deadline/status/urgency signals
(if present) — a deadline field or a status like "not started" is a strong
signal. These may be null; do not assume they exist, and rely on content
alone when they're missing.

Getting "actionable-mandatory" wrong is costly: it can trigger an automatic
action (e.g. creating a calendar event) with no human review. Getting
"actionable-optional" or "informational" wrong is comparatively low-cost —
the item is simply queued or indexed instead.

When an item sits near the boundary between mandatory and optional, or
between optional and informational, resolve the ambiguity toward the
LOWER-consequence label. This bias should be mild, not extreme — a clear,
well-signaled deadline should still be labeled mandatory with high
confidence. The bias only applies to genuinely ambiguous cases, not to
clear ones.

Respond only via the classify_item tool call, with:
- label: one of the three values above
- confidence: 0.0-1.0, calibrated — reserve values above 0.8 for cases with
  an explicit, unambiguous deadline signal
- rationale: one sentence explaining the call

Examples:
[boundary-case few-shots — to be written from real Notion content during
implementation, not left generic]
```

### Date normalization (new pipeline stage — runs right after fetch_item)

Relative dates are resolved into the content itself, before both storage and classify/extract — not just at extraction time. Two reasons: (1) stored/embedded content with "next Friday" is ambiguous forever at retrieval time; resolving at ingestion keeps stored content unambiguous. (2) it makes the extract call's job close to trivial for the common case — it reads an explicit date instead of doing date math.

- **Annotate, never overwrite**: `"the exam is next Friday [Aug 28, 2026]"`, not a blind replacement — preserves original phrasing, keeps a bad resolution visible/correctable rather than silently baked in.
- **Deterministic parser first, LLM fallback only for ambiguous cases**: a date-parsing library (e.g. `dateparser` with `RELATIVE_BASE` set to `item.last_edited_at`) handles the common cases cheaply with no model call. Only phrasing the parser can't confidently resolve escalates to an LLM call. If even that's ambiguous, leave the original text untouched — don't guess.
- Anchor is still `item.last_edited_at`, consistent with the extract-call anchor decision above.

### Extract interface (revised — operations agent replaces classify-embedded action_type + per-type Extractor)

`ClassificationResult` stays narrow — just the judgment call, no tool knowledge:

```python
@dataclass
class ClassificationResult:
    label: str
    confidence: float
    rationale: str
    raw_runs: list | None = None
```

If `label != "informational"`, the item is handed to a single **operations agent** call — one tool-calling turn, with the registered `ACTION_REGISTRY` tools exposed as its available functions. This merges what were two separate steps (action_type inference in classify, then a per-type `Extractor`) into one: the agent picks a tool and fills its parameters in the same motion. If nothing in the registry fits, it calls nothing — that absence is the "no tool available" signal, no separate reasoning needed to detect it.

```python
def operations_agent(item: Item) -> Optional[ActionSpec]:
    # sees ACTION_REGISTRY's tool schemas, proposes one call + arguments
    # returns None if no registered tool fits
    ...
```

**Critical constraint: the operations agent's output is a proposal, never a direct execution.** This matters most for untrusted content (Gmail) — if tool selection and tool firing happened in the same step, adversarial content that manipulates the agent's reasoning would go straight to a real action with nothing to catch it. Actual execution stays a separate, gated step:

```python
if classification.label == "actionable-mandatory" and spec.passes_confidence_gate():
    result = executor.create(spec)              # only here does anything real happen
    log_action_outcome(item, spec, result)        # embedded separately, immutable (see Storage)
else:
    queue_item(item, classification, spec)
```

All prior gating logic still applies unchanged: per-field confidence thresholds, the past-date sanity check, and the boundary-case bias — they now gate the operations agent's proposal rather than a separate Extractor's output, but the mechanism is identical.

- **Relative date anchor: `item.last_edited_at`**, not the time the pipeline happens to run. Matters most for backfill — a months-old page's "next Friday" must resolve against when it was written, not today.
- **Sanity check**: if the resolved date is already in the past, route to queue regardless of confidence — a past-dated auto-created event is never correct, and this usually signals stale content or a misresolved relative phrase.
- **Auto-action gating rule**: requires classify confidence AND all relevant `fields[...].confidence` in the `ActionSpec` to clear threshold, AND (for calendar-shaped actions) the resolved date to not be in the past. Any single failure routes to the queue with the partial spec attached (a pre-filled draft, not a blank item).

### Queue schema

Every reason an item reaches the queue is already implied by earlier decisions — this collects them into an explicit `queue_reason` rather than a generic "needs review" flag:

```python
@dataclass
class QueueEntry:
    id: UUID
    item_id: str                    # FK to source item
    chunk_id: Optional[str]         # set when a specific chunk triggered this (oversized items)
    classification_label: str
    classification_confidence: float
    action_type: Optional[str]      # None for informational/no-action queue entries
    action_spec: Optional[dict]     # pre-filled ActionSpec.fields, if extraction ran
    queue_reason: str               # "optional" | "low_confidence" | "no_tool_available" |
                                     # "past_date" | "date_conflict"
    status: str                     # "pending" | "dismissed" | "promoted"
    created_at: datetime
    resolved_at: Optional[datetime]
    resolution: Optional[str]       # what the user actually did — feeds threshold tuning
```

- **Unique constraint: one `pending` entry per `(item_id, chunk_id)`.** A re-edit while an entry is still unreviewed updates the existing entry rather than creating a duplicate.
- **Queue and the item lifecycle table are separate and stay separate.** Lifecycle tracks post-execution state ("what action currently exists"); queue tracks pre-execution state ("what's awaiting a human decision"). Promoting a queue entry is the bridge: triggers `ActionExecutor.create()`, then writes a lifecycle row.
- `resolution` closes the loop on the threshold-tuning approach already decided — every promote/dismiss produces a (confidence, outcome) pair usable for calibration later, with no separate labeling effort.

### Storage schema

Reuses patterns already established (content-hash diffing, inline embedding column) from the RAG knowledge base project where the same problem was already solved.

```
items:
  id (PK), source_type, source_id, url, title, last_edited_at,
  content_hash,   -- diff-on-edit, same pattern as the knowledge-base project
  created_at

chunks:
  id (PK), item_id (FK), chunk_index, chunk_type ('text' | 'table' | 'code'),
  content, embedding vector(768), chunk_hash, created_at, updated_at
  -- 768-dim, matches nomic-embed-text (run locally)
  -- chunk_type enforces the atomic-chunking rule (tables/code never split mid-block)

item_lifecycle:
  item_id, chunk_id, action_type, last_classification, last_extracted_date,
  action_id (nullable — the ActionExecutor-returned id, e.g. calendar_event_id),
  updated_at
  -- PK (item_id, chunk_id), consistent with the queue table's uniqueness constraint

audit_log:
  id, item_id, chunk_id, event_type ('classification' | 'extraction' | 'action_taken' | 'queued'),
  payload (jsonb — full ClassificationResult/ActionSpec at that moment), timestamp
  -- append-only; source of truth for "why did it do that," and feeds future confidence calibration
```

If action-outcome summaries are embedded at all (still a deferred nice-to-have), they live in their own table with their own `embedding` column — a physically separate collection from `chunks`, not a shared namespace, per the immutability/namespace-separation decision above.

### Not yet specified (blocking full handoff)

All previously blocking items are now resolved. Remaining items are non-blocking (see Open questions below).

---

## Local development surfaces

Neither is part of the design; both exist so the design can be *seen* working.

**Mock Notion (`ops/mock_notion.py` + `ops/notion_fixtures.json`).** Mocks the Notion **HTTP API**, not the connector — three endpoints (`/v1/search`, `/v1/pages/{id}`, `/v1/blocks/{id}/children`) served from an editable fixture file, reached by pointing `NOTION_API_BASE` at it. Mocking at the HTTP boundary rather than substituting a fake connector is the whole point: the real `NotionConnector` runs, so the crawl, the recursive block flattening, the depth limit, the markdown tables, and property-matching-by-type are all exercised. A fake connector would skip exactly the code most worth watching. Each page's `last_edited_time` defaults to the fixture file's mtime, so editing the file behaves like editing Notion: the ingest CAS accepts it, unchanged pages are rejected by the content hash before any model call, and a changed date reconciles onto the same calendar event.

**Dashboard (`GET /` on the existing API).** A single self-contained page over the endpoints that already existed plus a few reads: what was ingested, what the audit log says about each decision, what is queued for review (approve/dismiss inline), what is scheduled, and a box that runs either `search_content` or a full agent turn. Served by the same FastAPI app rather than as a separate frontend — a second process with its own build step would be more machinery than the thing it looks at. It reports prominently whether execution is a dry run, because "nothing was actually written to a calendar" is the single most important fact about a run.

---

## Deferred — general RAG memory layer (not yet designed)

Raised while discussing whether the Postgres store could also serve as memory for a separate RAG use case. None of this is decided — captured here so it isn't lost, to be designed later if this direction is pursued.

**Access & scope**
- Access boundary: what counts as "memory" (`items`/`chunks`) vs. Jake's assistant's own operational bookkeeping (`queue`, `item_lifecycle`, `audit_log`) — undecided whether any of the latter is exposed at all
- Interface for external access: a separate MCP server, a plain API, or something else — `search_content` as it exists is Jake's assistant-specific (grounding rule, attribution, item-title context baked in), likely wrong shape for a generic consumer
- Data sensitivity/scoping: whether Gmail's `untrusted` trust_level should translate into access scoping for external consumers, given it's already treated as more sensitive than Notion
- Schema stability: once something external depends on this store, changes (especially an embedding model switch, already flagged as requiring a full re-embed) become more consequential
- Write access beyond the direct-input path already designed: could an external consumer write back, not just the user directly? Undecided — would reopen the connector/trust/normalization design
- Namespacing across consumers, if there's ever more than one — possibly unnecessary at personal scale, but worth naming rather than assuming away

**Chunking & retrieval strategy**
- Current chunking (structure-aware, atomic table/code boundaries) was optimized for triage precision (letting classify isolate which chunk triggered an action), not necessarily for general retrieval quality — whether one strategy should serve both purposes is undecided
- Chunk overlap is not currently designed — structure-aware boundaries don't need it for triage, but general RAG retrieval has known context-loss failure modes at hard boundaries
- Retrieval is currently flat cosine top-k + threshold only — no hybrid retrieval or reranking, which were both explored for the separate RAG knowledge-base project and would be relevant questions again here if this becomes a general memory layer
- Chunk-level vs. item-level granularity may need to differ by consumer — triage wants a precise chunk, broader RAG queries may want "expand to the full parent item"
- Metadata filtering (by source, trust_level, recency, date range) not designed
- Freshness-weighted ranking — similar territory to the time-decay retrieval explored for [[study-copilot]] v2, arguably more relevant here given how much of Jake's assistant concerns what's currently due vs. stale

---

## Open questions (not yet decided)

- Model choice for classification/extraction: self-hosted (per the knowledge-base project's constraint) vs. hosted API (better reliability for structured extraction, viable at this low volume) — not yet decided for this project
- Relative date resolution anchor for extraction (proposed: page-edit timestamp, not query time — needs confirmation)
- Depth of Notion block flattening before it's classifier noise
- Concrete auto-action confidence threshold value (to be tuned once the queue has real usage data)
- Scope of image/attachment handling for v1: drop-with-placeholder vs. building OCR/captioning as part of the connector pipeline
- Exact heuristic for matching a source-native property to the normalized vocabulary (e.g. Notion: match by property type vs. by label synonyms, and how to disambiguate when multiple date-typed properties exist) — implementation detail, not a blocking design decision
