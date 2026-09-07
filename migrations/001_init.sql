-- 001_init.sql -- initial schema for Jake's assistant.
--
-- Everything lives in one Postgres: the vectors, the triage bookkeeping, the
-- review queue, and the audit log. Splitting the vectors into a dedicated
-- vector database would mean two-phase writes to keep an index agreeing with a
-- queue that needs transactions and joins.
--
-- Four ideas are load-bearing in the DDL rather than in application code:
--
--   * `items (source_type, source_id)` is UNIQUE, and every ingest is a
--     compare-and-swap on `last_edited_at`. There is no dedup table: a
--     redelivered webhook and a late out-of-order event both resolve to
--     "no rows updated, do nothing".
--   * `chunks.chunk_key` is stable identity (heading path + ordinal within the
--     section); `chunk_index` is mere position. The lifecycle and queue tables
--     reference chunk_key, so an edit that inserts a paragraph above an exam
--     notice does not repoint its calendar event at different content.
--   * `queue_entries` and `item_lifecycle` are separate tables. Pre-execution
--     state and post-execution state answer different questions and have
--     different lifetimes.
--   * `action_outcomes` has its own embedding column in its own table, never
--     sharing a vector space with source content. "Calendar updated: assessment
--     scheduled" must not surface at retrieval time as though it were the
--     substance of the assessment.

CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- items -- one row per source-native unit of content
-- ---------------------------------------------------------------------------

CREATE TABLE items (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    source_type       TEXT NOT NULL,   -- 'notion' | 'gmail' | 'manual'
    source_id         TEXT NOT NULL,   -- source-native id
    url               TEXT NOT NULL DEFAULT '',
    title             TEXT NOT NULL DEFAULT '',

    -- Markdown, never plain text. See models.Item.content.
    content           TEXT NOT NULL,
    content_hash      TEXT NOT NULL,

    created_at        TIMESTAMPTZ NOT NULL,

    -- The idempotency guard. Only strictly-newer edits are ever accepted, which
    -- is what makes redelivery and out-of-order delivery the same non-event.
    last_edited_at    TIMESTAMPTZ NOT NULL,

    -- Connector-normalized vocabulary. Frequently NULL (Gmail carries almost no
    -- structured metadata); consumers must never assume these are populated.
    deadline          TIMESTAMPTZ,
    status            TEXT,
    urgency_hint      TEXT,

    -- Full source-native passthrough for everything the fixed vocabulary misses.
    raw_properties    JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- 'trusted' | 'untrusted', inherited from the originating connector.
    trust_level       TEXT NOT NULL DEFAULT 'trusted',

    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT items_source_unique UNIQUE (source_type, source_id),
    CONSTRAINT items_trust_level_valid CHECK (trust_level IN ('trusted', 'untrusted'))
);

CREATE INDEX items_last_edited_idx ON items (last_edited_at DESC);
CREATE INDEX items_deadline_idx ON items (deadline) WHERE deadline IS NOT NULL;
CREATE INDEX items_source_type_idx ON items (source_type);


-- ---------------------------------------------------------------------------
-- chunks -- structure-aware, atomically-bounded, individually embedded
-- ---------------------------------------------------------------------------

CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id         UUID NOT NULL REFERENCES items (id) ON DELETE CASCADE,

    -- Stable across edits. This is what item_lifecycle and queue_entries join
    -- on; chunk_index is position and shifts on any insertion above.
    chunk_key       TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    heading_path    TEXT NOT NULL DEFAULT '',

    -- 'text' | 'table' | 'code'. Tables and code are never split mid-block: a
    -- table without its header row, or a function cut mid-body, means something
    -- different from what it meant whole.
    chunk_type      TEXT NOT NULL DEFAULT 'text',

    content         TEXT NOT NULL,

    -- SHA-256 of exactly the text that was embedded, so an equal hash implies
    -- an equal embedding input and "unchanged -> skip re-embed" is sound rather
    -- than merely plausible.
    content_hash    TEXT NOT NULL,

    -- 768 dims: nomic-embed-text, run locally. Switching embedding models later
    -- is a full corpus re-embed, not a config change -- different models produce
    -- different vector spaces, not merely different widths.
    embedding       VECTOR(768),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chunks_item_key_unique UNIQUE (item_id, chunk_key),
    CONSTRAINT chunks_type_valid CHECK (chunk_type IN ('text', 'table', 'code'))
);

CREATE INDEX chunks_item_idx ON chunks (item_id);
CREATE INDEX chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);


-- ---------------------------------------------------------------------------
-- item_lifecycle -- post-execution state: what action currently exists
-- ---------------------------------------------------------------------------

CREATE TABLE item_lifecycle (
    item_key             TEXT NOT NULL,   -- '<source_type>:<source_id>'
    chunk_key            TEXT NOT NULL,
    action_type          TEXT NOT NULL,

    last_classification  TEXT NOT NULL,
    last_extracted_date  TIMESTAMPTZ,

    -- Whatever the ActionExecutor returned -- a calendar event id today, some
    -- other destination's id tomorrow. Reconciliation deletes and updates
    -- through this, never by searching the destination for a matching event.
    action_id            TEXT,

    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (item_key, chunk_key)
);

CREATE INDEX item_lifecycle_item_idx ON item_lifecycle (item_key);


-- ---------------------------------------------------------------------------
-- queue_entries -- pre-execution state: what awaits a human decision
-- ---------------------------------------------------------------------------

CREATE TABLE queue_entries (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    item_key                  TEXT NOT NULL,
    -- NULL when the whole item queued rather than one triggering chunk.
    -- Coalesced in the uniqueness index below, since NULL <> NULL in SQL.
    chunk_key                 TEXT,

    classification_label      TEXT NOT NULL,
    classification_confidence REAL NOT NULL,

    action_type               TEXT,
    -- The pre-filled ActionSpec: a draft to accept, not a blank item to redo.
    action_spec               JSONB,

    queue_reason              TEXT NOT NULL,
    status                    TEXT NOT NULL DEFAULT 'pending',

    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at               TIMESTAMPTZ,
    -- What the user actually did. Feeds threshold calibration with no separate
    -- labelling effort.
    resolution                TEXT,

    CONSTRAINT queue_status_valid
        CHECK (status IN ('pending', 'dismissed', 'promoted')),
    CONSTRAINT queue_resolved_at_matches_status
        CHECK (status = 'pending' OR resolved_at IS NOT NULL)
);

-- One *pending* entry per (item, chunk): a re-edit while an entry is still
-- unreviewed updates that entry rather than stacking a second copy. Resolved
-- entries are history and may accumulate freely.
CREATE UNIQUE INDEX queue_pending_unique
    ON queue_entries (item_key, COALESCE(chunk_key, ''))
    WHERE status = 'pending';

CREATE INDEX queue_pending_idx ON queue_entries (created_at DESC) WHERE status = 'pending';
CREATE INDEX queue_item_idx ON queue_entries (item_key);


-- ---------------------------------------------------------------------------
-- action_outcomes -- immutable record of what actually happened
-- ---------------------------------------------------------------------------
--
-- Its own table, and therefore its own vector space. A record whose meaning
-- changes over time (scheduled -> completed -> scored) is not one mutable
-- embedding: this is append-only, and genuinely new information arrives as a
-- new Item through the normal ingestion path instead.

CREATE TABLE action_outcomes (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    item_key      TEXT NOT NULL,
    chunk_key     TEXT,
    action_type   TEXT NOT NULL,
    operation     TEXT NOT NULL,   -- 'create' | 'update' | 'delete'
    action_id     TEXT,

    summary       TEXT NOT NULL,   -- e.g. 'created calendar event: DBMS exam, 2026-08-28'
    payload       JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding     VECTOR(768),

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT action_outcomes_operation_valid
        CHECK (operation IN ('create', 'update', 'delete'))
);

CREATE INDEX action_outcomes_item_idx ON action_outcomes (item_key, created_at DESC);
CREATE INDEX action_outcomes_embedding_hnsw
    ON action_outcomes USING hnsw (embedding vector_cosine_ops);


-- ---------------------------------------------------------------------------
-- audit_log -- append-only "why did it do that"
-- ---------------------------------------------------------------------------

CREATE TABLE audit_log (
    id           BIGSERIAL PRIMARY KEY,

    -- 'classification' | 'extraction' | 'action_taken' | 'queued' |
    -- 'gate_blocked' | 'skipped' | 'injection_flagged' | 'reconciled'
    event_type   TEXT NOT NULL,

    item_key     TEXT,
    chunk_key    TEXT,

    -- The full ClassificationResult / ActionSpec as it stood at that moment.
    payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX audit_log_item_idx ON audit_log (item_key, created_at DESC);
CREATE INDEX audit_log_type_idx ON audit_log (event_type, created_at DESC);


-- ---------------------------------------------------------------------------
-- source_state -- per-connector cursor for incremental fetch
-- ---------------------------------------------------------------------------
--
-- In the database rather than a dotfile because the Gmail historyId and the
-- last-poll timestamp have to stay consistent with the items written in the
-- same transaction. A cursor advanced on disk after a failed commit would skip
-- messages permanently.

CREATE TABLE source_state (
    source_type     TEXT PRIMARY KEY,
    last_fetch_at   TIMESTAMPTZ,
    cursor          TEXT,          -- Gmail historyId, Notion pagination anchor, ...
    watch_expires_at TIMESTAMPTZ,  -- push subscriptions expire (Gmail: ~7 days)
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ---------------------------------------------------------------------------
-- updated_at maintenance
-- ---------------------------------------------------------------------------

CREATE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER items_touch_updated_at
    BEFORE UPDATE ON items
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

CREATE TRIGGER chunks_touch_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

CREATE TRIGGER item_lifecycle_touch_updated_at
    BEFORE UPDATE ON item_lifecycle
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
