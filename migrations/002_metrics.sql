-- 002_metrics.sql -- one durable row per processed item.
--
-- The in-process counters in `assistant.metrics` answer "what is happening now"
-- and reset with the process. This table answers the question they cannot: what
-- does an item cost, across a week of real mail, on this machine, with this
-- model. That question is only ever asked after a restart.
--
-- Deliberately not the audit log. `audit_log` records *decisions* and is read
-- by a person asking why something happened; this records *cost* and is read by
-- an aggregate query. Mixing them would mean either scanning JSONB payloads to
-- compute a median or writing latency into a table whose rows are quoted back
-- to the user as explanations.
--
-- Written inside the same transaction as everything else the item produced, so
-- a rolled-back item leaves no measurement of work that, as far as the rest of
-- the system is concerned, never happened.

CREATE TABLE pipeline_metrics (
    id                BIGSERIAL PRIMARY KEY,

    item_key          TEXT NOT NULL,
    source_type       TEXT NOT NULL,

    -- 'acted' | 'queued' | 'skipped' | 'indexed'. Ordered by consequence, so an
    -- item that both acted and queued counts as having acted.
    outcome           TEXT NOT NULL,
    label             TEXT,

    -- Wall clock for the whole item. Wall clock rather than CPU time because
    -- almost all of it is spent waiting on a model, and the wait is the cost.
    duration_ms       INTEGER NOT NULL,

    llm_calls         INTEGER NOT NULL DEFAULT 0,
    llm_ms            INTEGER NOT NULL DEFAULT 0,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,

    embed_calls       INTEGER NOT NULL DEFAULT 0,
    embed_ms          INTEGER NOT NULL DEFAULT 0,
    embedded_texts    INTEGER NOT NULL DEFAULT 0,

    -- Per-stage milliseconds. JSONB rather than columns: stages are a property
    -- of the current pipeline shape, and a stage added later should not be a
    -- migration.
    stages            JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pipeline_metrics_outcome_valid
        CHECK (outcome IN ('acted', 'queued', 'skipped', 'indexed'))
);

-- Every aggregate over this table is "the last N days", so the time index is
-- the only one worth having; the source/outcome breakdowns are grouped from a
-- window that this index has already narrowed.
CREATE INDEX pipeline_metrics_created_idx ON pipeline_metrics (created_at DESC);
CREATE INDEX pipeline_metrics_item_idx ON pipeline_metrics (item_key, created_at DESC);
