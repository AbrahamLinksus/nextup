# nextup

A personal assistant that ingests Notion pages and Gmail threads, triages every
item, acts automatically on the ones it is confident about, and queues the rest
for review.

The design goal is not "an agent that does things." It is a system that knows
the difference between what it may do unsupervised and what it must ask about.
Everything below follows from that split.

## Two execution models

**The pipeline** (`assistant/pipeline.py`) is deterministic. Every fetched item
— polled, pushed, or requested on demand — runs the same sequence, and the
trigger type never changes what happens downstream:

```
ingest guard (compare-and-swap)
  -> date normalization
  -> chunk
  -> embed  ||  classify (per chunk)        [in parallel]
  -> if actionable: injection check -> operations agent proposes
  -> GATE
  -> execute (and record lifecycle)  or  queue (with the draft attached)
  -> reconcile anything the edit invalidated
```

There are exactly two model calls in the actionable path — the classifier's
structured label and the operations agent's single proposal — and no
free-standing tool execution. Nobody is watching, so the model gets narrow calls
and a gate.

**The agent** (`assistant/agent.py`) is conversational and has real tool-calling
freedom. The user is present, reading the answer and able to say no, so the
model gets a tool list and a loop. Its system prompt leans on two rules:
answer only from what the tools returned, and name the source item behind any
retrieved claim.

## The gate

`assistant/gating.py` is the only place a proposal becomes permission to act.
Four rules:

1. **Classification confidence** — an uncertain label should not produce a
   certain action.
2. **Per-field confidence** — whole-proposal confidence hides the common
   failure: sure about the event, guessing at the date.
3. **Past dates** — a past-dated auto-created event means stale content or a
   misresolved relative phrase. Both want a human.
4. **Injection suspicion overrides everything** — a flagged item takes the
   conservative path regardless of reported confidence, because confidence is
   exactly what a manipulative item would try to inflate.

Failing a rule is not rejection. The item queues *with its partial spec
attached* — a pre-filled draft to accept, not a blank to redo.

## Quick start

Requires Python 3.12+, Docker (for Postgres), and either a local
[Ollama](https://ollama.com) or an Anthropic API key.

```bash
cp .env.example .env          # every value has a working default except credentials
docker compose up -d postgres # pgvector on :5434
uv sync --extra dev
uv run assistant migrate
```

To see the whole pipeline run with no Notion account and no cloud credentials,
point the connector at the fixture workspace:

```bash
uv run python ops/mock_notion.py          # fake Notion API on :8200
NOTION_API_BASE=http://localhost:8200 NOTION_API_KEY=anything \
  uv run assistant ingest notion
uv run assistant queue list
```

`ops/mock_notion.py` mocks the *HTTP API*, not the connector — the real
`NotionConnector` runs against checked-in fixtures you can edit in
`ops/notion_fixtures.json`. Edits take effect with no restart.

With `CALENDAR_MCP_COMMAND` unset, every action is a dry run; nothing is written
to a real calendar.

## CLI

```
assistant migrate                 create or update the database schema
assistant status                  what is stored, scheduled, and pending
assistant ingest [notion|gmail|all]  fetch a source and run the triage pipeline
assistant queue list              the pull-based review queue
assistant queue promote <id>      accept a queued draft (executes it)
assistant queue dismiss <id>      drop a queued draft
assistant remember <text>         store something directly into memory
assistant search <query>          semantic search over stored content
assistant chat [message]          talk to the assistant
assistant audit [--item KEY]      why the system did what it did
assistant upcoming [--days N]     what is currently scheduled
assistant metrics [--days N]      what the pipeline has cost, measured
assistant watch                   register push subscriptions
assistant serve                   run the webhook API
assistant poll                    run the scheduler (polling, backfill, renewal)
```

## API

`assistant serve` runs a FastAPI app (`assistant/api/main.py`) with a small
dashboard at `/`, read endpoints for the queue, audit log, metrics and upcoming
actions, and the Gmail/Notion webhook receivers.

Every route is authenticated one of two ways. The read/write API takes a shared
secret (`API_TOKEN`, sent as `Authorization: Bearer <token>`), because `/ask`
and `/ingest` spend model time and `/queue/{id}/promote` writes to a real
calendar. Webhooks cannot use that secret — Google has no way to learn it — so
each is verified the way its source signs deliveries: an OIDC token for Pub/Sub,
an HMAC over the raw body for Notion. With nothing configured the webhook
endpoints fail closed.

Binding to anything other than loopback without `API_TOKEN` set refuses to start
rather than serving an open API.

Push handling acknowledges immediately and processes afterwards: Pub/Sub retries
any delivery it does not get a prompt 2xx for, and processing an email takes
several model calls.

## Configuration

All settings live in `assistant/config.py` and are read from `.env`; see
`.env.example` for the annotated list. The ones worth knowing:

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` (fully local, no credentials) or `anthropic` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | via Ollama |
| `CLASSIFY_CONFIDENCE_THRESHOLD` | `0.80` | gate rule 1 |
| `FIELD_CONFIDENCE_THRESHOLD` | `0.75` | gate rule 2 |
| `NOTION_API_BASE` | `https://api.notion.com` | point at the mock to run offline |
| `CALENDAR_MCP_COMMAND` | *(empty)* | empty keeps every action a dry run |
| `API_TOKEN` | *(empty)* | optional on loopback, required otherwise |

## Tests

292 tests across 20 files, run against a real Postgres with pgvector rather than
mocks — the ingest compare-and-swap, the queue's partial unique index and the
vector similarity floor are behaviours of PostgreSQL, and a suite that mocked
them would be asserting that the mock works.

```bash
uv run ruff check .
uv run pytest
```

Database tests skip cleanly when Postgres is unreachable. CI asserts they did
not skip, so a misconfigured service container cannot look like a passing run.
A second job checks that `migrations/` applies to an empty database from
nothing, and that applying it again is a no-op.

## Layout

```
assistant/
  pipeline.py      the deterministic ingest -> gate -> act sequence
  gating.py        the four rules that decide act vs. queue
  agent.py         the conversational agent and its tool loop
  operations.py    the single-proposal operations agent
  classify.py      structured labelling
  guardrails.py    prompt-injection detection
  dates.py         relative-date resolution before anything reaches an LLM
  chunking.py      markdown-aware chunking
  embeddings.py    embedding via Ollama
  retrieval.py     vector search with a relevance floor
  repository.py    all SQL
  actions/         ACTION_REGISTRY; a new action type is one definition + executor
  connectors/      Notion, Gmail, manual; poll schedule derives from the registry
  api/             webhook receiver, read API, dashboard
migrations/        versioned SQL, applied in order and idempotently
ops/mock_notion.py a fake Notion HTTP API over checked-in fixtures
tests/
```

`jakes-assistant-design-log.md` is the running design record — the reasoning
behind each of these choices as it was made.
