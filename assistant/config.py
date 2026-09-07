"""Central configuration, loaded from the environment (and .env for local runs).

Every entry point -- the webhook API, the poll scheduler, the CLI, the
conversational agent -- reads its settings from here, so a model name or a
threshold is defined in exactly one place.
"""

from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Postgres -------------------------------------------------------
    # 5434 because 5432 is the system cluster and 5433 belongs to the RAG
    # knowledge-base stack; all three run on this host at once.
    postgres_host: str = "localhost"
    postgres_port: int = 5434
    postgres_user: str = "assistant"
    postgres_password: str = "assistant"
    postgres_db: str = "assistant"

    # --- Embeddings (Ollama, local) --------------------------------------
    # nomic-embed-text for the same reason as the knowledge-base project: an
    # 8192-token window suits embedding fuller documents, not just short chunks.
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # --- Reasoning model (classify / operations agent / conversation) -----
    # "ollama" keeps everything on this machine and needs no credentials;
    # "anthropic" trades that for markedly better structured tool-calling.
    # The choice is a config value rather than a code path because the design
    # deliberately left it open -- see the design log's Open questions.
    llm_provider: str = "ollama"
    anthropic_model: str = "claude-opus-5"
    ollama_reasoning_model: str = "qwen2.5:7b-instruct-q4_K_M"
    # Retrieved chunks plus the prompt have to fit; Ollama's 4096 default would
    # silently truncate the item being classified.
    reasoning_num_ctx: int = 8192

    # --- Triage gating ---------------------------------------------------
    # Deliberately conservative to start. The queue's (confidence, resolution)
    # pairs are what these get tuned from later -- there is no hand-labelled
    # eval set upfront.
    classify_confidence_threshold: float = 0.80
    field_confidence_threshold: float = 0.75
    # Items shorter than this with no normalized property signal skip
    # classification entirely and default to informational.
    min_content_chars: int = 30

    # --- Retrieval -------------------------------------------------------
    retrieval_top_k: int = 10
    # Without a floor, an unrelated query still returns the closest available
    # chunks and the agent answers confidently from noise. Below this, the
    # honest answer is "nothing indexed about that".
    #
    # 0.58 is measured, not guessed: with nomic-embed-text task prefixes, a
    # deliberately unrelated query scored 0.51 against a seeded corpus while a
    # genuinely relevant one scored 0.67. That gap is narrow, and it narrows
    # further as a corpus grows -- re-measure against real content rather than
    # trusting this number indefinitely.
    retrieval_min_similarity: float = 0.58

    # --- Time ------------------------------------------------------------
    # Fixed for v1 rather than derived per item (single user, one timezone).
    timezone: str = "Asia/Kolkata"

    # --- Connectors ------------------------------------------------------
    notion_api_key: str = ""
    notion_version: str = "2022-06-28"
    # Points at the real API by default. `ops/mock_notion.py` serves the same
    # three endpoints from a fixture file, so pointing this at it exercises the
    # connector -- block flattening, property matching, the crawl -- for real,
    # rather than testing a stub that skips all of it.
    notion_api_base: str = "https://api.notion.com"
    notion_poll_interval_hours: float = 12.0

    gmail_credentials_file: str = "secrets/gmail_credentials.json"
    gmail_token_file: str = "secrets/gmail_token.json"
    gmail_topic: str = ""  # projects/<project>/topics/<topic> for watch()
    gmail_query: str = "in:inbox -category:promotions"

    # --- Action execution (MCP) -------------------------------------------
    # A command that speaks MCP over stdio. Empty disables real execution and
    # the dry-run executor is used instead -- the default, so a fresh clone
    # cannot write to a real calendar by accident.
    calendar_mcp_command: str = ""
    calendar_mcp_args: str = ""
    calendar_id: str = "primary"

    # --- Application ------------------------------------------------------
    # Loopback by default. Binding anywhere else without API_TOKEN is refused at
    # startup rather than served unauthenticated (see api.security.enforce_bind).
    api_host: str = "127.0.0.1"
    api_port: int = 8100
    log_level: str = "INFO"

    # --- Request authentication -------------------------------------------
    # A shared secret, not user accounts: there is one user, and what needs
    # protecting is the *request* -- `/ask` and `/ingest` spend model time, and
    # `/queue/{id}/promote` writes to a real calendar. Sent as
    # `Authorization: Bearer <token>` or `X-API-Token`.
    #
    # Empty is allowed only while bound to loopback. The reasoning is the same
    # as the dry-run calendar default: the insecure configuration is the one
    # that cannot reach anything, and the moment it can, it must be configured.
    api_token: str = ""

    # --- Webhook verification ---------------------------------------------
    # Push endpoints are the one part of this system reachable by anyone who
    # learns the URL, and they cost model calls per delivery. Three independent
    # mechanisms, each enabled by configuring it:
    #
    # 1. `webhook_token` -- a secret in the push URL's query string. Pub/Sub
    #    supports this natively and it costs one comparison.
    # 2. `google_pubsub_audience` + `google_pubsub_service_account` -- full OIDC
    #    verification of the token Pub/Sub attaches to every push: RS256 against
    #    Google's rotating JWKS, plus issuer, audience, and the service account
    #    that is allowed to deliver.
    # 3. `notion_webhook_secret` -- the token from Notion's subscription
    #    handshake, used as the HMAC-SHA256 key over the raw request body.
    #
    # Anything configured must pass. Nothing configured means the endpoint
    # refuses to process deliveries at all, unless `webhook_allow_unverified` is
    # explicitly set -- fail closed, because the failure mode of the alternative
    # is an open endpoint nobody remembers is open.
    webhook_token: str = ""
    google_pubsub_audience: str = ""
    google_pubsub_service_account: str = ""
    notion_webhook_secret: str = ""
    webhook_allow_unverified: bool = False

    @property
    def database_url(self) -> str:
        """libpq-style connection string for psycopg."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def gmail_webhook_verifiers(self) -> list[str]:
        """Which Gmail push checks are configured. Empty means unverified."""
        configured = []
        if self.webhook_token:
            configured.append("url_token")
        if self.google_pubsub_audience and self.google_pubsub_service_account:
            configured.append("oidc")
        return configured

    @property
    def notion_webhook_verifiers(self) -> list[str]:
        return ["hmac"] if self.notion_webhook_secret else []


@lru_cache
def get_settings() -> Settings:
    return Settings()
