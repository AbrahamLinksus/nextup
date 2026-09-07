"""Who is allowed to make this request.

There are no user accounts here and there should not be: one person uses this
system. What needs authenticating is the *request*, and it needs it for two
different reasons, which is why there are two mechanisms rather than one.

**The API** is protected by a shared secret because its endpoints are not merely
readable -- `/ask` spends model time per call, `/ingest` spends it per item, and
`/queue/{id}/promote` writes to a real calendar. An unauthenticated one of those
is a way to spend someone else's GPU or put an event in their week.

**The webhooks** are protected differently because they are the one part of the
system that is *supposed* to be reachable by a stranger. A shared secret Google
does not know about is no use there, so each push source is verified the way
that source signs its deliveries: Pub/Sub attaches an OIDC token, and Notion
signs the body.

Both fail closed. An unset API token is tolerated only while the server is bound
to loopback, and an unverified webhook refuses to process deliveries at all
unless someone has explicitly said otherwise. The pattern is the same one the
dry-run calendar executor follows: the unconfigured state is the one that cannot
reach anything.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from fastapi import HTTPException, Request

from assistant.config import Settings, get_settings

log = structlog.get_logger(__name__)


class WebhookVerificationError(RuntimeError):
    """A delivery did not prove it came from the source it claims."""


# ---------------------------------------------------------------------------
# Shared-secret API auth
# ---------------------------------------------------------------------------

# Open because they carry nothing and cost nothing: the dashboard shell, which
# fetches everything else over authenticated XHR, and the liveness probe, which
# has to answer before anyone can be authenticated at all.
PUBLIC_PATHS = frozenset({"/", "/health"})

# Webhooks are not exempt from authentication -- they authenticate differently.
# Google has no way to learn this deployment's shared secret, so demanding it
# there would only mean turning the check off; the OIDC and HMAC verifiers below
# are what stand in its place.
SELF_VERIFIED_PREFIXES = ("/webhooks/",)


def presented_token(request: Request) -> str:
    """The token this request carries, from either accepted place.

    `Authorization: Bearer` is the header the dashboard and curl both reach for;
    `X-API-Token` exists because some proxies rewrite Authorization and losing
    the credential to infrastructure is a bad way to spend an afternoon.
    """
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return request.headers.get("x-api-token", "").strip()


def require_api_token(request: Request) -> None:
    """Reject a request that does not carry the configured shared secret.

    A no-op when no token is configured -- which `enforce_bind` has already
    guaranteed means the server is on loopback.
    """
    settings = get_settings()
    if not settings.api_token:
        return
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith(SELF_VERIFIED_PREFIXES):
        return
    # compare_digest rather than ==: a plain comparison returns as soon as two
    # bytes differ, and the time it took to do that is a readout of how much of
    # the token was right.
    if not hmac.compare_digest(presented_token(request), settings.api_token):
        raise HTTPException(status_code=401, detail="missing or invalid API token")


def enforce_bind(settings: Settings | None = None) -> None:
    """Refuse to serve an unauthenticated API on a reachable interface.

    Called at startup rather than per request, because the wrong answer here is
    a configuration mistake and configuration mistakes should be loud and
    immediate rather than discovered from an access log.
    """
    settings = settings or get_settings()
    if settings.api_token:
        return
    if _is_loopback(settings.api_host):
        log.warning(
            "api.unauthenticated_on_loopback",
            host=settings.api_host,
            detail="no API_TOKEN set; reachable only from this machine",
        )
        return
    raise RuntimeError(
        f"refusing to bind {settings.api_host} with no API_TOKEN set. "
        "Set API_TOKEN, or bind API_HOST=127.0.0.1 for local use."
    )


def _is_loopback(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Google Pub/Sub push (Gmail)
# ---------------------------------------------------------------------------

@dataclass
class GoogleTokenVerifier:
    """Verifies the OIDC token Pub/Sub attaches to every push delivery.

    Pub/Sub signs deliveries with a service account of the subscription owner's
    choosing, so a valid Google-issued token is not on its own evidence of
    anything -- anyone can get one. The check that matters is the pair: the
    audience this endpoint declared, and the exact service account allowed to
    deliver to it. Both are configuration, and both are required.

    Google rotates its signing keys, so the JWKS is cached with a TTL *and*
    refetched on an unseen `kid`. Cache-only would break at every rotation;
    fetch-per-request would put a network round trip in front of every email.
    """

    audience: str
    service_account: str
    jwks_url: str = "https://www.googleapis.com/oauth2/v3/certs"
    cache_ttl: float = 3600.0
    # accounts.google.com is issued both with and without the scheme depending
    # on the token's vintage; both are Google.
    issuers: tuple[str, ...] = ("https://accounts.google.com", "accounts.google.com")

    _keys: dict[str, Any] = field(default_factory=dict, repr=False)
    _fetched_at: float = 0.0

    async def verify(self, authorization: str | None) -> dict[str, Any]:
        """Return the token's claims, or raise WebhookVerificationError."""
        import jwt

        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise WebhookVerificationError("no bearer token on the push request")

        try:
            kid = jwt.get_unverified_header(token).get("kid", "")
        except jwt.PyJWTError as exc:
            raise WebhookVerificationError(f"unreadable token header: {exc}") from exc

        key = await self._key_for(kid)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=list(self.issuers),
                options={"require": ["exp", "iat", "aud", "iss"]},
            )
        except jwt.PyJWTError as exc:
            # The message is deliberately the library's: "signature expired" and
            # "audience mismatch" are different operational problems and
            # flattening them to "invalid" costs an hour of debugging later.
            raise WebhookVerificationError(f"token rejected: {exc}") from exc

        email = claims.get("email", "")
        if email != self.service_account:
            raise WebhookVerificationError(
                f"token is for {email!r}, not the configured push service account"
            )
        if not claims.get("email_verified", False):
            raise WebhookVerificationError("token's service account email is unverified")
        return claims

    async def _key_for(self, kid: str):
        import jwt

        if kid not in self._keys or self._stale():
            await self._refresh()
        if kid not in self._keys:
            raise WebhookVerificationError(f"no Google signing key matches kid={kid!r}")
        return jwt.PyJWK(self._keys[kid]).key

    def _stale(self) -> bool:
        return (time.monotonic() - self._fetched_at) > self.cache_ttl

    async def _refresh(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self.jwks_url)
            response.raise_for_status()
            jwks = response.json()
        self._keys = {key["kid"]: key for key in jwks.get("keys", []) if "kid" in key}
        self._fetched_at = time.monotonic()
        log.info("webhook.jwks_refreshed", keys=len(self._keys))


def verify_url_token(request: Request, expected: str) -> None:
    """Check the secret Pub/Sub carries in the push URL's query string.

    Weaker than the OIDC check on its own -- it travels in a URL, which is the
    kind of place secrets get logged -- but it is one string comparison, it is
    independent of Google's keys being reachable, and it is what stops a stray
    request from costing a Gmail history call.
    """
    if not hmac.compare_digest(request.query_params.get("token", ""), expected):
        raise WebhookVerificationError("push URL token missing or wrong")


async def verify_gmail_push(request: Request, verifier: GoogleTokenVerifier | None) -> list[str]:
    """Run every configured Gmail push check. Returns which ones passed.

    Every configured mechanism must pass, not any of them: two checks where the
    weaker one can satisfy the endpoint is one check with extra steps.
    """
    settings = get_settings()
    passed: list[str] = []

    if settings.webhook_token:
        verify_url_token(request, settings.webhook_token)
        passed.append("url_token")

    if verifier is not None:
        await verifier.verify(request.headers.get("authorization"))
        passed.append("oidc")

    if not passed:
        raise WebhookVerificationError("no push verification is configured for this endpoint")
    return passed


# ---------------------------------------------------------------------------
# Notion webhooks
# ---------------------------------------------------------------------------

def verify_notion_signature(body: bytes, header: str | None, secret: str) -> None:
    """Check Notion's HMAC over the delivery.

    The signature covers the *raw* body. Re-serializing the parsed JSON and
    hashing that would be wrong for any payload whose key order or spacing
    differs from what Notion sent, which is most of them -- so the caller reads
    bytes off the request and passes them here untouched.
    """
    if not header:
        raise WebhookVerificationError("no X-Notion-Signature header on the delivery")

    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(header.strip(), expected):
        raise WebhookVerificationError("Notion signature does not match the body")


def is_notion_handshake(payload: dict[str, Any]) -> bool:
    """The one Notion delivery that cannot be signature-checked.

    Notion posts `verification_token` once when a subscription is created, and
    that token *is* the HMAC key -- so this delivery necessarily arrives before
    the secret it would be verified with exists. It is exempt because it has to
    be, and it is safe to exempt because it carries no page id and starts no
    work: the handler echoes the token back and does nothing else.
    """
    return "verification_token" in payload


# ---------------------------------------------------------------------------
# Startup reporting
# ---------------------------------------------------------------------------

def build_google_verifier(settings: Settings | None = None) -> GoogleTokenVerifier | None:
    settings = settings or get_settings()
    if not (settings.google_pubsub_audience and settings.google_pubsub_service_account):
        return None
    return GoogleTokenVerifier(
        audience=settings.google_pubsub_audience,
        service_account=settings.google_pubsub_service_account,
    )


def describe_posture(settings: Settings | None = None) -> dict[str, Any]:
    """What is actually enforced right now, for /health and the startup log.

    Worth surfacing rather than leaving implicit: "I thought the token was set"
    is the normal way a secret ends up not being checked.
    """
    settings = settings or get_settings()
    return {
        "api_auth": "token" if settings.api_token else "none (loopback only)",
        "gmail_webhook": settings.gmail_webhook_verifiers or ["unverified"],
        "notion_webhook": settings.notion_webhook_verifiers or ["unverified"],
        "unverified_webhooks_allowed": settings.webhook_allow_unverified,
    }


def guard_unverified(source: str, configured: list[str]) -> None:
    """Refuse a delivery to an endpoint nobody has configured verification for."""
    if configured or get_settings().webhook_allow_unverified:
        return
    log.warning("webhook.refused_unconfigured", source=source)
    raise HTTPException(
        status_code=503,
        detail=(
            f"{source} webhook verification is not configured. Set the relevant "
            "secret, or set WEBHOOK_ALLOW_UNVERIFIED=true for local testing."
        ),
    )
