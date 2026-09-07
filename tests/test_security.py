"""Request authentication.

These tests are worth more than most of the suite per line, because the failure
mode of this module is silent: an endpoint that stopped checking a signature
behaves exactly like one that is checking it, right up until it does not. So
each mechanism is tested from both sides -- a valid credential is accepted, and
every way of getting it wrong is refused.

The OIDC tests sign real RS256 tokens with a key generated here and serve them
against a JWKS built from that key's public half. Verifying against a fake key
rather than a mocked-out `jwt.decode` is the whole point: what needs testing is
that the audience, the issuer, the expiry, and the service account are actually
checked, and a mock would assert that the mock was called.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from assistant.api.security import (
    GoogleTokenVerifier,
    WebhookVerificationError,
    describe_posture,
    enforce_bind,
    guard_unverified,
    is_notion_handshake,
    presented_token,
    require_api_token,
    verify_notion_signature,
    verify_url_token,
)
from assistant.config import Settings, get_settings

KID = "test-key-1"
AUDIENCE = "https://assistant.example/webhooks/gmail"
SERVICE_ACCOUNT = "pubsub-pusher@example.iam.gserviceaccount.com"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def verifier(signing_key):
    """A verifier whose JWKS is this test's key, with the network cut out."""
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})

    google = GoogleTokenVerifier(audience=AUDIENCE, service_account=SERVICE_ACCOUNT)

    async def fake_refresh():
        google._keys = {KID: jwk}
        google._fetched_at = time.monotonic()

    google._refresh = fake_refresh
    return google


def make_token(signing_key, **claims):
    payload = {
        "iss": "https://accounts.google.com",
        "aud": AUDIENCE,
        "email": SERVICE_ACCOUNT,
        "email_verified": True,
        "iat": int(time.time()) - 5,
        "exp": int(time.time()) + 600,
    }
    payload.update(claims)
    return jwt.encode(payload, signing_key, algorithm="RS256", headers={"kid": KID})


def request_with(path="/queue", headers=None, query=None):
    """The two attributes the security helpers actually read off a request."""
    return SimpleNamespace(
        url=SimpleNamespace(path=path),
        headers={k.lower(): v for k, v in (headers or {}).items()},
        query_params=query or {},
    )


# ---------------------------------------------------------------------------
# Google Pub/Sub OIDC
# ---------------------------------------------------------------------------

async def test_a_properly_signed_push_token_is_accepted(verifier, signing_key):
    claims = await verifier.verify(f"Bearer {make_token(signing_key)}")

    assert claims["email"] == SERVICE_ACCOUNT


async def test_a_token_for_another_audience_is_rejected(verifier, signing_key):
    """A Google-signed token is not evidence on its own -- anyone can get one."""
    token = make_token(signing_key, aud="https://someone-elses-service.example")

    with pytest.raises(WebhookVerificationError, match="rejected"):
        await verifier.verify(f"Bearer {token}")


async def test_a_token_from_another_service_account_is_rejected(verifier, signing_key):
    """Correct audience, valid signature, wrong sender: still not our pusher."""
    token = make_token(signing_key, email="someone-else@example.iam.gserviceaccount.com")

    with pytest.raises(WebhookVerificationError, match="not the configured push service account"):
        await verifier.verify(f"Bearer {token}")


async def test_an_expired_token_is_rejected(verifier, signing_key):
    token = make_token(signing_key, exp=int(time.time()) - 30, iat=int(time.time()) - 600)

    with pytest.raises(WebhookVerificationError, match="rejected"):
        await verifier.verify(f"Bearer {token}")


async def test_a_token_signed_by_a_different_key_is_rejected(verifier):
    """The check that makes the rest of them mean anything."""
    impostor = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = make_token(impostor)

    with pytest.raises(WebhookVerificationError, match="rejected"):
        await verifier.verify(f"Bearer {token}")


async def test_an_unverified_service_account_email_is_rejected(verifier, signing_key):
    token = make_token(signing_key, email_verified=False)

    with pytest.raises(WebhookVerificationError, match="unverified"):
        await verifier.verify(f"Bearer {token}")


async def test_a_missing_bearer_token_is_rejected(verifier):
    with pytest.raises(WebhookVerificationError, match="no bearer token"):
        await verifier.verify(None)


async def test_an_unknown_key_id_triggers_one_refresh_then_gives_up(verifier, signing_key):
    """Google rotates keys, so an unseen kid is a reason to refetch -- once.

    Retrying forever on an unknown kid would turn a malformed token into a
    denial-of-service against Google's JWKS endpoint.
    """
    refreshes = []
    original = verifier._refresh

    async def counting_refresh():
        refreshes.append(1)
        await original()

    verifier._refresh = counting_refresh
    token = jwt.encode({"iss": "x"}, signing_key, algorithm="RS256", headers={"kid": "unknown"})

    with pytest.raises(WebhookVerificationError, match="no Google signing key"):
        await verifier.verify(f"Bearer {token}")
    assert len(refreshes) == 1


# ---------------------------------------------------------------------------
# Notion HMAC
# ---------------------------------------------------------------------------

def signed(body: bytes, secret: str = "s3cret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_a_signature_over_the_exact_bytes_is_accepted():
    body = b'{"entity": {"id": "page-1"}}'

    verify_notion_signature(body, signed(body), "s3cret")  # does not raise


def test_a_signature_over_reserialized_json_does_not_match():
    """Why the handler reads raw bytes: the same object, spaced differently.

    `json.dumps` of the parsed body produces different bytes from what Notion
    sent, and hashing those would reject every genuine delivery.
    """
    sent = b'{"entity": {"id": "page-1"}}'
    reserialized = json.dumps(json.loads(sent), separators=(",", ":")).encode()

    assert reserialized != sent
    with pytest.raises(WebhookVerificationError):
        verify_notion_signature(reserialized, signed(sent), "s3cret")


def test_a_delivery_with_no_signature_header_is_rejected():
    with pytest.raises(WebhookVerificationError, match="no X-Notion-Signature"):
        verify_notion_signature(b"{}", None, "s3cret")


def test_a_signature_made_with_the_wrong_secret_is_rejected():
    body = b"{}"

    with pytest.raises(WebhookVerificationError, match="does not match"):
        verify_notion_signature(body, signed(body, "wrong-secret"), "s3cret")


def test_the_subscription_handshake_is_recognised():
    """It cannot be signed -- the token it carries is the key it would be signed with."""
    assert is_notion_handshake({"verification_token": "tok"})
    assert not is_notion_handshake({"entity": {"id": "page-1"}})


# ---------------------------------------------------------------------------
# URL token
# ---------------------------------------------------------------------------

def test_the_url_token_must_match():
    verify_url_token(request_with(query={"token": "abc"}), "abc")  # does not raise

    with pytest.raises(WebhookVerificationError):
        verify_url_token(request_with(query={"token": "abd"}), "abc")
    with pytest.raises(WebhookVerificationError):
        verify_url_token(request_with(query={}), "abc")


# ---------------------------------------------------------------------------
# Shared-secret API auth
# ---------------------------------------------------------------------------

@pytest.fixture
def with_api_token(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret-token")
    get_settings.cache_clear()
    yield "s3cret-token"
    get_settings.cache_clear()


def test_a_request_carrying_the_token_passes(with_api_token):
    require_api_token(request_with(headers={"Authorization": f"Bearer {with_api_token}"}))
    require_api_token(request_with(headers={"X-API-Token": with_api_token}))


def test_a_request_without_the_token_is_refused(with_api_token):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        require_api_token(request_with())
    assert raised.value.status_code == 401


def test_a_request_with_the_wrong_token_is_refused(with_api_token):
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        require_api_token(request_with(headers={"Authorization": "Bearer nearly-s3cret-token"}))


def test_the_dashboard_shell_and_health_stay_open(with_api_token):
    """They carry nothing, and /health has to answer before anyone is authenticated."""
    require_api_token(request_with(path="/"))
    require_api_token(request_with(path="/health"))


def test_webhooks_are_exempt_because_they_authenticate_differently(with_api_token):
    """Google cannot learn this deployment's shared secret; demanding it there
    would only mean turning the check off."""
    require_api_token(request_with(path="/webhooks/gmail"))


def test_both_header_forms_are_read():
    assert presented_token(request_with(headers={"Authorization": "Bearer abc"})) == "abc"
    assert presented_token(request_with(headers={"X-API-Token": "abc"})) == "abc"
    assert presented_token(request_with(headers={"Authorization": "Basic abc"})) == ""
    assert presented_token(request_with()) == ""


# ---------------------------------------------------------------------------
# Startup posture
# ---------------------------------------------------------------------------

def test_an_unauthenticated_api_may_bind_loopback():
    enforce_bind(Settings(api_token="", api_host="127.0.0.1"))
    enforce_bind(Settings(api_token="", api_host="localhost"))


def test_an_unauthenticated_api_refuses_a_reachable_interface():
    """The mistake is a configuration one, so it stops startup rather than
    showing up later in an access log."""
    with pytest.raises(RuntimeError, match="refusing to bind"):
        enforce_bind(Settings(api_token="", api_host="0.0.0.0"))


def test_a_token_permits_binding_anywhere():
    enforce_bind(Settings(api_token="set", api_host="0.0.0.0"))


def test_an_unconfigured_webhook_refuses_deliveries():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        guard_unverified("gmail", [])
    assert raised.value.status_code == 503


def test_the_escape_hatch_is_explicit(monkeypatch):
    monkeypatch.setenv("WEBHOOK_ALLOW_UNVERIFIED", "true")
    get_settings.cache_clear()
    try:
        guard_unverified("gmail", [])  # does not raise
    finally:
        get_settings.cache_clear()


def test_the_posture_reports_what_is_enforced():
    posture = describe_posture(
        Settings(api_token="", webhook_token="x", notion_webhook_secret="")
    )

    assert posture["api_auth"].startswith("none")
    assert posture["gmail_webhook"] == ["url_token"]
    assert posture["notion_webhook"] == ["unverified"]
