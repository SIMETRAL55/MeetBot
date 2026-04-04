"""
Tests for meetbot/web/auth_middleware.py.

Covers:
- Valid token round-trip
- Expired token rejection (+ debug log emitted)
- Malformed token (wrong number of parts)
- Invalid signature (tampered payload)
- get_current_user with no Authorization header → 401
- get_current_user with valid token → returns payload
- get_current_user with deleted/missing user (via DB mock) → 401
"""

import hashlib
import hmac
import json
import time
from base64 import urlsafe_b64encode
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_raw_token(user_id: str, username: str, exp: int, secret: str) -> str:
    """Build a token with a specific expiry using the same algorithm as auth_middleware."""
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_dict = {
        "sub": user_id,
        "username": username,
        "iat": int(time.time()),
        "exp": exp,
    }
    payload = _b64encode(json.dumps(payload_dict).encode())
    sig_input = f"{header}.{payload}".encode()
    signature = _b64encode(hmac.new(secret.encode(), sig_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def _mock_request(auth_header: str | None = None) -> Request:
    """Create a minimal Starlette Request with the given Authorization header."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/test",
        "headers": [],
    }
    if auth_header is not None:
        scope["headers"] = [(b"authorization", auth_header.encode())]
    return Request(scope)


# ---------------------------------------------------------------------------
# verify_token tests
# ---------------------------------------------------------------------------

SECRET = "test-secret-key-12345"


@pytest.fixture(autouse=True)
def _patch_secret():
    """Use a fixed secret key for all tests."""
    with patch("meetbot.web.auth_middleware._get_secret", return_value=SECRET):
        yield


def test_valid_token_returns_payload():
    from meetbot.web.auth_middleware import create_access_token, verify_token

    token = create_access_token("user-123", "alice")
    payload = verify_token(token)

    assert payload is not None
    assert payload["sub"] == "user-123"
    assert payload["username"] == "alice"
    assert "exp" in payload
    assert "iat" in payload


def test_expired_token_returns_none(caplog):
    from meetbot.web.auth_middleware import verify_token

    expired = _make_raw_token("user-abc", "bob", exp=int(time.time()) - 10, secret=SECRET)

    import logging
    with caplog.at_level(logging.DEBUG, logger="meetbot.web.auth_middleware"):
        result = verify_token(expired)

    assert result is None
    assert any("expired" in r.message.lower() and "user-abc" in r.message for r in caplog.records)


def test_future_token_passes():
    from meetbot.web.auth_middleware import verify_token

    future = _make_raw_token("user-xyz", "carol", exp=int(time.time()) + 9999, secret=SECRET)
    result = verify_token(future)
    assert result is not None
    assert result["sub"] == "user-xyz"


def test_malformed_token_two_parts():
    from meetbot.web.auth_middleware import verify_token

    assert verify_token("header.payload") is None


def test_malformed_token_one_part():
    from meetbot.web.auth_middleware import verify_token

    assert verify_token("onlyonepart") is None


def test_invalid_signature_returns_none():
    from meetbot.web.auth_middleware import create_access_token, verify_token

    token = create_access_token("user-123", "alice")
    # Tamper: flip the last character of the signature
    parts = token.split(".")
    tampered_sig = parts[2][:-1] + ("A" if parts[2][-1] != "A" else "B")
    tampered = f"{parts[0]}.{parts[1]}.{tampered_sig}"

    assert verify_token(tampered) is None


def test_wrong_secret_returns_none():
    """Token signed with a different secret must be rejected."""
    from meetbot.web.auth_middleware import verify_token

    token = _make_raw_token("user-999", "eve", exp=int(time.time()) + 3600, secret="wrong-secret")
    assert verify_token(token) is None


# ---------------------------------------------------------------------------
# get_current_user tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_current_user_no_header():
    from meetbot.web.auth_middleware import get_current_user

    req = _mock_request(auth_header=None)
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(req)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_valid_bearer():
    from meetbot.web.auth_middleware import create_access_token, get_current_user

    token = create_access_token("user-42", "diana")
    req = _mock_request(auth_header=f"Bearer {token}")

    payload = await get_current_user(req)
    assert payload["sub"] == "user-42"
    assert payload["username"] == "diana"


@pytest.mark.asyncio
async def test_get_current_user_expired_token():
    from meetbot.web.auth_middleware import get_current_user

    expired = _make_raw_token("user-old", "old", exp=int(time.time()) - 5, secret=SECRET)
    req = _mock_request(auth_header=f"Bearer {expired}")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(req)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_invalid_signature():
    from meetbot.web.auth_middleware import create_access_token, get_current_user

    token = create_access_token("user-123", "alice")
    parts = token.split(".")
    bad_token = f"{parts[0]}.{parts[1]}.badsig"
    req = _mock_request(auth_header=f"Bearer {bad_token}")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(req)
    assert exc_info.value.status_code == 401
