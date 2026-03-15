"""
JWT-based authentication middleware for the REST API.

Provides:
- JWT token generation and validation
- FastAPI dependency for protecting REST endpoints
- User extraction from Authorization header

Tokens are signed with the ``WEB_SECRET_KEY`` setting using HMAC-SHA256.
"""

import hashlib
import hmac
import json
import logging
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# Token lifetime: 24 hours
TOKEN_LIFETIME_SECONDS = 86400


def _b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return urlsafe_b64decode(s)


def _get_secret() -> str:
    from ..config import settings
    return settings.WEB_SECRET_KEY


def create_access_token(user_id: str, username: str) -> str:
    """
    Create a JWT-like access token (HMAC-SHA256 signed).

    Uses a simple 3-part format: header.payload.signature
    No external JWT library required.

    Args:
        user_id: User UUID.
        username: Username string.

    Returns:
        Signed token string.
    """
    secret = _get_secret()

    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_dict = {
        "sub": user_id,
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_LIFETIME_SECONDS,
    }
    payload = _b64encode(json.dumps(payload_dict).encode())

    signature_input = f"{header}.{payload}".encode()
    signature = _b64encode(
        hmac.new(secret.encode(), signature_input, hashlib.sha256).digest()
    )

    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode a token.

    Args:
        token: The token string to verify.

    Returns:
        Decoded payload dict if valid, None otherwise.
    """
    secret = _get_secret()

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        expected_sig = _b64encode(
            hmac.new(
                secret.encode(),
                f"{header_b64}.{payload_b64}".encode(),
                hashlib.sha256,
            ).digest()
        )
        if not hmac.compare_digest(signature_b64, expected_sig):
            return None

        # Decode payload
        payload = json.loads(_b64decode(payload_b64))

        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None


async def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency that extracts and validates the user from the
    Authorization header.

    Usage:
        @app.get("/api/protected")
        async def protected(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}

    Returns:
        Decoded token payload dict with keys: sub, username, iat, exp.

    Raises:
        HTTPException 401 if token is missing, invalid, or expired.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Support "Bearer <token>" format
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = auth_header

    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def get_optional_user(request: Request) -> Optional[dict]:
    """
    Extract user from Authorization header without raising on failure.

    Returns None if no valid token is present (for endpoints that work
    both authenticated and unauthenticated).
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    token = auth_header[7:] if auth_header.startswith("Bearer ") else auth_header
    return verify_token(token)
