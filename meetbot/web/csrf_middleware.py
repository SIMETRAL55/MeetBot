"""
CSRF protection middleware for MeetBot's REST API.

Strategy: Origin-header validation for state-changing requests that don't carry
a JWT Bearer token. Requests with a valid Bearer token are inherently CSRF-safe
because browsers cannot auto-attach custom Authorization headers cross-site.

This means:
- All authenticated API calls (JWT Bearer) bypass the check — no friction.
- Public endpoints called without Bearer are checked against allowed origins.
- Programmatic clients (curl, mobile) can still reach public endpoints as long
  as they include an allowed Origin header or supply a Bearer token.
"""

import logging
from urllib.parse import urlparse

from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.responses import Response

logger = logging.getLogger(__name__)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class CSRFMiddleware:
    """
    Pure ASGI middleware — more reliable than BaseHTTPMiddleware in NiceGUI's
    Starlette stack because it never buffers the request body or response.
    """

    def __init__(self, app: ASGIApp, *, allowed_origins: frozenset[str]) -> None:
        self.app = app
        self.allowed_origins = allowed_origins

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            method: str = scope["method"]
            path: str = scope["path"]

            if method not in _SAFE_METHODS and path.startswith("/api/"):
                headers: dict[bytes, bytes] = {
                    k.lower(): v for k, v in scope.get("headers", [])
                }
                auth = headers.get(b"authorization", b"").decode("latin-1")

                if not auth.startswith("Bearer "):
                    origin = headers.get(b"origin", b"").decode("latin-1")

                    if not origin:
                        referer = headers.get(b"referer", b"").decode("latin-1")
                        if referer and "://" in referer:
                            p = urlparse(referer)
                            origin = f"{p.scheme}://{p.netloc}"

                    if origin not in self.allowed_origins:
                        logger.warning(
                            "CSRF check failed — origin=%r path=%s method=%s",
                            origin,
                            path,
                            method,
                        )
                        response = Response("CSRF check failed", status_code=403)
                        await response(scope, receive, send)
                        return

        await self.app(scope, receive, send)
