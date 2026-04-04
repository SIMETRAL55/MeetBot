"""
Tests for meetbot/web/csrf_middleware.py.

Uses a minimal Starlette ASGI app with the middleware applied so the tests
are self-contained and don't require the full NiceGUI stack.
"""

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from meetbot.web.csrf_middleware import CSRFMiddleware

_ALLOWED = frozenset({
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
})


# ---------------------------------------------------------------------------
# Minimal app fixture
# ---------------------------------------------------------------------------

def _ok(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def _make_client() -> TestClient:
    app = Starlette(routes=[
        Route("/api/login", _ok, methods=["POST"]),
        Route("/api/data",  _ok, methods=["POST", "DELETE"]),
        Route("/api/info",  _ok, methods=["GET"]),
    ])
    app.add_middleware(CSRFMiddleware, allowed_origins=_ALLOWED)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    return _make_client()


# ---------------------------------------------------------------------------
# Safe methods (GET/HEAD/OPTIONS) — always allowed regardless of origin
# ---------------------------------------------------------------------------

def test_get_no_origin_allowed(client: TestClient):
    r = client.get("/api/info")
    assert r.status_code == 200


def test_get_evil_origin_allowed(client: TestClient):
    """GET is safe — CSRF check must NOT run."""
    r = client.get("/api/info", headers={"Origin": "http://evil.com"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Mutating requests with allowed origins — must pass
# ---------------------------------------------------------------------------

def test_post_localhost_3000_allowed(client: TestClient):
    r = client.post("/api/login", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 200


def test_post_127_allowed(client: TestClient):
    r = client.post("/api/login", headers={"Origin": "http://127.0.0.1:3000"})
    assert r.status_code == 200


def test_delete_allowed_origin(client: TestClient):
    r = client.delete("/api/data", headers={"Origin": "http://localhost:8080"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Mutating requests with evil / missing origin — must be blocked
# ---------------------------------------------------------------------------

def test_post_evil_origin_blocked(client: TestClient):
    r = client.post("/api/login", headers={"Origin": "http://evil.com"})
    assert r.status_code == 403


def test_post_no_origin_no_bearer_blocked(client: TestClient):
    """No origin and no Bearer token — must be blocked."""
    r = client.post("/api/login")
    assert r.status_code == 403


def test_delete_evil_origin_blocked(client: TestClient):
    r = client.delete("/api/data", headers={"Origin": "http://attacker.example"})
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Bearer token — bypasses CSRF check regardless of origin
# ---------------------------------------------------------------------------

def test_post_bearer_bypasses_evil_origin(client: TestClient):
    """JWT Bearer is CSRF-safe — middleware must not block it."""
    r = client.post(
        "/api/login",
        headers={
            "Origin": "http://evil.com",
            "Authorization": "Bearer some.jwt.token",
        },
    )
    assert r.status_code == 200


def test_post_bearer_no_origin_allowed(client: TestClient):
    """curl / mobile with Bearer and no Origin must get through."""
    r = client.post(
        "/api/data",
        headers={"Authorization": "Bearer some.jwt.token"},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Referer fallback (when Origin header absent)
# ---------------------------------------------------------------------------

def test_post_referer_allowed_origin(client: TestClient):
    """Referer from an allowed origin substitutes for missing Origin."""
    r = client.post(
        "/api/login",
        headers={"Referer": "http://localhost:3000/login"},
    )
    assert r.status_code == 200


def test_post_referer_evil_origin_blocked(client: TestClient):
    r = client.post(
        "/api/login",
        headers={"Referer": "http://evil.com/csrf-page"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Non-/api/ paths — middleware must not interfere
# ---------------------------------------------------------------------------

def test_non_api_path_not_affected():
    """Requests to paths outside /api/ are never checked."""
    app = Starlette(routes=[Route("/login", _ok, methods=["POST"])])
    app.add_middleware(CSRFMiddleware, allowed_origins=_ALLOWED)
    client = TestClient(app, raise_server_exceptions=True)
    # No origin, no bearer — but path is /login not /api/*, so it must pass
    r = client.post("/login")
    assert r.status_code == 200
