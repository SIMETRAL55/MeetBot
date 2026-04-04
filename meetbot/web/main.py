"""
MeetBot Web Application — NiceGUI entry point.

Initializes the NiceGUI application with:
- Database setup
- Session middleware for authentication
- Route registration for all pages
- Job queue startup/shutdown
- Real-time progress WebSocket integration
"""

import asyncio
import logging
import os
from pathlib import Path

# Disable ChromaDB's anonymous PostHog telemetry before any chromadb import.
# On air-gapped (offline) hosts this prevents a flood of DNS-resolution
# errors in the logs caused by background threads trying to reach
# us.i.posthog.com.  Setting the env var here (before chromadb is imported
# elsewhere in the process) is the most reliable disablement method.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

from nicegui import app, ui

from ..config import settings
from ..db.database import init_db

logger = logging.getLogger(__name__)


def _setup_database() -> None:
    """Initialize the SQLite database and tables."""
    init_db()
    logger.info("Database initialized")


def _setup_upload_dirs() -> None:
    """Ensure upload and results directories exist."""
    base = Path(settings.OUTPUT_DIR).parent
    (base / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Upload and results directories ready")


def _register_pages() -> None:
    """Register all NiceGUI pages and WebSocket endpoints."""
    # Import pages to trigger @ui.page registration
    from .pages import login  # noqa: F401
    from .pages import dashboard  # noqa: F401
    from .pages import upload  # noqa: F401
    from .pages import job_detail  # noqa: F401
    from .pages import query  # noqa: F401

    # Register FastAPI WebSocket endpoint for real-time job progress streaming.
    # NiceGUI exposes add_api_websocket_route() directly on its app object,
    # so we can register the route without reaching into app.fastapi_app.
    from .ws import ws_job_progress
    app.add_api_websocket_route("/ws/jobs/{job_id}", ws_job_progress)

    # Register REST API endpoints.
    from .api import (
        api_health,
        api_list_jobs,
        api_upload_job,
        api_delete_job,
        api_update_segment,
        api_job_status,
        api_job_query,
        api_job_download,
        api_job_audio,
        api_job_reindex,
        api_job_cancel,
        api_job_restart,
        api_auth_login,
        api_auth_register,
        api_firebase_login,
        api_forgot_password,
        api_reset_password,
        api_verify_register_otp,
        api_verify_reset_otp,
        api_resend_otp,
        api_get_chat_history,
        api_build_pageindex,
        api_cancel_pageindex,
        api_get_pageindex_tree,
        api_get_settings,
        api_update_setting,
        api_get_me,
        api_rename_speaker,
    )
    app.add_api_route("/api/health",                   api_health,       methods=["GET"])
    app.add_api_route("/api/auth/login",             api_auth_login,   methods=["POST"])
    app.add_api_route("/api/auth/register",          api_auth_register, methods=["POST"])
    app.add_api_route("/api/auth/firebase-login",    api_firebase_login, methods=["POST"])
    app.add_api_route("/api/auth/forgot-password",        api_forgot_password,        methods=["POST"])
    app.add_api_route("/api/auth/reset-password",         api_reset_password,         methods=["POST"])
    app.add_api_route("/api/auth/verify-register-otp",    api_verify_register_otp,    methods=["POST"])
    app.add_api_route("/api/auth/verify-reset-otp",       api_verify_reset_otp,       methods=["POST"])
    app.add_api_route("/api/auth/resend-otp",             api_resend_otp,             methods=["POST"])
    app.add_api_route("/api/jobs",                   api_list_jobs,    methods=["GET"])
    app.add_api_route("/api/jobs/upload",            api_upload_job,   methods=["POST"])
    app.add_api_route("/api/jobs/{job_id}",          api_delete_job,   methods=["DELETE"])
    app.add_api_route("/api/jobs/{job_id}/segments/{segment_id}", api_update_segment, methods=["PUT"])
    app.add_api_route("/api/jobs/{job_id}/status",   api_job_status,   methods=["GET"])
    app.add_api_route("/api/jobs/{job_id}/query",    api_job_query,    methods=["POST"])
    app.add_api_route("/api/jobs/{job_id}/download", api_job_download, methods=["GET"])
    app.add_api_route("/api/jobs/{job_id}/audio",    api_job_audio,    methods=["GET"])
    app.add_api_route("/api/jobs/{job_id}/reindex",  api_job_reindex,  methods=["POST"])
    app.add_api_route("/api/jobs/{job_id}/cancel",   api_job_cancel,   methods=["POST"])
    app.add_api_route("/api/jobs/{job_id}/restart",  api_job_restart,  methods=["POST"])
    app.add_api_route("/api/jobs/{job_id}/chat/history", api_get_chat_history, methods=["GET"])
    app.add_api_route("/api/jobs/{job_id}/build-pageindex", api_build_pageindex, methods=["POST"])
    app.add_api_route("/api/jobs/{job_id}/cancel-pageindex", api_cancel_pageindex, methods=["POST"])
    app.add_api_route("/api/jobs/{job_id}/pageindex-tree", api_get_pageindex_tree, methods=["GET"])
    app.add_api_route("/api/jobs/{job_id}/rename-speaker", api_rename_speaker,    methods=["POST"])
    app.add_api_route("/api/settings",       api_get_settings,    methods=["GET"])
    app.add_api_route("/api/settings/{key}", api_update_setting,  methods=["PUT"])
    app.add_api_route("/api/me",             api_get_me,          methods=["GET"])

    # Register WebSocket endpoint for streaming chat (RAG Q&A).
    from .ws_chat import ws_chat
    app.add_api_websocket_route("/ws/chat/{job_id}", ws_chat)

    logger.info("All pages and WebSocket endpoints registered")


async def _startup() -> None:
    """Application startup hook — start job queue worker."""
    from ..workers.queue import job_queue

    loop = asyncio.get_event_loop()
    job_queue.start(loop)
    logger.info("Job queue started")


async def _shutdown() -> None:
    """Application shutdown hook — stop job queue worker."""
    from ..workers.queue import job_queue

    job_queue.stop()
    logger.info("Job queue stopped")


def _auth_middleware(client) -> None:
    """
    Authentication middleware (NiceGUI on_connect handler).

    Receives a NiceGUI Client object (WebSocket client), not a Starlette
    Request. Use client.request.url.path to get the current page path.

    Redirects unauthenticated users to login page.
    Allows access to login page and static files without auth.
    """
    # Pages that don't require authentication
    PUBLIC_PATHS = {"/login", "/_nicegui", "/favicon.ico"}

    # client is a NiceGUI Client; the underlying Starlette Request is at
    # client.request — accessing .url.path gives the HTTP path.
    try:
        path = client.request.url.path
    except Exception:
        # request may be None in rare edge cases (e.g. prefetch); allow through
        return

    if any(path.startswith(p) for p in PUBLIC_PATHS):
        return

    user_id = app.storage.user.get("user_id")
    if user_id is None:
        return ui.navigate.to("/login")


def _apply_db_settings() -> None:
    """Load AppSetting overrides from the DB and hot-patch the live settings singleton."""
    from ..db.database import get_session
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        n = settings.apply_db_overrides(db)
        if n:
            logger.info("Applied %d DB setting override(s)", n)
    except Exception as exc:
        # DB may not have the table yet on first run before migration; non-fatal.
        logger.debug("_apply_db_settings: skipped — %s", exc)
    finally:
        db.close()


def create_app() -> None:
    """Configure and create the NiceGUI application."""
    from ..logging_conf import setup_logging
    setup_logging(level="INFO")

    logger.info("Configuring MeetBot web application...")

    # Setup
    _setup_database()
    _apply_db_settings()
    _setup_upload_dirs()

    # Rate limiting (slowapi)
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from .api import limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Security Headers Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as _StarletteRequest
    from starlette.responses import Response as _StarletteResponse

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: _StarletteRequest, call_next) -> _StarletteResponse:
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.googletagmanager.com; "
                "connect-src 'self' wss: ws: https://identitytoolkit.googleapis.com; "
                "img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline';"
            )
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    # CSRF Middleware (pure ASGI — avoids BaseHTTPMiddleware reliability issues)
    from .csrf_middleware import CSRFMiddleware
    _csrf_allowed_origins = frozenset({
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        f"http://localhost:{settings.WEB_PORT}",
        f"http://127.0.0.1:{settings.WEB_PORT}",
    })
    app.add_middleware(CSRFMiddleware, allowed_origins=_csrf_allowed_origins)

    # CORS Middleware
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication middleware
    app.on_connect(_auth_middleware)

    # Lifecycle hooks
    app.on_startup(_startup)
    app.on_shutdown(_shutdown)

    # Register pages
    _register_pages()

    logger.info("MeetBot web application configured")


def start_server(
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    """
    Start the MeetBot web server.

    Args:
        host: Host to bind to (default: from config).
        port: Port to listen on (default: from config).
        reload: Enable auto-reload for development.
    """
    create_app()

    _host = host or settings.WEB_HOST
    _port = port or settings.WEB_PORT

    logger.info(f"Starting MeetBot on http://{_host}:{_port}")

    ui.run(
        host=_host,
        port=_port,
        title="MeetBot",
        favicon="🎙️",
        storage_secret=settings.WEB_STORAGE_SECRET,
        reload=reload,
        show=False,
    )
