"""
REST API endpoints for MeetBot.

Registered on the NiceGUI FastAPI app via app.add_api_route().

# NOTE: Every function defined here MUST also be registered in main.py via
# app.add_api_route(). Forgetting main.py means the endpoint silently doesn't
# exist — no error is raised, the route simply isn't reachable.

Routes
------
GET  /api/jobs
    Returns all jobs for the first user (stubbed for single user usage).

POST /api/jobs/upload
    Accepts an audio file via multipart/form-data and starts processing.

GET  /api/jobs/{job_id}/status
    Returns job status, progress, and db_dir.

POST /api/jobs/{job_id}/query
    Body: {"q": "...", "llm_mode": "local" | "hf"}
    Returns the RAG answer and source segments for a completed job.
    ``llm_mode`` defaults to "local".

GET  /api/jobs/{job_id}/download?type=transcription|diarization|aligned
    Returns the requested JSON file as an attachment download.

DELETE /api/jobs/{job_id}
    Deletes the job and all associated files.
"""

import json as _json
import logging
from typing import Literal, Optional

from fastapi import HTTPException, Query as QueryParam, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from ..config import settings as _settings

# Module-level rate limiter — registered on app.state in main.py
limiter = Limiter(key_func=get_remote_address)

from ..db.database import get_session
from ..db.crud import get_job, build_aligned_json_from_db
from ..db.models import JobStatus
from datetime import timezone as _tz

logger = logging.getLogger(__name__)


def _utc_iso(dt) -> "str | None":
    """Serialise a datetime to an ISO-8601 string with explicit UTC offset.

    SQLite stores datetimes as plain text, so SQLAlchemy returns **naive**
    datetime objects (no tzinfo).  Without an explicit offset, JavaScript's
    ``new Date("2026-03-11T01:00:00")`` treats the value as *local* time,
    causing the "9 hours ago" timestamp regression on UTC+9 systems.

    Adding ``.replace(tzinfo=timezone.utc)`` ensures the output includes
    ``+00:00`` so browsers convert UTC → local time correctly.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz.utc)
    return dt.isoformat()


# ── Request / response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    q: str
    llm_mode: Literal["local", "hf"] = Field(
        "local",
        description="LLM backend: 'local' for AWQ model via transformers; 'hf' for HuggingFace Inference API",
    )

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    email: str

class FirebaseLoginRequest(BaseModel):
    id_token: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str

class VerifyOtpRequest(BaseModel):
    email: str
    otp: str

class ResendOtpRequest(BaseModel):
    email: str
    purpose: str  # "register" | "reset"


# ── Health Check ──────────────────────────────────────────────────────────────

async def api_health() -> JSONResponse:
    """Health check endpoint for container orchestration (K8s, Docker, etc.).

    Returns system status including database connectivity, GPU availability,
    queue state, and pipeline metrics.
    """
    import time
    from ..workers.queue import job_queue
    from ..logging_conf import pipeline_metrics

    health = {
        "status": "healthy",
        "timestamp": time.time(),
        "queue_size": job_queue.queue_size,
        "current_job": job_queue.current_job_id,
        "metrics": pipeline_metrics.get_summary(),
    }

    # Check database connectivity
    try:
        SessionLocal = get_session()
        db = SessionLocal()
        try:
            db.execute(__import__("sqlalchemy").text("SELECT 1"))
            health["database"] = "connected"
        finally:
            db.close()
    except Exception as e:
        health["database"] = f"error: {e}"
        health["status"] = "degraded"

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            health["gpu"] = {
                "available": True,
                "free_mb": round(free / 1024 / 1024),
                "total_mb": round(total / 1024 / 1024),
            }
        else:
            health["gpu"] = {"available": False}
    except ImportError:
        health["gpu"] = {"available": False}

    return JSONResponse(health)


# ── Auth Handlers ─────────────────────────────────────────────────────────────

async def api_auth_login(body: LoginRequest) -> JSONResponse:
    """Authenticate user and return user info with access token."""
    from ..web.auth import verify_password, check_account_lockout, record_failed_login, record_successful_login
    from ..web.auth_middleware import create_access_token
    from ..db.crud import get_user_by_username, get_user_by_email

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        # Support login by username OR email
        user = get_user_by_username(db, body.username)
        if user is None:
            user = get_user_by_email(db, body.username)

        if user is None:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # Block unverified email accounts (local auth only — Firebase users skip this)
        if user.email and not user.email_verified and not user.firebase_uid:
            return JSONResponse(
                {"detail": "email_not_verified", "email": user.email},
                status_code=403,
            )

        # Check account lockout before password verification
        locked_until = check_account_lockout(user)
        if locked_until is not None:
            raise HTTPException(
                status_code=423,
                detail=f"Account locked until {locked_until.isoformat()}. Too many failed login attempts.",
            )

        if not verify_password(body.password, user.password_hash):
            record_failed_login(db, user)
            raise HTTPException(status_code=401, detail="Invalid username or password")

        record_successful_login(db, user)
        db.refresh(user)
        db.expunge(user)
        token = create_access_token(user.id, user.username)
        return JSONResponse({
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name or user.username,
            "is_admin": user.is_admin,
            "access_token": token,
            "token_type": "bearer",
        })
    finally:
        db.close()

async def api_auth_register(body: RegisterRequest) -> JSONResponse:
    """Register a new user account and send OTP verification email."""
    import re
    from ..web.auth import hash_password, generate_otp, store_otp, send_otp_email
    from ..db.crud import create_user, get_user_by_username, get_user_by_email

    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", body.email):
        raise HTTPException(status_code=422, detail="Invalid email address")

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        if get_user_by_username(db, body.username):
            raise HTTPException(status_code=409, detail="Username already exists")
        if get_user_by_email(db, body.email):
            raise HTTPException(status_code=409, detail="Email already registered")

        raw_otp, otp_hash = generate_otp()

        # Send email BEFORE creating user — if SMTP fails, no orphan account is created
        try:
            send_otp_email(body.email, raw_otp, "register")
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            logger.error("register: OTP email failed: %s", exc)
            raise HTTPException(status_code=503, detail="Failed to send verification email")

        hashed = hash_password(body.password)
        user = create_user(db, body.username, hashed, display_name=body.display_name)
        user.email = body.email
        user.email_verified = False
        db.add(user)
        db.commit()

        store_otp(db, user, otp_hash, "register")

        return JSONResponse({"message": "otp_sent", "email": body.email}, status_code=201)
    finally:
        db.close()


@limiter.limit("20/minute")
async def api_firebase_login(request: "Request", body: FirebaseLoginRequest) -> JSONResponse:
    """Authenticate via Firebase ID token and return MeetBot access token."""
    from ..web.auth import verify_firebase_token, get_or_create_firebase_user
    from ..web.auth_middleware import create_access_token

    claims = verify_firebase_token(body.id_token)
    if claims is None:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    if not claims.get("email_verified", False):
        raise HTTPException(status_code=403, detail="EMAIL_NOT_VERIFIED")

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_or_create_firebase_user(db, claims)
        if user is None:
            raise HTTPException(status_code=500, detail="Failed to create user from Firebase token")

        token = create_access_token(user.id, user.username)
        return JSONResponse({
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name or user.username,
            "is_admin": user.is_admin,
            "access_token": token,
            "token_type": "bearer",
        })
    finally:
        db.close()


@limiter.limit("5/hour")
async def api_forgot_password(request: "Request", body: ForgotPasswordRequest) -> JSONResponse:
    """Initiate password reset via OTP — always returns 200 to prevent email enumeration."""
    import re
    from ..web.auth import generate_otp, store_otp, send_otp_email
    from ..db.crud import get_user_by_email

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", body.email):
        return JSONResponse({"message": "otp_sent"})

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_email(db, body.email)
        if user is not None:
            raw_otp, otp_hash = generate_otp()
            store_otp(db, user, otp_hash, "reset")
            try:
                send_otp_email(body.email, raw_otp, "reset")
            except Exception:
                logger.exception("forgot_password: OTP email failed (suppressed)")
    except Exception:
        logger.exception("forgot_password: unexpected error (suppressed)")
    finally:
        db.close()

    return JSONResponse({"message": "otp_sent"})


@limiter.limit("10/hour")
async def api_reset_password(request: "Request", body: ResetPasswordRequest) -> JSONResponse:
    """Complete password reset using a short-lived reset JWT."""
    from ..web.auth import hash_password
    from ..web.auth_middleware import verify_reset_token
    from ..db.crud import get_user_by_email

    if len(body.new_password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    claims = verify_reset_token(body.reset_token)
    if claims is None:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    email = claims["sub"]

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_email(db, email)
        if user is None:
            raise HTTPException(status_code=400, detail="Invalid or expired reset token")

        user.password_hash = hash_password(body.new_password)
        user.failed_login_attempts = 0
        user.locked_until = None
        db.add(user)
        db.commit()

        return JSONResponse({"message": "password_reset_success"})
    finally:
        db.close()


async def api_verify_register_otp(body: VerifyOtpRequest) -> JSONResponse:
    """Verify a registration OTP and issue a JWT on success."""
    from ..web.auth import verify_otp, clear_otp
    from ..web.auth_middleware import create_access_token
    from ..db.crud import get_user_by_email

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_email(db, body.email)
        if user is None or not verify_otp(user, body.otp, "register"):
            raise HTTPException(status_code=400, detail="invalid_or_expired_otp")

        user.email_verified = True
        db.add(user)
        db.commit()
        db.refresh(user)
        clear_otp(db, user)

        token = create_access_token(user.id, user.username)
        return JSONResponse({
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name or user.username,
            "is_admin": user.is_admin,
            "access_token": token,
            "token_type": "bearer",
        })
    finally:
        db.close()


async def api_verify_reset_otp(body: VerifyOtpRequest) -> JSONResponse:
    """Verify a password-reset OTP and return a short-lived reset JWT."""
    from ..web.auth import verify_otp, clear_otp
    from ..web.auth_middleware import create_reset_token
    from ..db.crud import get_user_by_email

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_email(db, body.email)
        if user is None or not verify_otp(user, body.otp, "reset"):
            raise HTTPException(status_code=400, detail="invalid_or_expired_otp")

        clear_otp(db, user)
        return JSONResponse({"reset_token": create_reset_token(body.email)})
    finally:
        db.close()


@limiter.limit("5/hour")
async def api_resend_otp(request: "Request", body: ResendOtpRequest) -> JSONResponse:
    """Resend an OTP — rate-limited, 60s per-user cooldown."""
    from datetime import datetime, timezone, timedelta
    from ..web.auth import generate_otp, store_otp, send_otp_email
    from ..db.crud import get_user_by_email

    if body.purpose not in ("register", "reset"):
        raise HTTPException(status_code=422, detail="purpose must be 'register' or 'reset'")

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_email(db, body.email)
        if user is not None:
            # Enforce 60s cooldown: if expires > now + 14 min, OTP was just issued
            if user.otp_expires:
                expires = user.otp_expires
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if expires > datetime.now(timezone.utc) + timedelta(minutes=14):
                    raise HTTPException(
                        status_code=429,
                        detail="Please wait before requesting another code",
                    )
            raw_otp, otp_hash = generate_otp()
            store_otp(db, user, otp_hash, body.purpose)
            try:
                send_otp_email(body.email, raw_otp, body.purpose)
            except Exception:
                logger.exception("resend_otp: email failed (suppressed)")
    finally:
        db.close()

    return JSONResponse({"message": "otp_sent"})


# ── Handlers ──────────────────────────────────────────────────────────────────

async def api_list_jobs(request: "Request") -> JSONResponse:
    """Return list of all jobs for the authenticated user.

    Falls back to the first user in DB if no valid token is present
    (backward compatibility during migration to token-based auth).
    """
    from ..web.auth_middleware import get_optional_user
    from fastapi import Request as _Req

    token_user = get_optional_user(request)

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        if token_user:
            user_id = token_user["sub"]
        else:
            # Legacy fallback: use first user (will be removed once frontend
            # is fully migrated to token-based auth)
            from ..db.models import User
            user = db.query(User).first()
            if not user:
                return JSONResponse([])
            user_id = user.id

        from ..db.crud import get_jobs_for_user
        jobs = get_jobs_for_user(db, user_id, limit=50)
        jobs_data = []
        for job in jobs:
            jobs_data.append({
                "id": job.id,
                "original_filename": job.original_filename,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": _utc_iso(job.created_at),
                "duration_seconds": job.duration_seconds,
                "file_size": job.file_size,
                "db_dir": job.db_dir,
                "progress_message": job.progress_message
            })
        return JSONResponse(jobs_data)
    finally:
        db.close()

@limiter.limit(lambda: _settings.RATE_LIMIT_UPLOAD)
async def api_upload_job(
    request: "Request",
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
) -> JSONResponse:
    """Upload an audio file and start processing."""
    from pathlib import Path
    import uuid
    from ..config import settings
    from ..workers.queue import job_queue
    from ..db.crud import create_job
    from ..web.auth_middleware import get_optional_user

    # Validate min/max speakers bounds
    if min_speakers is not None and (min_speakers < 1 or min_speakers > 20):
        raise HTTPException(status_code=422, detail="min_speakers must be between 1 and 20")
    if max_speakers is not None and (max_speakers < 1 or max_speakers > 20):
        raise HTTPException(status_code=422, detail="max_speakers must be between 1 and 20")

    original_name = file.filename
    if not original_name:
         raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(original_name).suffix.lower()
    if file_ext not in settings.get_allowed_extensions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(settings.get_allowed_extensions())}"
        )

    # Reject oversized uploads before writing to disk (check Content-Length header)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    token_user = get_optional_user(request)

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        if token_user:
            user_id = token_user["sub"]
        else:
            # Legacy fallback
            from ..db.models import User
            user = db.query(User).first()
            if not user:
                raise HTTPException(status_code=400, detail="No user found to assign job")
            user_id = user.id
             
        # Generate unique filename and save
        unique_name = f"{uuid.uuid4().hex}{file_ext}"
        base = Path(settings.OUTPUT_DIR).parent
        upload_dir = base / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / unique_name

        # Save file to disk
        with open(upload_path, "wb") as buffer:
             import shutil
             shutil.copyfileobj(file.file, buffer)

        saved_size = upload_path.stat().st_size
        if saved_size > max_bytes:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {settings.MAX_UPLOAD_SIZE_MB} MB.",
            )

        # Validate actual MIME type using libmagic (guards against disguised executables)
        import magic
        mime = magic.from_file(str(upload_path), mime=True)
        _ALLOWED_MIME_PREFIXES = ("audio/", "video/")
        if not any(mime.startswith(p) for p in _ALLOWED_MIME_PREFIXES):
            upload_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {mime}. Only audio and video files are accepted.",
            )

        job = create_job(
            db,
            user_id=user_id,
            filename=unique_name,
            original_filename=original_name,
            file_size=saved_size,
            language=language,
            backend=settings.TRANSCRIPTION_BACKEND,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        job_id_local = job.id
    finally:
        db.close()

    # Enqueue job for background processing
    await job_queue.enqueue(job_id_local)
    
    return JSONResponse(
        {"job_id": job_id_local, "status": "pending", "message": "Upload successful"},
        status_code=202,
    )

async def api_delete_job(job_id: str) -> JSONResponse:
    """Delete a job and all its data."""
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
            
        from ..db.crud import delete_job
        delete_job(db, job.id)
    finally:
        db.close()
        
    return JSONResponse({"status": "deleted"}, status_code=200)

async def api_update_segment(job_id: str, segment_id: str, body: dict) -> JSONResponse:
    """Update a specific segment and flush to JSON.
    
    ``segment_id`` can be either:
      - The actual UUID of the Segment row, OR
      - A numeric string representing the segment_index (0-based), which is
        what the Next.js frontend sends since the download JSON doesn't include
        DB IDs.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        from ..db.crud import (
            update_segment_text,
            update_segment_speaker,
            flush_segments_to_json,
            get_segments_for_job,
        )
        from ..db.models import Segment as SegmentModel

        # Resolve the real DB segment ID.
        # If the caller sent a numeric index, look up the Segment by
        # (job_id, segment_index).
        resolved_id = segment_id
        try:
            idx = int(segment_id)
            # It's a numeric index — look up by position
            seg = (
                db.query(SegmentModel)
                .filter(
                    SegmentModel.job_id == job_id,
                    SegmentModel.segment_index == idx,
                )
                .first()
            )
            if seg is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Segment at index {idx} not found for job {job_id[:8]}",
                )
            resolved_id = seg.id
        except ValueError:
            # Not numeric — assume it's already a UUID
            pass

        if "text" in body:
             update_segment_text(db, resolved_id, body["text"])
        if "speaker" in body:
             update_segment_speaker(db, resolved_id, body["speaker"])
             
        # Flush to the aligned JSON on disk so downloads and indexing are up to date!
        try:
             flush_segments_to_json(db, job_id)
        except Exception as e:
             logger.error(f"Failed to flush segments to JSON for job {job_id}: {e}")
             
    finally:
        db.close()
        
    return JSONResponse({"status": "updated"}, status_code=200)

async def api_job_status(job_id: str) -> JSONResponse:
    """Return current job status JSON."""
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
    finally:
        db.close()

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JSONResponse({
        # Include both "id" (matches api_list_jobs and the frontend Job type)
        # and "job_id" (legacy, kept for compatibility with any existing callers).
        "id": job.id,
        "job_id": job.id,
        "status": job.status.value,
        "progress": job.progress,
        "stage_progress": job.stage_progress,
        "progress_message": job.progress_message,
        "error_message": job.error_message,
        "db_dir": job.db_dir,
        "result_json_path": job.result_json_path,
        "duration_seconds": job.duration_seconds,
        "original_filename": job.original_filename,
        "file_size": job.file_size,
        "created_at": _utc_iso(job.created_at),
        "pageindex_status": job.pageindex_status,
    })


async def api_job_query(job_id: str, body: QueryRequest) -> JSONResponse:
    """Run a RAG query against the indexed job transcript."""
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
    finally:
        db.close()

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not ready — current status: {job.status.value}",
        )

    if not job.db_dir:
        raise HTTPException(
            status_code=409,
            detail="Job has no search index (indexing may have failed)",
        )

    question = body.q.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Query 'q' must not be empty")

    try:
        from ..services.query_service import QueryService
        from ..config import settings

        query_svc = QueryService()
        result = query_svc.query(
            question=question,
            db_dir=job.db_dir,
            embedding_model=settings.EMBEDDING_MODEL,
            k=settings.RAG_TOP_K,
            llm_mode=body.llm_mode,   # explicit mode, overrides USE_LOCAL_LLM
        )
        return JSONResponse({
            "job_id": job_id,
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "llm_backend": result.get("llm_backend", body.llm_mode),
        })

    except Exception as exc:
        logger.error(f"API query failed for job {job_id[:8]}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def api_get_chat_history(job_id: str) -> JSONResponse:
    """Ensure a ChatSession exists and return its message history."""
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        from ..db.crud import get_or_create_chat_session, get_chat_messages
        
        # This ensures the session exists, fixing the issue where WS fails if no session found.
        session = get_or_create_chat_session(db, job_id)
        messages = get_chat_messages(db, session.id)
        
        return JSONResponse([m.to_dict() for m in messages])
    finally:
        db.close()


# ── Download handler ──────────────────────────────────────────────────────────

_DOWNLOAD_TYPES = {"transcription", "diarization", "aligned"}


async def api_job_download(
    job_id: str,
    type: str = QueryParam("aligned", description="transcription | diarization | aligned"),
) -> FileResponse:
    """
    Download a raw output JSON file for a completed job.

    Query parameter ``type`` selects the file:
    - ``transcription`` — raw Whisper segments
    - ``diarization``   — raw Pyannote speaker segments
    - ``aligned``       — merged/labelled transcript (default)
    """
    if type not in _DOWNLOAD_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid type '{type}'. Must be one of: {sorted(_DOWNLOAD_TYPES)}",
        )

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        # Allow download from any terminal or near-terminal status.
        # REINDEXING means edits are being baked into the vector store — the
        # aligned transcript is still fully valid and downloadable.
        _downloadable = {
            JobStatus.COMPLETED,
            JobStatus.REINDEXING,
            JobStatus.FAILED,
        }
        if job.status not in _downloadable:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Transcript not yet available — job is still processing "
                    f"(status: {job.status.value})"
                ),
            )

        from pathlib import Path
        stem = Path(job.original_filename).stem
        download_name = f"{stem}_{type}.json"

        # ── aligned: always regenerate from DB so edits are never stale ──────
        if type == "aligned":
            try:
                payload = build_aligned_json_from_db(db, job_id)
            except Exception as exc:
                logger.error(
                    "api_job_download: could not build aligned JSON from DB for "
                    "job %s: %s", job_id[:8], exc, exc_info=True
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not build transcript: {exc}",
                ) from exc
            content = _json.dumps(payload, ensure_ascii=False, indent=2)
            return StreamingResponse(
                iter([content.encode("utf-8")]),
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="{download_name}"',
                },
            )

        # ── transcription / diarization: served from the raw on-disk file ─────
        path_map = {
            "transcription":  job.transcription_json_path,
            "diarization":    job.diarization_json_path,
        }
        file_path = path_map.get(type)

        if not file_path:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"The '{type}' output is not available for this job. "
                    "It may have been processed before this feature was added."
                ),
            )

        p = Path(file_path)
        if not p.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Output file not found on disk: {file_path}",
            )

        return FileResponse(
            path=str(p),
            media_type="application/json",
            filename=download_name,
        )
    finally:
        db.close()


# ── Audio serve handler ───────────────────────────────────────────────────────

async def api_job_audio(job_id: str) -> FileResponse:
    """
    Stream the original uploaded audio file for a job.

    GET /api/jobs/{job_id}/audio

    Used by the frontend audio player to play back the recording alongside
    the transcript.  The file is served from the uploads directory (not temp).

    Errors
    ------
    404  Job not found or audio file missing from disk.
    """
    from pathlib import Path as _Path
    from ..config import settings

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        filename = job.filename
    finally:
        db.close()

    upload_dir = _Path(settings.OUTPUT_DIR).parent / "data" / "uploads"
    audio_path = upload_dir / filename
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found on disk: {filename}",
        )

    ext = audio_path.suffix.lower()
    _mime_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".webm": "audio/webm",
    }
    media_type = _mime_map.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(audio_path),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )


# ── Reindex handler ───────────────────────────────────────────────────────────

async def api_job_reindex(job_id: str) -> JSONResponse:
    """
    Trigger a vector-index rebuild for a completed job.

    POST /api/jobs/{job_id}/reindex

    Behaviour
    ---------
    1. Validates that the job exists, belongs to the caller, and has a
       completed aligned result JSON on disk.
    2. Sets job status to REINDEXING.
    3. Enqueues the job in the same background queue used by the normal
       pipeline (the queue dispatcher picks ``run_reindex`` based on status).
    4. Returns 202 Accepted immediately; progress is streamed via the
       existing ``/ws/jobs/{job_id}`` WebSocket.

    Errors
    ------
    404  Job not found.
    409  Job is not in COMPLETED state, or the aligned result JSON is missing.
    500  Unexpected error when updating DB or enqueueing.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Reindex requires a COMPLETED or FAILED job — "
                    f"current status: {job.status.value}"
                ),
            )

        if not job.result_json_path:
            raise HTTPException(
                status_code=409,
                detail=(
                    "No aligned result JSON found for this job. "
                    "The job may need to be fully re-processed."
                ),
            )

        from pathlib import Path as _Path

        if not _Path(job.result_json_path).exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Aligned result JSON not found on disk: {job.result_json_path}. "
                    "The job may need to be fully re-processed."
                ),
            )

        # Mark job as reindexing
        from ..db.crud import update_job_status
        from ..db.models import JobStatus as _JS

        update_job_status(
            db,
            job_id,
            _JS.REINDEXING,
            progress=0,
            stage_progress=0,
            progress_message="Reindex queued...",
            log_line="🔄 Reindex requested",
        )
    finally:
        db.close()

    # Enqueue — the JobQueue dispatcher routes REINDEXING jobs to run_reindex()
    try:
        from ..workers.queue import job_queue

        await job_queue.enqueue(job_id)
        logger.info("Reindex enqueued for job %s", job_id[:8])
    except Exception as exc:
        logger.error("Failed to enqueue reindex for job %s: %s", job_id[:8], exc)
        raise HTTPException(status_code=500, detail=f"Failed to queue reindex: {exc}") from exc

    return JSONResponse(
        {"job_id": job_id, "status": "reindexing", "message": "Reindex started"},
        status_code=202,
    )


# ── Build PageIndex handler ──────────────────────────────────────────────────

# In-memory cancel flags for PageIndex builds (separate from pipeline cancel).
import threading as _pi_threading
_pageindex_cancel_flags: dict[str, bool] = {}
_pageindex_cancel_lock = _pi_threading.Lock()


def _pi_request_cancel(job_id: str) -> None:
    with _pageindex_cancel_lock:
        _pageindex_cancel_flags[job_id] = True


def _pi_is_cancelled(job_id: str) -> bool:
    with _pageindex_cancel_lock:
        return _pageindex_cancel_flags.get(job_id, False)


def _pi_clear(job_id: str) -> None:
    with _pageindex_cancel_lock:
        _pageindex_cancel_flags.pop(job_id, None)


class PageIndexCancelledError(Exception):
    """Raised when a PageIndex build is cancelled."""


async def api_build_pageindex(job_id: str) -> JSONResponse:
    """
    Trigger a PageIndex tree build for a completed job.

    POST /api/jobs/{job_id}/build-pageindex

    Behaviour
    ---------
    1. Validates that PAGEINDEX_ENABLED is True and the job is COMPLETED.
    2. Sets pageindex_status to "building".
    3. Runs PageIndex indexing asynchronously in a background thread.
    4. Returns 202 Accepted immediately.

    Errors
    ------
    404  Job not found.
    409  Job not completed, or PageIndex not enabled.
    """
    from ..config import settings as _settings

    if not _settings.PAGEINDEX_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="PageIndex is not enabled. Set PAGEINDEX_ENABLED=true in your .env file.",
        )

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Job must be COMPLETED to build PageIndex (current: {job.status.value})",
            )

        if job.pageindex_status == "building":
            raise HTTPException(
                status_code=409,
                detail="PageIndex is already being built for this job",
            )

        # Mark as building and clear any previous cancel flag
        _pi_clear(job_id)
        job.pageindex_status = "building"
        db.commit()
    finally:
        db.close()

    # Run in background thread via the job queue's thread pool
    import threading

    def _build():
        import asyncio as _aio
        _SessionLocal = get_session()
        _db = _SessionLocal()
        try:
            _job = get_job(_db, job_id)
            if _job is None:
                return

            # Read segments
            from ..db.crud import get_segments_for_job
            segments = get_segments_for_job(_db, job_id)
            if not segments:
                _job.pageindex_status = "failed"
                _db.commit()
                return

            seg_dicts = [s.to_dict() for s in segments]

            from ..services.rag.indexer_pageindex import PageIndexAdapter
            adapter = PageIndexAdapter()

            # Pass a cancel checker so build_index can abort between LLM calls
            def cancel_checker() -> None:
                if _pi_is_cancelled(job_id):
                    raise PageIndexCancelledError("PageIndex build cancelled by user")

            loop = _aio.new_event_loop()
            try:
                pi_path = loop.run_until_complete(
                    adapter.build_index(
                        segments=seg_dicts,
                        job_id=job_id,
                        filename=_job.original_filename,
                        cancel_checker=cancel_checker,
                    )
                )
            finally:
                loop.close()

            _job.pageindex_path = str(pi_path)
            _job.pageindex_status = "ready"
            _db.commit()
            logger.info("build_pageindex: job %s complete: %s", job_id[:8], pi_path)

        except PageIndexCancelledError:
            logger.info("build_pageindex: job %s cancelled by user", job_id[:8])
            try:
                _job = get_job(_db, job_id)
                if _job:
                    _job.pageindex_status = "cancelled"
                    _db.commit()
            except Exception:
                pass

        except Exception as exc:
            logger.error("build_pageindex: job %s failed: %s", job_id[:8], exc)
            try:
                _job = get_job(_db, job_id)
                if _job:
                    _job.pageindex_status = "failed"
                    _db.commit()
            except Exception:
                pass
        finally:
            _pi_clear(job_id)
            _db.close()

    t = threading.Thread(target=_build, name=f"pageindex-{job_id[:8]}", daemon=True)
    t.start()

    return JSONResponse(
        {"job_id": job_id, "status": "building", "message": "PageIndex build started"},
        status_code=202,
    )


async def api_cancel_pageindex(job_id: str) -> JSONResponse:
    """
    Cancel an in-progress PageIndex build.

    POST /api/jobs/{job_id}/cancel-pageindex

    Sets a cancel flag that the build thread checks between LLM calls.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.pageindex_status != "building":
            raise HTTPException(
                status_code=409,
                detail=f"PageIndex is not currently building (status: {job.pageindex_status})",
            )

        _pi_request_cancel(job_id)

        # Update status immediately so UI reflects it
        job.pageindex_status = "cancelled"
        db.commit()

        return JSONResponse(
            {"job_id": job_id, "status": "cancelled", "message": "PageIndex build cancel requested"},
        )
    finally:
        db.close()


async def api_get_pageindex_tree(job_id: str) -> JSONResponse:
    """
    Return the PageIndex JSON tree for a completed job.

    GET /api/jobs/{job_id}/pageindex-tree

    Returns the raw tree JSON written by PageIndexAdapter.build_index().
    404 if the job has no pageindex_path or the file does not exist.
    """
    import json as _j
    from pathlib import Path as _Path

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if not job.pageindex_path:
            raise HTTPException(status_code=404, detail="PageIndex not built for this job")

        tree_path = _Path(job.pageindex_path)
        if not tree_path.exists():
            raise HTTPException(status_code=404, detail="PageIndex tree file not found on disk")

        with tree_path.open("r", encoding="utf-8") as fh:
            tree = _j.load(fh)

        return JSONResponse(tree)
    finally:
        db.close()


# ── Cancel handler ────────────────────────────────────────────────────────────

# Job statuses from which cancellation is meaningful (job is actively running
# or waiting to run).
_CANCELLABLE_STATUSES = {
    JobStatus.PENDING,
    JobStatus.TRANSCRIBING,
    JobStatus.DIARIZING,
    JobStatus.ALIGNING,
    JobStatus.INDEXING,
    JobStatus.REINDEXING,
}

# Job statuses from which a restart is allowed.
_RESTARTABLE_STATUSES = {
    JobStatus.CANCELLED,
    JobStatus.FAILED,
}


async def api_job_cancel(job_id: str) -> JSONResponse:
    """
    Request cooperative cancellation of an in-progress job.

    POST /api/jobs/{job_id}/cancel

    Behaviour
    ---------
    1. Validates the job is in a cancellable state.
    2. Sets the cancel flag in ``CancelRegistry`` (checked by workers between stages).
    3. Updates the DB status to CANCELLED immediately so the UI reflects the
       request before the worker has a chance to pick it up.
    4. Returns 200 with the new status.

    The actual cancellation is *cooperative*: the worker detects the flag at
    the next inter-stage boundary and exits cleanly (GPU freed, no partial
    index left).  A job that is mid-transcription will finish that stage before
    stopping.

    Errors
    ------
    404  Job not found.
    409  Job is not in a cancellable state (already completed, failed, etc.).
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status not in _CANCELLABLE_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job cannot be cancelled from status '{job.status.value}'. "
                    f"Only active jobs ({', '.join(s.value for s in _CANCELLABLE_STATUSES)}) "
                    "can be cancelled."
                ),
            )

        # Arm the cancel flag first so any worker that dequeues *before* the
        # DB write also sees it.
        from ..workers.cancel import cancel_registry
        cancel_registry.request_cancel(job_id)

        from ..db.crud import update_job_status
        update_job_status(
            db, job_id, JobStatus.CANCELLED,
            progress=0,
            stage_progress=0,
            progress_message="Cancelled by user",
            log_line="⏹ Cancelled by user",
        )

        # Notify any live WebSocket progress subscribers so the UI updates
        # immediately without waiting for the next worker checkpoint.
        try:
            from ..workers.progress import progress_manager
            progress_manager.update(
                job_id, stage="cancelled", progress=0,
                message="Cancelled by user", status="cancelled",
            )
        except Exception:
            pass

        logger.info("api_job_cancel: job %s cancelled", job_id[:8])

    finally:
        db.close()

    return JSONResponse(
        {"job_id": job_id, "status": "cancelled", "message": "Cancellation requested"},
        status_code=200,
    )


async def api_job_restart(job_id: str) -> JSONResponse:
    """
    Restart a cancelled or failed job from the beginning.

    POST /api/jobs/{job_id}/restart

    Behaviour
    ---------
    1. Validates the job is in CANCELLED or FAILED state.
    2. Clears the cancel flag from ``CancelRegistry``.
    3. Resets progress fields and sets status back to PENDING.
    4. Enqueues the job for full pipeline re-execution.
    5. Returns 202 Accepted.

    Why restart from the beginning and not from the last safe stage:
    The pipeline caches Whisper and Pyannote outputs on disk (``use_cache=True``),
    so the expensive transcription and diarization stages are effectively free
    on restart — they load from disk in seconds.  Checkpoint-resume logic would
    add significant complexity with negligible practical benefit.

    Errors
    ------
    404  Job not found.
    409  Job is not in a restartable state.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status not in _RESTARTABLE_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job cannot be restarted from status '{job.status.value}'. "
                    f"Only cancelled or failed jobs can be restarted."
                ),
            )

        # Clear the cancel flag so the worker runs normally.
        from ..workers.cancel import cancel_registry
        cancel_registry.clear(job_id)

        # ── Clean derived artifacts to prevent duplication on restart ───
        from ..db.crud import (
            update_job_status,
            get_job as _get_job,
            delete_segments_for_job,
        )

        # 1. Delete stale segment rows so the pipeline re-inserts cleanly
        deleted_segs = delete_segments_for_job(db, job_id)
        if deleted_segs:
            logger.info(
                "api_job_restart: cleared %d stale segments for job %s",
                deleted_segs, job_id[:8],
            )

        # 2. Remove temp JSONL / embedding intermediates
        import shutil
        from pathlib import Path as _P
        from ..config import settings

        _temp_job_dir = _P(settings.TEMP_DIR) / job_id
        if _temp_job_dir.exists():
            shutil.rmtree(str(_temp_job_dir), ignore_errors=True)
            logger.debug("api_job_restart: removed temp dir %s", _temp_job_dir)

        update_job_status(
            db, job_id, JobStatus.PENDING,
            progress=0,
            stage_progress=0,
            progress_message="Restarting...",
            log_line="🔄 Job restarted by user",
        )
        # clear_error via direct ORM write — update_job_status only sets
        # error_message when the parameter is not None, so we clear it here.
        _job_reset = _get_job(db, job_id)
        if _job_reset:
            _job_reset.error_message = None
            _job_reset.logs = "[]"
            db.commit()
        logger.info("api_job_restart: job %s reset to PENDING", job_id[:8])
    finally:
        db.close()

    try:
        from ..workers.queue import job_queue
        await job_queue.enqueue(job_id)
        logger.info("api_job_restart: job %s enqueued", job_id[:8])
    except Exception as exc:
        logger.error("api_job_restart: enqueue failed for job %s: %s", job_id[:8], exc)
        raise HTTPException(status_code=500, detail=f"Failed to queue restart: {exc}") from exc

    return JSONResponse(
        {"job_id": job_id, "status": "pending", "message": "Job restarted"},
        status_code=202,
    )


# =============================================================================
# Settings & Profile endpoints
# =============================================================================

def _coerce(current, raw: str):
    """Coerce a DB string value to match the Python type of *current*."""
    if isinstance(current, bool):
        return raw.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw if raw.strip().lower() not in ("none", "null", "") else None


# Keys that are never returned in API responses (not even masked).
_NEVER_EXPOSE: set[str] = {"WEB_SECRET_KEY", "WEB_STORAGE_SECRET"}


SETTINGS_META: dict[str, dict] = {
    # ── Authentication ────────────────────────────────────────────────────
    "HF_API_TOKEN": {
        "description": "HuggingFace API token for gated models and HF Inference API.",
        "type": "password", "sensitive": True, "group": "Authentication",
        "restart_required": False,
    },
    # ── Models ────────────────────────────────────────────────────────────
    "WHISPER_MODEL": {
        "description": "HuggingFace Whisper model name used for ASR (HF backend).",
        "type": "string", "sensitive": False, "group": "Models",
        "restart_required": False,
    },
    "WHISPER_MODEL_SIZE": {
        "description": "Model size for local/faster-whisper backends (tiny/small/medium/large-v3).",
        "type": "string", "sensitive": False, "group": "Models",
        "restart_required": False,
    },
    "DIARIZATION_MODEL": {
        "description": "Pyannote model for speaker diarization.",
        "type": "string", "sensitive": False, "group": "Models",
        "restart_required": False,
    },
    "EMBEDDING_MODEL": {
        "description": "Sentence-transformer model path or HF model ID.",
        "type": "string", "sensitive": False, "group": "Models",
        "restart_required": False,
    },
    "HF_MODEL": {
        "description": "HuggingFace model ID for LLM inference via HF Inference API.",
        "type": "string", "sensitive": False, "group": "Models",
        "restart_required": False,
    },
    "HF_PROVIDER": {
        "description": "HF inference provider (auto/sambanova/novita/cerebras).",
        "type": "string", "sensitive": False, "group": "Models",
        "restart_required": False,
    },
    # ── Backends ──────────────────────────────────────────────────────────
    "TRANSCRIPTION_BACKEND": {
        "description": "Transcription backend: 'huggingface' (API) or 'local' / 'faster-whisper' (GPU).",
        "type": "string", "sensitive": False, "group": "Backends",
        "restart_required": False,
    },
    "USE_LOCAL_LLM": {
        "description": "Use a local AWQ model for LLM generation instead of HF Inference API.",
        "type": "boolean", "sensitive": False, "group": "Backends",
        "restart_required": False,
    },
    "LOCAL_LLM_MODEL_PATH": {
        "description": "Path to local AWQ model directory.",
        "type": "string", "sensitive": False, "group": "Backends",
        "restart_required": False,
    },
    "LOCAL_LLM_CONTEXT_SIZE": {
        "description": "Context window size in tokens for the local LLM.",
        "type": "integer", "sensitive": False, "group": "Backends",
        "restart_required": False,
    },
    "LOCAL_LLM_MAX_TOKENS": {
        "description": "Maximum output tokens per generation.",
        "type": "integer", "sensitive": False, "group": "Backends",
        "restart_required": False,
    },
    "LOCAL_LLM_TEMPERATURE": {
        "description": "Sampling temperature (0.0 = deterministic, 1.0+ = creative).",
        "type": "string", "sensitive": False, "group": "Backends",
        "restart_required": False,
    },
    # ── RAG & Retrieval ───────────────────────────────────────────────────
    "RAG_TOP_K": {
        "description": "Number of chunks to retrieve for each RAG query.",
        "type": "integer", "sensitive": False, "group": "RAG & Retrieval",
        "restart_required": False,
    },
    "SOURCE_SIM_THRESHOLD": {
        "description": "Min cosine similarity (0\u20131) for a source to appear in results.",
        "type": "string", "sensitive": False, "group": "RAG & Retrieval",
        "restart_required": False,
    },
    "SOURCE_MAX_RETURN": {
        "description": "Max number of sources to show after similarity filtering.",
        "type": "integer", "sensitive": False, "group": "RAG & Retrieval",
        "restart_required": False,
    },
    "RAG_RECALL_N": {
        "description": "ANN recall candidates fetched before reranking.",
        "type": "integer", "sensitive": False, "group": "RAG & Retrieval",
        "restart_required": False,
    },
    "RAG_MAX_CONTEXT_CHUNKS": {
        "description": "Max chunks fed to LLM after reranking.",
        "type": "integer", "sensitive": False, "group": "RAG & Retrieval",
        "restart_required": False,
    },
    "CHUNK_TOKENS": {
        "description": "Target chunk size in tokens for transcript chunking.",
        "type": "integer", "sensitive": False, "group": "RAG & Retrieval",
        "restart_required": False,
    },
    "CHUNK_OVERLAP": {
        "description": "Overlap tokens between consecutive chunks.",
        "type": "integer", "sensitive": False, "group": "RAG & Retrieval",
        "restart_required": False,
    },
    # ── Indexing Performance ──────────────────────────────────────────────
    "EMBED_BATCH_SIZE": {
        "description": "Embedding batch size. Lower = less RAM; higher = faster.",
        "type": "integer", "sensitive": False, "group": "Indexing Performance",
        "restart_required": False,
    },
    "PIPELINE_WORKERS": {
        "description": "Max concurrent pipeline jobs. Keep at 1 on GPU-constrained machines.",
        "type": "integer", "sensitive": False, "group": "Indexing Performance",
        "restart_required": False,
    },
    "MEMORY_WATCH_ENABLED": {
        "description": "Enable psutil memory monitoring during indexing.",
        "type": "boolean", "sensitive": False, "group": "Indexing Performance",
        "restart_required": False,
    },
    "MEMORY_WATCH_THRESHOLD_PCT": {
        "description": "RAM fraction (0\u20131) that triggers batch-size halving.",
        "type": "string", "sensitive": False, "group": "Indexing Performance",
        "restart_required": False,
    },
    "RAG_MEMORY_SAFE_MODE": {
        "description": "Enable memory-safe batched embedding and streamed inserts.",
        "type": "boolean", "sensitive": False, "group": "Indexing Performance",
        "restart_required": False,
    },
    # ── Web Server ────────────────────────────────────────────────────────
    "WEB_HOST": {
        "description": "Host to bind the web server to.",
        "type": "string", "sensitive": False, "group": "Web Server",
        "restart_required": True,
    },
    "WEB_PORT": {
        "description": "Port the web server listens on.",
        "type": "integer", "sensitive": False, "group": "Web Server",
        "restart_required": True,
    },
    "WEB_SECRET_KEY": {
        "description": "JWT signing key. Changing this invalidates all active sessions. Requires restart.",
        "type": "password", "sensitive": True, "group": "Web Server",
        "restart_required": True,
    },
    "WEB_STORAGE_SECRET": {
        "description": "NiceGUI session encryption key. Requires restart to take effect.",
        "type": "password", "sensitive": True, "group": "Web Server",
        "restart_required": True,
    },
    "MAX_UPLOAD_SIZE_MB": {
        "description": "Maximum audio upload size in megabytes.",
        "type": "integer", "sensitive": False, "group": "Web Server",
        "restart_required": False,
    },
    "ALLOWED_AUDIO_EXTENSIONS": {
        "description": "Comma-separated allowed audio file extensions.",
        "type": "string", "sensitive": False, "group": "Web Server",
        "restart_required": False,
    },
    "RATE_LIMIT_UPLOAD": {
        "description": "Upload rate limit per IP (slowapi format, e.g. '10/hour').",
        "type": "string", "sensitive": False, "group": "Web Server",
        "restart_required": False,
    },
    "RATE_LIMIT_CHAT": {
        "description": "Chat WebSocket rate limit per IP (e.g. '60/minute').",
        "type": "string", "sensitive": False, "group": "Web Server",
        "restart_required": False,
    },
    # ── PageIndex ─────────────────────────────────────────────────────────
    "PAGEINDEX_ENABLED": {
        "description": "Master toggle for the PageIndex feature.",
        "type": "boolean", "sensitive": False, "group": "PageIndex",
        "restart_required": False,
    },
    "PAGEINDEX_AUTO_INDEX": {
        "description": "Automatically build PageIndex during pipeline ingestion.",
        "type": "boolean", "sensitive": False, "group": "PageIndex",
        "restart_required": False,
    },
    "PAGEINDEX_LLM_BACKEND": {
        "description": "LLM backend for PageIndex calls (openrouter/ollama/openai/custom).",
        "type": "string", "sensitive": False, "group": "PageIndex",
        "restart_required": False,
    },
    "PAGEINDEX_LLM_BASE_URL": {
        "description": "OpenAI-compatible base URL for PageIndex LLM. Leave empty to auto-resolve.",
        "type": "string", "sensitive": False, "group": "PageIndex",
        "restart_required": False,
    },
    "PAGEINDEX_LLM_MODEL": {
        "description": "Model ID for PageIndex tree search LLM calls.",
        "type": "string", "sensitive": False, "group": "PageIndex",
        "restart_required": False,
    },
    "PAGEINDEX_LLM_API_KEY": {
        "description": "API key for the PageIndex LLM backend (OpenRouter/OpenAI).",
        "type": "password", "sensitive": True, "group": "PageIndex",
        "restart_required": False,
    },
    "PAGEINDEX_MAX_PAGE_TOKENS": {
        "description": "Max tokens per tree node during PageIndex indexing.",
        "type": "integer", "sensitive": False, "group": "PageIndex",
        "restart_required": False,
    },
    "PAGEINDEX_OUTPUT_DIR": {
        "description": "Directory for storing PageIndex JSON tree files.",
        "type": "string", "sensitive": False, "group": "PageIndex",
        "restart_required": False,
    },
}

_SENSITIVE_KEYS = {k for k, v in SETTINGS_META.items() if v["sensitive"]}


async def api_get_settings(request: Request) -> JSONResponse:
    """GET /api/settings — return all configurable settings (auth required)."""
    from .auth_middleware import get_current_user
    from ..db.crud import get_all_settings as _get_all_settings

    await get_current_user(request)  # raises 401 if not authenticated

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        db_rows = {row.key: row.value for row in _get_all_settings(db)}
    finally:
        db.close()

    result = []
    for key, meta in SETTINGS_META.items():
        if db_rows.get(key) is not None:
            raw_value = db_rows[key]
        else:
            live = getattr(_settings, key, None)
            raw_value = "" if live is None else str(live)

        if key in _NEVER_EXPOSE:
            display_value = ""
        elif meta["sensitive"]:
            display_value = "\u2022" * 8 if raw_value else ""
        else:
            display_value = raw_value

        result.append({
            "key": key,
            "value": display_value,
            "description": meta["description"],
            "type": meta["type"],
            "sensitive": meta["sensitive"],
            "group": meta["group"],
            "restart_required": meta["restart_required"],
        })
    return JSONResponse(result)


class UpdateSettingRequest(BaseModel):
    value: str


async def api_update_setting(
    request: Request, key: str, body: UpdateSettingRequest
) -> JSONResponse:
    """PUT /api/settings/{key} — update a setting value (admin only)."""
    from .auth_middleware import get_current_user
    from ..db.crud import get_user_by_id, upsert_setting

    token_user = await get_current_user(request)  # raises 401 if not authenticated

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_id(db, token_user["sub"])
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")

        if key not in SETTINGS_META:
            raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")

        upsert_setting(db, key, body.value)

        # Hot-patch the live singleton (skip never-override keys)
        if key not in _settings._NEVER_OVERRIDE:
            try:
                current = getattr(_settings, key, None)
                coerced = _coerce(current, body.value)
                object.__setattr__(_settings, key, coerced)
            except Exception:
                pass  # saved to DB; will apply on next restart
    finally:
        db.close()

    return JSONResponse({
        "ok": True,
        "restart_required": SETTINGS_META[key]["restart_required"],
    })


async def api_get_me(request: Request) -> JSONResponse:
    """GET /api/me — return the authenticated user's profile."""
    from .auth_middleware import get_current_user
    from ..db.crud import get_user_by_id

    token_user = await get_current_user(request)

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_id(db, token_user["sub"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return JSONResponse({
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name or user.username,
            "is_admin": user.is_admin,
            "created_at": _utc_iso(user.created_at),
            "last_login": _utc_iso(user.last_login),
        })
    finally:
        db.close()


async def api_rename_speaker(request: Request, job_id: str) -> JSONResponse:
    """POST /api/jobs/{job_id}/rename-speaker
    Body: {"old_name": "SPEAKER_00", "new_name": "Alice"}

    Bulk-renames all segments matching old_name in the given job and flushes
    updated segments to the on-disk JSON transcript.

    NOTE: Does NOT auto-trigger reindex — the user can use the existing
    "Reindex" button after renaming. This avoids blocking the response on
    a potentially long reindex, and allows the user to rename multiple
    speakers before reindexing once.
    """
    from .auth_middleware import get_optional_user
    from ..db import crud
    from ..db.database import get_session

    token_user = get_optional_user(request)
    if not token_user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=422)

    old_name = (body.get("old_name") or "").strip()
    new_name = (body.get("new_name") or "").strip()
    if not old_name or not new_name:
        return JSONResponse({"error": "old_name and new_name are required"}, status_code=422)

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = crud.get_job(db, job_id)
        if not job or job.user_id != token_user["sub"]:
            return JSONResponse({"error": "Not found"}, status_code=404)

        count = crud.bulk_update_speaker_name(db, job_id, old_name, new_name)
        crud.flush_segments_to_json(db, job_id)
    finally:
        db.close()

    return JSONResponse({"renamed": count})


async def api_delete_chat_message(request: Request, message_id: str) -> JSONResponse:
    """DELETE /api/chat/messages/{message_id}
    
    Deletes a user message and the following assistant response (the Q&A pair).
    Only the user who owns the job can delete messages.
    """
    from .auth_middleware import get_optional_user
    from ..db.models import ChatMessage, ChatSession, Job
    from ..db.database import get_session

    token_user = get_optional_user(request)
    if not token_user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        # 1. Fetch the message and verify it exists
        msg = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
        if msg is None:
            raise HTTPException(status_code=404, detail="Message not found")

        # 2. Verify ownership: msg -> session -> job -> user_id
        job = (
            db.query(Job)
            .join(ChatSession)
            .filter(ChatSession.id == msg.session_id)
            .first()
        )
        if job is None or job.user_id != token_user["sub"]:
            raise HTTPException(status_code=404, detail="Message not found")

        # 3. Rules: only allow starting delete from a "user" message
        if msg.role != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be deleted to remove a Q&A pair")

        # 4. Find the next message in the session (potential assistant response)
        # We look for the message immediately following this one in the same session.
        next_msg = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == msg.session_id,
                ChatMessage.created_at >= msg.created_at,
                ChatMessage.id != msg.id
            )
            .order_by(ChatMessage.created_at.asc())
            .first()
        )

        # 5. Delete assistant reply if it exists
        if next_msg and next_msg.role == "assistant":
            db.delete(next_msg)
            logger.info("Deleted assistant message %s following user msg %s", next_msg.id[:8], message_id[:8])

        # 6. Delete the user message itself
        db.delete(msg)
        db.commit()
        logger.info("Deleted user message %s and its Q&A pair", message_id[:8])

        return Response(status_code=204)
    finally:
        db.close()
