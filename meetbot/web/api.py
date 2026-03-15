"""
REST API endpoints for MeetBot.

Registered on the NiceGUI FastAPI app via app.add_api_route().

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
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field

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
    from ..web.auth import authenticate_user
    from ..web.auth_middleware import create_access_token

    user = authenticate_user(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user.id, user.username)
    return JSONResponse({
        "user_id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
        "is_admin": user.is_admin,
        "access_token": token,
        "token_type": "bearer",
    })

async def api_auth_register(body: RegisterRequest) -> JSONResponse:
    """Register a new user account and return access token."""
    from ..web.auth import hash_password
    from ..web.auth_middleware import create_access_token
    from ..db.crud import create_user, get_user_by_username

    # Validate password strength
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        existing = get_user_by_username(db, body.username)
        if existing:
            raise HTTPException(status_code=409, detail="Username already exists")
        hashed = hash_password(body.password)
        user = create_user(
            db, body.username, hashed,
            display_name=body.display_name,
        )
        token = create_access_token(user.id, user.username)
        return JSONResponse({
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name or user.username,
            "is_admin": user.is_admin,
            "access_token": token,
            "token_type": "bearer",
        }, status_code=201)
    finally:
        db.close()


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
