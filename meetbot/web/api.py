"""
REST API endpoints for MeetBot.

Registered on the NiceGUI FastAPI app via app.add_api_route().

Routes
------
GET  /api/jobs/{job_id}/status
    Returns job status, progress, and db_dir.

POST /api/jobs/{job_id}/query
    Body: {"q": "...", "llm_mode": "local" | "hf"}
    Returns the RAG answer and source segments for a completed job.
    ``llm_mode`` defaults to "local".

GET  /api/jobs/{job_id}/download?type=transcription|diarization|aligned
    Returns the requested JSON file as an attachment download.
"""

import json as _json
import logging
from typing import Literal, Optional

from fastapi import HTTPException, Query as QueryParam
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..db.database import get_session
from ..db.crud import get_job, build_aligned_json_from_db
from ..db.models import JobStatus

logger = logging.getLogger(__name__)


# ── Request / response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    q: str
    llm_mode: Literal["local", "hf"] = Field(
        "local",
        description="LLM backend: 'local' for llama.cpp GGUF; 'hf' for HuggingFace Inference API",
    )


# ── Handlers ──────────────────────────────────────────────────────────────────

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
