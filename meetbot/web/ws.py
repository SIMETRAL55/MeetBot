"""
WebSocket endpoint for real-time job progress streaming.

Endpoint:  GET /ws/jobs/{job_id}

Message shape (JSON, server → client):
    {
        "job_id":           "<uuid>",
        "stage":            "transcribing" | "diarizing" | "aligning" |
                            "indexing" | "completed" | "failed" | "pending",
        "stage_progress":   0.0 – 100.0,   # progress within the current stage
        "overall_progress": 0.0 – 100.0,   # weighted aggregate across all stages
        "logs":             ["...", ...],   # last ≤10 log lines
        "status":           "running" | "completed" | "failed",
        "message":          "<latest human-readable message>"
    }

Close codes sent by server:
    4004  Job not found.
    4005  Job already finished (last event delivered before close).

Auth note:
    NiceGUI stores session data in encrypted browser-local storage, not in a
    standard HTTP cookie that a raw Starlette WebSocket handler can read.
    For this air-gapped internal deployment, the job_id (UUID v4) acts as an
    unguessable capability token. A future iteration should add a short-lived
    signed token generated at page load.

Usage examples:
    Browser:
        const ws = new WebSocket("ws://localhost:8080/ws/jobs/<job_id>");
        ws.onmessage = e => console.log(JSON.parse(e.data));

    Python (websockets library):
        async with websockets.connect("ws://localhost:8080/ws/jobs/<job_id>") as ws:
            async for msg in ws:
                print(json.loads(msg))

    curl (websocat):
        websocat ws://localhost:8080/ws/jobs/<job_id>
"""

import asyncio
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect

from ..db.database import get_session
from ..db.crud import get_job
from ..db.models import JobStatus
from ..workers.progress import progress_manager

logger = logging.getLogger(__name__)

# Stages that indicate the job is still running.
# REINDEXING must be included — omitting it caused the WS handler to
# immediately read the DB, see status=reindexing, classify it as
# non-active, and send status="failed" before the worker even started.
ACTIVE_STATUSES = {
    JobStatus.PENDING,
    JobStatus.TRANSCRIBING,
    JobStatus.DIARIZING,
    JobStatus.ALIGNING,
    JobStatus.INDEXING,
    JobStatus.REINDEXING,
}


async def ws_job_progress(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket handler that streams job progress events for the given job_id.

    Lifecycle:
    1. Accept the WebSocket connection.
    2. Validate the job exists in the DB.
    3. If the job is already finished, send one final event and close.
    4. If the job is active/pending, subscribe to ProgressManager and relay
       events until the connection drops or the job reaches a terminal state.
    5. Unsubscribe and close cleanly in all cases.

    Args:
        websocket: The Starlette/FastAPI WebSocket connection.
        job_id: Job UUID from the URL path parameter.
    """
    await websocket.accept()
    logger.info(f"WS connected for job {job_id[:8]}")

    # ── 1. Validate job ──────────────────────────────────────────────────
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            await websocket.send_text(json.dumps({
                "error": "Job not found",
                "job_id": job_id,
            }))
            await websocket.close(code=4004)
            logger.warning(f"WS 4004: job {job_id[:8]} not found")
            return

        is_active = job.status in ACTIVE_STATUSES
        job_status_val = job.status.value

        # Build snapshot from current DB state
        try:
            job_logs = json.loads(job.logs or "[]")
        except Exception:
            job_logs = []

        snapshot = {
            "job_id": job_id,
            "stage": job_status_val,
            "stage_progress": round(getattr(job, "stage_progress", 0.0) or 0.0, 1),
            "overall_progress": round(job.progress or 0.0, 1),
            "logs": job_logs[-10:],
            "status": "running" if is_active else job_status_val,
            "message": job.progress_message or "",
        }
    finally:
        db.close()

    # ── 2. Already finished — send snapshot and close ───────────────────
    if not is_active:
        # Map terminal statuses: anything that is not "completed" is "failed"
        # (cancelled, reindexing-failed, etc.) so the UI knows the job stopped.
        snapshot["status"] = "completed" if job_status_val == "completed" else job_status_val
        await websocket.send_text(json.dumps(snapshot))
        await websocket.close(code=4005)
        logger.info(f"WS 4005: job {job_id[:8]} already {job_status_val}")
        return

    # ── 3. Active job — subscribe and relay events ───────────────────────
    # Send the current snapshot immediately so the client has initial state
    await websocket.send_text(json.dumps(snapshot))

    queue = progress_manager.subscribe(job_id)
    try:
        while True:
            # Wait for the next event with a 30-second heartbeat timeout
            try:
                event: dict = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a heartbeat ping so the connection stays alive
                # (also lets us detect stale connections)
                try:
                    await websocket.send_text(json.dumps({"ping": True}))
                except Exception:
                    break
                continue

            # Relay the event to the client
            try:
                await websocket.send_text(json.dumps(event))
            except (WebSocketDisconnect, RuntimeError):
                logger.info(f"WS disconnected for job {job_id[:8]}")
                break

            # Terminal event — close gracefully after final message
            if event.get("status") in ("completed", "failed"):
                await asyncio.sleep(0.1)  # brief flush pause
                try:
                    await websocket.close(code=1000)
                except Exception:
                    pass
                logger.info(
                    f"WS closed normally for job {job_id[:8]}: "
                    f"status={event.get('status')}"
                )
                break

    except WebSocketDisconnect:
        logger.info(f"WS client disconnected for job {job_id[:8]}")
    except Exception as e:
        logger.error(f"WS error for job {job_id[:8]}: {e}", exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        progress_manager.unsubscribe(job_id, queue)
        logger.info(f"WS unsubscribed for job {job_id[:8]}")
