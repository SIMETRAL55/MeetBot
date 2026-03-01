"""
WebSocket endpoint for streaming RAG chat.

Endpoint:  GET /ws/chat/{job_id}

Protocol (client → server, JSON):
    {"question": "...", "llm_mode": "local" | "hf"}

Protocol (server → client, JSON sequence):
    {"type": "sources",  "data": [...]}            # emitted first, before generation
    {"type": "token",    "data": "<str>"}           # one per generated token
    {"type": "done",     "llm_backend": "local|hf",
                         "full_answer": "..."}      # generation complete
    {"type": "error",    "data": "<message>"}       # on any error

Close codes:
    4004  Job not found.
    4009  Job not ready for querying (not COMPLETED or no db_dir).
    4010  Bad request payload (missing/invalid fields).

Auth: The job_id (UUID-v4) acts as an unguessable capability token for this
      internal deployment.  A future revision should add a short-lived signed token.

Usage (Python websockets library):
    async with websockets.connect("ws://localhost:8080/ws/chat/<job_id>") as ws:
        await ws.send(json.dumps({"question": "What was discussed?", "llm_mode": "local"}))
        async for msg in ws:
            event = json.loads(msg)
            if event["type"] == "token":
                print(event["data"], end="", flush=True)
            elif event["type"] == "done":
                print()
                break

Usage (websocat):
    websocat ws://localhost:8080/ws/chat/<job_id>
    > {"question":"Who spoke first?","llm_mode":"local"}
"""

import asyncio
import concurrent.futures
import json
import logging
from typing import Iterator, Dict, Any

from fastapi import WebSocket, WebSocketDisconnect

from ..db.database import get_session
from ..db.crud import get_job, get_or_create_chat_session, create_chat_message
from ..db.models import JobStatus
from ..config import settings

logger = logging.getLogger(__name__)

# One executor per process — prevents spawning unlimited threads on rapid queries
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ws_chat"
)


async def _relay_stream(
    websocket: WebSocket,
    stream_iter: Iterator[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run a synchronous streaming generator in the thread-pool and relay each
    event to the WebSocket.

    The generator runs in``_THREAD_POOL`` (background thread).
    Events are passed back to the async context via an ``asyncio.Queue``.

    Args:
        websocket: Active WebSocket connection.
        stream_iter: Synchronous generator that yields typed event dicts.

    Returns:
        The final "done" event dict (or an "error" event if generation failed).
    """
    loop    = asyncio.get_event_loop()
    q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def _worker():
        """Drain the generator and post events onto the asyncio queue."""
        try:
            for event in stream_iter:
                asyncio.run_coroutine_threadsafe(q.put(event), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "data": str(exc)}), loop
            ).result()
        finally:
            # Sentinel — signals the async consumer to stop
            asyncio.run_coroutine_threadsafe(q.put({"type": "_sentinel"}), loop).result()

    future = _THREAD_POOL.submit(_worker)

    last_event: Dict[str, Any] = {"type": "done", "llm_backend": "unknown", "full_answer": ""}

    while True:
        event = await q.get()
        if event.get("type") == "_sentinel":
            break

        await websocket.send_text(json.dumps(event))

        if event["type"] in ("done", "error"):
            last_event = event
            # Drain remaining sentinel without re-relaying
            while True:
                leftover = await q.get()
                if leftover.get("type") == "_sentinel":
                    break

    # Ensure the background thread has finished
    future.result(timeout=5.0)
    return last_event


async def ws_chat(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket handler for streaming RAG chat against an indexed job transcript.

    Lifecycle:
    1. Accept connection.
    2. Validate job exists and is COMPLETED with a db_dir.
    3. Receive a JSON message: {"question": "...", "llm_mode": "local"|"hf"}.
    4. Run retrieval + streaming generation via query_stream().
    5. Save user message + assistant message to ChatSession.
    6. Close gracefully.

    The connection handles exactly one question-answer pair per WebSocket
    connection.  To ask a follow-up, the client opens a new connection.
    This stateless design keeps the handler simple and avoids connection
    management complexity.
    """
    await websocket.accept()
    logger.info("ws_chat: connection accepted for job %s", job_id[:8])

    # ── 1. Validate job ──────────────────────────────────────────────────
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None:
            await websocket.send_text(json.dumps({"type": "error", "data": "Job not found"}))
            await websocket.close(code=4004)
            return

        if job.status != JobStatus.COMPLETED or not job.db_dir:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": (
                    f"Job not ready for queries (status={job.status.value}, "
                    f"db_dir={'set' if job.db_dir else 'missing'})"
                ),
            }))
            await websocket.close(code=4009)
            return

        job_db_dir = job.db_dir
        original_filename = job.original_filename

        # Ensure chat session exists (creates if needed)
        chat_session = get_or_create_chat_session(db, job_id)
        session_id = chat_session.id
    finally:
        db.close()

    # ── 2. Receive question ──────────────────────────────────────────────
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        payload = json.loads(raw)
    except asyncio.TimeoutError:
        await websocket.send_text(json.dumps({"type": "error", "data": "Timeout waiting for question"}))
        await websocket.close()
        return
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "data": f"Invalid payload: {exc}"}))
        await websocket.close(code=4010)
        return

    question = str(payload.get("question", "")).strip()
    if not question:
        await websocket.send_text(json.dumps({"type": "error", "data": "question must not be empty"}))
        await websocket.close(code=4010)
        return

    llm_mode: str = payload.get("llm_mode", "local")
    if llm_mode not in ("local", "hf"):
        llm_mode = "local"

    logger.info(
        "ws_chat: job=%s question=%r llm_mode=%s",
        job_id[:8], question[:80], llm_mode,
    )

    # ── 3. Persist user message ──────────────────────────────────────────
    db2 = SessionLocal()
    try:
        create_chat_message(db2, session_id, role="user", content=question)
    finally:
        db2.close()

    # ── 4. Stream generation ─────────────────────────────────────────────
    from ..services.query_service import QueryService

    query_svc   = QueryService()
    stream_iter = query_svc.query_stream(
        question=question,
        db_dir=job_db_dir,
        embedding_model=settings.EMBEDDING_MODEL,
        k=settings.RAG_TOP_K,
        llm_mode=llm_mode,
    )

    try:
        last_event = await _relay_stream(websocket, stream_iter)
    except WebSocketDisconnect:
        logger.info("ws_chat: client disconnected mid-stream for job %s", job_id[:8])
        return
    except Exception as exc:
        logger.error("ws_chat: relay error: %s", exc, exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "data": str(exc)}))
        except Exception:
            pass
        return

    # ── 5. Persist assistant message ─────────────────────────────────────
    if last_event.get("type") == "done":
        full_answer  = last_event.get("full_answer", "")
        llm_backend  = last_event.get("llm_backend", llm_mode)
        # Extract sources from the "sources" event that was already sent
        # They aren't in last_event; we re-query the stream's sources from
        # the _relay_stream log — but since stream is exhausted, we fall back
        # to saving without sources (the UI already received them live).
        db3 = SessionLocal()
        try:
            create_chat_message(
                db3, session_id,
                role="assistant",
                content=full_answer,
                llm_backend=llm_backend,
            )
        finally:
            db3.close()
        logger.info(
            "ws_chat: saved assistant message (%d chars) for job %s",
            len(full_answer), job_id[:8],
        )

    try:
        await websocket.close(code=1000)
    except Exception:
        pass

    logger.info("ws_chat: session complete for job %s", job_id[:8])
