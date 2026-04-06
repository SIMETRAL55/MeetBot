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
import threading
from typing import Iterator, Dict, Any, Optional

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
    abort_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """
    Run a synchronous streaming generator in the thread-pool and relay each
    event to the WebSocket.

    The generator runs in ``_THREAD_POOL`` (background thread).
    Events are passed back to the async context via an ``asyncio.Queue``.

    Persistence-safe design
    -----------------------
    ``last_event`` is captured BEFORE any ``send_text`` call so the caller
    always receives the full ``full_answer`` even when the WebSocket is
    already closed (user abort / early client disconnect).  If the socket is
    broken we keep draining the queue until the sentinel arrives so the
    generator thread can finish cleanly and ``future.result()`` does not hang.

    Args:
        websocket: Active WebSocket connection.
        stream_iter: Synchronous generator that yields typed event dicts.

    Returns:
        The final ``{"type": "done", "full_answer": "..."}`` dict.
        Falls back to ``{"type": "done", "full_answer": ""}`` if the generator
        never yielded a ``done`` event.
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
    # Track accumulated token content in case the generator never yields "done"
    # (e.g. client aborts mid-stream; we still want to save what was generated)
    accumulated_content: str = ""
    ws_broken: bool = False  # True after any send_text failure

    while True:
        event = await q.get()

        if event.get("type") == "_sentinel":
            # Generator exhausted without a "done" event (shouldn't happen in
            # normal operation, but guard against it so we always have content).
            if not last_event.get("full_answer") and accumulated_content:
                last_event = {
                    "type": "done",
                    "llm_backend": last_event.get("llm_backend", "unknown"),
                    "full_answer": accumulated_content,
                    "stopped": True,
                }
            break

        # ── PERSIST-FIRST: capture terminal events BEFORE sending ──────
        # If send_text raises (client disconnected), last_event is already
        # set so the caller can still persist the assistant message.
        is_terminal = event["type"] in ("done", "error")
        if is_terminal:
            last_event = event

        # Accumulate token content for fallback persistence on early abort
        if event["type"] == "token":
            accumulated_content += event.get("data", "")

        # ── Forward event to WebSocket (best-effort) ────────────────────
        if not ws_broken:
            try:
                await websocket.send_text(json.dumps(event))
            except Exception as send_exc:
                # Client disconnected (abort / network drop).  Log and
                # continue draining so the generator thread finishes.
                ws_broken = True
                # Signal the generator thread to abort (stops LLM token generation
                # immediately instead of waiting for the full response to finish).
                if abort_event is not None:
                    abort_event.set()
                logger.info(
                    "_relay_stream: WebSocket broken while sending %r event: %s",
                    event.get("type"), send_exc,
                )

        if is_terminal:
            # Drain remaining queue entries until the sentinel arrives so
            # the background thread slot is freed cleanly before we return.
            while True:
                leftover = await q.get()
                if leftover.get("type") == "_sentinel":
                    # Capture any leftover terminal events (edge case)
                    break
                if leftover.get("type") in ("done", "error"):
                    last_event = leftover
                elif leftover.get("type") == "token":
                    accumulated_content += leftover.get("data", "")
            break  # outer loop done

    # Await the background thread finishing without blocking the asyncio event loop.
    #
    # CRITICAL: concurrent.futures.Future.result() is a SYNCHRONOUS blocking call.
    # Calling it directly inside an async function freezes the entire FastAPI event
    # loop for however long the thread takes to finish (up to the timeout), which
    # prevents ALL other requests — including simple REST calls like GET /api/jobs/{id}
    # — from being processed.  This was the root cause of the "Loading Job Details..."
    # delay: the job-detail page's fetchJob() REST request could not complete while
    # the event loop was blocked here.
    #
    # asyncio.wrap_future() wraps the concurrent.futures.Future as an asyncio.Future
    # so we can await it without blocking.  asyncio.wait_for() enforces the timeout
    # asynchronously, and abort_event.set() (called as soon as the WS breaks) ensures
    # the generator stops quickly so this await returns in milliseconds on abort.
    try:
        await asyncio.wait_for(
            asyncio.wrap_future(future, loop=loop),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "_relay_stream: background thread did not finish within 300 s — "
            "generator may still be running"
        )
    except Exception as fut_exc:
        logger.warning("_relay_stream: background thread error: %s", fut_exc)

    # Final fallback: if we never got a "done" with content but accumulated
    # tokens were streamed (abort scenario), synthesise a "done" event.
    if not last_event.get("full_answer") and accumulated_content:
        last_event = {
            "type": "done",
            "llm_backend": last_event.get("llm_backend", "unknown"),
            "full_answer": accumulated_content,
            "stopped": True,
        }

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

    retrieval_method: str = payload.get("retrieval_method", "vector")
    if retrieval_method not in ("vector", "pageindex"):
        retrieval_method = "vector"

    starred_contexts: list = payload.get("starred_contexts", [])
    reference_context: str = payload.get("reference_context", "")
    reference_enabled: bool = payload.get("reference_enabled", False)

    logger.info(
        "ws_chat: job=%s question=%r llm_mode=%s retrieval_method=%s starred=%d ref_enabled=%s",
        job_id[:8], question[:80], llm_mode, retrieval_method,
        len(starred_contexts), reference_enabled,
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
    # abort_event: shared signal between relay and generator.
    # _relay_stream sets it the moment the WS breaks (client left the page),
    # and query_stream checks it inside the token loop so the LLM stops
    # generating instead of running to completion over the next 1-2 minutes.
    abort_event = threading.Event()
    stream_iter = query_svc.query_stream(
        question=question,
        db_dir=job_db_dir,
        embedding_model=settings.EMBEDDING_MODEL,
        k=settings.RAG_TOP_K,
        llm_mode=llm_mode,
        abort_event=abort_event,
        retrieval_method=retrieval_method,
        job_id=job_id,
        starred_contexts=starred_contexts,
        reference_context=reference_context,
        reference_enabled=reference_enabled,
    )

    # _relay_stream is now disconnect-safe: it captures last_event BEFORE
    # send_text and drains the queue on any WebSocket error, so it should
    # never raise WebSocketDisconnect itself.  We keep the handlers below as
    # a last-resort safety net for truly unexpected exceptions.
    try:
        last_event = await _relay_stream(websocket, stream_iter, abort_event=abort_event)
    except WebSocketDisconnect:
        # Should not happen with the new _relay_stream, but handle defensively.
        logger.info(
            "ws_chat: WebSocket disconnected during relay for job %s "
            "(assistant message may not be saved)",
            job_id[:8],
        )
        return
    except Exception as exc:
        logger.error("ws_chat: relay error: %s", exc, exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "data": str(exc)}))
        except Exception:
            pass
        return

    # ── 5. Persist assistant message ─────────────────────────────────────
    # Save whenever we have a "done" event — this includes the stopped/aborted
    # case where _relay_stream synthesised a done event from accumulated tokens.
    # Only skip saving for "error" events (no meaningful content to store).
    full_answer = last_event.get("full_answer", "")
    if last_event.get("type") == "done" and full_answer:
        llm_backend = last_event.get("llm_backend", llm_mode)
        stopped     = last_event.get("stopped", False)
        # "stopped" means the user aborted mid-generation; flag with status
        # so the history view can indicate partial answers.
        msg_status  = "stopped" if stopped else "completed"

        db3 = SessionLocal()
        try:
            # filtered_sources carries the answer-similarity filtered source list
            # produced by query_stream Stage 4.  Persisting it here means chat
            # history includes sources — they reappear correctly on page reload.
            filtered_sources = last_event.get("filtered_sources") or []
            create_chat_message(
                db3, session_id,
                role="assistant",
                content=full_answer,
                llm_backend=llm_backend,
                status=msg_status,
                sources=filtered_sources if filtered_sources else None,
                retrieval_method=retrieval_method,
            )
        except Exception as db_exc:
            logger.error(
                "ws_chat: DB write for assistant message failed (job %s): %s",
                job_id[:8], db_exc, exc_info=True,
            )
        finally:
            db3.close()
        logger.info(
            "ws_chat: saved assistant message (%d chars, %d sources, status=%s) for job %s",
            len(full_answer), len(filtered_sources), msg_status, job_id[:8],
        )
    elif not full_answer and last_event.get("type") == "done":
        logger.warning(
            "ws_chat: 'done' event had empty full_answer for job %s — "
            "skipping DB write (no content to save)",
            job_id[:8],
        )

    try:
        await websocket.close(code=1000)
    except Exception:
        pass

    logger.info("ws_chat: session complete for job %s", job_id[:8])
