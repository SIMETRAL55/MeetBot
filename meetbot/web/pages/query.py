"""
Query page — ChatGPT-like streaming RAG interface over indexed transcripts.

Provides:
- Persistent chat history stored in the database (per job)
- Token-by-token streaming answers via QueryService.query_stream()
- Sources panel shown before the answer begins
- Chat bubble layout (user right-aligned, assistant left-aligned)
- LLM backend selector (Local GGUF / HuggingFace Inference API)
- Clear history button
"""

import asyncio
import logging
import threading
from datetime import datetime

from nicegui import ui, app
from nicegui.client import Client as _NiceGuiClient

from ..auth import get_current_user_id
from ..components.nav import create_header
from ...config import settings
from ...db.crud import (
    create_chat_message,
    delete_chat_history,
    get_chat_messages,
    get_job,
    get_or_create_chat_session,
    update_chat_message,
)
from ...db.database import get_session
from ...db.models import JobStatus

logger = logging.getLogger(__name__)

# LLM mode options shown in the UI
_LLM_OPTIONS = {
    "local": "Local LLM",
    "hf":    "HuggingFace API",
}


@ui.page("/query/{job_id}")
async def query_page(job_id: str) -> None:  # noqa: C901
    """Render the ChatGPT-like query page for a specific job."""

    user_id = get_current_user_id()
    if not user_id:
        ui.navigate.to("/login")
        return

    SessionLocal = get_session()
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job is None or job.user_id != user_id:
            ui.navigate.to("/")
            ui.notify("Job not found", type="negative")
            return

        if job.status != JobStatus.COMPLETED or not job.db_dir:
            ui.navigate.to(f"/job/{job_id}")
            ui.notify("This job is not ready for querying", type="warning")
            return

        job_filename = job.original_filename
        job_db_dir = job.db_dir

        # Ensure a ChatSession exists and load message history
        session = get_or_create_chat_session(db, job_id)
        session_id = session.id
        history_msgs = get_chat_messages(db, session_id, limit=200)
    finally:
        db.close()

    # Fetch document count for the k-input upper bound.  Uses only the
    # lightweight chromadb count API (no model loading).
    import asyncio as _asyncio
    from ...services.query_service import QueryService as _QS
    try:
        _doc_count: int = await _asyncio.get_event_loop().run_in_executor(
            None, _QS.count_documents, job_db_dir
        )
    except Exception:
        _doc_count = 0
    # If count is unavailable, allow at least RAG_TOP_K docs
    doc_count_max: int = max(_doc_count, settings.RAG_TOP_K)

    # ─────────────────────────── page skeleton ───────────────────────────
    create_header()

    # Capture the NiceGUI Client for this browser tab.  The reference is
    # valid for the entire lifetime of the page; we use it to:
    #   • detect disconnect and signal the background stream thread
    #   • guard UI mutations after the client has been deleted (browser
    #     refresh / navigation) to suppress spurious NiceGUI warnings
    _page_client: _NiceGuiClient = ui.context.client

    with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-3"):

        # ── Top bar: breadcrumb + Clear history ──────────────────────────
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                with ui.row().classes("items-center gap-2 text-sm text-gray-500"):
                    ui.link("Dashboard", "/").classes("hover:text-blue-600")
                    ui.label("/")
                    ui.link(job_filename, f"/job/{job_id}").classes(
                        "hover:text-blue-600"
                    )
                    ui.label("/")
                    ui.label("Chat").classes("font-medium text-gray-800")
                ui.label(f"Chat: {job_filename}").classes("text-xl font-bold")

            async def do_clear() -> None:
                db2 = SessionLocal()
                try:
                    delete_chat_history(db2, job_id)
                finally:
                    db2.close()
                ui.navigate.to(f"/query/{job_id}")

            ui.button("Clear History", icon="delete_outline",
                      on_click=do_clear).props("flat color=red size=sm")

        # ── LLM backend selector + RAG k control ────────────────────────
        with ui.card().classes("w-full p-3"):
            with ui.row().classes("items-center gap-4 flex-wrap"):
                ui.label("LLM:").classes("text-sm font-semibold text-gray-700")
                default_mode = "local" if settings.USE_LOCAL_LLM else "hf"
                llm_mode_radio = ui.radio(
                    options=_LLM_OPTIONS,
                    value=default_mode,
                ).props("inline").classes("text-sm")

                hf_token_ok = bool(settings.get_hf_token())
                hf_warn = ui.badge(
                    "⚠ HF_API_TOKEN not set",
                    color="orange",
                ).classes(
                    "text-xs"
                    + ("" if default_mode == "hf" and not hf_token_ok else " hidden")
                )

                def _on_mode(e: object) -> None:
                    mode = getattr(e, "args", e) if not isinstance(e, str) else e
                    if mode == "hf" and not hf_token_ok:
                        hf_warn.classes(remove="hidden")
                    else:
                        hf_warn.classes(add="hidden")

                llm_mode_radio.on("update:model-value", _on_mode)

            # ── RAG k (number of retrieved sources) ──────────────────────
            with ui.row().classes("items-center gap-3 mt-2 flex-wrap"):
                ui.label("Sources to retrieve (k):").classes(
                    "text-sm font-semibold text-gray-700 whitespace-nowrap"
                )
                _k_default = min(settings.RAG_TOP_K, doc_count_max)
                rag_k_input = ui.number(
                    value=_k_default,
                    min=1,
                    max=doc_count_max,
                    step=1,
                    format="%.0f",
                ).props("dense outlined").classes("w-24").tooltip(
                    f"How many transcript chunks to fetch (1–{doc_count_max}). "
                    "Higher k → more context for the LLM, slightly slower retrieval."
                )
                ui.label(f"/ {doc_count_max} available").classes(
                    "text-xs text-gray-500"
                )

        # ── Chat scroll area ─────────────────────────────────────────────
        scroll = ui.scroll_area().classes(
            "w-full border rounded-lg bg-gray-50 p-4"
        ).style("height: 55vh;")
        with scroll:
            messages_col = ui.column().classes("w-full gap-3")

        # ── Helper: render one chat bubble (works for history + live) ────
        def _render_bubble(
            role: str,
            content: str,
            sources: list,
            llm_backend: str,
            stamp: str = "",
        ) -> None:
            sent = role == "user"
            with messages_col:
                with ui.chat_message(
                    name="You" if sent else "MeetBot",
                    stamp=stamp,
                    sent=sent,
                ).classes("w-full"):
                    if content:
                        ui.label(content).classes(
                            "whitespace-pre-wrap text-sm"
                        )
                    if sources and not sent:
                        with ui.expansion("Sources", icon="source").classes(
                            "w-full mt-2 text-xs"
                        ):
                            for idx, s in enumerate(sources, 1):
                                spk = s.get("speaker", "Unknown")
                                t0, t1 = s.get("start", 0), s.get("end", 0)
                                txt = s.get("text", "")
                                rel = s.get("relevance_pct")
                                ms, ss = int(t0 // 60), int(t0 % 60)
                                me, se = int(t1 // 60), int(t1 % 60)
                                with ui.expansion(
                                    f"[{idx}] {spk} — {ms}:{ss:02d}–{me}:{se:02d}"
                                ).classes("w-full"):
                                    if rel is not None:
                                        ui.badge(
                                            f"relevance {rel} %",
                                            color=(
                                                "green" if rel >= 60
                                                else "orange" if rel >= 30
                                                else "red"
                                            ),
                                        ).classes("text-xs mb-1")
                                    ui.label(txt).classes(
                                        "whitespace-pre-wrap text-xs text-gray-600"
                                    )
                    if llm_backend and not sent:
                        ui.badge(
                            _LLM_OPTIONS.get(llm_backend, llm_backend),
                            color="indigo",
                        ).classes("text-xs mt-1")

        # ── Render persistent history ────────────────────────────────────
        for msg in history_msgs:
            d = msg.to_dict()
            stamp = (
                msg.created_at.strftime("%H:%M")
                if msg.created_at
                else ""
            )
            msg_status = d.get("status", "completed")
            if msg_status == "streaming":
                # A "streaming" row means the server was generating when the
                # client last disconnected and the stream had not yet finished.
                # Show the partial content (may be empty) with a badge so the
                # user knows the response was never completed.  On the next
                # fresh query the session continues normally.
                with messages_col:
                    with ui.chat_message(
                        name="MeetBot",
                        stamp=stamp,
                        sent=False,
                    ).classes("w-full"):
                        if d.get("content"):
                            ui.label(d["content"]).classes("whitespace-pre-wrap text-sm")
                        ui.badge(
                            "⏳ Generation was in progress when you left — "
                            "answer may be incomplete",
                            color="orange",
                        ).classes("text-xs mt-1")
            else:
                _render_bubble(
                    msg.role,
                    d["content"],
                    d.get("sources", []),
                    msg.llm_backend or "",
                    stamp=stamp,
                )

        # Scroll to bottom after history is painted
        scroll.scroll_to(percent=1)

        # ── Input bar ────────────────────────────────────────────────────
        is_busy = {"value": False}
        # Holds the threading.Event for the current in-flight query so the
        # Stop button can set it from the NiceGUI async context.
        current_stop_event: dict = {"event": None}

        # ── Client disconnect → signal background stream thread ──────────
        # NiceGUI fires client.on_disconnect() as soon as the browser tab
        # closes, refreshes, or navigates away.  Signalling stop_event here
        # means the background thread stops even if CancelledError is
        # delivered slightly later (or never, depending on NiceGUI version).
        def _on_page_disconnect(_client: _NiceGuiClient | None = None) -> None:
            ev = current_stop_event["event"]
            if ev is not None and not ev.is_set():
                logger.info(
                    "query: client disconnected for job %s — signalling stream stop",
                    job_id,
                )
                ev.set()

        _page_client.on_disconnect(_on_page_disconnect)

        with ui.row().classes("w-full items-end gap-2 mt-1"):
            question_input = ui.textarea(
                placeholder="Ask a question about this transcript…",
            ).classes("flex-1").props("outlined autogrow rows=1 maxrows=6 dense")
            ask_btn = ui.button("", icon="send").props("color=primary round")
            stop_btn = (
                ui.button("", icon="stop_circle")
                .props("color=red round")
                .tooltip("Stop generating")
            )
            stop_btn.set_visibility(False)

            def _on_stop_click() -> None:
                ev = current_stop_event["event"]
                if ev is not None:
                    ev.set()
                stop_btn.disable()

            stop_btn.on_click(_on_stop_click)

        # ── Core query handler ───────────────────────────────────────────
        async def handle_query() -> None:  # noqa: C901
            if is_busy["value"]:
                return

            # Guard helper: returns True only while the browser tab that
            # initiated this query is still connected.  Checking _deleted
            # (the same flag NiceGUI checks internally in check_existence)
            # lets us skip UI mutations instead of generating a flood of
            # "Client has been deleted" warnings in the logs.
            def _live() -> bool:
                return not _page_client._deleted

            question = question_input.value.strip()
            if not question:
                ui.notify("Please enter a question", type="warning")
                return

            selected_mode: str = llm_mode_radio.value

            is_busy["value"] = True
            ask_btn.disable()
            question_input.disable()
            question_input.value = ""

            # Arm a fresh abort event and show the Stop button
            stop_event = threading.Event()
            current_stop_event["event"] = stop_event
            stop_btn.set_visibility(True)
            stop_btn.enable()

            now_stamp = datetime.now().strftime("%H:%M")

            # Persist + render user bubble immediately
            db_u = SessionLocal()
            try:
                create_chat_message(
                    db_u, session_id, role="user", content=question
                )
            finally:
                db_u.close()
            _render_bubble("user", question, [], "", stamp=now_stamp)

            # Create streaming placeholder BEFORE showing the UI bubble.
            # This row is visible on reload (with an "in progress" badge) even
            # if the client disconnects before streaming completes.  We update
            # it to 'completed'/'stopped'/'interrupted' when generation ends.
            _db_ph = SessionLocal()
            try:
                _ph_msg = create_chat_message(
                    _db_ph,
                    session_id,
                    role="assistant",
                    content="",
                    status="streaming",
                    llm_backend=selected_mode,
                )
                _stream_msg_id: str = _ph_msg.id
            except Exception as _ph_exc:
                logger.error(
                    "query: failed to create streaming placeholder: %s", _ph_exc
                )
                _stream_msg_id = ""
            finally:
                _db_ph.close()

            # Exactly-once flag: whichever of (done handler / CancelledError
            # handler) persists first sets this True so the other is a no-op.
            _message_saved: dict = {"value": False}

            # Create streaming assistant bubble
            with messages_col:
                with ui.chat_message(
                    name="MeetBot",
                    stamp=now_stamp,
                    sent=False,
                ).classes("w-full"):
                    typing_row = ui.row().classes("items-center gap-1")
                    with typing_row:
                        ui.spinner("dots", size="xs").classes("text-gray-400")
                        ui.label("Thinking…").classes(
                            "text-xs text-gray-400 italic"
                        )
                    sources_col = ui.column().classes("w-full gap-1")
                    answer_label = ui.label("").classes(
                        "whitespace-pre-wrap text-sm"
                    )
                    backend_badge = ui.badge("", color="indigo").classes(
                        "text-xs mt-1 hidden"
                    )

            scroll.scroll_to(percent=1)

            # Streaming via a thread executor + asyncio.Queue bridge
            event_q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()
            accumulated: list[str] = []
            collected_sources: list = []   # all k candidates
            final_sources: list = []        # post-answer filtered subset

            def _run_stream() -> None:
                from ...services.query_service import QueryService  # local import

                svc = QueryService()
                try:
                    _k_val = int(rag_k_input.value or settings.RAG_TOP_K)
                    _k_clamped = max(1, min(doc_count_max, _k_val))
                    for event in svc.query_stream(
                        question=question,
                        db_dir=job_db_dir,
                        embedding_model=settings.EMBEDDING_MODEL,
                        hf_model=settings.HF_MODEL,
                        k=_k_clamped,
                        llm_mode=selected_mode,
                        abort_event=stop_event,
                    ):
                        # IMPORTANT: do NOT check stop_event here before the put.
                        # query_stream() already checks abort_event internally and
                        # always emits a final 'done' event (with stopped=True when
                        # aborted).  Checking stop_event here would silently drop
                        # that 'done' event, causing the placeholder DB row to
                        # remain 'streaming' forever and the answer to be lost.
                        # Break only when the queue put fails (loop torn down).
                        try:
                            asyncio.run_coroutine_threadsafe(
                                event_q.put(event), loop
                            ).result(timeout=10)
                        except Exception:
                            # The asyncio loop is shutting down (page navigated
                            # away / refreshed).  Stop streaming cleanly.
                            stop_event.set()
                            break
                except Exception as exc:  # noqa: BLE001
                    try:
                        asyncio.run_coroutine_threadsafe(
                            event_q.put({"type": "error", "data": str(exc)}), loop
                        ).result(timeout=5)
                    except Exception:
                        pass
                finally:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            event_q.put({"type": "_sentinel"}), loop
                        ).result(timeout=5)
                    except Exception:
                        pass  # Loop already gone; sentinel not needed

            try:
                stream_future = loop.run_in_executor(None, _run_stream)

                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_q.get(), timeout=120.0
                        )
                    except asyncio.TimeoutError:
                        answer_label.set_text(
                            "[Timed out waiting for response]"
                        )
                        stop_event.set()
                        break

                    etype = event.get("type")

                    if etype == "_sentinel":
                        break

                    elif etype == "sources":
                        typing_row.set_visibility(False)
                        collected_sources = event.get("data", [])
                        if collected_sources:
                            with sources_col:
                                # Temporary "Retrieving…" label while LLM streams.
                                # The done handler will replace this with filtered sources.
                                with ui.expansion(
                                    f"Candidates ({len(collected_sources)} retrieved)",
                                    icon="manage_search",
                                ).classes("w-full text-xs text-gray-400"):
                                    for idx, s in enumerate(
                                        collected_sources, 1
                                    ):
                                        spk = s.get("speaker", "Unknown")
                                        t0 = s.get("start", 0)
                                        t1 = s.get("end", 0)
                                        ms = int(t0 // 60)
                                        ss = int(t0 % 60)
                                        me = int(t1 // 60)
                                        se = int(t1 % 60)
                                        ui.label(
                                            f"[{idx}] {spk} — "
                                            f"{ms}:{ss:02d}–{me}:{se:02d}"
                                        ).classes("text-xs text-gray-400")

                    elif etype == "token":
                        accumulated.append(event.get("data", ""))
                        answer_label.set_text("".join(accumulated))
                        scroll.scroll_to(percent=1)

                    elif etype == "done":
                        full_answer = "".join(accumulated)
                        was_stopped = event.get("stopped", False)
                        llm_be = event.get("llm_backend", selected_mode)
                        # Filtered sources from post-answer embedding comparison
                        final_sources = event.get("filtered_sources") or collected_sources
                        filtering_ok  = event.get("filtering_available", True)

                        # ── 1. DB save FIRST, before touching any UI element ──
                        # Persisting before DOM mutations means a socket drop
                        # between streaming completion and UI rebuild cannot
                        # lose the answer.  The save is isolated so any DB /
                        # JSON serialisation error is logged + badged without
                        # propagating out of this handler.
                        full_answer_to_save = (
                            (full_answer + "\n*(generation stopped)*" if full_answer
                             else "*(generation stopped)*")
                            if was_stopped else full_answer
                        )
                        _final_status = "stopped" if was_stopped else "completed"
                        _db_save_ok = True
                        if not _message_saved["value"] and _stream_msg_id:
                            db_a = SessionLocal()
                            try:
                                update_chat_message(
                                    db_a,
                                    _stream_msg_id,
                                    content=full_answer_to_save,
                                    status=_final_status,
                                    sources=final_sources,
                                    llm_backend=llm_be,
                                )
                                _message_saved["value"] = True
                                logger.info(
                                    "query: finalised assistant message %s "
                                    "(%d chars, status=%s)",
                                    _stream_msg_id[:8], len(full_answer_to_save),
                                    _final_status,
                                )
                            except Exception as _save_exc:
                                _db_save_ok = False
                                logger.error(
                                    "query: failed to finalise assistant message "
                                    "%s: %s",
                                    _stream_msg_id[:8], _save_exc, exc_info=True,
                                )
                            finally:
                                db_a.close()

                        # ── 2. UI updates — only if client still connected ────
                        # All DOM mutations gated on _live() so a browser
                        # refresh racing with the done event does not flood
                        # logs with NiceGUI "Client has been deleted" warnings.
                        # The answer is already safe in the DB regardless.
                        if _live():
                            # Replace candidate sources panel with filtered sources
                            sources_col.clear()
                            if final_sources:
                                with sources_col:
                                    with ui.expansion(
                                        f"Sources ({len(final_sources)})", icon="source"
                                    ).classes("w-full text-xs"):
                                        if not filtering_ok:
                                            ui.badge(
                                                "source filtering unavailable",
                                                color="grey",
                                            ).classes("text-xs mb-1")
                                        for idx, s in enumerate(final_sources, 1):
                                            spk = s.get("speaker", "Unknown")
                                            t0  = s.get("start", 0)
                                            t1  = s.get("end", 0)
                                            txt = s.get("text", "")
                                            rel = s.get("answer_relevance_pct",
                                                        s.get("relevance_pct"))
                                            ms, ss2 = int(t0 // 60), int(t0 % 60)
                                            me, se  = int(t1 // 60), int(t1 % 60)
                                            with ui.expansion(
                                                f"[{idx}] {spk} — {ms}:{ss2:02d}–{me}:{se:02d}"
                                            ).classes("w-full"):
                                                if rel is not None:
                                                    ui.badge(
                                                        f"relevance {rel} %",
                                                        color=(
                                                            "green"  if rel >= 60
                                                            else "orange" if rel >= 30
                                                            else "red"
                                                        ),
                                                    ).classes("text-xs mb-1")
                                                ui.label(txt).classes(
                                                    "whitespace-pre-wrap "
                                                    "text-xs text-gray-600"
                                                )

                            if was_stopped:
                                answer_label.set_text("".join(accumulated))
                                with messages_col:
                                    ui.badge("⏹ Generation stopped", color="grey").classes(
                                        "text-xs mt-1"
                                    )
                            else:
                                answer_label.set_text(full_answer)
                                backend_badge.set_text(
                                    _LLM_OPTIONS.get(llm_be, llm_be)
                                )
                                backend_badge.classes(remove="hidden")

                            if not _db_save_ok:
                                with messages_col:
                                    ui.badge(
                                        "⚠ Answer not saved — check server logs",
                                        color="orange",
                                    ).classes("text-xs mt-1")
                        break

                    elif etype == "error":
                        typing_row.set_visibility(False)
                        err_msg = event.get("data", "Unknown error")
                        answer_label.set_text(f"Error: {err_msg}")
                        answer_label.classes(add="text-red-600")
                        break

                await stream_future

            except asyncio.CancelledError:
                # Page was refreshed or navigated away while streaming.
                # Signal the background thread to stop, persist whatever we
                # have so the history survives reload, then re-raise so the
                # NiceGUI task lifecycle is handled correctly.
                logger.info(
                    "Query page disconnected mid-stream for job %s — "
                    "persisting partial answer (%d tokens)",
                    job_id, len(accumulated),
                )
                stop_event.set()
                if not _message_saved["value"] and _stream_msg_id:
                    partial_text = (
                        "".join(accumulated) + "\n*(generation interrupted)*"
                        if accumulated else ""
                    )
                    _db = SessionLocal()
                    try:
                        update_chat_message(
                            _db,
                            _stream_msg_id,
                            content=partial_text,
                            status="interrupted",
                            sources=collected_sources,
                            llm_backend=selected_mode,
                        )
                        _message_saved["value"] = True
                    except Exception as _db_exc:
                        logger.warning(
                            "Could not persist partial answer for msg %s: %s",
                            _stream_msg_id[:8], _db_exc,
                        )
                    finally:
                        _db.close()
                raise  # Let NiceGUI cancel the task normally

            except Exception as exc:  # noqa: BLE001
                logger.error("Streaming query failed: %s", exc, exc_info=True)
                try:
                    answer_label.set_text(f"Error: {exc}")
                    answer_label.classes(add="text-red-600")
                except Exception:
                    pass

            finally:
                # Always update pure-Python state regardless of client lifecycle.
                is_busy["value"] = False
                current_stop_event["event"] = None
                # Gate all NiceGUI element mutations on _live() to avoid the
                # "Client has been deleted but is still being used" warning that
                # fires whenever the browser navigates away while this coroutine
                # is still unwinding its finally block.
                if _live():
                    typing_row.set_visibility(False)
                    ask_btn.enable()
                    question_input.enable()
                    stop_btn.set_visibility(False)
                    stop_btn.enable()
                    scroll.scroll_to(percent=1)

        ask_btn.on_click(handle_query)

        async def _on_enter(e: object) -> None:
            args = getattr(e, "args", {}) or {}
            if not args.get("shiftKey"):
                await handle_query()

        question_input.on("keydown.enter", _on_enter)
