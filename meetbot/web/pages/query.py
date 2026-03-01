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

from nicegui import ui

from ..auth import get_current_user_id
from ..components.nav import create_header
from ...config import settings
from ...db.crud import (
    create_chat_message,
    delete_chat_history,
    get_chat_messages,
    get_job,
    get_or_create_chat_session,
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

    # ─────────────────────────── page skeleton ───────────────────────────
    create_header()

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

        # ── LLM backend selector ─────────────────────────────────────────
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
                                ms, ss = int(t0 // 60), int(t0 % 60)
                                me, se = int(t1 // 60), int(t1 % 60)
                                with ui.expansion(
                                    f"[{idx}] {spk} — {ms}:{ss:02d}–{me}:{se:02d}"
                                ).classes("w-full"):
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
            _render_bubble(
                msg.role,
                msg.content,
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
            collected_sources: list = []

            def _run_stream() -> None:
                from ...services.query_service import QueryService  # local import

                svc = QueryService()
                try:
                    for event in svc.query_stream(
                        question=question,
                        db_dir=job_db_dir,
                        embedding_model=settings.EMBEDDING_MODEL,
                        hf_model=settings.HF_MODEL,
                        k=settings.RAG_TOP_K,
                        llm_mode=selected_mode,
                        abort_event=stop_event,
                    ):
                        asyncio.run_coroutine_threadsafe(
                            event_q.put(event), loop
                        ).result(timeout=10)
                except Exception as exc:  # noqa: BLE001
                    asyncio.run_coroutine_threadsafe(
                        event_q.put({"type": "error", "data": str(exc)}), loop
                    ).result(timeout=5)
                finally:
                    asyncio.run_coroutine_threadsafe(
                        event_q.put({"type": "_sentinel"}), loop
                    ).result(timeout=5)

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
                        break

                    etype = event.get("type")

                    if etype == "_sentinel":
                        break

                    elif etype == "sources":
                        typing_row.set_visibility(False)
                        collected_sources = event.get("data", [])
                        if collected_sources:
                            with sources_col:
                                with ui.expansion(
                                    "Sources", icon="source"
                                ).classes("w-full text-xs"):
                                    for idx, s in enumerate(
                                        collected_sources, 1
                                    ):
                                        spk = s.get("speaker", "Unknown")
                                        t0 = s.get("start", 0)
                                        t1 = s.get("end", 0)
                                        txt = s.get("text", "")
                                        ms = int(t0 // 60)
                                        ss = int(t0 % 60)
                                        me = int(t1 // 60)
                                        se = int(t1 % 60)
                                        with ui.expansion(
                                            f"[{idx}] {spk} — "
                                            f"{ms}:{ss:02d}–{me}:{se:02d}"
                                        ).classes("w-full"):
                                            ui.label(txt).classes(
                                                "whitespace-pre-wrap "
                                                "text-xs text-gray-600"
                                            )

                    elif etype == "token":
                        accumulated.append(event.get("data", ""))
                        answer_label.set_text("".join(accumulated))
                        scroll.scroll_to(percent=1)

                    elif etype == "done":
                        full_answer = "".join(accumulated)
                        was_stopped = event.get("stopped", False)
                        llm_be = event.get("llm_backend", selected_mode)
                        if was_stopped:
                            answer_label.set_text(
                                ("\n".join(accumulated) if accumulated else "")
                            )
                            # Append a subtle stopped indicator
                            with messages_col:
                                ui.badge("⏹ Generation stopped", color="grey").classes(
                                    "text-xs mt-1"
                                )
                            full_answer_to_save = (
                                full_answer + "\n*(generation stopped)*"
                                if full_answer
                                else "*(generation stopped)*"
                            )
                        else:
                            full_answer_to_save = full_answer
                            backend_badge.set_text(
                                _LLM_OPTIONS.get(llm_be, llm_be)
                            )
                            backend_badge.classes(remove="hidden")
                        # Persist assistant message (partial or full)
                        db_a = SessionLocal()
                        try:
                            create_chat_message(
                                db_a,
                                session_id,
                                role="assistant",
                                content=full_answer_to_save,
                                sources=collected_sources,
                                llm_backend=llm_be,
                            )
                        finally:
                            db_a.close()
                        break

                    elif etype == "error":
                        typing_row.set_visibility(False)
                        err_msg = event.get("data", "Unknown error")
                        answer_label.set_text(f"Error: {err_msg}")
                        answer_label.classes(add="text-red-600")
                        break

                await stream_future

            except Exception as exc:  # noqa: BLE001
                logger.error("Streaming query failed: %s", exc, exc_info=True)
                answer_label.set_text(f"Error: {exc}")
                answer_label.classes(add="text-red-600")

            finally:
                typing_row.set_visibility(False)
                is_busy["value"] = False
                ask_btn.enable()
                question_input.enable()
                stop_btn.set_visibility(False)
                stop_btn.enable()
                current_stop_event["event"] = None
                scroll.scroll_to(percent=1)

        ask_btn.on_click(handle_query)

        async def _on_enter(e: object) -> None:
            args = getattr(e, "args", {}) or {}
            if not args.get("shiftKey"):
                await handle_query()

        question_input.on("keydown.enter", _on_enter)
