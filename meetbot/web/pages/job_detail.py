"""
Job detail page — transcript viewer, audio player, and editing.

Provides:
- Audio playback with seek-to-segment
- Full transcript display with editable speakers and text
- Pipeline progress display (if still processing)
- Export options
"""

import logging
from pathlib import Path

import httpx
from nicegui import ui, app

from ..auth import get_current_user_id
from ..components.nav import create_header
from ..components.audio_player import AudioPlayer
from ..components.transcript_table import TranscriptTable
from ..components.progress_bar import ProgressDisplay
from ...config import settings
from ...db.database import get_session
from ...db.crud import get_job
from ...db.models import JobStatus

logger = logging.getLogger(__name__)


@ui.page("/job/{job_id}")
def job_detail_page(job_id: str) -> None:
    """Render the job detail page."""
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

        job_status = job.status
        job_filename = job.original_filename
        job_stored_filename = job.filename
        job_duration = job.duration_seconds
        job_db_dir = job.db_dir
        job_error = job.error_message
    finally:
        db.close()

    create_header()

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-4"):
        # Breadcrumb
        with ui.row().classes("items-center gap-2 text-sm text-gray-500"):
            ui.link("Dashboard", "/").classes("hover:text-blue-600")
            ui.label("/")
            ui.label(job_filename).classes("font-medium text-gray-800")

        # Job header
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label(job_filename).classes("text-2xl font-bold")
                if job_duration:
                    minutes = int(job_duration // 60)
                    secs = int(job_duration % 60)
                    ui.label(f"Duration: {minutes}:{secs:02d}").classes(
                        "text-sm text-gray-500"
                    )

            # Action buttons (completed-only quick actions)
            with ui.row().classes("gap-2"):
                if job_status == JobStatus.COMPLETED and job_db_dir:
                    ui.button(
                        "Query",
                        icon="search",
                        on_click=lambda: ui.navigate.to(f"/query/{job_id}"),
                    ).props("color=teal outline")

                if job_status == JobStatus.COMPLETED:
                    async def _do_reindex(jid: str = job_id) -> None:
                        with ui.dialog() as confirm_dialog, ui.card():
                            ui.label(
                                "This will rebuild the search index from the current "
                                "transcript. Ongoing queries will be unavailable during "
                                "reindexing. Continue?"
                            ).classes("text-base")
                            with ui.row().classes("justify-end gap-2 mt-2"):
                                ui.button(
                                    "Cancel",
                                    on_click=lambda: confirm_dialog.submit(False),
                                ).props("flat")
                                ui.button(
                                    "Reindex",
                                    on_click=lambda: confirm_dialog.submit(True),
                                ).props("color=orange")

                        confirmed = await confirm_dialog
                        if not confirmed:
                            return
                        try:
                            async with httpx.AsyncClient() as client:
                                resp = await client.post(
                                    f"http://localhost:{settings.WEB_PORT}"
                                    f"/api/jobs/{jid}/reindex",
                                    timeout=10,
                                )
                            if resp.status_code == 202:
                                ui.notify("Reindex started", type="positive")
                                ui.navigate.to(f"/job/{jid}")
                            else:
                                detail = resp.json().get("detail", resp.text)
                                ui.notify(f"Reindex failed: {detail}", type="negative")
                        except Exception as exc:
                            ui.notify(f"Request error: {exc}", type="negative")

                    ui.button(
                        "Reindex",
                        icon="model_training",
                        on_click=_do_reindex,
                    ).props("color=orange outline").tooltip(
                        "Rebuild the search index from the current transcript"
                    )

                    with ui.row().classes("gap-1"):
                        ui.button(
                            "Transcript",
                            icon="download",
                            on_click=lambda jid=job_id: ui.navigate.to(
                                f"/api/jobs/{jid}/download?type=aligned"
                            ),
                        ).props("color=primary outline size=sm").tooltip(
                            "Aligned transcript JSON"
                        )
                        ui.button(
                            "Whisper",
                            icon="mic",
                            on_click=lambda jid=job_id: ui.navigate.to(
                                f"/api/jobs/{jid}/download?type=transcription"
                            ),
                        ).props("color=secondary outline size=sm").tooltip(
                            "Raw Whisper transcription JSON"
                        )
                        ui.button(
                            "Speakers",
                            icon="group",
                            on_click=lambda jid=job_id: ui.navigate.to(
                                f"/api/jobs/{jid}/download?type=diarization"
                            ),
                        ).props("color=secondary outline size=sm").tooltip(
                            "Speaker diarization JSON"
                        )

        # Content area depends on status
        _cancellable = {
            JobStatus.PENDING,
            JobStatus.TRANSCRIBING,
            JobStatus.DIARIZING,
            JobStatus.ALIGNING,
            JobStatus.INDEXING,
            JobStatus.REINDEXING,
        }

        if job_status in _cancellable:
            # ──────────────── Prominent processing action banner ────────────────
            # Placed at the TOP of the content area so it cannot be missed.
            with ui.card().classes(
                "w-full p-4 bg-orange-50 border border-orange-200"
            ):
                with ui.row().classes("w-full items-center justify-between flex-wrap gap-3"):
                    with ui.row().classes("items-center gap-3"):
                        ui.spinner("dots", size="md").classes("text-orange-500")
                        with ui.column().classes("gap-0"):
                            stage_label = (
                                "Reindexing transcript" if job_status == JobStatus.REINDEXING
                                else {
                                    JobStatus.PENDING:     "Queued — waiting to start",
                                    JobStatus.TRANSCRIBING: "Transcribing audio",
                                    JobStatus.DIARIZING:   "Identifying speakers",
                                    JobStatus.ALIGNING:    "Aligning transcript",
                                    JobStatus.INDEXING:    "Building search index",
                                }.get(job_status, "Processing")
                            )
                            ui.label(stage_label).classes(
                                "text-base font-semibold text-orange-700"
                            )
                            ui.label(
                                "Processing will stop cleanly at the end of the current stage."
                            ).classes("text-xs text-orange-500")

                    async def _do_cancel_content(jid: str = job_id) -> None:
                        with ui.dialog() as dlg, ui.card():
                            ui.label(
                                "Stop processing this job?"
                            ).classes("text-base font-semibold")
                            ui.label(
                                "The current stage will finish before the worker stops. "
                                "You can restart later — Whisper and Pyannote results are cached."
                            ).classes("text-sm text-gray-600 mt-1")
                            with ui.row().classes("justify-end gap-2 mt-3"):
                                ui.button(
                                    "Keep running",
                                    on_click=lambda: dlg.submit(False),
                                ).props("flat")
                                ui.button(
                                    "Stop processing",
                                    icon="stop_circle",
                                    on_click=lambda: dlg.submit(True),
                                ).props("color=red")
                        confirmed = await dlg
                        if not confirmed:
                            return
                        cancel_content_btn.props(add="loading")
                        cancel_content_btn.disable()
                        try:
                            async with httpx.AsyncClient() as hc:
                                resp = await hc.post(
                                    f"http://localhost:{settings.WEB_PORT}"
                                    f"/api/jobs/{jid}/cancel",
                                    timeout=10,
                                )
                            if resp.status_code == 200:
                                ui.notify("Cancellation requested", type="positive")
                                ui.navigate.to(f"/job/{jid}")
                            else:
                                detail = resp.json().get("detail", resp.text)
                                ui.notify(f"Cancel failed: {detail}", type="negative")
                                cancel_content_btn.props(remove="loading")
                                cancel_content_btn.enable()
                        except Exception as exc:
                            ui.notify(f"Request error: {exc}", type="negative")
                            cancel_content_btn.props(remove="loading")
                            cancel_content_btn.enable()

                    cancel_content_btn = ui.button(
                        "Stop Processing",
                        icon="stop_circle",
                        on_click=_do_cancel_content,
                    ).props("color=red").classes("text-sm")

            # Progress display below the banner
            def on_complete():
                ui.notify("Processing complete! Refreshing...", type="positive")
                ui.navigate.to(f"/job/{job_id}")

            def on_fail(error):
                ui.notify(f"Processing failed: {error}", type="negative")

            ProgressDisplay(
                job_id=job_id,
                on_complete=on_complete,
                on_fail=on_fail,
            )

        elif job_status == JobStatus.CANCELLED:
            with ui.card().classes("w-full p-4 bg-yellow-50 border border-yellow-200"):
                with ui.row().classes("w-full items-start justify-between flex-wrap gap-3"):
                    with ui.column().classes("gap-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("cancel", size="sm").classes("text-yellow-600")
                            ui.label("Processing Cancelled").classes(
                                "text-lg font-semibold text-yellow-700"
                            )
                        ui.label(
                            "This job was stopped before it could finish. "
                            "Whisper and Pyannote outputs are cached, so a restart is fast."
                        ).classes("text-sm text-yellow-600")

                    async def _do_restart_cancelled(jid: str = job_id) -> None:
                        restart_cancelled_btn.props(add="loading")
                        restart_cancelled_btn.disable()
                        try:
                            async with httpx.AsyncClient() as hc:
                                resp = await hc.post(
                                    f"http://localhost:{settings.WEB_PORT}"
                                    f"/api/jobs/{jid}/restart",
                                    timeout=10,
                                )
                            if resp.status_code == 202:
                                ui.notify("Job restarted", type="positive")
                                ui.navigate.to(f"/job/{jid}")
                            else:
                                detail = resp.json().get("detail", resp.text)
                                ui.notify(f"Restart failed: {detail}", type="negative")
                                restart_cancelled_btn.props(remove="loading")
                                restart_cancelled_btn.enable()
                        except Exception as exc:
                            ui.notify(f"Request error: {exc}", type="negative")
                            restart_cancelled_btn.props(remove="loading")
                            restart_cancelled_btn.enable()

                    restart_cancelled_btn = ui.button(
                        "Restart Processing",
                        icon="replay",
                        on_click=_do_restart_cancelled,
                    ).props("color=blue").classes("text-sm self-center")

        elif job_status == JobStatus.FAILED:
            with ui.card().classes("w-full p-4 bg-red-50 border border-red-200"):
                with ui.row().classes("w-full items-start justify-between flex-wrap gap-3"):
                    with ui.column().classes("gap-1 flex-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("error", size="sm").classes("text-red-600")
                            ui.label("Processing Failed").classes(
                                "text-lg font-semibold text-red-700"
                            )
                        ui.label(job_error or "Unknown error").classes(
                            "text-sm text-red-600 whitespace-pre-wrap"
                        )

                    async def _do_restart_failed(jid: str = job_id) -> None:
                        restart_failed_btn.props(add="loading")
                        restart_failed_btn.disable()
                        try:
                            async with httpx.AsyncClient() as hc:
                                resp = await hc.post(
                                    f"http://localhost:{settings.WEB_PORT}"
                                    f"/api/jobs/{jid}/restart",
                                    timeout=10,
                                )
                            if resp.status_code == 202:
                                ui.notify("Job restarted", type="positive")
                                ui.navigate.to(f"/job/{jid}")
                            else:
                                detail = resp.json().get("detail", resp.text)
                                ui.notify(f"Restart failed: {detail}", type="negative")
                                restart_failed_btn.props(remove="loading")
                                restart_failed_btn.enable()
                        except Exception as exc:
                            ui.notify(f"Request error: {exc}", type="negative")
                            restart_failed_btn.props(remove="loading")
                            restart_failed_btn.enable()

                    restart_failed_btn = ui.button(
                        "Restart Processing",
                        icon="replay",
                        on_click=_do_restart_failed,
                    ).props("color=blue").classes("text-sm self-center")

        elif job_status == JobStatus.COMPLETED:
            # Completed — show audio + transcript

            # Audio player
            upload_dir = (
                Path(settings.OUTPUT_DIR).parent / "data" / "uploads"
            )
            audio_path = upload_dir / job_stored_filename

            audio_player = None
            if audio_path.exists():
                # Serve the audio file
                audio_url = f"/audio/{job_stored_filename}"
                app.add_static_files("/audio", str(upload_dir))
                audio_player = AudioPlayer(audio_url)

            ui.separator()

            # Transcript
            ui.label("Transcript").classes("text-lg font-semibold mt-2")

            def on_seek(seconds: float) -> None:
                if audio_player:
                    audio_player.seek_to(seconds)

            TranscriptTable(job_id=job_id, on_seek=on_seek)



