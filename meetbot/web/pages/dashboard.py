"""
Dashboard page — shows all jobs with status, progress, and actions.

Provides:
- Job list with real-time status updates
- Status badges (color-coded)
- Links to job detail and query pages
- Job deletion
"""

import logging
from datetime import datetime, timezone

from nicegui import ui, app

from ..auth import get_current_user_id
from ..components.nav import create_header
from ...db.database import get_session
from ...db.crud import get_jobs_for_user, delete_job, get_job
from ...db.models import JobStatus
from ...workers.progress import progress_manager

logger = logging.getLogger(__name__)

# Status badge colors
STATUS_COLORS = {
    JobStatus.PENDING: "grey",
    JobStatus.TRANSCRIBING: "blue",
    JobStatus.DIARIZING: "purple",
    JobStatus.ALIGNING: "orange",
    JobStatus.INDEXING: "cyan",
    JobStatus.COMPLETED: "green",
    JobStatus.FAILED: "red",
}

STATUS_ICONS = {
    JobStatus.PENDING: "hourglass_empty",
    JobStatus.TRANSCRIBING: "mic",
    JobStatus.DIARIZING: "group",
    JobStatus.ALIGNING: "merge_type",
    JobStatus.INDEXING: "search",
    JobStatus.COMPLETED: "check_circle",
    JobStatus.FAILED: "error",
}


def _format_datetime(dt: datetime | None) -> str:
    """Format a datetime for display."""
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_duration(seconds: float | None) -> str:
    """Format duration in seconds to MM:SS."""
    if seconds is None or seconds == 0:
        return "—"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def _format_file_size(size_bytes: int | None) -> str:
    """Format file size in bytes to human-readable."""
    if size_bytes is None:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


@ui.page("/")
def dashboard_page() -> None:
    """Render the dashboard page with job list."""
    user_id = get_current_user_id()
    if not user_id:
        ui.navigate.to("/login")
        return

    create_header()

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-4"):
        # Page header
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Dashboard").classes("text-2xl font-bold")
            ui.button(
                "New Upload",
                icon="add",
                on_click=lambda: ui.navigate.to("/upload"),
            ).props("color=primary")

        # Job list container (refreshable)
        job_container = ui.column().classes("w-full gap-3")

        def refresh_jobs() -> None:
            """Load and display all jobs."""
            job_container.clear()

            SessionLocal = get_session()
            db = SessionLocal()
            try:
                jobs = get_jobs_for_user(db, user_id, limit=50)

                if not jobs:
                    with job_container:
                        with ui.card().classes("w-full p-8 items-center"):
                            ui.icon("mic_none").classes("text-6xl text-gray-300")
                            ui.label("No transcriptions yet").classes(
                                "text-xl text-gray-400 mt-2"
                            )
                            ui.button(
                                "Upload your first audio file",
                                icon="upload_file",
                                on_click=lambda: ui.navigate.to("/upload"),
                            ).props("color=primary outline").classes("mt-4")
                    return

                with job_container:
                    for job in jobs:
                        _render_job_card(job, refresh_jobs)
            finally:
                db.close()

        refresh_jobs()

        # Auto-refresh for active jobs
        timer = ui.timer(3.0, refresh_jobs)


def _render_job_card(job, refresh_callback) -> None:
    """Render a single job card."""
    color = STATUS_COLORS.get(job.status, "grey")
    icon = STATUS_ICONS.get(job.status, "help")

    with ui.card().classes("w-full p-4 hover:shadow-md transition-shadow"):
        with ui.row().classes("w-full items-center justify-between"):
            # Left: status icon + file info
            with ui.row().classes("items-center gap-3 flex-1"):
                ui.icon(icon).classes(f"text-2xl text-{color}-600")
                with ui.column().classes("gap-0"):
                    ui.label(job.original_filename).classes(
                        "text-base font-medium"
                    )
                    with ui.row().classes("gap-4 text-xs text-gray-500"):
                        ui.label(f"Created: {_format_datetime(job.created_at)}")
                        ui.label(f"Duration: {_format_duration(job.duration_seconds)}")
                        ui.label(f"Size: {_format_file_size(job.file_size)}")

            # Center: status badge + progress
            with ui.column().classes("items-center gap-1 min-w-[140px]"):
                ui.badge(
                    job.status.value.upper(),
                    color=color,
                ).props("outline")

                if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.PENDING):
                    ui.linear_progress(
                        value=job.progress / 100,
                        show_value=False,
                    ).classes("w-32").props(f"color={color}")
                    if job.progress_message:
                        ui.label(job.progress_message).classes(
                            "text-xs text-gray-500 truncate max-w-[140px]"
                        )

            # Right: action buttons
            with ui.row().classes("gap-1"):
                if job.status == JobStatus.COMPLETED:
                    ui.button(
                        icon="visibility",
                        on_click=lambda j=job: ui.navigate.to(f"/job/{j.id}"),
                    ).props("flat round color=primary size=sm").tooltip("View Transcript")

                    if job.db_dir:
                        ui.button(
                            icon="search",
                            on_click=lambda j=job: ui.navigate.to(
                                f"/query/{j.id}"
                            ),
                        ).props("flat round color=teal size=sm").tooltip("Query")

                elif job.status == JobStatus.FAILED:
                    ui.button(
                        icon="info",
                        on_click=lambda j=job: _show_error_dialog(j),
                    ).props("flat round color=red size=sm").tooltip("View Error")

                # Delete button (always available)
                ui.button(
                    icon="delete",
                    on_click=lambda j=job: _confirm_delete(j, refresh_callback),
                ).props("flat round color=grey size=sm").tooltip("Delete")


def _show_error_dialog(job) -> None:
    """Show error details for a failed job."""
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label("Error Details").classes("text-lg font-semibold")
        ui.separator()
        ui.label(job.error_message or "Unknown error").classes(
            "text-sm text-red-600 whitespace-pre-wrap"
        )
        ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


def _confirm_delete(job, refresh_callback) -> None:
    """Show delete confirmation dialog."""
    with ui.dialog() as dialog, ui.card().classes("w-80"):
        ui.label("Delete Job?").classes("text-lg font-semibold")
        ui.label(
            f'Are you sure you want to delete "{job.original_filename}"?'
        ).classes("text-sm text-gray-600")
        ui.label("This cannot be undone.").classes("text-xs text-red-500")

        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def do_delete():
                SessionLocal = get_session()
                db = SessionLocal()
                try:
                    # delete_job() cascades to segments, chat session/messages,
                    # on-disk files (audio upload, result JSONs) and vector store.
                    delete_job(db, job.id)
                finally:
                    db.close()
                dialog.close()
                ui.notify(f"Deleted: {job.original_filename}", type="info")
                refresh_callback()

            ui.button(
                "Delete", on_click=do_delete, color="red"
            ).props("flat")

    dialog.open()
