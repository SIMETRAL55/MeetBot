"""
Upload page — audio file upload with options.

Provides:
- Drag-and-drop file upload with validation
- Language and speaker count options
- File type and size validation
- Upload progress indication
"""

import asyncio
import logging
import uuid
from pathlib import Path

from nicegui import ui, app, events

from ..auth import get_current_user_id
from ..components.nav import create_header
from ...config import settings
from ...db.database import get_session
from ...db.crud import create_job
from ...workers.queue import job_queue

logger = logging.getLogger(__name__)


def _get_upload_dir() -> Path:
    """Get the upload directory."""
    base = Path(settings.OUTPUT_DIR).parent
    upload_dir = base / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@ui.page("/upload")
def upload_page() -> None:
    """Render the upload page."""
    user_id = get_current_user_id()
    if not user_id:
        ui.navigate.to("/login")
        return

    create_header()

    with ui.column().classes("w-full max-w-2xl mx-auto p-6 gap-6"):
        ui.label("Upload Audio").classes("text-2xl font-bold")

        with ui.card().classes("w-full p-6"):
            # Upload area
            ui.label("Audio File").classes("text-sm font-medium text-gray-600")
            upload_widget = ui.upload(
                label="Drop audio file here or click to browse",
                on_upload=lambda e: None,  # Handled in submit
                auto_upload=False,
                max_file_size=settings.get_max_upload_bytes(),
                max_files=1,
            ).classes("w-full").props(
                f'accept="{",".join(settings.get_allowed_extensions())}"'
            )

            allowed_exts = ", ".join(settings.get_allowed_extensions())
            ui.label(
                f"Supported formats: {allowed_exts} — "
                f"Max size: {settings.MAX_UPLOAD_SIZE_MB} MB"
            ).classes("text-xs text-gray-400 mt-1")

            ui.separator().classes("my-4")

            # Options
            ui.label("Options").classes("text-sm font-medium text-gray-600")

            with ui.row().classes("w-full gap-4"):
                language_select = ui.select(
                    label="Language",
                    options={
                        "": "Auto-detect",
                        "ja": "Japanese (日本語)",
                        "en": "English",
                        "zh": "Chinese (中文)",
                        "ko": "Korean (한국어)",
                    },
                    value="",
                ).classes("flex-1")

            with ui.row().classes("w-full gap-4"):
                min_speakers = ui.number(
                    label="Min Speakers",
                    value=None,
                    min=1,
                    max=20,
                    step=1,
                     ).classes("flex-1").props("clearable")
               format="%.0f",

                max_speakers = ui.number(
                    label="Max Speakers",
                    value=None,
                    min=1,
                    max=20,
                    step=1,
                    format="%.0f",
                ).classes("flex-1").props("clearable")

            ui.separator().classes("my-4")

            # Status area
            status_label = ui.label("").classes("text-sm hidden")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes(
                "w-full hidden"
            )

            # Track whether a file has been staged in the upload widget
            # (set to True the first time on_file_upload fires successfully)
            _upload_state: dict = {"file_staged": False}

            # Submit button — triggers the queued upload
            async def handle_submit() -> None:
                """Trigger the queued upload. Actual work is in on_file_upload."""
                submit_btn.disable()
                status_label.classes(remove="hidden")
                progress_bar.classes(remove="hidden")
                status_label.text = "Uploading…"
                progress_bar.value = 0.2

                # Ask the upload widget to push its queued file to the server.
                # on_file_upload fires once the file arrives.
                upload_widget.run_method("upload")

            submit_btn = ui.button(
                "Start Processing",
                icon="play_arrow",
                on_click=handle_submit,
            ).classes("w-full mt-2").props("color=primary size=lg")

            # Handle actual file upload event
            async def on_file_upload(e: events.UploadEventArguments) -> None:
                """
                Handle the uploaded file.

                NiceGUI delivers UploadEventArguments with a single attribute:
                    e.file  — a FileUpload (SmallFileUpload or LargeFileUpload)

                FileUpload attributes:
                    e.file.name          — original filename (str)
                    e.file.content_type  — MIME type (str)
                    e.file.size()        — file size in bytes (int)
                    await e.file.read()  — all bytes (async)
                    await e.file.save(path) — stream to disk (async, efficient)
                """
                try:
                    original_name = e.file.name
                    file_ext = Path(original_name).suffix.lower()

                    logger.info(
                        f"Upload received: {original_name!r} "
                        f"type={e.file.content_type!r}"
                    )

                    # Validate extension
                    if file_ext not in settings.get_allowed_extensions():
                        ui.notify(
                            f"Unsupported file type: {file_ext}. "
                            f"Allowed: {', '.join(settings.get_allowed_extensions())}",
                            type="negative",
                        )
                        submit_btn.enable()
                        return

                    # Validate size before persisting
                    file_size = e.file.size()
                    max_bytes = settings.get_max_upload_bytes()
                    if file_size > max_bytes:
                        ui.notify(
                            f"File too large: {file_size // 1_000_000} MB "
                            f"(max {settings.MAX_UPLOAD_SIZE_MB} MB)",
                            type="negative",
                        )
                        submit_btn.enable()
                        return

                    # Generate unique filename and save
                    unique_name = f"{uuid.uuid4().hex}{file_ext}"
                    upload_path = _get_upload_dir() / unique_name

                    progress_bar.value = 0.5
                    status_label.text = "Saving file…"

                    # Streams large files in chunks — does not load entire file into RAM
                    await e.file.save(upload_path)

                    saved_size = upload_path.stat().st_size
                    logger.info(
                        f"File saved: {original_name!r} → {unique_name} "
                        f"({saved_size:,} bytes)"
                    )

                    # Create job in database
                    SessionLocal = get_session()
                    db = SessionLocal()
                    try:
                        lang = language_select.value or None
                        min_spk = (
                            int(min_speakers.value) if min_speakers.value else None
                        )
                        max_spk = (
                            int(max_speakers.value) if max_speakers.value else None
                        )

                        job = create_job(
                            db,
                            user_id=user_id,
                            filename=unique_name,
                            original_filename=original_name,
                            file_size=saved_size,
                            language=lang,
                            backend="local",
                            min_speakers=min_spk,
                            max_speakers=max_spk,
                        )
                        job_id_local = job.id
                    finally:
                        db.close()

                    # Enqueue job for background processing
                    await job_queue.enqueue(job_id_local)
                    logger.info(f"Job {job_id_local[:8]} enqueued for processing")

                    _upload_state["file_staged"] = True
                    progress_bar.value = 1.0
                    status_label.text = "Processing started!"
                    status_label.classes(add="text-green-600")

                    ui.notify(
                        f"'{original_name}' queued for processing",
                        type="positive",
                    )

                    await asyncio.sleep(1.0)
                    ui.navigate.to("/")

                except Exception as exc:
                    logger.error(
                        f"File upload handling failed: {exc}", exc_info=True
                    )
                    ui.notify(f"Upload error: {exc}", type="negative")
                    status_label.text = f"Error: {exc}"
                    status_label.classes(add="text-red-600")
                    submit_btn.enable()

            upload_widget.on_upload(on_file_upload)
