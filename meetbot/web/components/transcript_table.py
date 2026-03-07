"""
Transcript table component with inline editing.

Renders an interactive table of transcript segments with:
- Timestamp display (click to seek audio)
- Speaker labels (editable inline)
- Text content (editable inline)
- Bulk speaker rename
"""

import logging
from typing import Callable, Optional

from nicegui import ui

from ...db.database import get_session
from ...db.crud import (
    update_segment_speaker,
    update_segment_text,
    bulk_update_speaker_name,
    get_segments_for_job,
    flush_segments_to_json,
    bump_transcript_version,
)
from ...db.models import Segment

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS.s format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:04.1f}"


# Speaker colors for visual distinction
SPEAKER_COLORS = [
    "blue", "green", "orange", "purple", "red",
    "teal", "pink", "indigo", "amber", "cyan",
]


def _speaker_color(speaker: str, speaker_list: list[str]) -> str:
    """Get a consistent color for a speaker."""
    if speaker in speaker_list:
        idx = speaker_list.index(speaker)
        return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
    return "grey"


class TranscriptTable:
    """
    Interactive transcript table with editing capabilities.

    Features:
    - Click timestamp to seek audio player
    - Click speaker badge to rename (single segment or all)
    - Click text to edit
    """

    def __init__(
        self,
        job_id: str,
        on_seek: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Create a transcript table.

        Args:
            job_id: Job ID to load segments for.
            on_seek: Callback when a timestamp is clicked (receives seconds).
        """
        self.job_id = job_id
        self.on_seek = on_seek
        self._container = ui.column().classes("w-full gap-0")
        self._segments: list[Segment] = []
        self._speakers: list[str] = []

        self.refresh()

    def refresh(self) -> None:
        """Reload segments from database and re-render."""
        SessionLocal = get_session()
        db = SessionLocal()
        try:
            self._segments = get_segments_for_job(db, self.job_id)
            # Build unique speaker list for coloring
            seen = []
            for seg in self._segments:
                if seg.speaker not in seen:
                    seen.append(seg.speaker)
            self._speakers = seen
        finally:
            db.close()

        self._render()

    def _render(self) -> None:
        """Render the transcript table."""
        self._container.clear()

        with self._container:
            if not self._segments:
                ui.label("No transcript segments available.").classes(
                    "text-gray-500 italic p-4"
                )
                return

            # Header
            with ui.row().classes(
                "w-full px-4 py-2 bg-gray-100 rounded-t text-xs "
                "font-semibold text-gray-600 gap-4"
            ):
                ui.label("Time").classes("w-24")
                ui.label("Speaker").classes("w-32")
                ui.label("Text").classes("flex-1")

            # Segments
            for seg in self._segments:
                self._render_segment(seg)

            # Summary
            ui.separator()
            with ui.row().classes("w-full px-4 py-2 text-xs text-gray-500 gap-4"):
                ui.label(f"{len(self._segments)} segments")
                ui.label(f"{len(self._speakers)} speakers: {', '.join(self._speakers)}")

    def _render_segment(self, segment: Segment) -> None:
        """Render a single segment row."""
        color = _speaker_color(segment.speaker, self._speakers)

        with ui.row().classes(
            "w-full px-4 py-2 border-b border-gray-100 items-start gap-4 "
            "hover:bg-gray-50 transition-colors"
        ):
            # Timestamp (clickable)
            time_str = _format_timestamp(segment.start_time)
            ts_btn = ui.button(
                time_str,
                on_click=lambda s=segment: self._seek(s.start_time),
            ).props("flat dense size=sm color=primary").classes("w-24 text-xs")
            ts_btn.tooltip("Click to play from here")

            # Speaker badge (clickable to rename)
            speaker_btn = ui.button(
                segment.speaker,
                on_click=lambda s=segment: self._edit_speaker(s),
            ).props(f"flat dense size=sm color={color}").classes("w-32 text-xs")
            speaker_btn.tooltip("Click to rename speaker")

            # Text (clickable to edit)
            text_label = ui.label(segment.text).classes(
                "flex-1 text-sm cursor-pointer hover:bg-blue-50 "
                "rounded px-2 py-1 transition-colors"
            )
            text_label.on("click", lambda s=segment: self._edit_text(s))
            text_label.tooltip("Click to edit text")

    def _seek(self, seconds: float) -> None:
        """Handle timestamp click — seek audio."""
        if self.on_seek:
            self.on_seek(seconds)

    def _edit_speaker(self, segment: Segment) -> None:
        """Show speaker rename dialog."""
        with ui.dialog() as dialog, ui.card().classes("w-80 p-4"):
            ui.label("Rename Speaker").classes("text-lg font-semibold")
            ui.label(f"Current: {segment.speaker}").classes(
                "text-sm text-gray-500"
            )

            new_name = ui.input(
                label="New Speaker Name",
                value=segment.speaker,
            ).classes("w-full").props("outlined")

            rename_all = ui.checkbox(
                f'Rename all "{segment.speaker}" segments',
                value=True,
            )

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def apply_rename():
                    name = new_name.value.strip()
                    if not name:
                        ui.notify("Speaker name cannot be empty", type="warning")
                        return

                    SessionLocal = get_session()
                    db = SessionLocal()
                    try:
                        if rename_all.value:
                            count = bulk_update_speaker_name(
                                db, self.job_id, segment.speaker, name
                            )
                            ui.notify(f"Renamed {count} segments", type="positive")
                        else:
                            update_segment_speaker(db, segment.id, name)
                            ui.notify("Speaker renamed", type="positive")
                        flush_segments_to_json(db, self.job_id)
                        bump_transcript_version(db, self.job_id)
                    except Exception as exc:
                        logger.error(
                            "apply_rename: flush_segments_to_json failed: %s", exc
                        )
                        ui.notify(
                            f"Speaker renamed in UI but transcript file could "
                            f"not be saved: {exc}",
                            type="warning",
                        )
                    finally:
                        db.close()

                    dialog.close()
                    self.refresh()

                ui.button("Apply", on_click=apply_rename).props("color=primary")

        dialog.open()

    def _edit_text(self, segment: Segment) -> None:
        """Show text edit dialog."""
        with ui.dialog() as dialog, ui.card().classes("w-[500px] p-4"):
            ui.label("Edit Segment Text").classes("text-lg font-semibold")
            ui.label(
                f"{segment.speaker} — "
                f"{_format_timestamp(segment.start_time)} - "
                f"{_format_timestamp(segment.end_time)}"
            ).classes("text-sm text-gray-500")

            text_area = ui.textarea(
                label="Text",
                value=segment.text,
            ).classes("w-full").props("outlined rows=4")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def apply_edit():
                    new_text = text_area.value.strip()
                    if not new_text:
                        ui.notify("Text cannot be empty", type="warning")
                        return

                    SessionLocal = get_session()
                    db = SessionLocal()
                    try:
                        update_segment_text(db, segment.id, new_text)
                        flush_segments_to_json(db, self.job_id)
                        bump_transcript_version(db, self.job_id)
                        ui.notify("Text updated and transcript file saved", type="positive")
                    except Exception as exc:
                        logger.error(
                            "apply_edit: flush_segments_to_json failed: %s", exc
                        )
                        ui.notify(
                            f"Text updated in UI but transcript file could "
                            f"not be saved: {exc}",
                            type="warning",
                        )
                    finally:
                        db.close()

                    dialog.close()
                    self.refresh()

                ui.button("Save", on_click=apply_edit).props("color=primary")

        dialog.open()
