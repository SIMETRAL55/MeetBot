"""
Real-time progress display component for pipeline jobs.

Primary update mechanism: polls the database every 1.5 s.
This is reliable and works without a persistent WebSocket from the NiceGUI
page context. The page-level WebSocket needed for live push would require
a dedicated background task per client — polling keeps complexity low while
still feeling near-real-time for jobs that take tens of seconds.

For external consumers (scripts, mobile, dashboards) that need true live
streaming, use the dedicated endpoint:  GET /ws/jobs/{job_id}
"""

import json
import logging

from nicegui import ui

from ...db.database import get_session
from ...db.crud import get_job
from ...db.models import JobStatus

logger = logging.getLogger(__name__)

# ── Stage metadata ────────────────────────────────────────────────────────────
# Each stage has a display label, Quasar color, and its overall-progress window
# (start %, end %) so we can render a two-level progress bar:
#   outer bar  = overall_progress (0-100, weighted across all stages)
#   inner chip = stage_progress   (0-100, within the current stage)

STAGE_META = {
    "pending":     {"label": "⏳ Waiting in queue",         "color": "grey",   "range": (0,   0)},
    "transcribing":{"label": "🎤 Transcribing audio",       "color": "blue",   "range": (0,  40)},
    "diarizing":   {"label": "👥 Identifying speakers",     "color": "purple", "range": (40, 65)},
    "aligning":    {"label": "🔗 Aligning transcript",       "color": "orange", "range": (65, 75)},
    "indexing":    {"label": "🔍 Building search index",    "color": "cyan",   "range": (75, 95)},
    "completed":   {"label": "✅ Complete!",                 "color": "green",  "range": (95, 100)},
    "failed":      {"label": "❌ Failed",                    "color": "red",    "range": (0,   0)},
}

# Pipeline stage order used to render the step-chip row
STAGE_ORDER = ["pending", "transcribing", "diarizing", "aligning", "indexing", "completed"]


class ProgressDisplay:
    """
    Enhanced real-time progress display for a pipeline job.

    Renders:
    - Overall progress bar (0-100 %) with colour coding
    - Stage-progress sub-bar showing progress within the current stage
    - Pipeline stage chips (breadcrumb-style step indicator)
    - Recent log lines from the worker
    - Auto-stops when the job reaches COMPLETED or FAILED
    """

    def __init__(
        self,
        job_id: str,
        on_complete=None,
        on_fail=None,
    ) -> None:
        self.job_id = job_id
        self.on_complete = on_complete
        self.on_fail = on_fail
        self._timer = None
        self._last_log_count = 0

        with ui.card().classes("w-full p-4 gap-3") as self._card:
            # ── Header row ──────────────────────────────────────────────
            with ui.row().classes("w-full items-center justify-between"):
                self._stage_label = ui.label("Initialising…").classes(
                    "text-base font-semibold"
                )
                self._pct_label = ui.label("0 %").classes(
                    "text-sm font-mono text-gray-500"
                )

            # ── Overall progress bar ─────────────────────────────────────
            self._overall_bar = ui.linear_progress(
                value=0, show_value=False
            ).classes("w-full").props("color=blue rounded")

            # ── Stage (within-stage) sub-bar ────────────────────────────
            with ui.row().classes("w-full items-center gap-2"):
                self._stage_icon = ui.label("Stage:").classes(
                    "text-xs text-gray-400 w-10"
                )
                self._stage_bar = ui.linear_progress(
                    value=0, show_value=False
                ).classes("flex-1").props("color=grey rounded size=xs")
                self._stage_pct = ui.label("0 %").classes(
                    "text-xs font-mono text-gray-400 w-10 text-right"
                )

            # ── Status message ───────────────────────────────────────────
            self._message = ui.label("").classes(
                "text-sm text-gray-600 whitespace-pre-wrap"
            )

            # ── Stage chips (mini step indicator) ────────────────────────
            with ui.row().classes("w-full gap-1 flex-wrap mt-1"):
                self._chips: dict[str, ui.badge] = {}
                for s in STAGE_ORDER:
                    meta = STAGE_META[s]
                    badge = ui.badge(
                        meta["label"].split(" ", 1)[-1],  # strip emoji for brevity
                        color="grey",
                    ).classes("text-xs")
                    self._chips[s] = badge

            # ── Log panel ────────────────────────────────────────────────
            ui.label("Pipeline log").classes("text-xs font-semibold text-gray-400 mt-2")
            self._log_area = ui.column().classes(
                "w-full bg-gray-50 rounded p-2 gap-0 max-h-40 overflow-y-auto font-mono"
            )

        # Poll every 1.5 s until the job reaches a terminal state
        self._timer = ui.timer(1.5, self._refresh)

    # ─────────────────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        """DB poll — update all UI elements from current job state."""
        SessionLocal = get_session()
        db = SessionLocal()
        try:
            job = get_job(db, self.job_id)
        finally:
            db.close()

        if job is None:
            self._stage_label.text = "Job not found"
            self._stop()
            return

        status_key = job.status.value
        meta = STAGE_META.get(status_key, STAGE_META["pending"])
        overall = float(job.progress or 0.0)
        stage_p = float(getattr(job, "stage_progress", 0.0) or 0.0)

        # Overall bar
        self._overall_bar.value = overall / 100
        self._overall_bar.props(f"color={meta['color']}")
        self._pct_label.text = f"{overall:.0f} %"

        # Stage sub-bar
        self._stage_bar.value = stage_p / 100
        self._stage_bar.props(f"color={meta['color']}")
        self._stage_pct.text = f"{stage_p:.0f} %"

        # Header label
        self._stage_label.text = meta["label"]

        # Message
        if job.progress_message:
            self._message.text = job.progress_message

        # Stage chips: active stage gets the real colour, others grey/green
        active_idx = STAGE_ORDER.index(status_key) if status_key in STAGE_ORDER else 0
        for i, s in enumerate(STAGE_ORDER):
            chip = self._chips[s]
            if i < active_idx:
                chip.props("color=green")
            elif i == active_idx:
                chip.props(f"color={meta['color']}")
            else:
                chip.props("color=grey")

        # Append *new* log lines to the log panel
        try:
            all_logs: list = json.loads(job.logs or "[]")
        except Exception:
            all_logs = []

        new_logs = all_logs[self._last_log_count:]
        self._last_log_count = len(all_logs)
        for line in new_logs:
            ui.label(line).classes(
                "text-xs text-gray-700 leading-tight py-0"
            ).move(self._log_area)

        # Terminal states
        if job.status == JobStatus.COMPLETED:
            self._overall_bar.value = 1.0
            self._pct_label.text = "100 %"
            self._stop()
            if self.on_complete:
                self.on_complete()
        elif job.status == JobStatus.FAILED:
            self._message.text = job.error_message or "Processing failed"
            self._message.classes(add="text-red-600")
            self._stop()
            if self.on_fail:
                self.on_fail(job.error_message)

    def _stop(self) -> None:
        """Deactivate the polling timer."""
        if self._timer:
            self._timer.deactivate()

    def stop(self) -> None:
        """Public stop — can be called externally."""
        self._stop()
