"""
Progress tracking and WebSocket push for pipeline jobs.

Provides a centralized ProgressManager that:
- Tracks progress state for each job (overall + per-stage)
- Notifies registered callbacks on each update
- Bridges worker-thread progress events to asyncio WebSocket subscribers
  via loop.call_soon_threadsafe, so the /ws/jobs/{job_id} endpoint can
  stream live JSON events without polling.

Architecture:
    Worker thread  →  progress_manager.update()
                           ↓
                   iterate self._callbacks  (legacy, kept for compat)
                           ↓
                   for each subscriber queue:
                       loop.call_soon_threadsafe(queue.put_nowait, event_dict)
                           ↓
                   FastAPI WS handler  →  WebSocket.send_json(event_dict)
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class JobProgress:
    """Current progress state for a single job."""
    job_id: str
    stage: str = "pending"
    stage_progress: float = 0.0   # 0-100 within the current stage
    progress: float = 0.0         # 0-100 weighted overall progress
    message: str = ""
    logs: list = field(default_factory=list)  # recent log lines (last 50)
    status: str = "running"       # "running" | "completed" | "failed"


class ProgressManager:
    """
    Thread-safe progress manager with callback + WebSocket pub/sub.

    Worker threads call update() with per-stage and overall progress.
    WebSocket handlers subscribe/unsubscribe per job_id via asyncio Queues.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._progress: dict[str, JobProgress] = {}
        self._callbacks: list[Callable[[str, str, float, str], None]] = []
        # per-job lists of asyncio.Queue — each WS connection gets one queue
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        # set from the main asyncio event loop when the queue worker starts
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Set the asyncio event loop for thread-safe queue puts.

        Called once from JobQueue.start() so the worker thread can safely
        push events into subscriber queues owned by the async layer.
        """
        self._loop = loop

    def register_callback(
        self, callback: Callable[[str, str, float, str], None]
    ) -> None:
        """Register a legacy progress callback (job_id, stage, progress, msg)."""
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(
        self, callback: Callable[[str, str, float, str], None]
    ) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """
        Subscribe to progress events for a job.

        Creates an asyncio.Queue and registers it for the given job_id.
        Call this from an async context (WebSocket handler) before the
        job starts pushing events.

        Args:
            job_id: Job to subscribe to.

        Returns:
            asyncio.Queue that will receive event dicts.
        """
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
            # Immediately deliver cached current state if available
            current = self._progress.get(job_id)
        if current is not None:
            q.put_nowait(self._build_event(current))
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        """
        Remove a subscriber queue for a job.

        Call this when the WebSocket connection closes.
        """
        with self._lock:
            lst = self._subscribers.get(job_id, [])
            if q in lst:
                lst.remove(q)

    def update(
        self,
        job_id: str,
        stage: str,
        progress: float,
        message: str,
        stage_progress: float = 0.0,
        status: str = "running",
    ) -> None:
        """
        Update progress for a job and notify all subscribers + callbacks.

        Thread-safe — called from the worker thread.

        Args:
            job_id: Job identifier.
            stage: Current pipeline stage name.
            progress: Weighted overall percentage (0-100).
            message: Human-readable status message.
            stage_progress: Within-stage percentage (0-100). Defaults to overall.
            status: "running" | "completed" | "failed".
        """
        with self._lock:
            current = self._progress.get(job_id)
            logs = list(current.logs) if current else []
            if message:
                logs.append(message)
                logs = logs[-50:]

            jp = JobProgress(
                job_id=job_id,
                stage=stage,
                stage_progress=stage_progress if stage_progress else progress,
                progress=progress,
                message=message,
                logs=logs,
                status=status,
            )
            self._progress[job_id] = jp
            callbacks = list(self._callbacks)
            subscribers = list(self._subscribers.get(job_id, []))

        # Build event dict once
        event = self._build_event(jp)

        # Push to async subscriber queues (thread-safe via loop bridge)
        if self._loop is not None and subscribers:
            for q in subscribers:
                self._loop.call_soon_threadsafe(q.put_nowait, event)

        # Fire legacy callbacks
        for cb in callbacks:
            try:
                cb(job_id, stage, progress, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    @staticmethod
    def _build_event(jp: JobProgress) -> dict:
        """Build the canonical WebSocket event dict from a JobProgress."""
        return {
            "job_id": jp.job_id,
            "stage": jp.stage,
            "stage_progress": round(jp.stage_progress, 1),
            "overall_progress": round(jp.progress, 1),
            "logs": jp.logs[-10:],   # send last 10 log lines per message
            "status": jp.status,
            "message": jp.message,
        }

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get current in-memory progress for a job."""
        with self._lock:
            return self._progress.get(job_id)

    def remove(self, job_id: str) -> None:
        """Remove progress tracking for a completed/failed job."""
        with self._lock:
            self._progress.pop(job_id, None)

    def make_callback(self, job_id: str) -> Callable[[str, float, str, float], None]:
        """
        Create a bound progress callback for the given job_id.

        Compatible with existing service progress_callback signature:
            callback(stage: str, overall_pct: float, message: str,
                     stage_pct: float = 0.0) -> None

        Args:
            job_id: Job to bind the callback to.
        """
        def callback(
            stage: str,
            overall_pct: float,
            message: str,
            stage_pct: float = 0.0,
        ) -> None:
            self.update(
                job_id,
                stage=stage,
                progress=overall_pct,
                message=message,
                stage_progress=stage_pct if stage_pct else overall_pct,
            )
        return callback


# Module-level singleton shared by all workers and web handlers
progress_manager = ProgressManager()
