"""
Job queue for background pipeline processing.

Provides a thread-safe queue that dispatches jobs to one or more worker threads
for GPU-bound processing.  Keep ``PIPELINE_WORKERS=1`` (the default) on
GPU-constrained machines to prevent VRAM contention between Whisper and Pyannote.

Important: we use ``queue.Queue`` (a standard threading queue) rather than
``asyncio.Queue`` because the worker lives in its own OS thread.  Mixing
asyncio.Queue with run_coroutine_threadsafe + a timeout creates "ghost"
coroutines that silently consume queued items without the worker ever seeing
them — causing jobs to stay stuck in PENDING forever.
"""

import asyncio
import logging
import queue
import threading
from typing import Optional

from .pipeline_worker import run_pipeline
from .reindex_worker import run_reindex

logger = logging.getLogger(__name__)


class JobQueue:
    """
    Configurable-concurrency job queue for pipeline processing.

    Jobs are enqueued from the async web layer and executed by
    ``settings.PIPELINE_WORKERS`` worker threads. This ensures:
    - GPU is used by only one model at a time (with PIPELINE_WORKERS=1)
    - No VRAM contention between Whisper, Pyannote, embeddings
    - Simple, predictable execution order (FIFO)
    """

    def __init__(self) -> None:
        # Use a plain threading Queue — fully safe to read/write across threads
        # without any asyncio bridge.  The worker thread calls .get(timeout=1)
        # directly; the async enqueue() calls .put() directly.
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._worker_threads: list[threading.Thread] = []
        self._running = False
        self._current_job_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def current_job_id(self) -> Optional[str]:
        """ID of the currently processing job, or None."""
        return self._current_job_id

    @property
    def queue_size(self) -> int:
        """Number of jobs waiting in the queue."""
        return self._queue.qsize()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the job queue worker(s).

        Args:
            loop: The asyncio event loop (for cross-thread communication).
        """
        if self._running:
            logger.warning("JobQueue already running")
            return

        self._loop = loop
        self._running = True

        # Give progress_manager a reference to the event loop so it can
        # bridge worker-thread updates into asyncio subscriber queues.
        from .progress import progress_manager
        progress_manager.set_loop(loop)

        from ..config import settings
        n_workers = max(1, settings.PIPELINE_WORKERS)
        for i in range(n_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"meetbot-pipeline-worker-{i}",
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)
        logger.info("Job queue started with %d worker(s)", n_workers)

    def stop(self) -> None:
        """Stop all job queue workers gracefully."""
        if not self._running:
            return

        self._running = False
        # Send one sentinel per worker thread to unblock each queue.get().
        for _ in self._worker_threads:
            self._queue.put(None)

        for t in self._worker_threads:
            if t.is_alive():
                t.join(timeout=5)
        self._worker_threads.clear()

        logger.info("Job queue workers stopped")

    async def enqueue(self, job_id: str) -> int:
        """
        Add a job to the processing queue.

        Args:
            job_id: The job ID to process.

        Returns:
            Position in queue (0-based, 0 means next up).
        """
        # queue.Queue.put() is non-blocking for an unbounded queue and fully
        # thread-safe — no asyncio plumbing required.
        position = self._queue.qsize()  # snapshot before put for position estimate
        self._queue.put(job_id)
        logger.info(f"Job {job_id[:8]} enqueued (position {position})")
        return position

    def _worker_loop(self) -> None:
        """Main worker loop — runs in a dedicated thread."""
        logger.info("Pipeline worker thread started")

        while self._running:
            try:
                # Block for up to 1 s waiting for a job.
                # queue.Queue.get() is thread-safe; no run_coroutine_threadsafe
                # indirection needed (that pattern creates ghost coroutines that
                # consume items without the worker ever seeing them).
                job_id = self._queue.get(timeout=1.0)

                if job_id is None:
                    # Sentinel value — shutdown signal
                    logger.debug("Worker received shutdown sentinel")
                    continue

                self._current_job_id = job_id
                logger.info(f"Dequeued job for processing: {job_id[:8]}")

                try:
                    # Check DB status to decide which worker to run.
                    # A REINDEXING status means only the Prepare→Index stages
                    # are needed; all other statuses run the full pipeline.
                    from ..db.database import get_session
                    from ..db.crud import get_job
                    from ..db.models import JobStatus

                    _ses = get_session()()
                    try:
                        _job = get_job(_ses, job_id)
                        _status = _job.status if _job else None
                    finally:
                        _ses.close()

                    if _status == JobStatus.REINDEXING:
                        run_reindex(job_id)
                    else:
                        run_pipeline(job_id)
                except Exception as e:
                    logger.error(
                        f"Pipeline worker error for job {job_id[:8]}: {e}",
                        exc_info=True,
                    )
                finally:
                    self._current_job_id = None

            except queue.Empty:
                # No job available within timeout — loop back and check _running
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Worker loop error: {e}", exc_info=True)

        logger.info("Pipeline worker thread stopped")


# Singleton instance
job_queue = JobQueue()
