"""
Cooperative cancellation registry for pipeline and reindex workers.

Provides a process-level thread-safe set of job IDs for which cancellation
has been requested.  Workers check ``cancel_registry.is_cancelled(job_id)``
at each stage boundary and raise ``JobCancelledError`` when the flag is set.

Design rationale
----------------
* **In-memory, not DB-polled per token** — workers run in a background thread;
  checking a set under a lock is O(1) and free of DB round-trips.
* **Cooperative, not preemptive** — cancellation takes effect at the next
  inter-stage check, so long-running stages (Whisper, Pyannote) run to
  natural completion before stopping.  This prevents half-written model
  outputs and torn GPU state.
* **Idempotent** — ``request_cancel`` can be called multiple times safely;
  ``clear`` removes the flag for a restarted job without affecting others.
"""

import threading
import logging

logger = logging.getLogger(__name__)


class JobCancelledError(Exception):
    """Raised inside a worker when cancel has been requested for a job."""


class CancelRegistry:
    """
    Thread-safe registry of job IDs that have been requested to cancel.

    The registry is checked by workers between pipeline stages.  It is
    never polled inside a single stage (transcription, diarization, etc.)
    because those stages run synchronous external libraries that cannot be
    interrupted safely mid-call.  Cancellation therefore takes effect at
    the *next* stage boundary after the request is received.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()

    def request_cancel(self, job_id: str) -> None:
        """
        Mark *job_id* as cancel-requested.

        Thread-safe; may be called from the asyncio event loop (web handler)
        while the worker thread is processing the job.
        """
        with self._lock:
            self._cancelled.add(job_id)
        logger.info("CancelRegistry: cancel requested for job %s", job_id[:8])

    def is_cancelled(self, job_id: str) -> bool:
        """
        Return True if cancellation has been requested for *job_id*.

        Thread-safe — called from the worker thread at each stage boundary.
        """
        with self._lock:
            return job_id in self._cancelled

    def clear(self, job_id: str) -> None:
        """
        Remove the cancel flag for *job_id* (e.g. when a job is restarted).

        Thread-safe; idempotent — safe to call even if the flag was never set.
        """
        with self._lock:
            self._cancelled.discard(job_id)
        logger.debug("CancelRegistry: cleared flag for job %s", job_id[:8])

    def check_and_raise(self, job_id: str) -> None:
        """
        Raise ``JobCancelledError`` if cancellation has been requested.

        Convenience helper for workers — replace per-stage ``if`` checks
        with a single call::

            cancel_registry.check_and_raise(job_id)
        """
        if self.is_cancelled(job_id):
            raise JobCancelledError(
                f"Job {job_id[:8]} cancelled by user request"
            )


# Process-level singleton — imported by pipeline_worker, reindex_worker,
# and the API cancel/restart handlers.
cancel_registry = CancelRegistry()
