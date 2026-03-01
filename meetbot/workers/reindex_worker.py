"""
Reindex worker — rebuilds the vector index for a completed job without
re-running the full audio pipeline.

Used when the user edits the transcription in the WebApp and wants queries to
reflect the corrected text.  Only the Prepare → Index stages are executed; the
expensive Whisper / Pyannote stages are skipped.

The worker:
1. Reads the *existing* aligned result JSON (``job.result_json_path``).
2. Wipes the old Chroma persist directory (prevents stale embeddings).
3. Re-prepares documents via ``PrepareDocsService``.
4. Rebuilds the index via ``IndexerService`` (``overwrite=True``).
5. Reports progress through the shared ``ProgressManager`` so the
   existing WebSocket subscribers see live updates without any UI changes.

Entry point
-----------
``run_reindex(job_id: str)`` — call this from the ``JobQueue`` worker thread.
It mirrors the shape of ``run_pipeline()`` so it can use the same queue.

Jobs in the queue are plain strings.  The queue distinguishes reindex tasks
from full-pipeline tasks by checking the JobStatus at dequeue time:

    if job.status == JobStatus.REINDEXING:
        run_reindex(job_id)
    else:
        run_pipeline(job_id)
"""

import json
import logging
import shutil
import traceback
from pathlib import Path

from ..config import settings
from ..db.database import get_session
from ..db.crud import get_job, update_job_status, update_job_result
from ..db.models import JobStatus
from .progress import progress_manager
from .cancel import cancel_registry, JobCancelledError

logger = logging.getLogger(__name__)


def _get_results_dir() -> Path:
    results_dir = Path(settings.OUTPUT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_reindex(job_id: str) -> None:  # noqa: C901
    """
    Rebuild the vector index for *job_id* without re-running the audio pipeline.

    Progress is sent through ``ProgressManager`` so the existing WebSocket
    infrastructure delivers live updates to the browser without modification.

    Args:
        job_id: The job whose index should be rebuilt.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    progress_cb = progress_manager.make_callback(job_id)

    def _stage(overall: float, stage: float, msg: str) -> None:
        """Push a progress event and persist it to the DB."""
        progress_cb("reindexing", overall, msg, stage)
        update_job_status(
            db,
            job_id,
            JobStatus.REINDEXING,
            progress=overall,
            stage_progress=stage,
            progress_message=msg,
            log_line=msg,
        )

    def _fail(error: str) -> None:
        progress_cb("failed", 0, f"Reindex failed: {error}", 0)
        update_job_status(
            db,
            job_id,
            JobStatus.FAILED,
            progress=0,
            stage_progress=0,
            progress_message=f"Reindex failed: {error}",
            error_message=error,
            log_line=f"❌ Reindex failed: {error}",
        )
        progress_manager.update(
            job_id,
            stage="failed",
            progress=0,
            message=error,
            status="failed",
        )

    try:
        job = get_job(db, job_id)
        if job is None:
            logger.error("run_reindex: job not found: %s", job_id)
            return

        # ── Validate prerequisites ─────────────────────────────────────
        result_path = job.result_json_path
        if not result_path or not Path(result_path).exists():
            _fail(
                "Aligned result JSON not found — cannot reindex. "
                "The job may need to be re-processed from scratch."
            )
            return

        _stage(0, 0, "Starting reindex...")
        logger.info("run_reindex: job=%s file=%s", job_id[:8], job.original_filename)

        # ── Wipe old vector store ──────────────────────────────────────
        _stage(5, 5, "Clearing old vector index...")
        collection_name = job_id[:8]
        db_root = str(Path(settings.VECTOR_DB_PATH).parent)
        old_db_dir = Path(db_root) / collection_name

        try:
            from ..services.query_service import _invalidate_chroma_cache as _inv_pre
            _inv_pre(job_id)
        except Exception:
            pass  # best-effort

        if old_db_dir.exists():
            try:
                shutil.rmtree(str(old_db_dir))
                logger.info(
                    "run_reindex: removed old index at %s", old_db_dir
                )
            except Exception as exc:
                logger.warning(
                    "run_reindex: could not remove old index (%s) — continuing",
                    exc,
                )

        # ── Cancel check: after clearing index, before embedding-heavy indexing ─
        cancel_registry.check_and_raise(job_id)

        # ── Prepare documents ─────────────────────────────────────────
        _stage(10, 10, "Preparing document chunks...")

        from ..services.prepare_docs import PrepareDocsService
        from ..services.indexer import IndexerService

        prepare_svc = PrepareDocsService()
        try:
            docs, prepared_path = prepare_svc.prepare(result_path)
        except Exception as exc:
            _fail(f"Document preparation failed: {exc}")
            return

        _stage(20, 20, f"Prepared {len(docs)} document chunks")
        logger.info(
            "run_reindex: prepared %d chunks from %s", len(docs), result_path
        )

        # ── Build index ───────────────────────────────────────────────
        indexer_svc = IndexerService()
        embedding_device = settings.EMBEDDING_DEVICE

        def _index_progress(stage: str, pct: float, msg: str) -> None:
            # Scale indexer 0→100 to overall 20→95
            overall = 20 + pct * 0.75
            _stage(overall, pct, msg)

        def _run_indexer(device: str) -> None:
            indexer_svc.build_index(
                str(prepared_path),
                persist_root=db_root,
                embedding_model=settings.EMBEDDING_MODEL,
                collection_name=collection_name,
                overwrite=True,
                progress_callback=_index_progress,
                device=device,
            )

        try:
            _run_indexer(embedding_device)
        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower() and embedding_device != "cpu":
                logger.warning(
                    "run_reindex: GPU OOM on indexing — retrying on CPU"
                )
                _stage(20, 5, "GPU OOM — retrying on CPU...")
                try:
                    import gc, torch  # noqa: E401
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass
                _run_indexer("cpu")
            else:
                raise

        # Clear the chromadb system cache a second time.  build_index returns
        # a Chroma / PersistentClient object that registers itself in
        # SharedSystemClient._identifier_to_system.  Clearing it here ensures
        # the next query creates a clean client that reads from the freshly
        # written SQLite rather than reusing the build-time connection.
        try:
            from ..services.query_service import _invalidate_chroma_cache as _inv_post
            _inv_post(job_id)
        except Exception:
            pass

        # ── Update DB ─────────────────────────────────────────────────
        db_dir = str(old_db_dir)
        update_job_result(db, job_id, db_dir=db_dir)

        progress_cb("completed", 100, "Reindex complete", 100.0)
        update_job_status(
            db,
            job_id,
            JobStatus.COMPLETED,
            progress=100,
            stage_progress=100,
            progress_message="Reindex complete",
            log_line="✅ Reindex complete",
        )
        progress_manager.update(
            job_id,
            stage="completed",
            progress=100,
            message="Reindex complete",
            stage_progress=100,
            status="completed",
        )
        logger.info("run_reindex: job=%s complete", job_id[:8])

    except JobCancelledError:
        logger.info("run_reindex: job=%s cancelled", job_id[:8])
        progress_cb("cancelled", 0, "Reindex cancelled by user", 0)
        update_job_status(
            db, job_id, JobStatus.CANCELLED,
            progress=0, stage_progress=0,
            progress_message="Cancelled by user",
            log_line="⏹ Reindex cancelled by user",
        )
        progress_manager.update(
            job_id, stage="cancelled", progress=0,
            message="Cancelled by user", status="cancelled",
        )

    except Exception as exc:
        error_msg = str(exc)
        tb = traceback.format_exc()
        logger.error(
            "run_reindex: job=%s failed: %s\n%s", job_id[:8], error_msg, tb
        )
        _fail(error_msg)

    finally:
        cancel_registry.clear(job_id)
        progress_manager.remove(job_id)
        db.close()
        _cleanup_gpu()


def _cleanup_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
