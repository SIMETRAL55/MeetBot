"""
Reindex worker — rebuilds the vector index for a completed job without
re-running the full audio pipeline.

Used when the user edits the transcription in the WebApp and wants queries to
reflect the corrected text.  Only the Prepare → Index stages are executed; the
expensive Whisper / Pyannote stages are skipped.

Uses the RAG pipeline with speaker-aware chunking, atomic swap indexing,
and transcript versioning.  The canonical transcript is read from the DB
(reflecting all inline edits), chunked, and indexed atomically so queries
never see a partially built index.

Entry point
-----------
``run_reindex(job_id: str)`` — call this from the ``JobQueue`` worker thread.
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

        _run_reindex_v2(job, db, job_id, _stage, _fail)

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


def _run_reindex_v2(job, db, job_id: str, _stage, _fail) -> None:
    """RAG v2 reindex: DB-canonical transcript → chunker → atomic swap index."""
    from ..db.crud import (
        get_segments_for_job,
        bump_transcript_version,
        flush_segments_to_json,
    )
    from ..services.rag.indexer import RAGIndexer, _invalidate_chroma_cache

    # Bump transcript version
    _stage(5, 5, "Bumping transcript version...")
    try:
        version = bump_transcript_version(db, job_id)
    except Exception:
        version = getattr(job, "transcript_version", 1) or 1

    # Flush DB segments to on-disk JSON (keeps file in sync with edits)
    _stage(8, 8, "Syncing transcript to disk...")
    try:
        flush_segments_to_json(db, job_id)
    except Exception as exc:
        logger.warning("run_reindex_v2: flush_segments failed: %s", exc)

    # Read canonical transcript from DB segments
    _stage(10, 10, "Reading canonical transcript from DB...")
    segments = get_segments_for_job(db, job_id)
    if not segments:
        _fail("No transcript segments found in DB")
        return

    seg_dicts = [s.to_dict() for s in segments]
    logger.info("run_reindex_v2: %d segments from DB (version=%d)", len(seg_dicts), version)

    # Cancel check before embedding-heavy work
    cancel_registry.check_and_raise(job_id)

    # Cancel check before embedding-heavy work
    cancel_registry.check_and_raise(job_id)

    _stage(15, 15, f"Preparing multi-level index for {len(seg_dicts)} segments...")
    logger.info("run_reindex_v2: building multilevel index for %d segments", len(seg_dicts))

    # Atomic build + swap index (multilevel: doc + segment + chunk)
    collection_name = job_id[:8]
    db_root = str(Path(settings.VECTOR_DB_PATH).parent)

    def _index_progress(stage: str, pct: float, msg: str) -> None:
        overall = 20 + pct * 0.75
        _stage(overall, pct, msg)

    def _cancel_check() -> None:
        cancel_registry.check_and_raise(job_id)

    def _run_indexer(device: str) -> str:
        return RAGIndexer.build_multilevel_index_atomic(
            segments=seg_dicts,
            persist_root=db_root,
            collection_name=collection_name,
            embedding_model=settings.EMBEDDING_MODEL,
            device=device,
            progress_callback=_index_progress,
            job_id=job_id,
            version=version,
            chunk_tokens=settings.CHUNK_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP,
            cancel_checker=_cancel_check,
        )

    embedding_device = settings.EMBEDDING_DEVICE
    try:
        db_dir = _run_indexer(embedding_device)
    except RuntimeError as oom_err:
        if "out of memory" in str(oom_err).lower() and embedding_device != "cpu":
            logger.warning("run_reindex_v2: GPU OOM — retrying on CPU")
            _stage(20, 5, "GPU OOM — retrying on CPU...")
            try:
                import gc as _gc
                import torch
                torch.cuda.empty_cache()
                _gc.collect()
            except Exception:
                pass
            db_dir = _run_indexer("cpu")
        else:
            raise

    # Cancel check after indexing
    cancel_registry.check_and_raise(job_id)

    # Update DB
    update_job_result(db, job_id, db_dir=db_dir)

    progress_cb = progress_manager.make_callback(job_id)
    progress_cb("completed", 100, "Reindex complete (v2)", 100.0)
    update_job_status(
        db, job_id, JobStatus.COMPLETED,
        progress=100, stage_progress=100,
        progress_message="Reindex complete",
        log_line="✅ Reindex complete (v2, atomic swap)",
    )
    progress_manager.update(
        job_id, stage="completed", progress=100,
        message="Reindex complete", stage_progress=100, status="completed",
    )
    logger.info("run_reindex_v2: job=%s complete (version=%d)", job_id[:8], version)


def _cleanup_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
