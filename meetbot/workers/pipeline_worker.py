"""
Pipeline worker — executes the transcription pipeline in a background thread.

Runs each stage (transcription → diarization → alignment → indexing) sequentially,
reporting progress via ProgressManager and updating job status in the database.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Optional

from ..config import settings
from ..db.database import get_session
from ..db.crud import (
    update_job_status,
    update_job_result,
    create_segments_from_aligned,
    get_job,
)
from ..db.models import JobStatus
from .progress import progress_manager
from .cancel import cancel_registry, JobCancelledError

logger = logging.getLogger(__name__)


def _get_upload_dir() -> Path:
    """Get the upload directory path."""
    base = Path(settings.OUTPUT_DIR).parent
    upload_dir = base / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _get_results_dir() -> Path:
    """Get the results directory path."""
    results_dir = Path(settings.OUTPUT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_pipeline(job_id: str) -> None:
    """
    Execute the full MeetBot pipeline for a job.

    Runs in a background thread. Updates DB status and pushes
    progress via ProgressManager throughout execution.

    Pipeline stages:
    1. Audio conversion (if needed)
    2. Transcription (Whisper)
    3. Diarization (Pyannote)
    4. Alignment
    5. Indexing (ChromaDB)

    Args:
        job_id: The job ID to process.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    progress_cb = progress_manager.make_callback(job_id)

    try:
        job = get_job(db, job_id)
        if job is None:
            logger.error(f"Job not found: {job_id}")
            return

        audio_path = str(_get_upload_dir() / job.filename)
        if not Path(audio_path).exists():
            _fail_job(db, job_id, f"Audio file not found: {audio_path}", progress_cb)
            return

        # Check for a pre-start cancel (e.g. job sat in queue while user cancelled)
        cancel_registry.check_and_raise(job_id)

        logger.info(f"Starting pipeline for job {job_id[:8]}: {job.original_filename}")

        # ── Stage 1: Transcription (overall 0 → 40 %) ─────────────────────
        _stage_update(db, job_id, JobStatus.TRANSCRIBING, progress_cb,
                      overall=0, stage=0, msg="Starting transcription...")

        from ..services.transcriber import TranscriberService
        from ..adapters.transcribers import get_transcriber_from_cli_arg
        from ..logging_conf import StageTimer, pipeline_metrics

        transcriber_adapter = get_transcriber_from_cli_arg(backend_arg=job.backend)
        transcriber_svc = TranscriberService(transcriber=transcriber_adapter)

        def transcribe_progress(stage: str, pct: float, msg: str) -> None:
            # pct is 0-100 within the transcription stage; scale to overall 0-40
            overall = pct * 0.4
            _stage_update(db, job_id, JobStatus.TRANSCRIBING, progress_cb,
                          overall=overall, stage=pct, msg=msg)

        with StageTimer("transcription", job_id=job_id):
            transcription = transcriber_svc.transcribe(
                audio_path,
                language=job.language,
                use_cache=True,
                progress_callback=transcribe_progress,
            )
        n_trans_segments = len(transcription.get("segments", []))
        done_msg = f"Transcription done — {n_trans_segments} segments"
        logger.info(f"Job {job_id[:8]}: {done_msg}")
        _stage_update(db, job_id, JobStatus.TRANSCRIBING, progress_cb,
                      overall=40, stage=100, msg=done_msg)

        # Save raw transcription JSON for per-format download
        trans_result_path = _get_results_dir() / f"{job_id}_transcription.json"
        with open(trans_result_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
        update_job_result(db, job_id, transcription_json_path=str(trans_result_path))

        # ── Memory cleanup between stages ───────────────────────────────
        del transcriber_svc, transcriber_adapter

        # CRITICAL: Explicitly unload the Whisper model from GPU before
        # diarization.  The class-level singleton (_model) keeps the model
        # in VRAM even after `del transcriber_adapter`.  Without this call,
        # diarization will OOM on GPUs with limited VRAM (e.g. 4 GiB).
        try:
            from ..adapters.transcribers.local_whisper import LocalWhisperTranscriber
            LocalWhisperTranscriber.cleanup()
        except Exception:
            pass  # safe to ignore if using HuggingFace backend

        try:
            from ..adapters.transcribers.faster_whisper import FasterWhisperTranscriber
            FasterWhisperTranscriber.cleanup()
        except Exception:
            pass  # safe to ignore if not using faster-whisper

        from ..utils.memory import cleanup_memory, log_memory_usage
        cleanup_memory("post-transcription")
        if settings.MEMORY_WATCH_ENABLED:
            log_memory_usage("post-transcription")

        # ── Cancel check: between Transcription and Diarization ────────────
        cancel_registry.check_and_raise(job_id)

        # ── Stage 2: Diarization (overall 40 → 65 %) ──────────────────────
        _stage_update(db, job_id, JobStatus.DIARIZING, progress_cb,
                      overall=40, stage=0, msg="Starting speaker identification...")

        from ..services.diarizer import DiarizationService

        diarizer_svc = DiarizationService()

        def diarize_progress(stage: str, pct: float, msg: str) -> None:
            # Scale diarization pct (0-100) to overall 40-65
            overall = 40 + pct * 0.25
            _stage_update(db, job_id, JobStatus.DIARIZING, progress_cb,
                          overall=overall, stage=pct, msg=msg)

        with StageTimer("diarization", job_id=job_id):
            diarization = diarizer_svc.diarize(
                audio_path,
                min_speakers=job.min_speakers,
                max_speakers=job.max_speakers,
                use_cache=True,
                progress_callback=diarize_progress,
            )
        n_dia_segments = len(diarization.get("segments", []))
        done_msg = f"Diarization done — {n_dia_segments} speaker segments"
        logger.info(f"Job {job_id[:8]}: {done_msg}")
        _stage_update(db, job_id, JobStatus.DIARIZING, progress_cb,
                      overall=65, stage=100, msg=done_msg)

        # Save raw diarization JSON for per-format download
        diar_result_path = _get_results_dir() / f"{job_id}_diarization.json"
        with open(diar_result_path, "w", encoding="utf-8") as f:
            json.dump(diarization, f, indent=2, ensure_ascii=False)
        update_job_result(db, job_id, diarization_json_path=str(diar_result_path))

        # ── Memory cleanup between stages ───────────────────────────────
        # Explicitly unload Pyannote model from GPU to free VRAM for indexing
        diarizer_svc.cleanup()
        del diarizer_svc
        cleanup_memory("post-diarization")
        if settings.MEMORY_WATCH_ENABLED:
            log_memory_usage("post-diarization")

        # ── Cancel check: between Diarization and Alignment ───────────────
        cancel_registry.check_and_raise(job_id)

        # ── Stage 3: Alignment (overall 65 → 75 %) ────────────────────────
        _stage_update(db, job_id, JobStatus.ALIGNING, progress_cb,
                      overall=65, stage=0, msg="Aligning transcript with speakers...")

        from ..services.aligner import AlignerService

        aligner = AlignerService()
        aligned = aligner.build_speaker_transcript(
            diarization.get("segments", []),
            transcription.get("segments", []),
        )
        n_aligned = len(aligned)
        logger.info(f"Job {job_id[:8]}: Alignment done — {n_aligned} segments")

        # Store segments in database
        _stage_update(db, job_id, JobStatus.ALIGNING, progress_cb,
                      overall=70, stage=50, msg=f"Saving {n_aligned} transcript segments...")
        create_segments_from_aligned(db, job_id, aligned)

        # Save result JSON
        from ..services.formatters import format_result_as_json

        final_output = format_result_as_json(aligned, job.original_filename)
        result_path = _get_results_dir() / f"{job_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        # Calculate duration
        duration = 0.0
        if aligned:
            duration = max(seg.get("end", 0) for seg in aligned)

        update_job_result(
            db, job_id,
            result_json_path=str(result_path),
            duration_seconds=duration,
        )
        _stage_update(db, job_id, JobStatus.ALIGNING, progress_cb,
                      overall=75, stage=100, msg="Transcript saved")

        # ── Memory cleanup: free alignment intermediates ──────────────────
        del aligner, aligned, transcription, diarization
        cleanup_memory("post-alignment")
        if settings.MEMORY_WATCH_ENABLED:
            log_memory_usage("post-alignment")

        # ── Cancel check: between Alignment and Indexing ──────────────────
        cancel_registry.check_and_raise(job_id)

        # ── Stage 4: Indexing (overall 75 → 95 %) ─────────────────────────
        _stage_update(db, job_id, JobStatus.INDEXING, progress_cb,
                      overall=75, stage=0, msg="Building search index...")

        # Indexing is FATAL — if it cannot complete on any device the job is
        # marked failed so the UI never shows a half-broken "ready" state.
        from ..db.crud import get_segments_for_job
        from ..services.rag.indexer import RAGIndexer

        # Read canonical segments from DB (inserted by the aligner stage)
        segments = get_segments_for_job(db, job_id)
        if not segments:
            raise RuntimeError("No transcript segments in DB — cannot build index")
        seg_dicts = [s.to_dict() for s in segments]
        version = getattr(segments[0], "transcript_version", 1) if segments else 1

        _stage_update(db, job_id, JobStatus.INDEXING, progress_cb,
                      overall=78, stage=10,
                      msg=f"Preparing multi-level index for {len(seg_dicts)} segments...")

        db_root = str(Path(settings.VECTOR_DB_PATH).parent)
        collection_name = job_id[:8]
        embedding_device = settings.EMBEDDING_DEVICE  # "cpu" by default

        def index_progress(stage: str, pct: float, msg: str) -> None:
            # Scale indexing pct (0-100) to overall 78-95
            overall = 78 + pct * 0.17
            _stage_update(db, job_id, JobStatus.INDEXING, progress_cb,
                          overall=overall, stage=10 + pct * 0.9, msg=msg)

        def _cancel_check() -> None:
            cancel_registry.check_and_raise(job_id)

        def _run_indexer(device: str) -> str:
            """Build the multi-level vector index atomically."""
            return RAGIndexer.build_multilevel_index_atomic(
                segments=seg_dicts,
                persist_root=db_root,
                collection_name=collection_name,
                embedding_model=settings.EMBEDDING_MODEL,
                device=device,
                progress_callback=index_progress,
                job_id=job_id,
                version=version,
                chunk_tokens=settings.CHUNK_TOKENS,
                chunk_overlap=settings.CHUNK_OVERLAP,
                cancel_checker=_cancel_check,
            )

        try:
            db_dir = _run_indexer(embedding_device)
        except RuntimeError as oom_err:
            _is_oom = "out of memory" in str(oom_err).lower()
            if _is_oom and embedding_device != "cpu":
                # ── GPU OOM: flush VRAM and retry on CPU ──────────────────
                logger.warning(
                    f"Job {job_id[:8]}: GPU indexing OOM — "
                    "clearing VRAM and retrying on CPU"
                )
                _stage_update(db, job_id, JobStatus.INDEXING, progress_cb,
                              overall=78, stage=5,
                              msg="GPU OOM — retrying indexing on CPU...")
                try:
                    import gc
                    import torch
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass
                # Retry on CPU — if this also fails, propagate to _fail_job
                db_dir = _run_indexer("cpu")
                logger.info(f"Job {job_id[:8]}: CPU indexing successful")
            else:
                raise  # Not OOM or already on CPU → propagate → _fail_job

        update_job_result(db, job_id, db_dir=db_dir)
        doc_count = RAGIndexer.get_doc_count(db_dir)
        logger.info(f"Job {job_id[:8]}: Indexing complete. Documents indexed (chunk level): {doc_count}")
        done_msg = f"Indexing done — {doc_count} chunks indexed ({len(seg_dicts)} segments)"
        logger.info(f"Job {job_id[:8]}: {done_msg}")
        _stage_update(db, job_id, JobStatus.INDEXING, progress_cb,
                      overall=95, stage=100, msg=done_msg)

        # ── Complete ───────────────────────────────────────────────────────
        progress_cb("completed", 100, "Processing complete", 100.0)
        update_job_status(
            db, job_id, JobStatus.COMPLETED,
            progress=100, stage_progress=100,
            progress_message="Processing complete",
            log_line="✅ Pipeline complete",
        )
        # Notify WebSocket subscribers that the job finished
        progress_manager.update(
            job_id, stage="completed", progress=100,
            message="Processing complete", stage_progress=100, status="completed",
        )
        logger.info(f"Job {job_id[:8]}: Pipeline complete")
        pipeline_metrics.record_job_complete()

    except JobCancelledError:
        logger.info("Job %s cancelled at stage boundary", job_id[:8])
        _cancel_job(db, job_id, progress_cb)

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Job {job_id[:8]} failed: {error_msg}\n{tb}")
        _fail_job(db, job_id, error_msg, progress_cb)
        pipeline_metrics.record_error()

    finally:
        cancel_registry.clear(job_id)  # remove flag regardless of outcome
        progress_manager.remove(job_id)
        db.close()
        # Clean up GPU memory after pipeline
        _cleanup_gpu()


def _stage_update(
    db,
    job_id: str,
    status: "JobStatus",
    progress_cb,
    overall: float,
    stage: float,
    msg: str,
) -> None:
    """
    Single helper that updates both ProgressManager and the DB in one call.

    Avoids duplicating the (progress_cb + update_job_status) pattern at
    every stage boundary.
    """
    progress_cb("transcribing" if status.value == "transcribing" else status.value,
                overall, msg, stage)
    update_job_status(
        db, job_id, status,
        progress=overall,
        stage_progress=stage,
        progress_message=msg,
        log_line=msg,
    )


def _cancel_job(
    db,
    job_id: str,
    progress_cb,
) -> None:
    """Mark a job as cancelled and notify WebSocket subscribers."""
    progress_cb("cancelled", 0, "Job cancelled by user", 0)
    update_job_status(
        db, job_id, JobStatus.CANCELLED,
        progress=0,
        stage_progress=0,
        progress_message="Cancelled by user",
        log_line="⏹ Cancelled by user",
    )
    progress_manager.update(
        job_id, stage="cancelled", progress=0,
        message="Cancelled by user", status="cancelled",
    )


def _fail_job(
    db,
    job_id: str,
    error_message: str,
    progress_cb,
) -> None:
    """Mark a job as failed with error details."""
    progress_cb("failed", 0, f"Failed: {error_message}", 0)
    update_job_status(
        db, job_id, JobStatus.FAILED,
        progress=0,
        stage_progress=0,
        progress_message=f"Failed: {error_message}",
        error_message=error_message,
        log_line=f"❌ Failed: {error_message}",
    )
    # Notify WebSocket subscribers of failure
    progress_manager.update(
        job_id, stage="failed", progress=0,
        message=error_message, status="failed",
    )


def _cleanup_gpu() -> None:
    """Release GPU memory after pipeline completion."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    except ImportError:
        pass
