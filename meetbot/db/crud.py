"""
CRUD operations for MeetBot database models.

Provides functions for creating, reading, updating, and deleting
User, Job, Segment, ChatSession, and ChatMessage records.
"""

import json as _json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from .models import User, Job, JobStatus, Segment, ChatSession, ChatMessage

logger = logging.getLogger(__name__)


# =============================================================================
# User CRUD
# =============================================================================

def create_user(
    db: Session,
    username: str,
    password_hash: str,
    display_name: Optional[str] = None,
    is_admin: bool = False,
) -> User:
    """
    Create a new user.

    Args:
        db: Database session.
        username: Unique username.
        password_hash: Bcrypt-hashed password.
        display_name: Optional display name.
        is_admin: Whether user has admin privileges.

    Returns:
        Created User instance.
    """
    user = User(
        username=username,
        password_hash=password_hash,
        display_name=display_name or username,
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"Created user: {username}")
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def update_user_last_login(db: Session, user: User) -> None:
    """Update user's last login timestamp."""
    user.last_login = datetime.now(timezone.utc)
    db.commit()


def list_users(db: Session) -> list[User]:
    """List all users."""
    return db.query(User).order_by(User.created_at).all()


# =============================================================================
# Job CRUD
# =============================================================================

def create_job(
    db: Session,
    user_id: str,
    filename: str,
    original_filename: str,
    file_size: Optional[int] = None,
    language: Optional[str] = None,
    backend: str = "local",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Job:
    """
    Create a new processing job.

    Args:
        db: Database session.
        user_id: ID of the user who created the job.
        filename: Server-side filename (UUID-based).
        original_filename: Original uploaded filename.
        file_size: File size in bytes.
        language: Language hint for transcription.
        backend: Transcription backend ('local' or 'huggingface').
        min_speakers: Minimum expected speakers.
        max_speakers: Maximum expected speakers.

    Returns:
        Created Job instance.
    """
    job = Job(
        user_id=user_id,
        filename=filename,
        original_filename=original_filename,
        file_size=file_size,
        language=language,
        backend=backend,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        status=JobStatus.PENDING,
        progress=0.0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    logger.info(f"Created job: {job.id[:8]} for {original_filename}")
    return job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    """Get job by ID."""
    return db.query(Job).filter(Job.id == job_id).first()


def get_jobs_for_user(
    db: Session,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[Job]:
    """Get all jobs for a user, newest first."""
    return (
        db.query(Job)
        .filter(Job.user_id == user_id)
        .order_by(Job.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_all_jobs(db: Session, limit: int = 100) -> list[Job]:
    """Get all jobs across all users, newest first."""
    return db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()


def get_pending_jobs(db: Session) -> list[Job]:
    """Get all pending jobs, oldest first."""
    return (
        db.query(Job)
        .filter(Job.status == JobStatus.PENDING)
        .order_by(Job.created_at.asc())
        .all()
    )


def update_job_status(
    db: Session,
    job_id: str,
    status: JobStatus,
    progress: float = 0.0,
    progress_message: Optional[str] = None,
    error_message: Optional[str] = None,
    stage_progress: float = 0.0,
    log_line: Optional[str] = None,
) -> Optional[Job]:
    """
    Update job status and progress.

    Args:
        db: Database session.
        job_id: Job ID to update.
        status: New status.
        progress: Overall progress percentage (0-100).
        progress_message: Human-readable status message.
        error_message: Error message (for FAILED status).
        stage_progress: Within-stage progress percentage (0-100).
        log_line: Optional log line to append to job.logs (kept last 50).

    Returns:
        Updated Job or None if not found.
    """
    job = get_job(db, job_id)
    if job is None:
        logger.warning(f"Job not found: {job_id}")
        return None

    job.status = status
    job.progress = progress
    job.stage_progress = stage_progress
    job.progress_message = progress_message

    if error_message is not None:
        job.error_message = error_message

    if log_line is not None:
        try:
            current_logs: list = _json.loads(job.logs or "[]")
        except Exception:
            current_logs = []
        current_logs.append(log_line)
        job.logs = _json.dumps(current_logs[-50:], ensure_ascii=False)  # keep last 50

    if status == JobStatus.TRANSCRIBING and job.started_at is None:
        job.started_at = datetime.now(timezone.utc)

    if status in (JobStatus.COMPLETED, JobStatus.FAILED):
        job.completed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(job)
    return job


def update_job_result(
    db: Session,
    job_id: str,
    result_json_path: Optional[str] = None,
    transcription_json_path: Optional[str] = None,
    diarization_json_path: Optional[str] = None,
    db_dir: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> Optional[Job]:
    """
    Update job with result paths.

    Args:
        db: Database session.
        job_id: Job ID to update.
        result_json_path: Path to aligned output JSON.
        transcription_json_path: Path to raw Whisper transcription JSON.
        diarization_json_path: Path to raw diarization JSON.
        db_dir: Path to ChromaDB directory.
        duration_seconds: Audio duration in seconds.

    Returns:
        Updated Job or None if not found.
    """
    job = get_job(db, job_id)
    if job is None:
        return None

    if result_json_path is not None:
        job.result_json_path = result_json_path
    if transcription_json_path is not None:
        job.transcription_json_path = transcription_json_path
    if diarization_json_path is not None:
        job.diarization_json_path = diarization_json_path
    if db_dir is not None:
        job.db_dir = db_dir
    if duration_seconds is not None:
        job.duration_seconds = duration_seconds

    db.commit()
    db.refresh(job)
    return job


def delete_job(db: Session, job_id: str) -> bool:
    """
    Delete a job and ALL associated data.

    Cascade order:
    1. ChatMessages (via ChatSession cascade)
    2. ChatSession
    3. Segments
    4. Job row
    5. On-disk files: uploaded audio, result JSONs, ChromaDB vector store

    SQLAlchemy ``cascade="all, delete-orphan"`` on Job.segments and
    Job.chat_session handles the ORM-level cascade automatically when the
    Job row is deleted via ``db.delete(job)``.

    Args:
        db: Database session.
        job_id: Job ID to delete.

    Returns:
        True if deleted, False if not found.
    """
    import shutil
    from pathlib import Path
    from ..config import settings

    job = get_job(db, job_id)
    if job is None:
        return False

    # ── Collect on-disk paths before object is removed from session ──────
    upload_dir = Path(settings.OUTPUT_DIR).parent / "data" / "uploads"
    uploaded_file = upload_dir / job.filename if job.filename else None
    result_json   = Path(job.result_json_path)  if job.result_json_path  else None
    trans_json    = Path(job.transcription_json_path) if job.transcription_json_path else None
    diar_json     = Path(job.diarization_json_path)   if job.diarization_json_path   else None
    db_dir        = Path(job.db_dir) if job.db_dir else None

    # ── Delete DB row (cascades to Segments, ChatSession, ChatMessages) ──
    db.delete(job)
    db.commit()
    logger.info(f"Deleted job DB record: {job_id[:8]}")

    # ── Delete on-disk files ─────────────────────────────────────────────
    for path in [uploaded_file, result_json, trans_json, diar_json]:
        if path and path.exists():
            try:
                path.unlink()
                logger.debug(f"Deleted file: {path}")
            except Exception as e:
                logger.warning(f"Could not delete {path}: {e}")

    if db_dir and db_dir.exists():
        try:
            shutil.rmtree(db_dir)
            logger.info(f"Deleted vector store: {db_dir}")
        except Exception as e:
            logger.warning(f"Could not delete vector store {db_dir}: {e}")

    return True


# =============================================================================
# Segment CRUD
# =============================================================================

def delete_segments_for_job(db: Session, job_id: str) -> int:
    """
    Delete ALL existing segments for a job.

    Used before re-inserting segments to ensure idempotency —
    restarting or re-running a job never duplicates segments.

    Args:
        db: Database session.
        job_id: Job whose segments should be deleted.

    Returns:
        Number of segments deleted.
    """
    count = (
        db.query(Segment)
        .filter(Segment.job_id == job_id)
        .delete(synchronize_session="fetch")
    )
    db.commit()
    if count:
        logger.info("Deleted %d existing segments for job %s", count, job_id[:8])
    return count


def create_segments_from_aligned(
    db: Session,
    job_id: str,
    aligned_segments: list[dict],
) -> list[Segment]:
    """
    Bulk create segments from aligned transcript data.

    **Idempotent**: deletes any existing segments for the job before
    inserting new ones, so restarts and re-runs never cause duplicates.

    Args:
        db: Database session.
        job_id: Job these segments belong to.
        aligned_segments: List of dicts with start, end, speaker, text.

    Returns:
        List of created Segment instances.
    """
    # ── Remove stale segments from a previous (interrupted) run ──────
    delete_segments_for_job(db, job_id)

    segments = []
    for idx, seg in enumerate(aligned_segments):
        segment = Segment(
            job_id=job_id,
            segment_index=idx,
            start_time=float(seg.get("start", 0)),
            end_time=float(seg.get("end", 0)),
            speaker=seg.get("speaker", "UNKNOWN"),
            text=seg.get("text", ""),
        )
        segments.append(segment)

    db.bulk_save_objects(segments)
    db.commit()
    logger.info(f"Created {len(segments)} segments for job {job_id[:8]}")
    return segments


def get_segments_for_job(db: Session, job_id: str) -> list[Segment]:
    """Get all segments for a job, ordered by index."""
    return (
        db.query(Segment)
        .filter(Segment.job_id == job_id)
        .order_by(Segment.segment_index)
        .all()
    )


def update_segment_speaker(
    db: Session,
    segment_id: str,
    speaker: str,
) -> Optional[Segment]:
    """
    Update a segment's speaker label.

    Args:
        db: Database session.
        segment_id: Segment ID.
        speaker: New speaker label.

    Returns:
        Updated Segment or None if not found.
    """
    segment = db.query(Segment).filter(Segment.id == segment_id).first()
    if segment is None:
        return None

    segment.speaker = speaker
    db.commit()
    db.refresh(segment)
    logger.debug(f"Updated segment {segment_id[:8]} speaker to '{speaker}'")
    return segment


def update_segment_text(
    db: Session,
    segment_id: str,
    text: str,
) -> Optional[Segment]:
    """
    Update a segment's text content.

    Args:
        db: Database session.
        segment_id: Segment ID.
        text: New text content.

    Returns:
        Updated Segment or None if not found.
    """
    segment = db.query(Segment).filter(Segment.id == segment_id).first()
    if segment is None:
        return None

    segment.text = text
    db.commit()
    db.refresh(segment)
    return segment


def bulk_update_speaker_name(
    db: Session,
    job_id: str,
    old_speaker: str,
    new_speaker: str,
) -> int:
    """
    Rename all occurrences of a speaker in a job.

    Args:
        db: Database session.
        job_id: Job ID.
        old_speaker: Current speaker label.
        new_speaker: New speaker label.

    Returns:
        Number of segments updated.
    """
    count = (
        db.query(Segment)
        .filter(Segment.job_id == job_id, Segment.speaker == old_speaker)
        .update({Segment.speaker: new_speaker})
    )
    db.commit()
    logger.info(
        f"Renamed '{old_speaker}' to '{new_speaker}' in job {job_id[:8]}: "
        f"{count} segments"
    )
    return count


# =============================================================================
# ChatSession / ChatMessage CRUD
# =============================================================================


def get_or_create_chat_session(db: Session, job_id: str) -> ChatSession:
    """
    Return the ChatSession for ``job_id``, creating it if it doesn't exist yet.

    This is the entry-point called whenever the query page opens.  Using
    ``get_or_create`` (rather than always-create) means a page refresh
    reconnects to the same history.

    Args:
        db: Database session.
        job_id: Parent job id.

    Returns:
        ChatSession instance (existing or newly created).
    """
    session = db.query(ChatSession).filter(ChatSession.job_id == job_id).first()
    if session is None:
        session = ChatSession(job_id=job_id)
        db.add(session)
        db.commit()
        db.refresh(session)
        logger.info(f"Created ChatSession for job {job_id[:8]}")
    return session


def get_chat_session(db: Session, job_id: str) -> Optional[ChatSession]:
    """Return the ChatSession for ``job_id`` or None if it doesn't exist."""
    return db.query(ChatSession).filter(ChatSession.job_id == job_id).first()


def create_chat_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
    llm_backend: Optional[str] = None,
    status: str = "completed",
    retrieval_method: Optional[str] = None,
) -> ChatMessage:
    """
    Append a new message to a ChatSession.

    Args:
        db: Database session.
        session_id: Parent ChatSession id.
        role: "user" | "assistant"
        content: Message text.
        sources: List of source dicts (serialised to JSON for assistant msgs).
        llm_backend: "local" | "hf" (for assistant messages).
        status: Persistence status — "streaming" | "completed" | "stopped" |
                "interrupted".  Use "streaming" when creating the placeholder
                row at the start of generation; update to a terminal status
                with ``update_chat_message`` when generation ends.
        retrieval_method: "vector" | "pageindex" (for assistant messages).

    Returns:
        Created ChatMessage.
    """
    sources_json = _json.dumps(sources, ensure_ascii=False) if sources else None
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources_json,
        llm_backend=llm_backend,
        status=status,
        retrieval_method=retrieval_method,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def update_chat_message(
    db: Session,
    msg_id: str,
    content: str,
    status: str,
    sources: Optional[list] = None,
    llm_backend: Optional[str] = None,
    retrieval_method: Optional[str] = None,
) -> Optional[ChatMessage]:
    """
    Update the content and status of an existing ChatMessage in-place.

    Used to finalise a "streaming" placeholder row created at the start of
    LLM generation.  Atomically replaces content + sources + status in a
    single transaction so the row is always consistent.

    Args:
        db: Database session.
        msg_id: ID of the ChatMessage to update.
        content: Final (or partial) message content.
        status: Terminal status — "completed" | "stopped" | "interrupted".
        sources: Updated source list (replaces previous value if provided).
        llm_backend: LLM backend label (replaces previous value if provided).
        retrieval_method: "vector" | "pageindex" (replaces previous value if provided).

    Returns:
        Updated ChatMessage or None if not found.
    """
    msg = db.query(ChatMessage).filter(ChatMessage.id == msg_id).first()
    if msg is None:
        logger.warning("update_chat_message: msg_id %s not found", msg_id[:8])
        return None
    msg.content = content
    msg.status = status
    if sources is not None:
        msg.sources = _json.dumps(sources, ensure_ascii=False)
    if llm_backend is not None:
        msg.llm_backend = llm_backend
    if retrieval_method is not None:
        msg.retrieval_method = retrieval_method
    db.commit()
    db.refresh(msg)
    return msg


def get_chat_messages(
    db: Session,
    session_id: str,
    limit: int = 200,
) -> list[ChatMessage]:
    """
    Return all messages in a session in chronological order.

    Args:
        db: Database session.
        session_id: ChatSession id.
        limit: Maximum number of messages to return (safety cap).

    Returns:
        List of ChatMessage instances, oldest first.
    """
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .limit(limit)
        .all()
    )


def delete_chat_history(db: Session, job_id: str) -> int:
    """
    Delete all chat messages for a job (keeps the session row).

    Useful for "clear history" without deleting the job itself.

    Args:
        db: Database session.
        job_id: Parent job id.

    Returns:
        Number of messages deleted.
    """
    session = get_chat_session(db, job_id)
    if session is None:
        return 0
    count = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .delete()
    )
    db.commit()
    logger.info(f"Cleared {count} chat messages for job {job_id[:8]}")
    return count


# =============================================================================
# Segment → JSON write-through
# =============================================================================


def flush_segments_to_json(db: Session, job_id: str) -> "Path":
    """
    Write the current SQLite segments for *job_id* back to the job's aligned
    result JSON file, updating ``speaker`` and ``text`` in place.

    All other fields (``start``, ``end``, top-level metadata such as
    ``input_file`` and ``duration_seconds``) are preserved unchanged.
    The write is atomic: a temporary file is created next to the target and
    then renamed into place so a concurrent read never sees a partial file.

    This keeps the JSON file (the source-of-truth for downloads and reindexing)
    in sync with every edit made through the web interface.

    Args:
        db: Database session.
        job_id: ID of the job whose segments should be flushed.

    Returns:
        Path to the (now updated) JSON file.

    Raises:
        ValueError: If the job or its result file cannot be found.
        RuntimeError: If the file cannot be written.
    """
    import os
    from pathlib import Path as _Path

    job = get_job(db, job_id)
    if job is None:
        raise ValueError(f"Job {job_id!r} not found")

    if not job.result_json_path:
        raise ValueError(
            f"Job {job_id[:8]} has no result_json_path — "
            "the pipeline may not have completed yet."
        )

    result_path = _Path(job.result_json_path)
    if not result_path.exists():
        raise ValueError(f"Result JSON not found on disk: {result_path}")

    # --------------------------------------------------------------------------
    # Load existing file – support both top-level-array and wrapped-object forms
    # --------------------------------------------------------------------------
    existing = _json.loads(result_path.read_text(encoding="utf-8"))

    if isinstance(existing, list):
        json_segments = existing
        is_array = True
    elif isinstance(existing, dict) and "segments" in existing:
        json_segments = existing["segments"]
        is_array = False
    else:
        raise ValueError(
            f"Unrecognised result JSON structure in {result_path}. "
            "Expected a top-level list or an object with a 'segments' key."
        )

    # --------------------------------------------------------------------------
    # Patch speaker + text from DB, keyed by segment_index (= position in file)
    # --------------------------------------------------------------------------
    db_segs = get_segments_for_job(db, job_id)
    db_by_idx = {seg.segment_index: seg for seg in db_segs}

    for i, jseg in enumerate(json_segments):
        db_seg = db_by_idx.get(i)
        if db_seg is not None:
            jseg["speaker"] = db_seg.speaker
            jseg["text"] = db_seg.text

    # --------------------------------------------------------------------------
    # Reconstitute document and write atomically
    # --------------------------------------------------------------------------
    if is_array:
        output = json_segments
    else:
        existing["segments"] = json_segments
        output = existing

    tmp_path = result_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(
            _json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(result_path))
    except Exception as exc:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Failed to write updated transcript to {result_path}: {exc}"
        ) from exc

    logger.info(
        "flush_segments_to_json: wrote %d segments to %s (job=%s)",
        len(json_segments),
        result_path,
        job_id[:8],
    )
    return result_path


def build_aligned_json_from_db(db: Session, job_id: str) -> dict:
    """
    Reconstruct the aligned transcript JSON payload from the DB segments,
    optionally enriched with top-level metadata from the on-disk result file.

    This is the **authoritative download source**: it always reflects the
    current state of the DB (including any inline edits) regardless of
    whether ``flush_segments_to_json`` has been called.  If the on-disk result
    file is readable its top-level metadata (``input_file``, ``duration_seconds``,
    etc.) is preserved; the ``segments`` list is replaced entirely from the DB.

    Args:
        db: Database session.
        job_id: ID of the job.

    Returns:
        A dict with a ``"segments"`` key containing up-to-date segment dicts
        and any preserved top-level metadata from the original result file.

    Raises:
        ValueError: If the job is not found.
    """
    job = get_job(db, job_id)
    if job is None:
        raise ValueError(f"Job {job_id!r} not found")

    db_segs = get_segments_for_job(db, job_id)

    # Try to load top-level metadata from the on-disk result file.
    # We discard the on-disk segments and replace with DB segments.
    metadata: dict = {}
    if job.result_json_path:
        from pathlib import Path as _P
        _rp = _P(job.result_json_path)
        if _rp.exists():
            try:
                existing = _json.loads(_rp.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    metadata = {k: v for k, v in existing.items() if k != "segments"}
            except Exception:
                pass  # metadata is optional; segments come from DB

    segments_out = [
        {
            "segment_index": seg.segment_index,
            "start": seg.start_time,
            "end": seg.end_time,
            "speaker": seg.speaker,
            "text": seg.text,
        }
        for seg in db_segs
    ]

    return {**metadata, "segments": segments_out}


# =============================================================================
# Streaming Persistence & Transcript Versioning
# =============================================================================


def flush_streaming_content(
    db: Session,
    msg_id: str,
    content_partial: str,
) -> Optional[ChatMessage]:
    """
    Update the partial content of a streaming ChatMessage.

    Called periodically during LLM generation to persist progress so
    page refreshes can display the partial answer.

    Args:
        db: Database session.
        msg_id: ID of the ChatMessage.
        content_partial: Accumulated partial content so far.

    Returns:
        Updated ChatMessage or None if not found.
    """
    from datetime import datetime, timezone

    msg = db.query(ChatMessage).filter(ChatMessage.id == msg_id).first()
    if msg is None:
        return None
    msg.content_partial = content_partial
    msg.updated_at = datetime.now(timezone.utc)
    db.commit()
    return msg


def bump_transcript_version(db: Session, job_id: str) -> int:
    """
    Increment the transcript version for a job.

    Called when a transcript edit is saved, before triggering reindex,
    so the new index knows which version it represents.

    Args:
        db: Database session.
        job_id: Job ID.

    Returns:
        The new version number.
    """
    job = get_job(db, job_id)
    if job is None:
        raise ValueError(f"Job {job_id!r} not found")
    current = getattr(job, "transcript_version", None) or 1
    job.transcript_version = current + 1
    db.commit()
    db.refresh(job)
    logger.info(
        "bump_transcript_version: job=%s version %d → %d",
        job_id[:8], current, job.transcript_version,
    )
    return job.transcript_version
