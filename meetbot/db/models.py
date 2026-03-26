"""
SQLAlchemy ORM models for MeetBot.

Defines the database schema:
- User: authentication and user management
- Job: audio processing pipeline jobs
- Segment: aligned transcript segments belonging to a job
- ChatSession: one-per-job persistent RAG chat session
- ChatMessage: individual user/assistant turns in a ChatSession
"""

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Text,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class JobStatus(str, enum.Enum):
    """Pipeline job status."""
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    ALIGNING = "aligning"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    # Triggered manually from the web UI to rebuild the vector index after
    # a transcript edit without re-running the full audio pipeline.
    REINDEXING = "reindexing"
    # Cooperative cancellation — set by the Cancel API endpoint; workers
    # detect this flag at inter-stage boundaries and stop gracefully.
    CANCELLED = "cancelled"


class User(Base):
    """User account for authentication."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    display_name = Column(String(120), nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_login = Column(DateTime, nullable=True)

    # Relationships
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(username='{self.username}', admin={self.is_admin})>"


class Job(Base):
    """Audio processing pipeline job."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    language = Column(String(10), nullable=True)
    status = Column(
        Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True
    )
    error_message = Column(Text, nullable=True)
    progress = Column(Float, default=0.0, nullable=False)
    # stage_progress tracks 0-100 progress *within* the current stage.
    # progress tracks the weighted overall 0-100 across all stages.
    stage_progress = Column(Float, default=0.0, nullable=False)
    progress_message = Column(String(255), nullable=True)
    # logs stores recent pipeline log lines as a JSON array (last 50 entries).
    logs = Column(Text, default="[]", nullable=False)

    # Pipeline configuration
    backend = Column(String(20), default="local", nullable=False)
    min_speakers = Column(Integer, nullable=True)
    max_speakers = Column(Integer, nullable=True)

    # Result paths
    result_json_path = Column(String(500), nullable=True)
    # Raw stage outputs — saved during pipeline so users can download each separately
    transcription_json_path = Column(String(500), nullable=True)
    diarization_json_path = Column(String(500), nullable=True)
    db_dir = Column(String(500), nullable=True)

    # PageIndex (vectorless LLM-based retrieval — optional)
    pageindex_path = Column(String(500), nullable=True)    # Path to PageIndex JSON tree
    pageindex_status = Column(String(20), nullable=True)   # pending|building|ready|failed

    # SHA256 of segment texts+speakers at last successful vector index build.
    # Allows reindex_worker to skip ChromaDB rebuilds when nothing changed.
    segments_hash = Column(String(64), nullable=True, default="")

    # Transcript versioning — bumped on each edit-save-reindex cycle so
    # vectors can track which version of the transcript they represent.
    transcript_version = Column(Integer, default=1, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="jobs")
    segments = relationship(
        "Segment", back_populates="job", cascade="all, delete-orphan",
        order_by="Segment.start_time"
    )
    chat_session = relationship(
        "ChatSession", back_populates="job", uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_jobs_user_status", "user_id", "status"),
        Index("ix_jobs_created", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Job(id='{self.id[:8]}', file='{self.original_filename}', "
            f"status={self.status.value})>"
        )


class Segment(Base):
    """Aligned transcript segment belonging to a job."""

    __tablename__ = "segments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False, index=True)
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    speaker = Column(String(50), nullable=False)
    text = Column(Text, nullable=False)

    # Relationships
    job = relationship("Job", back_populates="segments")

    __table_args__ = (
        Index("ix_segments_job_order", "job_id", "segment_index"),
    )

    def __repr__(self) -> str:
        return (
            f"<Segment(job='{self.job_id[:8]}', idx={self.segment_index}, "
            f"speaker='{self.speaker}')>"
        )

    def to_dict(self) -> dict:
        """Convert segment to dictionary."""
        return {
            "id": self.id,
            "segment_index": self.segment_index,
            "start": self.start_time,
            "end": self.end_time,
            "speaker": self.speaker,
            "text": self.text,
        }


# =============================================================================
# ChatSession / ChatMessage — persistent RAG conversation history
# =============================================================================


class ChatSession(Base):
    """
    One-per-job persistent chat session.

    Each Job has at most one ChatSession (via UNIQUE job_id).
    Created automatically on first query; reused on subsequent page loads.
    """

    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    job = relationship("Job", back_populates="chat_session")
    messages = relationship(
        "ChatMessage", back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

    def __repr__(self) -> str:
        return f"<ChatSession(job='{self.job_id[:8]}', msgs={len(self.messages)})>"


class ChatMessage(Base):
    """
    A single turn in a ChatSession.

    Roles:
        "user"      — question asked by the logged-in user
        "assistant" — RAG-generated answer
        "system"    — informational message (e.g. "Session started")

    The ``sources`` column stores serialised JSON (list of source dicts) for
    assistant messages.  ``llm_backend`` records which LLM produced the answer.

    The ``status`` column tracks the persistence state of assistant messages:
        "streaming"    — row created when generation started; content is empty;
                         shown on reload as "generation in progress"
        "completed"    — normal end-of-stream commit (default for all inserts)
        "stopped"      — user pressed Stop; partial content saved
        "interrupted"  — client disconnected mid-stream; partial content saved
    User messages always have status='completed'.
    """

    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role = Column(String(20), nullable=False)          # "user" | "assistant"
    content = Column(Text, nullable=False)
    # Partial content accumulator for streaming persistence — flushed
    # periodically during generation so page refresh shows progress.
    content_partial = Column(Text, nullable=True, default="")
    sources = Column(Text, nullable=True)              # JSON-encoded source list
    llm_backend = Column(String(20), nullable=True)    # "local" | "hf"
    retrieval_method = Column(String(20), nullable=True)  # "vector" | "pageindex" | null for legacy
    # streaming|completed|stopped|interrupted  (default covers legacy rows)
    status = Column(String(20), nullable=False, default="completed")
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc), nullable=True
    )

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index("ix_chat_messages_session", "session_id", "created_at"),
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dict for rendering/API responses."""
        import json as _json
        sources_list = []
        if self.sources:
            try:
                sources_list = _json.loads(self.sources)
            except Exception:
                pass
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "sources": sources_list,
            "llm_backend": self.llm_backend,
            "retrieval_method": self.retrieval_method,
            "status": self.status or "completed",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        preview = (self.content or "")[:40].replace("\n", " ")
        return f"<ChatMessage(role={self.role}, status={self.status}, preview={preview!r})>"
