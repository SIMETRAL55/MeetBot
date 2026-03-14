"""
SQLAlchemy database engine and session management.

Provides:
- SQLite engine creation with WAL mode for concurrent reads
- Session factory with proper lifecycle management
- Database initialization (table creation)
"""

import logging
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

logger = logging.getLogger(__name__)

# Module-level engine and session factory (initialized lazily)
_engine = None
_SessionLocal = None


def _get_db_path() -> Path:
    """Get database file path from config.

    Reads ``DB_PATH`` from settings (env var ``DB_PATH``).
    Defaults to ``./db/meetbot.db`` (relative to the working directory),
    which maps to ``/app/db/meetbot.db`` inside the Docker container —
    covered by the ``./db:/app/db`` bind mount so the database persists
    across container rebuilds.
    """
    from ..config import settings
    return Path(settings.DB_PATH).expanduser().resolve()


def get_engine(db_path: str | None = None):
    """
    Get or create the SQLAlchemy engine.

    Uses SQLite with WAL journal mode for better concurrent read performance.

    Args:
        db_path: Optional override for database file path.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is not None:
        return _engine

    if db_path is None:
        db_path = str(_get_db_path())

    db_url = f"sqlite:///{db_path}"
    logger.info(f"Creating database engine: {db_url}")

    _engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Enable WAL mode for better concurrent read performance
    @event.listens_for(_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return _engine


def get_session() -> sessionmaker:
    """
    Get the session factory.

    Returns:
        SQLAlchemy sessionmaker bound to the engine.
    """
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal

    engine = get_engine()
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Dependency-injection style session provider.

    Yields a database session and ensures proper cleanup.

    Yields:
        SQLAlchemy Session instance.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_migrations(engine) -> None:
    """
    Apply lightweight schema migrations for existing databases.

    SQLite does not support ALTER TABLE DROP/MODIFY, but ADD COLUMN is safe.
    We check existing columns via PRAGMA and add any that are missing.
    This allows upgrading an existing meetbot.db without data loss.
    """
    migrations = [
        # (table, column, sql_type, default_clause)
        # NOT NULL columns require a DEFAULT so existing rows satisfy the constraint
        ("jobs", "stage_progress", "REAL NOT NULL", "DEFAULT 0.0"),
        ("jobs", "logs",           "TEXT NOT NULL", "DEFAULT '[]'"),
        # Nullable timestamp columns — no DEFAULT needed
        ("jobs", "started_at",   "DATETIME", ""),
        ("jobs", "completed_at", "DATETIME", ""),
        # Nullable result-path columns added for multi-format download
        ("jobs", "transcription_json_path", "TEXT", ""),
        ("jobs", "diarization_json_path",   "TEXT", ""),
        # Chat message persistence status — 'completed' default covers all legacy rows
        ("chat_messages", "status", "TEXT NOT NULL", "DEFAULT 'completed'"),
        # Transcript versioning
        ("jobs", "transcript_version", "INTEGER NOT NULL", "DEFAULT 1"),
        # Streaming persistence: partial content accumulator + update timestamp
        ("chat_messages", "content_partial", "TEXT", "DEFAULT ''"),
        ("chat_messages", "updated_at", "DATETIME", ""),
    ]
    with engine.connect() as conn:
        for table, column, sql_type, default_clause in migrations:
            result = conn.execute(
                __import__('sqlalchemy').text(f"PRAGMA table_info({table})")
            )
            rows = result.fetchall()
            if not rows:
                # Table doesn't exist yet; it will be created with all columns
                # by Base.metadata.create_all(), so no ALTER TABLE needed.
                continue
            existing_columns = {row[1] for row in rows}
            if column not in existing_columns:
                ddl = f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
                if default_clause:
                    ddl += f" {default_clause}"
                conn.execute(__import__('sqlalchemy').text(ddl))
                conn.commit()
                logger.info(f"Migration: added column {table}.{column}")


def init_db(db_path: str | None = None) -> None:
    """
    Initialize the database — create all tables if they don't exist,
    then apply lightweight column migrations for existing databases.

    Args:
        db_path: Optional override for database file path.
    """
    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
    _run_migrations(engine)
    logger.info("Database tables created/verified.")


def reset_engine() -> None:
    """Reset the engine and session factory (for testing)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
