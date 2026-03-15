"""
SQLAlchemy database engine and session management.

Provides:
- Multi-database support (SQLite for dev, PostgreSQL for production)
- Session factory with proper lifecycle management
- Database initialization (table creation + migrations)

Configuration via environment variable:
    DATABASE_URL: Full database URL (e.g., "postgresql://user:pass@host/db")
                  Falls back to SQLite if not set.
    DB_PATH: SQLite file path (used only when DATABASE_URL is not set)
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


def _get_database_url() -> str:
    """Determine the database URL to use.

    Priority:
    1. DATABASE_URL env var (supports PostgreSQL, MySQL, SQLite, etc.)
    2. DB_PATH setting (SQLite file)

    Returns:
        SQLAlchemy database URL string.
    """
    import os
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Handle Heroku-style postgres:// URLs
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url

    # Fall back to SQLite
    from ..config import settings
    db_path = Path(settings.DB_PATH).expanduser().resolve()
    return f"sqlite:///{db_path}"


def _get_db_path() -> Path:
    """Get database file path from config (SQLite only)."""
    from ..config import settings
    return Path(settings.DB_PATH).expanduser().resolve()


def get_engine(db_path: str | None = None):
    """
    Get or create the SQLAlchemy engine.

    Supports both SQLite (with WAL mode) and PostgreSQL (with connection pooling).

    Args:
        db_path: Optional override for database file path (SQLite only).

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    if _engine is not None:
        return _engine

    if db_path is not None:
        db_url = f"sqlite:///{db_path}"
    else:
        db_url = _get_database_url()

    is_sqlite = db_url.startswith("sqlite")
    logger.info(f"Creating database engine: {db_url.split('@')[-1] if '@' in db_url else db_url}")

    engine_kwargs = {"echo": False}

    if is_sqlite:
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL connection pool settings
        engine_kwargs.update({
            "pool_size": 5,
            "max_overflow": 10,
            "pool_pre_ping": True,       # Verify connections before use
            "pool_recycle": 1800,         # Recycle connections every 30 min
        })

    _engine = create_engine(db_url, **engine_kwargs)

    if is_sqlite:
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

    For SQLite: uses PRAGMA table_info to detect missing columns and adds them.
    For PostgreSQL: uses information_schema.columns for the same purpose.
    """
    db_url = str(engine.url)
    is_sqlite = db_url.startswith("sqlite")

    migrations = [
        # (table, column, sqlite_type, pg_type, default_clause)
        ("jobs", "stage_progress", "REAL NOT NULL", "DOUBLE PRECISION NOT NULL", "DEFAULT 0.0"),
        ("jobs", "logs", "TEXT NOT NULL", "TEXT NOT NULL", "DEFAULT '[]'"),
        ("jobs", "started_at", "DATETIME", "TIMESTAMP WITH TIME ZONE", ""),
        ("jobs", "completed_at", "DATETIME", "TIMESTAMP WITH TIME ZONE", ""),
        ("jobs", "transcription_json_path", "TEXT", "TEXT", ""),
        ("jobs", "diarization_json_path", "TEXT", "TEXT", ""),
        ("chat_messages", "status", "TEXT NOT NULL", "VARCHAR(20) NOT NULL", "DEFAULT 'completed'"),
        ("jobs", "transcript_version", "INTEGER NOT NULL", "INTEGER NOT NULL", "DEFAULT 1"),
        ("chat_messages", "content_partial", "TEXT", "TEXT", "DEFAULT ''"),
        ("chat_messages", "updated_at", "DATETIME", "TIMESTAMP WITH TIME ZONE", ""),
    ]

    from sqlalchemy import text

    with engine.connect() as conn:
        for table, column, sqlite_type, pg_type, default_clause in migrations:
            if is_sqlite:
                result = conn.execute(text(f"PRAGMA table_info({table})"))
                rows = result.fetchall()
                if not rows:
                    continue
                existing_columns = {row[1] for row in rows}
                sql_type = sqlite_type
            else:
                # PostgreSQL / other databases
                result = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :table"
                ), {"table": table})
                rows = result.fetchall()
                if not rows:
                    continue
                existing_columns = {row[0] for row in rows}
                sql_type = pg_type

            if column not in existing_columns:
                ddl = f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
                if default_clause:
                    ddl += f" {default_clause}"
                conn.execute(text(ddl))
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
