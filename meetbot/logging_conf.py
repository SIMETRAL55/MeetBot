"""
Structured logging configuration for MeetBot.

Provides JSON-formatted logging for production deployments with:
- Correlation IDs per job
- Structured fields (timestamp, level, module, message, extras)
- Human-readable fallback for development
- Pipeline stage timing metrics

Environment variables:
    LOG_FORMAT: 'json' or 'text' (default) for human-readable output
    LOG_LEVEL: 'DEBUG', 'INFO' (default), 'WARNING', 'ERROR', 'CRITICAL'
    LOG_FILE: Optional file path for log output
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add job_id if present on the record (set via extra={})
        if hasattr(record, "job_id"):
            log_entry["job_id"] = record.job_id

        # Add pipeline stage if present
        if hasattr(record, "stage"):
            log_entry["stage"] = record.stage

        # Add duration if present (for timing metrics)
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
            if record.exc_text:
                log_entry["traceback"] = record.exc_text

        return json.dumps(log_entry, ensure_ascii=False)


class PipelineMetrics:
    """Simple in-process metrics collector for pipeline stages.

    Tracks timing and counts for each pipeline stage.
    Queryable via the /api/health endpoint.
    """

    def __init__(self):
        self._stage_times: dict[str, list[float]] = {}
        self._job_count = 0
        self._error_count = 0
        self._start_time = time.monotonic()

    def record_stage(self, stage: str, duration_ms: float) -> None:
        if stage not in self._stage_times:
            self._stage_times[stage] = []
        times = self._stage_times[stage]
        times.append(duration_ms)
        # Keep only last 100 measurements
        if len(times) > 100:
            self._stage_times[stage] = times[-100:]

    def record_job_complete(self) -> None:
        self._job_count += 1

    def record_error(self) -> None:
        self._error_count += 1

    def get_summary(self) -> dict:
        summary = {
            "uptime_seconds": round(time.monotonic() - self._start_time),
            "jobs_completed": self._job_count,
            "errors": self._error_count,
            "stages": {},
        }
        for stage, times in self._stage_times.items():
            if times:
                summary["stages"][stage] = {
                    "count": len(times),
                    "avg_ms": round(sum(times) / len(times)),
                    "min_ms": round(min(times)),
                    "max_ms": round(max(times)),
                }
        return summary


# Singleton metrics instance
pipeline_metrics = PipelineMetrics()


class StageTimer:
    """Context manager for timing pipeline stages.

    Usage:
        with StageTimer("transcription", job_id="abc123"):
            # ... do work ...
    """

    def __init__(self, stage: str, job_id: str = ""):
        self.stage = stage
        self.job_id = job_id
        self._start: float = 0

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.monotonic() - self._start) * 1000
        pipeline_metrics.record_stage(self.stage, duration_ms)

        logger = logging.getLogger("meetbot.pipeline")
        logger.info(
            "Stage %s completed in %.0f ms",
            self.stage,
            duration_ms,
            extra={
                "stage": self.stage,
                "duration_ms": round(duration_ms),
                "job_id": self.job_id,
            },
        )
        return False  # don't suppress exceptions


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for MeetBot.

    Uses JSON format in production (LOG_FORMAT=json) and human-readable
    format in development (LOG_FORMAT=text, default).

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs to
        format_string: Optional custom format string (uses default if None)
    """
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_level = os.getenv("LOG_LEVEL", level).upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_format == "json":
        formatter = JSONFormatter()
    else:
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    log_file = log_file or os.getenv("LOG_FILE")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.debug(f"Logging level: {log_level}, format: {log_format}")


# Configure logging at module import time (INFO level by default)
setup_logging("INFO")
