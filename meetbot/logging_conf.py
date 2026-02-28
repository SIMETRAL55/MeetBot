"""
Standard logging configuration for MeetBot.

Provides consistent logging across all modules with structured formatting,
log levels, and optional file output.

Example usage:
    from logging_conf import setup_logging

    setup_logging(level="INFO", log_file="meetbot.log")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for MeetBot.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs to
        format_string: Optional custom format string (uses default if None)

    Example:
        setup_logging("DEBUG", log_file="meetbot.log")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    formatter = logging.Formatter(format_string)

    # Get or create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.debug(f"Logging to file: {log_path}")

    root_logger.debug(f"Logging level: {level}")


# Configure logging at module import time (INFO level by default)
setup_logging("INFO")
