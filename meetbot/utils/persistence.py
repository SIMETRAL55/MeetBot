"""
Streaming token persistence helper.

Accumulates tokens during LLM generation and periodically flushes
partial content to the database.  This ensures page refreshes can
show progress even if the stream hasn't completed.

Thread-safe: designed to be called from the streaming background thread.
"""

import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Flush interval in seconds and token count threshold
_FLUSH_INTERVAL_SECS = 1.0
_FLUSH_TOKEN_COUNT = 128


class StreamingPersister:
    """
    Accumulates tokens and periodically flushes to the database.

    Usage::

        persister = StreamingPersister(message_id="abc123")
        for token in llm_stream:
            persister.append(token)
            # ... send token to UI ...
        persister.finalise(status="completed")

    The persister is thread-safe but designed for single-writer usage.
    """

    def __init__(
        self,
        message_id: str,
        flush_interval: float = _FLUSH_INTERVAL_SECS,
        flush_token_count: int = _FLUSH_TOKEN_COUNT,
    ):
        self.message_id = message_id
        self.flush_interval = flush_interval
        self.flush_token_count = flush_token_count

        self._buffer: list[str] = []
        self._total_tokens = 0
        self._unflushed_tokens = 0
        self._last_flush_time = time.monotonic()
        self._lock = threading.Lock()

    @property
    def content(self) -> str:
        """Return the full accumulated content."""
        with self._lock:
            return "".join(self._buffer)

    def append(self, token: str) -> None:
        """
        Append a token and flush to DB if interval/count threshold reached.

        Args:
            token: A single generated token string.
        """
        with self._lock:
            self._buffer.append(token)
            self._total_tokens += 1
            self._unflushed_tokens += 1

        should_flush = (
            self._unflushed_tokens >= self.flush_token_count
            or (time.monotonic() - self._last_flush_time) >= self.flush_interval
        )

        if should_flush:
            self._do_flush()

    def _do_flush(self) -> None:
        """Flush accumulated content to database."""
        if not self.message_id:
            return

        with self._lock:
            content = "".join(self._buffer)
            self._unflushed_tokens = 0
            self._last_flush_time = time.monotonic()

        try:
            from ..db.database import get_session
            SessionLocal = get_session()
            db = SessionLocal()
            try:
                from ..db.crud import flush_streaming_content
                flush_streaming_content(db, self.message_id, content)
                logger.debug(
                    "Saved chat message partial id=%s (%d chars)",
                    self.message_id[:8], len(content),
                )
            finally:
                db.close()
        except Exception as exc:
            logger.warning(
                "StreamingPersister: flush failed for msg %s: %s",
                self.message_id[:8], exc,
            )

    def finalise(
        self,
        status: str = "completed",
        sources: Optional[list] = None,
        llm_backend: Optional[str] = None,
    ) -> str:
        """
        Finalise the message: set content_final and terminal status.

        Args:
            status: Terminal status ("completed", "stopped", "interrupted").
            sources: Source list for the message.
            llm_backend: LLM backend label.

        Returns:
            The final accumulated content.
        """
        content = self.content

        if not self.message_id:
            return content

        try:
            from ..db.database import get_session
            from ..db.crud import update_chat_message
            SessionLocal = get_session()
            db = SessionLocal()
            try:
                update_chat_message(
                    db,
                    self.message_id,
                    content=content,
                    status=status,
                    sources=sources,
                    llm_backend=llm_backend,
                )
                logger.info(
                    "StreamingPersister: finalised msg %s (%d chars, status=%s)",
                    self.message_id[:8], len(content), status,
                )
            finally:
                db.close()
        except Exception as exc:
            logger.error(
                "StreamingPersister: finalise failed for msg %s: %s",
                self.message_id[:8], exc,
            )

        return content
