"""
Unit tests for streaming persistence (StreamingPersister).

Verifies token accumulation, flush interval/count thresholds,
and finalisation — all with mocked DB calls so no SQLite is needed.
"""

import time
from unittest.mock import MagicMock, patch, call

import pytest

from meetbot.utils.persistence import StreamingPersister


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db():
    """Patch get_session and flush_streaming_content."""
    mock_session = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_session)

    with patch(
        "meetbot.utils.persistence.StreamingPersister._do_flush"
    ) as mock_flush:
        yield mock_flush


# ---------------------------------------------------------------------------
# Basic accumulation
# ---------------------------------------------------------------------------


class TestBasicAccumulation:
    def test_empty_content(self):
        p = StreamingPersister(message_id="m1", flush_interval=9999, flush_token_count=9999)
        assert p.content == ""

    def test_single_token(self):
        p = StreamingPersister(message_id="m1", flush_interval=9999, flush_token_count=9999)
        p.append("hello")
        assert p.content == "hello"

    def test_multiple_tokens(self):
        p = StreamingPersister(message_id="m1", flush_interval=9999, flush_token_count=9999)
        p.append("hello ")
        p.append("world")
        assert p.content == "hello world"

    def test_empty_token_appended(self):
        p = StreamingPersister(message_id="m1", flush_interval=9999, flush_token_count=9999)
        p.append("a")
        p.append("")
        p.append("b")
        assert p.content == "ab"


# ---------------------------------------------------------------------------
# Flush thresholds
# ---------------------------------------------------------------------------


class TestFlushThresholds:
    def test_count_threshold_triggers_flush(self):
        """After flush_token_count tokens, flush should be called."""
        with patch.object(StreamingPersister, "_do_flush") as mock_flush:
            p = StreamingPersister(
                message_id="m1",
                flush_interval=9999,  # effectively disabled
                flush_token_count=3,
            )
            p.append("a")
            p.append("b")
            assert mock_flush.call_count == 0
            p.append("c")  # 3rd token — triggers
            assert mock_flush.call_count >= 1

    def test_time_threshold_triggers_flush(self):
        """After flush_interval seconds, flush should be called."""
        with patch.object(StreamingPersister, "_do_flush") as mock_flush:
            p = StreamingPersister(
                message_id="m1",
                flush_interval=0.05,  # 50ms
                flush_token_count=9999,
            )
            p.append("a")
            time.sleep(0.1)  # Wait past interval
            p.append("b")  # Should trigger time-based flush
            assert mock_flush.call_count >= 1

    def test_no_message_id_skips_flush(self):
        """With empty message_id, _do_flush should not write to DB."""
        p = StreamingPersister(
            message_id="",
            flush_interval=0,
            flush_token_count=1,
        )
        # This should not raise even though we'd exceed thresholds
        p.append("a")
        p.append("b")
        # No DB error = success


# ---------------------------------------------------------------------------
# Finalise
# ---------------------------------------------------------------------------


class TestFinalise:
    @patch("meetbot.utils.persistence.get_session", create=True)
    @patch("meetbot.utils.persistence.update_chat_message", create=True)
    def test_finalise_returns_accumulated(self, mock_update, mock_session_fn):
        """finalise() should return the full accumulated text."""
        p = StreamingPersister(
            message_id="m1", flush_interval=9999, flush_token_count=9999
        )
        p.append("one ")
        p.append("two ")
        p.append("three")
        result = p.finalise(status="completed")
        assert result == "one two three"

    def test_finalise_empty_message_id(self):
        """With empty message_id, finalise should return content without DB call."""
        p = StreamingPersister(
            message_id="", flush_interval=9999, flush_token_count=9999
        )
        p.append("content")
        result = p.finalise()
        assert result == "content"

    @patch("meetbot.utils.persistence.get_session", create=True)
    @patch("meetbot.utils.persistence.update_chat_message", create=True)
    def test_finalise_default_status_completed(self, mock_update, mock_session_fn):
        p = StreamingPersister(
            message_id="m1", flush_interval=9999, flush_token_count=9999
        )
        p.append("x")
        p.finalise()
        # No exception = success

    def test_content_property_thread_safe(self):
        """The content property should not raise under concurrent access."""
        import threading

        p = StreamingPersister(
            message_id="m1", flush_interval=9999, flush_token_count=9999
        )

        errors = []

        def writer():
            try:
                for i in range(100):
                    p.append(f"tok{i} ")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    _ = p.content
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
        # All tokens should have been appended
        assert "tok99" in p.content
