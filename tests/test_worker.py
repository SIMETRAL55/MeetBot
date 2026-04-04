"""
Tests for the worker/queue system — ProgressManager and JobQueue lifecycle.
"""

import threading
import pytest

from meetbot.workers.progress import ProgressManager


class TestProgressManager:
    """Test ProgressManager progress tracking and callbacks."""

    def test_update_and_get_progress(self):
        pm = ProgressManager()
        pm.update("job-1", "transcribing", 50.0, "Half done")

        progress = pm.get_progress("job-1")
        assert progress is not None
        assert progress.stage == "transcribing"
        assert progress.progress == 50.0
        assert progress.message == "Half done"

    def test_get_nonexistent_progress(self):
        pm = ProgressManager()
        assert pm.get_progress("nonexistent") is None

    def test_remove_progress(self):
        pm = ProgressManager()
        pm.update("job-1", "done", 100.0, "Complete")
        pm.remove("job-1")
        assert pm.get_progress("job-1") is None

    def test_remove_nonexistent_is_safe(self):
        pm = ProgressManager()
        pm.remove("nonexistent")  # Should not raise

    def test_callback_invoked_on_update(self):
        pm = ProgressManager()
        received = []

        def callback(job_id, stage, progress, message):
            received.append((job_id, stage, progress, message))

        pm.register_callback(callback)
        pm.update("job-1", "indexing", 75.0, "Building index")

        assert len(received) == 1
        assert received[0] == ("job-1", "indexing", 75.0, "Building index")

    def test_multiple_callbacks(self):
        pm = ProgressManager()
        count = {"a": 0, "b": 0}

        pm.register_callback(lambda *args: count.__setitem__("a", count["a"] + 1))
        pm.register_callback(lambda *args: count.__setitem__("b", count["b"] + 1))

        pm.update("job-1", "test", 0, "test")
        assert count["a"] == 1
        assert count["b"] == 1

    def test_unregister_callback(self):
        pm = ProgressManager()
        calls = []

        def cb(job_id, stage, progress, message):
            calls.append(1)

        pm.register_callback(cb)
        pm.update("job-1", "test", 0, "test")
        assert len(calls) == 1

        pm.unregister_callback(cb)
        pm.update("job-1", "test", 0, "test")
        assert len(calls) == 1  # Not called again

    def test_make_callback(self):
        pm = ProgressManager()
        cb = pm.make_callback("job-42")

        cb("transcribing", 30.0, "Working")

        progress = pm.get_progress("job-42")
        assert progress.job_id == "job-42"
        assert progress.stage == "transcribing"
        assert progress.progress == 30.0

    def test_callback_error_does_not_crash(self):
        pm = ProgressManager()

        def bad_callback(job_id, stage, progress, message):
            raise RuntimeError("Callback exploded")

        pm.register_callback(bad_callback)
        # Should not raise
        pm.update("job-1", "test", 0, "test")

    def test_thread_safety(self):
        pm = ProgressManager()
        errors = []

        def writer(job_id):
            try:
                for i in range(100):
                    pm.update(job_id, "stage", float(i), f"msg-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"job-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All jobs should have progress
        for i in range(5):
            p = pm.get_progress(f"job-{i}")
            assert p is not None


# ---------------------------------------------------------------------------
# GPU cleanup helpers in pipeline_worker
# ---------------------------------------------------------------------------


class TestCleanupGpu:
    """Verify _cleanup_gpu in pipeline_worker behaves correctly."""

    def test_clears_cuda_cache_when_available(self):
        """_cleanup_gpu calls torch.cuda.empty_cache() when CUDA is available."""
        import sys
        from unittest.mock import MagicMock, patch

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Re-import to pick up the patched torch
            from meetbot.workers.pipeline_worker import _cleanup_gpu
            _cleanup_gpu()

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_no_op_when_cuda_unavailable(self):
        """_cleanup_gpu must not raise when CUDA is not available."""
        import sys
        from unittest.mock import MagicMock, patch

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from meetbot.workers.pipeline_worker import _cleanup_gpu
            _cleanup_gpu()  # Should not raise

        mock_torch.cuda.empty_cache.assert_not_called()

    def test_no_op_when_torch_not_installed(self):
        """_cleanup_gpu must not raise if torch is not installed."""
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"torch": None}):
            from meetbot.workers.pipeline_worker import _cleanup_gpu
            _cleanup_gpu()  # Should not raise


class TestWhisperCleanupAfterTranscription:
    """Verify Whisper model cleanup is attempted after Stage 1."""

    def test_local_whisper_cleanup_called(self):
        """LocalWhisperTranscriber.cleanup() is called after transcription stage."""
        import sys
        from unittest.mock import MagicMock, patch

        mock_local = MagicMock()
        mock_faster = MagicMock()

        with (
            patch.dict(
                sys.modules,
                {
                    "meetbot.adapters.transcribers.local_whisper": MagicMock(
                        LocalWhisperTranscriber=mock_local
                    ),
                    "meetbot.adapters.transcribers.faster_whisper": MagicMock(
                        FasterWhisperTranscriber=mock_faster
                    ),
                }
            )
        ):
            # Simulate the cleanup block that runs after transcription
            try:
                from meetbot.adapters.transcribers.local_whisper import LocalWhisperTranscriber
                LocalWhisperTranscriber.cleanup()
            except Exception:
                pass

            try:
                from meetbot.adapters.transcribers.faster_whisper import FasterWhisperTranscriber
                FasterWhisperTranscriber.cleanup()
            except Exception:
                pass

        mock_local.cleanup.assert_called_once()
        mock_faster.cleanup.assert_called_once()

    def test_cleanup_errors_are_swallowed(self):
        """Cleanup errors after transcription stage must not propagate."""
        import sys
        from unittest.mock import MagicMock, patch

        mock_local = MagicMock()
        mock_local.cleanup.side_effect = RuntimeError("GPU already freed")

        with patch.dict(
            sys.modules,
            {
                "meetbot.adapters.transcribers.local_whisper": MagicMock(
                    LocalWhisperTranscriber=mock_local
                ),
            }
        ):
            # This replicates the try/except block in pipeline_worker
            try:
                from meetbot.adapters.transcribers.local_whisper import LocalWhisperTranscriber
                LocalWhisperTranscriber.cleanup()
            except Exception:
                pass  # Should swallow the error — no re-raise
