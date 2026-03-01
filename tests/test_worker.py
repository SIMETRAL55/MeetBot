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
