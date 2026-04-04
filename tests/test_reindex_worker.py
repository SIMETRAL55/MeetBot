"""
Tests for meetbot/workers/reindex_worker.py.

Covers:
- Successful reindex: job status COMPLETED, segments_hash updated
- PAGEINDEX_ENABLED=True: PageIndexAdapter called, pageindex_status="ready"
- PageIndex failure (raises): job still COMPLETED, pageindex_status="failed"
- Skip-if-unchanged: matching hash → skip ChromaDB rebuild, COMPLETED
- Job not found → early return (no crash)
- Missing result_json_path → _fail called
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

JOB_ID = "aaaabbbb-cccc-dddd-eeee-ffffffffffff"
KNOWN_HASH = "abc123def456" + "0" * 52  # 64-char hex-like hash

_WORKER_BASE = "meetbot.workers.reindex_worker"


def _make_segment(idx: int = 0, text: str = "Hello", speaker: str = "SPEAKER_00") -> MagicMock:
    seg = MagicMock()
    seg.to_dict.return_value = {
        "segment_index": idx,
        "start": float(idx),
        "end": float(idx + 1),
        "speaker": speaker,
        "text": text,
    }
    return seg


def _make_job(result_json_path: str = "/fake/result.json", segments_hash: str = "") -> MagicMock:
    job = MagicMock()
    job.result_json_path = result_json_path
    job.segments_hash = segments_hash
    job.original_filename = "meeting.mp3"
    job.transcript_version = 1
    job.pageindex_path = None
    job.pageindex_status = None
    return job


def _setup_db_mocks(job: MagicMock, segments: list):
    """Return a (db_mock, session_factory_mock) pair wired to return the given job/segments."""
    db = MagicMock()
    db.close = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    session_factory = MagicMock(return_value=db)
    return db, session_factory


# ---------------------------------------------------------------------------
# Successful reindex
# ---------------------------------------------------------------------------


def test_successful_reindex_completes_job():
    """run_reindex should set job to COMPLETED and update segments_hash."""
    job = _make_job()
    segments = [_make_segment(0, "Hello", "A"), _make_segment(1, "World", "B")]
    db, session_factory = _setup_db_mocks(job, segments)

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status") as mock_update_status,
        patch(f"{_WORKER_BASE}.update_job_result") as mock_update_result,
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=2),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
              return_value="/fake/chroma"),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
    ):
        mock_settings.PAGEINDEX_ENABLED = False
        mock_settings.VECTOR_DB_PATH = "/fake/chroma/collection"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20

        # result_json_path must exist
        mock_path_cls.return_value.exists.return_value = True
        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    # Final update_job_status call should be COMPLETED
    from meetbot.db.models import JobStatus
    completed_calls = [
        c for c in mock_update_status.call_args_list
        if len(c.args) > 2 and c.args[2] == JobStatus.COMPLETED
    ]
    assert completed_calls, "Expected at least one COMPLETED status update"

    # segments_hash should have been set on the job object
    assert job.segments_hash != ""
    db.commit.assert_called()


def test_segments_hash_written_after_indexing():
    """The segments_hash on the job must be updated after a successful index build."""
    from meetbot.services.rag.indexer import _hash_segments

    job = _make_job(segments_hash="")
    segments = [_make_segment(0, "Test text", "SPK")]
    db, session_factory = _setup_db_mocks(job, segments)

    expected_hash = _hash_segments([s.to_dict() for s in segments])

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status"),
        patch(f"{_WORKER_BASE}.update_job_result"),
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=1),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
              return_value="/fake/chroma"),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
    ):
        mock_settings.PAGEINDEX_ENABLED = False
        mock_settings.VECTOR_DB_PATH = "/fake/chroma/collection"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20
        mock_path_cls.return_value.exists.return_value = True
        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    assert job.segments_hash == expected_hash


# ---------------------------------------------------------------------------
# Skip-if-unchanged
# ---------------------------------------------------------------------------


def test_skip_if_unchanged_does_not_call_indexer():
    """If stored_hash matches, ChromaDB rebuild should be skipped."""
    from meetbot.services.rag.indexer import _hash_segments

    segments = [_make_segment(0, "Same text", "SPK")]
    seg_dicts = [s.to_dict() for s in segments]
    current_hash = _hash_segments(seg_dicts)

    job = _make_job(segments_hash=current_hash)
    db, session_factory = _setup_db_mocks(job, segments)

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status") as mock_update_status,
        patch(f"{_WORKER_BASE}.update_job_result") as mock_update_result,
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=1),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic") as mock_idx,
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
    ):
        mock_settings.PAGEINDEX_ENABLED = False
        mock_settings.VECTOR_DB_PATH = "/fake/chroma/collection"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20
        mock_path_cls.return_value.exists.return_value = True
        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    mock_idx.assert_not_called()

    from meetbot.db.models import JobStatus
    completed_calls = [
        c for c in mock_update_status.call_args_list
        if len(c.args) > 2 and c.args[2] == JobStatus.COMPLETED
    ]
    assert completed_calls


# ---------------------------------------------------------------------------
# PageIndex integration
# ---------------------------------------------------------------------------


def test_pageindex_enabled_calls_adapter():
    """When PAGEINDEX_ENABLED=True, PageIndexAdapter.build_index should be called."""
    job = _make_job()
    segments = [_make_segment(0, "Topic text", "SPK")]
    db, session_factory = _setup_db_mocks(job, segments)

    mock_adapter = MagicMock()
    mock_adapter.build_index = AsyncMock(return_value=Path("/fake/pi.json"))

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status"),
        patch(f"{_WORKER_BASE}.update_job_result"),
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=1),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
              return_value="/fake/chroma"),
        patch("meetbot.services.rag.indexer_pageindex.PageIndexAdapter",
              return_value=mock_adapter),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
    ):
        mock_settings.PAGEINDEX_ENABLED = True
        mock_settings.VECTOR_DB_PATH = "/fake/chroma/collection"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20
        mock_path_cls.return_value.exists.return_value = True
        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    mock_adapter.build_index.assert_called_once()
    assert job.pageindex_status == "ready"


def test_pageindex_failure_is_nonfatal():
    """PageIndex build failure must not fail the main job — status stays COMPLETED."""
    job = _make_job()
    segments = [_make_segment(0, "Topic text", "SPK")]
    db, session_factory = _setup_db_mocks(job, segments)

    mock_adapter = MagicMock()
    mock_adapter.build_index = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status") as mock_update_status,
        patch(f"{_WORKER_BASE}.update_job_result"),
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=1),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
              return_value="/fake/chroma"),
        patch("meetbot.services.rag.indexer_pageindex.PageIndexAdapter",
              return_value=mock_adapter),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
    ):
        mock_settings.PAGEINDEX_ENABLED = True
        mock_settings.VECTOR_DB_PATH = "/fake/chroma/collection"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20
        mock_path_cls.return_value.exists.return_value = True
        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    assert job.pageindex_status == "failed"

    from meetbot.db.models import JobStatus
    completed_calls = [
        c for c in mock_update_status.call_args_list
        if len(c.args) > 2 and c.args[2] == JobStatus.COMPLETED
    ]
    assert completed_calls, "Job should still be COMPLETED despite PageIndex failure"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_job_not_found_returns_early():
    """run_reindex should return silently if job doesn't exist."""
    db, session_factory = MagicMock(), MagicMock()
    session_factory.return_value = db

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=None),
        patch(f"{_WORKER_BASE}.progress_manager"),
        patch(f"{_WORKER_BASE}.cancel_registry"),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
    ):
        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)  # Should not raise


def test_missing_result_json_fails_job():
    """If result_json_path is missing, _fail should be called."""
    job = _make_job(result_json_path=None)
    db, session_factory = _setup_db_mocks(job, [])

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status") as mock_update_status,
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry"),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
    ):
        mock_pm.make_callback.return_value = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    from meetbot.db.models import JobStatus
    failed_calls = [
        c for c in mock_update_status.call_args_list
        if len(c.args) > 2 and c.args[2] == JobStatus.FAILED
    ]
    assert failed_calls, "Expected FAILED status when result JSON is missing"
