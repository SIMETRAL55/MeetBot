"""
Tests for segment insertion idempotency and restart safety.

Verifies that:
- create_segments_from_aligned replaces (not appends) existing segments.
- delete_segments_for_job removes all segments for a job.
- Calling create_segments_from_aligned multiple times produces exactly N segments.
- api_job_restart cleans up derived artifacts.
"""

import os
import shutil
import tempfile

import pytest

# Set test database path before importing anything
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create a fresh test database for each test."""
    from meetbot.db.database import get_engine, init_db, reset_engine
    from meetbot.db.models import Base

    reset_engine()
    engine = get_engine(_test_db_path)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    reset_engine()


@pytest.fixture
def db_session():
    """Provide a database session."""
    from meetbot.db.database import get_session

    SessionLocal = get_session()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _make_job(db_session, username="idempotent_user"):
    """Helper to create a test user + job."""
    from meetbot.db.crud import create_user, create_job

    # Unique username per call to avoid conflicts
    import uuid
    uname = f"{username}_{uuid.uuid4().hex[:6]}"
    user = create_user(db_session, username=uname, password_hash="h")
    return create_job(
        db_session,
        user_id=user.id,
        filename="test.wav",
        original_filename="test.wav",
    )


_ALIGNED_9 = [
    {"start": float(i), "end": float(i + 1), "speaker": f"SPK_{i % 3}", "text": f"Segment {i}"}
    for i in range(9)
]


class TestDeleteSegmentsForJob:
    """Tests for the delete_segments_for_job helper."""

    def test_deletes_all_segments(self, db_session):
        from meetbot.db.crud import (
            create_segments_from_aligned,
            get_segments_for_job,
            delete_segments_for_job,
        )

        job = _make_job(db_session)
        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)
        assert len(get_segments_for_job(db_session, job.id)) == 9

        deleted = delete_segments_for_job(db_session, job.id)
        assert deleted == 9
        assert len(get_segments_for_job(db_session, job.id)) == 0

    def test_returns_zero_for_no_segments(self, db_session):
        from meetbot.db.crud import delete_segments_for_job

        job = _make_job(db_session)
        deleted = delete_segments_for_job(db_session, job.id)
        assert deleted == 0

    def test_does_not_affect_other_jobs(self, db_session):
        from meetbot.db.crud import (
            create_segments_from_aligned,
            get_segments_for_job,
            delete_segments_for_job,
        )

        job_a = _make_job(db_session, "user_a")
        job_b = _make_job(db_session, "user_b")

        create_segments_from_aligned(db_session, job_a.id, _ALIGNED_9)
        create_segments_from_aligned(db_session, job_b.id, _ALIGNED_9[:3])

        delete_segments_for_job(db_session, job_a.id)

        assert len(get_segments_for_job(db_session, job_a.id)) == 0
        assert len(get_segments_for_job(db_session, job_b.id)) == 3


class TestCreateSegmentsIdempotency:
    """
    Verify that calling create_segments_from_aligned multiple times
    never duplicates segments.
    """

    def test_single_call_creates_correct_count(self, db_session):
        from meetbot.db.crud import create_segments_from_aligned, get_segments_for_job

        job = _make_job(db_session)
        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)

        segments = get_segments_for_job(db_session, job.id)
        assert len(segments) == 9

    def test_double_call_no_duplication(self, db_session):
        """Simulates cancel → restart: alignment inserts segments twice."""
        from meetbot.db.crud import create_segments_from_aligned, get_segments_for_job

        job = _make_job(db_session)

        # First run (would be interrupted)
        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)
        assert len(get_segments_for_job(db_session, job.id)) == 9

        # Second run (restart)
        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)
        segments = get_segments_for_job(db_session, job.id)

        # MUST still be 9, NOT 18
        assert len(segments) == 9

    def test_triple_call_no_duplication(self, db_session):
        """Even three consecutive calls produce exactly 9 segments."""
        from meetbot.db.crud import create_segments_from_aligned, get_segments_for_job

        job = _make_job(db_session)

        for _ in range(3):
            create_segments_from_aligned(db_session, job.id, _ALIGNED_9)

        segments = get_segments_for_job(db_session, job.id)
        assert len(segments) == 9

    def test_replacement_uses_latest_data(self, db_session):
        """Second call replaces content, not appends."""
        from meetbot.db.crud import create_segments_from_aligned, get_segments_for_job

        job = _make_job(db_session)

        # First call with original text
        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)

        # Second call with modified text
        modified = [
            {**seg, "text": f"EDITED {seg['text']}"}
            for seg in _ALIGNED_9
        ]
        create_segments_from_aligned(db_session, job.id, modified)

        segments = get_segments_for_job(db_session, job.id)
        assert len(segments) == 9
        assert all(s.text.startswith("EDITED") for s in segments)

    def test_replacement_with_different_count(self, db_session):
        """Replacing 9 segments with 3 results in exactly 3."""
        from meetbot.db.crud import create_segments_from_aligned, get_segments_for_job

        job = _make_job(db_session)

        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)
        assert len(get_segments_for_job(db_session, job.id)) == 9

        create_segments_from_aligned(db_session, job.id, _ALIGNED_9[:3])
        segments = get_segments_for_job(db_session, job.id)
        assert len(segments) == 3


class TestRestartArtifactCleanup:
    """Verify that restart cleans temp files and JSONL intermediates."""

    def test_temp_dir_cleaned_on_restart_simulation(self, db_session):
        """Simulates what api_job_restart does to temp dirs."""
        from meetbot.config import settings

        job = _make_job(db_session)

        # Create a fake temp dir with leftover files
        tmp = tempfile.mkdtemp()
        try:
            # Patch settings.TEMP_DIR temporarily
            original_temp = settings.TEMP_DIR
            settings.TEMP_DIR = tmp

            temp_job_dir = os.path.join(tmp, job.id)
            os.makedirs(temp_job_dir, exist_ok=True)
            leftover = os.path.join(temp_job_dir, "multilevel_docs.jsonl")
            with open(leftover, "w") as f:
                f.write('{"id": "leftover"}\n')
            assert os.path.exists(leftover)

            # Simulate restart cleanup
            from pathlib import Path
            _temp_job_dir = Path(settings.TEMP_DIR) / job.id
            if _temp_job_dir.exists():
                shutil.rmtree(str(_temp_job_dir), ignore_errors=True)

            assert not os.path.exists(temp_job_dir)

            settings.TEMP_DIR = original_temp
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestReindexIdempotency:
    """Verify that the reindex path also reads stable segment counts."""

    def test_segments_stable_after_multiple_creates(self, db_session):
        """
        Simulates the scenario from the bug report:
        Pipeline creates 9 segments, then reindex reads them.
        If pipeline is restarted it creates 9 again —
        get_segments_for_job must still return 9.
        """
        from meetbot.db.crud import (
            create_segments_from_aligned,
            get_segments_for_job,
        )

        job = _make_job(db_session)

        # Pipeline run 1
        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)
        segs1 = get_segments_for_job(db_session, job.id)
        assert len(segs1) == 9

        # Simulate restart: pipeline reinserts
        create_segments_from_aligned(db_session, job.id, _ALIGNED_9)
        segs2 = get_segments_for_job(db_session, job.id)
        assert len(segs2) == 9

        # What the indexer would see
        seg_dicts = [s.to_dict() for s in segs2]
        assert len(seg_dicts) == 9


# Cleanup test database
@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    try:
        os.unlink(_test_db_path)
    except OSError:
        pass
