"""
Tests for database layer — models, CRUD operations, and database lifecycle.
"""

import os
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
    # Drop all tables to ensure clean state
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


class TestUserCRUD:
    """Test User create/read/update operations."""

    def test_create_user(self, db_session):
        from meetbot.db.crud import create_user, get_user_by_username

        user = create_user(
            db_session,
            username="testuser",
            password_hash="fakehash",
            display_name="Test User",
        )
        assert user.id is not None
        assert user.username == "testuser"
        assert user.display_name == "Test User"
        assert user.is_admin is False

        found = get_user_by_username(db_session, "testuser")
        assert found is not None
        assert found.id == user.id

    def test_create_admin_user(self, db_session):
        from meetbot.db.crud import create_user

        admin = create_user(
            db_session,
            username="admin",
            password_hash="fakehash",
            is_admin=True,
        )
        assert admin.is_admin is True

    def test_get_nonexistent_user(self, db_session):
        from meetbot.db.crud import get_user_by_username

        user = get_user_by_username(db_session, "doesnotexist")
        assert user is None

    def test_duplicate_username_fails(self, db_session):
        from meetbot.db.crud import create_user
        from sqlalchemy.exc import IntegrityError

        create_user(db_session, username="dupe", password_hash="hash1")
        with pytest.raises(IntegrityError):
            create_user(db_session, username="dupe", password_hash="hash2")

    def test_update_last_login(self, db_session):
        from meetbot.db.crud import create_user, update_user_last_login

        user = create_user(db_session, username="logintest", password_hash="hash")
        assert user.last_login is None

        update_user_last_login(db_session, user)
        assert user.last_login is not None

    def test_list_users(self, db_session):
        from meetbot.db.crud import create_user, list_users

        create_user(db_session, username="user1", password_hash="h1")
        create_user(db_session, username="user2", password_hash="h2")

        users = list_users(db_session)
        assert len(users) == 2


class TestJobCRUD:
    """Test Job create/read/update/delete operations."""

    def _create_test_user(self, db_session):
        from meetbot.db.crud import create_user

        return create_user(
            db_session, username="jobuser", password_hash="hash"
        )

    def test_create_job(self, db_session):
        from meetbot.db.crud import create_job, get_job
        from meetbot.db.models import JobStatus

        user = self._create_test_user(db_session)
        job = create_job(
            db_session,
            user_id=user.id,
            filename="abc123.wav",
            original_filename="meeting.wav",
            file_size=1024000,
            language="ja",
        )

        assert job.id is not None
        assert job.status == JobStatus.PENDING
        assert job.original_filename == "meeting.wav"
        assert job.file_size == 1024000

        found = get_job(db_session, job.id)
        assert found is not None
        assert found.id == job.id

    def test_update_job_status(self, db_session):
        from meetbot.db.crud import create_job, update_job_status
        from meetbot.db.models import JobStatus

        user = self._create_test_user(db_session)
        job = create_job(
            db_session,
            user_id=user.id,
            filename="test.wav",
            original_filename="test.wav",
        )

        updated = update_job_status(
            db_session, job.id,
            JobStatus.TRANSCRIBING, 25.0, "Transcribing..."
        )
        assert updated.status == JobStatus.TRANSCRIBING
        assert updated.progress == 25.0
        assert updated.started_at is not None

    def test_update_job_failed(self, db_session):
        from meetbot.db.crud import create_job, update_job_status
        from meetbot.db.models import JobStatus

        user = self._create_test_user(db_session)
        job = create_job(
            db_session,
            user_id=user.id,
            filename="fail.wav",
            original_filename="fail.wav",
        )

        updated = update_job_status(
            db_session, job.id,
            JobStatus.FAILED,
            error_message="Out of VRAM",
        )
        assert updated.status == JobStatus.FAILED
        assert updated.error_message == "Out of VRAM"
        assert updated.completed_at is not None

    def test_get_jobs_for_user(self, db_session):
        from meetbot.db.crud import create_job, get_jobs_for_user

        user = self._create_test_user(db_session)
        create_job(db_session, user_id=user.id, filename="a.wav", original_filename="a.wav")
        create_job(db_session, user_id=user.id, filename="b.wav", original_filename="b.wav")

        jobs = get_jobs_for_user(db_session, user.id)
        assert len(jobs) == 2

    def test_delete_job(self, db_session):
        from meetbot.db.crud import create_job, delete_job, get_job

        user = self._create_test_user(db_session)
        job = create_job(
            db_session, user_id=user.id,
            filename="del.wav", original_filename="del.wav",
        )

        assert delete_job(db_session, job.id) is True
        assert get_job(db_session, job.id) is None

    def test_delete_nonexistent_job(self, db_session):
        from meetbot.db.crud import delete_job

        assert delete_job(db_session, "nonexistent-id") is False

    def test_update_job_result(self, db_session):
        from meetbot.db.crud import create_job, update_job_result

        user = self._create_test_user(db_session)
        job = create_job(
            db_session, user_id=user.id,
            filename="res.wav", original_filename="res.wav",
        )

        updated = update_job_result(
            db_session, job.id,
            result_json_path="/app/results/res.json",
            db_dir="/app/db/res",
            duration_seconds=120.5,
        )
        assert updated.result_json_path == "/app/results/res.json"
        assert updated.db_dir == "/app/db/res"
        assert updated.duration_seconds == 120.5


class TestSegmentCRUD:
    """Test Segment create/read/update operations."""

    def _create_test_job(self, db_session):
        from meetbot.db.crud import create_user, create_job

        user = create_user(db_session, username="seguser", password_hash="h")
        return create_job(
            db_session, user_id=user.id,
            filename="seg.wav", original_filename="seg.wav",
        )

    def test_create_segments(self, db_session):
        from meetbot.db.crud import create_segments_from_aligned, get_segments_for_job

        job = self._create_test_job(db_session)
        aligned = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "Hi there"},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00", "text": "How are you?"},
        ]

        create_segments_from_aligned(db_session, job.id, aligned)
        segments = get_segments_for_job(db_session, job.id)

        assert len(segments) == 3
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].text == "Hi there"
        assert segments[2].start_time == 10.0

    def test_update_segment_speaker(self, db_session):
        from meetbot.db.crud import (
            create_segments_from_aligned,
            get_segments_for_job,
            update_segment_speaker,
        )

        job = self._create_test_job(db_session)
        create_segments_from_aligned(db_session, job.id, [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Test"},
        ])

        segments = get_segments_for_job(db_session, job.id)
        updated = update_segment_speaker(db_session, segments[0].id, "田中さん")
        assert updated.speaker == "田中さん"

    def test_bulk_rename_speaker(self, db_session):
        from meetbot.db.crud import (
            create_segments_from_aligned,
            get_segments_for_job,
            bulk_update_speaker_name,
        )

        job = self._create_test_job(db_session)
        create_segments_from_aligned(db_session, job.id, [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "A"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "B"},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00", "text": "C"},
        ])

        count = bulk_update_speaker_name(
            db_session, job.id, "SPEAKER_00", "Alice"
        )
        assert count == 2

        segments = get_segments_for_job(db_session, job.id)
        speakers = [s.speaker for s in segments]
        assert speakers == ["Alice", "SPEAKER_01", "Alice"]

    def test_update_segment_text(self, db_session):
        from meetbot.db.crud import (
            create_segments_from_aligned,
            get_segments_for_job,
            update_segment_text,
        )

        job = self._create_test_job(db_session)
        create_segments_from_aligned(db_session, job.id, [
            {"start": 0.0, "end": 5.0, "speaker": "SPK", "text": "original"},
        ])

        segments = get_segments_for_job(db_session, job.id)
        updated = update_segment_text(db_session, segments[0].id, "corrected text")
        assert updated.text == "corrected text"

    def test_segment_to_dict(self, db_session):
        from meetbot.db.crud import create_segments_from_aligned, get_segments_for_job

        job = self._create_test_job(db_session)
        create_segments_from_aligned(db_session, job.id, [
            {"start": 1.5, "end": 3.5, "speaker": "SPK", "text": "hello"},
        ])

        segments = get_segments_for_job(db_session, job.id)
        d = segments[0].to_dict()
        assert d["start"] == 1.5
        assert d["end"] == 3.5
        assert d["speaker"] == "SPK"
        assert d["text"] == "hello"
        assert "id" in d

    def test_cascade_delete(self, db_session):
        """Deleting a job should delete its segments."""
        from meetbot.db.crud import (
            create_segments_from_aligned,
            get_segments_for_job,
            delete_job,
        )

        job = self._create_test_job(db_session)
        create_segments_from_aligned(db_session, job.id, [
            {"start": 0.0, "end": 5.0, "speaker": "SPK", "text": "test"},
        ])

        delete_job(db_session, job.id)
        segments = get_segments_for_job(db_session, job.id)
        assert len(segments) == 0


# Cleanup test database
@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    try:
        os.unlink(_test_db_path)
    except OSError:
        pass
