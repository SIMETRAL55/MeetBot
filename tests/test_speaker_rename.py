"""
Tests for bulk speaker rename — both the CRUD layer and the HTTP endpoint.
"""
import os
import tempfile

import pytest

_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_db():
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
    from meetbot.db.database import get_session
    SessionLocal = get_session()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def user_and_job(db_session):
    """Create a user + completed job with a few segments."""
    from meetbot.db import crud

    user = crud.create_user(db_session, username="renameuser", password_hash="h")
    job = crud.create_job(db_session, user_id=user.id, filename="test.mp3", original_filename="test.mp3")
    # Mark completed so flush_segments_to_json has something to write
    job.status = "completed"
    db_session.commit()

    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "Hello"},
        {"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "World"},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0, "text": "Hi there"},
    ]
    crud.create_segments_from_aligned(db_session, job.id, segments)

    return user, job


# ---------------------------------------------------------------------------
# CRUD-level tests
# ---------------------------------------------------------------------------

class TestBulkUpdateSpeakerName:
    def test_renames_all_matching_segments(self, db_session, user_and_job):
        from meetbot.db import crud
        _, job = user_and_job

        count = crud.bulk_update_speaker_name(db_session, job.id, "SPEAKER_00", "Alice")
        assert count == 2

        segments = crud.get_segments_for_job(db_session, job.id)
        alice_segs = [s for s in segments if s.speaker == "Alice"]
        speaker00_segs = [s for s in segments if s.speaker == "SPEAKER_00"]
        assert len(alice_segs) == 2
        assert len(speaker00_segs) == 0

    def test_unmatched_speaker_returns_zero(self, db_session, user_and_job):
        from meetbot.db import crud
        _, job = user_and_job

        count = crud.bulk_update_speaker_name(db_session, job.id, "SPEAKER_99", "Bob")
        assert count == 0

    def test_other_speakers_unchanged(self, db_session, user_and_job):
        from meetbot.db import crud
        _, job = user_and_job

        crud.bulk_update_speaker_name(db_session, job.id, "SPEAKER_00", "Alice")

        segments = crud.get_segments_for_job(db_session, job.id)
        speaker01_segs = [s for s in segments if s.speaker == "SPEAKER_01"]
        assert len(speaker01_segs) == 1

    def test_rename_to_same_name_no_op(self, db_session, user_and_job):
        from meetbot.db import crud
        _, job = user_and_job

        count = crud.bulk_update_speaker_name(db_session, job.id, "SPEAKER_00", "SPEAKER_00")
        # Should succeed (0 or 2 rows updated depending on DB engine, just no error)
        assert count >= 0


# ---------------------------------------------------------------------------
# HTTP endpoint tests (minimal starlette test client)
# ---------------------------------------------------------------------------

class TestRenameSpeakerEndpoint:
    def test_valid_rename_returns_200(self, db_session, user_and_job, monkeypatch):
        from meetbot.db import crud
        from meetbot.db.database import get_session
        import meetbot.web.auth_middleware as mw

        user, job = user_and_job

        # Patch get_optional_user to return the user dict
        monkeypatch.setattr(mw, "get_optional_user", lambda req: {"sub": user.id})
        # Patch get_session to return our test session factory
        monkeypatch.setattr("meetbot.web.api.get_session", get_session)
        # Stub flush (no output dir in tests)
        monkeypatch.setattr(crud, "flush_segments_to_json", lambda db, job_id: None)

        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from meetbot.web.api import api_rename_speaker

        app = FastAPI()
        app.add_api_route("/api/jobs/{job_id}/rename-speaker", api_rename_speaker, methods=["POST"])
        client = TestClient(app, raise_server_exceptions=True)

        resp = client.post(
            f"/api/jobs/{job.id}/rename-speaker",
            json={"old_name": "SPEAKER_00", "new_name": "Alice"},
        )
        assert resp.status_code == 200
        assert resp.json()["renamed"] == 2

    def test_missing_fields_returns_422(self, db_session, user_and_job, monkeypatch):
        from meetbot.db.database import get_session
        from meetbot.db import crud
        import meetbot.web.auth_middleware as mw

        user, job = user_and_job
        monkeypatch.setattr(mw, "get_optional_user", lambda req: {"sub": user.id})
        monkeypatch.setattr("meetbot.web.api.get_session", get_session)
        monkeypatch.setattr(crud, "flush_segments_to_json", lambda db, job_id: None)

        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from meetbot.web.api import api_rename_speaker

        app = FastAPI()
        app.add_api_route("/api/jobs/{job_id}/rename-speaker", api_rename_speaker, methods=["POST"])
        client = TestClient(app, raise_server_exceptions=True)

        resp = client.post(
            f"/api/jobs/{job.id}/rename-speaker",
            json={"old_name": "SPEAKER_00"},  # new_name missing
        )
        assert resp.status_code == 422

    def test_wrong_user_returns_404(self, db_session, user_and_job, monkeypatch):
        from meetbot.db.database import get_session
        from meetbot.db import crud
        import meetbot.web.auth_middleware as mw

        user, job = user_and_job
        # Patch to a DIFFERENT user ID
        monkeypatch.setattr(mw, "get_optional_user", lambda req: {"sub": "other-user-id"})
        monkeypatch.setattr("meetbot.web.api.get_session", get_session)
        monkeypatch.setattr(crud, "flush_segments_to_json", lambda db, job_id: None)

        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from meetbot.web.api import api_rename_speaker

        app = FastAPI()
        app.add_api_route("/api/jobs/{job_id}/rename-speaker", api_rename_speaker, methods=["POST"])
        client = TestClient(app, raise_server_exceptions=True)

        resp = client.post(
            f"/api/jobs/{job.id}/rename-speaker",
            json={"old_name": "SPEAKER_00", "new_name": "Alice"},
        )
        assert resp.status_code == 404

    def test_unauthenticated_returns_401(self, db_session, user_and_job, monkeypatch):
        from meetbot.db.database import get_session
        from meetbot.db import crud
        import meetbot.web.auth_middleware as mw

        _, job = user_and_job
        monkeypatch.setattr(mw, "get_optional_user", lambda req: None)
        monkeypatch.setattr("meetbot.web.api.get_session", get_session)
        monkeypatch.setattr(crud, "flush_segments_to_json", lambda db, job_id: None)

        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from meetbot.web.api import api_rename_speaker

        app = FastAPI()
        app.add_api_route("/api/jobs/{job_id}/rename-speaker", api_rename_speaker, methods=["POST"])
        client = TestClient(app, raise_server_exceptions=True)

        resp = client.post(
            f"/api/jobs/{job.id}/rename-speaker",
            json={"old_name": "SPEAKER_00", "new_name": "Alice"},
        )
        assert resp.status_code == 401
