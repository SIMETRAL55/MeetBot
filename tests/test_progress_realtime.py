"""
Tests for real-time progress features:
- ProgressManager pub/sub (asyncio queues, thread-safe bridging)
- DB migration (stage_progress + logs columns)
- update_job_status with stage_progress and log_line
- WebSocket endpoint handler (ws.py)
"""

import asyncio
import json
import os
import threading
import time
import pytest
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("MEETBOT_OUTPUT_DIR", "/tmp/meetbot_test_rt/results")

# ---------------------------------------------------------------------------
# DB fixtures (isolated SQLite file per test module run)
# ---------------------------------------------------------------------------

from meetbot.db.models import Base, JobStatus
from meetbot.db.database import _run_migrations


def _make_engine(path: str):
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_pragmas(conn, rec):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

    return engine


@pytest.fixture()
def db_session(tmp_path):
    """Provide a clean in-memory DB session for each test."""
    db_file = str(tmp_path / "test.db")
    engine = _make_engine(db_file)
    Base.metadata.create_all(bind=engine)
    _run_migrations(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess, engine
    sess.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from meetbot.db.crud import create_user, create_job, update_job_status
from meetbot.web.auth import hash_password


def _seed_job(sess, engine):
    """Create a user + job and return job.id."""
    import meetbot.db.database as dbmod
    from meetbot.db.database import reset_engine

    # Reset any stale engine/session cached from previous tests, then
    # inject the test-specific engine so all crud helpers use it.
    reset_engine()
    dbmod._engine = engine
    Session = sessionmaker(bind=engine)
    dbmod._SessionLocal = Session  # must match the global checked by get_session()

    db = Session()
    user = create_user(db, "rt_user", hash_password("pw"), is_admin=False)
    job = create_job(
        db,
        user_id=user.id,
        filename="test.wav",
        original_filename="test.wav",
        file_size=1024,
    )
    job_id = job.id
    db.close()
    return job_id


# ===========================================================================
# 1. DB migration — stage_progress and logs columns added
# ===========================================================================


class TestDbMigration:
    def test_columns_exist_after_migration(self, tmp_path):
        """_run_migrations() must add stage_progress and logs if absent."""
        db_file = str(tmp_path / "mig_test.db")
        engine = _make_engine(db_file)

        # Create table WITHOUT the new columns (simulate pre-migration state)
        with engine.connect() as conn:
            conn.execute(
                __import__("sqlalchemy").text(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        original_filename TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        progress REAL NOT NULL DEFAULT 0.0,
                        progress_message TEXT
                    )
                    """
                )
            )
            conn.commit()

        # Run migration — should add stage_progress and logs
        _run_migrations(engine)

        with engine.connect() as conn:
            result = conn.execute(
                __import__("sqlalchemy").text("PRAGMA table_info(jobs)")
            )
            cols = {row[1] for row in result.fetchall()}

        assert "stage_progress" in cols, "stage_progress column missing after migration"
        assert "logs" in cols, "logs column missing after migration"
        engine.dispose()

    def test_migration_idempotent(self, tmp_path):
        """Running migration twice must not raise."""
        db_file = str(tmp_path / "idem_test.db")
        engine = _make_engine(db_file)
        Base.metadata.create_all(bind=engine)
        _run_migrations(engine)
        _run_migrations(engine)  # second run — must not throw
        engine.dispose()


# ===========================================================================
# 2. update_job_status with stage_progress + log_line
# ===========================================================================


class TestUpdateJobStatus:
    def test_stage_progress_persisted(self, db_session):
        sess, engine = db_session
        job_id = _seed_job(sess, engine)

        from meetbot.db.crud import get_job
        import meetbot.db.database as dbmod

        db = sessionmaker(bind=engine)()
        update_job_status(
            db,
            job_id,
            JobStatus.TRANSCRIBING,
            progress=20.0,
            stage_progress=50.0,
            progress_message="halfway",
        )
        job = get_job(db, job_id)
        db.close()

        assert job.progress == pytest.approx(20.0)
        assert job.stage_progress == pytest.approx(50.0)
        assert job.progress_message == "halfway"

    def test_log_line_appended(self, db_session):
        sess, engine = db_session
        job_id = _seed_job(sess, engine)

        from meetbot.db.crud import get_job

        db = sessionmaker(bind=engine)()
        update_job_status(
            db, job_id, JobStatus.TRANSCRIBING,
            progress=10.0, log_line="First log"
        )
        update_job_status(
            db, job_id, JobStatus.TRANSCRIBING,
            progress=20.0, log_line="Second log"
        )
        job = get_job(db, job_id)
        db.close()

        stored = json.loads(job.logs)
        assert "First log" in stored
        assert "Second log" in stored

    def test_logs_capped_at_50(self, db_session):
        sess, engine = db_session
        job_id = _seed_job(sess, engine)

        from meetbot.db.crud import get_job

        db = sessionmaker(bind=engine)()
        for i in range(60):
            update_job_status(
                db, job_id, JobStatus.TRANSCRIBING,
                progress=float(i), log_line=f"line {i}"
            )
        job = get_job(db, job_id)
        db.close()

        stored = json.loads(job.logs)
        assert len(stored) == 50, f"Expected 50 logs, got {len(stored)}"
        assert stored[-1] == "line 59"


# ===========================================================================
# 3. ProgressManager pub/sub
# ===========================================================================


from meetbot.workers.progress import ProgressManager, JobProgress


class TestProgressManagerPubSub:
    def test_subscribe_receives_cached_state(self):
        """subscribe() after an update delivers the current state immediately."""
        pm = ProgressManager()

        # Pre-populate state
        pm._progress["job-1"] = JobProgress(
            job_id="job-1",
            stage="transcribing",
            stage_progress=25.0,
            progress=10.0,
            message="hi",
            status="running",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            q = pm.subscribe("job-1")
            event = loop.run_until_complete(
                asyncio.wait_for(q.get(), timeout=0.5)
            )
            assert event["job_id"] == "job-1"
            assert event["stage"] == "transcribing"
            assert event["stage_progress"] == pytest.approx(25.0)
        finally:
            loop.close()

    def test_update_pushes_to_subscriber(self):
        """update() from a worker thread must put events into subscriber queues."""
        pm = ProgressManager()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pm.set_loop(loop)

        received: list = []

        async def collect():
            q = pm.subscribe("job-2")
            # Skip the initial empty cached state (no pre-existing state here)
            # Worker thread fires update after a tiny delay
            event = await asyncio.wait_for(q.get(), timeout=2.0)
            received.append(event)

        def worker():
            time.sleep(0.05)
            pm.update(
                "job-2",
                stage="diarizing",
                progress=55.0,
                message="diarizing",
                stage_progress=80.0,
            )

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        try:
            loop.run_until_complete(collect())
        finally:
            t.join(timeout=2)
            loop.close()

        assert len(received) == 1
        ev = received[0]
        assert ev["stage"] == "diarizing"
        assert ev["overall_progress"] == pytest.approx(55.0)
        assert ev["stage_progress"] == pytest.approx(80.0)

    def test_unsubscribe_stops_delivery(self):
        """After unsubscribe, no more events are delivered to the queue."""
        pm = ProgressManager()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pm.set_loop(loop)

        async def run():
            q = pm.subscribe("job-3")
            pm.unsubscribe("job-3", q)
            # update after unsubscribe
            pm.update("job-3", stage="aligning", progress=70.0, message="x")
            # Queue should be empty
            assert q.empty()

        loop.run_until_complete(run())
        loop.close()

    def test_build_event_shape(self):
        """_build_event produces the required JSON-serialisable shape."""
        jp = JobProgress(
            job_id="abc",
            stage="indexing",
            stage_progress=33.3,
            progress=87.5,
            message="working",
            logs=["a", "b"],
            status="running",
        )
        ev = ProgressManager._build_event(jp)
        assert set(ev.keys()) == {
            "job_id", "stage", "stage_progress", "overall_progress",
            "logs", "status", "message",
        }
        # Must be JSON-serialisable
        json.dumps(ev)

    def test_logs_accumulated_in_progress(self):
        """Log lines accumulate in JobProgress across successive update() calls."""
        pm = ProgressManager()
        pm.update("j", "transcribing", 10, "first line")
        pm.update("j", "transcribing", 20, "second line")
        jp = pm.get_progress("j")
        assert "first line" in jp.logs
        assert "second line" in jp.logs

    def test_logs_capped_at_50(self):
        """JobProgress.logs never grows beyond 50 entries."""
        pm = ProgressManager()
        for i in range(60):
            pm.update("j", "transcribing", float(i), f"msg {i}")
        jp = pm.get_progress("j")
        assert len(jp.logs) <= 50

    def test_multiple_subscribers(self):
        """Two simultaneous subscribers each receive the same event."""
        pm = ProgressManager()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pm.set_loop(loop)

        received_a: list = []
        received_b: list = []

        async def run():
            qa = pm.subscribe("jj")
            qb = pm.subscribe("jj")

            def worker():
                time.sleep(0.02)
                pm.update("jj", stage="aligning", progress=68.0, message="y")

            t = threading.Thread(target=worker, daemon=True)
            t.start()

            received_a.append(await asyncio.wait_for(qa.get(), timeout=1.0))
            received_b.append(await asyncio.wait_for(qb.get(), timeout=1.0))
            t.join(timeout=1)

        loop.run_until_complete(run())
        loop.close()

        assert received_a[0]["overall_progress"] == pytest.approx(68.0)
        assert received_b[0]["overall_progress"] == pytest.approx(68.0)


# ===========================================================================
# 4. WebSocket handler unit tests (stub WebSocket)
# ===========================================================================


class MockWebSocket:
    """Minimal stub that mimics FastAPI's WebSocket interface."""

    def __init__(self):
        self.sent: list[str] = []
        self.closed: bool = False
        self.close_code: int | None = None
        self._accepted = False

    async def accept(self):
        self._accepted = True

    async def send_text(self, text: str):
        if self.closed:
            raise RuntimeError("WebSocket already closed")
        self.sent.append(text)

    async def close(self, code: int = 1000):
        self.closed = True
        self.close_code = code


class TestWsJobProgressHandler:
    def test_404_for_unknown_job(self, db_session):
        """Handler sends error + closes with 4004 if job does not exist."""
        from meetbot.web.ws import ws_job_progress
        import meetbot.db.database as dbmod
        from meetbot.db.database import reset_engine

        sess, engine = db_session
        reset_engine()
        dbmod._engine = engine
        dbmod._SessionLocal = sessionmaker(bind=engine)

        ws = MockWebSocket()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                ws_job_progress(ws, "00000000-0000-0000-0000-000000000000")
            )
        finally:
            loop.close()

        assert ws.close_code == 4004
        err = json.loads(ws.sent[0])
        assert "error" in err

    def test_completed_job_delivers_snapshot_and_closes(self, db_session):
        """Handler sends one snapshot event + closes with 4005 for a finished job."""
        from meetbot.web.ws import ws_job_progress
        import meetbot.db.database as dbmod
        from meetbot.db.database import reset_engine

        sess, engine = db_session
        reset_engine()
        dbmod._engine = engine
        dbmod._SessionLocal = sessionmaker(bind=engine)

        job_id = _seed_job(sess, engine)

        # Mark the job as COMPLETED
        db = sessionmaker(bind=engine)()
        update_job_status(
            db, job_id, JobStatus.COMPLETED,
            progress=100.0, stage_progress=100.0,
            progress_message="Done"
        )
        db.close()

        ws = MockWebSocket()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ws_job_progress(ws, job_id))
        finally:
            loop.close()

        assert ws.close_code == 4005
        snap = json.loads(ws.sent[0])
        assert snap["status"] == "completed"
        assert snap["overall_progress"] == pytest.approx(100.0)

    def test_active_job_receives_progress_events(self, db_session):
        """
        Handler subscribes and forwards live events until a 'completed' event.
        """
        from meetbot.web.ws import ws_job_progress
        import meetbot.db.database as dbmod
        from meetbot.db.database import reset_engine

        sess, engine = db_session
        reset_engine()
        dbmod._engine = engine
        dbmod._SessionLocal = sessionmaker(bind=engine)

        job_id = _seed_job(sess, engine)
        # Job starts PENDING (active)

        pm = __import__(
            "meetbot.workers.progress", fromlist=["progress_manager"]
        ).progress_manager

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pm.set_loop(loop)

        ws = MockWebSocket()

        async def run():
            # Run ws handler and simultaneously fire events from a thread
            handler_task = asyncio.create_task(ws_job_progress(ws, job_id))

            # Wait a moment for the handler to subscribe
            await asyncio.sleep(0.1)

            # Simulate worker thread updates
            def fire_events():
                time.sleep(0.05)
                pm.update(job_id, "transcribing", 20.0, "chunk 1", stage_progress=50)
                time.sleep(0.05)
                pm.update(
                    job_id, "completed", 100.0, "done",
                    stage_progress=100.0, status="completed"
                )

            t = threading.Thread(target=fire_events, daemon=True)
            t.start()
            t.join(timeout=3)

            # Wait for handler to finish
            await asyncio.wait_for(handler_task, timeout=3.0)

        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

        # Initial snapshot + at least the two events
        events = [json.loads(s) for s in ws.sent if not json.loads(s).get("ping")]
        assert any(e.get("stage") == "transcribing" for e in events), (
            f"transcribing event missing; got: {events}"
        )
        assert any(e.get("status") == "completed" for e in events), (
            f"completed event missing; got: {events}"
        )
        assert ws.close_code == 1000
