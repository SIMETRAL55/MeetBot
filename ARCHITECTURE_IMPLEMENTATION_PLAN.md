# MeetBot — Architecture Implementation Plan

> **Source of truth** for the production web application implementation.
> Update checkboxes as tasks are completed. This file enables continuation from any new session.

---

## Architecture Summary

| Dimension | Decision |
|---|---|
| **UI Framework** | NiceGUI (Python-only, Quasar components, WebSocket-native) |
| **Backend** | FastAPI (via NiceGUI) |
| **Task Queue** | `asyncio.Queue` + `threading.Thread` (single GPU worker) |
| **Database** | SQLite + SQLAlchemy (users, jobs, segments) |
| **Vector DB** | ChromaDB (existing, unchanged) |
| **Auth** | Session cookies + bcrypt |
| **Deployment** | Docker + nvidia-container-toolkit |
| **Models** | All local, sequential GPU access |
| **Target Users** | 5–25 non-technical users, internal org, air-gapped |
| **Languages** | Japanese primary, English secondary |

---

## Folder Structure

```
meetbot/
├── __init__.py                    # Existing
├── config.py                      # Existing (extend with web settings)
├── logging_conf.py                # Existing
├── app.py                         # Existing (pipeline orchestration)
├── cli.py                         # Existing (extend with manage commands)
├── adapters/                      # Existing (UNCHANGED)
├── services/                      # Existing (UNCHANGED)
├── utils/                         # Existing (UNCHANGED)
├── db/                            # NEW — Database layer
│   ├── __init__.py
│   ├── models.py                  # SQLAlchemy models (User, Job, Segment)
│   ├── database.py                # Engine + session factory
│   └── crud.py                    # CRUD operations
├── workers/                       # NEW — Background processing
│   ├── __init__.py
│   ├── queue.py                   # Job queue manager
│   ├── pipeline_worker.py         # Pipeline execution in thread
│   └── progress.py                # Progress tracking + WebSocket push
└── web/                           # NEW — Web application
    ├── __init__.py
    ├── main.py                    # NiceGUI app entry point
    ├── auth.py                    # Login/session management
    ├── pages/
    │   ├── __init__.py
    │   ├── login.py               # Login page
    │   ├── dashboard.py           # Job list / home
    │   ├── upload.py              # Audio upload page
    │   ├── job_detail.py          # Transcript viewer + editor
    │   └── query.py               # RAG query interface
    └── components/
        ├── __init__.py
        ├── transcript_table.py    # Editable transcript component
        ├── audio_player.py        # Audio player with seek
        ├── progress_bar.py        # Real-time progress
        └── nav.py                 # Navigation header
```

---

## Week 1 — Foundation

### Database Layer
- [x] Create `meetbot/db/__init__.py`
- [x] Create `meetbot/db/database.py` — SQLAlchemy engine, session factory, `init_db()`
- [x] Create `meetbot/db/models.py` — `User`, `Job`, `Segment` models
- [x] Create `meetbot/db/crud.py` — CRUD operations for all models

### Authentication
- [x] Create `meetbot/web/auth.py` — bcrypt password hashing, session management
- [x] Extend `meetbot/cli.py` — add `manage create-user` subcommand

### Job Queue & Worker
- [x] Create `meetbot/workers/__init__.py`
- [x] Create `meetbot/workers/progress.py` — `ProgressManager` with WebSocket push
- [x] Create `meetbot/workers/queue.py` — `JobQueue` with asyncio + threading
- [x] Create `meetbot/workers/pipeline_worker.py` — pipeline execution with progress callbacks

### NiceGUI Skeleton
- [x] Create `meetbot/web/__init__.py`
- [x] Create `meetbot/web/main.py` — NiceGUI app entry, middleware, startup/shutdown
- [x] Create `meetbot/web/components/__init__.py`
- [x] Create `meetbot/web/components/nav.py` — navigation header
- [x] Create `meetbot/web/pages/__init__.py`
- [x] Create `meetbot/web/pages/login.py` — login page

### Configuration
- [x] Extend `meetbot/config.py` — add web server settings (host, port, secret key)

---

## Week 2 — Core UI

### Dashboard & Upload
- [x] Create `meetbot/web/pages/dashboard.py` — job list with status badges, real-time updates
- [x] Create `meetbot/web/pages/upload.py` — file upload with progress

### Job Detail & Transcript Viewer
- [x] Create `meetbot/web/components/audio_player.py` — audio player with seek-to-timestamp
- [x] Create `meetbot/web/components/transcript_table.py` — editable transcript table
- [x] Create `meetbot/web/components/progress_bar.py` — real-time pipeline progress
- [x] Create `meetbot/web/pages/job_detail.py` — transcript viewer + editor + audio player

### Query Interface
- [x] Create `meetbot/web/pages/query.py` — RAG query with answer + sources display

---

## Week 3 — Polish & Deploy

### Error Handling & Validation
- [x] Add input validation on upload (file type, size limits)
- [x] Failed job display with error messages and retry
- [x] Graceful error handling in all pages

### Docker & Deployment
- [x] Create `Dockerfile`
- [x] Create `docker-compose.yml`
- [x] Create `.dockerignore`

### Testing & Documentation
- [x] Create `tests/test_db.py` — database CRUD tests
- [x] Create `tests/test_auth.py` — authentication tests
- [x] Create `tests/test_worker.py` — job queue tests
- [x] Update `requirements.txt` with new dependencies

---

---

## Week 4 — Real-time Progress Streaming

### Database
- [x] Add `stage_progress` (Float) column to `Job` model — within-stage 0–100 %
- [x] Add `logs` (Text/JSON array) column to `Job` model — last 50 log lines
- [x] Add `_run_migrations()` in `meetbot/db/database.py` — idempotent ALTER TABLE via PRAGMA table_info
- [x] Extend `update_job_status()` in `meetbot/db/crud.py` — `stage_progress` + `log_line` params, JSON log cap

### Worker / Progress Bus
- [x] Rewrite `meetbot/workers/progress.py` — asyncio Queue pub/sub, `subscribe()`/`unsubscribe()`, loop bridge via `call_soon_threadsafe`
- [x] Update `meetbot/workers/queue.py` — call `progress_manager.set_loop(loop)` on worker startup
- [x] Update `meetbot/workers/pipeline_worker.py` — `_stage_update()` helper, per-stage `stage_progress` calculation (transcription 0–40 %, diarization 40–65 %, alignment 65–75 %, indexing 75–95 %)

### WebSocket Endpoint
- [x] Create `meetbot/web/ws.py` — FastAPI WebSocket handler at `/ws/jobs/{job_id}`
  - Sends snapshot on connect; closes 4004 (not found) / 4005 (already finished)
  - Relays ProgressManager events until terminal state; 30 s heartbeat; clean close 1000
  - Event shape: `{job_id, stage, stage_progress, overall_progress, logs, status, message}`
- [x] Register route in `meetbot/web/main.py` via `app.add_api_websocket_route()`

### UI Component
- [x] Rewrite `meetbot/web/components/progress_bar.py` — dual progress bars (overall + stage), stage chip badges, scrollable log panel, 1.5 s poll interval

### Tests
- [x] Create `tests/test_progress_realtime.py` — 15 tests covering DB migration, CRUD, ProgressManager pub/sub, WebSocket handler (all passing)

---

## Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-01 | Initial implementation | Architecture finalized |
| 2026-03-02 | Real-time progress streaming (Week 4) | Per-stage WebSocket progress reporting |

---

## Notes

- Existing `adapters/`, `services/`, `utils/` are **not modified** — new layers wrap around them
- All processing is local (air-gapped) — no external API calls
- Single GPU worker thread — jobs processed sequentially
- Models loaded lazily, released after use to manage VRAM
