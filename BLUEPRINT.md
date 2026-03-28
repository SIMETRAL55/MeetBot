# MeetBot — Blueprint

## Elevator pitch

MeetBot turns raw audio recordings into a searchable, queryable knowledge base. Upload a meeting recording; MeetBot transcribes it, identifies speakers, and lets you ask natural-language questions like "What did Alice decide about the Q3 budget?" — getting back a cited answer with timestamps. It supports two retrieval strategies: fast vector similarity search (ChromaDB) and a deeper LLM-powered tree search (PageIndex) that understands the meeting's hierarchical topic structure.

## The problem it solves

> "I am a knowledge worker who attends 3–5 hours of meetings daily. I need to find a specific decision, action item, or quote from a meeting held 3 days ago. Currently I either re-listen to recordings (painful and slow), rely on imperfect human notes, or miss the information entirely. MeetBot lets me ask a question and get a cited answer in seconds."

## Technical architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interfaces                             │
│                                                                     │
│   Next.js Frontend (port 3000)    NiceGUI Pages (port 8080)        │
│   /job/[id]  /chat/[id]           /login  /dashboard  /upload      │
└────────────────────┬───────────────────────────┬────────────────────┘
                     │ REST /api/*                │ NiceGUI WS
                     │ WS /ws/*                   │ Session Auth
                     ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FastAPI + NiceGUI (Uvicorn, port 8080)             │
│                                                                     │
│  api.py (REST endpoints)     ws_chat.py (streaming chat WS)        │
│  ws.py (progress WS)         auth_middleware.py (JWT)              │
└──────────┬──────────────────────────────────────┬───────────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────┐              ┌───────────────────────────────┐
│   Pipeline Worker    │              │      Query Service             │
│  (background thread) │              │                               │
│                      │              │  retrieval_method=vector:     │
│  1. Transcription    │              │    ChromaDB ANN recall        │
│     (Whisper)        │              │    → cross-encoder rerank     │
│  2. GPU cleanup      │              │    → LLM generate             │
│  3. Diarization      │              │                               │
│     (Pyannote)       │              │  retrieval_method=pageindex:  │
│  4. GPU cleanup      │              │    load JSON tree             │
│  5. Alignment        │              │    → LLM tree search          │
│  6. Vector indexing  │              │    → collect segments         │
│     (ChromaDB)       │              │    → LLM generate             │
│  [4b. PageIndex]     │              └───────────────────────────────┘
└──────────┬───────────┘                          │
           │                                      │
           ▼                                      ▼
┌──────────────────────┐              ┌───────────────────────────────┐
│   SQLite / PostgreSQL│              │  ChromaDB (vector store)      │
│   (ORM: SQLAlchemy)  │              │  PageIndex JSON trees         │
│                      │              │  (db/pageindex/*.json)        │
│  User, Job, Segment  │              └───────────────────────────────┘
│  ChatSession, Message│
└──────────────────────┘
```

## Key technical decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Dual retrieval paths | Vector (ChromaDB) + PageIndex (LLM tree search) | Vector is fast; PageIndex understands topic hierarchy for complex multi-hop questions |
| Web framework | NiceGUI (wraps FastAPI) + Next.js frontend | NiceGUI for rapid server-rendered admin pages; Next.js for polished user-facing chat |
| GPU singleton pattern | Class-level `_model` + `cleanup()` classmethod | RTX 3050 (4 GB VRAM) cannot hold Whisper + Pyannote simultaneously; explicit cleanup between stages |
| PageIndex: non-fatal | PageIndex failures set `pageindex_status="failed"` but never fail the main job | Transcription/diarization is the core value; LLM tree building is optional enhancement |
| Async PageIndex in sync workers | `asyncio.new_event_loop()` + `loop.run_until_complete()` | Pipeline workers are sync threads; `asyncio.run()` raises if a running loop exists |
| Dual auth | NiceGUI session (cookie) + JWT Bearer for REST API | NiceGUI pages use built-in storage; REST API needs stateless JWT for Next.js frontend |
| Dual route registration | Both `api.py` (define) + `main.py` (register) | NiceGUI's `app.add_api_route()` must be called after the NiceGUI app is initialized |
| Embedding cache | `(model_name, device)` keyed singleton | Prevents OOM from loading multiple embedding model instances on 4 GB VRAM |

## Feature list

| Feature | Description | Status |
|---------|-------------|--------|
| Audio transcription | Whisper (local/faster/HuggingFace API) with model size config | Done |
| Speaker diarization | Pyannote 3.2 with GPU memory management | Done |
| Speaker alignment | Merge Whisper timestamps with Pyannote speaker labels | Done |
| Vector indexing | ChromaDB multi-level chunking + embedding (sentence-transformers) | Done |
| Vector RAG chat | Streaming WebSocket Q&A with ANN recall + cross-encoder rerank | Done |
| PageIndex build | LLM-driven 3-level topic tree from transcript segments | Done |
| PageIndex RAG chat | LLM agentic tree search (up to 4 iterations) → cited answer | Done |
| Transcript editing | Edit individual segments + re-index in background | Done |
| Chat history | Persistent per-job chat sessions (SQLite) | Done |
| Job management | Upload, cancel, restart, delete, download | Done |
| Auth | User registration, login, JWT, bcrypt passwords | Done |
| Audio streaming | Serve original audio file for in-browser playback | Done |
| PageIndex auto-build | Auto-trigger PageIndex after pipeline completes | Done (config flag) |
| Structured logging | Consistent log format across all modules | Planned |
| Rate limiting | Per-endpoint request throttling | Planned |
| File size validation | Max upload size enforcement | Planned |
| Parallel LLM calls | Concurrent subtopic/summary LLM calls in PageIndex build | Planned |

## Build roadmap

### Phase 0 — Core pipeline (Done)
Transcription → diarization → alignment → ChromaDB indexing. Streaming WebSocket chat with vector RAG. Basic auth. NiceGUI admin pages.

### Phase 1 — PageIndex (Done / In Progress)
LLM-based 3-level transcript tree. Agentic tree search retriever. `/build-pageindex` API. Frontend toggle (vector vs PageIndex). `pageindex_status` tracking on Job model.

### Phase 2 — Hardening (Planned)
- Fix pre-existing `test_query_rag.py` test failure
- Add missing test coverage: `ws_chat.py`, `reindex_worker.py`, `auth_middleware.py`
- Make `pageindex_env()` thread-safe (per-thread env injection or lock)
- Add file size validation on upload
- Add rate limiting (per-user or global)
- Parallelise PageIndex LLM calls (asyncio.gather for subtopics/summaries)

### Phase 3 — Scale (Future)
- PostgreSQL by default (currently optional)
- Celery/Redis task queue (currently in-process thread queue)
- Multi-GPU support
- S3/object storage for audio files
- OpenTelemetry tracing
