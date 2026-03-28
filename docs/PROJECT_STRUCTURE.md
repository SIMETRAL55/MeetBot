# MeetBot — Project Structure

## Folder layout

```
MeetBot/
├── meetbot/                              # Python backend package
│   ├── __init__.py                       # Package init; exports version
│   ├── cli.py                            # CLI entry point (transcribe, index, query, serve, create-user)
│   ├── config.py                         # Pydantic Settings — all env vars, defaults, validators
│   ├── adapters/
│   │   ├── transcribers/
│   │   │   ├── base.py                   # BaseTranscriber ABC
│   │   │   ├── factory.py                # Select backend from settings.TRANSCRIPTION_BACKEND
│   │   │   ├── local.py                  # OpenAI Whisper (original, float32)
│   │   │   ├── faster.py                 # faster-whisper (CTranslate2, class-level singleton + cleanup())
│   │   │   └── huggingface.py            # HuggingFace Inference API transcriber
│   │   ├── diarization.py                # Pyannote 3.2 speaker diarization (class-level singleton)
│   │   ├── embeddings.py                 # Sentence-transformer embedding (singleton keyed by model+device)
│   │   └── llm/
│   │       ├── base.py                   # BaseLLM ABC
│   │       ├── hf_api.py                 # HuggingFace Inference API LLM
│   │       ├── awq_llm.py                # Local AWQ quantized model (Qwen2.5)
│   │       └── openai_adapter.py         # pageindex_env() context manager (injects env vars for OpenAI SDK)
│   ├── services/
│   │   ├── transcriber.py                # Transcription service (calls adapter, caches result)
│   │   ├── diarizer.py                   # Diarization service (calls adapter, caches result)
│   │   ├── aligner.py                    # Merge Whisper segments with Pyannote speaker labels
│   │   ├── query_service.py              # RAG orchestration: routes on retrieval_method (vector|pageindex)
│   │   └── rag/
│   │       ├── chunker.py                # Transcript chunking for vector indexing
│   │       ├── indexer.py                # Multi-level ChromaDB indexer (embed + store)
│   │       ├── retriever.py              # ChromaDB ANN retriever
│   │       ├── reranker.py               # Cross-encoder reranker (filters/reorders candidates)
│   │       ├── retrieval_strategy.py     # RetrievalMethod enum; RetrievalResult dataclass
│   │       ├── prompts.py                # ALL LLM prompt templates (topic seg, subtopic, summary, root, search)
│   │       ├── indexer_pageindex.py      # PageIndexAdapter: build 3-level JSON tree; load from disk
│   │       └── retriever_pageindex.py    # PageIndexRetriever: agentic LLM tree search (up to 4 iters)
│   ├── workers/
│   │   ├── queue.py                      # In-process job queue (thread-based)
│   │   ├── cancel.py                     # Cancellation token registry
│   │   ├── progress.py                   # Progress reporting helpers
│   │   ├── pipeline_worker.py            # Main pipeline: Stages 1-6 + optional PageIndex (Stage 4b)
│   │   └── reindex_worker.py             # Re-index after transcript edits (vector + optional PageIndex rebuild)
│   ├── web/
│   │   ├── main.py                       # App setup: NiceGUI init, CORS, ALL route registration, lifespan
│   │   ├── api.py                        # REST API endpoint functions (registered in main.py)
│   │   ├── ws_chat.py                    # WebSocket streaming chat handler (retrieval_method routing)
│   │   ├── ws.py                         # WebSocket job progress handler
│   │   ├── auth.py                       # User login/register helpers (bcrypt, JWT)
│   │   ├── auth_middleware.py            # JWT Bearer middleware + get_optional_user() fallback
│   │   ├── pages/                        # NiceGUI server-rendered page views
│   │   └── components/                   # Reusable NiceGUI UI components
│   └── db/
│       ├── database.py                   # SQLAlchemy session factory + engine setup
│       ├── models.py                     # ORM models: User, Job, Segment, ChatSession, ChatMessage
│       ├── crud.py                       # CRUD helpers (get_job, create_message, update_segment, etc.)
│       └── migrations/
│           ├── env.py                    # Alembic env (render_as_batch=True for SQLite)
│           └── versions/
│               ├── 001_initial.py        # Initial schema (User, Job, Segment, ChatSession, ChatMessage)
│               └── 002_add_pageindex_columns.py  # Adds pageindex_path, pageindex_status to Job
├── frontend/
│   ├── package.json                      # Node dependencies (Next.js 16, React 19, Tailwind 4, shadcn/ui)
│   ├── next.config.js                    # Next.js config (API proxy to :8080)
│   ├── tailwind.config.js                # Tailwind CSS config
│   └── src/
│       ├── app/
│       │   ├── layout.tsx                # Root layout with providers
│       │   ├── page.tsx                  # Home / redirect
│       │   ├── job/[id]/page.tsx         # Job detail: transcript, segments, Build PageIndex panel
│       │   └── chat/[id]/page.tsx        # RAG chat: streaming messages, vector/pageindex toggle
│       ├── lib/
│       │   ├── api.ts                    # API client: all fetch calls, buildPageIndex(), auth helpers
│       │   └── hooks/
│       │       └── useChatWS.ts          # WebSocket hook: sendMessage() with retrieval_method param
│       └── types/
│           └── index.ts                  # TypeScript types: Job, Segment, ChatSource, ChatMessage
├── tests/
│   ├── test_auth.py                      # Auth endpoints: register, login, JWT validation
│   ├── test_db.py                        # DB CRUD: create/read/update/delete across all models
│   ├── test_query_rag.py                 # RAG query pipeline (1 pre-existing failure: TestCountDocuments)
│   ├── test_rag_chunking.py              # Transcript chunking logic
│   ├── test_indexer_batching.py          # ChromaDB indexer batch behavior
│   ├── test_indexer_multilevel.py        # Multi-level indexing correctness
│   ├── test_pageindex_integration.py     # 49 tests: env injection, adapter build/load, retriever search
│   ├── test_streaming_pipeline.py        # Chat streaming pipeline (WebSocket persistence)
│   ├── test_stream_persistence.py        # Chat message persistence across reconnects
│   ├── test_worker.py                    # Pipeline worker stage sequencing
│   └── test_segment_idempotency.py       # Segment edit + re-index idempotency
├── vendor/
│   └── pageindex_repo/                   # VectifyAI PageIndex library (reference only; NOT used in code)
├── db/
│   ├── meetbot.db                        # SQLite database file
│   └── pageindex/                        # PageIndex JSON output: {job_id}_tree.json + {job_id}_content_map.json
├── .cache_hf/                            # HuggingFace transcription/diarization cache (audio-hash keyed)
├── CLAUDE.md                             # Claude Code developer guidance (this project)
├── BLUEPRINT.md                          # Project blueprint: pitch, architecture, roadmap
├── ISSUES.md                             # Known issues, bugs, missing features audit
├── alembic.ini                           # Alembic database migration config
└── requirements.txt                      # Pinned Python dependencies
```

## Getting started

```bash
# 1. Clone and set up venv
git clone <repo>
cd MeetBot
python3.12 -m venv ../venv
source ../venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install system deps
sudo apt install ffmpeg

# 4. Create .env from example and fill in secrets
cp .env.example .env
# Required: HF_API_TOKEN, WEB_SECRET_KEY
# Optional for PageIndex: PAGEINDEX_ENABLED=true, PAGEINDEX_LLM_API_KEY

# 5. Run DB migrations
alembic upgrade head

# 6. Create initial admin user
python -m meetbot.cli create-user admin --password yourpassword --admin

# 7. Start backend (port 8080)
python -m meetbot.cli serve

# 8. Start frontend (new terminal, port 3000)
cd frontend
npm install
npm run dev

# 9. Open http://localhost:3000
```

## Key files and what they do

### `meetbot/web/main.py` + `meetbot/web/api.py` — Dual route registration

NiceGUI initializes the FastAPI app internally. All REST endpoints must be registered AFTER NiceGUI setup using `app.add_api_route(path, function, methods=[...])` in `main.py`. The handler functions live in `api.py`. This two-file split means **adding a new endpoint requires touching both files** — define the function in `api.py`, register the route in `main.py`. Missing the `main.py` step causes the endpoint to silently not exist (no 404, just not mounted).

### `meetbot/workers/pipeline_worker.py` — Main pipeline

Runs as a background thread. Executes 6 sequential stages: transcription (Whisper), GPU cleanup, diarization (Pyannote), GPU cleanup, alignment, and ChromaDB indexing. Each stage updates `job.status` and `job.progress_pct` in the DB. Stage 4b (PageIndex) is optional, non-fatal, and runs only when `PAGEINDEX_ENABLED=true` and `PAGEINDEX_AUTO_INDEX=true`. Any exception in Stage 4b sets `pageindex_status="failed"` and logs the error — it never propagates to fail the main job.

### `meetbot/services/query_service.py` — RAG orchestration

Routes incoming queries on `retrieval_method`. Vector path: ChromaDB ANN recall → MMR rerank → context assembly → LLM generation → source filtering by answer-embedding similarity. PageIndex path: load JSON tree from disk → build tree summary → agentic LLM tree search (up to 4 iterations with sufficiency check) → collect segments → LLM generation → return with `node_title`/`node_id` citations.

### `meetbot/web/ws_chat.py` — Streaming chat WebSocket

Accepts WebSocket connections at `/ws/chat/{job_id}`. Parses `retrieval_method` from the JSON payload. Runs the query in a thread-pool executor (max_workers=1 to serialize queries per session). Uses a persistence-first design: terminal events (complete/error) are saved to DB **before** sending to the WebSocket client, ensuring chat history is preserved even if the client disconnects mid-stream.

### `meetbot/services/rag/indexer_pageindex.py` — PageIndex tree builder

`PageIndexAdapter.build_index()` runs 5 steps: (A) topic segmentation via LLM, (B) subtopic segmentation per topic via LLM, (C) per-node summarization via LLM, (D) root synthesis via LLM, (E) deterministic content_map build. Saves `{job_id}_tree.json` and `{job_id}_content_map.json` to `PAGEINDEX_OUTPUT_DIR`. All LLM calls are synchronous (`_llm_call()`), routed through `pageindex_env()` which temporarily sets `OPENAI_BASE_URL`/`OPENAI_API_KEY`. Methods are marked `async` but don't use `await` internally — the async decoration exists for future parallelism.

### `meetbot/services/rag/retriever_pageindex.py` — PageIndex retriever

`PageIndexRetriever.retrieve()` implements an agentic tree search: it presents a tree summary to the LLM, asks it to select relevant `node_id`s, checks sufficiency, and iterates up to 4 times if more context is needed. Maps `node_id`s to segment ranges via the content_map, then returns `RetrievalResult` objects with `node_title`, `node_id`, and raw segment text.

### `meetbot/services/rag/prompts.py` — All LLM prompt templates

Single source of truth for every LLM prompt in the PageIndex system: `TOPIC_SEGMENTATION_PROMPT`, `TOPIC_SEGMENTATION_RETRY_PROMPT`, `SUBTOPIC_SEGMENTATION_PROMPT`, `NODE_SUMMARY_PROMPT`, `ROOT_SYNTHESIS_PROMPT`, and the tree-search prompts. Edit prompts here, not inline in adapter/retriever code.

### `meetbot/config.py` — Pydantic Settings

Loads all environment variables with defaults and validation. Key method: `get_pageindex_base_url()` which resolves the LLM backend URL based on `PAGEINDEX_LLM_BACKEND` (openrouter/ollama/openai/custom). `get_pageindex_output_dir()` returns a `Path` object for the JSON output directory.

### `meetbot/adapters/llm/openai_adapter.py` — `pageindex_env()`

A `contextlib.contextmanager` that temporarily sets `OPENAI_BASE_URL` and `OPENAI_API_KEY` in `os.environ`, yields, then restores the original values. This allows the OpenAI SDK to route calls to OpenRouter or Ollama without polluting global config. **Not thread-safe**: concurrent PageIndex jobs could see each other's env vars during the window. Consider adding a threading lock or switching to per-client instantiation.

### `meetbot/db/models.py` — ORM models

Five models: `User` (auth), `Job` (pipeline state + `pageindex_path` + `pageindex_status`), `Segment` (aligned transcript segments with `segment_index` for PageIndex mapping), `ChatSession` (one per job), `ChatMessage` (individual turns with `is_streaming` flag). `JobStatus` is a Python `Enum` stored as a string column.
