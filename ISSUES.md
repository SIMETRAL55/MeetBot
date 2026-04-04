# MeetBot — Issues

Audit of the codebase as of `feat/PageIndex` branch (2026-03-25).
Format: **[SEVERITY]** `file:line` — description / suggested fix.

---

## Critical (will break in production or corrupt state)

**[FIXED]** `CLAUDE.md` (directory structure section)
`meetbot/services/rag/transcript_to_md.py` was listed but had been deleted. Removed from CLAUDE.md directory structure and test table.

**[FIXED]** `CLAUDE.md` (Testing section)
Stale `test_transcript_to_md.py` reference and wrong test count. Updated test file table to reflect current 256-test suite.

---

## Bugs (wrong or fragile behavior)

**[FIXED]** `meetbot/services/rag/indexer_pageindex.py:482`
Content_map path derivation already uses `Path` stem manipulation (not fragile string replace). Verified correct: `tree_path.parent / (tree_path.stem.replace("_tree", "_content_map") + ".json")`.

**[FIXED]** `meetbot/adapters/llm/openai_adapter.py` — `pageindex_env()`
Already thread-safe: `openai.OpenAI()` is called with explicit `base_url` and `api_key` constructor args — no `os.environ` mutation for remote backends. Ollama lifecycle protected by `_ollama_lock`.

**[FIXED]** `meetbot/services/rag/indexer_pageindex.py` — `async def _segment_topics / _segment_subtopics / _summarize_node / _synthesize_root`
`_llm_call()` converted to `async def` using `asyncio.to_thread()`. Subtopic segmentation and node summarization now parallelized via `asyncio.gather()`. Exceptions from individual gather tasks fall back to `_fallback_*` methods.

---

## Missing features (referenced or needed but not implemented)

**[FIXED]** `meetbot/web/api.py` — Upload endpoint
File-size validation already implemented: Content-Length header checked pre-write (returns 413 if over limit), and post-write size check also present. Enforces `settings.MAX_UPLOAD_SIZE_MB` (default 500 MB).

**[FIXED]** `meetbot/web/api.py` — All endpoints
`slowapi` rate limiting added. Upload endpoint decorated with `@limiter.limit(settings.RATE_LIMIT_UPLOAD)`. Limiter mounted in `main.py` via `app.state.limiter` + `RateLimitExceeded` exception handler. New config vars: `RATE_LIMIT_UPLOAD` (default `"10/hour"`), `RATE_LIMIT_CHAT` (default `"60/minute"`).

**[FIXED]** `meetbot/services/rag/retriever_pageindex.py`
`cancel_checker: Optional[callable] = None` parameter already present in `search()` signature and called at the start of each iteration loop.

**[MISSING]** `meetbot/web/auth.py` / `meetbot/web/pages/`
No CSRF protection on NiceGUI server-rendered pages. Session cookie is the only protection.
Suggested fix: Add CSRF token validation for state-changing NiceGUI actions (upload, delete).

**[FIXED]** `.env.example`
`.env.example` created at `MeetBot/.env.example` with all variables from `config.py`, grouped by feature (core, auth, transcription, LLM, pageindex, DB, rate limiting), with placeholder values and comments.

---

## Code quality issues

**[FIXED]** `meetbot/web/main.py` + `meetbot/web/api.py` — Dual registration pattern
Comment block added at top of `api.py`: `# NOTE: Every function defined here MUST also be registered in main.py via app.add_api_route()`. Also documented in CLAUDE.md under "Adding New API Endpoints".

**[FIXED]** `meetbot/config.py` — JWT secret default
`model_post_init()` already generates a random key if `WEB_SECRET_KEY` is empty **and** emits a `WARNING` log: `"WEB_SECRET_KEY not set — generated a random key. Tokens will be invalidated on restart."` Verified in backend startup logs.

**[FIXED]** `meetbot/services/rag/prompts.py`
Schema comment blocks added to all 7 prompt constants: expected JSON schema and which module/method parses each response. Note: used `{{` / `}}` escaping so comments survive `.format()` calls.

**[QUALITY]** Various modules — Inconsistent logging
Some modules use `logger.debug`, some `logger.info`, some `logger.warning` for similar events. No structured logging format (e.g., JSON logs for production use).
Suggested fix: Define a logging config in `meetbot/__init__.py` or `cli.py` with consistent format; add `structlog` for production structured logging.

---

## Security issues

**[FIXED]** `meetbot/adapters/llm/openai_adapter.py` — `pageindex_env()`
Credentials already passed directly to `openai.OpenAI(base_url=..., api_key=...)` — `os.environ` is never mutated for remote backends.

**[FIXED]** `meetbot/web/auth_middleware.py`
Audited: `exp` claim validated in `verify_token()`. Expired tokens rejected with `logger.debug("JWT rejected: token expired for sub=%s", ...)`. `get_current_user()` validates user existence via DB lookup. Full test coverage added in `tests/test_auth_middleware.py`.

**[FIXED]** `meetbot/web/api.py` — `/api/jobs/upload`
`python-magic` MIME-type validation added after file write. Rejects non-audio/video files with HTTP 415. `libmagic1` system dep documented in setup. `python-magic==0.4.27` added to `requirements.txt`.

---

## Performance issues

**[FIXED]** `meetbot/services/rag/indexer_pageindex.py` — `build_index()`
`_llm_call()` uses `asyncio.to_thread()`. Subtopic segmentation and leaf-node summarization now run concurrently via `asyncio.gather(..., return_exceptions=True)`. Exceptions fall back to `_fallback_*` methods. Expected build time reduction: 3-5× for typical 5-topic transcripts.

**[FIXED]** `meetbot/services/rag/indexer.py` — ChromaDB reindex (skip-if-unchanged)
`Job.segments_hash` column added (migration `003_add_segments_hash`). `reindex_worker.py` computes SHA256 hash of segments before rebuilding; skips `build_multilevel_index_atomic()` entirely if hash matches stored value. Hash written back to DB after successful rebuild. Cuts reindex time to ~0 ms when transcript hasn't changed.

**[FIXED]** `meetbot/workers/pipeline_worker.py` — Single worker thread
`PIPELINE_WORKERS: int = Field(default=1)` added to `config.py`. `JobQueue.start()` spawns `settings.PIPELINE_WORKERS` worker threads. `stop()` joins all threads. Default remains 1 (safe for single-GPU setups).

---

## Missing tests

**[FIXED]** `meetbot/web/ws_chat.py`
`tests/test_ws_chat.py` added: 6 tests covering job-not-found (4004), job-not-completed (4009), missing/empty question (4010), invalid JSON (4010), valid query saves user+assistant messages with status="completed", and `retrieval_method="pageindex"` forwarded to `QueryService`.

**[FIXED]** `meetbot/workers/reindex_worker.py`
`tests/test_reindex_worker.py` added: 7 tests covering successful reindex (COMPLETED + hash written), exact hash value verification, skip-if-unchanged (indexer not called), PageIndex adapter called when enabled, PageIndex failure is non-fatal (job still COMPLETED), job-not-found early return, missing result JSON fails job.

**[FIXED]** `meetbot/web/auth_middleware.py`
`tests/test_auth_middleware.py` added: 11 tests covering valid token round-trip, expired token rejection + debug log emitted, future token passes, 2-part and 1-part malformed tokens, invalid signature, wrong secret, `get_current_user` with no header/valid bearer/expired/invalid-sig.

**[FIXED]** `meetbot/workers/pipeline_worker.py` — GPU cleanup stages
`tests/test_worker.py` extended: `TestCleanupGpu` (CUDA available/unavailable/torch-not-installed) and `TestWhisperCleanupAfterTranscription` (cleanup called, errors swallowed). Uses `patch.dict(sys.modules, {"torch": mock_torch})` — no real GPU needed.

**[FIXED]** `tests/test_pageindex_integration.py`
Added `TestBuildIndexSyncWorkerPath` class in `test_pageindex_integration.py` with two tests:
- `test_build_index_via_new_event_loop`: covers the `asyncio.new_event_loop()` + `loop.run_until_complete()` pattern from `pipeline_worker.py` Stage 4b.
- `test_build_index_sync_worker_llm_failure_uses_fallback`: verifies that LLM failures are handled by fallback logic (non-fatal design) and the tree is still produced.

**[FIXED]** `meetbot/web/api.py` — Edge cases
Added `tests/test_api_upload.py` covering: non-audio MIME type (415), oversized upload via Content-Length (413), oversized upload post-write (413), disallowed extension (400), PageIndex build when `PAGEINDEX_ENABLED=false` (409), job not found (404), job not completed (409).
