# MeetBot File Map (Repository Learning Guide)

This guide maps key top-level files and **every Python file under `meetbot/`**.

## Top-level directories
- `meetbot/`: application source (UI, services, adapters, workers).
- `tests/`: unit/integration-style tests for auth, db, workers, progress.
- `scripts/`: utility scripts (verification, now learner helper script).
- `docker-compose.yml`, `Dockerfile`: containerized run options.
- `requirements.txt`: Python dependency lock list.

---

## `meetbot/` file-by-file

### Core
- `meetbot/config.py`
  - Central env-driven settings object (`Settings`) and helper methods for token/device/path resolution.
  - **Why**: one validated source of truth for runtime behavior.
  - **I/O**: reads environment, returns typed config values.

- `meetbot/app.py`
  - `MeetBotPipeline` orchestrates run/build_index/answer_query from CLI or programmatic usage.
  - **Why**: high-level composition wrapper over service layer.
  - **I/O**: input audio/question, output dicts with artifacts and answers.

- `meetbot/cli.py`
  - Command handlers (`cmd_transcribe`, `cmd_index`, `cmd_query`, `cmd_serve`, etc.).
  - **Why**: scriptable interface for local/dev workflows.
  - **I/O**: argv -> service calls -> terminal output.

- `meetbot/logging_conf.py`
  - `setup_logging(...)` configures structured logging to console/file.

### Adapters
- `meetbot/adapters/diarization.py`
  - `LocalPyannoteAdapter` wraps pyannote inference; `get_diarization_adapter()` factory.
  - **Inputs**: wav/audio path + speaker bounds.
  - **Outputs**: normalized diarization segment dicts.

- `meetbot/adapters/embeddings.py`
  - `BaseEmbedding`, `HuggingFaceEmbedding` abstractions around embedding generation.
  - **Why**: decouple index/query services from embedding backend details.

- `meetbot/adapters/llm/base.py`
  - `BaseLLM` abstract contract for generation and (optionally) streaming.

- `meetbot/adapters/llm/hf_api.py`
  - Provider resolution helpers and `HFAPILLMAdapter` for HF Inference API calls.
  - Handles provider mismatch fallbacks.

- `meetbot/adapters/llm/local_llm.py`
  - `LocalLLMManager`, `LocalLLMAdapter`, lifecycle helpers (`get_local_llm`, `cleanup_local_llm`).
  - **Why**: singleton-ish loading and safe reuse of local quantized model.
  - **Inputs**: prompt + generation params.
  - **Outputs**: generated answer tokens/text.

- `meetbot/adapters/transcribers/base.py`
  - `BaseTranscriber` interface used by service layer.

- `meetbot/adapters/transcribers/factory.py`
  - `get_transcriber`, `get_transcriber_from_cli_arg` choose local or HF backend.

- `meetbot/adapters/transcribers/huggingface.py`
  - `HuggingFaceTranscriber` wraps remote ASR inference endpoint.

- `meetbot/adapters/transcribers/local_whisper.py`
  - `LocalWhisperTranscriber` wraps local Whisper model execution.

- `meetbot/adapters/__init__.py`, `meetbot/adapters/llm/__init__.py`, `meetbot/adapters/transcribers/__init__.py`
  - package exports.

### Services
- `meetbot/services/transcriber.py`
  - `TranscriberService` orchestrates audio prep/chunking/caching and calls transcriber adapter.
  - **RAG pipeline role**: creates initial transcript text segments.

- `meetbot/services/diarizer.py`
  - `DiarizationService` wraps diarization adapter + caching + progress hooks.

- `meetbot/services/aligner.py`
  - `AlignerService` merges transcript segments with speaker time regions.
  - key funcs: `_align`, `format_result_as_json`.

- `meetbot/services/formatters.py`
  - utility formatter to shape final JSON output payload.

- `meetbot/services/prepare_docs.py`
  - `PrepareDocsService` chunkifies aligned transcript into retrieval-friendly document units.

- `meetbot/services/indexer.py`
  - `IndexerService` embeds docs and persists Chroma vector index.
  - **RAG orchestration hotspot**: change chunk insertion/index options here.

- `meetbot/services/query_service.py`
  - `QueryService` = retrieval + prompt assembly + generation with local/HF LLM.
  - **Main RAG behavior switch points**:
    - retrieval `k`, filters, and source shaping,
    - prompt template construction,
    - backend mode selection.

- `meetbot/services/__init__.py`
  - package marker.

### Workers
- `meetbot/workers/queue.py`
  - `JobQueue`: enqueue/dequeue loop and dispatch policy.
  - **Why**: isolates long-running tasks from request thread.

- `meetbot/workers/pipeline_worker.py`
  - `run_pipeline(job_id)` full stage executor with DB/progress updates.
  - **Inputs**: job ID from queue.
  - **Outputs**: result JSON, segment rows, Chroma index, final status.

- `meetbot/workers/reindex_worker.py`
  - `run_reindex(job_id)` rebuilds index from edited transcript only.

- `meetbot/workers/progress.py`
  - `JobProgress`, `ProgressManager` in-memory pub/sub state for UI updates.

- `meetbot/workers/cancel.py`
  - `CancelRegistry` + `JobCancelledError` stage-boundary cancellation model.

- `meetbot/workers/__init__.py`
  - package marker.

### Utilities
- `meetbot/utils/audio.py`
  - `convert_to_wav(...)` helper using ffmpeg command flow.

- `meetbot/utils/audio_chunker.py`
  - audio probing, silence detection, chunk window generation, chunk stitching.
  - **Why**: makes large-file transcription robust on constrained hardware.
  - notable functions: `chunk_audio_for_transcription`, `stitch_chunk_results_to_json`.

- `meetbot/utils/cache.py`
  - path/key-based JSON cache save/load helpers for expensive model stages.

- `meetbot/utils/__init__.py`
  - package marker.

### Web layer
- `meetbot/web/main.py`
  - app bootstrap (`create_app`, `start_server`) + page/API/WS registration.

- `meetbot/web/api.py`
  - REST endpoints: job status, query, download, reindex, cancel/delete and chat helpers.

- `meetbot/web/ws.py`
  - WebSocket channel for job progress stream.

- `meetbot/web/ws_chat.py`
  - WebSocket channel for streaming chat responses and chat persistence integration.

- `meetbot/web/auth.py`
  - password hashing/verification and session helpers (`login_user`, `logout_user`).

- `meetbot/web/pages/login.py`
  - login/register page rendering.

- `meetbot/web/pages/dashboard.py`
  - job list cards, refresh, delete dialogs.

- `meetbot/web/pages/upload.py`
  - upload UI, input validation, job creation trigger.

- `meetbot/web/pages/job_detail.py`
  - transcript display/edit surface and reindex entrypoint.

- `meetbot/web/pages/query.py`
  - query/chat UI orchestration and source evidence rendering.

- `meetbot/web/pages/__init__.py`
  - package marker.

- `meetbot/web/components/nav.py`
  - shared header/navigation.

- `meetbot/web/components/progress_bar.py`
  - `ProgressDisplay` UI component bound to job progress state.

- `meetbot/web/components/transcript_table.py`
  - editable transcript table + speaker/name operations.

- `meetbot/web/components/audio_player.py`
  - simple audio playback wrapper for job detail context.

- `meetbot/web/components/__init__.py`, `meetbot/web/__init__.py`
  - package markers/exports.

### Package root
- `meetbot/__init__.py`
  - package marker.

---

## Where to change RAG behavior (exact hotspots)
1. `meetbot/services/query_service.py`
   - retrieval settings (`k`, filtering, how sources are selected), prompt creation, llm routing.
2. `meetbot/services/prepare_docs.py`
   - chunk size/boundary strategy before embedding.
3. `meetbot/services/indexer.py`
   - embedding model options and Chroma collection behavior.
4. `meetbot/adapters/llm/local_llm.py` + `meetbot/adapters/llm/hf_api.py`
   - generation parameters and backend-specific invocation behavior.
