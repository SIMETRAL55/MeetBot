# MeetBot

MeetBot turns audio recordings into searchable, queryable transcripts. Upload a meeting, interview, or podcast — MeetBot transcribes it, identifies who spoke when, aligns the two outputs, and builds a vector index so you can ask natural language questions about the content.

Everything can run locally. Cloud APIs are supported as well.

---

## Features

- **Transcription** — Uses OpenAI Whisper (local) or the HuggingFace Inference API
- **Speaker diarization** — Identifies and labels speakers via Pyannote Audio
- **Aligned transcripts** — Merges transcription and diarization into speaker-labelled segments
- **RAG-based Q&A** — Semantic search over ChromaDB with answer generation via a local GGUF model or HuggingFace API
- **Streaming answers** — Token-by-token generation in the web UI and over WebSocket
- **Persistent chat history** — Per-job conversation history stored in SQLite
- **Reindexing** — Rebuild the search index after editing a transcript in the web UI
- **Download outputs** — Export the aligned transcript, raw Whisper output, or diarization JSON
- **Web UI** — NiceGUI-based interface with real-time pipeline progress
- **REST + WebSocket API** — Machine-readable endpoints for all core operations
- **CLI** — Full command-line interface for batch processing

---

## Architecture

```
meetbot/
├── adapters/              LLM and transcription backend adapters
│   ├── llm/               Local (llama.cpp) and HuggingFace LLM clients
│   └── transcribers/      Whisper backends (local GPU, HF Inference API)
├── services/              Core business logic
│   ├── transcriber.py     Audio → raw segments
│   ├── diarizer.py        Audio → speaker intervals
│   ├── aligner.py         Merge transcript + diarization
│   ├── indexer.py         Build ChromaDB vector index
│   ├── prepare_docs.py    Chunk aligned segments for indexing
│   └── query_service.py   RAG retrieval + LLM generation
├── workers/               Background processing
│   ├── queue.py           Single-worker FIFO job queue
│   ├── pipeline_worker.py Full pipeline (transcribe → index)
│   └── reindex_worker.py  Index rebuild only (after transcript edit)
├── web/                   NiceGUI web application
│   ├── pages/             UI pages (dashboard, upload, job detail, chat)
│   ├── components/        Reusable UI components
│   ├── api.py             FastAPI REST endpoints
│   ├── ws.py              Job progress WebSocket
│   └── ws_chat.py         Streaming RAG chat WebSocket
├── db/                    SQLAlchemy models and CRUD
└── config.py              Pydantic Settings (env-var driven)
```

**Pipeline stages (full job):**
```
Upload → Transcription (Whisper) → Diarization (Pyannote) → Alignment → Indexing (Chroma)
```

**Reindex (after transcript edit):**
```
Edit transcript → [Reindex button] → Prepare docs → Build Chroma index
```

---

## Installation (local)

### Prerequisites

- Python 3.12
- `ffmpeg` installed and on `$PATH`
- (Optional) NVIDIA GPU + CUDA 12.x for faster transcription and diarization

### Steps

```bash
# 1. Clone
git clone https://github.com/your-org/meetbot.git
cd meetbot/MeetBot

# 2. Create virtual environment
python -m venv ../venv
source ../venv/bin/activate

# 3. Install PyTorch — choose the right variant for your hardware
#    CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#    CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Install MeetBot in editable mode
pip install -e .

# 6. (Optional) Build llama-cpp-python with GPU support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# 7. Copy and edit the configuration
cp .env.example .env   # or create .env from scratch
```

---

## Configuration

MeetBot is configured entirely through environment variables or a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `HF_API_TOKEN` | — | HuggingFace token; required for Pyannote and gated models |
| `HF_MODEL` | `deepseek-ai/DeepSeek-V3-0324` | Model used for HF Inference API Q&A |
| `HF_PROVIDER` | `auto` | HF inference provider (`auto`, `sambanova`, `novita`, etc.) |
| `TRANSCRIPTION_BACKEND` | `huggingface` | `local` (GPU Whisper) or `huggingface` (API) |
| `USE_LOCAL_LLM` | `false` | Use local GGUF model for Q&A instead of HF API |
| `LOCAL_LLM_MODEL_PATH` | `./models/rakutenai-7b-instruct-gguf` | Path to GGUF model directory |
| `LOCAL_LLM_GPU_LAYERS` | `15` | Model layers offloaded to GPU |
| `EMBEDDING_MODEL` | `./models/sarashina-embedding-v1-1b` | Local path or HF model ID for embeddings |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `OUTPUT_DIR` | `./results` | Where pipeline JSON outputs are stored |
| `VECTOR_DB_PATH` | `./db/chroma` | Root for Chroma persist directories |
| `RAG_TOP_K` | `4` | Number of context chunks retrieved per query |
| `WEB_HOST` | `0.0.0.0` | Web server bind address |
| `WEB_PORT` | `8080` | Web server port |
| `WEB_SECRET_KEY` | (random) | Session signing key — set explicitly in production |

---

## Running

### Web application

```bash
python -m meetbot.cli serve
```

Open `http://localhost:8080` in a browser.  Create an account on first run.

### CLI — transcribe a file

```bash
# Transcribe + diarize + index with HF backend
python -m meetbot.cli run path/to/meeting.mp3

# Force rebuild of an existing index
python -m meetbot.cli run path/to/meeting.mp3 --overwrite-index

# Use local Whisper + local LLM
python -m meetbot.cli run path/to/meeting.mp3 \
    --transcription-backend local \
    --use-local-llm
```

### CLI — query an indexed transcript

```bash
python -m meetbot.cli query db/abcd1234 "What were the main action items?"
```

---

## Docker

### CPU-only

```bash
docker build -t meetbot .

docker run -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/db:/app/db \
  --env-file .env \
  meetbot
```

### GPU (CUDA 12.1)

```bash
docker build --build-arg CUDA_VARIANT=cu121 -t meetbot-gpu .

docker run --gpus all -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/db:/app/db \
  --env-file .env \
  -e EMBEDDING_DEVICE=cuda \
  -e LOCAL_LLM_GPU_LAYERS=20 \
  meetbot-gpu
```

The build-time argument `CUDA_VARIANT` controls which PyTorch wheel is installed.
Accepted values: `cpu` (default), `cu118`, `cu121`.

**Volume mounts:**

| Mount | Contents |
|---|---|
| `/app/data` | Uploaded audio files |
| `/app/models` | Model weights (embedding model, GGUF) |
| `/app/db` | Chroma vector stores + SQLite database |
| `/app/results` | Pipeline output JSONs |

---

## REST API

All JSON endpoints require the server to be running.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/jobs/{id}/status` | Job status, progress, and paths |
| `POST` | `/api/jobs/{id}/query` | RAG query; body `{"q": "...", "llm_mode": "local\|hf"}` |
| `GET` | `/api/jobs/{id}/download?type=aligned\|transcription\|diarization` | Download output JSON |
| `POST` | `/api/jobs/{id}/reindex` | Trigger index rebuild from current transcript |

WebSocket endpoints:

| Path | Description |
|---|---|
| `/ws/jobs/{id}` | Real-time pipeline progress events (JSON) |
| `/ws/chat/{id}` | Streaming RAG chat; send `{"question":"...", "llm_mode":"..."}` |

---

## Reindexing after transcript edits

1. Open the job detail page for any completed job.
2. Edit speaker names or transcript text in the table.
3. Click **Reindex** (orange button in the header).
4. Confirm the dialog — the old index is wiped and rebuilt from the current transcript.
5. A progress bar appears while the new index is being built.
6. Once complete, the Query page will use the updated content.

---

## GPU notes

Transcription (Whisper) and diarization (Pyannote) benefit most from GPU. Embedding is fast on CPU for typical meeting lengths but can be enabled on CUDA with `EMBEDDING_DEVICE=cuda`.

If GPU memory runs out during indexing MeetBot automatically falls back to CPU for that stage and logs a warning.

For the local LLM (`USE_LOCAL_LLM=true`), `LOCAL_LLM_GPU_LAYERS` controls how many transformer layers are offloaded to VRAM. Start with `0` on machines with less than 6 GiB free VRAM after transcription; increase if you have headroom.

---

## Troubleshooting

**`ffmpeg` not found**
Install it: `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS).

**Pyannote download fails / 401 Unauthorized**
Set `HF_API_TOKEN` to a HuggingFace token that has accepted the Pyannote model conditions at https://huggingface.co/pyannote/speaker-diarization-3.1.

**HF Inference: "Model is not supported by provider"**
Leave `HF_PROVIDER=auto` and MeetBot will look up the correct provider for the selected model automatically. Pin it manually if you need a specific provider (e.g. `HF_PROVIDER=sambanova`).

**Local LLM loads but generates garbage**
Ensure the GGUF file matches the `LOCAL_LLM_MODEL_PATH` directory. Only GGUF quantised models are supported (not GPTQ or AWQ).

**`Error: 'Settings' object has no attribute X`**
A new environment variable was added. Either set it in `.env` or rely on the default; restarting the server is sufficient.

**Database migration errors on upgrade**
MeetBot runs lightweight column-add migrations on startup. If you see a schema error, back up `db/meetbot.db`, delete it, and restart — jobs will be lost but configuration is preserved.

---

## Running tests

```bash
source ../venv/bin/activate
python -m pytest tests/ --ignore=tests/test_characterization.py -v
```

Characterisation tests (`test_characterization.py`) require real model weights and a valid `HF_API_TOKEN`; skip them in CI with the `--ignore` flag above.

---

## License

MIT
