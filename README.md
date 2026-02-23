# MeetBot 🎙️

MeetBot helps you turn long audio conversations into something you can actually use:

- **Transcribe** audio (Whisper via Hugging Face)
- **Diarize** speakers (Pyannote)
- **Align** text with speaker turns
- **Prepare + index + query** transcripts for lightweight RAG workflows

In simple words: you can feed it meeting/interview audio and get structured speaker-labeled transcripts, then ask questions over that content.

---

## Why this project exists

Raw audio is hard to search and painful to review.
MeetBot turns that into:

- clean JSON outputs in `results/`
- prepared chunk documents in `prepared/`
- vector indexes in `db/`

So instead of replaying a 45-minute file, you can search and ask questions directly.

---

## Current architecture (human-readable)

The repository recently moved from flat scripts to a cleaner package layout:

- `meetbot/config/` → settings and environment config
- `meetbot/adapters/` → external model clients (HF / pyannote)
- `meetbot/infra/` → infrastructure helpers (cache, audio conversion/chunking)
- `meetbot/services/` → core business logic (transcribe, diarize, align, pipeline)
- `meetbot/cli/` → command entrypoint(s)
- `src/source/` → backward-compatible shims for older imports/scripts

This means features are easier to add without touching unrelated files.

---

## Quick start

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

> You also need **ffmpeg** available on your system path.

## 2) Configure environment

Create a `.env` file in repository root:

```env
HF_API_TOKEN=your_hf_token_here
# Optional aliases that are also supported:
# HF_HUB_TOKEN=...
# HUGGINGFACEHUB_API_TOKEN=...
```

## 3) Run the audio pipeline

Use the new package CLI:

```bash
python -m meetbot.cli.pipeline_cmd src/data/sample.mp3 --use-cache
```

Useful flags:

- `--language en` (or another hint language)
- `--no-cache` to bypass cache
- `--force-refresh` to refresh cache
- `--min-speakers 2 --max-speakers 4`

Output is written to `results/` with speaker-labeled JSON.

---

## Legacy compatibility

If you previously used scripts under `src/source/`, they still work as wrappers.
You can migrate gradually to the new `meetbot.*` modules.

---

## RAG workflow (prepare → index → query)

After generating transcript JSON:

### 1) Prepare documents

```bash
python src/source/prepare_docs.py results/sample.json --out-dir prepared
```

### 2) Build vector index

```bash
python src/source/build_index.py prepared/sample.jsonl --persist-root db
```

### 3) Ask questions

```bash
python src/source/query.py --db-root db --audio sample --question "What was the key decision?"
```

---

## Typical folders you will see

- `src/data/` → input audio samples
- `results/` → diarized transcript JSON outputs
- `prepared/` → normalized docs for embeddings
- `db/` → persisted Chroma indexes
- `.cache_hf/` → cached Hugging Face responses

---

## Testing

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

Run a quick compile sanity check:

```bash
python -m compileall meetbot src/source tests
```

---

## Common issues

### "No HF token found"
Set one of:

- `HF_API_TOKEN`
- `HF_HUB_TOKEN`
- `HUGGINGFACEHUB_API_TOKEN`

### pyannote / diarization errors
- Ensure `pyannote.audio` is installed (included in `requirements.txt`)
- Ensure `ffmpeg` is installed
- Ensure your HF token has access to required models

### Slow first run
Expected. First run downloads models and builds caches/indexes.

---

## Project status

This codebase is actively being refactored for better maintainability.
The new package layout is in place, with compatibility shims retained to avoid breaking existing usage.

If you’re contributing: prefer adding new logic under `meetbot/` rather than `src/source/`.
