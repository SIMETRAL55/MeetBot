# Lab 1: End-to-end pipeline with local LLM disabled

## Goal
Run upload→process→query using HF backend for answers.

## Commands
```bash
cd /workspace/MeetBot
source .venv/bin/activate
export USE_LOCAL_LLM=false
export TRANSCRIPTION_BACKEND=huggingface
python -m meetbot.cli serve
```

Then in browser:
1. Upload an audio file.
2. Wait for status `COMPLETED`.
3. Open Query and ask: `List action items`.

## Expected output excerpts
- Logs include stage transitions: `TRANSCRIBING`, `DIARIZING`, `ALIGNING`, `INDEXING`, `COMPLETED`.
- Query returns answer text and source snippets.

## Inspect filesystem
- `results/<job_id>_transcription.json`
- `results/<job_id>_diarization.json`
- `results/<job_id>.json`
- `db/<job_prefix>/` (Chroma files)
