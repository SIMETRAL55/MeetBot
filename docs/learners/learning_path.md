# Learning Path: Beginner → Advanced

## Module 0 — Setup and first run
- **Objectives**: create venv, install deps, start web app.
- **Commands**:
```bash
cd /workspace/MeetBot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m meetbot.cli serve
```
- **Walkthrough targets**: `README.md`, `meetbot/config.py`, `meetbot/web/main.py`.
- **Exercise**: open `http://localhost:8080`, register/login.
- **Expected**: dashboard loads with no jobs.
- **Quiz**: Q1 Why venv? Q2 Where port configured? Q3 command to run server?

## Module 1 — Upload and run full pipeline
- **Objectives**: create job and inspect outputs.
- **Commands**:
```bash
python -m meetbot.cli serve
# in UI upload sample audio (or your own)
```
- **Walkthrough targets**: `meetbot/web/pages/upload.py`, `meetbot/workers/queue.py`, `meetbot/workers/pipeline_worker.py`.
- **Exercise**: upload audio and watch progress to COMPLETED.
- **Expected**: files in `results/` and vector folder in `db/`.
- **Quiz**: Q1 stage order? Q2 where progress stored? Q3 where aligned JSON saved?

## Module 2 — Transcription deep dive
- **Objectives**: understand local vs HF transcription.
- **Commands**:
```bash
export TRANSCRIPTION_BACKEND=local
python -m meetbot.cli run src/data/sample.mp3 || true
```
- **Targets**: `meetbot/services/transcriber.py`, `meetbot/adapters/transcribers/*`, `meetbot/utils/audio_chunker.py`.
- **Exercise**: switch backend to `huggingface` and compare logs.
- **Expected**: differing latency/model load behavior.
- **Quiz**: Q1 backend factory file? Q2 why chunking exists? Q3 what is language hint?

## Module 3 — Diarization + alignment
- **Objectives**: speaker segments and merge logic.
- **Commands**:
```bash
python -m meetbot.cli run path/to/audio.wav
```
- **Targets**: `meetbot/services/diarizer.py`, `meetbot/adapters/diarization.py`, `meetbot/services/aligner.py`.
- **Exercise**: inspect diarization/transcription/aligned JSON outputs.
- **Expected**: aligned rows have `start/end/speaker/text`.
- **Quiz**: Q1 diarization model? Q2 align input types? Q3 why alignment needed?

## Module 4 — Indexing and RAG retrieval
- **Objectives**: prepared docs, embeddings, vector search.
- **Commands**:
```bash
python -m meetbot.cli index results/<job>.json
python -m meetbot.cli query db/<collection> "What decisions were made?"
```
- **Targets**: `meetbot/services/prepare_docs.py`, `meetbot/services/indexer.py`, `meetbot/services/query_service.py`.
- **Exercise**: change `RAG_TOP_K` and re-query.
- **Expected**: source chunks change.
- **Quiz**: Q1 what is top-k? Q2 where prompt built? Q3 where vector db lives?

## Module 5 — Local LLM + HF fallback
- **Objectives**: run local GGUF and switch modes.
- **Commands**:
```bash
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL_PATH=./models/<model>.gguf
python -m meetbot.cli query db/<collection> "Summarize action items"
```
- **Targets**: `meetbot/adapters/llm/local_llm.py`, `meetbot/adapters/llm/hf_api.py`.
- **Exercise**: set local off and use HF API.
- **Expected**: answer returned from alternate backend.
- **Quiz**: Q1 what is GGUF? Q2 what controls GPU offload? Q3 fallback path?

## Module 6 — Debugging common issues
- **Objectives**: handle OOM, detached ORM, torchaudio mismatch.
- **Commands**:
```bash
python -m pytest tests/test_worker.py -v
python -m pytest tests/test_db.py -v
```
- **Targets**: `docs/learners/debug_recipes.md`.
- **Exercise**: reproduce one issue and apply documented fix.
- **Expected**: stable rerun.
- **Quiz**: Q1 DetachedInstanceError cause? Q2 OOM mitigation? Q3 check torch versions?

## Module 7 — Extend features safely
- **Objectives**: add endpoint and adjust prompts.
- **Commands**:
```bash
# add endpoint in meetbot/web/api.py, register in web/main.py
python -m pytest tests/ -q
```
- **Targets**: `meetbot/web/api.py`, `meetbot/web/main.py`, `meetbot/services/query_service.py`.
- **Exercise**: create `/api/healthz` and add prompt tweak.
- **Expected**: endpoint responds and query style changes.
- **Quiz**: Q1 where are routes registered? Q2 where prompt lives? Q3 how to verify quickly?
