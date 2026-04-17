# Line-by-line Commentary: 10 High-impact Functions

> These are teaching snippets (trimmed for readability) from repository functions.

## 1) `run_pipeline` (`meetbot/workers/pipeline_worker.py`)
```python
# fetch job -> validate audio path -> stage updates
job = get_job(db, job_id)
if job is None: return
...
transcription = transcriber_svc.transcribe(...)
diarization = diarizer_svc.diarize(...)
aligned = aligner.build_speaker_transcript(...)
create_segments_from_aligned(db, job_id, aligned)
...
indexer_svc.build_index(...)
update_job_status(..., JobStatus.COMPLETED)
```
- Gets one job and performs full stage sequence.
- Uses status/progress updates at each stage for UI.
- Failure mode: any exception triggers `_fail_job` with error logged.

## 2) `JobQueue` loop (`meetbot/workers/queue.py`)
```python
while not stop_event.is_set():
    job_id = queue.get(timeout=...)
    run_pipeline(job_id)
```
- Consumes queued IDs in order.
- Keeps HTTP thread free.

## 3) `TranscriberService.transcribe` (`meetbot/services/transcriber.py`)
```python
wav_path = convert_to_wav(audio_path)
if should_chunk(wav_path):
    results = chunk_audio_for_transcription(...)
else:
    results = adapter.transcribe(wav_path, ...)
```
- Normalizes audio, optionally chunks large files.
- Failure mode: ffmpeg missing or backend model errors.

## 4) `DiarizationService.diarize` (`meetbot/services/diarizer.py`)
```python
adapter = get_diarization_adapter()
result = adapter.diarize(audio_path, min_speakers=..., max_speakers=...)
```
- Wraps diarization adapter with cache/progress scaffolding.

## 5) `AlignerService.build_speaker_transcript` (`meetbot/services/aligner.py`)
```python
for t in transcription_segments:
    speaker = find_best_overlap_speaker(t, diarization_segments)
    aligned.append({...})
```
- Assigns speaker label to each transcript chunk via time overlap.
- Failure mode: empty diarization yields unknown/default speakers.

## 6) `PrepareDocsService.prepare` (`meetbot/services/prepare_docs.py`)
```python
for seg in aligned_segments:
    docs.append({"page_content": text, "metadata": {...}})
write_jsonl(docs)
```
- Converts transcript rows into retrieval documents.

## 7) `IndexerService.build_index` (`meetbot/services/indexer.py`)
```python
emb = HuggingFaceEmbeddings(model_name=embedding_model)
vectordb = Chroma.from_documents(docs, emb, persist_directory=...)
vectordb.persist()
```
- Generates embeddings and stores vectors on disk.
- Failure mode: GPU OOM; worker may retry CPU.

## 8) `QueryService.query` (`meetbot/services/query_service.py`)
```python
retriever = vectordb.as_retriever(search_kwargs={"k": k})
sources = retriever.get_relevant_documents(question)
prompt = build_prompt(question, sources)
answer = llm.generate(prompt)
```
- Core RAG orchestration point.
- Change here to alter prompt style, retrieval depth, source formatting.

## 9) `LocalLLMManager.load` (`meetbot/adapters/llm/local_llm.py`)
```python
if not loaded:
    self.model = AutoModelForCausalLM.from_pretrained(path, gpu_layers=...)
```
- Lazy-loads local GGUF model and keeps instance available.
- Failure mode: wrong path/model format.

## 10) `websocket_chat_endpoint` flow (`meetbot/web/ws_chat.py`)
```python
async for msg in websocket:
    question = parse(msg)
    result = query_service.query(...)
    await websocket.send_json({"token": ...})
```
- Streams model output to client and may persist chat history.
- Failure mode: session/job lookup errors, disconnected clients.
