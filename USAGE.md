# RAG Pipeline — Usage & Configuration Guide

## Overview

MeetBot ships with a **production-grade RAG pipeline** featuring:

- **Speaker-aware chunking** — segments are grouped into overlapping chunks
  that respect speaker boundaries.
- **Two-stage retrieval** — ANN recall (50+ candidates) followed by
  Maximal Marginal Relevance (MMR) reranking for a compact, non-redundant
  context window.
- **Answer-aware source filtering** — after the LLM generates an answer,
  source chunks are re-scored against the answer embedding so only the most
  relevant sources are shown.
- **Hierarchical summarisation** — map-reduce summarisation for
  "what was discussed?" / summary queries.
- **Atomic-swap reindex** — indexes are built in a temp directory and
  swapped atomically so queries never see a half-built index.
- **Streaming persistence** — partial answers are flushed to the database
  every 1 s / 128 tokens, so page refreshes show progress.
- **Transcript versioning** — every edit-save bumps a transcript version
  counter, enabling the reindex pipeline to detect stale indexes.

---

## Configuration

All settings can be set via environment variables or directly in
`meetbot/config.py`.

| Setting | Env var | Default | Description |
|---------|---------|---------|-------------|
| `RAG_RECALL_N` | `RAG_RECALL_N` | `50` | Number of candidates to retrieve from the vector store (stage 1). |
| `RAG_MAX_CONTEXT_CHUNKS` | `RAG_MAX_CONTEXT_CHUNKS` | `6` | Maximum chunks to pass to the LLM after MMR reranking (stage 2). |
| `CHUNK_TOKENS` | `CHUNK_TOKENS` | `300` | Target chunk size in estimated tokens (~4 chars/token). |
| `CHUNK_OVERLAP` | `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks in estimated tokens. |

---

## Edit → Save → Reindex flow

1. Open the **Job Detail** page and click on a transcript segment to edit.
2. Modify the text or speaker name, then click **Save**.
3. The save:
   - Updates the segment in the database.
   - Flushes the canonical transcript JSON to disk.
   - **Bumps the transcript version**.
4. Click **Reindex** on the job detail page to rebuild the vector index.
5. The reindex worker:
   - Reads segments from the database (canonical source of truth).
   - Chunks them with the speaker-aware chunker.
   - Builds a new Chroma collection in a `.tmp` directory.
   - Atomically swaps `.tmp` → live and removes the old dir.
   - Bumps the transcript version in the DB.

### What happens during reindex?

| Event | User experience |
|-------|-----------------|
| Build phase | Progress bar shows "Indexing…" |
| Swap phase | Takes < 1 s — queries keep using the old index until the rename completes |
| Swap complete | Queries immediately use the new index |
| Failure | Old index is restored automatically; error shown in progress |

---

## UI Controls

### Retrieval count (k)

On the **Query** page, the numeric input next to the chat lets you control
how many context chunks are sent to the LLM. This corresponds to the
`RAG_MAX_CONTEXT_CHUNKS` setting but can be adjusted per query.

### Stop / Cancel

- **Stop button** (during streaming) — sends a cooperative stop signal;
  the LLM finishes the current token and the partial answer is saved.
- **Cancel button** (on the dashboard or job detail) — marks the in-flight
  job for cancellation. The worker checks for cancellation at stage
  boundaries and raises `JobCancelledError`.
- **Restart button** — resets the job status to `queued` and re-enqueues it.

---

## Module Organisation

```
meetbot/services/rag/
├── __init__.py         # Package exports
├── chunker.py          # Speaker-aware transcript chunking
├── indexer.py          # Atomic-swap Chroma indexer
├── retriever.py        # ANN recall from vector store
├── reranker.py         # MMR diversity selection
├── selector.py         # Answer-embedding source filtering
└── summarizer.py       # Hierarchical map-reduce summarisation
```

Support modules:

```
meetbot/utils/persistence.py    # Streaming token → DB flusher
meetbot/db/crud.py              # flush_streaming_content(), bump_transcript_version()
```

---

## Running Tests

```bash
cd MeetBot
python -m pytest tests/ -v
```

All tests mock heavy dependencies (embedding models, ChromaDB) and run
in < 1 s without GPU.

---

## Streaming Persistence

During LLM generation the `StreamingPersister` utility:

1. Accumulates tokens in memory.
2. Every **1 second** or **128 tokens** (whichever comes first), flushes
   the partial content to `ChatMessage.content_partial` in the database.
3. On page refresh, the query page checks for messages with status
   `"streaming"` and displays `content_partial` with an orange badge.
4. When the stream finishes, `finalise()` sets the terminal status
   (`completed`, `stopped`, `interrupted`) and final content.

This ensures **no answer is lost** even if the browser tab is closed
mid-stream.

---

## Safety & Rollback

- **Atomic swap recovery**: if the rename fails, the indexer restores
  the old directory automatically.
- **DB migrations**: new columns (`transcript_version`, `content_partial`,
  `updated_at`) have safe defaults and `nullable=True`, so existing data
  is unaffected.
- **Legacy archive**: a pre-removal archive of the legacy RAG modules is
  available via `scripts/restore_legacy.sh` for emergency rollback.
