"""
Atomic-swap ChromaDB indexer for RAG.

Builds a new index into a temporary collection/namespace, then atomically
swaps it to the live namespace.  This ensures queries never see a partially
built index and reindex operations are safe even under concurrent reads.

Memory-safe streaming design
-----------------------------
The entire pipeline is streaming end-to-end:

    JSONL-on-disk → stream_jsonl() → chunked_iterable() → embed batch → collection.add() → del → gc

No full doc-list is ever materialised in memory.  The hash for metadata is
computed incrementally during the JSONL write phase and the total doc count
is tracked as a running counter — both avoid a second pass over the data.

Embedding and insertion happen in configurable batches (``EMBED_BATCH_SIZE``).
Between batches, memory is freed aggressively (``gc.collect`` + CUDA cache
flush).  The indexer reuses the process-level embedding model singleton to
avoid reloading large models per request.
"""

import gc
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]
CancelChecker = Callable[[], None]  # raises if cancelled


def _get_embedding_singleton(model_name: str, device: str = "cpu"):
    """Return cached HuggingFaceEmbeddings via the query_service singleton."""
    from ..query_service import _get_embedding_model
    return _get_embedding_model(model_name, device)


def _noop_cancel() -> None:
    """Default no-op cancellation checker."""
    pass


class RAGIndexer:
    """
    Build and manage ChromaDB vector indices with atomic swap support.

    Features:
    - Memory-safe batched embedding and insertion
    - Atomic swap: build temp → rename → delete old
    - Rich metadata per vector (job_id, chunk_id, start, end, speaker, version)
    - Uses cached embedding model singleton
    - Optional psutil-based memory monitoring per batch
    - Cooperative cancellation between batches
    """

    @staticmethod
    def compute_hash(docs: List[Dict[str, Any]], model_name: str) -> str:
        """Compute SHA256 hash of document texts + model name."""
        hasher = hashlib.sha256()
        for doc in docs:
            hasher.update(doc.get("text", "").encode("utf-8"))
        hasher.update(model_name.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def build_index_atomic(
        docs: List[Dict[str, Any]],
        persist_root: str,
        collection_name: str,
        embedding_model: str,
        device: str = "cpu",
        progress_callback: Optional[ProgressCallback] = None,
        job_id: str = "",
        version: int = 1,
        cancel_checker: CancelChecker = _noop_cancel,
    ) -> str:
        """
        Build a new vector index atomically.

        Steps:
        1. Build index in a temp directory (collection_name + ".tmp")
        2. If old index exists, rename to collection_name + ".old"
        3. Rename temp to live collection_name
        4. Delete old index

        Args:
            docs: List of document dicts with 'id', 'text', 'metadata' keys.
            persist_root: Root directory for all vector stores.
            collection_name: Name for the Chroma collection/subdirectory.
            embedding_model: Embedding model name or path.
            device: Compute device ("cpu" or "cuda").
            progress_callback: Optional progress reporter.
            job_id: Job ID for metadata.
            version: Transcript version for metadata.
            cancel_checker: Callable that raises if cancellation requested.

        Returns:
            Path to the live index directory.

        Raises:
            RuntimeError: If index creation fails.
        """
        if not docs:
            raise ValueError("No documents to index")

        root = Path(persist_root)
        root.mkdir(parents=True, exist_ok=True)

        live_dir = root / collection_name
        temp_dir = root / f"{collection_name}.tmp"
        old_dir = root / f"{collection_name}.old"

        # Clean up any leftover temp/old dirs from previous failed attempts
        for d in [temp_dir, old_dir]:
            if d.exists():
                try:
                    shutil.rmtree(str(d))
                except Exception as exc:
                    logger.warning("Could not remove leftover dir %s: %s", d, exc)

        if progress_callback:
            progress_callback("indexing", 10, "Preparing documents for indexing...")

        logger.info(
            "RAGIndexer: building index for %d docs (collection=%s, device=%s)",
            len(docs), collection_name, device,
        )

        # Step 1: Build index in temp directory (batched & memory-safe)
        try:
            RAGIndexer._build_chroma_index_batched(
                docs=docs,
                persist_dir=str(temp_dir),
                collection_name=collection_name,
                embedding_model=embedding_model,
                device=device,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker,
                total_hint=len(docs),
            )
        except Exception as exc:
            # Clean up temp on failure
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)
            raise RuntimeError(f"Failed to build temp index: {exc}") from exc

        logger.info("Built temp index for job %s at %s", job_id[:8] if job_id else "?", temp_dir)

        # Step 2: Atomic swap
        if progress_callback:
            progress_callback("indexing", 90, "Swapping index atomically...")

        # Invalidate chromadb cache before swap
        _invalidate_chroma_cache(job_id)

        try:
            if live_dir.exists():
                live_dir.rename(old_dir)
                logger.debug("Renamed live → old: %s → %s", live_dir, old_dir)

            temp_dir.rename(live_dir)
            logger.debug("Renamed temp → live: %s → %s", temp_dir, live_dir)

            # Clean up old
            if old_dir.exists():
                shutil.rmtree(str(old_dir), ignore_errors=True)
                logger.debug("Removed old index: %s", old_dir)

        except Exception as exc:
            # Attempt recovery: if live_dir doesn't exist but old does, restore
            if not live_dir.exists() and old_dir.exists():
                old_dir.rename(live_dir)
                logger.warning("Recovered live index from old after swap failure")
            raise RuntimeError(f"Atomic swap failed: {exc}") from exc

        # Invalidate cache after swap
        _invalidate_chroma_cache(job_id)

        # Write metadata
        meta = {
            "hash": RAGIndexer.compute_hash(docs, embedding_model),
            "model": embedding_model,
            "n_docs": len(docs),
            "collection_name": collection_name,
            "job_id": job_id,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = live_dir / ".index_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if progress_callback:
            progress_callback("indexing", 100, f"✓ Index built ({len(docs)} docs)")

        logger.info("Swapped index for job %s (%d docs)", job_id[:8] if job_id else "?", len(docs))
        return str(live_dir)

    # ------------------------------------------------------------------
    # Memory-safe batched Chroma builder (streaming)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chroma_index_batched(
        docs: Iterable[Dict[str, Any]],
        persist_dir: str,
        collection_name: str,
        embedding_model: str,
        device: str = "cpu",
        progress_callback: Optional[ProgressCallback] = None,
        cancel_checker: CancelChecker = _noop_cancel,
        total_hint: int = 0,
    ) -> int:
        """Build a Chroma index using batched embedding & incremental inserts.

        Accepts any **iterable** of doc dicts — including a generator backed
        by a JSONL file — so the full doc list never needs to reside in
        memory.

        Flow per batch:
            cancel_checker() → embed_documents(texts) → collection.add()
            → del embeddings → cleanup_memory()

        Args:
            docs: Iterable of doc dicts (``id``, ``text``, ``metadata``).
                  May be a list, generator, or any iterable.
            persist_dir: Directory for the Chroma persistent client.
            collection_name: Chroma collection name.
            embedding_model: Embedding model name or path.
            device: Compute device (``"cpu"`` or ``"cuda"``).
            progress_callback: Optional ``(stage, pct, msg)`` reporter.
            cancel_checker: Callable that raises if cancellation requested.
            total_hint: Expected number of docs (for progress reporting).
                        If 0, progress percentages are approximate.

        Returns:
            Number of documents actually inserted.
        """
        from ...config import settings
        from ...utils.memory import (
            cleanup_memory,
            log_memory_usage,
            check_memory_pressure,
            chunked_iterable,
        )

        batch_size = settings.EMBED_BATCH_SIZE
        memory_watch = settings.MEMORY_WATCH_ENABLED
        threshold = settings.MEMORY_WATCH_THRESHOLD_PCT
        checkpoint_interval = settings.INDEX_BATCH_PERSIST_CHECKPOINT

        if progress_callback:
            progress_callback("indexing", 30, f"Loading embedding model on {device.upper()}...")

        # ── Load embedder (singleton — no extra RAM) ─────────────────────
        embedder = _get_embedding_singleton(embedding_model, device)

        # ── Create empty Chroma collection ───────────────────────────────
        import chromadb

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        # Disable telemetry per-client as a belt-and-suspenders measure
        # (the process-level env var in main.py is the primary control).
        _chroma_settings = chromadb.Settings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=_chroma_settings,
        )
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(name=collection_name)

        if memory_watch:
            log_memory_usage("pre-embedding")

        # ── Batch embed + insert (streaming) ─────────────────────────────
        total = total_hint  # may be 0 if unknown
        inserted = 0
        batch_idx = 0

        if progress_callback:
            if total:
                n_batches_est = (total + batch_size - 1) // batch_size
                progress_callback(
                    "indexing", 40,
                    f"Embedding ~{total} docs in ~{n_batches_est} batches "
                    f"(batch_size={batch_size})...",
                )
            else:
                progress_callback(
                    "indexing", 40,
                    f"Embedding docs in batches (batch_size={batch_size})...",
                )

        t_start = time.monotonic()

        for batch in chunked_iterable(docs, batch_size):
            # Cooperative cancellation
            cancel_checker()

            batch_t0 = time.monotonic()

            texts = [d["text"] for d in batch]
            ids = [d.get("id", str(inserted + i)) for i, d in enumerate(batch)]
            metadatas = [d.get("metadata", {}) for d in batch]

            # Embed this batch
            try:
                embeddings = embedder.embed_documents(texts)
            except Exception as exc:
                # If memory pressure caused OOM, try halving batch once
                if memory_watch and check_memory_pressure(threshold) and len(texts) > 1:
                    logger.warning(
                        "Batch %d embed failed (%s) under memory pressure — "
                        "halving batch and retrying",
                        batch_idx + 1, exc,
                    )
                    cleanup_memory("oom-retry")
                    half = len(texts) // 2
                    emb1 = embedder.embed_documents(texts[:half])
                    emb2 = embedder.embed_documents(texts[half:])
                    embeddings = emb1 + emb2
                    del emb1, emb2
                else:
                    raise

            # Insert into Chroma
            collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=texts,
                embeddings=embeddings,
            )

            inserted += len(batch)
            batch_idx += 1

            # Free batch memory immediately
            del embeddings, texts, ids, metadatas, batch
            cleanup_memory(f"batch-{batch_idx}")

            batch_elapsed = time.monotonic() - batch_t0

            # Progress & logging
            if total:
                pct = 40 + (inserted / total) * 45  # 40–85 range
            else:
                pct = min(84, 40 + batch_idx * 2)  # approximate
            if progress_callback:
                progress_callback(
                    "indexing", pct,
                    f"Indexed batch {batch_idx} ({inserted}"
                    f"{'/' + str(total) if total else ''} docs, "
                    f"{batch_elapsed:.2f}s)",
                )

            if memory_watch and batch_idx % max(1, (total // batch_size // 5) if total else 5) == 0:
                log_memory_usage(f"batch-{batch_idx}")

            # Checkpoint logging
            if checkpoint_interval and batch_idx % checkpoint_interval == 0:
                logger.info(
                    "Checkpoint: %d batches complete (%d docs inserted)",
                    batch_idx, inserted,
                )

        elapsed = time.monotonic() - t_start

        if memory_watch:
            log_memory_usage("post-embedding")

        logger.info(
            "RAGIndexer: built Chroma index via streaming batched inserts "
            "(%d docs, %d batches, %.1fs)",
            inserted, batch_idx, elapsed,
        )
        return inserted

    # ------------------------------------------------------------------
    # Legacy single-call builder (kept as fallback, not used by default)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chroma_index(
        docs: List[Dict[str, Any]],
        persist_dir: str,
        collection_name: str,
        embedding_model: str,
        device: str = "cpu",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Build a Chroma index in the given directory using LangChain or direct API.

        .. deprecated::
            Use :meth:`_build_chroma_index_batched` instead for memory-safe
            indexing.  This method is kept as a fallback for debugging when
            ``RAG_MEMORY_SAFE_MODE=false``.
        """

        if progress_callback:
            progress_callback("indexing", 30, f"Loading embedding model on {device.upper()}...")

        # Try LangChain approach first
        try:
            from langchain_core.documents import Document

            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            embedding = _get_embedding_singleton(embedding_model, device)

            if progress_callback:
                progress_callback("indexing", 50, f"Embedding {len(docs)} documents...")

            lc_docs = [
                Document(page_content=d["text"], metadata=d.get("metadata", {}))
                for d in docs
            ]

            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback("indexing", 60, "Creating vector index...")

            vectordb = Chroma.from_documents(
                documents=lc_docs,
                embedding=embedding,
                persist_directory=persist_dir,
                collection_name=collection_name,
            )

            # Persist if needed
            try:
                vectordb.persist()
            except AttributeError:
                pass

            logger.info("RAGIndexer: built Chroma index via LangChain (%d docs)", len(docs))
            return

        except Exception as exc:
            logger.warning("LangChain indexing failed: %s — trying direct chromadb", exc)

        # Fallback: sentence-transformers + chromadb direct
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            import numpy as np

            if progress_callback:
                progress_callback("indexing", 40, f"Loading SentenceTransformer on {device}...")

            sbert = SentenceTransformer(embedding_model, device=device)

            texts = [d["text"] for d in docs]
            if progress_callback:
                progress_callback("indexing", 55, f"Computing embeddings for {len(texts)} docs...")

            batch_size = 8 if device == "cuda" else 16
            embeddings = sbert.encode(texts, batch_size=batch_size, convert_to_numpy=True)

            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            _settings = chromadb.Settings(anonymized_telemetry=False)
            client = chromadb.PersistentClient(path=persist_dir, settings=_settings)

            try:
                collection = client.get_collection(collection_name)
            except Exception:
                collection = client.create_collection(name=collection_name)

            ids = [d.get("id", str(i)) for i, d in enumerate(docs)]
            metadatas = [d.get("metadata", {}) for d in docs]

            collection.upsert(
                ids=ids,
                metadatas=metadatas,
                documents=texts,
                embeddings=embeddings.tolist(),
            )

            logger.info("RAGIndexer: built Chroma index via chromadb (%d docs)", len(docs))

        except Exception as exc:
            raise RuntimeError(f"All indexing approaches failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Multi-level index builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_multilevel_index_atomic(
        segments: List[Dict[str, Any]],
        persist_root: str,
        collection_name: str,
        embedding_model: str,
        device: str = "cpu",
        progress_callback: Optional[ProgressCallback] = None,
        job_id: str = "",
        version: int = 1,
        chunk_tokens: int = 300,
        chunk_overlap: int = 50,
        cancel_checker: CancelChecker = _noop_cancel,
    ) -> str:
        """
        Build a multi-level vector index atomically from raw transcript segments.

        Memory-safe: writes intermediate doc lists to JSONL on disk, then
        streams them in batches through the embedder.  No large in-memory
        list of all docs is built.

        Creates **three** embedding levels in a single Chroma collection:

        1. ``level="document"`` — One vector for the full concatenated transcript.
           Best for summarisation and broad "what was discussed" queries.

        2. ``level="segment"`` — One vector per original segment with per-turn
           ``speaker``, ``start``, and ``end`` metadata preserved exactly.
           Best for speaker-specific and timestamped queries.

        3. ``level="chunk"`` — Overlapping chunks built by the speaker-aware Chunker.
           Best for localised factual retrieval with MMR reranking.

        Design: **single collection, ``level`` metadata filter** (vs. separate
        collections).  Trade-offs:
        - ✅ One atomic swap covers all three levels.
        - ✅ Single model-load batch; all vectors in one call.
        - ✅ Chroma WHERE filtering is O(n) but transcripts are small (< 1 k segs).
        - ⚠️  Care needed: ``count_documents`` must now filter by level.

        Args:
            segments: Raw transcript segments (list of dicts with text/speaker/start/end).
            persist_root: Root directory for vector stores.
            collection_name: Chroma collection / subdirectory name.
            embedding_model: Embedding model name or path.
            device: "cpu" or "cuda".
            progress_callback: Optional progress reporter.
            job_id: Job identifier for metadata.
            version: Transcript version for stale-index detection.
            chunk_tokens: Target chunk size (passed to Chunker).
            chunk_overlap: Overlap tokens between chunks (passed to Chunker).
            cancel_checker: Callable that raises if cancellation requested.

        Returns:
            Path string to the live index directory (same as build_index_atomic).
        """
        from .chunker import Chunker
        from ...config import settings
        from ...utils.memory import cleanup_memory, log_memory_usage

        temp_dir_path = Path(settings.TEMP_DIR) / job_id
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = temp_dir_path / "multilevel_docs.jsonl"

        memory_watch = settings.MEMORY_WATCH_ENABLED
        if memory_watch:
            log_memory_usage("multilevel-start")

        # Incremental hash — updated per-doc during JSONL write
        hasher = hashlib.sha256()

        # ── Write all docs to JSONL on disk (streaming, never hold all) ──
        total_docs = 0

        root = Path(persist_root)
        root.mkdir(parents=True, exist_ok=True)
        live_dir = root / collection_name
        temp_index_dir = root / f"{collection_name}.tmp"
        old_dir = root / f"{collection_name}.old"

        # Clean up any leftover temp/old dirs
        for d in [temp_index_dir, old_dir]:
            if d.exists():
                try:
                    shutil.rmtree(str(d))
                except Exception as exc:
                    logger.warning("Could not remove leftover dir %s: %s", d, exc)

        try:
            with open(jsonl_path, "w", encoding="utf-8") as fh:
                # ── 1. Document-level doc ────────────────────────────────
                full_text = " ".join(
                    (s.get("text") or s.get("transcript") or "").strip()
                    for s in segments
                ).strip()
                if not full_text:
                    raise ValueError("No text found in segments — cannot build index")

                doc_entry = {
                    "id": f"{collection_name}_doc",
                    "text": full_text,
                    "metadata": {
                        "level": "document",
                        "job_id": job_id,
                        "doc_id": collection_name,
                        "start": segments[0].get("start", 0.0) if segments else 0.0,
                        "end": segments[-1].get("end", 0.0) if segments else 0.0,
                        "speaker": "multiple",
                        "version": version,
                    },
                }
                fh.write(json.dumps(doc_entry, ensure_ascii=False) + "\n")
                hasher.update(full_text.encode("utf-8"))
                total_docs += 1
                del full_text, doc_entry

                # ── 2. Segment-level docs ────────────────────────────────
                seg_count = 0
                for i, seg in enumerate(segments):
                    text = (seg.get("text") or seg.get("transcript") or "").strip()
                    if not text:
                        continue
                    seg_entry = {
                        "id": f"{collection_name}_seg_{i}",
                        "text": text,
                        "metadata": {
                            "level": "segment",
                            "job_id": job_id,
                            "segment_id": i,
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", 0.0),
                            "speaker": seg.get("speaker") or seg.get("spk") or "unknown",
                            "version": version,
                        },
                    }
                    fh.write(json.dumps(seg_entry, ensure_ascii=False) + "\n")
                    hasher.update(text.encode("utf-8"))
                    seg_count += 1
                    total_docs += 1
                del seg_entry  # last ref

                cancel_checker()

                # ── 3. Chunk-level docs ──────────────────────────────────
                chunker = Chunker(chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap)
                chunk_count = 0
                for c in chunker.chunk_segments_iter(segments, job_id=job_id, version=version):
                    doc = c.to_doc()  # already has level="chunk"
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    hasher.update(doc.get("text", "").encode("utf-8"))
                    chunk_count += 1
                    total_docs += 1
                del chunker
                cleanup_memory("jsonl-write-done")

            # Finalise hash (include model name for cache-busting)
            hasher.update(embedding_model.encode("utf-8"))
            doc_hash = hasher.hexdigest()

            logger.info(
                "RAGIndexer: multilevel index — 1 doc + %d segments + %d chunks = %d total "
                "(written to JSONL, hash=%s)",
                seg_count, chunk_count, total_docs, doc_hash[:12],
            )

            if progress_callback:
                progress_callback("indexing", 10, "Preparing documents for indexing...")

            # ── Stream from JSONL → batched builder (no list materialisation) ──
            try:
                inserted = RAGIndexer._build_chroma_index_batched(
                    docs=_stream_jsonl(jsonl_path),
                    persist_dir=str(temp_index_dir),
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    device=device,
                    progress_callback=progress_callback,
                    cancel_checker=cancel_checker,
                    total_hint=total_docs,
                )
            except Exception as exc:
                if temp_index_dir.exists():
                    shutil.rmtree(str(temp_index_dir), ignore_errors=True)
                raise RuntimeError(f"Failed to build temp index: {exc}") from exc

            # ── Atomic swap (temp → live) ────────────────────────────────
            if progress_callback:
                progress_callback("indexing", 90, "Swapping index atomically...")

            _invalidate_chroma_cache(job_id)

            try:
                if live_dir.exists():
                    live_dir.rename(old_dir)
                    logger.debug("Renamed live → old: %s → %s", live_dir, old_dir)

                temp_index_dir.rename(live_dir)
                logger.debug("Renamed temp → live: %s → %s", temp_index_dir, live_dir)

                if old_dir.exists():
                    shutil.rmtree(str(old_dir), ignore_errors=True)
                    logger.debug("Removed old index: %s", old_dir)
            except Exception as exc:
                if not live_dir.exists() and old_dir.exists():
                    old_dir.rename(live_dir)
                    logger.warning("Recovered live index from old after swap failure")
                raise RuntimeError(f"Atomic swap failed: {exc}") from exc

            _invalidate_chroma_cache(job_id)

            # Write metadata with pre-computed hash and count
            meta = {
                "hash": doc_hash,
                "model": embedding_model,
                "n_docs": inserted,
                "collection_name": collection_name,
                "job_id": job_id,
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_path = live_dir / ".index_meta.json"
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
            )

            if progress_callback:
                progress_callback("indexing", 100, f"✓ Index built ({inserted} docs)")

            if memory_watch:
                log_memory_usage("multilevel-end")

            logger.info(
                "Swapped multilevel index for job %s (%d docs)",
                job_id[:8] if job_id else "?", inserted,
            )
            return str(live_dir)

        finally:
            # Cleanup temp JSONL
            try:
                if jsonl_path.exists():
                    jsonl_path.unlink()
                if temp_dir_path.exists() and not any(temp_dir_path.iterdir()):
                    temp_dir_path.rmdir()
            except Exception:
                pass

    @staticmethod
    def get_doc_count(db_dir: str) -> int:
        """Return *chunk-level* document count from a Chroma collection.

        For multi-level indexes this filters by ``level="chunk"`` so the
        returned number reflects the number of retrievable chunk vectors —
        the value the UI uses as the upper bound for the k selector.

        Falls back to the total collection count for legacy (pre-multilevel)
        indexes that have no ``level`` metadata field.
        """
        from pathlib import Path
        db_path = Path(db_dir)
        if not db_path.exists():
            return 0
        try:
            import chromadb
            _settings = chromadb.Settings(anonymized_telemetry=False)
            client = chromadb.PersistentClient(path=str(db_path), settings=_settings)
            collection = client.get_collection(db_path.name)
            # Try chunk-level filter first
            try:
                chunk_count = collection.count()  # fallback: total
                results = collection.get(where={"level": {"$eq": "chunk"}}, limit=1)
                if results and results.get("ids"):
                    # Count via query — chromadb has no direct count-with-filter
                    all_chunks = collection.get(where={"level": {"$eq": "chunk"}})
                    chunk_count = len(all_chunks.get("ids", []))
                return chunk_count
            except Exception:
                return collection.count()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _hash_segments(segments: List[Dict[str, Any]]) -> str:
    """Return a SHA256 hex-digest of the segment texts + speakers.

    Used by the reindex worker to skip ChromaDB rebuilds when the transcript
    has not changed since the last successful index.
    """
    hasher = hashlib.sha256()
    for seg in segments:
        hasher.update((seg.get("text") or "").encode("utf-8"))
        hasher.update((seg.get("speaker") or "").encode("utf-8"))
    return hasher.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    .. deprecated::
        Use :func:`_stream_jsonl` for memory-safe streaming reads.
        Kept for backward-compatible callers that require a ``list``.
    """
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _stream_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Lazily yield dicts from a JSONL file one line at a time.

    Unlike :func:`_read_jsonl`, this never loads the entire file into
    memory — each doc is parsed, yielded, and immediately eligible
    for garbage collection.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _invalidate_chroma_cache(job_id: str = "") -> None:
    """Clear chromadb's process-level SharedSystemClient cache."""
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
        logger.debug("Cleared chromadb cache (job=%s)", job_id[:8] if job_id else "?")
    except Exception as exc:
        logger.debug("chromadb cache clear skipped: %s", exc)
    gc.collect()
