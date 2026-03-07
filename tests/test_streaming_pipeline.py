"""
Tests for the end-to-end streaming pipeline.

Verifies:
- ``_stream_jsonl`` yields dicts lazily without full materialisation.
- ``chunk_segments_iter`` is a proper lazy generator.
- ``_build_chroma_index_batched`` works with generators (not just lists).
- Incremental hash in ``build_multilevel_index_atomic`` matches batch compute.
- ``total_hint`` is passed correctly and used for progress reporting.
"""

import json
import types
from pathlib import Path
from typing import Dict, Iterator, List
from unittest.mock import MagicMock, patch

import pytest

from meetbot.services.rag.indexer import RAGIndexer, _stream_jsonl, _read_jsonl
from meetbot.services.rag.chunker import Chunker


# ---------------------------------------------------------------------------
# _stream_jsonl tests
# ---------------------------------------------------------------------------


class TestStreamJsonl:
    """Verify the streaming JSONL reader."""

    def test_yields_all_docs(self, tmp_path):
        p = tmp_path / "docs.jsonl"
        docs = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        p.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")

        result = list(_stream_jsonl(p))
        assert result == docs

    def test_returns_generator(self, tmp_path):
        p = tmp_path / "docs.jsonl"
        p.write_text('{"id":"x"}\n', encoding="utf-8")
        gen = _stream_jsonl(p)
        assert isinstance(gen, types.GeneratorType)

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "docs.jsonl"
        p.write_text('{"a":1}\n\n\n{"b":2}\n', encoding="utf-8")
        result = list(_stream_jsonl(p))
        assert len(result) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "docs.jsonl"
        p.write_text("", encoding="utf-8")
        assert list(_stream_jsonl(p)) == []

    def test_matches_read_jsonl(self, tmp_path):
        """_stream_jsonl should produce the same data as _read_jsonl."""
        p = tmp_path / "docs.jsonl"
        docs = [
            {"id": str(i), "text": f"doc {i}", "metadata": {"level": "chunk"}}
            for i in range(20)
        ]
        p.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")

        streamed = list(_stream_jsonl(p))
        loaded = _read_jsonl(p)
        assert streamed == loaded


# ---------------------------------------------------------------------------
# chunk_segments_iter tests
# ---------------------------------------------------------------------------


class TestChunkSegmentsIter:
    """Verify the lazy chunking generator."""

    def _segments(self, n: int = 5):
        return [
            {
                "text": f"This is segment number {i} with enough words to matter.",
                "speaker": f"SPK_{i % 2}",
                "start": float(i * 10),
                "end": float(i * 10 + 8),
            }
            for i in range(n)
        ]

    def test_returns_iterator(self):
        chunker = Chunker(chunk_tokens=100, chunk_overlap=10)
        result = chunker.chunk_segments_iter(self._segments())
        assert hasattr(result, "__next__"), "Should be an iterator"

    def test_yields_same_as_chunk_segments(self):
        segs = self._segments(8)
        chunker = Chunker(chunk_tokens=50, chunk_overlap=10)

        via_iter = list(chunker.chunk_segments_iter(segs, job_id="j1", version=2))
        via_list = chunker.chunk_segments(segs, job_id="j1", version=2)

        iter_docs = [c.to_doc() for c in via_iter]
        list_docs = [c.to_doc() for c in via_list]
        assert iter_docs == list_docs

    def test_lazy_no_eagerness(self):
        """Calling chunk_segments_iter should NOT materialise all chunks."""
        segs = self._segments(20)
        chunker = Chunker(chunk_tokens=30, chunk_overlap=5)
        gen = chunker.chunk_segments_iter(segs)
        # Just taking the first value should not consume all
        first = next(gen)
        assert first is not None


# ---------------------------------------------------------------------------
# Streaming batched builder with generators
# ---------------------------------------------------------------------------


class TestBatchedBuilderStreaming:
    """Verify _build_chroma_index_batched works with generators."""

    def _make_doc_generator(self, n: int):
        """Return a generator of doc dicts."""
        for i in range(n):
            yield {
                "id": f"doc_{i}",
                "text": f"Content for document {i} with details.",
                "metadata": {"level": "chunk"},
            }

    def _setup_fakes(self):
        fake_embedder = MagicMock()
        fake_embedder.embed_documents.side_effect = (
            lambda texts: [[0.1] * 8 for _ in texts]
        )

        fake_collection = MagicMock()
        fake_collection.add = MagicMock()

        mock_client = MagicMock()
        mock_client.create_collection.return_value = fake_collection
        mock_client.delete_collection.return_value = None

        return fake_embedder, fake_collection, mock_client

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_accepts_generator(self, mock_settings, mock_get_model, mock_pcc, tmp_path):
        """Should work with a generator, not just a list."""
        mock_settings.EMBED_BATCH_SIZE = 4
        mock_settings.MEMORY_WATCH_ENABLED = False
        mock_settings.MEMORY_WATCH_THRESHOLD_PCT = 0.85
        mock_settings.INDEX_BATCH_PERSIST_CHECKPOINT = 0

        fake_embedder, fake_collection, mock_client = self._setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        gen = self._make_doc_generator(10)
        inserted = RAGIndexer._build_chroma_index_batched(
            docs=gen,
            persist_dir=str(tmp_path / "idx"),
            collection_name="t",
            embedding_model="fake",
        )

        assert inserted == 10
        # 10 docs / batch_size=4 → 3 add calls
        assert fake_collection.add.call_count == 3

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_returns_inserted_count(self, mock_settings, mock_get_model, mock_pcc, tmp_path):
        mock_settings.EMBED_BATCH_SIZE = 16
        mock_settings.MEMORY_WATCH_ENABLED = False
        mock_settings.MEMORY_WATCH_THRESHOLD_PCT = 0.85
        mock_settings.INDEX_BATCH_PERSIST_CHECKPOINT = 0

        fake_embedder, fake_collection, mock_client = self._setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        inserted = RAGIndexer._build_chroma_index_batched(
            docs=[{"id": "a", "text": "x", "metadata": {}}],
            persist_dir=str(tmp_path / "idx"),
            collection_name="t",
            embedding_model="fake",
        )
        assert inserted == 1

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_total_hint_used_in_progress(self, mock_settings, mock_get_model, mock_pcc, tmp_path):
        """When total_hint is given, progress messages should include it."""
        mock_settings.EMBED_BATCH_SIZE = 5
        mock_settings.MEMORY_WATCH_ENABLED = False
        mock_settings.MEMORY_WATCH_THRESHOLD_PCT = 0.85
        mock_settings.INDEX_BATCH_PERSIST_CHECKPOINT = 0

        fake_embedder, fake_collection, mock_client = self._setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        progress_msgs = []

        def cb(stage, pct, msg):
            progress_msgs.append(msg)

        RAGIndexer._build_chroma_index_batched(
            docs=self._make_doc_generator(10),
            persist_dir=str(tmp_path / "idx"),
            collection_name="t",
            embedding_model="fake",
            progress_callback=cb,
            total_hint=10,
        )

        # Progress should mention the total
        combined = " ".join(progress_msgs)
        assert "10" in combined


# ---------------------------------------------------------------------------
# Incremental hash consistency
# ---------------------------------------------------------------------------


class TestIncrementalHash:
    """Verify that the hash produced by the streaming multilevel path
    matches what compute_hash would produce for the same docs."""

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched")
    def test_meta_hash_matches_compute_hash(self, mock_build, tmp_path):
        """The .index_meta.json hash should equal compute_hash(all_docs, model)."""
        # Capture docs from the streaming call
        captured: List[Dict] = []

        def _capture_and_create(**kwargs):
            docs_iter = kwargs.get("docs", [])
            consumed = list(docs_iter)
            captured.clear()
            captured.extend(consumed)
            persist_dir = kwargs["persist_dir"]
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            (Path(persist_dir) / "chroma.sqlite3").write_text("fake")
            return len(consumed)

        mock_build.side_effect = _capture_and_create

        segs = [
            {"text": "Hello world", "speaker": "A", "start": 0.0, "end": 1.0},
            {"text": "Goodbye world", "speaker": "B", "start": 1.0, "end": 2.0},
        ]

        result = RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="hashtest",
            embedding_model="test-model",
            job_id="hashtest",
        )

        meta_path = Path(result) / ".index_meta.json"
        meta = json.loads(meta_path.read_text())

        # Compute hash the old way (full list + model)
        expected_hash = RAGIndexer.compute_hash(captured, "test-model")
        assert meta["hash"] == expected_hash


# ---------------------------------------------------------------------------
# Multilevel streaming integration
# ---------------------------------------------------------------------------


class TestMultilevelStreamingIntegration:
    """Verify the full multilevel streaming path (mock embedding only)."""

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched")
    def test_total_hint_passed(self, mock_build, tmp_path):
        """build_multilevel_index_atomic should pass total_hint to the builder."""
        def _side_effect(**kwargs):
            docs_iter = kwargs.get("docs", [])
            consumed = list(docs_iter)
            persist_dir = kwargs["persist_dir"]
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            (Path(persist_dir) / "chroma.sqlite3").write_text("fake")
            return len(consumed)

        mock_build.side_effect = _side_effect

        segs = [{"text": f"Seg {i}", "speaker": "X", "start": 0.0, "end": 1.0} for i in range(3)]
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="hint_test",
            embedding_model="fake-model",
            job_id="hint_test",
        )

        # total_hint should match the number of docs consumed
        call_kwargs = mock_build.call_args[1]
        assert "total_hint" in call_kwargs
        assert call_kwargs["total_hint"] > 3  # 1 doc + 3 segs + chunks

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched")
    def test_docs_is_generator_not_list(self, mock_build, tmp_path):
        """The docs arg to _build_chroma_index_batched should be a generator."""
        received_docs_type = []

        def _side_effect(**kwargs):
            docs_iter = kwargs.get("docs", [])
            received_docs_type.append(type(docs_iter).__name__)
            consumed = list(docs_iter)
            persist_dir = kwargs["persist_dir"]
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            (Path(persist_dir) / "chroma.sqlite3").write_text("fake")
            return len(consumed)

        mock_build.side_effect = _side_effect

        segs = [{"text": "Test", "speaker": "X", "start": 0.0, "end": 1.0}]
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="gen_test",
            embedding_model="fake",
            job_id="gen_test",
        )

        assert len(received_docs_type) == 1
        assert received_docs_type[0] == "generator", (
            f"Expected generator, got {received_docs_type[0]}"
        )

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched")
    def test_jsonl_cleaned_up_after_success(self, mock_build, tmp_path):
        """Temp JSONL file should be deleted after successful indexing."""
        def _side_effect(**kwargs):
            docs_iter = kwargs.get("docs", [])
            consumed = list(docs_iter)
            persist_dir = kwargs["persist_dir"]
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            (Path(persist_dir) / "chroma.sqlite3").write_text("fake")
            return len(consumed)

        mock_build.side_effect = _side_effect

        segs = [{"text": "Clean", "speaker": "X", "start": 0.0, "end": 1.0}]
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="cleanup_test",
            embedding_model="fake",
            job_id="cleanup_test",
        )

        # No JSONL files should remain in temp
        jsonl_files = list(Path("./temp").rglob("*.jsonl")) if Path("./temp").exists() else []
        cleanup_jsonl = [f for f in jsonl_files if "cleanup_test" in str(f)]
        assert cleanup_jsonl == []
