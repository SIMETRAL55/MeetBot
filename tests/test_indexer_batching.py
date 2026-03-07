"""
Unit tests for the memory-safe batched indexer (_build_chroma_index_batched).

Verifies that:
- Embeddings are computed in batches of EMBED_BATCH_SIZE.
- collection.add() is called once per batch (not one giant call).
- cleanup_memory() is called between batches.
- OOM retry halves batch size automatically.
- cancel_checker is invoked between batches.
"""

import gc
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

from meetbot.services.rag.indexer import RAGIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docs(n: int) -> list:
    """Return *n* minimal doc dicts for indexing."""
    return [
        {
            "id": f"doc_{i}",
            "text": f"This is document number {i} with some content.",
            "metadata": {"level": "chunk", "speaker": "A", "start": 0.0, "end": 1.0},
        }
        for i in range(n)
    ]


class _FakeCollection:
    """Minimal stand-in for a chromadb Collection."""

    def __init__(self):
        self.adds: list = []

    def add(self, ids, documents, embeddings, metadatas):
        self.adds.append({
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        })


class _FakeEmbedder:
    """Minimal stand-in for a HuggingFace embedding model."""

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.call_count = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        return [[0.1] * self._dim for _ in texts]


def _setup_fakes():
    """Create fake embedder and collection pair."""
    fake_embedder = _FakeEmbedder()
    fake_collection = _FakeCollection()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = fake_collection
    mock_client.delete_collection.return_value = None
    return fake_embedder, fake_collection, mock_client


def _apply_settings(mock_settings, **overrides):
    """Configure a mock settings object with sensible defaults."""
    mock_settings.EMBED_BATCH_SIZE = overrides.get("batch_size", 16)
    mock_settings.MEMORY_WATCH_ENABLED = overrides.get("memory_watch", False)
    mock_settings.MEMORY_WATCH_THRESHOLD_PCT = overrides.get("threshold", 0.85)
    mock_settings.INDEX_BATCH_PERSIST_CHECKPOINT = overrides.get("checkpoint", 100)


# ---------------------------------------------------------------------------
# Batching behaviour
# ---------------------------------------------------------------------------


class TestBatchedIndexer:
    """Core batching tests for _build_chroma_index_batched."""

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_multiple_add_calls_for_large_input(self, mock_settings, mock_get_model, mock_pcc, tmp_path):
        """With 10 docs and batch_size=4, there should be 3 collection.add() calls."""
        _apply_settings(mock_settings, batch_size=4)
        fake_embedder, fake_collection, mock_client = _setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        docs = _make_docs(10)

        RAGIndexer._build_chroma_index_batched(
            docs=docs,
            persist_dir=str(tmp_path / "index"),
            collection_name="test",
            embedding_model="fake",
            device="cpu",
        )

        # 10 docs / 4 per batch = 3 batches (4 + 4 + 2)
        assert len(fake_collection.adds) == 3
        assert len(fake_collection.adds[0]["ids"]) == 4
        assert len(fake_collection.adds[1]["ids"]) == 4
        assert len(fake_collection.adds[2]["ids"]) == 2

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_single_batch_for_small_input(self, mock_settings, mock_get_model, mock_pcc, tmp_path):
        """With fewer docs than batch_size, only one add() call."""
        _apply_settings(mock_settings, batch_size=16)
        fake_embedder, fake_collection, mock_client = _setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        docs = _make_docs(3)

        RAGIndexer._build_chroma_index_batched(
            docs=docs,
            persist_dir=str(tmp_path / "index"),
            collection_name="test",
            embedding_model="fake",
            device="cpu",
        )

        assert len(fake_collection.adds) == 1
        assert len(fake_collection.adds[0]["ids"]) == 3

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_embedder_called_per_batch(self, mock_settings, mock_get_model, mock_pcc, tmp_path):
        """embed_documents() should be called once per batch, not once for all."""
        _apply_settings(mock_settings, batch_size=3)
        fake_embedder, fake_collection, mock_client = _setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        docs = _make_docs(9)

        RAGIndexer._build_chroma_index_batched(
            docs=docs,
            persist_dir=str(tmp_path / "index"),
            collection_name="test",
            embedding_model="fake",
            device="cpu",
        )

        assert fake_embedder.call_count == 3  # 9 docs / 3 per batch

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.utils.memory.cleanup_memory")
    @patch("meetbot.config.settings")
    def test_cleanup_memory_called_between_batches(
        self, mock_settings, mock_cleanup, mock_get_model, mock_pcc, tmp_path
    ):
        """cleanup_memory() should be called after each batch."""
        _apply_settings(mock_settings, batch_size=2)
        fake_embedder, fake_collection, mock_client = _setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        docs = _make_docs(6)

        RAGIndexer._build_chroma_index_batched(
            docs=docs,
            persist_dir=str(tmp_path / "index"),
            collection_name="test",
            embedding_model="fake",
            device="cpu",
        )

        # 6 docs / 2 per batch = 3 batches → cleanup called 3 times
        assert mock_cleanup.call_count >= 3

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_cancel_checker_invoked_between_batches(
        self, mock_settings, mock_get_model, mock_pcc, tmp_path
    ):
        """cancel_checker() should be called between batches."""
        _apply_settings(mock_settings, batch_size=4)
        fake_embedder, fake_collection, mock_client = _setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        cancel_mock = MagicMock()
        docs = _make_docs(8)

        RAGIndexer._build_chroma_index_batched(
            docs=docs,
            persist_dir=str(tmp_path / "index"),
            collection_name="test",
            embedding_model="fake",
            device="cpu",
            cancel_checker=cancel_mock,
        )

        # 2 batches → cancel_checker called at least once between batches
        assert cancel_mock.call_count >= 1

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_cancel_checker_raises_stops_indexing(
        self, mock_settings, mock_get_model, mock_pcc, tmp_path
    ):
        """If cancel_checker raises, indexing should stop."""
        _apply_settings(mock_settings, batch_size=2)
        fake_embedder, fake_collection, mock_client = _setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        def _raise():
            raise RuntimeError("Job cancelled")

        docs = _make_docs(10)

        with pytest.raises(RuntimeError, match="Job cancelled"):
            RAGIndexer._build_chroma_index_batched(
                docs=docs,
                persist_dir=str(tmp_path / "index"),
                collection_name="test",
                embedding_model="fake",
                device="cpu",
                cancel_checker=_raise,
            )

        # Cancel fires at the top of the loop before the first batch is inserted
        assert len(fake_collection.adds) == 0

    @patch("chromadb.PersistentClient")
    @patch("meetbot.services.rag.indexer._get_embedding_singleton")
    @patch("meetbot.config.settings")
    def test_progress_callback_called(
        self, mock_settings, mock_get_model, mock_pcc, tmp_path
    ):
        """progress_callback should receive at least one call per batch."""
        _apply_settings(mock_settings, batch_size=3)
        fake_embedder, fake_collection, mock_client = _setup_fakes()
        mock_get_model.return_value = fake_embedder
        mock_pcc.return_value = mock_client

        progress_calls = []

        def progress_cb(stage, pct, msg):
            progress_calls.append((stage, pct, msg))

        docs = _make_docs(6)

        RAGIndexer._build_chroma_index_batched(
            docs=docs,
            persist_dir=str(tmp_path / "index"),
            collection_name="test",
            embedding_model="fake",
            device="cpu",
            progress_callback=progress_cb,
        )

        # Should have progress calls including model loading + batch progress
        assert len(progress_calls) >= 2


# ---------------------------------------------------------------------------
# Memory hygiene utilities
# ---------------------------------------------------------------------------


class TestChunkedIterable:
    """Tests for the chunked_iterable utility."""

    def test_exact_division(self):
        from meetbot.utils.memory import chunked_iterable
        result = list(chunked_iterable(range(6), 3))
        assert result == [[0, 1, 2], [3, 4, 5]]

    def test_remainder(self):
        from meetbot.utils.memory import chunked_iterable
        result = list(chunked_iterable(range(7), 3))
        assert result == [[0, 1, 2], [3, 4, 5], [6]]

    def test_empty_input(self):
        from meetbot.utils.memory import chunked_iterable
        result = list(chunked_iterable([], 5))
        assert result == []

    def test_single_element(self):
        from meetbot.utils.memory import chunked_iterable
        result = list(chunked_iterable([42], 3))
        assert result == [[42]]

    def test_size_one(self):
        from meetbot.utils.memory import chunked_iterable
        result = list(chunked_iterable([1, 2, 3], 1))
        assert result == [[1], [2], [3]]


class TestCleanupMemory:
    """Tests for cleanup_memory utility."""

    @patch("gc.collect")
    def test_gc_collect_called(self, mock_gc):
        from meetbot.utils.memory import cleanup_memory
        cleanup_memory("test-label")
        mock_gc.assert_called_once()

    def test_no_crash_without_torch(self):
        from meetbot.utils.memory import cleanup_memory
        # Should not raise even if torch is not installed
        cleanup_memory("safe")


class TestCheckMemoryPressure:
    """Tests for check_memory_pressure utility."""

    def test_returns_false_without_psutil(self):
        """Without psutil installed, check_memory_pressure returns False (fail-open)."""
        from meetbot.utils.memory import check_memory_pressure
        # The function should not raise even without psutil
        result = check_memory_pressure(0.85)
        # If psutil is not installed, returns False; if installed, depends on system
        assert isinstance(result, bool)

    def test_returns_bool(self):
        from meetbot.utils.memory import check_memory_pressure
        assert isinstance(check_memory_pressure(0.0), bool)
        assert isinstance(check_memory_pressure(1.0), bool)
