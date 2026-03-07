"""
Unit tests for the atomic reindex pipeline.

Tests verify:
- Atomic swap directory management (temp → live → old cleanup)
- Hash computation for cache invalidation
- Metadata writing
- Error recovery (swap failure → restore old)

Heavy deps (embedding model, ChromaDB internals) are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meetbot.services.rag.indexer import RAGIndexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_root(tmp_path):
    """Yield a temporary root directory for index storage."""
    return tmp_path / "db"


@pytest.fixture()
def sample_docs():
    return [
        {"id": "d1", "text": "Hello world", "metadata": {"speaker": "A"}},
        {"id": "d2", "text": "Goodbye", "metadata": {"speaker": "B"}},
    ]


def _mock_build_that_creates_dir(**kwargs):
    """Side-effect for _build_chroma_index_batched that creates the temp dir."""
    docs = kwargs.get("docs", [])
    n = len(list(docs)) if not isinstance(docs, list) else len(docs)
    persist_dir = kwargs["persist_dir"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    (Path(persist_dir) / "chroma.sqlite3").write_text("fake")
    return n


# ---------------------------------------------------------------------------
# compute_hash
# ---------------------------------------------------------------------------


class TestComputeHash:
    def test_deterministic(self, sample_docs):
        h1 = RAGIndexer.compute_hash(sample_docs, "model_a")
        h2 = RAGIndexer.compute_hash(sample_docs, "model_a")
        assert h1 == h2

    def test_different_model_changes_hash(self, sample_docs):
        h1 = RAGIndexer.compute_hash(sample_docs, "model_a")
        h2 = RAGIndexer.compute_hash(sample_docs, "model_b")
        assert h1 != h2

    def test_different_docs_changes_hash(self):
        d1 = [{"text": "foo"}]
        d2 = [{"text": "bar"}]
        h1 = RAGIndexer.compute_hash(d1, "m")
        h2 = RAGIndexer.compute_hash(d2, "m")
        assert h1 != h2

    def test_empty_docs(self):
        h = RAGIndexer.compute_hash([], "m")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex digest


# ---------------------------------------------------------------------------
# build_index_atomic — directory management
# ---------------------------------------------------------------------------


class TestBuildIndexAtomic:
    """Test the swap logic by mocking the actual Chroma build step."""

    @patch.object(RAGIndexer, "_build_chroma_index_batched", side_effect=_mock_build_that_creates_dir)
    def test_creates_live_dir(self, mock_build, tmp_root, sample_docs):
        """After a successful build, the live directory should exist."""
        result = RAGIndexer.build_index_atomic(
            docs=sample_docs,
            persist_root=str(tmp_root),
            collection_name="test_coll",
            embedding_model="fake_model",
        )
        live_dir = Path(result)
        assert live_dir.exists()
        assert live_dir.name == "test_coll"

    @patch.object(RAGIndexer, "_build_chroma_index_batched", side_effect=_mock_build_that_creates_dir)
    def test_temp_dir_cleaned_up(self, mock_build, tmp_root, sample_docs):
        """The .tmp directory should not exist after success."""
        RAGIndexer.build_index_atomic(
            docs=sample_docs,
            persist_root=str(tmp_root),
            collection_name="test_coll",
            embedding_model="fake_model",
        )
        temp_dir = tmp_root / "test_coll.tmp"
        assert not temp_dir.exists()

    @patch.object(RAGIndexer, "_build_chroma_index_batched", side_effect=_mock_build_that_creates_dir)
    def test_old_dir_cleaned_up_after_swap(self, mock_build, tmp_root, sample_docs):
        """When replacing an existing live index, the .old dir should be removed."""
        live_dir = tmp_root / "test_coll"
        live_dir.mkdir(parents=True)
        (live_dir / "dummy.txt").write_text("old content")

        RAGIndexer.build_index_atomic(
            docs=sample_docs,
            persist_root=str(tmp_root),
            collection_name="test_coll",
            embedding_model="fake_model",
        )

        old_dir = tmp_root / "test_coll.old"
        assert not old_dir.exists()
        assert live_dir.exists()

    @patch.object(RAGIndexer, "_build_chroma_index_batched", side_effect=_mock_build_that_creates_dir)
    def test_metadata_written(self, mock_build, tmp_root, sample_docs):
        """The .index_meta.json file should exist and be valid JSON."""
        RAGIndexer.build_index_atomic(
            docs=sample_docs,
            persist_root=str(tmp_root),
            collection_name="test_coll",
            embedding_model="my_model",
            job_id="abc123",
            version=3,
        )
        meta_path = tmp_root / "test_coll" / ".index_meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["model"] == "my_model"
        assert meta["n_docs"] == len(sample_docs)
        assert meta["job_id"] == "abc123"
        assert meta["version"] == 3
        assert "hash" in meta
        assert "created_at" in meta

    @patch.object(RAGIndexer, "_build_chroma_index_batched", side_effect=_mock_build_that_creates_dir)
    def test_progress_callback_called(self, mock_build, tmp_root, sample_docs):
        """The optional progress callback should receive calls."""
        calls = []

        def cb(stage, pct, msg):
            calls.append((stage, pct, msg))

        RAGIndexer.build_index_atomic(
            docs=sample_docs,
            persist_root=str(tmp_root),
            collection_name="pc",
            embedding_model="m",
            progress_callback=cb,
        )
        assert len(calls) >= 2

    def test_empty_docs_raises(self, tmp_root):
        with pytest.raises(ValueError, match="No documents"):
            RAGIndexer.build_index_atomic(
                docs=[],
                persist_root=str(tmp_root),
                collection_name="x",
                embedding_model="m",
            )

    @patch.object(RAGIndexer, "_build_chroma_index_batched", side_effect=RuntimeError("boom"))
    def test_build_failure_cleans_temp(self, mock_build, tmp_root, sample_docs):
        """If _build_chroma_index_batched fails, the temp dir should be cleaned up."""
        with pytest.raises(RuntimeError, match="Failed to build temp index"):
            RAGIndexer.build_index_atomic(
                docs=sample_docs,
                persist_root=str(tmp_root),
                collection_name="fail_coll",
                embedding_model="m",
            )
        temp_dir = tmp_root / "fail_coll.tmp"
        assert not temp_dir.exists()

    @patch.object(RAGIndexer, "_build_chroma_index_batched", side_effect=_mock_build_that_creates_dir)
    def test_leftover_temp_cleaned_before_build(self, mock_build, tmp_root, sample_docs):
        """If a .tmp dir exists from a previous failed run, it's cleaned up."""
        leftover = tmp_root / "test_coll.tmp"
        leftover.mkdir(parents=True)
        (leftover / "stale.txt").write_text("stale")

        RAGIndexer.build_index_atomic(
            docs=sample_docs,
            persist_root=str(tmp_root),
            collection_name="test_coll",
            embedding_model="m",
        )
        assert (tmp_root / "test_coll").exists()
        assert not leftover.exists()


# ---------------------------------------------------------------------------
# get_doc_count
# ---------------------------------------------------------------------------


class TestGetDocCount:
    def test_nonexistent_dir_returns_zero(self, tmp_root):
        count = RAGIndexer.get_doc_count(str(tmp_root / "nope"))
        assert count == 0
