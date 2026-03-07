"""
Unit tests for multilevel indexing (build_multilevel_index_atomic).

Verifies that the method correctly produces document-level,
segment-level, and chunk-level docs and streams them through
_build_chroma_index_batched.

No real embedding model is invoked — _build_chroma_index_batched is mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from meetbot.services.rag.indexer import RAGIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segments(n: int = 3):
    """Return *n* minimal transcript segment dicts."""
    return [
        {
            "text": f"This is segment {i} spoken by SPEAKER_{i % 2:02d}.",
            "speaker": f"SPEAKER_{i % 2:02d}",
            "start": float(i * 10),
            "end": float(i * 10 + 8),
        }
        for i in range(n)
    ]


# Module-level list to capture docs consumed by the streaming mock
_captured_docs: list = []


def _mock_build_creates_dir(**kwargs):
    """Mock that consumes the docs iterable, stores for inspection, and creates dir."""
    docs_iter = kwargs.get("docs", [])
    consumed = list(docs_iter)
    _captured_docs.clear()
    _captured_docs.extend(consumed)
    persist_dir = kwargs["persist_dir"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    (Path(persist_dir) / "chroma.sqlite3").write_text("fake")
    return len(consumed)


# ---------------------------------------------------------------------------
# Level composition
# ---------------------------------------------------------------------------


class TestMultilevelDocComposition:
    """Verify the correct mix of levels is forwarded to Chroma."""

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_all_three_levels_present(self, mock_build, tmp_path):
        segs = _make_segments(3)
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        levels = [d["metadata"]["level"] for d in all_docs]
        assert "document" in levels
        assert "segment" in levels
        assert "chunk" in levels

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_exactly_one_document_level_doc(self, mock_build, tmp_path):
        segs = _make_segments(4)
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        doc_docs = [d for d in all_docs if d["metadata"]["level"] == "document"]
        assert len(doc_docs) == 1

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_segment_count_matches_input(self, mock_build, tmp_path):
        n = 5
        segs = _make_segments(n)
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        seg_docs = [d for d in all_docs if d["metadata"]["level"] == "segment"]
        assert len(seg_docs) == n

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_chunk_docs_have_level_chunk(self, mock_build, tmp_path):
        segs = _make_segments(3)
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        chunk_docs = [d for d in all_docs if d["metadata"]["level"] == "chunk"]
        assert len(chunk_docs) >= 1
        for d in chunk_docs:
            assert d["metadata"]["level"] == "chunk"


# ---------------------------------------------------------------------------
# Segment-level metadata correctness
# ---------------------------------------------------------------------------


class TestSegmentLevelMetadata:
    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_segment_speaker_preserved(self, mock_build, tmp_path):
        segs = [
            {"text": "Hello", "speaker": "Alice", "start": 0.0, "end": 5.0},
            {"text": "Hi there", "speaker": "Bob", "start": 5.0, "end": 10.0},
        ]
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        seg_docs = [d for d in all_docs if d["metadata"]["level"] == "segment"]
        speakers = {d["metadata"]["speaker"] for d in seg_docs}
        assert "Alice" in speakers
        assert "Bob" in speakers

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_segment_timestamps_preserved(self, mock_build, tmp_path):
        segs = [{"text": "Test", "speaker": "X", "start": 1.5, "end": 3.7}]
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        seg_docs = [d for d in all_docs if d["metadata"]["level"] == "segment"]
        assert seg_docs[0]["metadata"]["start"] == 1.5
        assert seg_docs[0]["metadata"]["end"] == 3.7


# ---------------------------------------------------------------------------
# Document-level content
# ---------------------------------------------------------------------------


class TestDocumentLevelContent:
    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_document_text_contains_all_segments(self, mock_build, tmp_path):
        segs = [
            {"text": "Alpha", "speaker": "A", "start": 0.0, "end": 1.0},
            {"text": "Beta", "speaker": "B", "start": 1.0, "end": 2.0},
            {"text": "Gamma", "speaker": "A", "start": 2.0, "end": 3.0},
        ]
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        doc_doc = next(d for d in all_docs if d["metadata"]["level"] == "document")
        assert "Alpha" in doc_doc["text"]
        assert "Beta" in doc_doc["text"]
        assert "Gamma" in doc_doc["text"]


# ---------------------------------------------------------------------------
# Empty input edge case
# ---------------------------------------------------------------------------


class TestMultilevelEdgeCases:
    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_single_segment(self, mock_build, tmp_path):
        segs = [{"text": "Only sentence.", "speaker": "Solo", "start": 0.0, "end": 2.0}]
        RAGIndexer.build_multilevel_index_atomic(
            segments=segs,
            persist_root=str(tmp_path),
            collection_name="testcol",
            embedding_model="fake-model",
        )

        all_docs = _captured_docs
        levels = {d["metadata"]["level"] for d in all_docs}
        assert "document" in levels
        assert "segment" in levels
        # Chunk level may or may not be present for a single short segment
        # depending on tokenisation — just verify no crash.
