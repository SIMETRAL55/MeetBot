"""
Integration-style tests for the multilevel atomic reindex flow.

Verifies that build_multilevel_index_atomic:
- Performs an atomic temp→live directory swap
- Writes .index_meta.json with total doc count
- Cleans up the old directory
- Returns the live directory path

_build_chroma_index_batched is mocked to avoid embedding-model dependencies.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from meetbot.services.rag.indexer import RAGIndexer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _mock_build_creates_dir(**kwargs):
    """Simulate _build_chroma_index_batched by consuming docs and creating dir."""
    docs_iter = kwargs.get("docs", [])
    consumed = list(docs_iter)  # consume generator to get count
    persist_dir = kwargs["persist_dir"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    (Path(persist_dir) / "chroma.sqlite3").write_text("fake-db")
    return len(consumed)


def _segments(n: int = 3):
    return [
        {
            "text": f"Segment {i}.",
            "speaker": f"SPK_{i % 2:02d}",
            "start": float(i * 5),
            "end": float(i * 5 + 4),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Atomic swap behaviour
# ---------------------------------------------------------------------------


class TestMultilevelAtomicSwap:

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_live_dir_created(self, _mock, tmp_path):
        result = RAGIndexer.build_multilevel_index_atomic(
            segments=_segments(),
            persist_root=str(tmp_path),
            collection_name="testjob",
            embedding_model="fake-model",
            job_id="testjob",
        )
        live = Path(result)
        assert live.exists()
        assert live.is_dir()

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_no_temp_dir_left_behind(self, _mock, tmp_path):
        RAGIndexer.build_multilevel_index_atomic(
            segments=_segments(),
            persist_root=str(tmp_path),
            collection_name="testjob2",
            embedding_model="fake-model",
            job_id="testjob2",
        )
        # .tmp directory must be gone after successful build
        temp_dirs = list(tmp_path.glob("*.tmp"))
        assert temp_dirs == [], f"Unexpected .tmp dirs: {temp_dirs}"

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_no_old_dir_left_behind(self, _mock, tmp_path):
        # Run twice so there's an old dir to clean up on the second run
        for _ in range(2):
            RAGIndexer.build_multilevel_index_atomic(
                segments=_segments(),
                persist_root=str(tmp_path),
                collection_name="testjob3",
                embedding_model="fake-model",
                job_id="testjob3",
            )

        old_dirs = list(tmp_path.glob("*.old"))
        assert old_dirs == [], f"Unexpected .old dirs: {old_dirs}"

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_meta_file_written(self, _mock, tmp_path):
        result = RAGIndexer.build_multilevel_index_atomic(
            segments=_segments(3),
            persist_root=str(tmp_path),
            collection_name="testjob4",
            embedding_model="fake-model",
            job_id="testjob4",
        )
        meta_path = Path(result) / ".index_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "n_docs" in meta
        # 1 doc + 3 segments + N chunks — must be > 3
        assert meta["n_docs"] > 3

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_returns_string_path(self, _mock, tmp_path):
        result = RAGIndexer.build_multilevel_index_atomic(
            segments=_segments(),
            persist_root=str(tmp_path),
            collection_name="testjob5",
            embedding_model="fake-model",
        )
        assert isinstance(result, str)

    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_collection_name_matches_requested(self, _mock, tmp_path):
        result = RAGIndexer.build_multilevel_index_atomic(
            segments=_segments(),
            persist_root=str(tmp_path),
            collection_name="mycollection",
            embedding_model="fake-model",
        )
        # The live dir name should match the collection name
        assert Path(result).name == "mycollection"


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestMultilevelProgressCallback:
    @patch("meetbot.services.rag.indexer.RAGIndexer._build_chroma_index_batched",
           side_effect=_mock_build_creates_dir)
    def test_progress_callback_called(self, _mock, tmp_path):
        calls = []

        def cb(stage, pct, msg):
            calls.append((stage, pct, msg))

        RAGIndexer.build_multilevel_index_atomic(
            segments=_segments(2),
            persist_root=str(tmp_path),
            collection_name="cb_test",
            embedding_model="fake-model",
            progress_callback=cb,
        )

        assert len(calls) > 0
        # At least one call should mark 100 % completion
        max_pct = max(pct for _, pct, _ in calls)
        assert max_pct == 100
