"""
Unit tests for QueryService RAG source-filtering and document-count helpers.

Tests are designed to run without loading any real embedding model or LLM.
All heavy dependencies (HuggingFaceEmbeddings, chromadb) are mocked.
"""

import types
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from meetbot.services.query_service import QueryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(text: str, speaker: str = "SPK_00", start: float = 0.0) -> Dict[str, Any]:
    """Minimal source dict as produced by query_stream step 2."""
    return {
        "text": text,
        "speaker": speaker,
        "start": start,
        "end": start + 5.0,
        "audio_file": "test.mp3",
        "relevance_pct": 80,
    }


def _unit_vec(dim: int, nonzero_idx: int) -> List[float]:
    """Return a unit vector with 1.0 at nonzero_idx and 0 elsewhere."""
    v = [0.0] * dim
    v[nonzero_idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# _score_to_relevance_pct
# ---------------------------------------------------------------------------

class TestScoreToRelevancePct:
    def test_zero_distance_is_100(self):
        assert QueryService._score_to_relevance_pct(0.0) == 100

    def test_half_distance(self):
        assert QueryService._score_to_relevance_pct(0.5) == 50

    def test_negative_score_is_none(self):
        assert QueryService._score_to_relevance_pct(-1.0) is None

    def test_clamped_at_zero_for_high_distance(self):
        assert QueryService._score_to_relevance_pct(1.5) == 0

    def test_full_distance_is_zero(self):
        assert QueryService._score_to_relevance_pct(1.0) == 0


# ---------------------------------------------------------------------------
# _filter_sources_by_answer_similarity (mocking embedding model)
# ---------------------------------------------------------------------------

class TestFilterSourcesByAnswerSimilarity:
    """Tests for the post-answer source filtering step."""

    DIM = 4  # small embedding dim for tests

    def _make_embedder_mock(self, embeddings: List[List[float]]) -> MagicMock:
        """
        Return a mock HuggingFaceEmbeddings whose embed_documents returns
        the given list in order.
        """
        mock = MagicMock()
        mock.embed_documents.return_value = embeddings
        return mock

    def test_high_similarity_sources_are_kept(self):
        """Sources whose text is very similar to the answer are returned."""
        # answer vector: identical to doc 0's vector
        answer_emb = _unit_vec(self.DIM, 0)  # [1, 0, 0, 0]
        doc0_emb   = _unit_vec(self.DIM, 0)  # sim = 1.0  → kept
        doc1_emb   = _unit_vec(self.DIM, 3)  # sim = 0.0  → dropped (threshold=0.3)

        mock_embedder = self._make_embedder_mock([answer_emb, doc0_emb, doc1_emb])

        candidates = [
            _make_source("highly relevant text", speaker="SPK_A", start=0.0),
            _make_source("unrelated text",       speaker="SPK_B", start=10.0),
        ]

        with patch(
            "meetbot.services.query_service._get_embedding_model",
            return_value=mock_embedder,
        ):
            filtered, ok = QueryService._filter_sources_by_answer_similarity(
                answer_text="answer text",
                candidates=candidates,
                embedding_model="dummy-model",
                threshold=0.30,
                max_return=5,
            )

        assert ok is True
        assert len(filtered) == 1
        assert filtered[0]["speaker"] == "SPK_A"
        assert "answer_relevance_pct" in filtered[0]
        assert filtered[0]["answer_relevance_pct"] == 100

    def test_fallback_keeps_best_when_nothing_above_threshold(self):
        """If no candidate exceeds threshold, the top-1 is returned as fallback."""
        answer_emb = _unit_vec(self.DIM, 0)
        doc0_emb   = _unit_vec(self.DIM, 3)  # sim=0 < threshold
        doc1_emb   = _unit_vec(self.DIM, 2)  # sim=0 < threshold

        mock_embedder = self._make_embedder_mock([answer_emb, doc0_emb, doc1_emb])

        candidates = [
            _make_source("doc A", start=0.0),
            _make_source("doc B", start=5.0),
        ]

        with patch(
            "meetbot.services.query_service._get_embedding_model",
            return_value=mock_embedder,
        ):
            filtered, ok = QueryService._filter_sources_by_answer_similarity(
                answer_text="some answer",
                candidates=candidates,
                embedding_model="dummy",
                threshold=0.50,
                max_return=5,
            )

        assert ok is True
        assert len(filtered) == 1  # fallback: exactly one returned

    def test_max_return_caps_results(self):
        """max_return is respected even when many candidates are similar."""
        # All docs identical to answer → all have sim=1.0
        answer_emb = _unit_vec(self.DIM, 0)
        doc_embs   = [_unit_vec(self.DIM, 0)] * 6  # 6 identical docs

        mock_embedder = self._make_embedder_mock([answer_emb] + doc_embs)
        candidates = [_make_source(f"doc {i}", start=float(i)) for i in range(6)]

        with patch(
            "meetbot.services.query_service._get_embedding_model",
            return_value=mock_embedder,
        ):
            filtered, ok = QueryService._filter_sources_by_answer_similarity(
                answer_text="answer",
                candidates=candidates,
                embedding_model="dummy",
                threshold=0.10,
                max_return=3,
            )

        assert ok is True
        assert len(filtered) == 3

    def test_embedding_failure_returns_fallback(self):
        """When embedding computation raises, all candidates are returned with ok=False."""
        candidates = [_make_source("doc A"), _make_source("doc B")]

        with patch(
            "meetbot.services.query_service._get_embedding_model",
            side_effect=RuntimeError("model unavailable"),
        ):
            filtered, ok = QueryService._filter_sources_by_answer_similarity(
                answer_text="answer",
                candidates=candidates,
                embedding_model="dummy",
                threshold=0.30,
                max_return=5,
            )

        assert ok is False
        assert filtered == candidates  # original list returned unchanged

    def test_empty_candidates_returns_empty(self):
        filtered, ok = QueryService._filter_sources_by_answer_similarity(
            answer_text="answer",
            candidates=[],
            embedding_model="dummy",
        )
        assert filtered == []
        assert ok is True

    def test_empty_answer_returns_candidates_unchanged(self):
        """Blank answer skips filtering and returns candidates as-is."""
        candidates = [_make_source("doc A")]
        filtered, ok = QueryService._filter_sources_by_answer_similarity(
            answer_text="   ",
            candidates=candidates,
            embedding_model="dummy",
        )
        assert filtered == candidates
        assert ok is True

    def test_answer_relevance_pct_enriched(self):
        """Returned dicts carry answer_relevance_pct; original keys are preserved."""
        answer_emb = _unit_vec(self.DIM, 0)
        doc_emb    = _unit_vec(self.DIM, 0)  # sim=1.0

        mock_embedder = self._make_embedder_mock([answer_emb, doc_emb])
        candidates = [_make_source("relevant", speaker="SPK_X", start=3.0)]

        with patch(
            "meetbot.services.query_service._get_embedding_model",
            return_value=mock_embedder,
        ):
            filtered, ok = QueryService._filter_sources_by_answer_similarity(
                answer_text="answer",
                candidates=candidates,
                embedding_model="dummy",
                threshold=0.0,
                max_return=5,
            )

        assert ok is True
        assert len(filtered) == 1
        result = filtered[0]
        # Original fields preserved
        assert result["speaker"] == "SPK_X"
        assert result["start"] == 3.0
        assert result["text"] == "relevant"
        # New field added
        assert result["answer_relevance_pct"] == 100


# ---------------------------------------------------------------------------
# count_documents
# ---------------------------------------------------------------------------

class TestCountDocuments:
    def test_returns_zero_for_nonexistent_path(self, tmp_path):
        count = QueryService.count_documents(str(tmp_path / "nonexistent"))
        assert count == 0

    def test_returns_collection_count(self, tmp_path):
        import sys
        import types as _types

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        fake_chromadb = _types.SimpleNamespace(
            PersistentClient=MagicMock(return_value=mock_client)
        )
        with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
            count = QueryService.count_documents(str(tmp_path))

        assert count == 42
        mock_collection.count.assert_called_once()

    def test_returns_zero_on_exception(self, tmp_path):
        import sys, types as _types

        broken_chromadb = _types.SimpleNamespace(
            PersistentClient=MagicMock(side_effect=RuntimeError("corrupt db"))
        )
        with patch.dict(sys.modules, {"chromadb": broken_chromadb}):
            count = QueryService.count_documents(str(tmp_path))
        assert count == 0
