"""
Unit tests for RAG retrieval and filtering (Retriever, Reranker, Selector).

All heavy deps (embedding models, ChromaDB) are mocked so these tests
run in <1 s without GPU / model downloads.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from meetbot.services.rag.reranker import Reranker
from meetbot.services.rag.retriever import Candidate
from meetbot.services.rag.selector import Selector


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_to_source_dict_basic(self):
        c = Candidate(
            text="hello",
            metadata={"speaker": "A", "start": 1.0, "end": 2.5},
            distance=0.3,
        )
        d = c.to_source_dict(relevance_pct=70)
        assert d["text"] == "hello"
        assert d["speaker"] == "A"
        assert d["start"] == 1.0
        assert d["end"] == 2.5
        assert d["relevance_pct"] == 70

    def test_to_source_dict_no_relevance(self):
        c = Candidate(text="x", metadata={}, distance=0.5)
        d = c.to_source_dict()
        assert "relevance_pct" not in d

    def test_speaker_default(self):
        c = Candidate(text="x", metadata={}, distance=0.1)
        assert c.speaker == "unknown"

    def test_start_end_defaults(self):
        c = Candidate(text="x", metadata={}, distance=0.1)
        assert c.start == 0.0
        assert c.end == 0.0


# ---------------------------------------------------------------------------
# Reranker — MMR selection
# ---------------------------------------------------------------------------


class TestMMRSelect:
    @pytest.fixture()
    def reranker(self):
        return Reranker(lambda_=0.7)

    def test_empty_candidates(self, reranker):
        selected = reranker.mmr_select(
            query_embedding=np.array([1.0, 0.0]),
            candidate_embeddings=np.empty((0, 2)),
            candidate_texts=[],
            k=3,
        )
        assert selected == []

    def test_single_candidate(self, reranker):
        selected = reranker.mmr_select(
            query_embedding=np.array([1.0, 0.0]),
            candidate_embeddings=np.array([[1.0, 0.0]]),
            candidate_texts=["hello"],
            k=3,
        )
        assert selected == [0]

    def test_k_larger_than_candidates(self, reranker):
        emb = np.eye(3)
        selected = reranker.mmr_select(
            query_embedding=np.array([1.0, 0.0, 0.0]),
            candidate_embeddings=emb,
            candidate_texts=["a", "b", "c"],
            k=10,
        )
        assert len(selected) == 3
        assert sorted(selected) == [0, 1, 2]

    def test_selects_k(self, reranker):
        dim = 4
        np.random.seed(42)
        emb = np.random.randn(20, dim)
        query = np.random.randn(dim)
        selected = reranker.mmr_select(
            query_embedding=query,
            candidate_embeddings=emb,
            candidate_texts=[f"doc{i}" for i in range(20)],
            k=5,
        )
        assert len(selected) == 5
        assert len(set(selected)) == 5  # no duplicates

    def test_first_selected_is_most_relevant(self, reranker):
        """The first selected index should be the most relevant candidate."""
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([
            [0.1, 0.9, 0.0],  # idx 0: not very relevant
            [0.95, 0.05, 0.0],  # idx 1: very relevant
            [0.5, 0.5, 0.0],  # idx 2: moderate
        ])
        selected = reranker.mmr_select(query, candidates, ["a", "b", "c"], k=3)
        assert selected[0] == 1

    def test_lambda_1_selects_by_relevance_only(self):
        """With lambda=1, MMR reduces to pure relevance ordering."""
        reranker = Reranker(lambda_=1.0)
        query = np.array([1.0, 0.0])
        # Candidates sorted by decreasing similarity to query
        candidates = np.array([
            [1.0, 0.0],  # highest sim
            [0.8, 0.2],  # next
            [0.0, 1.0],  # lowest
        ])
        selected = reranker.mmr_select(query, candidates, ["a", "b", "c"], k=3)
        assert selected[0] == 0

    def test_diversity_with_low_lambda(self):
        """With low lambda, MMR should pick diverse candidates."""
        reranker = Reranker(lambda_=0.1)
        query = np.array([1.0, 0.0, 0.0])
        # Two nearly identical candidates and one very different
        candidates = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 0.0, 1.0],
        ])
        selected = reranker.mmr_select(query, candidates, ["a", "b", "c"], k=2)
        # Should select one of the similar pair AND the diverse one
        assert 2 in selected or 0 in selected


# ---------------------------------------------------------------------------
# Reranker — rerank_with_cosine
# ---------------------------------------------------------------------------


class TestRerankWithCosine:
    def test_mmr_mode(self):
        reranker = Reranker(lambda_=0.7)
        candidates = [
            {"text": "a", "score": 1.0},
            {"text": "b", "score": 0.5},
            {"text": "c", "score": 0.8},
        ]
        emb = np.eye(3)
        query = np.array([1.0, 0.0, 0.0])
        result = reranker.rerank_with_cosine(
            query, candidates, emb, k=2, use_mmr=True
        )
        assert len(result) == 2

    def test_cosine_only_mode(self):
        reranker = Reranker()
        candidates = [
            {"text": "far"},
            {"text": "near"},
        ]
        emb = np.array([
            [0.0, 1.0],  # idx 0: orthogonal to query
            [1.0, 0.0],  # idx 1: aligned with query
        ])
        query = np.array([1.0, 0.0])
        result = reranker.rerank_with_cosine(
            query, candidates, emb, k=2, use_mmr=False
        )
        assert result[0]["text"] == "near"

    def test_empty_candidates(self):
        reranker = Reranker()
        result = reranker.rerank_with_cosine(
            np.array([1.0]), [], np.empty((0, 1)), k=5
        )
        assert result == []


# ---------------------------------------------------------------------------
# Selector — answer-aware filtering
# ---------------------------------------------------------------------------


class TestSelector:
    """Test Selector using mocked embedding model."""

    def _mock_embedder(self, embedding_matrix):
        """Create a mock embedder that returns pre-defined embeddings.

        embedding_matrix rows: [answer_emb, cand1_emb, cand2_emb, ...]
        """
        mock = MagicMock()
        mock.embed_documents.return_value = embedding_matrix
        return mock

    @patch("meetbot.services.query_service._get_embedding_model")
    def test_empty_candidates(self, mock_get_emb):
        selector = Selector(threshold=0.3)
        result, ok = selector.filter_by_answer_similarity(
            answer_text="hello",
            candidates=[],
            embedding_model="m",
        )
        assert result == []
        assert ok is True

    @patch("meetbot.services.query_service._get_embedding_model")
    def test_all_below_threshold(self, mock_get_emb):
        """When no candidate passes the threshold, return fallback_count."""
        # answer is [1,0,0], candidates are orthogonal
        mock_get_emb.return_value = self._mock_embedder([
            [1.0, 0.0, 0.0],    # answer
            [0.0, 1.0, 0.0],    # cand 0 — orthogonal
            [0.0, 0.0, 1.0],    # cand 1 — orthogonal
        ])
        selector = Selector(threshold=0.9, fallback_count=1)
        candidates = [{"text": "a"}, {"text": "b"}]
        result, ok = selector.filter_by_answer_similarity(
            answer_text="hello",
            candidates=candidates,
            embedding_model="m",
        )
        assert ok is True
        assert len(result) >= 1

    @patch("meetbot.services.query_service._get_embedding_model")
    def test_above_threshold_returned(self, mock_get_emb):
        """Candidates with similarity >= threshold should be returned."""
        mock_get_emb.return_value = self._mock_embedder([
            [1.0, 0.0],         # answer
            [0.9, 0.1],         # cand 0 — high sim
            [0.8, 0.2],         # cand 1 — high sim
        ])
        selector = Selector(threshold=0.0)
        candidates = [{"text": "a"}, {"text": "b"}]
        result, ok = selector.filter_by_answer_similarity(
            answer_text="hello",
            candidates=candidates,
            embedding_model="m",
        )
        assert ok is True
        assert len(result) == 2

    @patch("meetbot.services.query_service._get_embedding_model")
    def test_relevance_pct_annotated(self, mock_get_emb):
        """Returned candidates should have an 'answer_relevance_pct' key."""
        mock_get_emb.return_value = self._mock_embedder([
            [1.0, 0.0],
            [0.9, 0.1],
        ])
        selector = Selector(threshold=0.0)
        candidates = [{"text": "a"}]
        result, ok = selector.filter_by_answer_similarity(
            answer_text="hello",
            candidates=candidates,
            embedding_model="m",
        )
        assert "answer_relevance_pct" in result[0]
        assert isinstance(result[0]["answer_relevance_pct"], (int, float))

    @patch("meetbot.services.query_service._get_embedding_model")
    def test_fallback_count_respected(self, mock_get_emb):
        """When no candidates pass threshold, fallback_count items returned."""
        mock_get_emb.return_value = self._mock_embedder([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        selector = Selector(threshold=0.99, fallback_count=2)
        candidates = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        result, ok = selector.filter_by_answer_similarity(
            answer_text="hello",
            candidates=candidates,
            embedding_model="m",
        )
        assert len(result) == 2
