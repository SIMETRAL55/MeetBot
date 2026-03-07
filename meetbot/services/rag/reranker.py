"""
Maximal Marginal Relevance (MMR) reranker and optional cross-encoder wrapper.

Produces a compact, non-redundant candidate set from a larger pool of ANN
recall results.  Supports:
- Cosine-similarity MMR (default, no extra model needed)
- Optional cross-encoder reranking (stub for future use)

The reranker works on CPU and does not load any additional models unless
cross-encoder mode is explicitly requested.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """
    MMR-based reranker for RAG candidate selection.

    Parameters
    ----------
    lambda_ : float
        Trade-off between relevance and diversity (0 = max diversity,
        1 = max relevance). Default 0.7.
    """

    def __init__(self, lambda_: float = 0.7):
        self.lambda_ = lambda_

    def mmr_select(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_texts: List[str],
        k: int = 6,
    ) -> List[int]:
        """
        Select k candidates using Maximal Marginal Relevance.

        MMR balances relevance to the query with diversity among selected
        candidates, reducing redundancy in the final context.

        Args:
            query_embedding: Query vector (1D numpy array).
            candidate_embeddings: Matrix of candidate vectors (N x D).
            candidate_texts: List of candidate texts (for logging).
            k: Number of candidates to select.

        Returns:
            List of selected indices into the candidate arrays.
        """
        if len(candidate_embeddings) == 0:
            return []

        n = len(candidate_embeddings)
        k = min(k, n)

        # Normalise embeddings
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        c_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9
        c_normed = candidate_embeddings / c_norms

        # Query-candidate similarities
        query_sims = c_normed @ q_norm

        selected: List[int] = []
        remaining = set(range(n))

        for _ in range(k):
            if not remaining:
                break

            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = float(query_sims[idx])

                # Max similarity to already-selected candidates
                if selected:
                    sel_embs = c_normed[selected]
                    sims_to_selected = sel_embs @ c_normed[idx]
                    max_sim_to_selected = float(np.max(sims_to_selected))
                else:
                    max_sim_to_selected = 0.0

                # MMR score
                mmr_score = self.lambda_ * relevance - (1 - self.lambda_) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        logger.info(
            "MMR: selected %d/%d candidates (lambda=%.2f)",
            len(selected), n, self.lambda_,
        )
        return selected

    def rerank_with_cosine(
        self,
        query_embedding: np.ndarray,
        candidates: List[Dict[str, Any]],
        candidate_embeddings: np.ndarray,
        k: int = 6,
        use_mmr: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cosine similarity, optionally with MMR.

        Args:
            query_embedding: Query embedding vector.
            candidates: List of candidate source dicts.
            candidate_embeddings: Embedding matrix for candidates.
            k: Number of candidates to return.
            use_mmr: If True, apply MMR; otherwise just take top-k by cosine.

        Returns:
            Reranked list of candidate dicts (up to k items).
        """
        if not candidates or len(candidate_embeddings) == 0:
            return candidates[:k]

        if use_mmr and len(candidates) > k:
            selected_indices = self.mmr_select(
                query_embedding, candidate_embeddings,
                [c.get("text", "") for c in candidates],
                k=k,
            )
            return [candidates[i] for i in selected_indices]
        else:
            # Simple cosine ranking
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            c_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9
            c_normed = candidate_embeddings / c_norms
            sims = c_normed @ q_norm
            ranked = sorted(range(len(candidates)), key=lambda i: sims[i], reverse=True)
            return [candidates[i] for i in ranked[:k]]
