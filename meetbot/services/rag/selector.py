"""
Answer-embedding source selector for RAG pipeline.

After the LLM generates an answer, computes the answer embedding and
filters candidates by cosine similarity to the answer.  This ensures
only genuinely relevant sources are shown to the user, not just
topographically close chunks.

Uses the same embedding model as indexing for consistency.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Selector:
    """
    Answer-aware source selector.

    Filters candidate source chunks by cosine similarity between the
    answer embedding and each candidate's embedding.  If no candidate
    passes the threshold, falls back to the top 1-2 candidates.

    Parameters
    ----------
    threshold : float
        Minimum cosine similarity for inclusion (default 0.60).
    max_return : int
        Maximum number of sources to return (default 5).
    fallback_count : int
        Number of fallback sources if none pass threshold (default 2).
    """

    def __init__(
        self,
        threshold: float = 0.60,
        max_return: int = 5,
        fallback_count: int = 2,
    ):
        self.threshold = threshold
        self.max_return = max_return
        self.fallback_count = fallback_count

    def filter_by_answer_similarity(
        self,
        answer_text: str,
        candidates: List[Dict[str, Any]],
        embedding_model: str,
        device: str = "cpu",
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Filter candidates by similarity to the generated answer.

        Embeds the answer and all candidate texts in a single batch,
        computes cosine similarity, and filters by threshold.

        Args:
            answer_text: The fully accumulated LLM answer.
            candidates: List of source dicts with "text" key.
            embedding_model: Model name/path for embedding.
            device: Compute device.

        Returns:
            (filtered_sources, filtering_available)
            - filtered_sources: enriched with "answer_relevance_pct"
            - filtering_available: False if embedding failed (returns
              candidates unchanged)
        """
        if not answer_text.strip() or not candidates:
            return candidates, True

        try:
            from ..query_service import _get_embedding_model

            embedder = _get_embedding_model(embedding_model, device)

            # Batch embed: answer + all candidates
            texts = [answer_text] + [c.get("text", "") for c in candidates]
            all_embeddings = embedder.embed_documents(texts)

            answer_emb = np.array(all_embeddings[0], dtype=np.float32)
            a_norm = float(np.linalg.norm(answer_emb))
            if a_norm > 1e-9:
                answer_emb /= a_norm

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for i, cand in enumerate(candidates):
                doc_emb = np.array(all_embeddings[i + 1], dtype=np.float32)
                d_norm = float(np.linalg.norm(doc_emb))
                if d_norm > 1e-9:
                    doc_emb /= d_norm
                sim = float(np.dot(answer_emb, doc_emb))
                scored.append((sim, cand))

            # Sort descending by similarity
            scored.sort(key=lambda x: x[0], reverse=True)

            logger.info(
                "Selector: answer similarity scores: %s",
                [round(s, 3) for s, _ in scored],
            )

            # Filter by threshold
            filtered = [
                {**cand, "answer_relevance_pct": max(0, round(sim * 100))}
                for sim, cand in scored
                if sim >= self.threshold
            ][:self.max_return]

            # Fallback: always return at least fallback_count sources
            if not filtered and scored:
                filtered = [
                    {**cand, "answer_relevance_pct": max(0, round(sim * 100))}
                    for sim, cand in scored[:self.fallback_count]
                ]
                logger.info(
                    "Selector: no candidate above threshold %.2f; "
                    "keeping %d fallback(s)",
                    self.threshold, len(filtered),
                )
            else:
                logger.info(
                    "Selector: %d/%d candidates selected (threshold=%.2f)",
                    len(filtered), len(candidates), self.threshold,
                )

            logger.info(
                "Final selected sources: %s",
                [
                    {"speaker": s.get("speaker"), "relevance": s.get("answer_relevance_pct")}
                    for s in filtered
                ],
            )

            return filtered, True

        except Exception as exc:
            logger.warning(
                "Selector: answer-similarity filtering failed (%s) — "
                "returning all %d candidates unfiltered",
                exc, len(candidates),
            )
            return candidates, False
