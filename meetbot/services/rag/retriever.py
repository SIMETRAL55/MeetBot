"""
ANN recall retriever for RAG pipeline.

Abstracts ChromaDB vector search, returning candidate documents with
embeddings and metadata for downstream reranking/selection.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A retrieved candidate document with metadata and embedding."""

    text: str
    metadata: Dict[str, Any]
    distance: float  # lower = more similar
    embedding: Optional[np.ndarray] = None

    @property
    def speaker(self) -> str:
        return self.metadata.get("speaker", "unknown")

    @property
    def start(self) -> float:
        return float(self.metadata.get("start", 0))

    @property
    def end(self) -> float:
        return float(self.metadata.get("end", 0))

    def to_source_dict(self, relevance_pct: Optional[int] = None) -> Dict[str, Any]:
        """Convert to the source dict format expected by the UI."""
        d = {
            "text": self.text,
            "audio_file": self.metadata.get("audio_file", ""),
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
        }
        if relevance_pct is not None:
            d["relevance_pct"] = relevance_pct
        return d


class Retriever:
    """
    ANN recall retriever using ChromaDB.

    Loads the vectorstore once per query and retrieves the top-N most
    similar documents to the query.  Returns Candidate objects with
    metadata and optionally embeddings for downstream processing.

    Parameters
    ----------
    embedding_model : str
        Path or name of the embedding model.
    device : str
        Compute device for embeddings ("cpu" or "cuda").
    """

    def __init__(self, embedding_model: str, device: str = "cpu"):
        self.embedding_model = embedding_model
        self.device = device

    def recall(
        self,
        query: str,
        db_dir: str,
        top_n: int = 50,
        include_embeddings: bool = False,
        level: Optional[str] = None,
    ) -> List[Candidate]:
        """
        Retrieve top-N candidates from the vector store.

        Args:
            query: The user query string.
            db_dir: Path to the ChromaDB persist directory.
            top_n: Maximum number of candidates to retrieve.
            include_embeddings: If True, fetch embeddings for MMR.
            level: Optional metadata filter (``"document"``, ``"segment"``,
                   or ``"chunk"``).  When specified, only vectors with
                   matching ``level`` metadata are returned.  Falls back to
                   an un-filtered search if the level filter yields no
                   results (backward-compat with pre-multilevel indexes).

        Returns:
            List of Candidate objects sorted by relevance (best first).
        """
        from pathlib import Path

        db_path = Path(db_dir)
        if not db_path.exists():
            raise FileNotFoundError(f"Vector database not found: {db_dir}")

        collection_name = db_path.name

        # Try LangChain Chroma first
        candidates = self._recall_langchain(
            query, db_dir, collection_name, top_n, include_embeddings, level
        )
        if candidates is not None:
            # Fallback to no-filter if level filter produced 0 results
            if not candidates and level is not None:
                logger.info(
                    "Retriever: level=%r filter returned 0 results — retrying without filter",
                    level,
                )
                candidates = self._recall_langchain(
                    query, db_dir, collection_name, top_n, include_embeddings, None
                )
            if candidates is not None:
                return candidates

        # Fallback to chromadb direct
        candidates = self._recall_chromadb(
            query, db_dir, collection_name, top_n, include_embeddings, level
        )
        if candidates is not None:
            # Fallback to no-filter if level filter produced 0 results
            if not candidates and level is not None:
                logger.info(
                    "Retriever: level=%r filter returned 0 via chromadb — retrying without filter",
                    level,
                )
                candidates = self._recall_chromadb(
                    query, db_dir, collection_name, top_n, include_embeddings, None
                )
            if candidates is not None:
                return candidates

        logger.error("Retriever: all recall methods failed for %s", db_dir)
        return []

    def _recall_langchain(
        self,
        query: str,
        db_dir: str,
        collection_name: str,
        top_n: int,
        include_embeddings: bool,
        level: Optional[str] = None,
    ) -> Optional[List[Candidate]]:
        """Try LangChain Chroma retrieval."""
        try:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            from ..query_service import _get_embedding_model

            embedding = _get_embedding_model(self.embedding_model, self.device)
            vectordb = Chroma(
                persist_directory=db_dir,
                embedding_function=embedding,
                collection_name=collection_name,
            )

            # Apply level filter when specified
            lc_filter = {"level": level} if level is not None else None

            # Get results with scores
            if hasattr(vectordb, "similarity_search_with_score"):
                results = vectordb.similarity_search_with_score(
                    query, k=top_n, filter=lc_filter
                )
            elif hasattr(vectordb, "similarity_search"):
                docs = vectordb.similarity_search(query, k=top_n, filter=lc_filter)
                results = [(d, -1.0) for d in docs]
            else:
                return None

            candidates = []
            for doc, score in results:
                candidates.append(Candidate(
                    text=getattr(doc, "page_content", str(doc)),
                    metadata=getattr(doc, "metadata", {}),
                    distance=score,
                ))

            logger.info(
                "Retriever: recalled %d candidates via LangChain (level=%r)",
                len(candidates), level,
            )
            return candidates

        except Exception as exc:
            logger.warning("LangChain recall failed: %s", exc)
            return None

    def _recall_chromadb(
        self,
        query: str,
        db_dir: str,
        collection_name: str,
        top_n: int,
        include_embeddings: bool,
        level: Optional[str] = None,
    ) -> Optional[List[Candidate]]:
        """Fallback to chromadb direct API."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=db_dir)
            collection = client.get_collection(collection_name)

            include = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")

            # Build where clause for level filter
            where = {"level": {"$eq": level}} if level is not None else None

            query_kwargs = dict(
                query_texts=[query],
                n_results=top_n,
                include=include,
            )
            if where is not None:
                query_kwargs["where"] = where

            results = collection.query(**query_kwargs)

            candidates = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            embs = results.get("embeddings", [[]])[0] if include_embeddings else [None] * len(docs)

            for i, text in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = float(dists[i]) if i < len(dists) else -1.0
                emb = np.array(embs[i], dtype=np.float32) if embs[i] is not None else None

                candidates.append(Candidate(
                    text=text,
                    metadata=meta,
                    distance=dist,
                    embedding=emb,
                ))

            logger.info(
                "Retriever: recalled %d candidates via chromadb (level=%r)",
                len(candidates), level,
            )
            return candidates

        except Exception as exc:
            logger.warning("chromadb recall failed: %s", exc)
            return None

    @staticmethod
    def score_to_relevance_pct(score: float) -> Optional[int]:
        """Convert distance score to 0-100 relevance percentage."""
        if score < 0:
            return None
        return max(0, round((1.0 - score) * 100))
