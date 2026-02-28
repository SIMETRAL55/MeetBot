"""Embedding adapters for vector encoding."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each as list of floats)
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate single embedding for query text.

        Args:
            text: Query string to embed

        Returns:
            Single embedding vector
        """
        pass


class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace embedding model wrapper."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize HuggingFace embedding model.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for embedding computation
        """
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise RuntimeError(
                "langchain-huggingface not installed. "
                "Install with: pip install langchain-huggingface"
            )

        logger.info(f"Loading embedding model: {model_name} (device={device})")
        self.model_name = model_name
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        logger.debug(f"Embedding {len(texts)} texts (batch_size={self.batch_size})")
        embeddings = self.embedding.embed_documents(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for query text."""
        logger.debug(f"Embedding query ({len(text)} chars)")
        embedding = self.embedding.embed_query(text)
        return embedding
