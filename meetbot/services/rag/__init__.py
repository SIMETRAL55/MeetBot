"""
Production-grade RAG pipeline for MeetBot.

Modules
-------
- chunker    : Speaker-aware transcript chunking with overlap
- indexer    : Atomic-swap ChromaDB indexing
- retriever  : ANN recall from vector store
- reranker   : MMR diversity selection & optional cross-encoder
- selector   : Answer-embedding source filtering
- summarizer : Hierarchical summarization helper
"""

__all__ = [
    "Chunker",
    "RAGIndexer",
    "Retriever",
    "Reranker",
    "Selector",
    "Summarizer",
]
