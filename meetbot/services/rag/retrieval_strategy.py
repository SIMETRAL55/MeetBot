"""
Retrieval strategy router for MeetBot.

Provides a ``RetrievalMethod`` enum and a ``route_retrieval()`` function that
dispatches queries to either the existing vector-based (ChromaDB) pipeline or
the new PageIndex tree-search pipeline.

Both pipelines return ``RetrievalResult`` objects with a unified interface so
the downstream LLM generation step can consume them identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class RetrievalMethod(str, Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"        # Existing ChromaDB pipeline
    PAGEINDEX = "pageindex"  # VectifyAI PageIndex tree search


@dataclass
class RetrievalResult:
    """Unified result format across retrieval methods."""
    text: str
    metadata: Dict[str, Any]
    score: float                  # 0-1, higher = more relevant
    retrieval_method: str         # "vector" or "pageindex"
    source_ref: str               # e.g. "chunk_42" or "node_0003"
