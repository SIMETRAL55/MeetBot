"""
Query intent detection for multi-level RAG retrieval.

Determines which embedding level (document / segment / chunk) to search
based on simple keyword heuristics applied to the user query.

Design rationale
----------------
Intent detection maps a query to the most appropriate granularity:

- **document** — broad summarisation queries benefit from the single
  document-level vector that encodes the entire transcript.  Precise
  factual retrieval would be *worse* with this level because a single
  vector cannot represent every detail.

- **segment** — speaker/time queries need per-utterance vectors so that
  exact speaker labels and timestamps are preserved.  Chunk-level vectors
  blend multiple speakers and lose per-turn precision.

- **chunk** — the default fallback.  Overlapping chunks with MMR reranking
  produce the best trade-off between coverage and context density for
  open-ended factual questions.

Thresholds / heuristics are intentionally simple so they are easy to read,
test, and extend.  A production system could replace this with a small
intent-classification model if accuracy becomes critical.
"""

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

# The three retrieval levels
RetrievalLevel = Literal["document", "segment", "chunk"]

# ── Keyword lists ──────────────────────────────────────────────────────────
_DOCUMENT_PATTERNS = [
    r"\bsummariz",          # summarize / summarise / summary
    r"\bsummary\b",
    r"\boverview\b",
    r"\boverall\b",
    r"\bgive me a (brief|short|quick|full)?\s*(summary|overview|recap)",
    r"\bwhat was discussed\b",
    r"\bwhat (were|are) the (main|key|primary|major) (points|topics|themes|issues)",
    r"\bwhat happened\b",
    r"\brecap\b",
    r"\bhigh.level\b",
    r"\bmain points?\b",
    r"\bkey (points?|takeaways?|highlights?)\b",
    # Japanese
    r"要約",
    r"まとめ",
    r"概要",
    r"全体",
]

_SEGMENT_PATTERNS = [
    r"\bwho said\b",
    r"\bwho mentioned\b",
    r"\bwhen did\b",
    r"\bdid .{1,30} say\b",
    r"\bwhat did .{1,30} say\b",
    r"\b(at what|which) time\b",
    r"\btimestamp\b",
    r"\bspeaker\b",
    r"\bwhich speaker\b",
    r"\bwho spoke\b",
    r"\bwho talked about\b",
    r"\bmentioned by\b",
    r"\bsaid by\b",
    r"\baccording to\b",
    # Japanese
    r"言った",
    r"話した",
    r"誰が",
    r"いつ",
]

# Pre-compile patterns for speed
_DOCUMENT_RE = [re.compile(p, re.IGNORECASE) for p in _DOCUMENT_PATTERNS]
_SEGMENT_RE = [re.compile(p, re.IGNORECASE) for p in _SEGMENT_PATTERNS]


def detect_intent(query: str) -> RetrievalLevel:
    """
    Classify query intent and return the appropriate retrieval level.

    Decision tree
    -------------
    1. Check document-level patterns → ``"document"``
    2. Check segment-level patterns  → ``"segment"``
    3. Default                       → ``"chunk"``

    Parameters
    ----------
    query : str
        The raw user query string.

    Returns
    -------
    RetrievalLevel
        One of ``"document"``, ``"segment"``, or ``"chunk"``.
    """
    q = query.strip()

    for pat in _DOCUMENT_RE:
        if pat.search(q):
            logger.info("IntentDetector: document-level (matched: %r)", pat.pattern)
            return "document"

    for pat in _SEGMENT_RE:
        if pat.search(q):
            logger.info("IntentDetector: segment-level (matched: %r)", pat.pattern)
            return "segment"

    logger.info("IntentDetector: chunk-level (default fallback)")
    return "chunk"
