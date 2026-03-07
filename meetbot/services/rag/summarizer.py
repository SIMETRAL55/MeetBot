"""
Hierarchical summarization helper for RAG pipeline.

Provides a map-reduce style summarization that works well with long
transcripts by:
1. Summarizing individual chunks (map phase)
2. Combining chunk summaries into a final summary (reduce phase)

This is useful for "summarize the meeting" type queries where the
entire document needs to be considered, not just the most relevant chunks.
"""

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Hierarchical transcript summarizer.

    Splits the full transcript into manageable groups, summarizes each,
    then combines the summaries into a final, coherent summary.

    This is a helper — the actual LLM calls are delegated to the caller
    via a generate function.
    """

    def __init__(self, max_chunk_chars: int = 3000, max_combine_chunks: int = 8):
        """
        Args:
            max_chunk_chars: Maximum characters per map-phase chunk.
            max_combine_chunks: Maximum number of chunk summaries to
                combine in one reduce call.
        """
        self.max_chunk_chars = max_chunk_chars
        self.max_combine_chunks = max_combine_chunks

    def build_map_prompts(
        self,
        segments: List[Dict[str, Any]],
        query: str = "Summarize this meeting.",
    ) -> List[str]:
        """
        Build prompts for the map phase (individual chunk summarization).

        Groups segments into chunks of max_chunk_chars and creates a
        summarization prompt for each.

        Args:
            segments: List of transcript segment dicts (text, speaker, start, end).
            query: The user's summarization request.

        Returns:
            List of formatted prompts for the map phase.
        """
        groups = self._group_segments(segments)

        prompts = []
        for i, group in enumerate(groups):
            context = "\n".join(
                f"[{s.get('speaker', '?')} {s.get('start', '?')}-{s.get('end', '?')}] "
                f"{s.get('text', '')}"
                for s in group
            )
            prompt = (
                f"Summarize the following section ({i + 1}/{len(groups)}) of a meeting transcript. "
                f"Focus on key points, decisions, and action items.\n\n"
                f"User request: {query}\n\n"
                f"Transcript section:\n{context}\n\n"
                f"Summary:"
            )
            prompts.append(prompt)

        logger.info(
            "Summarizer: built %d map prompts from %d segments",
            len(prompts), len(segments),
        )
        return prompts

    def build_reduce_prompt(
        self,
        chunk_summaries: List[str],
        query: str = "Summarize this meeting.",
    ) -> str:
        """
        Build a reduce prompt that combines chunk summaries.

        Args:
            chunk_summaries: List of summaries from the map phase.
            query: The user's original request.

        Returns:
            A formatted prompt for the reduce phase.
        """
        combined = "\n\n---\n\n".join(
            f"Section {i + 1} Summary:\n{s}"
            for i, s in enumerate(chunk_summaries)
        )
        prompt = (
            f"The following are summaries of different sections of a meeting transcript. "
            f"Combine them into a single, coherent summary that addresses the user's request.\n\n"
            f"User request: {query}\n\n"
            f"Section summaries:\n{combined}\n\n"
            f"Combined summary:"
        )
        return prompt

    def is_summary_query(self, query: str) -> bool:
        """
        Heuristic to detect if a query is asking for a full-document summary.

        Args:
            query: The user's question.

        Returns:
            True if the query appears to be a summarization request.
        """
        query_lower = query.lower().strip()
        summary_keywords = [
            "summarize", "summary", "summarise", "要約", "まとめ",
            "overview", "key points", "main points", "takeaways",
            "what was discussed", "what happened", "recap",
            "meeting notes", "minutes", "action items",
            "全体", "概要",
        ]
        return any(kw in query_lower for kw in summary_keywords)

    def _group_segments(self, segments: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group segments into chunks based on max_chunk_chars."""
        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        current_chars = 0

        for seg in segments:
            text = seg.get("text", "")
            seg_chars = len(text)

            if current_chars + seg_chars > self.max_chunk_chars and current_group:
                groups.append(current_group)
                current_group = []
                current_chars = 0

            current_group.append(seg)
            current_chars += seg_chars

        if current_group:
            groups.append(current_group)

        return groups
