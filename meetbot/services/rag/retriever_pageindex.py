"""
PageIndex tree-search retriever for MeetBot.

Implements LLM-driven retrieval over a PageIndex tree structure.  The LLM
receives the tree hierarchy and a query, then returns a list of node IDs that
are most likely to contain the answer.  The retriever maps those node IDs back
to transcript segment ranges and collects the text.

Uses the same OpenAI-compatible endpoint as indexing (env var injection via
``pageindex_env``), so it works with OpenRouter, local Ollama, or OpenAI.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ...config import settings

logger = logging.getLogger(__name__)

# Prompt template for LLM tree search (from PageIndex docs, Section 4.4).
SEARCH_PROMPT = """You are given a query and the tree structure of a meeting transcript.
Find all nodes that are likely to contain the answer to the query.

Query: {query}

PageIndex Tree:
{tree_summary}

Respond with ONLY valid JSON in this exact format (no other text):
{{"thinking": "your brief reasoning", "node_list": ["0001", "0003"]}}
"""


class PageIndexRetriever:
    """LLM-based tree search over a PageIndex structure."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or settings.PAGEINDEX_LLM_MODEL
        self.base_url = base_url or settings.get_pageindex_base_url()
        self.api_key = api_key or settings.PAGEINDEX_LLM_API_KEY or "not-needed"

    async def search(
        self,
        query: str,
        tree: Dict[str, Any],
        segments: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Search the PageIndex tree for nodes relevant to the query.

        Steps:
            1. Build a compact tree summary for the LLM prompt
            2. Send tree + query to LLM via OpenAI-compatible API
            3. Parse returned node_list
            4. Map node IDs -> segment ranges
            5. Collect segment text from matched nodes
            6. Return results with node references

        Args:
            query: User's question.
            tree: The PageIndex tree (loaded from JSON).
            segments: Original transcript segments for text lookup.

        Returns:
            List of source dicts compatible with MeetBot's ChatSource format,
            enriched with node_title and node_id fields.
        """
        # Step 1: Build tree summary
        tree_summary = self._build_tree_summary(tree)

        # Step 2: Call LLM
        node_ids = await self._llm_tree_search(query, tree_summary)

        if not node_ids:
            logger.warning("PageIndex retriever: LLM returned no nodes for query: %s", query[:80])
            return []

        # Step 3-4: Map node IDs to segment indices
        node_map = self._build_node_map(tree)
        matched_segments = []

        for nid in node_ids:
            node = node_map.get(nid)
            if node is None:
                logger.debug("PageIndex retriever: node %s not found in tree", nid)
                continue

            seg_indices = node.get("segment_indices", [])
            node_title = node.get("title", f"Node {nid}")

            for si in seg_indices:
                if 0 <= si < len(segments):
                    seg = segments[si]
                    matched_segments.append({
                        "segment_index": si,
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "speaker": seg.get("speaker", "Unknown"),
                        "text": seg.get("text", ""),
                        "node_title": node_title,
                        "node_id": nid,
                        "distance": 0.0,
                        "relevance": 100,
                    })

        # Deduplicate by segment_index (a segment may appear in multiple nodes)
        seen = set()
        unique = []
        for s in matched_segments:
            if s["segment_index"] not in seen:
                seen.add(s["segment_index"])
                unique.append(s)

        logger.info(
            "PageIndex retriever: query=%s -> %d nodes -> %d segments",
            query[:40], len(node_ids), len(unique),
        )
        return unique

    async def _llm_tree_search(self, query: str, tree_summary: str) -> List[str]:
        """Send tree + query to LLM and parse the node_list response."""
        import openai

        prompt = SEARCH_PROMPT.format(query=query, tree_summary=tree_summary)

        try:
            from ...adapters.llm.openai_adapter import pageindex_env

            with pageindex_env(settings):
                client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )

            content = response.choices[0].message.content or ""
            return self._parse_node_list(content)

        except Exception as exc:
            logger.error("PageIndex LLM tree search failed: %s", exc)
            return []

    def _parse_node_list(self, response_text: str) -> List[str]:
        """Extract node_list from LLM JSON response with fallback parsing."""
        # Try direct JSON parse
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "node_list" in data:
                return [str(n) for n in data["node_list"]]
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "node_list" in data:
                    return [str(n) for n in data["node_list"]]
            except json.JSONDecodeError:
                pass

        # Fallback: find any JSON object in the response
        json_match = re.search(r'\{[^{}]*"node_list"\s*:\s*\[.*?\][^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return [str(n) for n in data.get("node_list", [])]
            except json.JSONDecodeError:
                pass

        logger.warning("PageIndex retriever: could not parse node_list from LLM response")
        return []

    def _build_tree_summary(self, tree: Dict[str, Any], indent: int = 0) -> str:
        """Build a compact text representation of the tree for the LLM prompt."""
        lines = []

        # Root container: real PageIndex uses "structure"; fallback uses "children".
        # Neither key has a node_id, so we unwrap and recurse into children directly.
        if "structure" in tree or ("children" in tree and "node_id" not in tree):
            top_nodes = tree.get("structure", tree.get("children", []))
            for child in top_nodes:
                lines.append(self._build_tree_summary(child, indent))
            return "\n".join(lines)

        # Regular node
        prefix = "  " * indent
        node_id = tree.get("node_id", "?")
        title = tree.get("title", "")
        seg_indices = tree.get("segment_indices", [])

        if title:
            seg_info = f" (segments: {seg_indices[0]}-{seg_indices[-1]})" if seg_indices else ""
            lines.append(f"{prefix}[{node_id}] {title}{seg_info}")

        # Subnodes: real PageIndex uses "nodes"; fallback uses "children"
        for child in tree.get("nodes", tree.get("children", [])):
            lines.append(self._build_tree_summary(child, indent + 1))

        return "\n".join(lines)

    def _build_node_map(self, tree: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Build a flat map from node_id -> node dict for fast lookup."""
        node_map: Dict[str, Dict[str, Any]] = {}

        def _walk(node: Dict) -> None:
            nid = node.get("node_id")
            if nid:
                node_map[nid] = node
            # Real PageIndex uses "nodes"; fallback uses "children"
            for child in node.get("nodes", node.get("children", [])):
                _walk(child)

        # Walk from top-level nodes — real PageIndex uses "structure"; fallback uses "children"
        top_nodes = tree.get("structure", tree.get("children", []))
        for node in top_nodes:
            _walk(node)

        return node_map
