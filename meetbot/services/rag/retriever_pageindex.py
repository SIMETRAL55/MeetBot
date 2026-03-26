"""
PageIndex tree-search retriever for MeetBot.

Implements an agentic LLM-driven retrieval loop over a 3-level PageIndex
tree.  The LLM receives a compact tree summary (titles + summaries + node_ids)
and selects 1-3 nodes per iteration.  A sufficiency check determines whether
enough context has been gathered or more iterations are needed (up to 4).

Uses the same OpenAI-compatible endpoint as indexing (env var injection via
``pageindex_env``), so it works with OpenRouter, local Ollama, or OpenAI.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ...config import settings
from .prompts import SUFFICIENCY_CHECK_PROMPT, TREE_SEARCH_PROMPT
from .retrieval_strategy import RetrievalResult

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4


def _parse_json(text: str) -> Any:
    """Extract JSON from an LLM response, with fallback parsing."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


class PageIndexRetriever:
    """Agentic LLM-based tree search over a PageIndex structure."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or settings.PAGEINDEX_LLM_MODEL
        self.base_url = base_url or settings.get_pageindex_base_url()
        self.api_key = api_key or settings.PAGEINDEX_LLM_API_KEY or "not-needed"

    # -- LLM call ----------------------------------------------------------

    def _llm_call(self, prompt: str) -> str:
        """Single synchronous LLM call via ``pageindex_env()``."""
        from ...adapters.llm.openai_adapter import pageindex_env

        with pageindex_env(settings):
            import openai
            client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )
            return response.choices[0].message.content or ""

    # -- Tree summary ------------------------------------------------------

    @staticmethod
    def _build_tree_summary(tree: Dict[str, Any]) -> str:
        """Serialize tree to text with titles + summaries + node_ids only."""
        lines: list[str] = []

        def _walk(node: Dict, indent: int = 0) -> None:
            nid = node.get("node_id", "")
            title = node.get("title", "")
            summary_obj = node.get("summary", "")
            if isinstance(summary_obj, dict):
                summary_text = summary_obj.get("summary", "")
            else:
                summary_text = str(summary_obj) if summary_obj else ""

            prefix = "  " * indent
            line = f"{prefix}[{nid}] {title}"
            if summary_text:
                line += f" -- {summary_text}"
            lines.append(line)

            for child in node.get("children", []):
                _walk(child, indent + 1)

        _walk(tree)
        return "\n".join(lines)

    # -- Chat history formatting -------------------------------------------

    @staticmethod
    def _format_chat_history(chat_history: Optional[List[Dict]]) -> str:
        """Format chat history for LLM prompts."""
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history[-6:]:  # Last 6 messages max
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # -- Agentic search loop -----------------------------------------------

    async def search(
        self,
        query: str,
        tree: Dict[str, Any],
        content_map: Dict[str, str],
        chat_history: Optional[List[Dict]] = None,
        cancel_checker: Optional[callable] = None,
    ) -> List[RetrievalResult]:
        """Agentic search: up to MAX_ITERATIONS of select-then-check.

        Args:
            cancel_checker: Optional callable checked between iterations.
                Should raise an exception to abort the search.

        Returns a list of RetrievalResult objects for the selected nodes.
        """
        tree_summary = self._build_tree_summary(tree)
        chat_hist_text = self._format_chat_history(chat_history)
        chat_history_block = (
            f"Chat history:\n{chat_hist_text}" if chat_hist_text else ""
        )

        all_selected: list[str] = []
        all_content: list[str] = []

        for iteration in range(MAX_ITERATIONS):
            if cancel_checker:
                cancel_checker()
            # a. Tree search
            search_prompt = TREE_SEARCH_PROMPT.format(
                query=query,
                chat_history_block=chat_history_block,
                tree_summary=tree_summary,
            )
            try:
                raw = self._llm_call(search_prompt)
                parsed = _parse_json(raw)
                node_ids = parsed.get("node_ids", []) if isinstance(parsed, dict) else []
                node_ids = [str(n) for n in node_ids if str(n) not in all_selected]
            except Exception as exc:
                logger.warning("Tree search LLM failed (iter %d): %s", iteration, exc)
                break

            if not node_ids:
                logger.debug("No new nodes selected at iteration %d", iteration)
                break

            # b. Fetch content for newly selected nodes
            new_content_parts: list[str] = []
            for nid in node_ids:
                text = content_map.get(nid, "")
                if text:
                    new_content_parts.append(f"--- [{nid}] ---\n{text}")

            all_selected.extend(node_ids)
            all_content.extend(new_content_parts)

            # c. Sufficiency check
            sufficiency_prompt = SUFFICIENCY_CHECK_PROMPT.format(
                query=query,
                chat_history_block=chat_history_block,
                previously_selected_nodes=", ".join(all_selected),
                retrieved_content="\n\n".join(all_content),
            )
            try:
                raw = self._llm_call(sufficiency_prompt)
                check = _parse_json(raw)
                if isinstance(check, dict) and check.get("sufficient", False):
                    logger.info(
                        "PageIndex retriever: sufficient after %d iterations, %d nodes",
                        iteration + 1, len(all_selected),
                    )
                    break
                # If insufficient, the LLM may suggest additional node_ids
                extra = check.get("node_ids", []) if isinstance(check, dict) else []
                if extra:
                    for nid in extra:
                        nid = str(nid)
                        if nid not in all_selected and nid in content_map:
                            all_selected.append(nid)
                            text = content_map.get(nid, "")
                            if text:
                                all_content.append(f"--- [{nid}] ---\n{text}")
            except Exception as exc:
                logger.warning("Sufficiency check failed (iter %d): %s", iteration, exc)
                break

        if not all_selected:
            logger.warning("PageIndex retriever: no nodes selected for query: %s", query[:80])
            return []

        # Build RetrievalResult list
        node_map = self._build_flat_node_map(tree)
        results: list[RetrievalResult] = []
        for nid in all_selected:
            text = content_map.get(nid, "")
            node_info = node_map.get(nid, {})
            results.append(RetrievalResult(
                text=text,
                metadata={
                    "node_id": nid,
                    "node_title": node_info.get("title", nid),
                    "level": node_info.get("level"),
                    "speakers": node_info.get("speakers", []),
                    "start_segment": node_info.get("start_segment"),
                    "end_segment": node_info.get("end_segment"),
                },
                score=1.0,
                retrieval_method="pageindex",
                source_ref=nid,
            ))

        logger.info(
            "PageIndex retriever: query=%s -> %d nodes selected",
            query[:40], len(results),
        )
        return results

    # -- Node map ----------------------------------------------------------

    @staticmethod
    def _build_flat_node_map(tree: Dict[str, Any]) -> Dict[str, Dict]:
        """Flat map from node_id to node dict."""
        nmap: dict[str, dict] = {}

        def _walk(node: Dict) -> None:
            nid = node.get("node_id")
            if nid:
                nmap[nid] = node
            for child in node.get("children", []):
                _walk(child)

        _walk(tree)
        return nmap
