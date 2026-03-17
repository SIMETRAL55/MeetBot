"""
PageIndex indexing adapter for MeetBot transcripts.

Wraps VectifyAI's PageIndex ``md_to_tree()`` to build a hierarchical tree
from meeting transcript segments.  Uses env var injection (Approach A) so
PageIndex vendor code is never modified.

Pipeline:
    segments -> transcript_to_md -> temp .md file -> md_to_tree() -> JSON tree
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ...config import settings

logger = logging.getLogger(__name__)


class PageIndexAdapter:
    """Wraps VectifyAI's PageIndex for MeetBot transcripts."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or settings.PAGEINDEX_LLM_MODEL
        self.base_url = base_url or settings.get_pageindex_base_url()
        self.api_key = api_key or settings.PAGEINDEX_LLM_API_KEY or "not-needed"

    async def build_index(
        self,
        segments: List[Dict],
        job_id: str,
        output_dir: Optional[Path] = None,
        filename: str = "Untitled Meeting",
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """
        Build a PageIndex tree from transcript segments.

        Steps:
            1. Convert segments -> Markdown via transcript_to_md
            2. Write temp .md file
            3. Set OPENAI_BASE_URL + OPENAI_API_KEY env vars for PageIndex
            4. Call pageindex.md_to_tree(md_path, model=self.model)
            5. Restore original env vars
            6. Post-process: enrich nodes with segment index mappings
            7. Save resulting JSON tree to output_dir/{job_id}_pageindex.json
            8. Return path to the JSON tree

        Args:
            segments: Aligned transcript segments [{speaker, text, start, end, segment_index}].
            job_id: Job identifier.
            output_dir: Directory for the output JSON tree. Defaults to PAGEINDEX_OUTPUT_DIR.
            filename: Meeting name for the document title.
            progress_callback: Optional (stage, pct, msg) callback.

        Returns:
            Path to the saved PageIndex JSON tree file.
        """
        if output_dir is None:
            output_dir = settings.get_pageindex_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback("pageindex", 0, "Converting transcript to Markdown...")

        # Step 1: Convert to Markdown
        from .transcript_to_md import convert as transcript_to_md
        result = transcript_to_md(segments, filename=filename)
        md_content = result.markdown
        line_to_segment = result.line_to_segment

        if progress_callback:
            progress_callback("pageindex", 10, "Markdown conversion complete")

        # Step 2: Write temp .md file
        temp_dir = Path(settings.TEMP_DIR) / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        md_path = temp_dir / "transcript.md"
        md_path.write_text(md_content, encoding="utf-8")

        if progress_callback:
            progress_callback("pageindex", 15, "Building PageIndex tree (LLM calls)...")

        # Step 3-4: Call PageIndex with env var injection
        try:
            from ...adapters.llm.openai_adapter import pageindex_env

            with pageindex_env(settings):
                tree = await self._call_md_to_tree(str(md_path))
        except Exception as exc:
            logger.error("PageIndex build_index failed for job %s: %s", job_id[:8], exc)
            raise

        if progress_callback:
            progress_callback("pageindex", 80, "Post-processing tree...")

        # Step 5: Post-process — enrich nodes with segment index mappings
        tree = self._enrich_tree_with_segments(tree, line_to_segment)

        if progress_callback:
            progress_callback("pageindex", 90, "Saving PageIndex tree...")

        # Step 6: Save JSON tree
        output_path = output_dir / f"{job_id}_pageindex.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

        # Clean up temp markdown file
        try:
            md_path.unlink(missing_ok=True)
        except Exception:
            pass

        if progress_callback:
            progress_callback("pageindex", 100, "PageIndex tree built successfully")

        logger.info(
            "PageIndex tree built for job %s: %s (%d bytes)",
            job_id[:8], output_path, output_path.stat().st_size,
        )
        return output_path

    async def _call_md_to_tree(self, md_path: str) -> Dict[str, Any]:
        """
        Call PageIndex's md_to_tree function.

        If running in sync context, wraps in asyncio.
        Falls back to a simple heading-based tree if PageIndex import fails.
        """
        try:
            from pageindex import md_to_tree
            # md_to_tree is async in PageIndex
            tree = await md_to_tree(md_path, model=self.model)
            return tree
        except ImportError:
            logger.warning(
                "PageIndex library not installed. "
                "Falling back to simple heading-based tree."
            )
            return self._build_simple_tree(md_path)
        except Exception as exc:
            logger.error("md_to_tree() failed: %s. Falling back to simple tree.", exc)
            return self._build_simple_tree(md_path)

    def _build_simple_tree(self, md_path: str) -> Dict[str, Any]:
        """
        Build a simple tree from Markdown headings when PageIndex is unavailable.

        Parses ## and ### headings to create a basic hierarchy without LLM calls.
        """
        content = Path(md_path).read_text(encoding="utf-8")
        lines = content.split("\n")

        tree: Dict[str, Any] = {"title": "", "children": [], "content_lines": []}
        current_h2: Optional[Dict] = None
        current_h3: Optional[Dict] = None
        node_counter = 0

        for i, line in enumerate(lines):
            line_num = i + 1
            if line.startswith("# ") and not line.startswith("## "):
                tree["title"] = line[2:].strip()
            elif line.startswith("## "):
                node_counter += 1
                current_h2 = {
                    "node_id": f"{node_counter:04d}",
                    "title": line[3:].strip(),
                    "children": [],
                    "content_lines": [],
                    "start_line": line_num,
                    "end_line": line_num,
                }
                current_h3 = None
                tree["children"].append(current_h2)
            elif line.startswith("### "):
                node_counter += 1
                current_h3 = {
                    "node_id": f"{node_counter:04d}",
                    "title": line[4:].strip(),
                    "children": [],
                    "content_lines": [],
                    "start_line": line_num,
                    "end_line": line_num,
                }
                if current_h2 is not None:
                    current_h2["children"].append(current_h3)
            elif line.strip():
                target = current_h3 or current_h2
                if target is not None:
                    target["content_lines"].append(line_num)
                    target["end_line"] = line_num

        return tree

    def _enrich_tree_with_segments(
        self,
        tree: Dict[str, Any],
        line_to_segment: Dict[int, int],
    ) -> Dict[str, Any]:
        """
        Add segment_indices to each tree node based on line-to-segment mapping.

        Handles both the real PageIndex md_to_tree() output format (uses
        ``"structure"`` at root and ``"nodes"`` within each node, with only
        ``"line_num"`` for position) and the fallback _build_simple_tree format
        (uses ``"children"`` keys with ``"start_line"``/``"end_line"``).
        """
        def _subnodes(node: Dict) -> List[Dict]:
            # Real PageIndex uses "nodes"; fallback uses "children"
            return node.get("nodes", node.get("children", []))

        def _walk(node: Dict, siblings: List[Dict], pos: int) -> None:
            content_lines: List[int] = node.get("content_lines", [])

            # Determine start line: real format has "line_num", fallback has "start_line"
            start_line: Optional[int] = node.get("start_line") or node.get("line_num")
            end_line: Optional[int] = node.get("end_line")

            if end_line is None and start_line is not None:
                # Infer end from next sibling's start line
                if pos + 1 < len(siblings):
                    next_start = (
                        siblings[pos + 1].get("start_line")
                        or siblings[pos + 1].get("line_num")
                    )
                    end_line = (next_start - 1) if next_start else start_line + 500
                else:
                    # Last sibling: scan a generous range to find all remaining lines
                    end_line = start_line + 2000

            seg_indices: set = set()

            # Collect from explicit content_lines (fallback format)
            for ln in content_lines:
                if ln in line_to_segment:
                    seg_indices.add(line_to_segment[ln])

            # Scan the full line range of this node
            if start_line is not None and end_line is not None:
                for ln in range(start_line, end_line + 1):
                    if ln in line_to_segment:
                        seg_indices.add(line_to_segment[ln])

            node["segment_indices"] = sorted(seg_indices)

            children = _subnodes(node)
            for i, child in enumerate(children):
                _walk(child, children, i)

        # Walk top-level nodes — real PageIndex uses "structure"; fallback uses "children"
        top_nodes = tree.get("structure", tree.get("children", []))
        for i, node in enumerate(top_nodes):
            _walk(node, top_nodes, i)

        return tree

    def load_index(self, job_id: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Load a previously built PageIndex tree from disk."""
        if output_dir is None:
            output_dir = settings.get_pageindex_output_dir()

        tree_path = output_dir / f"{job_id}_pageindex.json"
        if not tree_path.exists():
            raise FileNotFoundError(f"PageIndex tree not found: {tree_path}")

        with open(tree_path, "r", encoding="utf-8") as f:
            return json.load(f)
