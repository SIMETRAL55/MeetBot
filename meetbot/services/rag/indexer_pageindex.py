"""
PageIndex indexing adapter for MeetBot transcripts.

Builds a 3-level hierarchical JSON tree directly from raw transcript
segments via LLM calls — no Markdown conversion, no vendor dependency.

Pipeline:
    segments → topic segmentation (LLM)
             → sub-topic segmentation (LLM per topic)
             → per-node summarization (LLM per node)
             → root synthesis (LLM)
             → content_map build (deterministic)

Output files:
    {job_id}_tree.json        — the hierarchical tree
    {job_id}_content_map.json — node_id → raw transcript text
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ...config import settings
from .prompts import (
    NODE_SUMMARY_PROMPT,
    ROOT_SYNTHESIS_PROMPT,
    SUBTOPIC_SEGMENTATION_PROMPT,
    TOPIC_SEGMENTATION_PROMPT,
    TOPIC_SEGMENTATION_RETRY_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Convert float seconds to ``HH:MM:SS``."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_segments_for_llm(segments: List[Dict]) -> str:
    """Format segment dicts as numbered lines for LLM prompts.

    Each line: ``[{index}] SPEAKER_XX [{start}-{end}]: {text}``
    """
    lines: list[str] = []
    for i, seg in enumerate(segments):
        idx = seg.get("segment_index", i)
        speaker = seg.get("speaker", "UNKNOWN")
        start = _fmt_time(seg.get("start", 0))
        end = _fmt_time(seg.get("end", 0))
        text = seg.get("text", "")
        lines.append(f"[{idx}] {speaker} [{start}-{end}]: {text}")
    return "\n".join(lines)


def _parse_json(text: str) -> Any:
    """Extract JSON from an LLM response, with fallback parsing."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Regex extract first [...] or {...}
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


# ── Adapter ──────────────────────────────────────────────────────────────────

class PageIndexAdapter:
    """Builds and loads a 3-level transcript-native PageIndex tree."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or settings.PAGEINDEX_LLM_MODEL
        self.base_url = base_url or settings.get_pageindex_base_url()
        self.api_key = api_key or settings.PAGEINDEX_LLM_API_KEY or "not-needed"
        self.tree: Dict[str, Any] = {}
        self.content_map: Dict[str, str] = {}

    # ── LLM call ─────────────────────────────────────────────────────────

    async def _llm_call(self, prompt: str) -> str:
        """Async LLM call — runs the blocking HTTP request in a thread pool."""
        model = self.model
        base_url = self.base_url
        api_key = self.api_key

        def _sync() -> str:
            from ...adapters.llm.openai_adapter import pageindex_env, get_pageindex_client

            with pageindex_env(settings):
                client = get_pageindex_client(settings)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                )
                return response.choices[0].message.content or ""

        return await asyncio.to_thread(_sync)

    # ── Step A: Topic segmentation ───────────────────────────────────────

    async def _segment_topics(self, segments: List[Dict]) -> List[Dict]:
        """Identify 4-7 major topic blocks via LLM."""
        formatted = _format_segments_for_llm(segments)
        max_seg = len(segments) - 1

        # If the formatted text is very long, truncate to keep within typical
        # context windows (~128k chars ≈ ~32k tokens).  We keep first and last
        # portions so the LLM can see the full range.
        MAX_CHARS = 100_000
        if len(formatted) > MAX_CHARS:
            half = MAX_CHARS // 2
            formatted = (
                formatted[:half]
                + "\n\n... [middle segments omitted for brevity] ...\n\n"
                + formatted[-half:]
            )
            logger.info(
                "Topic segmentation: truncated %d segments (%d chars) for LLM prompt",
                len(segments), len(formatted),
            )

        prompt = TOPIC_SEGMENTATION_PROMPT.format(
            segments=formatted, max_segment=max_seg,
        )
        try:
            raw = await self._llm_call(prompt)
            if not raw or not raw.strip():
                logger.warning("Topic segmentation: LLM returned empty response")
                raise ValueError("Empty LLM response")
            topics = _parse_json(raw)
            if isinstance(topics, list) and topics:
                return topics
        except Exception as exc:
            logger.warning("Topic segmentation parse failed, retrying: %s", exc)

        # Retry with stronger instruction
        retry_prompt = TOPIC_SEGMENTATION_RETRY_PROMPT.format(
            segments=formatted, max_segment=max_seg,
        )
        try:
            raw = await self._llm_call(retry_prompt)
            if not raw or not raw.strip():
                logger.warning("Topic segmentation retry: LLM returned empty response")
                raise ValueError("Empty LLM response")
            topics = _parse_json(raw)
            if isinstance(topics, list) and topics:
                return topics
        except Exception as exc:
            logger.warning("Topic segmentation retry failed: %s", exc)

        # Fallback: equal-sized blocks
        return self._fallback_topic_split(segments)

    @staticmethod
    def _fallback_topic_split(segments: List[Dict]) -> List[Dict]:
        """Split segments into roughly equal blocks of ~15."""
        n = len(segments)
        block_size = 15
        topics: list[dict] = []
        start = 0
        idx = 0
        while start < n:
            end = min(start + block_size - 1, n - 1)
            topics.append({
                "start_segment": start,
                "end_segment": end,
                "topic_title": f"Topic {idx + 1}",
            })
            start = end + 1
            idx += 1
        return topics

    # ── Step B: Sub-topic segmentation ───────────────────────────────────

    async def _segment_subtopics(
        self, segments: List[Dict], topic: Dict,
    ) -> List[Dict]:
        """Split one topic block into 2-4 sub-clusters via LLM."""
        s, e = topic["start_segment"], topic["end_segment"]
        topic_segs = segments[s: e + 1]
        formatted = _format_segments_for_llm(topic_segs)

        prompt = SUBTOPIC_SEGMENTATION_PROMPT.format(segments=formatted)
        try:
            raw = await self._llm_call(prompt)
            subs = _parse_json(raw)
            if isinstance(subs, list) and subs:
                return subs
        except Exception as exc:
            logger.warning("Subtopic segmentation failed for %s: %s", topic.get("topic_title"), exc)

        # Fallback: group consecutive same-speaker segments
        return self._fallback_speaker_turns(topic_segs, offset=s)

    @staticmethod
    def _fallback_speaker_turns(
        segments: List[Dict], offset: int = 0,
    ) -> List[Dict]:
        """Group consecutive same-speaker segments as subtopics."""
        if not segments:
            return []
        groups: list[dict] = []
        cur_speaker = segments[0].get("speaker", "UNKNOWN")
        start = 0
        for i, seg in enumerate(segments):
            spk = seg.get("speaker", "UNKNOWN")
            if spk != cur_speaker:
                groups.append({
                    "start_segment": start + offset,
                    "end_segment": i - 1 + offset,
                    "title": f"{cur_speaker}: discussion",
                    "speakers": [cur_speaker],
                })
                cur_speaker = spk
                start = i
        # Last group
        groups.append({
            "start_segment": start + offset,
            "end_segment": len(segments) - 1 + offset,
            "title": f"{cur_speaker}: discussion",
            "speakers": [cur_speaker],
        })
        return groups

    # ── Step C: Per-node summarization ───────────────────────────────────

    async def _summarize_node(
        self, segments: List[Dict], start: int, end: int,
    ) -> Dict:
        """Summarize a segment range via LLM."""
        node_segs = segments[start: end + 1]
        text_block = "\n".join(
            f"{seg.get('speaker', 'UNKNOWN')}: {seg.get('text', '')}"
            for seg in node_segs
        )
        prompt = NODE_SUMMARY_PROMPT.format(segments_text=text_block)
        try:
            raw = await self._llm_call(prompt)
            result = _parse_json(raw)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            logger.warning("Node summarization failed: %s", exc)

        # Fallback
        preview = text_block[:200]
        return {
            "summary": f"{preview}...",
            "decision": None,
            "action_items": [],
            "owner": None,
            "keywords": [],
        }

    # ── Step D: Root synthesis ───────────────────────────────────────────

    async def _synthesize_root(
        self, topics: List[Dict], filename: str,
    ) -> Dict:
        """Produce a meeting-level title and summary."""
        topic_text = "\n".join(
            f"- {t.get('topic_title', 'Untitled')}: "
            f"{t.get('summary', {}).get('summary', '')}"
            for t in topics
        )
        prompt = ROOT_SYNTHESIS_PROMPT.format(topic_summaries=topic_text)
        try:
            raw = await self._llm_call(prompt)
            result = _parse_json(raw)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            logger.warning("Root synthesis failed: %s", exc)

        return {
            "title": filename,
            "summary": f"Meeting with {len(topics)} topics discussed.",
        }

    # ── Step E: Content map (deterministic) ──────────────────────────────

    @staticmethod
    def _build_content_map(
        tree: Dict[str, Any], segments: List[Dict],
    ) -> Dict[str, str]:
        """Build ``node_id → raw transcript text`` mapping."""
        cmap: dict[str, str] = {}

        def _walk(node: Dict) -> None:
            nid = node.get("node_id")
            if nid:
                s = node.get("start_segment", 0)
                e = node.get("end_segment", 0)
                lines: list[str] = []
                for seg in segments[s: e + 1]:
                    spk = seg.get("speaker", "UNKNOWN")
                    st = _fmt_time(seg.get("start", 0))
                    et = _fmt_time(seg.get("end", 0))
                    txt = seg.get("text", "")
                    lines.append(f"{spk} [{st}-{et}]: {txt}")
                cmap[nid] = "\n".join(lines)
            for child in node.get("children", []):
                _walk(child)

        # Walk ROOT and all descendants
        _walk(tree)
        return cmap

    # ── build_index ──────────────────────────────────────────────────────

    async def build_index(
        self,
        segments: List[Dict],
        job_id: str,
        output_dir: Optional[Path] = None,
        filename: str = "Untitled Meeting",
        progress_callback: Optional[Callable] = None,
        cancel_checker: Optional[Callable] = None,
    ) -> Path:
        """Build a 3-level PageIndex tree and content map from segments.

        Returns the path to the saved ``{job_id}_tree.json`` file.

        Args:
            cancel_checker: Optional callable that raises an exception if
                the build should be aborted. Called between LLM stages.
        """
        if output_dir is None:
            output_dir = settings.get_pageindex_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        def _check_cancel() -> None:
            if cancel_checker:
                cancel_checker()

        if progress_callback:
            progress_callback("pageindex", 0, "Starting topic segmentation...")

        # Step A — topic segmentation
        _check_cancel()
        topics = await self._segment_topics(segments)
        if progress_callback:
            progress_callback("pageindex", 10, f"Found {len(topics)} topics")

        # Step B — sub-topic segmentation per topic (parallel)
        _check_cancel()
        subtopics_results = await asyncio.gather(
            *[self._segment_subtopics(segments, t) for t in topics],
            return_exceptions=True,
        )
        for t, result in zip(topics, subtopics_results):
            if isinstance(result, Exception):
                logger.warning("Subtopic segmentation failed for %s, using fallback: %s", t.get("topic_title"), result)
                t["subtopics"] = self._fallback_speaker_turns(
                    segments[t["start_segment"]: t["end_segment"] + 1],
                    offset=t["start_segment"],
                )
            else:
                t["subtopics"] = result
        if progress_callback:
            progress_callback("pageindex", 30, "Sub-topic segmentation complete")

        # Step C — summarize every node (topics + subtopics, parallel)
        _check_cancel()
        nodes_to_summarize: list[tuple[dict, int, int]] = []
        for t in topics:
            nodes_to_summarize.append((t, t["start_segment"], t["end_segment"]))
            for sub in t.get("subtopics", []):
                nodes_to_summarize.append((sub, sub["start_segment"], sub["end_segment"]))

        summary_results = await asyncio.gather(
            *[self._summarize_node(segments, start, end) for _, start, end in nodes_to_summarize],
            return_exceptions=True,
        )
        for (node, start, end), result in zip(nodes_to_summarize, summary_results):
            if isinstance(result, Exception):
                logger.warning("Node summarization failed, using fallback: %s", result)
                text_block = "\n".join(
                    f"{seg.get('speaker', 'UNKNOWN')}: {seg.get('text', '')}"
                    for seg in segments[start: end + 1]
                )
                node["summary"] = {
                    "summary": f"{text_block[:200]}...",
                    "decision": None,
                    "action_items": [],
                    "owner": None,
                    "keywords": [],
                }
            else:
                node["summary"] = result
        if progress_callback:
            progress_callback("pageindex", 50, "Node summarization complete")

        # Step D — root synthesis
        _check_cancel()
        root_info = await self._synthesize_root(topics, filename)
        if progress_callback:
            progress_callback("pageindex", 70, "Root synthesis complete")

        # Assemble tree
        participants = sorted({
            seg.get("speaker", "UNKNOWN") for seg in segments
        })

        tree_children: list[dict] = []
        for ti, t in enumerate(topics):
            subtopic_children: list[dict] = []
            for si, sub in enumerate(t.get("subtopics", [])):
                subtopic_children.append({
                    "node_id": f"T{ti}-S{si}",
                    "level": 2,
                    "title": sub.get("title", f"Subtopic {si}"),
                    "start_segment": sub["start_segment"],
                    "end_segment": sub["end_segment"],
                    "speakers": sub.get("speakers", []),
                    "summary": sub.get("summary", {}),
                    "children": [],
                })
            tree_children.append({
                "node_id": f"T{ti}",
                "level": 1,
                "title": t.get("topic_title", f"Topic {ti}"),
                "start_segment": t["start_segment"],
                "end_segment": t["end_segment"],
                "summary": t.get("summary", {}),
                "children": subtopic_children,
            })

        tree: Dict[str, Any] = {
            "node_id": "ROOT",
            "level": 0,
            "title": root_info.get("title", filename),
            "summary": root_info.get("summary", ""),
            "participants": participants,
            "n_segments": len(segments),
            "children": tree_children,
        }

        # Step E — content map
        content_map = self._build_content_map(tree, segments)
        if progress_callback:
            progress_callback("pageindex", 90, "Saving PageIndex tree...")

        # Save outputs
        tree_path = output_dir / f"{job_id}_tree.json"
        cmap_path = output_dir / f"{job_id}_content_map.json"

        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        with open(cmap_path, "w", encoding="utf-8") as f:
            json.dump(content_map, f, indent=2, ensure_ascii=False)

        if progress_callback:
            progress_callback("pageindex", 100, "PageIndex tree built successfully")

        logger.info(
            "PageIndex tree built for job %s: %s (%d bytes, %d nodes in content_map)",
            job_id[:8], tree_path, tree_path.stat().st_size, len(content_map),
        )
        return tree_path

    # ── load_index ───────────────────────────────────────────────────────

    @classmethod
    def load_index(cls, path: str) -> "PageIndexAdapter":
        """Load a previously built tree and content_map from disk.

        ``path`` is the tree JSON path (stored in ``job.pageindex_path``).
        The content_map path is derived by replacing ``_tree.json`` with
        ``_content_map.json``.

        Returns a ``PageIndexAdapter`` instance with ``.tree`` and
        ``.content_map`` populated.
        """
        tree_path = Path(path)
        if not tree_path.exists():
            raise FileNotFoundError(f"PageIndex tree not found: {tree_path}")

        cmap_path = tree_path.parent / (tree_path.stem.replace("_tree", "_content_map") + ".json")

        with open(tree_path, "r", encoding="utf-8") as f:
            tree = json.load(f)

        content_map: Dict[str, str] = {}
        if cmap_path.exists():
            with open(cmap_path, "r", encoding="utf-8") as f:
                content_map = json.load(f)

        instance = cls()
        instance.tree = tree
        instance.content_map = content_map
        return instance
