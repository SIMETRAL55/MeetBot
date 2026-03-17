"""
Convert aligned transcript segments to structured Markdown for PageIndex.

PageIndex's ``md_to_tree()`` builds a hierarchical tree from Markdown headings.
This module converts MeetBot's ``[{speaker, text, start, end}, ...]`` segment
list into a heading-structured Markdown document so PageIndex can produce a
meaningful tree.

Structuring strategy
--------------------
- Group consecutive segments by the same speaker into "speaker turns".
- Each turn becomes a ``## Speaker: {name} ({start} - {end})`` heading.
- Individual segments within a turn become timestamped lines.
- If a single speaker talks for longer than ``MAX_TURN_SECONDS`` (default 300),
  the turn is split into sub-sections with ``### {start} - {end}`` headings.

The returned Markdown string also includes a mapping from line numbers back to
segment indices, enabling retrieval-time lookback from PageIndex nodes to the
original segments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# If a single speaker talks for more than this many seconds without interruption,
# insert a time-window sub-heading to give PageIndex more granular structure.
MAX_TURN_SECONDS = 300  # 5 minutes


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class ConversionResult:
    """Result of transcript-to-Markdown conversion."""
    markdown: str
    line_to_segment: Dict[int, int]  # 1-based line number -> segment index


def convert(
    segments: List[Dict],
    filename: str = "Untitled Meeting",
) -> ConversionResult:
    """
    Convert aligned transcript segments to structured Markdown.

    Args:
        segments: List of segment dicts with keys: speaker, text, start, end.
                  May also contain 'segment_index'.
        filename: Meeting name for the top-level heading.

    Returns:
        ConversionResult with the Markdown string and line-to-segment mapping.
    """
    if not segments:
        return ConversionResult(markdown=f"# Meeting Transcript: {filename}\n", line_to_segment={})

    lines: List[str] = []
    line_to_segment: Dict[int, int] = {}

    # Title
    lines.append(f"# Meeting Transcript: {filename}")
    lines.append("")

    # Group consecutive segments by speaker into turns
    turns = _group_speaker_turns(segments)

    for turn in turns:
        turn_start = turn[0].get("start", 0)
        turn_end = turn[-1].get("end", 0)
        speaker = turn[0].get("speaker", "Unknown")
        turn_duration = turn_end - turn_start

        # Speaker turn heading
        lines.append(f"## Speaker: {speaker} ({_fmt_time(turn_start)} - {_fmt_time(turn_end)})")
        lines.append("")

        if turn_duration > MAX_TURN_SECONDS:
            # Split long monologues into time-window sub-sections
            _add_long_turn_with_subsections(lines, line_to_segment, turn)
        else:
            # Normal turn: list all segments
            for seg in turn:
                seg_idx = seg.get("segment_index", 0)
                start_s = _fmt_time(seg.get("start", 0))
                end_s = _fmt_time(seg.get("end", 0))
                text = seg.get("text", "").strip()
                line_num = len(lines) + 1
                lines.append(f"[{start_s} - {end_s}] {speaker}: {text}")
                line_to_segment[line_num] = seg_idx

            lines.append("")

    markdown = "\n".join(lines)
    return ConversionResult(markdown=markdown, line_to_segment=line_to_segment)


def _group_speaker_turns(segments: List[Dict]) -> List[List[Dict]]:
    """Group consecutive segments by the same speaker into turns."""
    if not segments:
        return []

    turns: List[List[Dict]] = []
    current_turn: List[Dict] = [segments[0]]

    for seg in segments[1:]:
        if seg.get("speaker") == current_turn[0].get("speaker"):
            current_turn.append(seg)
        else:
            turns.append(current_turn)
            current_turn = [seg]

    turns.append(current_turn)
    return turns


def _add_long_turn_with_subsections(
    lines: List[str],
    line_to_segment: Dict[int, int],
    turn: List[Dict],
) -> None:
    """Add a long speaker turn with time-window sub-headings."""
    turn_start = turn[0].get("start", 0)
    speaker = turn[0].get("speaker", "Unknown")

    # Split into MAX_TURN_SECONDS windows
    window_start = turn_start
    window_segs: List[Dict] = []

    for seg in turn:
        seg_start = seg.get("start", 0)
        if seg_start - window_start >= MAX_TURN_SECONDS and window_segs:
            # Emit current window
            window_end = window_segs[-1].get("end", 0)
            lines.append(f"### {_fmt_time(window_start)} - {_fmt_time(window_end)}")
            lines.append("")
            for ws in window_segs:
                seg_idx = ws.get("segment_index", 0)
                start_s = _fmt_time(ws.get("start", 0))
                end_s = _fmt_time(ws.get("end", 0))
                text = ws.get("text", "").strip()
                line_num = len(lines) + 1
                lines.append(f"[{start_s} - {end_s}] {speaker}: {text}")
                line_to_segment[line_num] = seg_idx
            lines.append("")
            window_start = seg_start
            window_segs = []

        window_segs.append(seg)

    # Emit final window
    if window_segs:
        window_end = window_segs[-1].get("end", 0)
        lines.append(f"### {_fmt_time(window_start)} - {_fmt_time(window_end)}")
        lines.append("")
        for ws in window_segs:
            seg_idx = ws.get("segment_index", 0)
            start_s = _fmt_time(ws.get("start", 0))
            end_s = _fmt_time(ws.get("end", 0))
            text = ws.get("text", "").strip()
            line_num = len(lines) + 1
            lines.append(f"[{start_s} - {end_s}] {speaker}: {text}")
            line_to_segment[line_num] = seg_idx
        lines.append("")
