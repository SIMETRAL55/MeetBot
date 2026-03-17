"""
Unit tests for meetbot.services.rag.transcript_to_md.

Tests cover:
- Basic single-speaker conversion
- Multi-speaker turn grouping
- Long monologue sub-section splitting
- Empty segment list edge case
- Time formatting helper
- Line-to-segment mapping accuracy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from meetbot.services.rag.transcript_to_md import convert, _fmt_time, _group_speaker_turns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_seg(idx: int, speaker: str, start: float, end: float, text: str) -> dict:
    return {"segment_index": idx, "speaker": speaker, "start": start, "end": end, "text": text}


SIMPLE_SEGMENTS = [
    make_seg(0, "Alice", 0.0,   10.0, "Hello everyone."),
    make_seg(1, "Bob",   10.0,  25.0, "Good morning."),
    make_seg(2, "Alice", 25.0,  40.0, "Let us begin."),
]

SAME_SPEAKER_SEGMENTS = [
    make_seg(0, "Alice", 0.0,  10.0, "First sentence."),
    make_seg(1, "Alice", 10.0, 20.0, "Second sentence."),
    make_seg(2, "Alice", 20.0, 30.0, "Third sentence."),
]


# ---------------------------------------------------------------------------
# _fmt_time
# ---------------------------------------------------------------------------

class TestFmtTime:
    def test_under_a_minute(self):
        assert _fmt_time(45.0) == "00:45"

    def test_exact_minute(self):
        assert _fmt_time(60.0) == "01:00"

    def test_over_an_hour(self):
        assert _fmt_time(3661.0) == "01:01:01"

    def test_zero(self):
        assert _fmt_time(0.0) == "00:00"

    def test_fractional_seconds_truncated(self):
        # fractional part should be truncated to integer seconds
        assert _fmt_time(90.9) == "01:30"


# ---------------------------------------------------------------------------
# _group_speaker_turns
# ---------------------------------------------------------------------------

class TestGroupSpeakerTurns:
    def test_alternating_speakers(self):
        turns = _group_speaker_turns(SIMPLE_SEGMENTS)
        assert len(turns) == 3
        assert turns[0][0]["speaker"] == "Alice"
        assert turns[1][0]["speaker"] == "Bob"
        assert turns[2][0]["speaker"] == "Alice"

    def test_same_speaker_grouped(self):
        turns = _group_speaker_turns(SAME_SPEAKER_SEGMENTS)
        assert len(turns) == 1
        assert len(turns[0]) == 3

    def test_empty_input(self):
        assert _group_speaker_turns([]) == []


# ---------------------------------------------------------------------------
# convert()
# ---------------------------------------------------------------------------

class TestConvert:
    def test_empty_returns_title_only(self):
        result = convert([], filename="Test Meeting")
        assert "# Meeting Transcript: Test Meeting" in result.markdown
        assert result.line_to_segment == {}

    def test_contains_title(self):
        result = convert(SIMPLE_SEGMENTS, filename="Q3 Review")
        assert "# Meeting Transcript: Q3 Review" in result.markdown

    def test_speaker_headings_present(self):
        result = convert(SIMPLE_SEGMENTS)
        assert "## Speaker: Alice" in result.markdown
        assert "## Speaker: Bob" in result.markdown

    def test_segment_text_present(self):
        result = convert(SIMPLE_SEGMENTS)
        assert "Hello everyone." in result.markdown
        assert "Good morning." in result.markdown
        assert "Let us begin." in result.markdown

    def test_timestamp_format_in_lines(self):
        result = convert(SIMPLE_SEGMENTS)
        # Segments have timestamps like [00:00 - 00:10]
        assert "[00:00 - 00:10] Alice: Hello everyone." in result.markdown

    def test_line_to_segment_mapping_populated(self):
        result = convert(SIMPLE_SEGMENTS)
        # Every segment should be mapped to at least one line
        mapped_segment_indices = set(result.line_to_segment.values())
        assert 0 in mapped_segment_indices
        assert 1 in mapped_segment_indices
        assert 2 in mapped_segment_indices

    def test_same_speaker_contiguous_one_heading(self):
        result = convert(SAME_SPEAKER_SEGMENTS)
        # Only one "## Speaker: Alice" heading
        heading_count = result.markdown.count("## Speaker: Alice")
        assert heading_count == 1

    def test_all_lines_are_strings(self):
        result = convert(SIMPLE_SEGMENTS)
        assert isinstance(result.markdown, str)
        assert len(result.markdown) > 0

    def test_long_monologue_gets_subsections(self):
        """A single speaker talking for >5 minutes should get ### sub-headings."""
        # Build 20 segments from t=0 to t=1200 (20 minutes)
        long_segs = [
            make_seg(i, "Alice", i * 60.0, (i + 1) * 60.0, f"Segment {i}")
            for i in range(20)
        ]
        result = convert(long_segs)
        # Should have at least one ### heading
        assert "### " in result.markdown

    def test_line_segment_mapping_correct_index(self):
        """line_to_segment values should be the actual segment_index values."""
        result = convert(SIMPLE_SEGMENTS)
        # The mapping should reference the segment indices we set (0, 1, 2)
        all_mapped = set(result.line_to_segment.values())
        assert all_mapped.issubset({0, 1, 2})

    def test_missing_segment_index_defaults_to_zero(self):
        """Segments without segment_index key should default to 0."""
        segs = [
            {"speaker": "Alice", "start": 0.0, "end": 5.0, "text": "Hello"},
        ]
        result = convert(segs)
        # Should not raise; line_to_segment values are 0
        assert all(v == 0 for v in result.line_to_segment.values())

    def test_markdown_is_parseable_headings(self):
        """The output should have no malformed heading lines."""
        result = convert(SIMPLE_SEGMENTS)
        lines = result.markdown.split("\n")
        for line in lines:
            if line.startswith("#"):
                # Heading must start with "# " (space after hash)
                assert line.lstrip("#")[0] == " ", f"Malformed heading: {line!r}"
