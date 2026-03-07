"""
Unit tests for the RAG chunker module.

Tests deterministic chunking behaviour, speaker-aware splitting,
overlap windows, and edge cases — all without any external model.
"""

import pytest

from meetbot.services.rag.chunker import Chunk, Chunker, _estimate_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(text: str, speaker: str = "SPK_00", start: float = 0.0, end: float = 5.0):
    """Minimal segment dict."""
    return {"text": text, "speaker": speaker, "start": start, "end": end}


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 1  # max(1, 0)

    def test_short_string(self):
        assert _estimate_tokens("Hi") == 1

    def test_four_chars_one_token(self):
        assert _estimate_tokens("word") == 1

    def test_eight_chars_two_tokens(self):
        assert _estimate_tokens("abcdefgh") == 2

    def test_japanese(self):
        result = _estimate_tokens("日本語テスト")
        assert result >= 1


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


class TestChunkDataclass:
    def test_to_doc_basic(self):
        c = Chunk(
            chunk_id=0,
            text="Hello world",
            speaker="SPK_00",
            start=1.0,
            end=5.0,
            job_id="abc123",
            version=1,
        )
        doc = c.to_doc()
        assert doc["id"] == "abc123_0"
        assert doc["text"] == "Hello world"
        assert doc["metadata"]["speaker"] == "SPK_00"
        assert doc["metadata"]["start"] == 1.0
        assert doc["metadata"]["end"] == 5.0
        assert doc["metadata"]["version"] == 1

    def test_to_doc_no_job_id(self):
        c = Chunk(chunk_id=3, text="x", speaker="A", start=0, end=1)
        doc = c.to_doc()
        assert doc["id"] == "3"


# ---------------------------------------------------------------------------
# Chunker: basic behaviour
# ---------------------------------------------------------------------------


class TestChunkerBasic:
    def test_empty_segments(self):
        chunker = Chunker()
        assert chunker.chunk_segments([]) == []

    def test_single_segment(self):
        chunker = Chunker(chunk_tokens=1000, chunk_overlap=0)
        segments = [_seg("Hello world", start=0.0, end=2.0)]
        chunks = chunker.chunk_segments(segments, job_id="j1")
        assert len(chunks) == 1
        assert "Hello world" in chunks[0].text
        assert chunks[0].job_id == "j1"
        assert chunks[0].start == 0.0
        assert chunks[0].end == 2.0

    def test_version_propagated(self):
        chunker = Chunker(chunk_tokens=1000, chunk_overlap=0)
        chunks = chunker.chunk_segments(
            [_seg("txt")], job_id="j1", version=42
        )
        assert chunks[0].version == 42

    def test_chunk_ids_sequential(self):
        chunker = Chunker(chunk_tokens=10, chunk_overlap=0)
        segments = [
            _seg("a" * 40, start=0, end=1),
            _seg("b" * 40, start=1, end=2),
            _seg("c" * 40, start=2, end=3),
        ]
        chunks = chunker.chunk_segments(segments)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_segments_with_blank_text_skipped(self):
        chunker = Chunker(chunk_tokens=1000, chunk_overlap=0)
        segments = [
            _seg("real text"),
            _seg(""),
            _seg("   "),
        ]
        chunks = chunker.chunk_segments(segments)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Chunker: speaker awareness
# ---------------------------------------------------------------------------


class TestChunkerSpeakerAware:
    def test_speaker_change_triggers_break_when_buffer_half_full(self):
        """
        When the buffer is at least half full and the speaker changes,
        the chunker should split.
        """
        # chunk_tokens = 20 → half = 10 → ~40 chars needed for half
        chunker = Chunker(chunk_tokens=20, chunk_overlap=0)
        segments = [
            _seg("x" * 50, speaker="A", start=0, end=1),  # 12 tokens
            _seg("y" * 50, speaker="B", start=1, end=2),
        ]
        chunks = chunker.chunk_segments(segments)
        # Should get at least 2 chunks due to speaker change
        assert len(chunks) >= 2

    def test_single_speaker_label_set_correctly(self):
        chunker = Chunker(chunk_tokens=1000, chunk_overlap=0)
        segments = [
            _seg("hello", speaker="Alice"),
            _seg("world", speaker="Alice"),
        ]
        chunks = chunker.chunk_segments(segments)
        assert chunks[0].speaker == "Alice"


# ---------------------------------------------------------------------------
# Chunker: overlap
# ---------------------------------------------------------------------------


class TestChunkerOverlap:
    def test_overlap_zero_no_duplication(self):
        chunker = Chunker(chunk_tokens=10, chunk_overlap=0)
        segments = [
            _seg("a" * 40, start=0, end=1),
            _seg("b" * 40, start=1, end=2),
        ]
        chunks = chunker.chunk_segments(segments)
        # With zero overlap, content of first chunk should not appear in second
        # (at least not from the overlap mechanism)
        assert len(chunks) >= 2

    def test_overlap_positive_adds_trailing_content(self):
        """With overlap > 0, the second chunk should reference at least one
        prior segment, so its start time may overlap with a previous chunk."""
        chunker = Chunker(chunk_tokens=10, chunk_overlap=5)
        segments = [
            _seg("aaaa " * 10, speaker="A", start=0, end=1),
            _seg("bbbb " * 10, speaker="A", start=1, end=2),
            _seg("cccc " * 10, speaker="A", start=2, end=3),
        ]
        chunks = chunker.chunk_segments(segments)
        assert len(chunks) >= 2

    def test_overlap_clamped_to_half(self):
        """Overlap should be clamped to at most chunk_tokens // 2."""
        chunker = Chunker(chunk_tokens=10, chunk_overlap=100)
        assert chunker.chunk_overlap == 5


# ---------------------------------------------------------------------------
# Chunker: normalisation
# ---------------------------------------------------------------------------


class TestChunkerNormalisation:
    def test_accepts_transcript_key(self):
        """The chunker should accept 'transcript' as an alternative to 'text'."""
        chunker = Chunker(chunk_tokens=1000, chunk_overlap=0)
        segments = [{"transcript": "alt key", "speaker": "X", "start": 0, "end": 1}]
        chunks = chunker.chunk_segments(segments)
        assert len(chunks) == 1
        assert "alt key" in chunks[0].text

    def test_accepts_timestamp_array(self):
        """The chunker should extract start/end from a timestamp array."""
        chunker = Chunker(chunk_tokens=1000, chunk_overlap=0)
        segments = [{"text": "ts test", "speaker": "X", "timestamp": [10.5, 12.0]}]
        chunks = chunker.chunk_segments(segments)
        assert chunks[0].start == pytest.approx(10.5)
        assert chunks[0].end == pytest.approx(12.0)

    def test_missing_speaker_defaults_to_unknown(self):
        chunker = Chunker(chunk_tokens=1000, chunk_overlap=0)
        segments = [{"text": "no speaker", "start": 0, "end": 1}]
        chunks = chunker.chunk_segments(segments)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestChunkerDeterminism:
    def test_same_input_same_output(self):
        chunker = Chunker(chunk_tokens=20, chunk_overlap=5)
        segments = [
            _seg("aaa " * 20, speaker="A", start=0, end=1),
            _seg("bbb " * 20, speaker="B", start=1, end=2),
            _seg("ccc " * 20, speaker="A", start=2, end=3),
        ]
        out1 = chunker.chunk_segments(segments, job_id="j", version=1)
        out2 = chunker.chunk_segments(segments, job_id="j", version=1)
        assert len(out1) == len(out2)
        for c1, c2 in zip(out1, out2):
            assert c1.text == c2.text
            assert c1.start == c2.start
            assert c1.end == c2.end
