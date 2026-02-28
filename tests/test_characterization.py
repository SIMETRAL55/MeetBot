"""
Characterization tests for MeetBot core modules.

These tests capture the current behavior of critical modules before refactoring.
They serve as regression tests to ensure behavior doesn't change during refactoring.
"""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from source.align import (
    overlap,
    split_transcript_chunk,
    build_speaker_transcript,
    format_result_as_json,
)
from source.utils.cache import (
    _make_key,
    cache_path_for,
    save_to_cache,
    load_from_cache,
    CACHE_DIR,
)


class TestOverlapFunction:
    """Test the overlap calculation function."""

    def test_basic_overlap(self):
        """Test basic interval overlap."""
        assert overlap(1.0, 3.0, 2.0, 4.0) == 1.0
        assert overlap(1.0, 2.0, 2.0, 3.0) == 0.0
        assert overlap(1.0, 4.0, 2.0, 3.0) == 1.0

    def test_no_overlap(self):
        """Test intervals that don't overlap."""
        assert overlap(1.0, 2.0, 3.0, 4.0) == 0.0
        assert overlap(5.0, 6.0, 1.0, 2.0) == 0.0

    def test_complete_overlap(self):
        """Test one interval completely within another."""
        assert overlap(1.0, 10.0, 2.0, 3.0) == 1.0
        assert overlap(2.0, 3.0, 1.0, 10.0) == 1.0

    def test_none_values(self):
        """Test handling of None values."""
        assert overlap(None, 3.0, 2.0, 4.0) == 0.0
        assert overlap(1.0, None, 2.0, 4.0) == 0.0
        assert overlap(1.0, 3.0, None, 4.0) == 0.0
        assert overlap(1.0, 3.0, 2.0, None) == 0.0

    def test_identical_intervals(self):
        """Test identical intervals."""
        assert overlap(1.0, 3.0, 1.0, 3.0) == 2.0


class TestSplitTranscriptChunk:
    """Test transcript splitting at boundaries."""

    def test_no_cuts_within_chunk(self):
        """Test when no cut times fall within chunk."""
        chunk = {"start": 1.0, "end": 3.0, "text": "hello world"}
        cuts = [0.5, 5.0]
        result = split_transcript_chunk(chunk, cuts)
        assert len(result) == 1
        assert result[0]["text"] == "hello world"
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 3.0

    def test_single_cut_within_chunk(self):
        """Test splitting with one cut point."""
        chunk = {"start": 1.0, "end": 3.0, "text": "hello world"}
        cuts = [2.0]
        result = split_transcript_chunk(chunk, cuts)
        assert len(result) == 2
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 2.0
        assert result[1]["start"] == 2.0
        assert result[1]["end"] == 3.0

    def test_multiple_cuts_within_chunk(self):
        """Test splitting with multiple cut points."""
        chunk = {"start": 0.0, "end": 4.0, "text": "a b c d"}
        cuts = [1.0, 2.0, 3.0]
        result = split_transcript_chunk(chunk, cuts)
        assert len(result) == 4
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.0

    def test_empty_text(self):
        """Test chunk with empty text."""
        chunk = {"start": 1.0, "end": 3.0, "text": ""}
        cuts = [2.0]
        result = split_transcript_chunk(chunk, cuts)
        assert len(result) == 2
        for part in result:
            assert part["text"] == ""

    def test_invalid_chunk_ignored(self):
        """Test invalid chunks (None times or invalid ranges)."""
        chunk = {"start": None, "end": 3.0, "text": "hello"}
        result = split_transcript_chunk(chunk, [2.0])
        assert len(result) == 1
        assert result[0] == chunk

    def test_end_less_than_start(self):
        """Test when end <= start."""
        chunk = {"start": 3.0, "end": 1.0, "text": "hello"}
        result = split_transcript_chunk(chunk, [2.0])
        assert len(result) == 1
        assert result[0] == chunk


class TestBuildSpeakerTranscript:
    """Test speaker assignment and merging logic."""

    def test_simple_alignment(self):
        """Test basic whisper-to-diarization alignment."""
        whisper = [
            {"start": 0.0, "end": 2.0, "text": "hello world"},
            {"start": 2.0, "end": 4.0, "text": "foo bar"},
        ]
        diarization = [
            {"start": 0.0, "end": 3.0, "speaker": "Speaker_0"},
            {"start": 3.0, "end": 4.0, "speaker": "Speaker_1"},
        ]
        result = build_speaker_transcript(diarization, whisper)
        assert all("speaker" in item for item in result)
        assert all("start" in item and "end" in item for item in result)

    def test_merging_same_speaker(self):
        """Test that contiguous same-speaker segments merge."""
        whisper = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        diarization = [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker_0"},
        ]
        result = build_speaker_transcript(diarization, whisper)
        # Should merge into one segment
        assert len(result) <= 2

    def test_missing_whisper_times(self):
        """Test handling of whisper segments with missing start/end."""
        whisper = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": None, "end": 2.0, "text": "world"},  # invalid
            {"start": 2.0, "end": 3.0, "text": "foo"},
        ]
        diarization = [
            {"start": 0.0, "end": 3.0, "speaker": "Speaker_0"},
        ]
        result = build_speaker_transcript(diarization, whisper)
        # Should skip the invalid segment and process the valid ones
        assert all("start" in item for item in result)

    def test_fallback_to_nearest_speaker(self):
        """Test fallback when no overlap found."""
        whisper = [
            {"start": 10.0, "end": 11.0, "text": "isolated"},
        ]
        diarization = [
            {"start": 0.0, "end": 1.0, "speaker": "Speaker_0"},
            {"start": 5.0, "end": 6.0, "speaker": "Speaker_1"},
        ]
        result = build_speaker_transcript(diarization, whisper)
        # Should assign to nearest speaker
        assert result[0]["speaker"] in ["Speaker_0", "Speaker_1"]

    def test_empty_inputs(self):
        """Test with empty inputs."""
        result = build_speaker_transcript([], [])
        assert result == []

        result = build_speaker_transcript(
            [{"start": 0.0, "end": 1.0, "speaker": "Speaker_0"}], []
        )
        assert result == []

    def test_diarization_None_values(self):
        """Test diarization segments with None values."""
        whisper = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
        ]
        diarization = [
            {"start": None, "end": 1.0, "speaker": "Speaker_0"},
            {"start": 0.0, "end": None, "speaker": "Speaker_1"},
        ]
        # Should not crash
        result = build_speaker_transcript(diarization, whisper)
        assert len(result) > 0


class TestFormatResultAsJson:
    """Test JSON formatting and output file handling."""

    def test_basic_formatting(self):
        """Test basic JSON formatting."""
        transcript = [
            {"start": 0.0, "end": 1.5, "speaker": "Speaker_0", "text": "hello"},
            {"start": 1.5, "end": 3.0, "speaker": "Speaker_1", "text": "world"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_result_as_json(transcript, "test.wav", tmpdir)
            assert len(result) == 2
            assert result[0]["speaker"] == "Speaker 1"
            assert result[1]["speaker"] == "Speaker 2"

    def test_speaker_id_mapping(self):
        """Test that speaker IDs are correctly mapped."""
        transcript = [
            {"start": 0.0, "end": 1.0, "speaker": "Speaker_0", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker": "Speaker_1", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker": "Speaker_0", "text": "c"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_result_as_json(transcript, "test.wav", tmpdir)
            # Same speaker IDs should map to same names
            assert result[0]["speaker"] == result[2]["speaker"]
            assert result[1]["speaker"] != result[0]["speaker"]

    def test_timestamp_rounding(self):
        """Test that timestamps are rounded to 2 decimals."""
        transcript = [
            {
                "start": 1.23456,
                "end": 2.98765,
                "speaker": "Speaker_0",
                "text": "test",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_result_as_json(transcript, "test.wav", tmpdir)
            assert result[0]["start"] == 1.23
            assert result[0]["end"] == 2.99

    def test_missing_fields(self):
        """Test handling of missing fields."""
        transcript = [
            {"speaker": "Speaker_0", "text": "hello"},  # missing start/end
            {"start": 1.0, "end": 2.0, "text": "world"},  # missing speaker
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_result_as_json(transcript, "test.wav", tmpdir)
            assert result[0]["start"] == 0.0  # default
            # Second speaker (unknown) gets mapped to its assigned ID
            assert result[1]["speaker"] in ["Speaker 1", "Speaker 2"]

    def test_file_creation(self):
        """Test that output file is actually created."""
        transcript = [
            {"start": 0.0, "end": 1.0, "speaker": "Speaker_0", "text": "hello"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            format_result_as_json(transcript, "myaudio.wav", tmpdir)
            output_file = Path(tmpdir) / "myaudio.json"
            assert output_file.exists()


class TestCacheKeyGeneration:
    """Test cache key generation determinism."""

    def test_key_determinism(self):
        """Test that same inputs produce same keys."""
        key1 = _make_key("model1", "audio.wav", {"param": "value"})
        key2 = _make_key("model1", "audio.wav", {"param": "value"})
        assert key1 == key2

    def test_different_models_different_keys(self):
        """Test that different models produce different keys."""
        key1 = _make_key("model1", "audio.wav")
        key2 = _make_key("model2", "audio.wav")
        assert key1 != key2

    def test_different_paths_different_keys(self):
        """Test that different audio paths produce different keys."""
        key1 = _make_key("model1", "/path/audio1.wav")
        key2 = _make_key("model1", "/path/audio2.wav")
        assert key1 != key2

    def test_different_params_different_keys(self):
        """Test that different params produce different keys."""
        key1 = _make_key("model1", "audio.wav", {"param": "value1"})
        key2 = _make_key("model1", "audio.wav", {"param": "value2"})
        assert key1 != key2

    def test_param_order_irrelevant(self):
        """Test that parameter order doesn't matter."""
        key1 = _make_key("model", "audio.wav", {"a": 1, "b": 2})
        key2 = _make_key("model", "audio.wav", {"b": 2, "a": 1})
        assert key1 == key2


class TestCacheSaveLoad:
    """Test cache save and load operations."""

    def test_save_and_load_simple(self):
        """Test saving and loading simple JSON data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("source.utils.cache.CACHE_DIR", Path(tmpdir)):
                data = {"key": "value", "number": 42}
                path = save_to_cache("model", "audio.wav", data)
                loaded = load_from_cache("model", "audio.wav")
                assert loaded == data

    def test_save_and_load_complex_structure(self):
        """Test with nested structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("source.utils.cache.CACHE_DIR", Path(tmpdir)):
                data = {
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "hello"},
                        {"start": 1.0, "end": 2.0, "text": "world"},
                    ],
                    "metadata": {"model": "test"},
                }
                save_to_cache("model", "audio.wav", data)
                loaded = load_from_cache("model", "audio.wav")
                assert loaded == data

    def test_cache_miss(self):
        """Test that missing cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("source.utils.cache.CACHE_DIR", Path(tmpdir)):
                result = load_from_cache("model", "audio.wav")
                assert result is None

    def test_corrupt_cache_cleanup(self):
        """Test that corrupt cache files are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            with patch("source.utils.cache.CACHE_DIR", cache_dir):
                # Create a corrupt cache file
                key = _make_key("model", "audio.wav")
                cache_file = cache_dir / f"{key}.json"
                cache_file.parent.mkdir(exist_ok=True, parents=True)
                cache_file.write_text("{ invalid json")

                # Try to load - should remove corrupt file
                result = load_from_cache("model", "audio.wav")
                assert result is None
                assert not cache_file.exists()  # corrupt file was removed

    def test_save_with_extra_params(self):
        """Test saving with extra parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("source.utils.cache.CACHE_DIR", Path(tmpdir)):
                data = {"result": "success"}
                extra = {"language": "en", "beam_size": 5}
                path = save_to_cache("model", "audio.wav", data, extra)
                loaded = load_from_cache("model", "audio.wav", extra)
                assert loaded == data

    def test_save_non_json_serializable(self):
        """Test saving non-JSON-serializable objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("source.utils.cache.CACHE_DIR", Path(tmpdir)):
                # Mock object that's not JSON-serializable
                class CustomObj:
                    def __str__(self):
                        return "custom_object"

                obj = CustomObj()
                path = save_to_cache("model", "audio.wav", obj)
                loaded = load_from_cache("model", "audio.wav")
                # Should be converted to string representation
                assert loaded is not None
                assert "raw_text" in loaded


class TestCachePathGeneration:
    """Test cache path generation."""

    def test_cache_path_format(self):
        """Test that cache paths have expected format."""
        path = cache_path_for("model", "audio.wav")
        assert path.suffix == ".json"
        assert len(path.stem) == 64  # sha256 hex is 64 chars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
