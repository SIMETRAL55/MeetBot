"""Output formatting utilities for MeetBot."""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def format_result_as_json(
    aligned_segments: List[Dict],
    audio_path: str,
) -> Dict[str, Any]:
    """
    Format aligned segments as final JSON output.

    Args:
        aligned_segments: Output from aligner with speaker attribution
        audio_path: Path to original audio file

    Returns:
        Formatted output dictionary ready for JSON serialization

    Example:
        >>> segments = [{"start": 0.0, "end": 5.0, "speaker": "Speaker 1", "text": "Hello"}]
        >>> result = format_result_as_json(segments, "audio.wav")
    """
    audio_file = Path(audio_path).name

    # Calculate total duration
    total_duration = 0.0
    if aligned_segments:
        total_duration = aligned_segments[-1].get("end", 0.0)

    output = {
        "input_file": audio_file,
        "duration_seconds": float(total_duration),
        "segments": aligned_segments,
        "n_segments": len(aligned_segments),
    }

    return output
