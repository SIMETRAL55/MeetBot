# Backward-compatible shim. Prefer: meetbot.services.alignment_service
from meetbot.services.alignment_service import (
    build_speaker_transcript,
    format_result_as_json,
    overlap,
    split_transcript_chunk,
)

__all__ = [
    "overlap",
    "split_transcript_chunk",
    "build_speaker_transcript",
    "format_result_as_json",
]
