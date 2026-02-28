"""Alignment and formatting services for speaker-attributed transcripts."""

import logging
from typing import List, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)


class AlignerService:
    """Aligns transcription segments with speaker diarization segments."""

    @staticmethod
    def build_speaker_transcript(
        dia_segments: List[Dict],
        whisper_segments: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Align transcription with speaker diarization.

        Args:
            dia_segments: Diarization segments from diarizer
            whisper_segments: Transcription segments from transcriber

        Returns:
            List of aligned segments with speaker attribution
        """
        # Import legacy alignment logic
        import sys
        from pathlib import Path

        sys.path.insert(
            0,
            str(Path(__file__).parent.parent.parent / "source"),
        )

        try:
            from align import build_speaker_transcript as align_legacy
            return align_legacy(dia_segments, whisper_segments)
        except ImportError:
            logger.warning("Could not import legacy alignment, using fallback")
            return _fallback_align(dia_segments, whisper_segments)

    def get_aligned_transcript(
        self,
        audio_path: str,
        diarization: Dict,
        transcription: Dict,
    ) -> List[Dict[str, Any]]:
        """
        Full alignment pipeline with transcript service.

        Args:
            audio_path: Path to original audio
            diarization: Output from DiarizationService
            transcription: Output from TranscriberService

        Returns:
            Aligned speaker transcript segments
        """
        return self.build_speaker_transcript(
            diarization.get("segments", []),
            transcription.get("segments", []),
        )


def _fallback_align(
    dia_segments: List[Dict],
    whisper_segments: List[Dict],
) -> List[Dict[str, Any]]:
    """
    Fallback alignment algorithm if legacy code unavailable.

    Simple strategy: assign each transcription segment to the speaker
    whose diarization segment has maximum overlap.
    """
    aligned = []

    for trans in whisper_segments:
        if trans.get("start") is None or trans.get("end") is None:
            continue

        trans_start = trans["start"]
        trans_end = trans["end"]

        # Find overlapping speaker segment
        best_speaker = None
        best_overlap = 0

        for dia in dia_segments:
            dia_start = dia.get("start", 0)
            dia_end = dia.get("end", 0)

            # Calculate overlap duration
            overlap_start = max(trans_start, dia_start)
            overlap_end = min(trans_end, dia_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia.get("speaker", "Unknown")

        if best_speaker is None and dia_segments:
            # Fallback: assign to nearest speaker
            nearest = min(
                dia_segments,
                key=lambda d: abs((d.get("start", 0) + d.get("end", 0)) / 2 - (trans_start + trans_end) / 2),
            )
            best_speaker = nearest.get("speaker", "Unknown")

        aligned.append(
            {
                "start": trans_start,
                "end": trans_end,
                "speaker": best_speaker or "Unknown",
                "text": trans.get("text", "").strip(),
            }
        )

    # Merge consecutive same-speaker segments
    merged = []
    for seg in aligned:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            # Extend previous segment
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    return merged


def format_result_as_json(
    aligned_segments: List[Dict],
    audio_path: str,
) -> Dict[str, Any]:
    """
    Format aligned segments as final JSON output.

    Args:
        aligned_segments: Output from aligner
        audio_path: Path to original audio file

    Returns:
        Formatted output dictionary ready for JSON serialization
    """
    # Import legacy formatter if available
    import sys
    from pathlib import Path

    sys.path.insert(
        0,
        str(Path(__file__).parent.parent.parent / "source"),
    )

    try:
        from align import format_result_as_json as format_legacy
        return format_legacy(aligned_segments, audio_path)
    except ImportError:
        logger.warning("Could not import legacy formatter, using fallback")
        return _fallback_format(aligned_segments, audio_path)


def _fallback_format(
    aligned_segments: List[Dict],
    audio_path: str,
) -> Dict[str, Any]:
    """Fallback JSON formatter."""
    from pathlib import Path

    return {
        "input_file": str(audio_path),
        "segments": aligned_segments,
        "n_segments": len(aligned_segments),
        "duration_seconds": (
            aligned_segments[-1].get("end", 0)
            if aligned_segments
            else 0
        ),
    }
