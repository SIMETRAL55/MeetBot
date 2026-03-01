"""Alignment and formatting services for speaker-attributed transcripts."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Maximum silence gap (seconds) between two consecutive same-speaker segments
# that may be merged into a single output segment.  Gap larger than this value
# preserves the boundary so that pauses / speaker hand-offs are visible.
# Keep small (0.5 s) to avoid collapsing whole speaker turns into one block.
MAX_MERGE_GAP: float = 0.5


class AlignerService:
    """Aligns transcription segments with speaker diarization segments."""

    @staticmethod
    def build_speaker_transcript(
        dia_segments: List[Dict],
        whisper_segments: List[Dict],
        max_merge_gap: float = MAX_MERGE_GAP,
    ) -> List[Dict[str, Any]]:
        """
        Align transcription with speaker diarization.

        Each Whisper segment is assigned to the diarization speaker who has the
        most time-overlap with it.  Adjacent segments — same speaker, gap below
        *max_merge_gap* — are merged into one output segment.

        Args:
            dia_segments: Diarization segments from diarizer.
            whisper_segments: Transcription segments from transcriber.
            max_merge_gap: Maximum silence gap (s) between same-speaker segments
                           that are allowed to merge.  Default 0.5 s.

        Returns:
            List of aligned segments with speaker attribution.
        """
        return _align(dia_segments, whisper_segments, max_merge_gap=max_merge_gap)

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


def _align(
    dia_segments: List[Dict],
    whisper_segments: List[Dict],
    max_merge_gap: float = MAX_MERGE_GAP,
) -> List[Dict[str, Any]]:
    """
    Core alignment: map each Whisper segment to the speaker with maximum
    time-overlap, then merge only adjacent same-speaker segments whose gap
    is within *max_merge_gap* seconds.

    Design constraints
    ------------------
    * Whisper timestamps are the authoritative boundaries.  We never split or
      re-chunk them — only assign speaker labels and optionally merge.
    * Merging requires BOTH same speaker AND small gap.  This preserves pauses
      and prevents whole speaker turns from collapsing into one block.
    """
    logger.debug(
        "Aligning %d Whisper segments against %d diarization segments "
        "(max_merge_gap=%.2f s)",
        len(whisper_segments), len(dia_segments), max_merge_gap,
    )

    # ── Step 1: assign each Whisper segment a speaker label ──────────────
    labelled: List[Dict[str, Any]] = []
    skipped = 0

    for trans in whisper_segments:
        t_start = trans.get("start")
        t_end   = trans.get("end")
        if t_start is None or t_end is None:
            skipped += 1
            continue

        best_speaker: str | None = None
        best_overlap: float = 0.0

        for dia in dia_segments:
            d_start = dia.get("start", 0.0)
            d_end   = dia.get("end",   0.0)
            overlap = max(0.0, min(t_end, d_end) - max(t_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia.get("speaker", "Unknown")

        if best_speaker is None and dia_segments:
            # No overlap at all — fall back to nearest segment mid-point
            mid = (t_start + t_end) / 2.0
            nearest = min(
                dia_segments,
                key=lambda d: abs((d.get("start", 0) + d.get("end", 0)) / 2.0 - mid),
            )
            best_speaker = nearest.get("speaker", "Unknown")

        text = trans.get("text", "").strip()
        if not text:
            # Keep empty segments — dropping them silently hides content.
            # Caller can decide what to do with empty strings.
            pass

        labelled.append({
            "start":   t_start,
            "end":     t_end,
            "speaker": best_speaker or "Unknown",
            "text":    text,
        })

    logger.debug(
        "Labelling done: %d segments assigned, %d skipped (missing timestamps)",
        len(labelled), skipped,
    )

    # ── Step 2: merge only when same speaker AND gap ≤ max_merge_gap ─────
    # The previous code merged unconditionally on same speaker, turning 37
    # segments into 9 by collapsing entire speaker turns.
    merged: List[Dict[str, Any]] = []
    merge_count = 0

    for seg in labelled:
        if merged:
            prev = merged[-1]
            gap  = seg["start"] - prev["end"]
            if prev["speaker"] == seg["speaker"] and gap <= max_merge_gap:
                # Same speaker, tiny gap — extend the previous segment
                prev["end"]   = seg["end"]
                prev["text"] += (" " if prev["text"] else "") + seg["text"]
                merge_count  += 1
                continue
        merged.append(dict(seg))  # new boundary

    logger.info(
        "Alignment complete: %d Whisper → %d labelled → %d final segments "
        "(%d merges, gap threshold=%.2f s)",
        len(whisper_segments), len(labelled), len(merged),
        merge_count, max_merge_gap,
    )
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
    from pathlib import Path

    return {
        "input_file": str(audio_path),
        "segments": aligned_segments,
        "n_segments": len(aligned_segments),
        "duration_seconds": (
            aligned_segments[-1].get("end", 0.0)
            if aligned_segments
            else 0.0
        ),
    }
