"""
Audio chunking utilities for handling large audio files.

Breaks large audio files into overlapping chunks with smart silence-detection
boundaries, transcribes each chunk, and stitches results back together with
absolute timestamps preserved.

This module adapts the chunking strategy from the original src/source/utils/chunker.py
to work seamlessly with the modular MeetBot architecture.
"""

import subprocess
import tempfile
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants for chunking behavior
DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB WAV file threshold
DEFAULT_CHUNK_SECONDS = 120.0  # ~2 minute chunks
DEFAULT_OVERLAP_SECONDS = 1.0  # 1 second overlap between chunks
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run subprocess command and return (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        raise RuntimeError(f"Required command not found: {cmd[0]}")


def ffprobe_duration(path: str) -> Optional[float]:
    """Return duration in seconds using ffprobe, or None if ffprobe unavailable/fails."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        rc, out, err = run_cmd(cmd)
        if rc != 0:
            logger.debug("ffprobe failed: %s", err.strip())
            return None
        return float(out.strip())
    except Exception as e:
        logger.debug("ffprobe_duration error: %s", e)
        return None


def ffmpeg_extract_segment(
    src: str,
    dst: str,
    start: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    codec: str = "pcm_s16le",
) -> None:
    """Extract audio segment using ffmpeg with re-encoding to desired parameters."""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        str(src),
        "-t",
        str(duration),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        codec,
        str(dst),
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg extract failed: {err}")


def detect_silences_ffmpeg(
    path: str, min_silence_len: float = 0.3, silence_thresh_db: int = -35
) -> List[Tuple[float, float]]:
    """
    Detect silences using ffmpeg's silencedetect filter.
    Returns list of (silence_start, silence_end). If ffmpeg missing or fails, returns [].
    """
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_len}",
            "-f",
            "null",
            "-",
        ]
        rc, out, err = run_cmd(cmd)
        # Parse lines like: [silencedetect @ 0x...] silence_start: 12.345
        silences = []
        start = None
        for line in err.splitlines():
            line = line.strip()
            if "silence_start:" in line:
                try:
                    start = float(line.split("silence_start:")[1].strip())
                except Exception:
                    start = None
            elif "silence_end:" in line:
                try:
                    end_part = line.split("silence_end:")[1].split("|")[0].strip()
                    end = float(end_part)
                    if start is None:
                        start = max(0.0, end - min_silence_len)
                    silences.append((start, end))
                    start = None
                except Exception:
                    start = None
        return silences
    except Exception as e:
        logger.debug("detect_silences_ffmpeg failed: %s", e)
        return []


def estimate_bytes_per_second(path: str, duration: Optional[float] = None) -> Optional[float]:
    """Estimate bytes per second for a file."""
    try:
        sz = Path(path).stat().st_size
        if duration is None:
            duration = ffprobe_duration(path)
        if duration and duration > 0:
            return sz / float(duration)
        return None
    except Exception:
        return None


def compute_chunk_duration_for_size(
    path: str, target_max_bytes: int = DEFAULT_MAX_BYTES, nominal_chunk_seconds: float = DEFAULT_CHUNK_SECONDS
) -> float:
    """
    Compute safe chunk duration (seconds) so that converted chunk doesn't exceed target_max_bytes.
    """
    dur = ffprobe_duration(path)
    bps = estimate_bytes_per_second(path, dur)
    if bps is None:
        logger.info("Could not estimate bytes/sec; using nominal chunk seconds %.1fs", nominal_chunk_seconds)
        return nominal_chunk_seconds
    # Conservative estimate (WAV is larger than compressed formats)
    conservative_bps = bps * 0.8
    if conservative_bps <= 0:
        return nominal_chunk_seconds
    max_seconds = max(1.0, math.floor(target_max_bytes / conservative_bps))
    chosen = min(nominal_chunk_seconds, float(max_seconds))
    if chosen < 1.0:
        chosen = 1.0
    logger.info(
        "Estimated bytes/sec=%.1f, choosing chunk duration=%.1fs (target=%d bytes)",
        conservative_bps,
        chosen,
        target_max_bytes,
    )
    return chosen


def generate_chunk_windows(
    total_duration: float, chunk_seconds: float, overlap_seconds: float
) -> List[Tuple[float, float]]:
    """
    Return list of (start, end) windows covering [0, total_duration], with specified overlap.
    """
    if total_duration <= 0:
        return []
    step = max(0.1, chunk_seconds - overlap_seconds)
    windows = []
    start = 0.0
    while start < total_duration:
        end = min(total_duration, start + chunk_seconds)
        windows.append((round(start, 6), round(end, 6)))
        start = start + step
    return windows


# ============================================================================
# Segment normalization and stitching
# ============================================================================

def _ensure_float(x):
    """Safely convert to float."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _extract_normalized_segments_from_raw(chunk_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize one chunk_result entry into list of segments with absolute times.

    chunk_result expected keys:
      - "start": absolute start offset of this chunk (seconds) OR None
      - "raw": the model raw response (dict) possibly containing "segments" or "chunks"
      - "segments": optional pre-normalized segments (relative or absolute)

    Returns: list of {"start": float_or_None, "end": float_or_None, "text": str}
    """
    out = []
    base_offset = _ensure_float(chunk_result.get("start") or 0.0)

    segs = chunk_result.get("segments")
    raw = chunk_result.get("raw") or {}

    if not segs:
        # Extract from raw shapes
        if isinstance(raw, dict) and isinstance(raw.get("segments"), list):
            segs = raw.get("segments")
        elif isinstance(raw, dict) and isinstance(raw.get("chunks"), list):
            # Chunks often have 'timestamp': [s,e]
            segs = []
            for c in raw.get("chunks", []):
                ts = c.get("timestamp") or c.get("time") or [None, None]
                segs.append({"start": ts[0], "end": ts[1], "text": c.get("text", "")})
        elif isinstance(raw, dict) and "text" in raw:
            segs = [{"start": 0.0, "end": None, "text": raw.get("text", "")}]
        elif isinstance(raw, list):
            segs = raw
        else:
            segs = []

    for s in segs:
        # s may be a dict-like ASR segment
        if isinstance(s, dict):
            start = s.get("start") if s.get("start") is not None else (s.get("timestamp") or [None, None])[0]
            end = s.get("end") if s.get("end") is not None else (s.get("timestamp") or [None, None])[1]
            text = s.get("text") or s.get("transcript") or ""
        else:
            # Fallback for objects
            start = getattr(s, "start", None)
            end = getattr(s, "end", None)
            text = getattr(s, "text", str(s))

        start = _ensure_float(start)
        end = _ensure_float(end)

        # If start/end appear to be relative, add base_offset
        if start is not None:
            if base_offset and start < (base_offset - 0.1):
                start = start + base_offset
            elif base_offset and start < 10000 and start < base_offset:
                start = start + base_offset
        if end is not None:
            if base_offset and end < (base_offset - 0.1):
                end = end + base_offset
            elif base_offset and end < 10000 and end < base_offset:
                end = end + base_offset

        out.append({"start": start, "end": end, "text": (text or "").strip()})
    return out


def stitch_chunk_results_to_json(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert list of per-chunk results into final single JSON structure with:
      - text (full concatenated)
      - chunks: [ {text, timestamp: [start,end]} ... ]
      - inferred_languages: aggregated list
      - diarization_segments: concatenated (if any)

    Normalizes per-chunk segment times to absolute times using chunk["start"]
    Sorts segments globally and trims overlaps.
    """
    if not chunk_results:
        return {"text": "", "chunks": [], "inferred_languages": [], "diarization_segments": []}

    # Sort chunk_results by chunk start
    def _key(cr):
        if cr.get("start") is not None:
            return float(cr.get("start"))
        if cr.get("index") is not None:
            return int(cr.get("index"))
        return 0

    chunk_results_sorted = sorted(chunk_results, key=_key)

    inferred_languages = []
    diarization_segments = []
    all_segments: List[Dict[str, Any]] = []

    for cr in chunk_results_sorted:
        raw = cr.get("raw") or {}
        # Aggregate languages
        if isinstance(raw, dict):
            langs = raw.get("inferred_languages") or raw.get("languages") or raw.get("language")
            if isinstance(langs, list):
                for L in langs:
                    if L and L not in inferred_languages:
                        inferred_languages.append(L)
            elif isinstance(langs, str) and langs not in inferred_languages:
                inferred_languages.append(langs)
            ds = raw.get("diarization_segments")
            if isinstance(ds, list) and ds:
                for dseg in ds:
                    diarization_segments.append(dseg)

        # Normalized segments for this chunk (absolute times)
        normalized = _extract_normalized_segments_from_raw(cr)
        for s in normalized:
            # Skip completely empty ones
            if s.get("start") is None and s.get("end") is None and not s.get("text"):
                continue
            all_segments.append(s)

    # Sort globally by start
    all_segments = sorted(all_segments, key=lambda x: (x["start"] if x["start"] is not None else 1e12))

    # Trim overlaps and build chunk list
    overlap_tolerance = 0.25
    merged_segments: List[Dict[str, Any]] = []
    for seg in all_segments:
        if not merged_segments:
            merged_segments.append(seg.copy())
            continue
        prev = merged_segments[-1]
        # If any end missing: cannot reliably trim, just append
        if prev.get("end") is None or seg.get("start") is None:
            merged_segments.append(seg.copy())
            continue
        # If seg starts before prev end, trim seg.start to prev.end
        if seg["start"] < prev["end"] - overlap_tolerance:
            logger.debug("Trimming overlapping segment start %.3f to prev.end %.3f", seg["start"], prev["end"])
            seg["start"] = prev["end"]
        # Drop if invalid
        if seg.get("end") is not None and seg["start"] >= seg["end"]:
            # Zero or negative duration -> skip, but preserve text
            if seg.get("text"):
                if prev.get("text"):
                    prev["text"] = (prev.get("text").rstrip() + " " + seg.get("text").lstrip()).strip()
                else:
                    prev["text"] = seg.get("text")
            continue
        # Otherwise append
        merged_segments.append(seg.copy())

    # Build chunks from merged segments
    chunks_out = []
    full_text_parts = []
    for s in merged_segments:
        st = s.get("start")
        ed = s.get("end")
        txt = (s.get("text") or "").strip()
        chunks_out.append({"text": txt, "timestamp": [st, ed]})
        if txt:
            full_text_parts.append(txt)

    full_text = " ".join(full_text_parts).strip()

    result = {
        "text": full_text,
        "chunks": chunks_out,
        "inferred_languages": inferred_languages,
        "diarization_segments": diarization_segments,
    }
    return result


# ============================================================================
# Main chunking function
# ============================================================================

def chunk_audio_for_transcription(
    audio_path: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    nominal_chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    use_silence_detection: bool = True,
    silence_min_len: float = 0.3,
    silence_thresh_db: int = -35,
    tmp_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk audio file into transcribable segments.

    Returns a list of dicts:
      [{"chunk_path": "<path>", "start": <sec>, "end": <sec>, "duration": <sec>, "index": i}, ... ]

    Args:
        audio_path: Path to audio file (any format)
        max_bytes: Maximum WAV file size before chunking (bytes)
        sample_rate: Target sample rate (Hz)
        channels: Target channels (1=mono)
        nominal_chunk_seconds: Ideal chunk duration (seconds)
        overlap_seconds: Overlap between consecutive chunks
        use_silence_detection: Whether to snap to silence boundaries
        silence_min_len: Minimum silence duration to detect (seconds)
        silence_thresh_db: Silence threshold (dB)
        tmp_dir: Directory for temporary chunk files

    Returns:
        List of chunk metadata dicts
    """
    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(audio_path)

    tmpdir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="chunks_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure we have a WAV at target sample rate and channels
    wav_path = src
    try:
        # Try using existing convert_to_wav if in package
        from .audio import convert_to_wav as _convert_to_wav
        wav_path = Path(_convert_to_wav(str(src), output_dir=str(tmpdir)))
    except Exception:
        # Fallback to ffmpeg conversion
        logger.debug("convert_to_wav not available; falling back to ffmpeg conversion")
        fallback_wav = tmpdir / (src.stem + "_16k_mono.wav")
        dur = ffprobe_duration(str(src))
        if dur is None:
            raise RuntimeError(f"Could not determine audio duration for {src}")
        ffmpeg_extract_segment(str(src), str(fallback_wav), 0.0, dur, sample_rate, channels, "pcm_s16le")
        wav_path = fallback_wav

    # 2. Compute duration
    total_dur = ffprobe_duration(str(wav_path)) or 0.0
    if total_dur == 0:
        raise RuntimeError("Could not determine audio duration for chunking.")

    # 3. Compute safe chunk duration based on max_bytes
    chunk_seconds = compute_chunk_duration_for_size(
        str(wav_path), target_max_bytes=max_bytes, nominal_chunk_seconds=nominal_chunk_seconds
    )

    # 4. Detect silence boundaries (optional)
    silence_windows: List[Tuple[float, float]] = []
    if use_silence_detection:
        try:
            silence_windows = detect_silences_ffmpeg(
                str(wav_path), min_silence_len=silence_min_len, silence_thresh_db=silence_thresh_db
            )
            logger.info("Detected %d silence windows", len(silence_windows))
        except Exception as e:
            logger.debug("Silence detection failed: %s", e)
            silence_windows = []

    # Build a set of candidate snap points (silence midpoints)
    snap_points = sorted({(s + e) / 2.0 for (s, e) in silence_windows})

    # 5. Generate windows and snap boundaries to nearest silence
    windows = generate_chunk_windows(total_dur, chunk_seconds, overlap_seconds)
    final_windows: List[Tuple[float, float]] = []
    SNAP_TOLERANCE = min(3.0, chunk_seconds / 4.0)

    for start, end in windows:
        snapped_start = start
        snapped_end = end
        # Snap start to nearest snap point
        for p in snap_points:
            if abs(p - start) <= SNAP_TOLERANCE and p <= end:
                snapped_start = max(0.0, p - 0.01)
                break
        # Snap end similarly
        for p in reversed(snap_points):
            if abs(p - end) <= SNAP_TOLERANCE and p >= start:
                snapped_end = min(total_dur, p + 0.01)
                break
        # Ensure window is valid
        if snapped_end <= snapped_start + 0.05:
            snapped_end = min(total_dur, snapped_start + max(0.5, overlap_seconds + 0.1))
        final_windows.append((round(snapped_start, 6), round(snapped_end, 6)))

    # 6. Extract chunk files (WAV format for transcription)
    chunks: List[Dict[str, Any]] = []
    for i, (s, e) in enumerate(final_windows):
        dur = max(0.01, e - s)
        out_name = f"{src.stem}_chunk_{i:03d}.wav"
        out_path = tmpdir / out_name
        try:
            ffmpeg_extract_segment(str(wav_path), str(out_path), s, dur, sample_rate, channels, "pcm_s16le")
        except Exception as exc:
            logger.warning("Failed to extract chunk %d: %s", i, exc)
            raise
        chunks.append({"index": i, "chunk_path": str(out_path), "start": s, "end": e, "duration": dur})

    logger.info("Created %d chunks in %s", len(chunks), tmpdir)
    return chunks
