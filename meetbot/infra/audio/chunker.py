# utils/chunker.py
import os
import subprocess
import tempfile
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# constants (override from settings if you prefer)
DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
DEFAULT_TARGET_CODEC = "flac"  # HF supports flac, m4a, opus, etc.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_SECONDS = 120.0  # nominal chunk size in seconds (will be adjusted)
DEFAULT_OVERLAP_SECONDS = 1.0  # small overlap to avoid cutting words


# -------------------------
# helpers: ffprobe / ffmpeg wrappers (robust)
# -------------------------
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run subprocess command and return (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        raise RuntimeError(f"Required command not found: {cmd[0]}")


def ffprobe_duration(path: str) -> Optional[float]:
    """Return duration in seconds using ffprobe, or None if ffprobe unavailable / fails."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ]
        rc, out, err = run_cmd(cmd)
        if rc != 0:
            logger.debug("ffprobe failed: %s", err.strip())
            return None
        return float(out.strip())
    except Exception as e:
        logger.debug("ffprobe_duration error: %s", e)
        return None


def ffmpeg_extract_segment(src: str, dst: str, start: float, duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
                           channels: int = DEFAULT_CHANNELS, codec: str = DEFAULT_TARGET_CODEC) -> None:
    """
    Extract a segment using ffmpeg with re-encoding to desired sample rate / channels / codec.
    Overwrites dst.
    """
    start_str = str(start)
    duration_str = str(duration)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", start_str, "-i", str(src),
        "-t", duration_str,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", codec,
        str(dst)
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg extract failed: {err}")


def detect_silences_ffmpeg(path: str,
                           min_silence_len: float = 0.3,
                           silence_thresh_db: int = -35) -> List[Tuple[float, float]]:
    """
    Detect silences using ffmpeg's silencedetect filter.
    Returns list of (silence_start, silence_end). If ffmpeg missing or fails, returns [].
    Note: ffmpeg outputs silencedetect to stderr; we parse it.
    """
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-af", f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_len}",
            "-f", "null", "-"
        ]
        rc, out, err = run_cmd(cmd)
        # parse lines like: [silencedetect @ 0x...] silence_start: 12.345
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
                        # sometimes only end appears -> use end - min_silence_len as approximate start
                        start = max(0.0, end - min_silence_len)
                    silences.append((start, end))
                    start = None
                except Exception:
                    start = None
        return silences
    except Exception as e:
        logger.debug("detect_silences_ffmpeg failed: %s", e)
        return []


# -------------------------
# chunk boundary calculation
# -------------------------
def estimate_bytes_per_second(path: str, duration: Optional[float] = None) -> Optional[float]:
    """Estimate bytes per second for a file. If duration missing try ffprobe otherwise None."""
    try:
        sz = Path(path).stat().st_size
        if duration is None:
            duration = ffprobe_duration(path)
        if duration and duration > 0:
            return sz / float(duration)
        return None
    except Exception:
        return None


def compute_chunk_duration_for_size(path: str, target_max_bytes: int = DEFAULT_MAX_BYTES,
                                    nominal_chunk_seconds: float = DEFAULT_CHUNK_SECONDS) -> float:
    """
    Compute a safe chunk duration (seconds) so that converted chunk doesn't exceed target_max_bytes.
    Uses bytes/sec estimation from the source file (uncompressed wav) as heuristic and
    clamps to nominal chunk seconds if needed.
    """
    dur = ffprobe_duration(path)
    bps = estimate_bytes_per_second(path, dur)
    if bps is None:
        logger.info("Could not estimate bytes/sec; using nominal chunk seconds %.1fs", nominal_chunk_seconds)
        return nominal_chunk_seconds
    # assume re-encoded compressed flac will be smaller than wav; be conservative and use 0.6 factor
    conservative_bps = bps * 0.6
    if conservative_bps <= 0:
        return nominal_chunk_seconds
    max_seconds = max(1.0, math.floor(target_max_bytes / conservative_bps))
    chosen = min(nominal_chunk_seconds, float(max_seconds))
    if chosen < 1.0:
        # ensure at least 1 second
        chosen = 1.0
    logger.info("Estimated bytes/sec=%.1f, choosing chunk duration=%.1fs (target=%d bytes)",
                conservative_bps, chosen, target_max_bytes)
    return chosen


def generate_chunk_windows(total_duration: float, chunk_seconds: float, overlap_seconds: float) -> List[Tuple[float, float]]:
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


# -------------------------
# Helper: normalize & stitch utilities (new)
# -------------------------
def _ensure_float(x):
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
        # extract from raw shapes
        if isinstance(raw, dict) and isinstance(raw.get("segments"), list):
            segs = raw.get("segments")
        elif isinstance(raw, dict) and isinstance(raw.get("chunks"), list):
            # chunks often have 'timestamp': [s,e]
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
        # s may be a dict-like ASR chunk
        if isinstance(s, dict):
            start = s.get("start") if s.get("start") is not None else (s.get("timestamp") or [None, None])[0]
            end = s.get("end") if s.get("end") is not None else (s.get("timestamp") or [None, None])[1]
            text = s.get("text") or s.get("transcript") or ""
        else:
            # fallback for objects
            start = getattr(s, "start", None)
            end = getattr(s, "end", None)
            text = getattr(s, "text", str(s))

        start = _ensure_float(start)
        end = _ensure_float(end)

        # If start/end appear to be relative (i.e., small numbers) we add base_offset.
        # Heuristic: if base_offset > 0 and start is not None and start < 1e6, but avoid double-adding when start already >= base_offset.
        if start is not None:
            if base_offset and start < (base_offset - 0.1):
                # if start significantly smaller than base_offset, probably relative => add base_offset
                start = start + base_offset
            elif base_offset and start < 10000 and start < base_offset:
                # also treat as relative
                start = start + base_offset
            # else keep as-is (already absolute)
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
    Behavior:
      - Normalizes per-chunk segment times to absolute times using chunk["start"]
      - Sorts segments globally
      - Trims very small overlaps to avoid duplicated words from chunk overlaps
      - Does not discard textual content; it concatenates all texts in chronological order
    """
    if not chunk_results:
        return {"text": "", "chunks": [], "inferred_languages": [], "diarization_segments": []}

    # sort chunk_results by chunk start if present or by index
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
        # aggregate languages
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

        # normalized segments for this chunk (absolute times)
        normalized = _extract_normalized_segments_from_raw(cr)
        for s in normalized:
            # skip completely empty ones
            if s.get("start") is None and s.get("end") is None and not s.get("text"):
                continue
            all_segments.append(s)

    # sort globally by start (None will come last)
    all_segments = sorted(all_segments, key=lambda x: (x["start"] if x["start"] is not None else 1e12))

    # trim overlaps and build chunk list and full text
    # If segments overlap slightly (< overlap_tolerance), we cut the later start to prev end
    overlap_tolerance = 0.25
    merged_segments: List[Dict[str, Any]] = []
    for seg in all_segments:
        if not merged_segments:
            merged_segments.append(seg.copy())
            continue
        prev = merged_segments[-1]
        # if any end missing: we cannot reliably trim, just append
        if prev.get("end") is None or seg.get("start") is None:
            merged_segments.append(seg.copy())
            continue
        # if seg starts before prev end, trim seg.start to prev.end
        if seg["start"] < prev["end"] - overlap_tolerance:
            logger.debug("Trimming overlapping segment start %.3f to prev.end %.3f", seg["start"], prev["end"])
            seg["start"] = prev["end"]
        # drop if invalid
        if seg.get("end") is not None and seg["start"] >= seg["end"]:
            # zero or negative duration after trimming -> skip but keep text by appending to prev if prev has no text duplication
            if seg.get("text"):
                # append text to prev (separated by space) to avoid losing content
                if prev.get("text"):
                    prev["text"] = (prev.get("text").rstrip() + " " + seg.get("text").lstrip()).strip()
                else:
                    prev["text"] = seg.get("text")
            continue
        # otherwise append
        merged_segments.append(seg.copy())

    # build chunks out of merged_segments (atomic)
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
        "diarization_segments": diarization_segments
    }
    return result


# -------------------------
# chunking main function (unchanged)
# -------------------------
def chunk_audio_for_hf(audio_path: str,
                       *,
                       max_bytes: int = DEFAULT_MAX_BYTES,
                       target_codec: str = DEFAULT_TARGET_CODEC,
                       sample_rate: int = DEFAULT_SAMPLE_RATE,
                       channels: int = DEFAULT_CHANNELS,
                       nominal_chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
                       overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
                       use_silence_detection: bool = True,
                       silence_min_len: float = 0.3,
                       silence_thresh_db: int = -35,
                       tmp_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Chunk the audio file into HF-friendly per-chunk files (FLAC by default).
    Returns a list of dicts:
      [{"chunk_path": "<path>", "start": <sec>, "end": <sec>, "duration": <sec>, "index": i}, ... ]
    Notes:
      - always converts the source to 16k mono WAV first (keeps original intact).
      - requires ffmpeg/ffprobe for best results; gracefully degrades if not available.
    """
    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(audio_path)

    tmpdir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="chunks_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    # 1) ensure we have a WAV at 16k mono (use caller's convert_to_wav if available)
    wav_path = src
    try:
        # import lazily to avoid circular imports
        from utils.audio_utils import convert_to_wav as _convert_to_wav
        wav_path = Path(_convert_to_wav(str(src)))
    except Exception:
        # fallback: try ffmpeg re-encode to 16k mono wav
        logger.debug("convert_to_wav not available or failed; falling back to ffmpeg conversion")
        fallback_wav = tmpdir / (src.stem + "_16k_mono.wav")
        ffmpeg_extract_segment(str(src), str(fallback_wav), 0.0, ffprobe_duration(str(src)) or 0.0,
                               sample_rate, channels, "pcm_s16le")
        wav_path = fallback_wav

    # 2) compute duration
    total_dur = ffprobe_duration(str(wav_path)) or 0.0
    if total_dur == 0:
        raise RuntimeError("Could not determine audio duration for chunking.")

    # 3) compute safe chunk duration based on max_bytes
    chunk_seconds = compute_chunk_duration_for_size(str(wav_path),
                                                    target_max_bytes=max_bytes,
                                                    nominal_chunk_seconds=nominal_chunk_seconds)

    # 4) detect silence boundaries (optional)
    silence_windows: List[Tuple[float, float]] = []
    if use_silence_detection:
        try:
            silence_windows = detect_silences_ffmpeg(str(wav_path),
                                                     min_silence_len=silence_min_len,
                                                     silence_thresh_db=silence_thresh_db)
            logger.info("Detected %d silence windows", len(silence_windows))
        except Exception as e:
            logger.debug("Silence detection failed: %s", e)
            silence_windows = []

    # Build a set of candidate snap points (silence midpoints)
    snap_points = sorted({(s + e) / 2.0 for (s, e) in silence_windows})

    # 5) generate windows and snap boundaries to nearest silence (within tolerance)
    windows = generate_chunk_windows(total_dur, chunk_seconds, overlap_seconds)
    final_windows: List[Tuple[float, float]] = []
    SNAP_TOLERANCE = min(3.0, chunk_seconds / 4.0)  # seconds within which we will snap to silence
    for start, end in windows:
        # try snap start to nearest silence after (or before within tol)
        snapped_start = start
        snapped_end = end
        # snap start to nearest snap point in (start - tol, start + tol)
        for p in snap_points:
            if abs(p - start) <= SNAP_TOLERANCE and p <= end:
                snapped_start = max(0.0, p - 0.01)
                break
        # snap end similarly
        for p in reversed(snap_points):
            if abs(p - end) <= SNAP_TOLERANCE and p >= start:
                snapped_end = min(total_dur, p + 0.01)
                break
        # ensure we didn't invert window
        if snapped_end <= snapped_start + 0.05:
            snapped_end = min(total_dur, snapped_start + max(0.5, overlap_seconds + 0.1))
        final_windows.append((round(snapped_start, 6), round(snapped_end, 6)))

    # 6) extract chunk files into target codec (flac) with ffmpeg
    chunks: List[Dict[str, Any]] = []
    for i, (s, e) in enumerate(final_windows):
        dur = max(0.01, e - s)
        out_name = f"{src.stem}_chunk_{i:03d}.{target_codec}"
        out_path = tmpdir / out_name
        try:
            ffmpeg_extract_segment(str(wav_path), str(out_path), s, dur, sample_rate, channels, target_codec)
        except Exception as exc:
            logger.warning("Failed to extract chunk %d, falling back to simple copy: %s", i, exc)
            # fallback to copying whole file for tiny files
            out_path = tmpdir / f"{src.stem}_chunk_{i:03d}.wav"
            ffmpeg_extract_segment(str(wav_path), str(out_path), s, dur, sample_rate, channels, "pcm_s16le")
        chunks.append({"index": i, "chunk_path": str(out_path), "start": s, "end": e, "duration": dur})

    logger.info("Created %d chunks in %s", len(chunks), tmpdir)
    return chunks


# -------------------------
# transcription per-chunk (no caching here) and normalization
# -------------------------
def transcribe_chunks_no_cache(client,
                                 chunks: List[Dict[str, Any]],
                                 model_name: str,
                                 *,
                                 asr_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Transcribe each chunk using the provided HF client. No cache read/write happens here.
    Returns list of dicts:
      [{"index": i, "raw": raw_response, "segments": [...], "start": <chunk_start>, "end": <chunk_end>}, ...]
    """
    results = []
    asr_params = asr_params or {}

    for ch in chunks:
        logger.info("Transcribing chunk %d (start=%.2fs end=%.2fs) -> %s", ch["index"], ch["start"], ch["end"], ch["chunk_path"])
        raw = client.transcribe_whisper(ch["chunk_path"], **(asr_params or {}))
        # normalize segments for chunk (relative->absolute)
        segs = _extract_normalized_segments_from_raw({"raw": raw, "segments": None, "start": ch["start"]})
        results.append({"index": ch["index"], "raw": raw, "segments": segs, "start": ch["start"], "end": ch["end"], "chunk_path": ch["chunk_path"]})
    return results


def stitch_chunked_segments(chunked_results: List[Dict[str, Any]],
                            *,
                            overlap_tolerance: float = 0.25) -> Dict[str, Any]:
    """
    Wrapper to create final JSON (delegates to stitch_chunk_results_to_json).
    """
    return stitch_chunk_results_to_json(chunked_results)


# -------------------------
# high-level helper: full flow for a given audio file
# -------------------------
def transcribe_large_audio(audio_path: str,
                           client,
                           *,
                           model_name: str,
                           max_bytes: int = DEFAULT_MAX_BYTES,
                           codec: str = DEFAULT_TARGET_CODEC,
                           nominal_chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
                           overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
                           use_silence_detection: bool = True,
                           asr_params: Optional[Dict[str, Any]] = None,
                           tmp_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    High-level convenience function:
      - chunk_audio_for_hf(...)
      - transcribe_chunks_no_cache(...)
      - stitch_chunked_segments(...)

    Returns:
      {
        "chunks": [ ... ],
        "chunk_results": [ ... ],
        "final": { "text":..., "chunks": [...], ... },
        "raw_chunks": [raw1, raw2, ...]
      }

    Note: this function does NOT save caches. Calling code (transcribe.py) should save the final JSON if desired.
    """
    asr_params = asr_params or {}
    chunks = chunk_audio_for_hf(audio_path,
                                max_bytes=max_bytes,
                                target_codec=codec,
                                nominal_chunk_seconds=nominal_chunk_seconds,
                                overlap_seconds=overlap_seconds,
                                use_silence_detection=use_silence_detection,
                                tmp_dir=tmp_dir)
    # transcribe each chunk (no cache read/write inside this file)
    chunk_results = transcribe_chunks_no_cache(client, chunks, model_name, asr_params=asr_params)
    # stitch into final JSON structure
    final = stitch_chunked_segments(chunk_results)
    raw_list = [c.get("raw") for c in chunk_results]
    return {"chunks": chunks, "chunk_results": chunk_results, "final": final, "raw_chunks": raw_list}
