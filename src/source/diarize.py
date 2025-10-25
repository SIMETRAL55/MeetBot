# diarize.py
import logging
from typing import List, Dict, Any, Optional
from utils.cache import load_from_cache, save_to_cache, cache_path_for
from hf_client import HFInferenceClient
from config import settings
from utils.audio_utils import convert_to_wav
from pathlib import Path

logger = logging.getLogger(__name__)
client = HFInferenceClient()

def diarize(audio_path: str,
            use_cache: bool = True,
            force_refresh: bool = False,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None) -> Dict[str, Any]:
    """
    Diarize the given audio file and return:
      {
        "raw": <raw_response>,
        "segments": [ {"start": float, "end": float, "speaker": str}, ... ],
        "from_cache": bool,
        "cache_path": "<path or None>"
      }

    - Converts input to 16k mono WAV before processing (pyannote expects 16k mono).
    - Uses cache keyed on model + original audio path + diarization params.
    """
    model_name = settings.DIARIZATION_MODEL
    params = {}
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    raw = None
    cache_path = None

    # try load cache
    if use_cache and not force_refresh:
        raw = load_from_cache(model_name, audio_path, extra=params)
        if raw is not None:
            cache_path = cache_path_for(model_name, audio_path, extra=params)
            logger.info("Loaded diarization raw response from cache: %s", cache_path)

    # convert to WAV 16k mono before calling pyannote (recommended)
    converted_audio_path = audio_path
    if Path(audio_path).suffix.lower() != ".wav":
        logger.info("Converting audio for diarization -> WAV (16k mono)")
        converted_audio_path = convert_to_wav(audio_path)

    if raw is None:
        logger.info("Calling local pyannote diarization pipeline...")
        # hf_client.diarize_pyannote will run the local pipeline and return normalized segments
        result = client.diarize_pyannote(converted_audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        raw = result.get("raw")
        segments = result.get("segments", [])
        if use_cache:
            cache_path = save_to_cache(model_name, audio_path, {"raw": raw, "segments": segments}, extra=params)
            logger.info("Saved diarization raw response to cache: %s", cache_path)
    else:
        # normalize cached segments
        cached = raw
        # cached expected to contain dict with "raw" and "segments" keys 
        segments = cached.get("segments", []) if isinstance(cached, dict) else []

    # final normalization: ensure floats and sorted
    norm_segments: List[Dict[str, Any]] = []
    for s in segments:
        try:
            start = float(s["start"])
            end = float(s["end"])
            speaker = s.get("speaker", "unknown")
            norm_segments.append({"start": start, "end": end, "speaker": speaker})
        except Exception:
            logger.warning("Skipping malformed diarization segment: %s", s)

    norm_segments = sorted(norm_segments, key=lambda x: x["start"])

    return {
        "raw": raw,
        "segments": norm_segments,
        "from_cache": raw is not None and cache_path is not None and use_cache and not force_refresh,
        "cache_path": str(cache_path) if cache_path is not None else None
    }
