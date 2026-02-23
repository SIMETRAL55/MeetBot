import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from meetbot.adapters.hf_client import HFInferenceClient
from meetbot.config.settings import settings
from meetbot.infra.audio.convert import convert_to_wav
from meetbot.infra.cache.hf_cache import cache_path_for, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)
client = HFInferenceClient()


def diarize(
    audio_path: str,
    use_cache: bool = True,
    force_refresh: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    model_name = settings.DIARIZATION_MODEL
    params: Dict[str, Any] = {}
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    raw: Any = None
    cache_path = None

    if use_cache and not force_refresh:
        raw = load_from_cache(model_name, audio_path, extra=params)
        if raw is not None:
            cache_path = cache_path_for(model_name, audio_path, extra=params)
            logger.info("Loaded diarization raw response from cache: %s", cache_path)

    converted_audio_path = audio_path
    if Path(audio_path).suffix.lower() != ".wav":
        logger.info("Converting audio for diarization -> WAV (16k mono)")
        converted_audio_path = convert_to_wav(audio_path)

    if raw is None:
        result = client.diarize_pyannote(converted_audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        raw = result.get("raw")
        segments = result.get("segments", [])
        if use_cache:
            cache_path = save_to_cache(model_name, audio_path, {"raw": raw, "segments": segments}, extra=params)
            logger.info("Saved diarization raw response to cache: %s", cache_path)
    else:
        segments = raw.get("segments", []) if isinstance(raw, dict) else []

    normalized: List[Dict[str, Any]] = []
    for seg in segments:
        try:
            normalized.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker": seg.get("speaker", "unknown"),
                }
            )
        except Exception:
            logger.warning("Skipping malformed diarization segment: %s", seg)
    normalized.sort(key=lambda x: x["start"])

    return {
        "raw": raw,
        "segments": normalized,
        "from_cache": raw is not None and cache_path is not None and use_cache and not force_refresh,
        "cache_path": str(cache_path) if cache_path is not None else None,
    }
