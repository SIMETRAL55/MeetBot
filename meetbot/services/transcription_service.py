import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from meetbot.adapters.hf_client import HFInferenceClient
from meetbot.config.settings import settings
from meetbot.infra.audio.chunker import transcribe_large_audio
from meetbot.infra.cache.hf_cache import cache_path_for, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)
client = HFInferenceClient()


def transcribe(
    audio_path: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    del task
    params: Dict[str, Any] = {}
    if language:
        params["language"] = language

    model_name = settings.WHISPER_MODEL
    raw = None
    cache_path = None

    if use_cache and not force_refresh:
        raw = load_from_cache(model_name, audio_path, extra=params)
        if raw is not None:
            cache_path = cache_path_for(model_name, audio_path, extra=params)
            logger.info("Loaded Whisper raw response from cache: %s", cache_path)

    if Path(audio_path).suffix.lower() != ".wav":
        logger.info("Input is not wav; chunker/ffmpeg handling will normalize audio chunks.")

    if raw is None:
        logger.info("Calling HF Whisper inference...")
        raw = transcribe_large_audio(
            audio_path,
            client,
            model_name=settings.WHISPER_MODEL,
            max_bytes=20 * 1024 * 1024,
            nominal_chunk_seconds=300,
            overlap_seconds=1.0,
            use_silence_detection=True,
        )
        if use_cache:
            cache_path = save_to_cache(model_name, audio_path, raw["final"], extra=params)
            logger.info("Saved Whisper raw response to cache: %s", cache_path)

    segments: List[Dict[str, Any]] = []
    if isinstance(raw, dict) and "chunks" in raw and isinstance(raw["chunks"], list):
        for c in raw["chunks"]:
            segments.append(
                {
                    "start": float(c["timestamp"][0]) if c.get("timestamp") and c["timestamp"][0] is not None else None,
                    "end": float(c["timestamp"][1]) if c.get("timestamp") and c["timestamp"][1] is not None else None,
                    "text": c.get("text", "").strip(),
                }
            )
    elif isinstance(raw, dict) and "segments" in raw and isinstance(raw["segments"], list):
        for s in raw["segments"]:
            segments.append(
                {
                    "start": float(s["start"]) if s.get("start") is not None else None,
                    "end": float(s["end"]) if s.get("end") is not None else None,
                    "text": s.get("text", "").strip(),
                }
            )
    else:
        segments.append({"start": 0.0, "end": None, "text": raw.get("text", "").strip()})

    return {
        "raw": raw,
        "segments": segments,
        "from_cache": raw is not None and cache_path is not None and use_cache and not force_refresh,
        "cache_path": str(cache_path) if cache_path is not None else None,
    }
