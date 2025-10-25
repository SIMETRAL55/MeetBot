# transcribe.py
from hf_client import HFInferenceClient
from typing import List, Dict, Any
from utils.chunker import transcribe_large_audio
from utils.cache import load_from_cache, save_to_cache, cache_path_for
from config import settings
import logging
from utils.audio_utils import convert_to_wav
from pathlib import Path

logger = logging.getLogger(__name__)
client = HFInferenceClient()

def transcribe(audio_path: str, language: str = None, task: str = "transcribe",
               use_cache: bool = True, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Transcribe an audio file using Hugging Face Whisper API with caching and format conversion.
    Returns:
      {
        "raw": <raw_response>,
        "segments": [...],
        "from_cache": True|False,
        "cache_path": "<path or None>"
      }
    """
    params = {}
    if language:
        params["language"] = language

    model_name = settings.WHISPER_MODEL
    raw = None
    cache_path = None

    # Use cache if available
    if use_cache and not force_refresh:
        raw = load_from_cache(model_name, audio_path, extra=params)
        if raw is not None:
            cache_path = cache_path_for(model_name, audio_path, extra=params)
            logger.info(f"Loaded Whisper raw response from cache: {cache_path}")

    # Convert to WAV for Whisper if needed
    # converted_audio_path = audio_path
    if Path(audio_path).suffix.lower() != ".wav":
        logger.info(f"Converting {audio_path} to WAV for Whisper API...")
        # converted_audio_path = convert_to_wav(audio_path)
        # converted_audio_path = "/home/alkris/whisper/temp/Kanto_Radio.wav"

    # Call API if no cache found
    if raw is None:
        logger.info("Calling HF Whisper inference...")
        # raw = client.transcribe_whisper(converted_audio_path, **params)
        raw = transcribe_large_audio(
                audio_path,         # input audio
                client,
                model_name=settings.WHISPER_MODEL,
                max_bytes=20*1024*1024,    # keep as before
                nominal_chunk_seconds=300, # or tweak
                overlap_seconds=1.0,
                use_silence_detection=True
        )
        
        logger.info("HF Whisper inference completed.")
        print("==========RAW OUTPUT WHISPER==============")
        print(raw["final"])
        if use_cache:
            cache_path = save_to_cache(model_name, audio_path, raw["final"], extra=params)
            logger.info(f"Saved Whisper raw response to cache: {cache_path}")
            
    segments: List[Dict[str, Any]] = []

    if isinstance(raw, dict) and "chunks" in raw and isinstance(raw["chunks"], list):
        # If your raw uses "chunks"
        for c in raw["chunks"]:
            seg = {
                "start": float(c["timestamp"][0]) if c.get("timestamp") and c["timestamp"][0] is not None else None,
                "end": float(c["timestamp"][1]) if c.get("timestamp") and c["timestamp"][1] is not None else None,
                "text": c.get("text", "").strip()
            }
            segments.append(seg)
    elif isinstance(raw, dict) and "segments" in raw and isinstance(raw["segments"], list):
        # Or if raw uses "segments"
        for s in raw["segments"]:
            seg = {
                "start": float(s["start"]) if s.get("start") is not None else None,
                "end": float(s["end"]) if s.get("end") is not None else None,
                "text": s.get("text", "").strip()
            }
            segments.append(seg)
    else:
        # fallback: whole text
        segments.append({"start": 0.0, "end": None, "text": raw.get("text", "").strip()})

    return {
        "raw": raw,
        "segments": segments,
        "from_cache": raw is not None and cache_path is not None and use_cache and not force_refresh,
        "cache_path": str(cache_path) if cache_path is not None else None
    }