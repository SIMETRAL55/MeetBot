"""Caching utilities for transcription and diarization results."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get cache directory from config or use default."""
    try:
        from ...config import settings
        return settings.get_cache_dir()
    except Exception:
        # Fallback if config import fails
        cache_dir = Path(".cache_hf")
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir


_CACHE_DIR: Optional[Path] = None


def cache_dir() -> Path:
    """Get or create cache directory (lazy-initialized)."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = get_cache_dir()
    return _CACHE_DIR


def _make_key(
    model_name: str,
    audio_path: str,
    extra: Optional[dict] = None,
) -> str:
    """
    Create a deterministic cache key.

    Uses SHA256 hash of model name + absolute audio path + optional parameters.
    Ensures same inputs always produce same key, and different inputs never collide.

    Args:
        model_name: Name of the model used (e.g., "openai/whisper-large-v3")
        audio_path: Path to audio file (converted to absolute path)
        extra: Optional dict of parameters (e.g., {"language": "en"})

    Returns:
        str: 64-character hex SHA256 hash
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"|")
    h.update(Path(audio_path).resolve().as_posix().encode("utf-8"))
    if extra:
        # Stable serialization (sorted keys)
        h.update(
            json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
    return h.hexdigest()


def cache_path_for(
    model_name: str,
    audio_path: str,
    extra: Optional[dict] = None,
) -> Path:
    """
    Get cache file path for model + audio + parameters.

    Returns:
        Path: Path to cache JSON file (may not exist yet)

    Example:
        cache_file = cache_path_for("openai/whisper-large-v3", "audio.wav")
        # Returns: Path(".cache_hf/a1b2c3d4e5f6....json")
    """
    key = _make_key(model_name, audio_path, extra)
    return cache_dir() / f"{key}.json"


def save_to_cache(
    model_name: str,
    audio_path: str,
    payload: Any,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save payload to cache as JSON.

    If payload is not directly JSON-serializable, attempts to convert to string.
    Creates parent directory if needed.

    Args:
        model_name: Name of the model used
        audio_path: Path to audio file
        payload: Data to cache (should be JSON-serializable dict)
        extra: Optional parameters dict

    Returns:
        Path: Path to cache file written

    Example:
        result = transcriber.transcribe("audio.wav")
        save_to_cache("openai/whisper-large-v3", "audio.wav", result)
    """
    p = cache_path_for(model_name, audio_path, extra)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        with p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.debug(f"Cached to: {p}")
    except TypeError:
        # Fallback: convert non-JSON-serializable to string
        logger.warning(
            f"Payload not JSON-serializable, converting to string and caching"
        )
        with p.open("w", encoding="utf-8") as fh:
            json.dump(
                {"raw_text": str(payload)},
                fh,
                ensure_ascii=False,
                indent=2,
            )

    return p


def load_from_cache(
    model_name: str,
    audio_path: str,
    extra: Optional[dict] = None,
) -> Optional[Any]:
    """
    Load cached result if available.

    Returns None if cache file doesn't exist or is corrupted (and removes corrupt file).

    Args:
        model_name: Name of the model used
        audio_path: Path to audio file
        extra: Optional parameters dict

    Returns:
        dict or None: Cached data if available, None otherwise

    Example:
        cached_result = load_from_cache("openai/whisper-large-v3", "audio.wav")
        if cached_result:
            print("Using cached transcription")
    """
    p = cache_path_for(model_name, audio_path, extra)

    if not p.exists():
        return None

    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug(f"Loaded from cache: {p}")
        return data
    except Exception as e:
        logger.warning(f"Cache file corrupted: {p}, removing ({e})")
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        return None
