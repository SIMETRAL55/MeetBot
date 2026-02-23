# utils/cache.py
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Tuple

CACHE_DIR = Path(".cache_hf")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def _make_key(model_name: str, audio_path: str, extra: Optional[dict] = None) -> str:
    """
    Deterministic key from model name + absolute audio path + optional extras (e.g. params).
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"|")
    h.update(Path(audio_path).resolve().as_posix().encode("utf-8"))
    if extra:
        # stable serialization
        h.update(json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()

def cache_path_for(model_name: str, audio_path: str, extra: Optional[dict] = None) -> Path:
    key = _make_key(model_name, audio_path, extra)
    return CACHE_DIR / f"{key}.json"

def save_to_cache(model_name: str, audio_path: str, payload: Any, extra: Optional[dict] = None) -> Path:
    """
    Save payload as JSON to cache. If payload is not JSON-serializable,
    try to coerce to str.
    Returns the path written.
    """
    p = cache_path_for(model_name, audio_path, extra)
    data = payload
    try:
        # attempt to dump directly
        with p.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except TypeError:
        # fallback: convert to string
        with p.open("w", encoding="utf-8") as fh:
            json.dump({"raw_text": str(data)}, fh, ensure_ascii=False, indent=2)
    return p

def load_from_cache(model_name: str, audio_path: str, extra: Optional[dict] = None) -> Optional[Any]:
    p = cache_path_for(model_name, audio_path, extra)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        # If corrupt, remove and return None
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        return None
