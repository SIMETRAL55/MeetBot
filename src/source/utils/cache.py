# Backward-compatible shim. Prefer: meetbot.infra.cache.hf_cache
from meetbot.infra.cache.hf_cache import cache_path_for, load_from_cache, save_to_cache

__all__ = ["cache_path_for", "load_from_cache", "save_to_cache"]
