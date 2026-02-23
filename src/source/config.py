# Backward-compatible shim. Prefer: meetbot.config.settings
from meetbot.config.settings import Settings, settings

__all__ = ["Settings", "settings"]
