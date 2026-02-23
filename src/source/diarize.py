# Backward-compatible shim. Prefer: meetbot.services.diarization_service
from meetbot.services.diarization_service import diarize

__all__ = ["diarize"]
