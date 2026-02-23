# Backward-compatible shim. Prefer: meetbot.services.transcription_service
from meetbot.services.transcription_service import transcribe

__all__ = ["transcribe"]
