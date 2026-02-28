"""Transcriber adapter implementations and factory."""

from .base import BaseTranscriber
from .factory import get_transcriber, get_transcriber_from_cli_arg
from .local_whisper import LocalWhisperTranscriber
from .huggingface import HuggingFaceTranscriber

__all__ = [
    "BaseTranscriber",
    "get_transcriber",
    "get_transcriber_from_cli_arg",
    "LocalWhisperTranscriber",
    "HuggingFaceTranscriber",
]
