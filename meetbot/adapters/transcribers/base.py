"""Abstract base class for transcriber backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTranscriber(ABC):
    """
    Abstract base class for all transcriber implementations.

    Defines the interface for audio transcription backends (local Whisper, HF API, etc.).
    All transcribers must implement the transcribe() method returning timestamped segments.

    Example:
        transcriber = get_transcriber("local")
        result = transcriber.transcribe("audio.wav", language="en")
        # result = {"segments": [{"start": 0.0, "end": 5.0, "text": "Hello"}, ...]}
    """

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.)
            **kwargs: Backend-specific parameters (language, etc.)

        Returns:
            Dict with transcription result containing:
                - segments: List of timestamped segments
                  Example: [{"start": 0.0, "end": 5.0, "text": "Hello"}]
                - text: Full transcription text (optional)
                - language: Detected language (optional)
                - duration: Audio duration (optional)
                - Any other backend-specific fields

        Raises:
            RuntimeError: If transcription fails
            FileNotFoundError: If audio file not found
        """
        pass
