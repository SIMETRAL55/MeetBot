"""HuggingFace API Whisper transcriber."""

import logging
from typing import Any, Dict, Optional

from huggingface_hub import InferenceClient

from .base import BaseTranscriber
from ...config import settings

logger = logging.getLogger(__name__)


class HuggingFaceTranscriber(BaseTranscriber):
    """
    Transcriber using HuggingFace Inference API.

    Features:
    - Uses HuggingFace's Inference API for Whisper transcription
    - Automatic audio chunking for large files
    - Supports language specification
    - Uses configured HuggingFace token from settings/env

    Example:
        >>> transcriber = HuggingFaceTranscriber()
        >>> result = transcriber.transcribe("audio.wav", language="en")
    """

    def __init__(
        self,
        token: Optional[str] = None,
        provider: str = "fal-ai",
    ):
        """
        Initialize HuggingFaceTranscriber.

        Args:
            token: HF API token. If None, uses settings.get_hf_token()
            provider: Provider for InferenceClient (default: "fal-ai")

        Raises:
            ValueError: If no token provided and none in environment
            RuntimeError: If InferenceClient initialization fails
        """
        self.token = token or settings.get_hf_token()
        if not self.token:
            raise ValueError(
                "No HuggingFace token provided. Set HF_API_TOKEN, HF_HUB_TOKEN, or "
                "HUGGINGFACEHUB_API_TOKEN environment variable."
            )

        self.provider = provider

        try:
            self.client = InferenceClient(provider=provider, token=self.token)
            logger.info(f"Initialized HuggingFaceTranscriber with provider={provider}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize HuggingFace InferenceClient: {e}"
            ) from e

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using HuggingFace Whisper API.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, FLAC, etc.)
            language: Language code (e.g., 'en', 'es'). If None, auto-detects.
            **kwargs: Additional parameters (ignored)

        Returns:
            Dict with transcription result containing:
                - chunks or segments: List of segments with timestamps
                - text: Full transcription text (optional)
                - Any other fields returned by HF API

        Raises:
            RuntimeError: If API call fails
            FileNotFoundError: If audio file not found
        """
        model = settings.WHISPER_MODEL
        extra_body: Dict[str, Any] = {}

        # Request timestamps for segments
        extra_body["return_timestamps"] = True

        # Add language if specified
        if language:
            extra_body["generate_kwargs"] = {"language": language}

        logger.info(f"Calling HuggingFace ASR for model={model} on {audio_path}")

        try:
            result = self.client.automatic_speech_recognition(
                audio_path, model=model, extra_body=extra_body
            )
            logger.debug(f"HF API result type: {type(result)}")

            # Convert result to dict if needed
            if isinstance(result, dict):
                return result
            else:
                # Try to convert to dict
                try:
                    converted = dict(result)
                    logger.info("Successfully converted HF result to dict")
                    return converted
                except Exception:
                    try:
                        converted = result.__dict__
                        logger.info("Successfully converted HF result to dict via __dict__")
                        return converted
                    except Exception:
                        logger.warning(
                            "Could not convert HF result to dict, returning as string"
                        )
                        return {"result": str(result), "chunks": []}

        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            raise
        except Exception as e:
            logger.error(f"HuggingFace transcription failed: {e}")
            raise RuntimeError(
                f"Failed to transcribe with HuggingFace: {e}"
            ) from e
