"""Factory for creating transcriber instances based on backend selection."""

import logging
import os
from typing import Optional

from .base import BaseTranscriber
from .huggingface import HuggingFaceTranscriber
from .local_whisper import LocalWhisperTranscriber

logger = logging.getLogger(__name__)


def get_transcriber(backend: Optional[str] = None, **kwargs) -> BaseTranscriber:
    """
    Create a transcriber instance based on backend selection.

    Priority order for backend selection:
    1. Explicit `backend` parameter
    2. TRANSCRIPTION_BACKEND environment variable
    3. Default to 'huggingface'

    Args:
        backend: Backend type ('local' or 'huggingface'). If None, reads from
                 TRANSCRIPTION_BACKEND env var or defaults to 'huggingface'.
        **kwargs: Backend-specific keyword arguments

    Returns:
        BaseTranscriber instance (LocalWhisperTranscriber or HuggingFaceTranscriber)

    Raises:
        ValueError: If backend is unknown
        RuntimeError: If backend initialization fails

    Example:
        # Use environment variable
        transcriber = get_transcriber()

        # Override with explicit backend
        transcriber = get_transcriber(backend="local", model_size="base")
    """
    if backend is None:
        backend = os.getenv("TRANSCRIPTION_BACKEND", "huggingface").lower()

    logger.info(f"Creating transcriber with backend: {backend}")

    if backend == "local":
        logger.debug("Using LocalWhisperTranscriber")
        # Extract local-specific kwargs
        model_size = kwargs.pop("model_size", "small")
        device = kwargs.pop("device", None)
        return LocalWhisperTranscriber(model_size=model_size, device=device)

    elif backend == "huggingface":
        logger.debug("Using HuggingFaceTranscriber")
        # Extract HF-specific kwargs
        token = kwargs.pop("token", None)
        provider = kwargs.pop("provider", "fal-ai")
        return HuggingFaceTranscriber(token=token, provider=provider)

    else:
        raise ValueError(
            f"Unknown transcription backend: {backend}. Valid values: 'local', 'huggingface'"
        )


def get_transcriber_from_cli_arg(backend_arg: Optional[str] = None) -> BaseTranscriber:
    """
    Create a transcriber instance from CLI argument.

    Convenience function that wraps get_transcriber() for CLI use.

    Args:
        backend_arg: Backend type from CLI argument (overrides env var if provided)

    Returns:
        BaseTranscriber instance

    Raises:
        ValueError: If backend is unknown
        RuntimeError: If backend initialization fails
    """
    return get_transcriber(backend=backend_arg)
