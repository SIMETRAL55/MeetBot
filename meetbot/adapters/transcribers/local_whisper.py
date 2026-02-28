"""Local Whisper transcriber using OpenAI's Whisper model with GPU-support."""

import logging
from typing import Any, Dict, Optional

from .base import BaseTranscriber

logger = logging.getLogger(__name__)


class LocalWhisperTranscriber(BaseTranscriber):
    """
    Transcriber using local OpenAI Whisper model with GPU support.

    Features:
    - Runs entirely locally (no API calls)
    - GPU acceleration via CUDA (auto-detects)
    - CPU fallback if CUDA unavailable
    - Singleton model management (load once, reuse)
    - Configurable model sizes (tiny, base, small, medium, large)

    Example:
        >>> transcriber = LocalWhisperTranscriber(model_size="base")
        >>> result = transcriber.transcribe("audio.wav", language="en")
    """

    _model = None  # Class-level singleton for model

    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        """
        Initialize LocalWhisperTranscriber.

        Args:
            model_size: Whisper model size
                       ('tiny', 'base', 'small', 'medium', 'large')
                       Larger models are more accurate but slower.
            device: Device to use ('cuda' or 'cpu'). If None, auto-detects.

        Raises:
            RuntimeError: If openai-whisper not installed or model load fails
        """
        self.model_size = model_size
        self.device = device or self._detect_device()
        self._load_model()

    @staticmethod
    def _detect_device() -> str:
        """
        Detect available device (CUDA or CPU).

        Returns:
            str: 'cuda' if available, 'cpu' otherwise
        """
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA detected: using GPU for Whisper inference")
                return "cuda"
        except ImportError:
            logger.warning("torch not installed; falling back to CPU")
        except Exception as e:
            logger.warning(f"Error detecting CUDA: {e}; falling back to CPU")

        logger.warning(
            "Using CPU for Whisper inference (slower). "
            "Install torch + CUDA for GPU acceleration."
        )
        return "cpu"

    def _load_model(self):
        """
        Load Whisper model (singleton pattern - only load once per process).

        Uses class-level _model singleton to avoid reloading on each init.

        Raises:
            RuntimeError: If model loading fails
        """
        if LocalWhisperTranscriber._model is not None:
            logger.debug("Reusing cached Whisper model")
            return

        try:
            import whisper

            logger.info(
                f"Loading Whisper model '{self.model_size}' on device '{self.device}'..."
            )
            LocalWhisperTranscriber._model = whisper.load_model(
                self.model_size, device=self.device
            )
            logger.info("✓ Whisper model loaded successfully")
        except ImportError as e:
            raise RuntimeError(
                "openai-whisper not installed. Install with: pip install openai-whisper"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using local Whisper model.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, FLAC, etc.)
            language: Language code (e.g., 'en', 'es', 'fr'). If None, auto-detects.
            **kwargs: Additional parameters for whisper.transcribe()
                     (task, beam_size, best_of, temperature, etc.)

        Returns:
            Dict with transcription result:
                {
                    "text": "Full transcription text",
                    "segments": [
                        {
                            "id": 0,
                            "seek": 0,
                            "start": 0.0,
                            "end": 5.0,
                            "text": " Segment text",
                            "tokens": [50364, 3231, 364],
                            "temperature": 0.0,
                            "avg_logprob": -0.5,
                            "compression_ratio": 1.5,
                            "no_speech_prob": 0.001
                        },
                        ...
                    ],
                    "language": "en"
                }

        Raises:
            RuntimeError: If transcription fails
            FileNotFoundError: If audio file not found
        """
        if LocalWhisperTranscriber._model is None:
            raise RuntimeError("Model not loaded")

        logger.info(
            f"Transcribing {audio_path} with Whisper "
            f"(model={self.model_size}, device={self.device})"
        )

        # Set default parameters for GPU-safe transcription
        transcribe_params = {
            "fp16": self.device == "cuda",  # Use fp16 only on CUDA
            "temperature": 0.0,  # Deterministic
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,  # More robust splicing
            "verbose": False,
        }

        # Allow overrides from kwargs
        transcribe_params.update(kwargs)

        # Add language if specified
        if language:
            transcribe_params["language"] = language

        try:
            result = LocalWhisperTranscriber._model.transcribe(
                audio_path, **transcribe_params
            )
            logger.info(
                f"✓ Transcription completed: {len(result.get('segments', []))} segments"
            )
            return result
        except FileNotFoundError as e:
            logger.error(f"Audio file not found: {audio_path}")
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe {audio_path}: {e}") from e

    @classmethod
    def cleanup(cls):
        """
        Unload the model from memory and free GPU resources.

        Call this on shutdown or before loading a different model.
        """
        if cls._model is not None:
            logger.info("Unloading Whisper model")
            cls._model = None

            # Also clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
            except Exception:
                pass
