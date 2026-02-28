"""Diarization adapters for speaker identification."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LocalPyannoteAdapter:
    """
    Diarization adapter using local Pyannote model.

    Performs speaker diarization directly with Pyannote on local GPU/CPU.
    No HuggingFace API calls - pure local inference.
    """

    def __init__(self):
        """Initialize local diarization adapter."""
        from ..config import settings
        self.api_token = settings.get_hf_token()
        self.pipeline = None

    def diarize_pyannote(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization using local Pyannote model.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            Dict with diarization results including segments with speaker labels and timestamps
        """
        try:
            import torch
            from pyannote.audio import Pipeline

            logger.info(f"Diarizing {Path(audio_path).name} using local Pyannote...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Load Pyannote pipeline (cache it after first load)
            if self.pipeline is None:
                logger.info("Loading Pyannote speaker-diarization-3.1 model...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.api_token,
                )
                self.pipeline.to(torch.device(device))
                logger.info("✓ Pyannote model loaded")

            # Run diarization
            logger.info(f"Running diarization inference...")
            diarization = self.pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # Convert Pyannote output to standard format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "label": speaker,
                })

            logger.info(f"✓ Diarization completed: {len(segments)} speaker segments")
            return {"segments": segments}

        except Exception as e:
            logger.error(f"Local Pyannote diarization failed: {e}")
            raise RuntimeError(f"Failed to perform local diarization: {e}") from e


def get_diarization_adapter() -> LocalPyannoteAdapter:
    """Factory function to get local diarization adapter."""
    return LocalPyannoteAdapter()

