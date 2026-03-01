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
            import torchaudio
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

            # ── Pre-load and resample to 16 kHz ──────────────────────────────
            # Passing the file path directly to pyannote can produce mismatched
            # chunk sizes when the source sample rate is not 16000 Hz (e.g. M4A
            # files are often 44100 Hz). Pyannote's embedder expects exactly
            # 160000 samples (10 s × 16000 Hz) per chunk; a foreign sample rate
            # yields differently-sized chunks that torch.vstack() refuses to
            # stack, raising the "Sizes of tensors must match" error.
            # Solution: load → resample → pass as a {"waveform", "sample_rate"}
            # dict so pyannote works entirely in 16000 Hz space.
            TARGET_SR = 16_000
            waveform, orig_sr = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                # Mix down to mono
                waveform = waveform.mean(dim=0, keepdim=True)

            if orig_sr != TARGET_SR:
                logger.info(
                    f"Resampling audio from {orig_sr} Hz → {TARGET_SR} Hz "
                    "for Pyannote compatibility"
                )
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=TARGET_SR
                )
                waveform = resampler(waveform)

            audio_input = {"waveform": waveform, "sample_rate": TARGET_SR}

            # Run diarization
            logger.info("Running diarization inference...")
            diarization = self.pipeline(
                audio_input,
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

