"""Diarization adapters for speaker identification."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _load_audio_with_fallback(audio_path: str):
    """
    Load audio as a (waveform, sample_rate) tuple using torchaudio.

    torchaudio >= 2.0 ships its own ffmpeg C++ extension.  When that extension
    is unavailable (e.g. container built on a host without libavcodec headers)
    it falls back to the soundfile backend, which cannot decode M4A/AAC/MP3.

    Strategy:
      1. Try torchaudio.load() directly — fast path for WAV/FLAC/OGG and for
         containers where the ffmpeg extension was compiled successfully.
      2. On format-not-recognised failure, transcode to a temporary WAV via a
         system ffmpeg subprocess (always present in the Docker image) and
         load the WAV instead.  The temp file is deleted immediately after.
    """
    import torchaudio

    # Diagnostic
    try:
        backend = torchaudio.get_audio_backend()
    except Exception:
        backend = "<unknown>"
    logger.debug("torchaudio backend: %s", backend)
    logger.debug("Loading audio: %s", audio_path)

    try:
        return torchaudio.load(audio_path)
    except Exception as primary_err:
        # Only attempt the ffmpeg fallback when the error looks like a codec
        # / format issue — not for file-not-found or permission errors.
        err_str = str(primary_err).lower()
        if not any(k in err_str for k in ("format", "codec", "decode", "recogni", "open")):
            raise

        logger.warning(
            "torchaudio.load() failed for %s (%s) — attempting ffmpeg transcode fallback",
            Path(audio_path).name,
            primary_err,
        )

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError(
                "torchaudio cannot decode the audio file and ffmpeg is not "
                "available as a fallback. Install ffmpeg in your environment."
            ) from primary_err

        logger.info("ffmpeg found at %s — transcoding to temporary WAV", ffmpeg_bin)

        tmp_dir = tempfile.mkdtemp(prefix="meetbot_audio_")
        try:
            tmp_wav = os.path.join(tmp_dir, "converted.wav")
            cmd = [
                ffmpeg_bin,
                "-y",               # overwrite output
                "-i", audio_path,   # input
                "-ar", "16000",     # resample to 16 kHz (target for pyannote anyway)
                "-ac", "1",         # mono
                "-f", "wav",        # force WAV container
                tmp_wav,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg transcode failed (exit {result.returncode}): "
                    f"{result.stderr[-500:]}"
                ) from primary_err

            logger.info("ffmpeg transcode succeeded → %s", tmp_wav)
            waveform, sr = torchaudio.load(tmp_wav)
            return waveform, sr
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
            #
            # _load_audio_with_fallback() handles formats (e.g. M4A/AAC) that
            # torchaudio's soundfile backend cannot decode by transcoding to a
            # temporary WAV via ffmpeg before loading.
            TARGET_SR = 16_000
            waveform, orig_sr = _load_audio_with_fallback(audio_path)

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

    def cleanup(self) -> None:
        """Unload the Pyannote pipeline from memory and free GPU resources."""
        if self.pipeline is not None:
            logger.info("Unloading Pyannote diarization model")
            del self.pipeline
            self.pipeline = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared after Pyannote unload")
            except Exception:
                pass


def get_diarization_adapter() -> LocalPyannoteAdapter:
    """Factory function to get local diarization adapter."""
    return LocalPyannoteAdapter()

