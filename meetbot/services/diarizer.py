"""Diarization service for speaker identification and temporal segmentation."""

import logging
import os
from typing import Optional, Any, Dict, Callable

logger = logging.getLogger(__name__)

# Progress callback type: (stage: str, progress: float 0-100, message: str) -> None
ProgressCallback = Callable[[str, float, str], None]


class DiarizationService:
    """
    Speaker diarization service using Pyannote.

    Identifies and segments speakers in audio with temporal boundaries.
    """

    def __init__(self):
        """Initialize diarization service."""
        self._adapter = None

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            use_cache: Whether to use cache
            force_refresh: Force fresh diarization
            progress_callback: Optional callback for progress updates (stage, progress 0-100, message)
            **kwargs: Additional parameters

        Returns:
            Dict with keys:
                - segments: List of speaker segments
                - from_cache: Whether result was cached
                - cache_path: Path to cache file if applicable
        """
        # MIGRATION: Gradually replace with native implementation
        # For now, import legacy code
        from ..utils.cache import load_from_cache, save_to_cache, cache_path_for
        from ..config import settings

        if progress_callback:
            progress_callback("diarization", 5, "Initializing diarization service...")

        model_name = settings.DIARIZATION_MODEL
        cache_key = {"min": min_speakers, "max": max_speakers}

        # Check cache
        raw = None
        cache_path = None
        from_cache = False

        if use_cache and not force_refresh:
            if progress_callback:
                progress_callback("diarization", 10, "Checking cache...")
            raw = load_from_cache(model_name, audio_path, extra=cache_key)
            if raw is not None:
                cache_path = cache_path_for(model_name, audio_path, extra=cache_key)
                from_cache = True
                if progress_callback:
                    progress_callback("diarization", 100, "✓ Loaded from cache")
                logger.info(f"Loaded diarization from cache: {cache_path}")

        # Perform diarization if not cached
        if raw is None:
            if progress_callback:
                progress_callback("diarization", 20, "Loading diarization model...")
            logger.info("Running diarization pipeline...")

            try:
                from ..adapters.diarization import get_diarization_adapter

                if progress_callback:
                    progress_callback("diarization", 40, "Processing audio for speaker identification...")
                adapter = get_diarization_adapter()
                self._adapter = adapter
                raw = adapter.diarize_pyannote(
                    audio_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                if progress_callback:
                    progress_callback("diarization", 80, "Diarization complete")
            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                if progress_callback:
                    progress_callback("diarization", 0, f"✗ Error: {str(e)}")
                raise

            if use_cache:
                if progress_callback:
                    progress_callback("diarization", 85, "Caching results...")
                cache_path = save_to_cache(model_name, audio_path, raw, extra=cache_key)
                logger.info(f"Cached diarization: {cache_path}")

        # Extract and normalize segments
        if progress_callback:
            progress_callback("diarization", 90, "Normalizing speaker segments...")
        segments = raw.get("segments", []) if isinstance(raw, dict) else []

        if progress_callback:
            progress_callback("diarization", 100, f"✓ Diarization complete ({len(segments)} segments)")

        return {
            "raw": raw,
            "segments": segments,
            "from_cache": from_cache,
            "cache_path": str(cache_path) if cache_path else None,
        }

    def cleanup(self) -> None:
        """Release GPU resources held by the diarization adapter."""
        if self._adapter is not None:
            try:
                self._adapter.cleanup()
            except Exception:
                pass
            self._adapter = None
