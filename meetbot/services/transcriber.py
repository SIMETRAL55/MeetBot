"""
Transcription service orchestration.

This service coordinates audio transcription across multiple backends
with caching and normalization.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List, Callable

logger = logging.getLogger(__name__)

# Progress callback type: (stage: str, progress: float 0-100, message: str) -> None
ProgressCallback = Callable[[str, float, str], None]


class TranscriberService:
    """
    High-level transcription service orchestration.

    Handles:
    - Backend selection
    - Caching
    - Result normalization
    """

    def __init__(self, transcriber=None):
        """
        Initialize transcriber service.

        Args:
            transcriber: BaseTranscriber instance (e.g., from get_transcriber())
        """
        self.transcriber = transcriber

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with caching support.

        For large files, automatically uses chunking strategy to handle
        files that would exceed Whisper's size limits. Transparent to caller.

        Args:
            audio_path: Path to audio file
            language: Language hint for Whisper
            use_cache: Whether to use cache
            force_refresh: Force fresh transcription
            progress_callback: Optional callback for progress updates (stage, progress 0-100, message)
            **kwargs: Additional parameters

        Returns:
            Dict with keys:
                - segments: List of transcription segments
                - from_cache: Whether result was cached
                - cache_path: Path to cache file if caching enabled
        """
        from ..utils.cache import load_from_cache, save_to_cache, cache_path_for
        from ..utils.audio import convert_to_wav
        from ..config import settings

        if not self.transcriber:
            from ..adapters.transcribers import get_transcriber
            self.transcriber = get_transcriber()

        if progress_callback:
            progress_callback("transcription", 5, "Initializing transcription service...")

        model_name = settings.WHISPER_MODEL
        params = {}
        if language:
            params["language"] = language

        # Check cache
        raw = None
        cache_path = None
        from_cache = False

        if use_cache and not force_refresh:
            if progress_callback:
                progress_callback("transcription", 10, "Checking cache...")
            raw = load_from_cache(model_name, audio_path, extra=params)
            if raw is not None:
                cache_path = cache_path_for(model_name, audio_path, extra=params)
                from_cache = True
                if progress_callback:
                    progress_callback("transcription", 100, "✓ Loaded from cache")
                logger.info(f"Loaded transcription from cache: {cache_path}")

        # Transcribe if not cached
        if raw is None:
            logger.info("Calling transcriber backend...")
            try:
                # 1. Convert to WAV first
                if progress_callback:
                    progress_callback("transcription", 15, "Converting audio to WAV format...")
                wav_path = convert_to_wav(audio_path, output_dir="temp")

                try:
                    # 2. Check file size to determine if chunking is needed
                    if progress_callback:
                        progress_callback("transcription", 20, "Analyzing audio file...")
                    wav_size = Path(wav_path).stat().st_size
                    should_chunk = (
                        settings.AUDIO_CHUNK_ENABLE
                        and wav_size >= settings.AUDIO_CHUNK_SIZE_BYTES
                    )

                    if should_chunk:
                        logger.info(
                            f"File size {wav_size / 1024 / 1024:.1f} MB exceeds threshold "
                            f"({settings.AUDIO_CHUNK_SIZE_BYTES / 1024 / 1024:.1f} MB). Using chunking strategy."
                        )
                        if progress_callback:
                            progress_callback("transcription", 25, f"File size: {wav_size / 1024 / 1024:.1f} MB - using chunking strategy...")
                        raw = self._transcribe_with_chunking(
                            wav_path, language=language, progress_callback=progress_callback, **kwargs
                        )
                    else:
                        # Direct transcription for small files
                        logger.info(f"File size {wav_size / 1024 / 1024:.1f} MB. Using direct transcription.")
                        if progress_callback:
                            progress_callback("transcription", 30, f"Transcribing audio (size: {wav_size / 1024 / 1024:.1f} MB)...")
                        raw = self.transcriber.transcribe(wav_path, language=language, **kwargs)
                        if progress_callback:
                            progress_callback("transcription", 80, "Transcription complete")
                finally:
                    # Always clean up the temporary WAV file after transcription
                    try:
                        import os as _os
                        if Path(wav_path).exists():
                            _os.remove(wav_path)
                            logger.debug(f"Cleaned up temporary WAV: {wav_path}")
                    except Exception as _cleanup_err:
                        logger.warning(f"Could not clean up temp WAV {wav_path}: {_cleanup_err}")

            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                if progress_callback:
                    progress_callback("transcription", 0, f"✗ Error: {str(e)}")
                raise

            if use_cache:
                if progress_callback:
                    progress_callback("transcription", 85, "Caching results...")
                cache_path = save_to_cache(model_name, audio_path, raw, extra=params)
                logger.info(f"Cached transcription: {cache_path}")

        # Normalize segments format
        if progress_callback:
            progress_callback("transcription", 90, "Normalizing transcript format...")
        segments = []
        if isinstance(raw, dict) and "chunks" in raw:
            for c in raw["chunks"]:
                seg = {
                    "start": float(c["timestamp"][0])
                    if c.get("timestamp") and c["timestamp"][0] is not None
                    else None,
                    "end": float(c["timestamp"][1])
                    if c.get("timestamp") and c["timestamp"][1] is not None
                    else None,
                    "text": c.get("text", "").strip(),
                }
                if seg["start"] is not None and seg["end"] is not None:
                    segments.append(seg)
        elif isinstance(raw, dict) and "segments" in raw:
            segments = raw["segments"]

        if progress_callback:
            progress_callback("transcription", 100, f"✓ Transcription complete ({len(segments)} segments)")

        return {
            "raw": raw,
            "segments": segments,
            "from_cache": from_cache,
            "cache_path": str(cache_path) if cache_path else None,
        }

    def _transcribe_with_chunking(
        self,
        wav_path: str,
        language: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe large audio file using chunking strategy.

        Breaks audio into overlapping chunks, transcribes each chunk,
        and stitches results back together with coherent absolute timestamps.

        Args:
            wav_path: Path to WAV file (already converted)
            language: Language hint
            progress_callback: Optional callback for progress updates
            **kwargs: Additional transcription parameters

        Returns:
            Dict with stitched transcription results
        """
        from ..utils.audio_chunker import (
            chunk_audio_for_transcription,
            stitch_chunk_results_to_json,
            _extract_normalized_segments_from_raw,
        )
        from ..config import settings

        # 1. Chunk the audio
        if progress_callback:
            progress_callback("transcription", 30, "Chunking large audio file...")
        logger.info("Chunking audio file for transcription...")
        chunks = chunk_audio_for_transcription(
            wav_path,
            max_bytes=settings.AUDIO_CHUNK_SIZE_BYTES,
            nominal_chunk_seconds=settings.AUDIO_CHUNK_NOMINAL_DURATION,
            overlap_seconds=settings.AUDIO_CHUNK_OVERLAP_SECONDS,
            use_silence_detection=settings.AUDIO_CHUNK_USE_SILENCE_DETECTION,
        )

        if progress_callback:
            progress_callback("transcription", 35, f"Created {len(chunks)} chunks - starting transcription...")

        # 2. Transcribe each chunk
        logger.info(f"Transcribing {len(chunks)} chunks...")
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(
                f"  Chunk {chunk['index']}: {chunk['start']:.1f}s - {chunk['end']:.1f}s"
            )
            if progress_callback:
                chunk_progress = 35 + (50 * (i / len(chunks)))  # 35-85%
                progress_callback("transcription", chunk_progress, f"Transcribing chunk {i +1}/{len(chunks)}...")
            try:
                raw_chunk = self.transcriber.transcribe(
                    chunk["chunk_path"], language=language, **kwargs
                )
                # Normalize segment timestamps to absolute times
                normalized_segs = _extract_normalized_segments_from_raw(
                    {"raw": raw_chunk, "segments": None, "start": chunk["start"]}
                )
                chunk_results.append(
                    {
                        "index": chunk["index"],
                        "raw": raw_chunk,
                        "segments": normalized_segs,
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "chunk_path": chunk["chunk_path"],
                    }
                )
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {chunk['index']}: {e}")
                raise

        # 3. Stitch results together
        if progress_callback:
            progress_callback("transcription", 85, "Stitching chunk results...")
        logger.info("Stitching chunk results...")
        stitched = stitch_chunk_results_to_json(chunk_results)
        logger.info(
            f"✓ Stitched {len(chunks)} chunks into {len(stitched.get('chunks', []))} segments"
        )

        return stitched

