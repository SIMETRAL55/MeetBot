"""
Transcription service orchestration.

This service coordinates audio transcription across multiple backends
with caching and normalization.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

logger = logging.getLogger(__name__)


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

        model_name = settings.WHISPER_MODEL
        params = {}
        if language:
            params["language"] = language

        # Check cache
        raw = None
        cache_path = None
        from_cache = False

        if use_cache and not force_refresh:
            raw = load_from_cache(model_name, audio_path, extra=params)
            if raw is not None:
                cache_path = cache_path_for(model_name, audio_path, extra=params)
                from_cache = True
                logger.info(f"Loaded transcription from cache: {cache_path}")

        # Transcribe if not cached
        if raw is None:
            logger.info("Calling transcriber backend...")
            try:
                # 1. Convert to WAV first
                wav_path = convert_to_wav(audio_path, output_dir="temp")

                # 2. Check file size to determine if chunking is needed
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
                    raw = self._transcribe_with_chunking(
                        wav_path, language=language, **kwargs
                    )
                else:
                    # Direct transcription for small files
                    logger.info(f"File size {wav_size / 1024 / 1024:.1f} MB. Using direct transcription.")
                    raw = self.transcriber.transcribe(wav_path, language=language, **kwargs)

            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                raise

            if use_cache:
                cache_path = save_to_cache(model_name, audio_path, raw, extra=params)
                logger.info(f"Cached transcription: {cache_path}")

        # Normalize segments format
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe large audio file using chunking strategy.

        Breaks audio into overlapping chunks, transcribes each chunk,
        and stitches results back together with coherent absolute timestamps.

        Args:
            wav_path: Path to WAV file (already converted)
            language: Language hint
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
        logger.info("Chunking audio file for transcription...")
        chunks = chunk_audio_for_transcription(
            wav_path,
            max_bytes=settings.AUDIO_CHUNK_SIZE_BYTES,
            nominal_chunk_seconds=settings.AUDIO_CHUNK_NOMINAL_DURATION,
            overlap_seconds=settings.AUDIO_CHUNK_OVERLAP_SECONDS,
            use_silence_detection=settings.AUDIO_CHUNK_USE_SILENCE_DETECTION,
        )

        # 2. Transcribe each chunk
        logger.info(f"Transcribing {len(chunks)} chunks...")
        chunk_results = []
        for chunk in chunks:
            logger.info(
                f"  Chunk {chunk['index']}: {chunk['start']:.1f}s - {chunk['end']:.1f}s"
            )
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
        logger.info("Stitching chunk results...")
        stitched = stitch_chunk_results_to_json(chunk_results)
        logger.info(
            f"✓ Stitched {len(chunks)} chunks into {len(stitched.get('chunks', []))} segments"
        )

        return stitched

