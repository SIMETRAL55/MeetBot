"""Document preparation service for RAG encoding.

.. deprecated::
    This module is superseded by ``services.rag.chunker.Chunker`` which
    provides speaker-aware token-level chunking with overlap.  No production
    code paths import this module any longer.  It is retained only for
    reference and will be removed in a future release.
"""
import warnings as _warnings
_warnings.warn(
    "meetbot.services.prepare_docs is deprecated — use meetbot.services.rag.chunker instead",
    DeprecationWarning,
    stacklevel=2,
)

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regex to extract speaker prefix from text (e.g., "SPEAKER_00: text here")
SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z0-9_\-\s]{1,40}):\s*(.*)$")


class PrepareDocsService:
    """
    Prepare transcription/diarization output for vector indexing.

    Converts aligned transcript JSON to JSONL documents with metadata,
    including speaker information and timestamps for RAG retrieval.
    """

    def __init__(self):
        """Initialize document preparation service."""
        pass

    @staticmethod
    def _extract_speaker_and_text(raw_text: str) -> Tuple[Optional[str], str]:
        """
        Extract speaker prefix from text if present.

        Args:
            raw_text: Text that may start with "SPEAKER_00: ..."

        Returns:
            Tuple of (speaker_label_or_none, cleaned_text)
        """
        if not raw_text:
            return None, ""

        match = SPEAKER_PREFIX_RE.match(raw_text)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            return speaker, text

        return None, raw_text.strip()

    @staticmethod
    def _get_timestamp(segment: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract start/end timestamps from segment (flexible format handling).

        Args:
            segment: Segment dict with various possible timestamp formats

        Returns:
            Tuple of (start_time_or_none, end_time_or_none)
        """
        # Try direct keys
        if "start" in segment or "end" in segment:
            try:
                start = segment.get("start")
                end = segment.get("end")
                return (
                    float(start) if start is not None else None,
                    float(end) if end is not None else None,
                )
            except (ValueError, TypeError):
                pass

        # Try timestamp array
        timestamps = (
            segment.get("timestamp")
            or segment.get("timestamps")
            or segment.get("time")
            or segment.get("ts")
        )
        if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
            try:
                return (
                    float(timestamps[0]) if timestamps[0] is not None else None,
                    float(timestamps[1]) if timestamps[1] is not None else None,
                )
            except (ValueError, TypeError):
                pass

        return None, None

    @staticmethod
    def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
        """
        Load either JSON array, JSON object with segments field, or JSONL file format.

        Supports three formats:
        1. JSON array: [{segment}, {segment}, ...]
        2. JSON object with segments: {"segments": [{segment}, ...], ...}
        3. JSONL: one segment per line

        Args:
            path: Path to JSON array, JSON object, or JSONL file

        Returns:
            List of segment dictionaries

        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []

        # Check if JSON array or JSON object
        first_char = text.lstrip()[0]
        if first_char in ("[", "{"):
            try:
                data = json.loads(text)

                # Case 1: JSON array
                if isinstance(data, list):
                    return data

                # Case 2: JSON object with 'segments' field (from format_result_as_json)
                if isinstance(data, dict) and "segments" in data:
                    segments = data["segments"]
                    if isinstance(segments, list):
                        return segments
                    else:
                        raise ValueError("Expected 'segments' field to be a list")

                # Case 3: Single JSON object (treat as segments list with one item)
                if isinstance(data, dict):
                    return [data]

                raise ValueError(f"Unexpected JSON structure: {type(data)}")

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}") from e

        # Load JSONL format (one object per line)
        docs = []
        for line_num, line in enumerate(text.split("\n"), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
                docs.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

        return docs

    def prepare(
        self,
        transcript_json: str,
        output_dir: str = "prepared",
    ) -> Tuple[List[Dict[str, Any]], Path]:
        """
        Prepare transcript/diarization JSON for vector indexing.

        Args:
            transcript_json: Path to transcript JSON from aligner service
            output_dir: Directory to save prepared JSONL

        Returns:
            Tuple of (documents_list, output_path)
                - documents_list: List of prepared document dicts with id, text, metadata
                - output_path: Path to written JSONL file

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input format is invalid
        """
        input_path = Path(transcript_json)
        output_path = Path(output_dir)

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Load input
        logger.info(f"Loading transcript from {input_path}")
        try:
            segments = self._load_json_or_jsonl(input_path)
        except Exception as e:
            logger.error(f"Failed to load input: {e}")
            raise

        audio_basename = input_path.stem
        output_file = output_path / f"{audio_basename}.jsonl"

        # Process segments
        documents = []
        for idx, segment in enumerate(segments):
            # Extract text
            text = (
                segment.get("text")
                or segment.get("transcript")
                or segment.get("chunk_text")
                or ""
            ).strip()

            # Extract speaker
            speaker = (
                segment.get("speaker")
                or segment.get("spk")
                or segment.get("speaker_label")
            )

            if speaker is None:
                # Try to extract from text prefix
                extracted_speaker, cleaned_text = self._extract_speaker_and_text(text)
                if extracted_speaker:
                    speaker = extracted_speaker
                    text = cleaned_text

            # Normalize empty speaker
            if not speaker:
                speaker = "unknown"

            # Extract timestamps
            start, end = self._get_timestamp(segment)

            # Build document text (include speaker in search text)
            if speaker and speaker.lower() != "unknown":
                doc_text = f"{speaker}: {text}".strip()
            else:
                doc_text = text

            # Create document
            doc = {
                "id": f"{audio_basename}_{idx}",
                "text": doc_text,
                "metadata": {
                    "audio_file": audio_basename,
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "chunk_id": idx,
                },
            }
            documents.append(doc)

        # Write JSONL output
        logger.info(f"Writing {len(documents)} documents to {output_file}")
        with output_file.open("w", encoding="utf-8") as fh:
            for doc in documents:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info(f"✓ Prepared {len(documents)} documents -> {output_file}")
        return documents, output_file
