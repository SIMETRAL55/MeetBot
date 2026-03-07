"""
Speaker-aware transcript chunker with configurable token size and overlap.

Produces chunks from aligned transcript segments, respecting speaker
boundaries where possible and attaching metadata (start/end timestamps,
speaker labels) for downstream RAG retrieval.

The chunker is deterministic given the same inputs and settings.

Streaming API
-------------
``chunk_segments_iter()`` yields :class:`Chunk` objects lazily so that
downstream consumers (e.g. the JSONL writer in the indexer) never need
to hold all chunks in memory simultaneously.  The convenience method
``chunk_segments()`` materialises the iterator into a list for callers
that need random access.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata for vector indexing."""

    chunk_id: int
    text: str
    speaker: str
    start: float
    end: float
    job_id: str = ""
    version: int = 1
    segment_indices: List[int] = field(default_factory=list)

    def to_doc(self) -> Dict[str, Any]:
        """Convert to the document format expected by the indexer."""
        # Serialise segment_indices as a comma-separated string for Chroma
        # (Chroma metadata values must be str/int/float/bool, not lists).
        seg_ids_str = ",".join(str(i) for i in self.segment_indices)
        return {
            "id": f"{self.job_id}_{self.chunk_id}" if self.job_id else str(self.chunk_id),
            "text": self.text,
            "metadata": {
                "level": "chunk",
                "job_id": self.job_id,
                "chunk_id": self.chunk_id,
                "segment_ids": seg_ids_str,
                "speaker": self.speaker,
                "start": self.start,
                "end": self.end,
                "version": self.version,
            },
        }


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for multilingual text."""
    return max(1, len(text) // 4)


class Chunker:
    """
    Speaker-aware transcript chunker.

    Combines consecutive segments into chunks of approximately
    ``chunk_tokens`` tokens, with ``chunk_overlap`` token overlap between
    adjacent chunks.  Speaker boundaries are respected: a chunk break is
    always preferred at a speaker change.

    Parameters
    ----------
    chunk_tokens : int
        Target chunk size in estimated tokens (default 300).
    chunk_overlap : int
        Number of overlap tokens between consecutive chunks (default 50).
    """

    def __init__(
        self,
        chunk_tokens: int = 300,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_tokens = max(1, chunk_tokens)
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_tokens // 2))

    def chunk_segments(
        self,
        segments: List[Dict[str, Any]],
        job_id: str = "",
        version: int = 1,
    ) -> List[Chunk]:
        """
        Chunk a list of aligned transcript segments.

        Each segment dict should have at minimum:
            - text (str)
            - speaker (str)
            - start (float)
            - end (float)

        Returns a list of Chunk objects ready for indexing.

        For memory-efficient streaming, use :meth:`chunk_segments_iter` instead.
        """
        chunks = list(self.chunk_segments_iter(segments, job_id=job_id, version=version))
        logger.info(
            "Chunker: %d segments → %d chunks (target=%d tokens, overlap=%d)",
            len(segments), len(chunks), self.chunk_tokens, self.chunk_overlap,
        )
        return chunks

    def chunk_segments_iter(
        self,
        segments: List[Dict[str, Any]],
        job_id: str = "",
        version: int = 1,
    ) -> Iterator[Chunk]:
        """
        Yield chunks lazily from aligned transcript segments.

        Identical logic to :meth:`chunk_segments` but returns a generator
        so downstream consumers can process chunks one at a time without
        holding all chunks in memory.

        Each segment dict should have at minimum:
            - text (str)
            - speaker (str)
            - start (float)
            - end (float)

        Yields:
            :class:`Chunk` objects ready for indexing.
        """
        if not segments:
            return

        # Normalise segments into a uniform format
        normed = self._normalise_segments(segments)
        if not normed:
            return

        chunk_count = 0
        buf_text: List[str] = []
        buf_tokens = 0
        buf_start: float = normed[0]["start"]
        buf_end: float = normed[0]["end"]
        buf_speaker: str = normed[0]["speaker"]
        buf_indices: List[int] = []

        # Track overlap window (last N tokens worth of segments)
        overlap_segs: List[Dict[str, Any]] = []

        for i, seg in enumerate(normed):
            seg_tokens = _estimate_tokens(seg["text"])
            seg_speaker = seg["speaker"]

            # Decide whether to flush current buffer
            should_flush = False
            if buf_tokens + seg_tokens > self.chunk_tokens and buf_text:
                should_flush = True
            elif seg_speaker != buf_speaker and buf_tokens >= self.chunk_tokens // 2 and buf_text:
                # Prefer breaking at speaker change if we already have
                # a reasonable amount of content
                should_flush = True

            if should_flush:
                # Emit chunk
                chunk = Chunk(
                    chunk_id=chunk_count,
                    text=" ".join(buf_text),
                    speaker=buf_speaker if self._single_speaker(buf_text, buf_speaker, normed, buf_indices) else "multiple",
                    start=buf_start,
                    end=buf_end,
                    job_id=job_id,
                    version=version,
                    segment_indices=list(buf_indices),
                )
                yield chunk
                chunk_count += 1

                # Compute overlap: take trailing segments whose total tokens
                # fall within chunk_overlap
                overlap_segs = []
                overlap_tokens = 0
                for j in range(len(buf_indices) - 1, -1, -1):
                    idx = buf_indices[j]
                    s = normed[idx]
                    st = _estimate_tokens(s["text"])
                    if overlap_tokens + st > self.chunk_overlap and overlap_segs:
                        break
                    overlap_segs.insert(0, s)
                    overlap_tokens += st

                # Start new buffer with overlap content
                buf_text = [s["text"] for s in overlap_segs]
                buf_tokens = overlap_tokens
                buf_start = overlap_segs[0]["start"] if overlap_segs else seg["start"]
                buf_end = overlap_segs[-1]["end"] if overlap_segs else seg["start"]
                buf_speaker = seg_speaker
                buf_indices = [s["_idx"] for s in overlap_segs]

            # Add current segment to buffer
            prefix = f"{seg_speaker}: " if seg_speaker and seg_speaker.lower() != "unknown" else ""
            buf_text.append(f"{prefix}{seg['text']}")
            buf_tokens += seg_tokens
            buf_end = seg["end"]
            if not buf_indices:
                buf_start = seg["start"]
            buf_indices.append(i)

        # Final chunk
        if buf_text:
            chunk = Chunk(
                chunk_id=chunk_count,
                text=" ".join(buf_text),
                speaker=buf_speaker if self._single_speaker(buf_text, buf_speaker, normed, buf_indices) else "multiple",
                start=buf_start,
                end=buf_end,
                job_id=job_id,
                version=version,
                segment_indices=list(buf_indices),
            )
            yield chunk

    def _normalise_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalise segment dicts to a uniform format with _idx tracking."""
        normed = []
        for i, seg in enumerate(segments):
            text = (
                seg.get("text")
                or seg.get("transcript")
                or seg.get("chunk_text")
                or ""
            ).strip()
            if not text:
                continue

            speaker = (
                seg.get("speaker")
                or seg.get("spk")
                or seg.get("speaker_label")
                or "unknown"
            )

            start = self._get_float(seg, "start", 0.0)
            end = self._get_float(seg, "end", start)

            normed.append({
                "text": text,
                "speaker": speaker,
                "start": start,
                "end": end,
                "_idx": i,
            })
        return normed

    @staticmethod
    def _get_float(d: dict, key: str, default: float) -> float:
        """Safely extract a float from a dict."""
        val = d.get(key)
        if val is None:
            # Try timestamp array
            ts = d.get("timestamp") or d.get("timestamps")
            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                try:
                    if key == "start":
                        return float(ts[0]) if ts[0] is not None else default
                    else:
                        return float(ts[1]) if ts[1] is not None else default
                except (ValueError, TypeError):
                    pass
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _single_speaker(
        buf_text: List[str],
        primary: str,
        normed: List[Dict[str, Any]],
        indices: List[int],
    ) -> bool:
        """Check if all segments in the buffer have the same speaker."""
        if not indices:
            return True
        speakers = {normed[i]["speaker"] for i in indices if i < len(normed)}
        return len(speakers) <= 1
