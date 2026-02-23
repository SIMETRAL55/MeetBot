from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptSegment:
    start: Optional[float]
    end: Optional[float]
    text: str


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass
class AlignedSegment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class TranscriptionResult:
    raw: Any
    segments: List[Dict[str, Any]]
    from_cache: bool
    cache_path: Optional[str]


@dataclass
class DiarizationResult:
    raw: Any
    segments: List[Dict[str, Any]]
    from_cache: bool
    cache_path: Optional[str]
