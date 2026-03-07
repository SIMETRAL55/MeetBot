"""
Memory management utilities for the MeetBot pipeline.

Provides:
- ``cleanup_memory()``: aggressive GC + optional CUDA cache flush.
- ``log_memory_usage()``: psutil-based RSS/percent logging.
- ``check_memory_pressure()``: returns True when system RAM exceeds threshold.
- ``chunked_iterable()``: lazy batch iterator for streaming processing.
"""

import gc
import logging
from typing import Iterable, Iterator, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def cleanup_memory(label: str = "") -> None:
    """Run garbage collection and optionally flush CUDA cache.

    Args:
        label: descriptive tag included in the log message.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("cleanup_memory(%s): gc + CUDA cache cleared", label)
            return
    except ImportError:
        pass
    logger.debug("cleanup_memory(%s): gc.collect done", label)


def log_memory_usage(label: str = "") -> float:
    """Log current process RSS via *psutil* and return usage in MB.

    Returns 0.0 if *psutil* is not installed (non-fatal).
    """
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        pct = psutil.virtual_memory().percent
        logger.info(
            "Memory [%s]: RSS=%.1f MB, system=%.1f%%", label, rss_mb, pct,
        )
        return rss_mb
    except ImportError:
        logger.debug("psutil not installed — memory logging skipped")
        return 0.0
    except Exception as exc:
        logger.debug("Memory log failed: %s", exc)
        return 0.0


def check_memory_pressure(threshold_pct: float = 0.85) -> bool:
    """Return True if system RAM usage exceeds *threshold_pct* (0.0–1.0).

    Returns False if *psutil* is unavailable (fail-open).
    """
    try:
        import psutil
        used_pct = psutil.virtual_memory().percent / 100.0
        return used_pct >= threshold_pct
    except ImportError:
        return False
    except Exception:
        return False


def chunked_iterable(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """Yield successive *size*-length lists from *iterable*.

    Unlike ``itertools.batched`` (Python 3.12+), this works on any Python 3.9+
    and always returns plain lists suitable for embedding APIs.

    >>> list(chunked_iterable([1,2,3,4,5], 2))
    [[1, 2], [3, 4], [5]]
    """
    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
