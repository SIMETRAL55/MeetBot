"""
OpenAI-compatible adapter for PageIndex LLM calls.

Uses environment variable injection (Approach A from the integration plan)
to route PageIndex's ``openai`` SDK calls to the configured backend without
modifying any PageIndex vendor code.

The adapter:
1. Saves the current ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` env vars
2. Sets them to the PageIndex-specific backend values
3. Yields control so the caller can invoke PageIndex functions
4. Restores the original env vars (even on exception)

Usage::

    from meetbot.adapters.llm.openai_adapter import pageindex_env

    with pageindex_env(settings):
        # All openai SDK calls inside this block go to the PageIndex backend
        result = await md_to_tree(md_path, model=settings.PAGEINDEX_LLM_MODEL)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meetbot.config import Settings

logger = logging.getLogger(__name__)

_UNSET = object()


@contextmanager
def pageindex_env(settings: "Settings"):
    """
    Context manager that injects OpenAI env vars for PageIndex calls.

    Sets ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``, and (for local backends)
    ``OLLAMA_KEEP_ALIVE`` before yielding, then restores originals.

    Args:
        settings: MeetBot Settings instance with PAGEINDEX_* fields.
    """
    base_url = settings.get_pageindex_base_url()
    api_key = settings.PAGEINDEX_LLM_API_KEY or "not-needed"

    # Save originals
    saved = {}
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OLLAMA_KEEP_ALIVE"):
        saved[key] = os.environ.get(key, _UNSET)

    try:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = api_key

        # For local Ollama backend, set KEEP_ALIVE=0 so model unloads after use
        if settings.PAGEINDEX_LLM_BACKEND == "local":
            os.environ["OLLAMA_KEEP_ALIVE"] = "0"

        logger.debug(
            "pageindex_env: injected OPENAI_BASE_URL=%s, model=%s, backend=%s",
            base_url, settings.PAGEINDEX_LLM_MODEL, settings.PAGEINDEX_LLM_BACKEND,
        )
        yield

    finally:
        # Restore originals
        for key, val in saved.items():
            if val is _UNSET:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        logger.debug("pageindex_env: restored original env vars")
