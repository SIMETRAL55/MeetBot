"""
OpenAI-compatible adapter for PageIndex LLM calls.

Supports multiple backends:
- ``openrouter``, ``openai``, ``custom``: Remote OpenAI-compatible APIs
- ``ollama`` / ``local``: Local Ollama server

Uses environment variable injection to route the ``openai`` SDK calls to
the selected backend.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING
from urllib.request import urlopen

if TYPE_CHECKING:
    from meetbot.config import Settings

logger = logging.getLogger(__name__)

_UNSET = object()

# How long to wait for Ollama to become reachable after auto-starting it
_OLLAMA_STARTUP_TIMEOUT = 30  # seconds
_OLLAMA_HOST = "127.0.0.1"
_OLLAMA_PORT = 11434

# Module-level state for managing a self-started Ollama process.
# _ollama_proc  — the Popen handle if WE started it; None otherwise.
# _ollama_refs  — number of active pageindex_env contexts currently using Ollama.
# _ollama_lock  — protects both variables across background threads.
_ollama_proc: "subprocess.Popen[bytes] | None" = None
_ollama_refs: int = 0
_ollama_lock = threading.Lock()


def _ollama_reachable() -> bool:
    """Return True if the Ollama HTTP port is open."""
    try:
        with socket.create_connection((_OLLAMA_HOST, _OLLAMA_PORT), timeout=1):
            return True
    except OSError:
        return False


def _ensure_ollama_running() -> None:
    """
    Start ``ollama serve`` in the background if it is not already running.
    Increments the reference count.  Waits up to ``_OLLAMA_STARTUP_TIMEOUT``
    seconds for the port to open.  Raises RuntimeError if Ollama cannot be
    reached after that window.
    """
    global _ollama_proc, _ollama_refs

    with _ollama_lock:
        _ollama_refs += 1

        if _ollama_reachable():
            # Already running (either by us or the user) — nothing to do.
            return

        logger.info("Ollama not running — starting 'ollama serve' automatically…")
        try:
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            _ollama_refs -= 1
            raise RuntimeError(
                "Cannot find 'ollama' on PATH. "
                "Install Ollama from https://ollama.com/download and make sure "
                "it is on your PATH, then retry."
            )

        _ollama_proc = proc

    # Wait for the port outside the lock so other threads aren't blocked.
    deadline = time.monotonic() + _OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _ollama_reachable():
            logger.info("Ollama is now reachable on port %d.", _OLLAMA_PORT)
            return
        time.sleep(0.5)

    with _ollama_lock:
        _ollama_refs -= 1
    raise RuntimeError(
        f"Started 'ollama serve' but it did not become reachable on "
        f"{_OLLAMA_HOST}:{_OLLAMA_PORT} within {_OLLAMA_STARTUP_TIMEOUT}s."
    )


def _release_ollama() -> None:
    """
    Decrement the reference count.  If it reaches 0 and WE started the
    Ollama process, terminate it so it does not consume memory while idle.
    """
    global _ollama_proc, _ollama_refs

    proc = None
    with _ollama_lock:
        _ollama_refs = max(0, _ollama_refs - 1)
        if _ollama_refs == 0 and _ollama_proc is not None:
            proc = _ollama_proc
            _ollama_proc = None

    if proc is not None:
        logger.info("No active PageIndex sessions — stopping Ollama to free memory.")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to stop Ollama process: %s", exc)


def _ensure_model_available(model: str, model_path: str = "") -> None:
    """
    Check if *model* is already registered in Ollama; create it from a local
    HuggingFace directory if not.

    When *model_path* is provided, writes a temporary Modelfile and runs
    ``ollama create`` via subprocess to import the safetensors weights — no
    internet download required.
    """
    import json
    import tempfile

    base = f"http://{_OLLAMA_HOST}:{_OLLAMA_PORT}"

    # 1. List local models — if already registered, nothing to do.
    try:
        with urlopen(f"{base}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        local_names = {m["name"] for m in data.get("models", [])}
        if model in local_names:
            return
        if ":" not in model and f"{model}:latest" in local_names:
            return
    except Exception as exc:
        logger.warning("Could not list Ollama models: %s — will attempt create.", exc)

    # 2. Create the model from a local safetensors directory.
    if not model_path:
        raise RuntimeError(
            f"Model '{model}' is not available in Ollama and "
            f"PAGEINDEX_LLM_MODEL_PATH is not set. Either pull the model "
            f"manually ('ollama pull {model}') or set PAGEINDEX_LLM_MODEL_PATH "
            f"to a local HuggingFace model directory."
        )

    abs_path = os.path.abspath(model_path)
    if not os.path.isdir(abs_path):
        raise RuntimeError(
            f"PAGEINDEX_LLM_MODEL_PATH '{abs_path}' is not a directory."
        )

    logger.info(
        "Creating Ollama model '%s' from local directory '%s' …", model, abs_path
    )

    # Write a Modelfile and use `ollama create` CLI — more reliable for
    # safetensors import than the REST API.
    modelfile_content = f"FROM {abs_path}\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".Modelfile", delete=False
        ) as f:
            f.write(modelfile_content)
            modelfile_path = f.name

        result = subprocess.run(
            ["ollama", "create", model, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Clean up temp file
        try:
            os.unlink(modelfile_path)
        except OSError:
            pass

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"'ollama create {model}' failed (exit {result.returncode}): "
                f"{stderr or result.stdout.strip()}"
            )

        logger.info("Model '%s' created successfully from local weights.", model)

    except FileNotFoundError:
        raise RuntimeError(
            "Cannot find 'ollama' on PATH.  Install Ollama from "
            "https://ollama.com/download and make sure it is on your PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"'ollama create {model}' timed out after 600s. "
            f"The model conversion may be too slow for this machine."
        )


def get_pageindex_client(settings: "Settings") -> "openai.OpenAI":
    """
    Build and return a configured ``openai.OpenAI`` client for PageIndex LLM calls.

    For the ``vertexai`` backend, fetches a short-lived access token via
    Application Default Credentials (``google.auth.default``). The token is
    refreshed on every call so it is always valid.

    For all other backends, uses the static ``PAGEINDEX_LLM_API_KEY``.
    """
    import openai

    base_url = settings.get_pageindex_base_url()

    if settings.PAGEINDEX_LLM_BACKEND == "vertexai":
        try:
            from google.auth import default as google_auth_default
            import google.auth.transport.requests as google_requests
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-auth' package is required for PAGEINDEX_LLM_BACKEND='vertexai'. "
                "Install it with: pip install google-auth"
            ) from exc

        credentials, _ = google_auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google_requests.Request())
        api_key = credentials.token
        logger.debug("vertexai: refreshed access token (len=%d)", len(api_key or ""))
    else:
        api_key = settings.PAGEINDEX_LLM_API_KEY or "not-needed"

    return openai.OpenAI(base_url=base_url, api_key=api_key)


@contextmanager
def pageindex_env(settings: "Settings"):
    """
    Context manager that prepares the environment for PageIndex LLM calls.

    For ollama/local backends: starts Ollama (if needed) and ensures the
    requested model is available.  Releases the reference on exit so Ollama
    is stopped when no PageIndex sessions are active.

    For remote backends (openrouter, openai, custom): no environment
    mutations are performed — callers pass ``base_url`` and ``api_key``
    directly to ``openai.OpenAI()``, which is thread-safe.

    Args:
        settings: MeetBot Settings instance with PAGEINDEX_* fields.
    """
    backend = settings.PAGEINDEX_LLM_BACKEND

    try:
        if backend in ("local", "ollama"):
            _ensure_ollama_running()
            _ensure_model_available(
                settings.PAGEINDEX_LLM_MODEL,
                model_path=settings.PAGEINDEX_LLM_MODEL_PATH,
            )

        logger.debug(
            "pageindex_env: backend=%s, model=%s",
            backend, settings.PAGEINDEX_LLM_MODEL,
        )
        yield

    finally:
        if backend in ("local", "ollama"):
            _release_ollama()
