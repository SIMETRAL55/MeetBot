"""MeetBot: Audio transcription, diarization, and RAG-based query system.

This package provides a modular, production-ready pipeline for:
- Audio transcription (multi-backend: local Whisper or HuggingFace API)
- Speaker diarization (Pyannote)
- Speaker-attributed transcript generation
- Vector embedding and semantic search (ChromaDB)
- RAG-based question answering (local or HF API LLM)

Key modules:
    config: Centralized configuration and environment management
    cli: Command-line interface entry point
    app: High-level pipeline orchestration
    services: Core services (transcriber, diarizer, indexer, query_service)
    adapters: Backend adapters (LLM, transcription engines)
    utils: Helper utilities and shared functions
"""

__version__ = "1.0.0"
__author__ = "MeetBot Team"

from . import config  # noqa: F401

__all__ = ["config"]
