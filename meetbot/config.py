"""
Centralized configuration management for MeetBot.

Loads configuration from environment variables and .env files using pydantic-settings.
Provides sensible defaults for all components and validates settings at startup.

Environment variables (in priority order):
    HF_API_TOKEN, HF_HUB_TOKEN, HUGGINGFACEHUB_API_TOKEN: HuggingFace API token
    TRANSCRIPTION_BACKEND: 'local' or 'huggingface' (default: 'huggingface')
    USE_LOCAL_LLM: 'true'/'false', enable local quantized LLM (default: 'false')
    LOCAL_LLM_MODEL_PATH: Path to GGUF model (default: './rakutenai-7b-instruct-gguf')
    LOCAL_LLM_GPU_LAYERS: Number of GPU layers (default: 20)
    LOCAL_LLM_CONTEXT_SIZE: Context window in tokens (default: 2048)
    LOCAL_LLM_MAX_TOKENS: Max output tokens (default: 256)
    VECTOR_DB_PATH: Path to ChromaDB directory (default: './db/sample')
    EMBEDDING_MODEL_PATH: HuggingFace embedding model (default: 'sarashina-embedding-v1-1b')
    DEVICE: 'cuda' or 'cpu' (default: auto-detect)

Example usage:
    from config import settings

    token = settings.get_hf_token()
    model_path = settings.LOCAL_LLM_MODEL_PATH
    use_local = settings.USE_LOCAL_LLM
"""

from pathlib import Path
from typing import Optional, ClassVar
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load .env file at module import time
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # =========================================================================
    # HuggingFace Configuration (transcription, diarization, embedding, LLM API)
    # =========================================================================
    HF_API_TOKEN: Optional[str] = Field(
        None,
        env="HF_API_TOKEN",
        description="HuggingFace API token for gated models and inference",
    )
    HF_HUB_TOKEN: Optional[str] = Field(
        None,
        env="HF_HUB_TOKEN",
        description="Alternative HF token (deprecated, use HF_API_TOKEN)",
    )
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = Field(
        None,
        env="HUGGINGFACEHUB_API_TOKEN",
        description="Alternative HF token for LangChain compatibility",
    )

    # =========================================================================
    # Model Configuration
    # =========================================================================
    WHISPER_MODEL: str = Field(
        "openai/whisper-large-v3",
        env="WHISPER_MODEL",
        description="HuggingFace Whisper model name for ASR",
    )

    DIARIZATION_MODEL: str = Field(
        "pyannote/speaker-diarization-3.1",
        env="DIARIZATION_MODEL",
        description="HuggingFace Pyannote model for speaker diarization",
    )

    DIARIZATION_MODEL_REVISION: ClassVar[str] = "main"

    EMBEDDING_MODEL: str = Field(
        "./models/sarashina-embedding-v1-1b",
        env="EMBEDDING_MODEL",
        description="Local path to embedding model directory or HuggingFace model ID",
    )

    HF_MODEL: str = Field(
        "deepseek-ai/DeepSeek-V3-0324",
        env="HF_MODEL",
        description="HuggingFace model ID used for inference via the Inference API",
    )

    HF_PROVIDER: str = Field(
        "auto",
        env="HF_PROVIDER",
        description=(
            "HuggingFace Inference provider (e.g. 'sambanova', 'novita', 'cerebras'). "
            "Set to 'auto' to select the first live provider for the chosen model automatically."
        ),
    )

    # =========================================================================
    # Transcription Backend Configuration
    # =========================================================================
    TRANSCRIPTION_BACKEND: str = Field(
        "huggingface",
        env="TRANSCRIPTION_BACKEND",
        description="Transcription backend: 'local' (GPU Whisper) or 'huggingface' (API)",
    )

    # =========================================================================
    # Local LLM Configuration (for quantized GGUF models)
    # =========================================================================
    USE_LOCAL_LLM: bool = Field(
        False,
        env="USE_LOCAL_LLM",
        description="Use local quantized LLM instead of HuggingFace API",
    )

    LOCAL_LLM_MODEL_PATH: str = Field(
        "./models/rakutenai-7b-instruct-gguf",
        env="LOCAL_LLM_MODEL_PATH",
        description="Path to local quantized GGUF model directory or file",
    )

    LOCAL_LLM_GPU_LAYERS: int = Field(
        15,
        env="LOCAL_LLM_GPU_LAYERS",
        description=(
            "Number of model layers to offload to GPU (higher = more VRAM, faster). "
            "Defaults to 0 (CPU-only) because at query time most VRAM is occupied by "
            "Pyannote/Whisper residuals. Set to 8 or higher only if you have "
            ">=1 GiB VRAM free after all other models finish."
        ),
    )

    LOCAL_LLM_CONTEXT_SIZE: int = Field(
        2048,
        env="LOCAL_LLM_CONTEXT_SIZE",
        description="Context window size in tokens (max prompt + response length)",
    )

    LOCAL_LLM_MAX_TOKENS: int = Field(
        128,
        env="LOCAL_LLM_MAX_TOKENS",
        description="Maximum output tokens per generation (128 is safe for meeting Q&A; increase for longer answers)",
    )

    LOCAL_LLM_TEMPERATURE: float = Field(
        0.7,
        env="LOCAL_LLM_TEMPERATURE",
        description="Sampling temperature (0.0 = deterministic, 1.0+ = random)",
    )

    # =========================================================================
    # Vector Database & Retrieval Configuration
    # =========================================================================
    VECTOR_DB_PATH: str = Field(
        "./db/sample",
        env="VECTOR_DB_PATH",
        description="Path to ChromaDB persistent storage directory",
    )

    VECTOR_DB_COLLECTION_NAME: str = Field(
        "meetbot",
        env="VECTOR_DB_COLLECTION_NAME",
        description="ChromaDB collection name",
    )

    RAG_TOP_K: int = Field(
        4,
        env="RAG_TOP_K",
        description="Number of documents to retrieve in RAG queries",
    )

    # =========================================================================
    # Device & Performance Configuration
    # =========================================================================
    DEVICE: Optional[str] = Field(
        None,
        env="DEVICE",
        description="Compute device: 'cuda', 'cpu', or None for auto-detection",
    )

    EMBEDDING_DEVICE: str = Field(
        "cpu",
        env="EMBEDDING_DEVICE",
        description="Device for embedding model ('cuda' or 'cpu')",
    )

    # =========================================================================
    # Cache & I/O Configuration
    # =========================================================================
    CACHE_DIR: str = Field(
        "./.cache_hf",
        env="CACHE_DIR",
        description="Directory for caching transcription and diarization results",
    )

    OUTPUT_DIR: str = Field(
        "./results",
        env="OUTPUT_DIR",
        description="Directory for output JSON files",
    )

    PREPARED_DOCS_DIR: str = Field(
        "./prepared",
        env="PREPARED_DOCS_DIR",
        description="Directory for prepared JSONL documents before indexing",
    )

    # =========================================================================
    # Audio Chunking Configuration (for large file handling)
    # =========================================================================
    AUDIO_CHUNK_ENABLE: bool = Field(
        True,
        env="AUDIO_CHUNK_ENABLE",
        description="Enable audio chunking for large files",
    )

    AUDIO_CHUNK_SIZE_BYTES: int = Field(
        100 * 1024 * 1024,
        env="AUDIO_CHUNK_SIZE_BYTES",
        description="WAV file size threshold (bytes) for triggering chunking (100 MB default)",
    )

    AUDIO_CHUNK_OVERLAP_SECONDS: float = Field(
        1.0,
        env="AUDIO_CHUNK_OVERLAP_SECONDS",
        description="Overlap duration between consecutive chunks (seconds)",
    )

    AUDIO_CHUNK_NOMINAL_DURATION: float = Field(
        120.0,
        env="AUDIO_CHUNK_NOMINAL_DURATION",
        description="Target chunk duration (seconds, ~2 minutes default)",
    )

    AUDIO_CHUNK_USE_SILENCE_DETECTION: bool = Field(
        True,
        env="AUDIO_CHUNK_USE_SILENCE_DETECTION",
        description="Snap chunk boundaries to silence points for cleaner cuts",
    )

    # =========================================================================
    # Transcription/Diarization Defaults
    # =========================================================================
    DEFAULT_LANGUAGE: Optional[str] = Field(
        None,
        env="DEFAULT_LANGUAGE",
        description="Default language hint for Whisper (e.g., 'en')",
    )

    # =========================================================================
    # Web Application Configuration
    # =========================================================================
    WEB_HOST: str = Field(
        "0.0.0.0",
        env="WEB_HOST",
        description="Host to bind the web server to",
    )

    WEB_PORT: int = Field(
        8080,
        env="WEB_PORT",
        description="Port to listen on for the web server",
    )

    WEB_SECRET_KEY: str = Field(
        "meetbot-change-this-secret-key-in-production",
        env="WEB_SECRET_KEY",
        description="Secret key for session encryption (CHANGE IN PRODUCTION)",
    )

    WEB_STORAGE_SECRET: str = Field(
        "meetbot-storage-secret-change-me",
        env="WEB_STORAGE_SECRET",
        description="Secret key for NiceGUI user storage encryption",
    )

    MAX_UPLOAD_SIZE_MB: int = Field(
        500,
        env="MAX_UPLOAD_SIZE_MB",
        description="Maximum upload file size in megabytes",
    )

    ALLOWED_AUDIO_EXTENSIONS: str = Field(
        ".wav,.mp3,.m4a,.flac,.aac,.ogg,.wma,.opus",
        env="ALLOWED_AUDIO_EXTENSIONS",
        description="Comma-separated list of allowed audio file extensions",
    )

    def get_allowed_extensions(self) -> list[str]:
        """Get allowed audio extensions as a list."""
        return [ext.strip().lower() for ext in self.ALLOWED_AUDIO_EXTENSIONS.split(",")]

    def get_max_upload_bytes(self) -> int:
        """Get maximum upload size in bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # =========================================================================
    # Pydantic Configuration
    # =========================================================================
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Allow unknown env vars without failing
    }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_hf_token(self) -> Optional[str]:
        """
        Get HuggingFace API token from first available environment variable.

        Priority order:
        1. HF_API_TOKEN (recommended)
        2. HF_HUB_TOKEN (deprecated)
        3. HUGGINGFACEHUB_API_TOKEN (LangChain legacy)

        Returns:
            str: HuggingFace token or None if not set
        """
        return (
            self.HF_API_TOKEN
            or self.HF_HUB_TOKEN
            or self.HUGGINGFACEHUB_API_TOKEN
        )

    def get_device(self) -> str:
        """
        Get compute device with fallback to CPU if CUDA unavailable.

        Auto-detects CUDA availability if device not configured.

        Returns:
            str: 'cuda' or 'cpu'
        """
        if self.DEVICE:
            return self.DEVICE

        try:
            import torch

            if torch.cuda.is_available():
                logger.debug("CUDA detected, using GPU")
                return "cuda"
        except (ImportError, Exception):
            pass

        logger.debug("CUDA not available, using CPU")
        return "cpu"

    def get_vector_db_path(self) -> Path:
        """Get vector database path as pathlib.Path."""
        return Path(self.VECTOR_DB_PATH).expanduser().resolve()

    def get_cache_dir(self) -> Path:
        """Get cache directory path as pathlib.Path."""
        return Path(self.CACHE_DIR).expanduser().resolve()

    def get_output_dir(self) -> Path:
        """Get output directory path as pathlib.Path."""
        return Path(self.OUTPUT_DIR).expanduser().resolve()

    def get_prepared_docs_dir(self) -> Path:
        """Get prepared documents directory path as pathlib.Path."""
        return Path(self.PREPARED_DOCS_DIR).expanduser().resolve()


# ============================================================================
# Singleton Settings Instance
# ============================================================================

settings = Settings()

logger.debug(
    "Settings loaded: backend=%s, use_local_llm=%s, device=%s",
    settings.TRANSCRIPTION_BACKEND,
    settings.USE_LOCAL_LLM,
    settings.get_device(),
)
