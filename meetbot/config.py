"""
Centralized configuration management for MeetBot.

Loads configuration from environment variables and .env files using pydantic-settings.
Provides sensible defaults for all components and validates settings at startup.

Environment variables (in priority order):
    HF_API_TOKEN, HF_HUB_TOKEN, HUGGINGFACEHUB_API_TOKEN: HuggingFace API token
    TRANSCRIPTION_BACKEND: 'local' or 'huggingface' (default: 'huggingface')
    USE_LOCAL_LLM: 'true'/'false', enable local quantized LLM (default: 'false')
    LOCAL_LLM_MODEL_PATH: Path to AWQ model directory (default: './models/qwen2.5-7B')
    LOCAL_LLM_CONTEXT_SIZE: Context window in tokens (default: 2048)
    LOCAL_LLM_MAX_TOKENS: Max output tokens (default: 256)
    VECTOR_DB_PATH: Path to ChromaDB directory (default: './db/sample')
    EMBEDDING_MODEL_PATH: HuggingFace embedding model (default: 'all-MiniLM-L6-v2')
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
        "./models/all-MiniLM-L6-v2",
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

    WHISPER_MODEL_SIZE: str = Field(
        "small",
        env="WHISPER_MODEL_SIZE",
        description=(
            "Whisper model size for local/faster-whisper backends. "
            "Options: tiny, base, small, medium, large-v2, large-v3. "
            "For RTX 3050 4 GB: 'small' (recommended) or 'medium'. "
            "'large-v3' requires ~6 GB VRAM and will OOM on 4 GB cards."
        ),
    )

    # =========================================================================
    # Local LLM Configuration (AWQ quantized models via transformers + autoawq)
    # =========================================================================
    USE_LOCAL_LLM: bool = Field(
        False,
        env="USE_LOCAL_LLM",
        description="Use local quantized LLM instead of HuggingFace API",
    )

    LOCAL_LLM_MODEL_PATH: str = Field(
        "./models/qwen2.5-1.5B",
        env="LOCAL_LLM_MODEL_PATH",
        description="Path to local AWQ model directory (must contain config.json and safetensors shards)",
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
    # Database Configuration
    # =========================================================================
    DB_PATH: str = Field(
        "./db/meetbot.db",
        env="DB_PATH",
        description="Path to the SQLite database file (must be on a persistent volume in Docker)",
    )

    VECTOR_DB_PATH: str = Field(
        "./db/sample",
        env="VECTOR_DB_PATH",
        description="Path to ChromaDB persistent storage directory",
    )

    # =========================================================================
    # Vector Database & Retrieval Configuration
    # =========================================================================
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

    SOURCE_SIM_THRESHOLD: float = Field(
        0.30,
        env="SOURCE_SIM_THRESHOLD",
        description=(
            "Minimum cosine similarity (0–1) between the generated answer embedding "
            "and a candidate source chunk for that chunk to appear in the final "
            "sources list.  Lower = show more sources; higher = stricter."
        ),
    )

    SOURCE_MAX_RETURN: int = Field(
        5,
        env="SOURCE_MAX_RETURN",
        description=(
            "Maximum number of sources to display after answer-embedding filtering. "
            "Must be ≤ RAG_TOP_K."
        ),
    )

    # ── New RAG pipeline settings ────────────────────────────────────────
    RAG_RECALL_N: int = Field(
        50,
        env="RAG_RECALL_N",
        description=(
            "Stage-1 ANN recall: how many candidates to fetch from the vector "
            "store before reranking.  Higher = better recall, slightly slower."
        ),
    )

    RAG_MAX_CONTEXT_CHUNKS: int = Field(
        6,
        env="RAG_MAX_CONTEXT_CHUNKS",
        description=(
            "Maximum number of context chunks to feed to the LLM after "
            "reranking.  Controls prompt length vs. answer quality."
        ),
    )

    CHUNK_TOKENS: int = Field(
        300,
        env="CHUNK_TOKENS",
        description=(
            "Target chunk size in estimated tokens for transcript chunking. "
            "Smaller = more precise retrieval; larger = more context per chunk."
        ),
    )

    CHUNK_OVERLAP: int = Field(
        50,
        env="CHUNK_OVERLAP",
        description=(
            "Number of overlap tokens between consecutive chunks to preserve "
            "context continuity across chunk boundaries."
        ),
    )

    RAG_V2_ENABLED: bool = Field(
        True,
        env="RAG_V2_ENABLED",
        description=(
            "DEPRECATED — always True.  The legacy RAG pipeline has been "
            "removed.  This field is kept for one release to avoid breaking "
            "existing .env files that set it.  Will be removed in a future "
            "version."
        ),
    )

    # ── Memory-safe indexing settings ────────────────────────────────────
    EMBED_BATCH_SIZE: int = Field(
        # Default 32.  all-MiniLM-L6-v2 is a small 22 M-param model (~90 MB)
        # with 384-dim outputs — far less memory pressure than the previous
        # 1.1 B-param sarashina model.  Batch-32 is efficient on typical
        # hardware; increase via EMBED_BATCH_SIZE for even faster throughput.
        32,
        env="EMBED_BATCH_SIZE",
        description=(
            "Number of documents to embed in a single batch during indexing. "
            "Lower = less peak RAM; higher = faster throughput."
        ),
    )

    INDEX_BATCH_PERSIST_CHECKPOINT: int = Field(
        100,
        env="INDEX_BATCH_PERSIST_CHECKPOINT",
        description=(
            "Write a checkpoint file every N batches during indexing so "
            "progress can be inspected (and potentially resumed)."
        ),
    )

    MEMORY_WATCH_ENABLED: bool = Field(
        True,
        env="MEMORY_WATCH_ENABLED",
        description="Enable psutil-based memory monitoring during indexing.",
    )

    MEMORY_WATCH_THRESHOLD_PCT: float = Field(
        0.85,
        env="MEMORY_WATCH_THRESHOLD_PCT",
        description=(
            "Fraction of total system RAM (0.0–1.0). If usage exceeds this "
            "during embedding, batch size is halved and the batch retried."
        ),
    )

    RAG_MEMORY_SAFE_MODE: bool = Field(
        True,
        env="RAG_MEMORY_SAFE_MODE",
        description=(
            "Master toggle for memory-safe indexing (batched embedding, "
            "streamed inserts, aggressive gc). Set to False to revert to "
            "single-call indexing for debugging."
        ),
    )

    TEMP_DIR: str = Field(
        "./temp",
        env="TEMP_DIR",
        description="Directory for temporary files (JSONL chunks, checkpoints).",
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
        "cuda",
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
    # PageIndex Configuration (vectorless LLM-based retrieval — optional)
    # =========================================================================
    PAGEINDEX_ENABLED: bool = Field(
        False,
        env="PAGEINDEX_ENABLED",
        description="Master toggle for PageIndex integration. When False, all "
                    "PageIndex features are disabled (zero regression risk).",
    )

    PAGEINDEX_AUTO_INDEX: bool = Field(
        False,
        env="PAGEINDEX_AUTO_INDEX",
        description="Automatically build PageIndex during pipeline ingestion "
                    "(Stage 4b). When False, PageIndex must be triggered manually "
                    "via the 'Build PageIndex' button or API endpoint.",
    )

    PAGEINDEX_LLM_BACKEND: str = Field(
        "openrouter",
        env="PAGEINDEX_LLM_BACKEND",
        description="LLM backend for PageIndex calls: 'vertexai' (Vertex AI via gcloud ADC, "
                    "recommended), 'google' (Gemini via Google AI Studio API key), "
                    "'openrouter' (free models), 'ollama' or 'local' "
                    "(Ollama offline fallback), 'openai', or 'custom'.",
    )

    VERTEXAI_PROJECT: str = Field(
        "",
        env="VERTEXAI_PROJECT",
        description="Google Cloud project ID or number for Vertex AI. Required when "
                    "PAGEINDEX_LLM_BACKEND='vertexai'.",
    )

    VERTEXAI_LOCATION: str = Field(
        "us-central1",
        env="VERTEXAI_LOCATION",
        description="Google Cloud region for Vertex AI endpoints. "
                    "Default: 'us-central1'.",
    )

    PAGEINDEX_LLM_BASE_URL: str = Field(
        "",
        env="PAGEINDEX_LLM_BASE_URL",
        description="OpenAI-compatible API base URL for PageIndex LLM calls. "
                    "Leave empty to auto-resolve from PAGEINDEX_LLM_BACKEND.",
    )

    PAGEINDEX_LLM_MODEL: str = Field(
        "qwen/qwen3-next-80b-a3b-instruct:free",
        env="PAGEINDEX_LLM_MODEL",
        description="Model ID for PageIndex LLM calls. Default is the free "
                    "Qwen3 80B MoE on OpenRouter. Use 'qwen2.5:3b' for local fallback.",
    )

    PAGEINDEX_LLM_API_KEY: str = Field(
        "",
        env="PAGEINDEX_LLM_API_KEY",
        description="API key for the PageIndex LLM backend. Required for "
                    "OpenRouter/OpenAI. Use 'not-needed' for local servers.",
    )

    PAGEINDEX_LLM_MODEL_PATH: str = Field(
        "",
        env="PAGEINDEX_LLM_MODEL_PATH",
        description="Path to a local HuggingFace model directory (safetensors). "
                    "Used by the 'ollama' backend to import models without internet.",
    )

    PAGEINDEX_MAX_PAGE_TOKENS: int = Field(
        20000,
        env="PAGEINDEX_MAX_PAGE_TOKENS",
        description="Maximum tokens per tree node during PageIndex indexing.",
    )

    PAGEINDEX_OUTPUT_DIR: str = Field(
        "./db/pageindex",
        env="PAGEINDEX_OUTPUT_DIR",
        description="Directory for storing PageIndex JSON tree files.",
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
        "",
        env="WEB_SECRET_KEY",
        description="Secret key for JWT signing and session encryption. "
                    "A random key is generated if not set, but this means "
                    "tokens are invalidated on every restart. "
                    "Set via WEB_SECRET_KEY env var in production.",
    )

    WEB_STORAGE_SECRET: str = Field(
        "",
        env="WEB_STORAGE_SECRET",
        description="Secret key for NiceGUI user storage encryption. "
                    "A random key is generated if not set.",
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

    PIPELINE_WORKERS: int = Field(
        1,
        env="PIPELINE_WORKERS",
        description="Max concurrent pipeline jobs. Keep at 1 for GPU-constrained machines "
                    "to prevent VRAM contention between Whisper and Pyannote.",
    )

    RATE_LIMIT_UPLOAD: str = Field(
        "10/hour",
        env="RATE_LIMIT_UPLOAD",
        description="Upload rate limit per IP address (slowapi format, e.g. '10/hour').",
    )

    RATE_LIMIT_CHAT: str = Field(
        "60/minute",
        env="RATE_LIMIT_CHAT",
        description="Chat WebSocket connection rate limit per IP address.",
    )

    # =========================================================================
    # Firebase Authentication (optional — enables social login)
    # =========================================================================
    FIREBASE_PROJECT_ID: str = Field(
        "",
        env="FIREBASE_PROJECT_ID",
        description="Firebase project ID (e.g. 'my-meetbot-app'). Required for Firebase auth.",
    )

    FIREBASE_WEB_API_KEY: str = Field(
        "",
        env="FIREBASE_WEB_API_KEY",
        description="Firebase Web API Key from the Firebase console (Browser key).",
    )

    # =========================================================================
    # SMTP Email Configuration (for password reset emails)
    # =========================================================================
    SMTP_HOST: str = Field(
        "",
        env="SMTP_HOST",
        description="SMTP server hostname (e.g. 'smtp.gmail.com').",
    )

    SMTP_PORT: int = Field(
        587,
        env="SMTP_PORT",
        description="SMTP server port. 587 for STARTTLS (recommended), 465 for SSL.",
    )

    SMTP_USER: str = Field(
        "",
        env="SMTP_USER",
        description="SMTP login username (usually the sender email address).",
    )

    SMTP_PASSWORD: str = Field(
        "",
        env="SMTP_PASSWORD",
        description="SMTP login password or app-specific password.",
    )

    SMTP_FROM: str = Field(
        "noreply@meetbot.app",
        env="SMTP_FROM",
        description="From address used in outgoing emails.",
    )

    # =========================================================================
    # Google Analytics 4
    # =========================================================================
    GA4_MEASUREMENT_ID: str = Field(
        "",
        env="GA4_MEASUREMENT_ID",
        description="GA4 Measurement ID (e.g. 'G-XXXXXXXXXX'). Leave empty to disable analytics.",
    )

    # =========================================================================
    # Runtime DB Overrides
    # =========================================================================

    # Keys that must never be hot-patched at runtime (security / structural).
    # WEB_SECRET_KEY changing mid-session would invalidate all live tokens.
    _NEVER_OVERRIDE: ClassVar[set[str]] = {
        "WEB_SECRET_KEY",
        "WEB_STORAGE_SECRET",
        "DB_PATH",
    }

    def apply_db_overrides(self, session) -> int:
        """Load AppSetting rows from DB and hot-patch the live settings singleton.

        Called once at server startup after the DB is initialised. Any key in
        ``_NEVER_OVERRIDE`` is silently skipped.

        Args:
            session: SQLAlchemy Session (caller is responsible for closing it).

        Returns:
            Number of overrides successfully applied.
        """
        from .db.models import AppSetting as _AppSetting
        _log = logging.getLogger("meetbot.config")
        rows = session.query(_AppSetting).all()
        applied = 0
        for row in rows:
            key = row.key
            if key in self._NEVER_OVERRIDE:
                continue
            if not hasattr(self, key):
                _log.debug("apply_db_overrides: unknown key %s — skipped", key)
                continue
            current = getattr(self, key)
            try:
                coerced = _coerce(current, row.value)
                object.__setattr__(self, key, coerced)
                applied += 1
            except Exception as exc:
                _log.warning("apply_db_overrides: skip %s — %s", key, exc)
        return applied

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

    def model_post_init(self, __context) -> None:
        """Generate random secrets if not provided; warn about insecure defaults."""
        import logging
        import secrets
        _log = logging.getLogger("meetbot.config")

        _insecure_defaults = {
            "meetbot-change-this-secret-key-in-production",
            "meetbot-storage-secret-change-me",
        }

        if not self.WEB_SECRET_KEY or self.WEB_SECRET_KEY in _insecure_defaults:
            generated = secrets.token_urlsafe(32)
            object.__setattr__(self, "WEB_SECRET_KEY", generated)
            _log.warning(
                "WEB_SECRET_KEY not set — generated a random key. "
                "Tokens will be invalidated on restart. "
                "Set WEB_SECRET_KEY env var for persistence."
            )

        if not self.WEB_STORAGE_SECRET or self.WEB_STORAGE_SECRET in _insecure_defaults:
            generated = secrets.token_urlsafe(32)
            object.__setattr__(self, "WEB_STORAGE_SECRET", generated)
            _log.warning(
                "WEB_STORAGE_SECRET not set — generated a random key. "
                "Set WEB_STORAGE_SECRET env var for persistence."
            )

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

    def get_pageindex_base_url(self) -> str:
        """Resolve the LLM base URL based on the configured backend."""
        if self.PAGEINDEX_LLM_BASE_URL:
            return self.PAGEINDEX_LLM_BASE_URL
        if self.PAGEINDEX_LLM_BACKEND == "vertexai":
            loc = self.VERTEXAI_LOCATION or "us-central1"
            proj = self.VERTEXAI_PROJECT
            if not proj:
                raise ValueError(
                    "VERTEXAI_PROJECT must be set when PAGEINDEX_LLM_BACKEND='vertexai'."
                )
            return (
                f"https://{loc}-aiplatform.googleapis.com/v1/projects/{proj}"
                f"/locations/{loc}/endpoints/openapi"
            )
        _backend_urls = {
            "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
            "local": "http://localhost:11434/v1",
            "ollama": "http://localhost:11434/v1",
        }
        return _backend_urls.get(self.PAGEINDEX_LLM_BACKEND, "https://openrouter.ai/api/v1")

    def get_pageindex_output_dir(self) -> Path:
        """Get PageIndex output directory as pathlib.Path, creating if needed."""
        p = Path(self.PAGEINDEX_OUTPUT_DIR).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

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
# Type coercion helper for DB overrides
# ============================================================================


def _coerce(current, raw: str):
    """Coerce a DB string value to match the Python type of *current*.

    Handles bool, int, float, str, and Optional[str] (None when raw is
    'none', 'null', or empty string).
    """
    if isinstance(current, bool):
        return raw.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    # str / Optional[str] / None
    return raw if raw.strip().lower() not in ("none", "null", "") else None


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
