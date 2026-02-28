"""
Local LLM adapter for quantized GGUF models.

Provides inference using locally loaded GGUF models via llama_cpp or ctransformers.
Supports GPU acceleration with memory management optimized for consumer GPUs (4-8GB VRAM).

Example:
    >>> from adapters.llm import LocalLLMAdapter
    >>> llm = LocalLLMAdapter(model_path="./rakutenai-7b-instruct-gguf")
    >>> response = llm.generate("What is Python?", max_tokens=256)
"""

import logging
from pathlib import Path
from typing import Any, List, Optional
from functools import lru_cache

from .base import BaseLLM

logger = logging.getLogger(__name__)

# Try to import llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# Try to import torch for CUDA utilities
try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL_DIR = "./rakutenai-7b-instruct-gguf"
DEFAULT_MODEL_FILENAME = "RakutenAI-7B-q3_K_M.gguf"
FALLBACK_MODEL_PATTERNS = ["*.gguf", "model.gguf", "*q4*.gguf", "*4bit*.gguf"]

# GPU memory management
DEFAULT_N_GPU_LAYERS = 20  # Aggressive GPU usage for RTX 3050
DEFAULT_CONTEXT_SIZE = 2048  # Safe context window for Q4_K_M quantization
DEFAULT_MAX_TOKENS = 256  # Max output tokens per generation
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


# ============================================================================
# LocalLLMManager: Singleton for Model Lifecycle Management
# ============================================================================


class LocalLLMManager:
    """
    Singleton manager for locally loaded GGUF models.

    Handles:
    - Model path resolution (local priority, fallback to HuggingFace)
    - Singleton model loading (avoid reloading on each call)
    - GPU memory management (CUDA cache cleanup)
    - Error handling with graceful fallbacks

    Attributes:
        model_path: Path to GGUF model file
        n_gpu_layers: Number of layers to offload to GPU
        n_ctx: Context window size in tokens
    """

    _instance: Optional["LocalLLMManager"] = None
    _model: Optional[Llama] = None

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        n_ctx: int = DEFAULT_CONTEXT_SIZE,
    ):
        """
        Initialize LocalLLMManager.

        Args:
            model_path: Path to GGUF model file or directory
            n_gpu_layers: Number of model layers to keep on GPU (default: 20)
            n_ctx: Context window size in tokens (default: 2048)

        Raises:
            RuntimeError: If llama_cpp is not installed
        """
        if Llama is None:
            raise RuntimeError(
                "llama_cpp not installed. Install with:\n"
                "  pip install llama-cpp-python\n"
                "For CUDA support:\n"
                "  CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python"
            )

        self.model_path = self._verify_model_path(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        logger.info(
            "LocalLLMManager initialized: model=%s, gpu_layers=%d, ctx=%d",
            self.model_path,
            self.n_gpu_layers,
            self.n_ctx,
        )

    def _verify_model_path(self, model_path_or_dir: str) -> str:
        """
        Verify and resolve model path.

        Priority order:
        1. If model_path_or_dir is a file that exists, use it
        2. If it's a directory, search for GGUF file inside
        3. If not found, log warning and return original (may auto-download)

        Args:
            model_path_or_dir: Path to model file or directory

        Returns:
            str: Resolved path to GGUF model file

        Raises:
            FileNotFoundError: If model file cannot be found or created
        """
        path = Path(model_path_or_dir)

        # Case 1: Direct file path that exists
        if path.is_file() and path.suffix == ".gguf":
            logger.info("Using GGUF model file: %s", path)
            return str(path)

        # Case 2: Directory that exists - search for GGUF
        if path.is_dir():
            logger.info("Searching for GGUF in directory: %s", path)
            for gguf_file in path.glob("*.gguf"):
                logger.info("Found GGUF model: %s", gguf_file)
                return str(gguf_file)

            # Try fallback patterns
            for pattern in FALLBACK_MODEL_PATTERNS:
                gguf_files = list(path.glob(pattern))
                if gguf_files:
                    logger.info("Found model matching %s: %s", pattern, gguf_files[0])
                    return str(gguf_files[0])

            logger.error("No GGUF files found in %s", path)
            raise FileNotFoundError(
                f"No GGUF model files found in {path}. Expected files: {FALLBACK_MODEL_PATTERNS}"
            )

        # Case 3: Path doesn't exist yet
        logger.warning(
            "Model path does not exist: %s. Model will be downloaded from HuggingFace "
            "if llama_cpp supports it.",
            model_path_or_dir,
        )
        return model_path_or_dir

    def load_model(self) -> Llama:
        """
        Load GGUF model with GPU memory management.

        Implements retry strategy:
        - First attempt: Use configured n_gpu_layers
        - On CUDA OOM: Reduce layers and retry
        - Multiple retries with exponential reduction

        Returns:
            Llama: Loaded model instance

        Raises:
            RuntimeError: If model cannot be loaded after retries
        """
        if self._model is not None:
            logger.debug("Model already loaded, returning cached instance")
            return self._model

        logger.info("Loading GGUF model from %s...", self.model_path)

        # Attempt model loading with fallback on CUDA OOM
        gpu_layers = self.n_gpu_layers
        last_error = None

        for attempt in range(3):  # Max 3 attempts
            try:
                logger.info(
                    "Attempt %d: Loading model with n_gpu_layers=%d",
                    attempt + 1,
                    gpu_layers,
                )

                self._model = Llama(
                    model_path=str(self.model_path),
                    n_gpu_layers=gpu_layers,
                    n_ctx=self.n_ctx,
                    verbose=False,
                )

                logger.info(
                    "✓ Model loaded successfully with %d GPU layers", gpu_layers
                )
                return self._model

            except RuntimeError as e:
                last_error = e
                if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                    # CUDA OOM - reduce layers and retry
                    gpu_layers = max(0, gpu_layers - 5)
                    logger.warning(
                        "CUDA OOM detected. Retrying with n_gpu_layers=%d: %s",
                        gpu_layers,
                        str(e)[:100],
                    )
                else:
                    # Different error - don't retry
                    raise
            except Exception as e:
                logger.error("Failed to load model: %s", e)
                raise

        # All attempts failed
        raise RuntimeError(
            f"Failed to load GGUF model after {attempt + 1} attempts. "
            f"Last error: {last_error}"
        )

    def cleanup(self) -> None:
        """
        Clean up model and GPU memory.

        Releases model instance and clears CUDA cache to free up GPU memory
        for subsequent operations.
        """
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Model instance deleted")

        # Clean GPU memory if torch is available
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

    @classmethod
    def get_instance(cls, **kwargs) -> "LocalLLMManager":
        """
        Get or create singleton instance.

        Args:
            **kwargs: Arguments to pass to __init__ if creating new instance
                     (model_path, n_gpu_layers, n_ctx)

        Returns:
            LocalLLMManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance


# ============================================================================
# LocalLLMAdapter: BaseLLM Implementation
# ============================================================================


class LocalLLMAdapter(BaseLLM):
    """
    LLM adapter for local quantized GGUF models.

    Implements BaseLLM interface for inference with local quantized models.
    Supports GPU acceleration and memory-efficient inference.

    Attributes:
        model_path: Path to GGUF model file or directory
        n_gpu_layers: Number of layers to offload to GPU
        max_tokens: Maximum output tokens per generation
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Example:
        >>> llm = LocalLLMAdapter(model_path="./rakutenai-7b-instruct-gguf")
        >>> response = llm.generate("What is machine learning?")
        >>> llm.close()  # Clean up resources
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_DIR,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        n_ctx: int = DEFAULT_CONTEXT_SIZE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ):
        """
        Initialize LocalLLMAdapter.

        Args:
            model_path: Path to GGUF model file or directory
            n_gpu_layers: Number of layers to keep on GPU
            n_ctx: Context window size in tokens
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._manager: Optional[LocalLLMManager] = None

    def _ensure_manager(self) -> LocalLLMManager:
        """Lazy-load model manager."""
        if self._manager is None:
            self._manager = LocalLLMManager.get_instance(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
            )
        return self._manager

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate response from prompt using local GGUF model.

        Args:
            prompt: Input prompt text
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            stop: Stop sequences (ignored for GGUF)
            **kwargs: Extra parameters (ignored)

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If generation fails
        """
        manager = self._ensure_manager()
        model = manager.load_model()

        # Use defaults if not overridden
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        try:
            logger.info("Generating response (max_tokens=%d)...", max_tokens)

            # Call GGUF model with safe parameters
            response = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,  # Don't echo input
            )

            # Extract text from response
            # llama_cpp returns: {"choices": [{"text": "..."}]}
            if isinstance(response, dict) and "choices" in response:
                text = response["choices"][0].get("text", "").strip()
            else:
                text = str(response).strip()

            if not text:
                logger.warning("Empty response from model")
                return "I could not generate a response."

            logger.info("Generated %d characters", len(text))
            return text

        except Exception as e:
            logger.error("Model inference failed: %s", e)
            raise RuntimeError(f"Local LLM inference failed: {e}") from e

        finally:
            # Clean CUDA memory between calls
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> None:
        """Clean up model and resources."""
        if self._manager is not None:
            self._manager.cleanup()
            self._manager = None


# ============================================================================
# Helper Functions
# ============================================================================


def get_local_llm(
    model_path: Optional[str] = None,
    n_gpu_layers: Optional[int] = None,
) -> Optional[LocalLLMAdapter]:
    """
    Factory function to create local LLM with environment variable support.

    Checks environment variables:
    - USE_LOCAL_LLM: Enable/disable local model (default: False)
    - LOCAL_LLM_MODEL_PATH: Override model path
    - LOCAL_LLM_GPU_LAYERS: Override GPU layers

    Args:
        model_path: Override model path (env var takes precedence if set)
        n_gpu_layers: Override GPU layers (env var takes precedence if set)

    Returns:
        LocalLLMAdapter instance if enabled, None otherwise

    Example:
        >>> llm = get_local_llm()
        >>> if llm:
        ...     response = llm.generate("What is AI?")
    """
    import os
    from ...config import settings

    use_local = settings.USE_LOCAL_LLM
    if not use_local:
        logger.debug("Local LLM disabled (USE_LOCAL_LLM not set or false)")
        return None

    # Use configuration with env var overrides
    llm_model_path = settings.LOCAL_LLM_MODEL_PATH or model_path or DEFAULT_MODEL_DIR
    llm_gpu_layers = settings.LOCAL_LLM_GPU_LAYERS or n_gpu_layers or DEFAULT_N_GPU_LAYERS

    logger.info(
        "Creating local LLM: model=%s, gpu_layers=%d",
        llm_model_path,
        llm_gpu_layers,
    )

    try:
        return LocalLLMAdapter(
            model_path=llm_model_path,
            n_gpu_layers=llm_gpu_layers,
        )
    except Exception as e:
        logger.error("Failed to create local LLM: %s", e)
        return None


def cleanup_local_llm() -> None:
    """
    Cleanup function to release local LLM resources.

    Call this when shutting down to properly release GPU memory.
    """
    manager = LocalLLMManager._instance
    if manager is not None:
        manager.cleanup()
        LocalLLMManager._instance = None
        logger.info("Local LLM cleaned up")
