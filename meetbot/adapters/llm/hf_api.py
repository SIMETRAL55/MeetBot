"""
HuggingFace Inference API adapter for LLM access.

Provides inference using HuggingFace's Inference API with support for
various open-source models and gated models with proper authentication.

Example:
    >>> from adapters.llm import HFAPILLMAdapter
    >>> llm = HFAPILLMAdapter(token="hf_xxxx")
    >>> response = llm.generate("What is Python?")
"""

import logging
from typing import Any, List, Optional

from .base import BaseLLM
from ...config import settings

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


class HFAPILLMAdapter(BaseLLM):
    """
    LLM adapter using HuggingFace Inference API.

    Supports inference against any model available on HuggingFace hub.
    Gated models require HF_API_TOKEN to be set.

    Attributes:
        token: HuggingFace API token
        model: Model name on HuggingFace hub
        provider: Inference provider (fal-ai, replicate, etc.)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        provider: str = "fal-ai",
    ):
        """
        Initialize HFAPILLMAdapter.

        Args:
            token: HuggingFace API token (uses settings if not provided)
            model: Model name on HuggingFace hub
            provider: Inference provider

        Raises:
            RuntimeError: If huggingface_hub not installed
            ValueError: If token is required but not provided
        """
        if InferenceClient is None:
            raise RuntimeError(
                "huggingface_hub not installed. Install with:\n"
                "  pip install huggingface_hub"
            )

        self.token = token or settings.get_hf_token()
        self.model = model
        self.provider = provider

        if not self.token:
            logger.warning(
                "No HuggingFace token provided. Some gated models may not be accessible."
            )

        try:
            self.client = InferenceClient(
                token=self.token,
                provider=self.provider,
            )
            logger.info(
                "HFAPILLMAdapter initialized: model=%s, provider=%s",
                model,
                provider,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InferenceClient: {e}") from e

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
        Generate response using HuggingFace Inference API.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Extra parameters passed to API

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If API call fails
        """
        try:
            logger.info("Calling HF Inference API for model %s", self.model)

            # Build request parameters
            request_kwargs = {
                "max_new_tokens": max_tokens or 256,
                "details": True,
            }

            if temperature is not None:
                request_kwargs["temperature"] = temperature

            if top_p is not None:
                request_kwargs["top_p"] = top_p

            if stop:
                request_kwargs["stop_sequences"] = stop

            # Add any extra kwargs
            request_kwargs.update(kwargs)

            # Call inference
            response = self.client.text_generation(
                prompt,
                **request_kwargs,
            )

            # Extract text from response
            if isinstance(response, dict):
                text = response.get("generated_text", "").strip()
            elif hasattr(response, "generated_text"):
                text = response.generated_text.strip()
            else:
                text = str(response).strip()

            if not text:
                logger.warning("Empty response from HF API")
                return "I could not generate a response."

            logger.info("Generated %d characters", len(text))
            return text

        except Exception as e:
            logger.error("HF API inference failed: %s", e)
            raise RuntimeError(f"HF API inference failed: {e}") from e

    def close(self) -> None:
        """Close API connection (no-op for HTTP clients)."""
        logger.debug("HFAPILLMAdapter closed")
