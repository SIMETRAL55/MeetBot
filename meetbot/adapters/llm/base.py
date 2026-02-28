"""Base interface for LLM backends in MeetBot."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.

    Defines common interface for both local quantized models and HuggingFace API models.
    Implementations must provide a generate() method that returns plain text responses.

    Example usage:
        llm = LocalLLMAdapter()
        response = llm.generate("What is machine learning?", max_tokens=256)
    """

    @abstractmethod
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
        Generate response from prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum output tokens (implementation-dependent)
            temperature: Sampling temperature (0.0 = deterministic, 1.0+ = random)
            top_p: Nucleus sampling parameter (0.0-1.0)
            stop: Stop sequences (may not be supported by all backends)
            **kwargs: Backend-specific parameters

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources and release memory.

        Called on shutdown or backend switch. Implementations should release
        GPU memory, file handles, etc.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self.close()
        return False
