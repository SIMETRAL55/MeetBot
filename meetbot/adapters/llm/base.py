"""Base interface for LLM backends in MeetBot."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterator, Optional, List


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.

    Defines common interface for both local quantized models and HuggingFace API models.
    Implementations must provide a ``generate()`` method that returns plain text and
    a ``generate_stream()`` method that yields text tokens one-by-one.
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
        Generate response from prompt (blocking, returns full text).

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

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Generate response from prompt, yielding token strings one-by-one.

        Default implementation calls ``generate()`` and yields the response as a
        single chunk.  Subclasses should override with true token-level streaming
        when the underlying backend supports it.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Backend-specific parameters

        Yields:
            str: Individual token or sub-word pieces
        """
        yield self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

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
