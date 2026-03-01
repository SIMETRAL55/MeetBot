import json
import logging
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .base import BaseLLM
from ...config import settings

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

# ─────────────────────────────────────────────────────────────────────────────
# Provider resolution cache  {(model_id, provider_setting) -> resolved_provider}
# ─────────────────────────────────────────────────────────────────────────────
_PROVIDER_CACHE: Dict[Tuple[str, str], str] = {}

#: Error substrings that indicate the *provider* doesn't support this model/task
_PROVIDER_MISMATCH_PHRASES = (
    "not supported by provider",
    "model is not supported",
    "task not supported",
    "no endpoint found",
)


def _is_provider_mismatch(err_str: str) -> bool:
    low = err_str.lower()
    return any(p in low for p in _PROVIDER_MISMATCH_PHRASES)


def resolve_provider(model: str, token: Optional[str], wanted: str) -> str:
    """
    Return a concrete provider name for *model*.

    If *wanted* is ``"auto"`` the HF model-card API is queried and the first
    live provider for the ``conversational`` task is returned.  Falls back to
    ``"sambanova"`` when the API call fails or returns no live providers.

    Results are cached for the process lifetime to avoid redundant HTTP calls.

    Args:
        model:  HuggingFace model ID, e.g. ``"deepseek-ai/DeepSeek-V3-0324"``
        token:  HF API token (improves rate-limits on the model-card API)
        wanted: Provider name or ``"auto"``

    Returns:
        Concrete provider name string.
    """
    if wanted != "auto":
        logger.debug("Provider pinned to '%s' by configuration", wanted)
        return wanted

    cache_key = (model, wanted)
    if cache_key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[cache_key]

    resolved = _query_first_live_provider(model, token)
    _PROVIDER_CACHE[cache_key] = resolved
    return resolved


def _query_first_live_provider(model: str, token: Optional[str]) -> str:
    """Query the HF API for the model's live inference providers."""
    fallback = "sambanova"
    url = (
        f"https://huggingface.co/api/models/{model}"
        "?expand[]=inferenceProviderMapping"
    )
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        mapping: dict = data.get("inferenceProviderMapping", {})
        # Prefer conversational task; live status only
        candidates = [
            prov
            for prov, info in mapping.items()
            if info.get("status") == "live" and info.get("task") == "conversational"
        ]
        if candidates:
            chosen = candidates[0]
            logger.info(
                "Auto-resolved provider for '%s': %s (candidates: %s)",
                model,
                chosen,
                candidates,
            )
            return chosen

        logger.warning(
            "No live conversational providers found for '%s'. "
            "Falling back to '%s'.",
            model,
            fallback,
        )
    except Exception as exc:
        logger.warning(
            "Could not query HF provider mapping for '%s' (%s). "
            "Using fallback provider '%s'.",
            model,
            exc,
            fallback,
        )

    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────

class HFAPILLMAdapter(BaseLLM):
    """
    LLM adapter using HuggingFace Inference API (chat_completion endpoint).

    Supports ``conversational`` task models via any HF-supported provider.
    Pass ``provider="auto"`` (default) to auto-select the correct provider
    for the given model.

    Attributes:
        token:    HuggingFace API token
        model:    HuggingFace model ID
        provider: Resolved (concrete) provider name
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = "deepseek-ai/DeepSeek-V3-0324",
        provider: str = "auto",
    ):
        """
        Initialise HFAPILLMAdapter.

        Args:
            token:    HF API token (reads settings if omitted)
            model:    HuggingFace model ID
            provider: Provider name or ``"auto"`` for automatic resolution

        Raises:
            RuntimeError: If huggingface_hub is not installed
        """
        if InferenceClient is None:
            raise RuntimeError(
                "huggingface_hub not installed. Install with:\n"
                "  pip install huggingface_hub"
            )

        self.token = token or settings.get_hf_token()
        self.model = model
        # Resolve the provider (may issue one HTTP call on first use)
        self.provider = resolve_provider(model, self.token, provider)

        if not self.token:
            logger.warning(
                "No HuggingFace token provided. "
                "Some gated models may not be accessible."
            )

        try:
            self.client = InferenceClient(
                token=self.token,
                provider=self.provider,
            )
            logger.info(
                "HFAPILLMAdapter initialised: model=%s, provider=%s",
                model,
                self.provider,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialise InferenceClient: {e}"
            ) from e

    # ── helpers ────────────────────────────────────────────────────────────

    def _build_chat_kwargs(
        self,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[List[str]],
        extra: dict,
    ) -> dict:
        kw: dict = {"max_tokens": max_tokens or 256, "model": self.model}
        if temperature is not None:
            kw["temperature"] = temperature
        if top_p is not None:
            kw["top_p"] = top_p
        if stop:
            kw["stop"] = stop
        kw.update(extra)
        return kw

    # ── public interface ───────────────────────────────────────────────────

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
        Generate a response (non-streaming) via HF chat_completion.

        Args:
            prompt:      Input text (sent as a user message)
            max_tokens:  Maximum output tokens
            temperature: Sampling temperature
            top_p:       Nucleus sampling parameter
            stop:        Stop sequences
            **kwargs:    Extra parameters forwarded to chat_completion

        Returns:
            Generated response text

        Raises:
            RuntimeError: On API or token errors
        """
        logger.info(
            "HF generate: model=%s provider=%s", self.model, self.provider
        )
        kw = self._build_chat_kwargs(max_tokens, temperature, top_p, stop, kwargs)
        try:
            resp = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **kw,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                logger.warning("Empty response from HF API")
                return "I could not generate a response."
            logger.info("HF generate: %d chars returned", len(text))
            return text
        except Exception as exc:
            logger.error("HF generate failed: %s", exc)
            raise RuntimeError(f"HF API inference failed: {exc}") from exc

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
        Stream response tokens via HF chat_completion.

        Automatically falls back to non-streaming and yields the full answer
        as a single chunk if the provider signals that streaming is not
        supported for this model.

        This is a *synchronous* generator — run it in a background thread via
        ``asyncio.run_in_executor`` to keep the event loop responsive.

        Args:
            prompt:      Input text
            max_tokens:  Maximum output tokens
            temperature: Sampling temperature
            top_p:       Nucleus sampling parameter
            stop:        Stop sequences
            **kwargs:    Extra parameters forwarded to chat_completion

        Yields:
            Individual token strings

        Raises:
            RuntimeError: On auth, rate-limit, or unrecoverable API errors
        """
        if not self.token:
            raise RuntimeError(
                "HF_API_TOKEN is not set. "
                "Export HF_API_TOKEN=hf_... before starting the server."
            )

        logger.info(
            "HF stream: model=%s provider=%s streaming=True",
            self.model,
            self.provider,
        )

        kw = self._build_chat_kwargs(max_tokens, temperature, top_p, stop, kwargs)
        kw["stream"] = True

        try:
            for chunk in self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **kw,
            ):
                delta = chunk.choices[0].delta
                tok = delta.content if delta.content is not None else ""
                if tok:
                    yield tok
            return  # normal completion

        except Exception as first_exc:
            err_str = str(first_exc)

            # ── rate-limit — surface immediately ──────────────────────────
            if "429" in err_str or "rate limit" in err_str.lower():
                raise RuntimeError(
                    f"HuggingFace rate limit exceeded (429): {err_str}"
                ) from first_exc

            # ── provider doesn't support streaming → non-stream fallback ──
            if _is_provider_mismatch(err_str):
                logger.warning(
                    "HF stream: provider '%s' does not support streaming for '%s'. "
                    "Falling back to non-streaming. (reason: %s)",
                    self.provider,
                    self.model,
                    err_str,
                )
                try:
                    kw_ns = dict(kw)
                    kw_ns.pop("stream", None)
                    resp = self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        **kw_ns,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    if text:
                        logger.info(
                            "HF stream: fallback succeeded, yielding %d chars as "
                            "single chunk (streaming=False)",
                            len(text),
                        )
                        yield text
                    return
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"HF API non-stream fallback also failed: {fallback_exc}"
                    ) from fallback_exc

            raise RuntimeError(f"HF API streaming failed: {err_str}") from first_exc

    def close(self) -> None:
        """Close API connection (no-op for HTTP clients)."""
        logger.debug("HFAPILLMAdapter closed")

