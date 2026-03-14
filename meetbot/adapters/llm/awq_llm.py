"""
Local LLM adapter for AWQ-quantized models (safetensors format).

Uses HuggingFace transformers with the AutoAWQ backend for 4-bit quantized
inference.  Supports Qwen2.5-Instruct and any other model whose config.json
carries ``quantization_config.quant_method == "awq"``.

Model loading is done once per process (singleton) and reused across queries.
Prompt chat-template formatting is delegated to the tokenizer so the adapter
works with any chat model, not just Qwen.

Requirements:
    pip install autoawq accelerate transformers
"""

import logging
import threading
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# Defaults — callers can override via constructor or generate() kwargs
# ---------------------------------------------------------------------------
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


class AwqLLMAdapter(BaseLLM):
    """
    LLM adapter for AWQ-quantized models loaded via HuggingFace transformers.

    The model directory must contain config.json (with quantization_config),
    safetensors weight shards, and tokenizer files — exactly what
    ``huggingface-cli download`` produces.

    Attributes
    ----------
    model_path:
        Path to the model directory (e.g. ``./models/qwen2.5-7B``).
    max_new_tokens:
        Default cap on generated tokens (overridable per-call).
    temperature:
        Default sampling temperature (overridable per-call).
    top_p:
        Default nucleus sampling parameter (overridable per-call).
    """

    # -----------------------------------------------------------------------
    # Process-level singleton state — shared across all AwqLLMAdapter instances
    # -----------------------------------------------------------------------
    _model = None
    _tokenizer = None
    _loaded_path: Optional[str] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _ensure_loaded(self):
        """Load model + tokenizer on first call; return (model, tokenizer)."""
        cls = type(self)

        # Fast path — already loaded for this model path
        if cls._model is not None and cls._loaded_path == self.model_path:
            return cls._model, cls._tokenizer

        with cls._lock:
            # Re-check inside the lock
            if cls._model is not None and cls._loaded_path == self.model_path:
                return cls._model, cls._tokenizer

            logger.info("Loading AWQ model from %s ...", self.model_path)

            # Compatibility shim: autoawq 0.2.9 imports PytorchGELUTanh from
            # transformers.activations, which was removed in transformers 4.52+.
            # Patch it with the modern GELUTanh equivalent before importing awq.
            try:
                import transformers.activations as _acts
                if not hasattr(_acts, "PytorchGELUTanh"):
                    _acts.PytorchGELUTanh = _acts.GELUTanh
            except Exception:
                pass

            # Verify autoawq is installed — it registers the GEMM kernel
            # that transformers delegates to under the hood.
            try:
                import awq  # noqa: F401
            except ImportError as _exc:
                raise RuntimeError(
                    "autoawq is required for AWQ model inference.\n"
                    "Install with:  pip install autoawq accelerate\n"
                    f"Underlying error: {_exc}"
                ) from _exc

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise RuntimeError(
                    "transformers is required.\n"
                    "Install with:  pip install transformers accelerate"
                )

            # AWQ hard-blocks CPU/disk offloading — the entire model must fit
            # in GPU VRAM.  Check before attempting to load to give a clear,
            # actionable error rather than a cryptic transformers ValueError.
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError(
                    "AWQ inference requires a CUDA-capable GPU (none detected).\n"
                    "Set USE_LOCAL_LLM=false in your .env to use the "
                    "HuggingFace Inference API instead."
                )

            free_vram, total_vram = torch.cuda.mem_get_info(0)
            model_dir = Path(self.model_path)
            disk_bytes = sum(
                f.stat().st_size
                for f in model_dir.rglob("*.safetensors")
                if f.is_file()
            )
            if disk_bytes > 0 and free_vram < disk_bytes:
                raise RuntimeError(
                    f"Insufficient GPU VRAM to load AWQ model.\n"
                    f"  Model on disk : {disk_bytes / 1e9:.2f} GB\n"
                    f"  GPU free VRAM : {free_vram / 1e9:.2f} GB  "
                    f"(total {total_vram / 1e9:.2f} GB)\n"
                    "AWQ models cannot be split across CPU+GPU.\n"
                    "Options:\n"
                    "  • Set USE_LOCAL_LLM=false in .env to use the "
                    "HuggingFace Inference API\n"
                    f"  • Use a smaller AWQ model that fits within "
                    f"{free_vram / 1e9:.1f} GB VRAM"
                )

            logger.info(
                "GPU VRAM check passed: %.2f GB free, model %.2f GB on disk",
                free_vram / 1e9,
                disk_bytes / 1e9,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cuda:0",
                dtype=torch.float16,
                trust_remote_code=False,
            )
            model.eval()

            cls._model = model
            cls._tokenizer = tokenizer
            cls._loaded_path = self.model_path
            logger.info("AWQ model ready (device_map=cuda:0)")
            return model, tokenizer

    def _tokenize(
        self,
        prompt: str,
        messages: Optional[List[dict]],
        tokenizer,
    ):
        """
        Build tokenized input tensors.

        When *messages* is provided the tokenizer's chat template is applied,
        which produces the correct ChatML format for Qwen2.5-Instruct
        (and other instruct-tuned models with a tokenizer_config.json that
        declares a chat_template).  Falls back to encoding the raw *prompt*
        string when no messages are supplied.
        """
        if messages:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        return tokenizer(text, return_tensors="pt")

    # -----------------------------------------------------------------------
    # BaseLLM interface
    # -----------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        messages: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a complete response (blocking).

        Parameters
        ----------
        prompt:
            Raw prompt string (used as-is when *messages* is None).
        messages:
            Optional list of ``{"role": ..., "content": ...}`` dicts.
            When provided the tokenizer's chat template is applied instead
            of encoding the raw prompt directly.  This is the recommended
            path for chat/instruct models.
        """
        model, tokenizer = self._ensure_loaded()
        inputs = self._tokenize(prompt, messages, tokenizer)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens or self.max_new_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        try:
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)
        except Exception as exc:
            raise RuntimeError(f"AWQ inference failed: {exc}") from exc
        finally:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Slice off the input tokens; decode only what the model generated
        new_ids = output_ids[0][input_ids.shape[-1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        messages: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream response tokens one-by-one via HuggingFace TextIteratorStreamer.

        Generation runs in a daemon background thread; the main thread iterates
        the streamer queue.  This method is a synchronous generator — call it
        from a background thread (e.g. via ``asyncio.to_thread``) to keep the
        event loop free.
        """
        model, tokenizer = self._ensure_loaded()
        inputs = self._tokenize(prompt, messages, tokenizer)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            # transformers too old — fall back to returning in one chunk
            yield self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=messages,
            )
            return

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens or self.max_new_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        gen_thread = threading.Thread(
            target=model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        try:
            gen_thread.start()
            for token_text in streamer:
                if token_text:
                    yield token_text
        except Exception as exc:
            raise RuntimeError(f"AWQ streaming failed: {exc}") from exc
        finally:
            gen_thread.join(timeout=10)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> None:
        """Unload the model and free GPU memory."""
        cls = type(self)
        with cls._lock:
            if cls._model is not None:
                del cls._model
                cls._model = None
                cls._tokenizer = None
                cls._loaded_path = None
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("AWQ model unloaded")


def cleanup_awq_llm() -> None:
    """
    Release the process-level AWQ model singleton and free GPU memory.

    Call this on application shutdown to ensure VRAM is returned to the OS
    before the process exits.
    """
    adapter = AwqLLMAdapter.__new__(AwqLLMAdapter)
    adapter.close()

