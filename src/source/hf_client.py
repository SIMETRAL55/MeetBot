# Backward-compatible shim. Prefer: meetbot.adapters.hf_client
from meetbot.adapters.hf_client import HFInferenceClient

__all__ = ["HFInferenceClient"]
