#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL_PATH=${LOCAL_LLM_MODEL_PATH:-./models/your-model.gguf}
export LOCAL_LLM_GPU_LAYERS=${LOCAL_LLM_GPU_LAYERS:-0}
python -m meetbot.cli serve
