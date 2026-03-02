#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export USE_LOCAL_LLM=false
export TRANSCRIPTION_BACKEND=huggingface
python -m meetbot.cli serve
