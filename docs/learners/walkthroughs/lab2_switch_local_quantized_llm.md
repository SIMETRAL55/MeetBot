# Lab 2: Switch to local quantized LLM (GGUF)

## Goal
Use local inference for query answers.

## Commands
```bash
cd /workspace/MeetBot
source .venv/bin/activate
mkdir -p models
# Place a GGUF model file in models/ (manual download)
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL_PATH=./models/your-model.gguf
export LOCAL_LLM_GPU_LAYERS=0
python -m meetbot.cli serve
```

Ask a query on completed job.

## Expected excerpts
- Log indicates local LLM adapter selected.
- First query may be slower due to model load.

## Inspect
- confirm env vars with `env | rg 'USE_LOCAL_LLM|LOCAL_LLM'`
