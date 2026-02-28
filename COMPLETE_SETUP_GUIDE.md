# MeetBot - Complete Setup Guide

Complete step-by-step instructions for setting up MeetBot from scratch in a new environment.

## Quick Start (5 minutes)

### Prerequisites
- **Python**: 3.8+ (tested on 3.10, 3.11, 3.12)
- **System Dependencies**:
  - ffmpeg (required for audio processing)
  - CUDA 11.8+ (optional but recommended for GPU acceleration)
  - CMake (optional, only for local LLM)

### Installation Steps

```bash
# 1. Clone repository (if not already done)
git clone <repo-url>
cd MeetBot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate           # Linux/macOS
# or: venv\Scripts\activate        # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. (Optional) Install PyTorch with GPU support
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# pip install torch torchvision torchaudio

# 5. Install all dependencies
pip install -r requirements.txt

# 6. (Optional) Install local LLM support with GPU
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# 7. Install MeetBot in development mode
pip install -e .

# 8. Verify installation
python -c "from meetbot.config import settings; print('✓ MeetBot ready!')"
```

## Detailed Setup by Platform

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3 python3-dev python3-venv \
    ffmpeg \
    build-essential cmake

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# (Optional) For NVIDIA GPU support
# Install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
# Then install PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Continue with pip installs
pip install --upgrade pip
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
pip install -e .
```

### macOS

```bash
# Install using Homebrew
brew install ffmpeg python-tk@3.12

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Note: Local LLM (llama-cpp-python) will compile for CPU on macOS
# For Metal GPU acceleration: pip install llama-cpp-python -c cmake.args="-DLLAMA_METAL=on"

pip install -e .
```

### Windows

```powershell
# Install prerequisites:
# 1. Download ffmpeg from https://ffmpeg.org/download.html
#    - Download "full" build
#    - Extract to C:\ffmpeg
#    - Add C:\ffmpeg\bin to PATH environment variable

# 2. Install Visual C++ Build Tools (for compilation):
#    https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install llama-cpp-python
pip install -e .
```

## Configuration

### 1. Create .env file

```bash
cp .env.example .env  # if available, or create new
```

Edit `.env`:

```env
# HuggingFace API Token (required for API-based transcription)
# Get token from: https://huggingface.co/settings/tokens
HF_API_TOKEN=hf_your_token_here

# Transcription Backend
TRANSCRIPTION_BACKEND=local    # 'local' for GPU Whisper, 'huggingface' for API

# Local LLM Configuration (optional, for private inference)
USE_LOCAL_LLM=true
LOCAL_LLM_MODEL_PATH=./models/rakutenai-7b-instruct-gguf
LOCAL_LLM_GPU_LAYERS=20        # Adjust based on VRAM (more = faster but uses more memory)

# Vector Database
VECTOR_DB_PATH=./db/sample
EMBEDDING_MODEL_PATH=./models/sarashina-embedding-v1-1b

# Audio Processing
AUDIO_CHUNK_ENABLE=true
AUDIO_CHUNK_SIZE_BYTES=104857600  # 100 MB
```

### 2. Download/Prepare Models

Models auto-download on first use, but you can pre-download:

```bash
# Whisper model (small = 468 MB, medium = 1.5 GB, large = 2.9 GB)
python -c "import whisper; whisper.load_model('small')"

# Pyannote (speaker diarization)
python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='YOUR_HF_TOKEN')"

# Embeddings
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sarashina-embedding-v1-1b')"

# Local LLM (optional)
# Download GGUF model manually and place in ./models/rakutenai-7b-instruct-gguf/
```

## Verify Installation

```bash
# Test imports
python -c "
from meetbot.config import settings
from meetbot.services.transcriber import TranscriberService
from meetbot.services.diarizer import DiarizationService
from meetbot.adapters.diarization import get_diarization_adapter
from meetbot.adapters.llm import get_local_llm
print('✓ All modules imported successfully!')
"

# Test CLI
python -m meetbot.cli --help

# Test with sample audio
python -m meetbot.cli transcribe data/sample.mp3 --backend local --out results/test.json
```

## Troubleshooting

### ffmpeg not found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Add to PATH and restart Python
# https://ffmpeg.org/download.html
```

### CUDA not available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install correct PyTorch version for your CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of VRAM during diarization
```bash
# Reduce LOCAL_LLM_GPU_LAYERS in .env
LOCAL_LLM_GPU_LAYERS=5  # Start low, increase gradually

# Or use CPU only
DEVICE=cpu
```

### llama-cpp-python build fails
```bash
# Install CMake
sudo apt-get install cmake  # Ubuntu/Debian
brew install cmake          # macOS

# Try again
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

## Usage Examples

### Transcribe and Diarize

```bash
# Using local GPU (Whisper + Pyannote)
python -m meetbot.cli transcribe data/sample.mp3 \
  --backend local \
  --language en \
  --min-speakers 2 \
  --out results/transcription.json

# Using HuggingFace API
python -m meetbot.cli transcribe data/sample.mp3 \
  --backend huggingface \
  --out results/transcription.json
```

### Build Search Index

```bash
python -m meetbot.cli index results/transcription.json \
  --db-root db \
  --embedding-model ./models/sarashina-embedding-v1-1b
```

### Query with RAG

```bash
# Query using local Rakuten LLM
python -m meetbot.cli query db/transcription "patient symptoms?" \
  --use-local-llm

# Query using HuggingFace API
python -m meetbot.cli query db/transcription "patient symptoms?"
```

## System Requirements

### Minimum (CPU only)
- 8 GB RAM
- 10 GB disk space (for models)
- CPU: Modern multi-core processor

### Recommended (GPU)
- 16+ GB RAM
- 20 GB disk space
- GPU: NVIDIA RTX 3050+ with 4GB VRAM minimum
- CUDA 11.8 or 12.1

### Optimal (High-performance)
- 32+ GB RAM
- SSD with 50+ GB space
- GPU: NVIDIA RTX 4070+ or better
- CUDA 12.1

## Environment Variables Summary

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_API_TOKEN` | HuggingFace API authentication | None |
| `TRANSCRIPTION_BACKEND` | Whisper backend (local/huggingface) | huggingface |
| `USE_LOCAL_LLM` | Enable local quantized LLM | false |
| `LOCAL_LLM_MODEL_PATH` | Path to GGUF model | ./models/rakutenai-7b-instruct-gguf |
| `LOCAL_LLM_GPU_LAYERS` | GPU layers for LLM | 20 |
| `VECTOR_DB_PATH` | Path to vector database | ./db/sample |
| `EMBEDDING_MODEL_PATH` | Embeddings model path | ./models/sarashina-embedding-v1-1b |
| `DEVICE` | Compute device (cuda/cpu) | auto-detect |

## Support & Updates

- Issues: Check GitHub issues
- Documentation: See USAGE.md, RAG_QUICK_REFERENCE.md
- Updates: `pip install --upgrade -r requirements.txt`

---

Happy transcribing! 🎉
