# syntax=docker/dockerfile:1

# =============================================================================
# MeetBot — Production Dockerfile  (multi-stage, CPU + GPU)
#
# Build (CPU-only — default):
#   docker build -t meetbot .
#
# Build (CUDA 12.1 GPU):
#   docker build --build-arg CUDA_VARIANT=cu121 -t meetbot-gpu .
#
# Run (CPU-only):
#   docker run -p 8080:8080 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/db:/app/db \
#     --env-file .env \
#     meetbot
#
# Run (GPU):
#   docker run --gpus all -p 8080:8080 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/db:/app/db \
#     --env-file .env \
#     -e EMBEDDING_DEVICE=cuda \
#     -e LOCAL_LLM_GPU_LAYERS=20 \
#     meetbot-gpu
# =============================================================================

ARG CUDA_VARIANT=cpu
# Accepted values: cpu | cu118 | cu121

# ---------------------------------------------------------------------------
# Stage 1 – builder
# Full build toolchain, compiles llama-cpp-python and installs all deps.
# The final runtime image copies only the installed packages.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ARG CUDA_VARIANT

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time system dependencies (compiler, cmake for llama-cpp-python, audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        libsndfile1-dev \
        libffi-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --upgrade pip setuptools wheel

# ── PyTorch (must be installed before packages that depend on it) ──────────
RUN if [ "$CUDA_VARIANT" = "cu118" ]; then \
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu118; \
    elif [ "$CUDA_VARIANT" = "cu121" ]; then \
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ── Core Python dependencies (everything except llama-cpp-python) ──────────
COPY requirements.txt .
RUN grep -v -E "^(llama-cpp-python|#|[[:space:]]*$)" requirements.txt \
        > requirements_core.txt \
    && pip install -r requirements_core.txt

# ── llama-cpp-python with optional CUDA backend ────────────────────────────
RUN if [ "$CUDA_VARIANT" = "cu118" ] || [ "$CUDA_VARIANT" = "cu121" ]; then \
        CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
        pip install llama-cpp-python; \
    else \
        pip install llama-cpp-python; \
    fi

# Install the project package
COPY setup.py .
COPY meetbot/ meetbot/
RUN pip install -e .

# ---------------------------------------------------------------------------
# Stage 2 – runtime
# Slim image without build toolchain.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ARG CUDA_VARIANT

LABEL org.opencontainers.image.title="MeetBot"
LABEL org.opencontainers.image.description="AI meeting transcription and Q&A assistant"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime system deps: ffmpeg (audio processing), tini (proper PID 1)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1000 meetbot \
    && useradd --uid 1000 --gid meetbot --shell /bin/bash --create-home meetbot

WORKDIR /app

# Copy Python packages installed in the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --chown=meetbot:meetbot meetbot/ ./meetbot/
COPY --chown=meetbot:meetbot setup.py ./

# Persistent volume mount points — override with -v in docker run:
#   /app/data/uploads    uploaded audio files
#   /app/models          embedding + LLM model weights (large, keep as volume)
#   /app/db              Chroma vector store + SQLite DB
#   /app/results         JSON pipeline outputs
RUN mkdir -p data/uploads models db results prepared \
    && chown -R meetbot:meetbot /app

# Default configuration — override with -e or --env-file
ENV OUTPUT_DIR=/app/results \
    VECTOR_DB_PATH=/app/db/chroma \
    LOCAL_LLM_MODEL_PATH=/app/models/rakutenai-7b-instruct-gguf \
    EMBEDDING_MODEL=/app/models/sarashina-embedding-v1-1b \
    EMBEDDING_DEVICE=cpu \
    TRANSCRIPTION_BACKEND=huggingface \
    USE_LOCAL_LLM=false \
    HF_PROVIDER=auto \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8080

EXPOSE 8080

# tini as PID 1: forwards signals cleanly and reaps zombie processes
ENTRYPOINT ["/usr/bin/tini", "--"]

USER meetbot

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8080/ > /dev/null || exit 1

CMD ["python", "-m", "meetbot.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
