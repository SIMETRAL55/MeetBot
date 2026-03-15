# syntax=docker/dockerfile:1

# =============================================================================
# MeetBot — Production Dockerfile  (multi-stage, CPU + GPU)
#
# Build (CPU-only — default):
#   docker build -t meetbot .
#
# Build (CUDA 11.8 GPU):
#   docker build --build-arg CUDA_VARIANT=cu118 -t meetbot-gpu .
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
#     meetbot-gpu
# =============================================================================

# CUDA_VARIANT must be declared *before* the first FROM so Docker makes it
# available in all subsequent FROM ... AS ... lines.
ARG CUDA_VARIANT=cpu
# Accepted values: cpu | cu118 | cu121

# =============================================================================
# Builder base images — one per supported CUDA_VARIANT.
# Only the stage matching CUDA_VARIANT is materialised; the others are skipped.
# =============================================================================

# ---------------------------------------------------------------------------
# base-builder-cpu  — plain Python slim, no CUDA tooling
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base-builder-cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ffmpeg \
        libsndfile1-dev \
        libffi-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# base-builder-cu118  — NVIDIA CUDA 11.8 devel (ships nvcc + cuBLAS headers)
#
# libcuda.so.1 fix:
#   The devel image ships a linker stub at /usr/local/cuda/lib64/stubs/libcuda.so
#   but the dynamic linker resolves -lcuda by soname (libcuda.so.1).  We create
#   the missing versioned symlink and register the stubs directory with ldconfig
#   so the linker finds it without any special -L flags at the system level.
#   This is the standard approach for GPU-less build hosts.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base-builder-cu118

ENV DEBIAN_FRONTEND=noninteractive \
    # nvcc must be on PATH for CMake's FindCUDAToolkit
    PATH="/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}" \
    # Tell CMake exactly where the toolkit lives — prevents the
    # "Could not find nvcc / CUDA Toolkit not found" error.
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"

# Install Python 3.12 from the deadsnakes PPA (NVIDIA Ubuntu base has none)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        cmake \
        git \
        curl \
        ffmpeg \
        libsndfile1-dev \
        libffi-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3.12 -m ensurepip --upgrade \
    # Create the versioned stub symlink expected by the dynamic linker
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
              /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

# ---------------------------------------------------------------------------
# base-builder-cu121  — NVIDIA CUDA 12.1 devel (ships nvcc + cuBLAS headers)
# Same libcuda.so.1 stub treatment as cu118 above.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base-builder-cu121

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        cmake \
        git \
        curl \
        ffmpeg \
        libsndfile1-dev \
        libffi-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3.12 -m ensurepip --upgrade \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
              /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

# =============================================================================
# Stage 1 — builder
# Inherits the correct base via ARG expansion: base-builder-${CUDA_VARIANT}
# =============================================================================
FROM base-builder-${CUDA_VARIANT} AS builder

ARG CUDA_VARIANT

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # All packages land in a venv at a fixed, well-known path.
    # The COPY --from=builder in the runtime stage is then identical for every
    # variant regardless of which base image was chosen.
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

RUN python -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# ── PyTorch — installed before other deps so pip never re-resolves it ───────
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

# ── Core Python dependencies ─────────────────────────────────────────────────
# Exclude:
#   • torch / torchvision / torchaudio — already installed above
#   • blank lines / comments
COPY requirements.txt .
RUN grep -v -E "^(torch|torchvision|torchaudio|#|[[:space:]]*$)" \
        requirements.txt > requirements_core.txt \
    && pip install -r requirements_core.txt

# Install the project package
COPY setup.py .
COPY meetbot/ meetbot/
RUN pip install -e .

# =============================================================================
# Runtime base images — slim counterparts of the builder bases.
# They carry CUDA *runtime* libraries needed by torch/autoawq at run time
# but omit the compiler + headers to keep image size down.
# =============================================================================
FROM python:3.12-slim AS base-runtime-cpu

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base-runtime-cu118

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base-runtime-cu121

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# =============================================================================
# Stage 2 — runtime (final image)
# =============================================================================
FROM base-runtime-${CUDA_VARIANT} AS runtime

ARG CUDA_VARIANT

LABEL org.opencontainers.image.title="MeetBot"
LABEL org.opencontainers.image.description="AI meeting transcription and Q&A assistant"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    # Force torchaudio to prefer the ffmpeg I/O backend over soundfile.
    # soundfile (libsndfile) cannot decode M4A/AAC/MP3; ffmpeg handles all
    # common audio formats.  The fallback in diarization.py also uses ffmpeg
    # subprocess if this backend is unavailable.
    TORCHAUDIO_BACKEND=ffmpeg

# Runtime system deps: ffmpeg (audio decoding), libgomp (OpenMP for torch),
# tini (PID 1 signal forwarding + zombie reaping)
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

# Copy the entire virtual environment from the builder — identical path
# (/opt/venv) for every variant, so this instruction never changes.
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY --chown=meetbot:meetbot meetbot/ ./meetbot/
COPY --chown=meetbot:meetbot setup.py ./

# Persistent volume mount points
#   /app/data/uploads    uploaded audio files
#   /app/models          embedding + LLM model weights
#   /app/db              Chroma vector store + SQLite DB
#   /app/results         JSON pipeline outputs
#   /app/temp            transient pipeline working files
RUN mkdir -p data/uploads models db results prepared temp .cache_hf \
    && chown -R meetbot:meetbot /app

# Default configuration — override with -e or --env-file
ENV OUTPUT_DIR=/app/results \
    VECTOR_DB_PATH=/app/db/chroma \
    DB_PATH=/app/db/meetbot.db \
    TEMP_DIR=/app/temp \
    LOCAL_LLM_MODEL_PATH=/app/models/qwen2.5-7B \
    EMBEDDING_MODEL=/app/models/all-MiniLM-L6-v2 \
    EMBEDDING_DEVICE=cpu \
    TRANSCRIPTION_BACKEND=huggingface \
    USE_LOCAL_LLM=false \
    HF_PROVIDER=auto \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8080/ > /dev/null || exit 1

# tini as PID 1: forwards signals cleanly and reaps zombie processes
ENTRYPOINT ["/usr/bin/tini", "--"]

USER meetbot

CMD ["python", "-m", "meetbot.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
