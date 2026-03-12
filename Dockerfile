# PersonaPlex + Qwen3-TTS Streaming Dual GPU Server
# Supports both Koyeb and RunPod Serverless deployments.
# Nginx reverse proxy routes traffic on port 8998:
#   /api/chat        → PersonaPlex (WebSocket, port 8999)
#   /v1/*            → Qwen3-TTS Streaming (REST+streaming, port 8880)
#   /health-tts      → Qwen3-TTS health
#   /*               → PersonaPlex (default)
#
# RunPod: handler.py starts Supervisor internally and proxies requests.

ARG BASE_IMAGE="nvidia/cuda"
ARG BASE_IMAGE_TAG="12.4.1-devel-ubuntu22.04"
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# NVIDIA runtime env
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Install system dependencies (combined for both services)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopus-dev \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    nginx \
    supervisor \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ─── PersonaPlex Setup ────────────────────────────────────────────────────────
WORKDIR /app/moshi/
COPY moshi/ /app/moshi/

# Create PersonaPlex virtual environment and install dependencies
RUN uv venv /app/moshi/.venv --python 3.12
RUN uv sync

# ─── faster-qwen3-tts Setup ──────────────────────────────────────────────────
WORKDIR /app/tts/

# Create TTS virtual environment
RUN python3 -m venv /app/tts/.venv

# Install PyTorch with CUDA 12.1
RUN /app/tts/.venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /app/tts/.venv/bin/pip install --no-cache-dir \
    "torch>=2.5.1" \
    "torchaudio>=2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121

# Install faster-qwen3-tts (pulls qwen-tts, transformers, etc.)
RUN /app/tts/.venv/bin/pip install --no-cache-dir \
    "faster-qwen3-tts>=0.2.4" \
    "aiohttp>=3.9.0" \
    "librosa" \
    "scipy"

# Install flash-attn for faster inference (requires devel image for nvcc)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN /app/tts/.venv/bin/pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn build failed, falling back to SDPA"

# Copy TTS server
COPY qwen3-tts/tts_streaming_server.py /app/tts/tts_streaming_server.py

# ─── Nginx + Supervisor Config ────────────────────────────────────────────────
COPY nginx.conf /app/nginx.conf
COPY supervisord.conf /app/supervisord.conf

# Create necessary temp directories for nginx
RUN mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi

# Create cache directories
RUN mkdir -p /root/.cache /tmp/numba_cache

# ─── RunPod Handler + Start Script ───────────────────────────────────────────
RUN pip install --no-cache-dir runpod
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ─── Runtime ──────────────────────────────────────────────────────────────────
WORKDIR /app
EXPOSE 8998

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8998/health || exit 1

ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
