# syntax=docker/dockerfile:1.7
# Simplified PyTorch development environment with CUDA 12.9 support

ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM
ARG TARGETARCH

# ==============================================================================
# Base Stage - NVIDIA CUDA with Ubuntu 24.04
# ==============================================================================
FROM --platform=$TARGETPLATFORM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04 AS base

# Build arguments
ARG BUILDKIT_INLINE_CACHE=1
ARG APP_USER=ubuntu
ARG APP_UID=1000
ARG APP_GID=1000

# System dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    ca-certificates curl git wget unzip \
    # Build essentials
    build-essential pkg-config \
    # Python and development
    python3 python3-pip python3-venv python3-dev \
    # System utilities
    sudo tini vim tree \
    # GPU utilities
    nvidia-utils-555 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get autoremove -y && apt-get autoclean

# ==============================================================================
# User Setup Stage
# ==============================================================================
FROM base AS user-setup

# Create user with proper permissions (if it doesn't already exist)
RUN if ! getent group ${APP_USER} > /dev/null; then \
        groupadd -g ${APP_GID} ${APP_USER}; \
    fi && \
    if ! id -u ${APP_USER} > /dev/null 2>&1; then \
        useradd -m -u ${APP_UID} -g ${APP_USER} -s /bin/bash ${APP_USER}; \
    fi && \
    usermod -aG sudo ${APP_USER} && \
    echo "${APP_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${APP_USER} && \
    chmod 0440 /etc/sudoers.d/${APP_USER} && \
    mkdir -p /workspaces/torch-starter && \
    mkdir -p /home/${APP_USER}/.cache/{huggingface,torch,uv,pip} && \
    mkdir -p /home/${APP_USER}/.local/bin && \
    chown -R ${APP_USER}:${APP_USER} /home/${APP_USER} /workspaces

# ==============================================================================
# UV and Python Environment Stage
# ==============================================================================
FROM user-setup AS python-env

USER ${APP_USER}
WORKDIR /workspaces/torch-starter

# Install UV package manager
ENV UV_INSTALL_DIR=/home/${APP_USER}/.local/bin \
    UV_CACHE_DIR=/home/${APP_USER}/.cache/uv \
    PATH=/home/${APP_USER}/.local/bin:${PATH}

RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version

# Copy project files for dependency installation
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml ./
COPY --chown=${APP_USER}:${APP_USER} verify_setup/ ./verify_setup/

# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    uv lock && \
    uv venv --python 3.12 && \
    uv sync --frozen && \
    # Verify installations
    .venv/bin/python verify_setup/verify_python.py && \
    .venv/bin/python verify_setup/verify_torch.py

# ==============================================================================
# Final Stage
# ==============================================================================
FROM python-env AS final

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspaces/torch-starter \
    HF_HOME=/home/${APP_USER}/.cache/huggingface \
    TORCH_HOME=/home/${APP_USER}/.cache/torch \
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all

# Setup shell integration
RUN echo 'source /workspaces/torch-starter/.venv/bin/activate' >> ~/.bashrc && \
    echo 'cd /workspaces/torch-starter' >> ~/.bashrc && \
    echo 'echo "🚀 PyTorch DevContainer Ready! UV environment activated."' >> ~/.bashrc

EXPOSE 8888 8000 6006

# Health check
HEALTHCHECK --interval=300s --timeout=30s --start-period=60s --retries=2 \
    CMD .venv/bin/python verify_setup/health_check.py || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# Labels
LABEL org.opencontainers.image.title="torch-starter-simplified" \
      org.opencontainers.image.description="Simplified PyTorch development environment with CUDA 12.9" \
      org.opencontainers.image.version="1.0.0" \
      cuda.version="12.9" \
      pytorch.version="2.8.0"