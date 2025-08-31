# syntax=docker/dockerfile:1.7
# PyTorch development environment with NVIDIA CUDA 12.9 support

ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM
ARG TARGETARCH

# ==============================================================================
# Base Stage - NVIDIA CUDA with Ubuntu 24.04 from NVIDIA Container Registry
# ==============================================================================
FROM --platform=$TARGETPLATFORM nvcr.io/nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04 AS base

# Build arguments
ARG BUILDKIT_INLINE_CACHE=1
ARG APP_USER=ubuntu
ARG APP_UID=1000
ARG APP_GID=1000

# System dependencies with optimized caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    ca-certificates curl git wget unzip \
    # Build essentials for compiling packages
    build-essential pkg-config \
    # Python and development tools
    python3 python3-pip python3-venv python3-dev \
    # System utilities
    sudo tini vim nano tree htop \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get autoremove -y && apt-get autoclean

# ==============================================================================
# User and Environment Setup
# ==============================================================================

# Create user with proper permissions
RUN if ! getent group ${APP_USER} > /dev/null; then \
        groupadd -g ${APP_GID} ${APP_USER}; \
    fi && \
    if ! id -u ${APP_USER} > /dev/null 2>&1; then \
        useradd -m -u ${APP_UID} -g ${APP_USER} -s /bin/bash ${APP_USER}; \
    fi && \
    usermod -aG sudo ${APP_USER} && \
    echo "${APP_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${APP_USER} && \
    chmod 0440 /etc/sudoers.d/${APP_USER}

# Create necessary directories with proper ownership
RUN mkdir -p /workspaces/torch-starter && \
    mkdir -p /home/${APP_USER}/.cache/{huggingface,torch,uv,pip} && \
    mkdir -p /home/${APP_USER}/.local/bin && \
    chown -R ${APP_USER}:${APP_USER} /home/${APP_USER} /workspaces

# Switch to non-root user
USER ${APP_USER}
WORKDIR /workspaces/torch-starter

# ==============================================================================
# UV Package Manager Installation
# ==============================================================================

# Install UV package manager with optimized settings
ENV UV_INSTALL_DIR=/home/${APP_USER}/.local/bin \
    UV_CACHE_DIR=/home/${APP_USER}/.cache/uv \
    PATH=/home/${APP_USER}/.local/bin:${PATH}

RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version

# ==============================================================================
# Python Environment Setup
# ==============================================================================

# Copy project configuration files (only essential files for dependency resolution)
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml ./

# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    # Initialize UV and create virtual environment
    uv venv --python 3.12 && \
    # Generate lockfile
    uv lock --verbose && \
    # Install core dependencies first
    uv sync --frozen && \
    # Verify basic installation
    .venv/bin/python -c "import sys; print(f'Python {sys.version}')" && \
    # Test PyTorch installation (allow failure during build)
    .venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}')" || echo "PyTorch verification will be done post-build"

# ==============================================================================
# Final Configuration
# ==============================================================================

# Environment variables for optimal performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspaces/torch-starter \
    HF_HOME=/home/${APP_USER}/.cache/huggingface \
    TORCH_HOME=/home/${APP_USER}/.cache/torch \
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    # UV optimizations
    UV_SYSTEM_PYTHON=1 \
    UV_CACHE_DIR=/home/${APP_USER}/.cache/uv \
    # CUDA optimizations
    CUDA_MODULE_LOADING=LAZY

# Setup shell integration for seamless development
RUN echo '# Torch-starter development environment' >> ~/.bashrc && \
    echo 'if [ -f /workspaces/torch-starter/.venv/bin/activate ]; then' >> ~/.bashrc && \
    echo '  source /workspaces/torch-starter/.venv/bin/activate' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc && \
    echo 'cd /workspaces/torch-starter 2>/dev/null || true' >> ~/.bashrc && \
    echo 'echo "🚀 PyTorch DevContainer Ready!"' >> ~/.bashrc && \
    echo 'if [ -f "./dev.sh" ]; then echo "💡 Run ./dev.sh help for available commands"; fi' >> ~/.bashrc

# Expose commonly used ports
EXPOSE 8888 8000 6006 8080

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# Metadata labels
LABEL org.opencontainers.image.title="torch-starter-nvidia" \
      org.opencontainers.image.description="PyTorch development environment with NVIDIA CUDA 12.9 support" \
      org.opencontainers.image.version="1.1.0" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04" \
      cuda.version="12.9" \
      pytorch.version="2.8.0" \
      nvidia.container.registry="true"