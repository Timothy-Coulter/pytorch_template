#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Environment Setup Script for torch-starter
# ==============================================================================

# Colors
COLOR_RESET="\033[0m"
COLOR_RED="\033[0;31m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_BLUE="\033[0;34m"

log_error()   { echo -e "${COLOR_RED}[ERROR] $*${COLOR_RESET}" >&2; }
log_success() { echo -e "${COLOR_GREEN}[ OK ] $*${COLOR_RESET}"; }
log_warning() { echo -e "${COLOR_YELLOW}[WARN] $*${COLOR_RESET}"; }
log_info()    { echo -e "${COLOR_BLUE}[INFO] $*${COLOR_RESET}"; }

echo "🔧 Setting up torch-starter environment..."

# Ensure we're in the right directory
cd /workspaces/torch-starter

# Check if pyproject.toml exists, if not clone the repository
if [ ! -f "pyproject.toml" ]; then
    log_info "Cloning repository..."
    git clone https://github.com/Timothy-Coulter/ubuntu_p3_12_torch_devcontainer.git /tmp/repo
    cp -r /tmp/repo/* /tmp/repo/.[!.]* /workspaces/torch-starter/ 2>/dev/null || true
    rm -rf /tmp/repo
    log_success "Repository cloned"
fi

# Fix ownership
sudo chown -R ubuntu:ubuntu /workspaces/torch-starter

# Sync dependencies if UV environment exists
if [ -d ".venv" ]; then
    log_info "Syncing UV environment..."
    uv sync --frozen || {
        log_warning "UV sync failed, trying without frozen lock..."
        uv sync || log_error "UV sync failed completely"
    }
    log_success "Dependencies synced"
else
    log_warning "Virtual environment not found, creating..."
    uv venv --python 3.12
    uv sync
    log_success "Virtual environment created and dependencies installed"
fi

# Make scripts executable
if [ -f "dev.sh" ]; then
    chmod +x dev.sh
    log_success "dev.sh made executable"
fi

# Run basic verification
if [ -f "dev.sh" ] && [ -d ".venv" ]; then
    log_info "Running basic verification..."
    ./dev.sh verify-setup || log_warning "Verification had some issues, but continuing..."
else
    log_info "Running basic Python test..."
    .venv/bin/python -c "import sys; print(f'Python {sys.version}')" || log_error "Python test failed"
    .venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}')" || log_warning "PyTorch import failed"
fi

log_success "Environment setup completed!"
echo "🎉 You can now run './dev.sh help' for available commands"