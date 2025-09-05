#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Environment Setup Script for torch-starter (Fixed)
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

# Check if we're in a valid torch-starter directory
if [ ! -f "pyproject.toml" ]; then
    log_error "pyproject.toml not found - are we in the right directory?"
    exit 1
fi

# Fix ownership (important for mounted volumes)
if [ -w "/workspaces/torch-starter" ]; then
    log_info "Fixing file ownership..."
    sudo chown -R ubuntu:ubuntu /workspaces/torch-starter 2>/dev/null || log_warning "Could not fix ownership"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    log_warning "Virtual environment not found, creating..."
    uv venv --python 3.12
    log_success "Virtual environment created"
fi

# Sync dependencies
log_info "Syncing dependencies with UV..."
if uv sync; then
    log_success "Dependencies synced successfully"
else
    log_warning "UV sync failed, trying without lockfile..."
    if uv sync --no-lock; then
        log_success "Dependencies synced (no lockfile)"
    else
        log_error "UV sync failed completely, falling back to pip"
        .venv/bin/pip install torch torchvision torchaudio numpy pandas transformers jupyter
    fi
fi

# Make scripts executable
if [ -f "dev.sh" ]; then
    chmod +x dev.sh
    log_success "dev.sh made executable"
fi

# Create necessary directories
mkdir -p {data,logs,models,checkpoints,outputs,notebooks,examples}
log_success "Project directories created"

# Run basic verification
log_info "Running basic verification..."
if .venv/bin/python -c "import sys; print(f'✅ Python {sys.version}')"; then
    log_success "Python verification passed"
else
    log_error "Python verification failed"
    exit 1
fi

if .venv/bin/python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"; then
    log_success "PyTorch verification passed"
    
    # Check CUDA if available
    if .venv/bin/python -c "import torch; print('✅ CUDA available' if torch.cuda.is_available() else '⚠️ CUDA not available')"; then
        log_info "CUDA status checked"
    fi
else
    log_warning "PyTorch verification failed - might need manual installation"
fi

# Copy verification scripts to workspace if they don't exist
if [ ! -d "verify_setup" ]; then
    log_info "Creating basic verification scripts..."
    mkdir -p verify_setup
    
    cat > verify_setup/verify_python.py << 'EOF'
#!/usr/bin/env python3
import sys
print(f"Python {sys.version}")
print(f"Executable: {sys.executable}")
print("✅ Python verification passed")
EOF

    cat > verify_setup/verify_torch.py << 'EOF'
#!/usr/bin/env python3
try:
    import torch
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
    print("✅ PyTorch verification passed")
except ImportError as e:
    print(f"❌ PyTorch not available: {e}")
    sys.exit(1)
EOF

    chmod +x verify_setup/*.py
    log_success "Verification scripts created"
fi

log_success "Environment setup completed!"
echo "🎉 You can now run './dev.sh help' for available commands"