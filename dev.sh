#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Simplified Development Script for torch-starter (Fixed)
# ==============================================================================

# Colors
COLOR_RESET="\033[0m"
COLOR_RED="\033[0;31m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_BLUE="\033[0;34m"

log_error()   { echo -e "${COLOR_RED}

cmd_benchmark() {
  ensure_venv
  $(python_exec) -c "
import time
print('⚡ Performance Benchmarks')
print('=' * 40)

# CPU benchmark
print('\n🚀 CPU Benchmark')
start = time.time()
result = sum(i**2 for i in range(100000))
cpu_time = time.time() - start
print(f'  Python computation: {cpu_time:.3f}s')

# NumPy benchmark
try:
    import numpy as np
    start = time.time()
    a = np.random.randn(1000, 1000)
    b = np.random.randn(1000, 1000)
    c = np.dot(a, b)
    numpy_time = time.time() - start
    print(f'  NumPy matrix mult: {numpy_time:.3f}s')
    print(f'  NumPy speedup: {cpu_time/numpy_time:.1f}x')
except ImportError:
    print('  ⚠️ NumPy not available')

# GPU benchmark
print('\n🔥 GPU Benchmark')
try:
    import torch
    if not torch.cuda.is_available():
        print('  ⚠️ CUDA not available')
    else:
        device = torch.device('cuda:0')
        
        # CPU vs GPU comparison
        a_cpu = torch.randn(1000, 1000)
        b_cpu = torch.randn(1000, 1000)
        start = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_torch_time = time.time() - start
        print(f'  PyTorch CPU: {cpu_torch_time:.3f}s')
        
        # GPU benchmark
        a_gpu = torch.randn(1000, 1000, device=device)
        b_gpu = torch.randn(1000, 1000, device=device)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f'  PyTorch GPU: {gpu_time:.3f}s')
        print(f'  GPU speedup: {cpu_torch_time/gpu_time:.1f}x')
        
        # Memory info
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f'  GPU memory: {allocated:.1f} MB')
        
        # Cleanup
        del a_cpu, b_cpu, c_cpu, a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()
        
except ImportError:
    print('  ⚠️ PyTorch not available')
"
}

cmd_doctor() {
  section "System Health Check"
  cmd_verify_setup
  
  log_info "Checking system resources..."
  $(python_exec) -c "
import platform
import psutil if 'psutil' in locals() else None

print('💻 System Information')
print('=' * 30)
print(f'OS: {platform.system()} {platform.release()}')
print(f'Architecture: {platform.machine()}')
print(f'Python: {platform.python_version()}')

# Try to get memory info
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f'Memory: {mem.available//1024//1024}MB / {mem.total//1024//1024}MB available')
except ImportError:
    # Fallback for systems without psutil
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                mem_info[key.strip()] = value.strip()
        
        total = int(mem_info.get('MemTotal', '0').split()[0]) // 1024
        available = int(mem_info.get('MemAvailable', '0').split()[0]) // 1024
        print(f'Memory: {available}MB / {total}MB available')
    except:
        print('Memory: Unknown')

# Container check
from pathlib import Path
if Path('/.dockerenv').exists():
    print('Environment: Docker container ✅')
else:
    print('Environment: Native/VM')
"
  
  log_info "Running performance tests..."
  cmd_benchmark || log_warning "Performance tests had issues"
  
  log_success "Health check completed"
}

cmd_help() {
  cat <<EOF
Usage: $0 <command> [options]

Environment Management:
  sync              Sync dependencies with UV
  install <pkg>     Install a package with UV
  clean             Clean caches and temporary files

Development:
  verify-setup      Verify Python, PyTorch, and CUDA setup
  test [args]       Run tests with pytest
  lint              Run ruff and mypy (read-only)
  lint-fix          Run ruff with auto-fix
  format            Format code with ruff
  shell             Start shell with activated environment

Services:
  jupyter [port]    Start Jupyter Lab (default port: 8888)

Diagnostics:
  gpu-info          Show GPU information
  benchmark         Run performance benchmarks
  doctor            Complete system health check

Utilities:
  help              Show this help message
EOF
}

# ==============================================================================
# Main dispatcher
# ==============================================================================

cmd="${1:-help}"
shift || true

case "$cmd" in
  sync)             cmd_sync ;;
  install)          cmd_install "$@" ;;
  clean)            cmd_clean ;;
  verify-setup)     cmd_verify_setup ;;
  test)             cmd_test "$@" ;;
  lint)             cmd_lint ;;
  lint-fix)         cmd_lint_fix ;;
  format)           cmd_format ;;
  shell)            cmd_shell ;;
  jupyter)          cmd_jupyter "$@" ;;
  gpu-info)         cmd_gpu_info ;;
  benchmark)        cmd_benchmark ;;
  doctor)           cmd_doctor ;;
  help|""|-h|--help) cmd_help ;;
  *) 
    log_error "Unknown command: $cmd"
    cmd_help
    exit 1 
    ;;
esac[ERROR] $*${COLOR_RESET}" >&2; }
log_success() { echo -e "${COLOR_GREEN}[ OK ] $*${COLOR_RESET}"; }
log_warning() { echo -e "${COLOR_YELLOW}[WARN] $*${COLOR_RESET}"; }
log_info()    { echo -e "${COLOR_BLUE}[INFO] $*${COLOR_RESET}"; }

section() { echo -e "\n${COLOR_BLUE}=== $* ===${COLOR_RESET}"; }

# Check if virtual environment exists
ensure_venv() {
  if [[ ! -d ".venv" ]]; then
    log_error "Virtual environment not found. Run: uv venv"
    exit 1
  fi
}

# Python executable
python_exec() {
  echo ".venv/bin/python"
}

# ==============================================================================
# Commands
# ==============================================================================

cmd_sync() {
  log_info "Syncing dependencies with UV..."
  if uv sync; then
    log_success "Dependencies synced"
  else
    log_warning "UV sync failed, trying without lockfile..."
    uv sync --no-lock || log_error "UV sync failed completely"
  fi
}

cmd_install() {
  local package="${1:-}"
  if [[ -z "$package" ]]; then
    log_error "Usage: $0 install <package>"
    exit 1
  fi
  log_info "Installing $package..."
  uv add "$package"
}

cmd_clean() {
  log_info "Cleaning environment..."
  rm -rf .pytest_cache __pycache__ */__pycache__ build dist *.egg-info
  find . -type f -name '*.pyc' -delete 2>/dev/null || true
  uv cache clean 2>/dev/null || true
}

cmd_verify_setup() {
  section "Environment Verification"
  ensure_venv
  
  local py
  py=$(python_exec)
  
  log_info "Running comprehensive verification..."
  
  # Basic Python verification
  if "$py" -c "import sys; print(f'✅ Python {sys.version}')"; then
    log_success "Python verification passed"
  else
    log_error "Python verification failed"
    return 1
  fi
  
  # PyTorch verification
  if "$py" -c "import torch; print(f'✅ PyTorch {torch.__version__}')"; then
    log_success "PyTorch verification passed"
    
    # CUDA check
    "$py" -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA {torch.version.cuda} with {torch.cuda.device_count()} device(s)')
    # Simple CUDA test
    x = torch.tensor([1.0], device='cuda')
    print('✅ CUDA tensor creation successful')
else:
    print('⚠️ CUDA not available - CPU mode only')
"
  else
    log_warning "PyTorch verification failed"
  fi
  
  # Run verification scripts if they exist
  if [ -f "verify_setup/verify_python.py" ]; then
    "$py" verify_setup/verify_python.py || log_warning "Python verification script failed"
  fi
  
  if [ -f "verify_setup/verify_torch.py" ]; then
    "$py" verify_setup/verify_torch.py || log_warning "PyTorch verification script failed"
  fi
  
  log_success "Environment verification completed"
}

cmd_test() {
  ensure_venv
  if [[ -d "tests" ]] && command -v .venv/bin/pytest &> /dev/null; then
    .venv/bin/pytest -v "${@}"
  else
    log_warning "pytest not available or no tests directory found"
    # Run basic tests
    $(python_exec) -c "
print('🧪 Running basic tests...')
import torch
import numpy as np

# Basic functionality test
x = torch.tensor([1, 2, 3])
y = np.array([1, 2, 3])
print('✅ Basic tensor operations work')

if torch.cuda.is_available():
    x_cuda = torch.tensor([1, 2, 3], device='cuda')
    print('✅ CUDA tensor operations work')

print('✅ All basic tests passed')
"
  fi
}

cmd_lint() {
  ensure_venv
  if command -v .venv/bin/ruff &> /dev/null; then
    .venv/bin/ruff check .
  else
    log_warning "ruff not available"
  fi
  
  if command -v .venv/bin/mypy &> /dev/null; then
    .venv/bin/mypy . || log_warning "mypy check had issues"
  else
    log_warning "mypy not available"
  fi
}

cmd_format() {
  ensure_venv
  if command -v .venv/bin/ruff &> /dev/null; then
    log_info "Formatting code with ruff..."
    .venv/bin/ruff format .
    log_success "Code formatting completed"
  else
    log_warning "ruff not available for formatting"
  fi
}

cmd_lint_fix() {
  ensure_venv
  if command -v .venv/bin/ruff &> /dev/null; then
    log_info "Running ruff linter with auto-fix..."
    .venv/bin/ruff check . --fix
    log_success "Lint fixes applied"
  else
    log_warning "ruff not available for linting"
  fi
}

cmd_jupyter() {
  ensure_venv
  local port="${1:-8888}"
  if command -v .venv/bin/jupyter &> /dev/null; then
    log_info "Starting Jupyter Lab on port $port..."
    .venv/bin/jupyter lab --ip=0.0.0.0 --port="$port" --no-browser --allow-root
  else
    log_error "Jupyter not available. Install with: ./dev.sh install jupyter"
  fi
}

cmd_shell() {
  ensure_venv
  log_info "Starting shell with activated environment..."
  bash --init-file <(echo "source .venv/bin/activate; echo '🐍 Virtual environment activated'")
}

cmd_gpu_info() {
  ensure_venv
  $(python_exec) -c "
print('🖥️ GPU Information')
print('=' * 30)

# Check nvidia-smi
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                name, memory, driver = [x.strip() for x in line.split(',')]
                print(f'GPU {i}: {name}')
                print(f'  Memory: {memory}')
                print(f'  Driver: v{driver}')
    else:
        print('❌ nvidia-smi not available')
except FileNotFoundError:
    print('❌ nvidia-smi not found')

# Check PyTorch CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f'\n🔥 PyTorch CUDA Info:')
        print(f'  Devices: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props