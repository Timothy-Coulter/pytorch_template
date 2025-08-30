#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Simplified Development Script for torch-starter
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
  uv sync
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
  find . -type f -name '*.pyc' -delete
  uv cache clean 2>/dev/null || true
}

cmd_verify_setup() {
  section "Environment Verification"
  ensure_venv
  
  local py
  py=$(python_exec)
  
  log_info "Running comprehensive verification..."
  
  # Run verification scripts
  "$py" verify_setup/verify_python.py || return 1
  "$py" verify_setup/verify_torch.py || return 1
  "$py" verify_setup/verify_gpu.py || log_warning "GPU verification failed"
  "$py" verify_setup/verify_libs.py || log_warning "Some libraries missing"
  
  log_success "Environment verification completed"
}

cmd_test() {
  ensure_venv
  .venv/bin/pytest -v "${@}"
}

cmd_lint() {
  ensure_venv
  .venv/bin/ruff check .
  .venv/bin/mypy .
}

cmd_format() {
  ensure_venv
  log_info "Formatting code with ruff..."
  .venv/bin/ruff format .
  log_success "Code formatting completed"
}

cmd_lint_fix() {
  ensure_venv
  log_info "Running ruff linter with auto-fix..."
  .venv/bin/ruff check . --fix
  log_success "Lint fixes applied"
}

cmd_typecheck() {
  ensure_venv
  log_info "Running mypy type checker..."
  .venv/bin/mypy .
  log_success "Type checking completed"
}

cmd_docstring_check() {
  ensure_venv
  log_info "Checking docstring coverage..."
  .venv/bin/interrogate . --verbose
  log_success "Docstring coverage check completed"
}

cmd_all_checks() {
  ensure_venv
  log_info "Running all code quality checks..."
  
  section "Formatting"
  cmd_format
  
  section "Linting with fixes"
  cmd_lint_fix
  
  section "Type checking"
  cmd_typecheck
  
  section "Docstring coverage"
  cmd_docstring_check || log_warning "Docstring coverage below threshold"
  
  section "Tests"
  cmd_test
  
  log_success "All checks completed successfully"
}

cmd_jupyter() {
  ensure_venv
  local port="${1:-8888}"
  log_info "Starting Jupyter Lab on port $port..."
  .venv/bin/jupyter lab --ip=0.0.0.0 --port="$port" --no-browser --allow-root
}

cmd_shell() {
  ensure_venv
  log_info "Starting shell with activated environment..."
  bash --init-file <(echo "source .venv/bin/activate")
}

cmd_gpu_info() {
  ensure_venv
  $(python_exec) verify_setup/gpu_info.py
}

cmd_benchmark() {
  ensure_venv
  $(python_exec) verify_setup/benchmark.py
}

cmd_doctor() {
  section "System Health Check"
  cmd_verify_setup
  
  log_info "Checking system resources..."
  $(python_exec) verify_setup/system_info.py
  
  log_info "Running performance tests..."
  $(python_exec) verify_setup/performance_test.py || log_warning "Performance tests had issues"
  
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
  typecheck         Run mypy type checker
  docstring-check   Check docstring coverage with interrogate
  all-checks        Run format, lint-fix, typecheck, docstring-check, and tests
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
  typecheck)        cmd_typecheck ;;
  docstring-check)  cmd_docstring_check ;;
  all-checks)       cmd_all_checks ;;
  jupyter)          cmd_jupyter "$@" ;;
  shell)            cmd_shell ;;
  gpu-info)         cmd_gpu_info ;;
  benchmark)        cmd_benchmark ;;
  doctor)           cmd_doctor ;;
  help|""|-h|--help) cmd_help ;;
  *) 
    log_error "Unknown command: $cmd"
    cmd_help
    exit 1 
    ;;
esac