#!/usr/bin/env python3
"""Verify Python environment setup."""

import sys
import platform
from pathlib import Path


def main():
    print("🐍 Python Environment Verification")
    print("=" * 40)
    
    # Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 11):
        print("❌ Python 3.11+ required")
        return False
        
    print("✅ Python version OK")
    
    # Platform info
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python executable: {sys.executable}")
    
    # Virtual environment check
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment")
    
    # Workspace check
    workspace = Path("/workspaces/torch-starter")
    if workspace.exists():
        print(f"✅ Workspace found: {workspace}")
    else:
        print(f"⚠️  Workspace not found: {workspace}")
    
    print("\n✅ Python environment verification passed")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)