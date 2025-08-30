#!/usr/bin/env python3
"""Verify GPU and CUDA setup."""

import sys
import subprocess


def run_command(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def main():
    print("🖥️  GPU and CUDA Verification")
    print("=" * 40)
    
    # Check nvidia-smi
    success, stdout, stderr = run_command("nvidia-smi --version")
    if success:
        version_line = stdout.split('\n')[0] if stdout else "Unknown version"
        print(f"✅ nvidia-smi available: {version_line}")
    else:
        print("❌ nvidia-smi not available")
        print("⚠️  Running in CPU-only mode")
        return False
    
    # Get GPU information
    success, stdout, stderr = run_command(
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
    )
    if success and stdout:
        print("\n📊 GPU Information:")
        for line in stdout.split('\n'):
            if line.strip():
                name, memory, driver = [x.strip() for x in line.split(',')]
                print(f"  Device: {name}")
                print(f"  Memory: {memory}")
                print(f"  Driver: v{driver}")
    
    # Check CUDA runtime version
    success, stdout, stderr = run_command("nvcc --version")
    if success:
        for line in stdout.split('\n'):
            if 'release' in line.lower():
                print(f"✅ CUDA toolkit: {line.strip()}")
                break
    else:
        print("⚠️  CUDA toolkit not found (using runtime only)")
    
    # Test PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print("\n🔥 PyTorch CUDA Test:")
            device = torch.cuda.get_device_name(0)
            print(f"  Active device: {device}")
            
            # Simple computation test
            a = torch.randn(1000, 1000, device='cuda')
            b = torch.randn(1000, 1000, device='cuda')
            torch.cuda.synchronize()
            
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            print("  ✅ CUDA computation test passed")
            
            # Memory check
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"  GPU memory allocated: {allocated:.1f} MB")
            
            # Cleanup
            del a, b, c
            torch.cuda.empty_cache()
            
        else:
            print("❌ PyTorch CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch CUDA test failed: {e}")
        return False
    
    print("\n✅ GPU verification passed")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)