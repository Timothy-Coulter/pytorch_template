# verify_setup/gpu_info.py
#!/usr/bin/env python3
"""Display GPU information."""

import subprocess
import sys


def main():
    print("🖥️  GPU Information")
    print("=" * 30)
    
    try:
        # Get detailed GPU info
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu", "--format=csv"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            headers = [h.strip() for h in lines[0].split(',')]
            
            for i, line in enumerate(lines[1:], 1):
                values = [v.strip() for v in line.split(',')]
                print(f"\nGPU {i-1}:")
                for header, value in zip(headers, values):
                    print(f"  {header}: {value}")
        else:
            print("❌ nvidia-smi command failed")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # PyTorch GPU info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🔥 PyTorch GPU Info:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {props.name}")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Total Memory: {props.total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# verify_setup/benchmark.py
#!/usr/bin/env python3
"""Run performance benchmarks."""

import time
import sys


def benchmark_cpu():
    """CPU benchmark."""
    print("🚀 CPU Benchmark")
    
    # Python computation
    start = time.time()
    result = sum(i**2 for i in range(100000))
    python_time = time.time() - start
    print(f"  Python computation: {python_time:.3f}s")
    
    # NumPy benchmark
    try:
        import numpy as np
        start = time.time()
        a = np.random.randn(1000, 1000)
        b = np.random.randn(1000, 1000)
        c = np.dot(a, b)
        numpy_time = time.time() - start
        print(f"  NumPy matrix mult: {numpy_time:.3f}s")
        
        speedup = python_time / numpy_time if numpy_time > 0 else 0
        print(f"  NumPy speedup: {speedup:.1f}x")
        
    except ImportError:
        print("  ⚠️  NumPy not available")


def benchmark_gpu():
    """GPU benchmark."""
    print("\n🔥 GPU Benchmark")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("  ⚠️  CUDA not available")
            return
        
        device = torch.device('cuda:0')
        
        # Warm up
        for _ in range(3):
            a = torch.randn(100, 100, device=device)
            torch.cuda.synchronize()
            del a
        
        # CPU benchmark
        a_cpu = torch.randn(1000, 1000)
        b_cpu = torch.randn(1000, 1000)
        start = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start
        print(f"  PyTorch CPU: {cpu_time:.3f}s")
        
        # GPU benchmark
        a_gpu = torch.randn(1000, 1000, device=device)
        b_gpu = torch.randn(1000, 1000, device=device)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  PyTorch GPU: {gpu_time:.3f}s")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"  GPU speedup: {speedup:.1f}x")
        
        # Memory info
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory: {allocated:.1f} MB")
        
        # Cleanup
        del a_cpu, b_cpu, c_cpu, a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()
        
    except ImportError:
        print("  ⚠️  PyTorch not available")
    except Exception as e:
        print(f"  ❌ GPU benchmark failed: {e}")


def main():
    print("⚡ Performance Benchmarks")
    print("=" * 40)
    
    benchmark_cpu()
    benchmark_gpu()
    
    print("\n✅ Benchmarks completed")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# verify_setup/system_info.py
#!/usr/bin/env python3
"""Display system information."""

import sys
import platform
import subprocess
from pathlib import Path


def get_memory_info():
    """Get memory information."""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        
        mem_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                mem_info[key.strip()] = value.strip()
        
        total = int(mem_info.get('MemTotal', '0').split()[0]) / 1024  # MB
        available = int(mem_info.get('MemAvailable', '0').split()[0]) / 1024  # MB
        
        return f"{available:.0f}MB / {total:.0f}MB available"
    except:
        return "Unknown"


def get_disk_info():
    """Get disk information."""
    try:
        result = subprocess.run(['df', '-h', '/workspaces'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                return f"{parts[3]} available of {parts[1]}"
    except:
        pass
    return "Unknown"


def main():
    print("💻 System Information")
    print("=" * 30)
    
    # Basic system info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Memory: {get_memory_info()}")
    print(f"Disk: {get_disk_info()}")
    
    # Container info
    if Path("/.dockerenv").exists():
        print("Environment: Docker container")
    else:
        print("Environment: Native/VM")
    
    # CUDA info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            print(f"NVIDIA Driver: {driver_version}")
    except:
        print("NVIDIA Driver: Not available")
    
    # Workspace info
    workspace = Path("/workspaces/torch-starter")
    if workspace.exists():
        print(f"Workspace: {workspace} ✅")
    else:
        print(f"Workspace: {workspace} ❌")
    
    print("\n✅ System information gathered")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# verify_setup/performance_test.py
#!/usr/bin/env python3
"""Quick performance tests."""

import time
import sys


def test_imports():
    """Test import performance."""
    print("📦 Testing import performance...")
    
    packages = ['numpy', 'pandas', 'torch', 'transformers']
    
    for pkg in packages:
        try:
            start = time.time()
            __import__(pkg)
            import_time = time.time() - start
            print(f"  {pkg}: {import_time:.2f}s")
        except ImportError:
            print(f"  {pkg}: Not available")


def test_computation():
    """Test basic computation."""
    print("\n⚡ Testing computation performance...")
    
    # Python test
    start = time.time()
    sum(i**2 for i in range(50000))
    python_time = time.time() - start
    print(f"  Python: {python_time:.3f}s")
    
    # NumPy test
    try:
        import numpy as np
        start = time.time()
        a = np.random.randn(500, 500)
        b = np.random.randn(500, 500)
        c = np.dot(a, b)
        numpy_time = time.time() - start
        print(f"  NumPy: {numpy_time:.3f}s")
    except ImportError:
        print("  NumPy: Not available")


def main():
    print("🔬 Quick Performance Tests")
    print("=" * 35)
    
    test_imports()
    test_computation()
    
    print("\n✅ Performance tests completed")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)