#!/usr/bin/env python3
"""Verify core libraries installation."""

import sys
import time


def check_package(name, description=""):
    """Check if a package can be imported."""
    try:
        start = time.time()
        module = __import__(name)
        import_time = time.time() - start
        
        version = getattr(module, "__version__", "unknown")
        desc_str = f" ({description})" if description else ""
        print(f"  ✅ {name:<15} v{version:<12} {desc_str} - {import_time:.3f}s")
        return True
        
    except ImportError:
        desc_str = f" ({description})" if description else ""
        print(f"  ❌ {name:<15} MISSING      {desc_str}")
        return False
    except Exception as e:
        print(f"  ⚠️  {name:<15} ERROR        ({str(e)[:50]})")
        return False


def main():
    print("📦 Core Libraries Verification")
    print("=" * 40)
    
    # Core scientific computing
    print("\nCore Scientific Computing:")
    core_packages = [
        ("numpy", "Scientific computing"),
        ("pandas", "Data manipulation"),
        ("scipy", "Scientific algorithms"),
        ("matplotlib", "Plotting"),
        ("sklearn", "Machine learning"),
    ]
    
    core_success = 0
    for pkg, desc in core_packages:
        if check_package(pkg, desc):
            core_success += 1
    
    # ML/DL packages
    print("\nMachine Learning:")
    ml_packages = [
        ("transformers", "Transformer models"),
        ("datasets", "ML datasets"),
        ("accelerate", "Distributed training"),
        ("safetensors", "Safe tensor storage"),
        ("tqdm", "Progress bars"),
    ]
    
    ml_success = 0
    for pkg, desc in ml_packages:
        if check_package(pkg, desc):
            ml_success += 1
    
    # Optional packages
    print("\nOptional:")
    optional_packages = [
        ("jupyter", "Jupyter notebooks"),
        ("ipykernel", "Jupyter kernel"),
        ("seaborn", "Statistical plots"),
        ("plotly", "Interactive plots"),
    ]
    
    optional_success = 0
    for pkg, desc in optional_packages:
        if check_package(pkg, desc):
            optional_success += 1
    
    # Summary
    total_core = len(core_packages)
    total_ml = len(ml_packages)
    total_optional = len(optional_packages)
    
    print(f"\n📊 Summary:")
    print(f"  Core packages: {core_success}/{total_core}")
    print(f"  ML packages: {ml_success}/{total_ml}")
    print(f"  Optional packages: {optional_success}/{total_optional}")
    
    # Determine success
    critical_missing = (core_success < total_core) or (ml_success < total_ml - 1)
    
    if critical_missing:
        print("\n❌ Critical packages missing")
        return False
    else:
        print("\n✅ Library verification passed")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)