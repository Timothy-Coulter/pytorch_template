#!/usr/bin/env python3
"""Verify PyTorch and CUDA setup."""

import sys
import time


def main():
    print("🔥 PyTorch Verification")
    print("=" * 40)
    
    try:
        # Test PyTorch import
        start_time = time.time()
        import torch
        import_time = time.time() - start_time
        
        print(f"✅ PyTorch v{torch.__version__} imported ({import_time:.2f}s)")
        
        # CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA devices: {device_count}")
            
            # Get CUDA version
            cuda_version = torch.version.cuda
            print(f"PyTorch CUDA version: {cuda_version}")
            
            # Test basic CUDA operation
            try:
                x = torch.tensor([1.0], device='cuda')
                print(f"✅ CUDA tensor test passed: {x.device}")
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ CUDA tensor test failed: {e}")
                return False
                
        else:
            print("⚠️  CUDA not available - CPU mode only")
        
        # Test torchvision if available
        try:
            import torchvision
            print(f"✅ torchvision v{torchvision.__version__}")
        except ImportError:
            print("⚠️  torchvision not available")
        
        print("\n✅ PyTorch verification passed")
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch verification failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)