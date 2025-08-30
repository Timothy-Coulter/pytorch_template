#!/usr/bin/env python3
"""Simple health check for container."""

import sys


def main():
    """Quick health check for Docker healthcheck."""
    try:
        # Test Python
        import torch
        
        # Test basic functionality
        x = torch.tensor([1.0])
        
        # Test CUDA if available (but don't fail if not)
        if torch.cuda.is_available():
            x_cuda = torch.tensor([1.0], device='cuda')
            del x_cuda
            torch.cuda.empty_cache()
        
        print("✅ Health check passed")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)