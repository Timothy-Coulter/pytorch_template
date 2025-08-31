#!/usr/bin/env python3
"""
NVIDIA CUDA PyTorch Example
Demonstrates CUDA functionality with the NVIDIA container registry base.
"""

import time
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def get_device_info() -> dict:
    """Get comprehensive device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "devices": []
    }
    
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1024**3,
                "multiprocessors": props.multi_processor_count,
            }
            info["devices"].append(device_info)
    
    return info


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for CUDA demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def create_synthetic_dataset(
    num_samples: int = 1000,
    input_size: int = 784,
    num_classes: int = 10,
    batch_size: int = 64
) -> DataLoader:
    """Create synthetic dataset for demonstration."""
    # Generate random data
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def benchmark_computation(device: torch.device, matrix_size: int = 2048) -> Tuple[float, float]:
    """Benchmark matrix multiplication on specified device."""
    # Generate random matrices
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    
    # Warm-up (important for GPU timing)
    for _ in range(3):
        _ = torch.mm(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    result = torch.mm(a, b)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU computation to finish
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    # Calculate performance metrics
    flops = 2 * matrix_size**3  # Floating point operations for matrix multiplication
    gflops = flops / computation_time / 1e9
    
    return computation_time, gflops


def train_model_demo(device: torch.device, num_epochs: int = 3) -> None:
    """Demonstrate training a model with CUDA."""
    print(f"\n🧠 Training Demo on {device}")
    print("=" * 40)
    
    # Create model and move to device
    model = SimpleNeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataset
    dataloader = create_synthetic_dataset()
    
    model.train()
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:  # Print every 5 batches
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.6f}")
    
    total_time = time.time() - total_start_time
    print(f"\n✅ Training completed in {total_time:.2f}s")
    
    # Memory usage (CUDA only)
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"📊 GPU Memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached")


def main():
    """Main demonstration function."""
    print("🚀 NVIDIA CUDA PyTorch Demonstration")
    print("=" * 50)
    
    # Display device information
    device_info = get_device_info()
    print("🖥️  Device Information:")
    print(f"  CUDA Available: {device_info['cuda_available']}")
    print(f"  Device Count: {device_info['device_count']}")
    
    if device_info['cuda_available']:
        print(f"  Current Device: {device_info['current_device']}")
        for device_data in device_info['devices']:
            print(f"  GPU {device_data['id']}: {device_data['name']}")
            print(f"    Compute Capability: {device_data['compute_capability']}")
            print(f"    Total Memory: {device_data['total_memory_gb']:.1f} GB")
            print(f"    Multiprocessors: {device_data['multiprocessors']}")
    
    # Benchmark CPU vs GPU
    print("\n⚡ Performance Benchmark")
    print("=" * 30)
    
    # CPU benchmark
    cpu_device = torch.device('cpu')
    cpu_time, cpu_gflops = benchmark_computation(cpu_device)
    print(f"CPU: {cpu_time:.3f}s, {cpu_gflops:.1f} GFLOPS")
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda:0')
        gpu_time, gpu_gflops = benchmark_computation(gpu_device)
        print(f"GPU: {gpu_time:.3f}s, {gpu_gflops:.1f} GFLOPS")
        
        speedup = cpu_time / gpu_time
        print(f"🏎️  GPU Speedup: {speedup:.1f}x")
        
        # Training demonstration
        train_model_demo(gpu_device)
        
        # Cleanup
        torch.cuda.empty_cache()
    else:
        print("⚠️  GPU not available, skipping GPU benchmark")
        train_model_demo(cpu_device)
    
    print("\n✅ Demonstration completed successfully!")


if __name__ == "__main__":
    main()