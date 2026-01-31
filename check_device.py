#!/usr/bin/env python3
"""
Quick device check for PyTorch acceleration.
"""

import torch
import platform

print("=" * 60)
print("PyTorch Device Check")
print("=" * 60)

print(f"\nSystem: {platform.system()} {platform.machine()}")
print(f"PyTorch version: {torch.__version__}")

print("\n" + "-" * 60)
print("Available Devices:")
print("-" * 60)

# Check CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA: Available")
    print(f"  Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("✗ CUDA: Not available")

# Check MPS (Apple Silicon)
if hasattr(torch.backends, 'mps'):
    if torch.backends.mps.is_available():
        print(f"\n✓ MPS (Apple Silicon): Available")
        if torch.backends.mps.is_built():
            print("  MPS backend is built")
    else:
        print("\n✗ MPS (Apple Silicon): Not available")
        if not torch.backends.mps.is_built():
            print("  Reason: MPS not built in this PyTorch version")
            print("  Install PyTorch with MPS support:")
            print("  pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu")
else:
    print("\n✗ MPS: Not supported (requires PyTorch 1.12+)")

# Test device
print("\n" + "-" * 60)
print("Device Selection:")
print("-" * 60)

if torch.cuda.is_available():
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    device_name = "Apple Silicon GPU (MPS)"
else:
    device = "cpu"
    device_name = "CPU"

print(f"Selected device: {device} ({device_name})")

# Quick performance test
print("\n" + "-" * 60)
print("Performance Test (1000x1000 matrix multiply):")
print("-" * 60)

import time

for test_device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else []):
    try:
        x = torch.randn(1000, 1000, device=test_device)
        y = torch.randn(1000, 1000, device=test_device)

        # Warmup
        for _ in range(3):
            z = torch.matmul(x, y)

        # Benchmark
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        if test_device in ["cuda", "mps"]:
            torch.cuda.synchronize() if test_device == "cuda" else None
        elapsed = time.time() - start

        print(f"{test_device.upper():6s}: {elapsed*10:.2f} ms/iteration")
    except Exception as e:
        print(f"{test_device.upper():6s}: Error - {e}")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)

if device == "cpu":
    print("⚠️  WARNING: No GPU acceleration available!")
    print("   AdaRound quantization will be VERY slow on CPU.")
    print()
    if platform.system() == "Darwin":  # macOS
        print("   For Apple Silicon Macs:")
        print("   1. Check if you have Apple Silicon (M1/M2/M3)")
        print("   2. Install PyTorch with MPS support:")
        print("      pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu")
    else:
        print("   For NVIDIA GPUs:")
        print("   1. Install CUDA toolkit")
        print("   2. Install PyTorch with CUDA support:")
        print("      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
else:
    print(f"✓ Using {device.upper()} acceleration - good to go!")

print("=" * 60)
