#!/usr/bin/env python3
"""Quick test to verify command-line arguments are being parsed correctly."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
parser.add_argument("--output-dir", type=str, default="./quantized_models/model_adaround_xl")
parser.add_argument("--n-calib", type=int, default=128)
parser.add_argument("--adaround-iters", type=int, default=2000)
parser.add_argument("--adaround-lr", type=float, default=1e-3)
parser.add_argument("--layer-batch-size", type=int, default=16)

args = parser.parse_args()

print("=" * 80)
print("PARSED ARGUMENTS:")
print("=" * 80)
print(f"  Model path: {args.model_path}")
print(f"  Output dir: {args.output_dir}")
print(f"  Calibration samples: {args.n_calib}")
print(f"  AdaRound iterations: {args.adaround_iters}")
print(f"  Learning rate: {args.adaround_lr}")
print(f"  Layer batch size: {args.layer_batch_size}")
print("=" * 80)

if args.adaround_iters != 2000:
    print("✅ CUSTOM ARGUMENTS DETECTED!")
else:
    print("❌ USING DEFAULTS - YOUR ARGUMENTS WERE NOT PARSED!")
