"""
Simple Uniform INT4 Quantization (No Scaling)

This is a simplified baseline that just quantizes all weights to INT4
without any scaling or optimization. Use this as a fair comparison baseline.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import random
import numpy as np


@torch.no_grad()
def quantize_weight_int4(W):
    """Quantize weights to INT4 (per-channel symmetric)."""
    # Per-channel (per output feature) quantization
    W_abs_max = W.abs().max(dim=1, keepdim=True)[0]
    W_abs_max = W_abs_max.clamp(min=1e-8)

    # INT4 range: [-8, 7]
    scale = W_abs_max / 7.0
    W_quant = torch.round(W / scale).clamp(-8, 7)
    W_dequant = W_quant * scale

    return W_dequant


def quantize_model(model):
    """Quantize all linear layers to INT4."""
    print("\nQuantizing all linear layers to INT4...")
    quantized_count = 0

    for name, module in tqdm(list(model.named_modules()), desc="Quantizing"):
        if isinstance(module, nn.Linear):
            module.weight.data = quantize_weight_int4(module.weight.data)
            quantized_count += 1

    print(f"✅ Quantized {quantized_count} layers")
    return model


def main():
    parser = argparse.ArgumentParser(description="Simple INT4 quantization")
    parser.add_argument("--model-name", type=str, default="openbmb/MiniCPM-2B-sft-bf16")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_simple_int4")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("Simple Uniform INT4 Quantization")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print("Method: Per-channel symmetric INT4, no scaling")
    print("="*80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Quantize
    model = quantize_model(model)

    # Save
    print(f"\nSaving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
