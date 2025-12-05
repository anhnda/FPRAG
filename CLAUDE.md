# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing and comparing **Fast-R-PRAQ** (Fast Risk-aware Post-activation Quantization) against **AWQ** (Activation-aware Weight Quantization) for Large Language Model compression. The project focuses on quantizing the MiniCPM-2B model to 4-bit precision while preserving model quality.

### Core Concept

Fast-R-PRAQ addresses a critical limitation in existing quantization methods: it accounts for "risky dead neurons" - channels with large negative pre-activations (mean < -3 for SiLU) that appear inactive but have:
- High variance (large std)
- Large weight magnitudes
- Risk of "resurrection" when quantization noise pushes them over the activation threshold

The key insight is that quantization noise is proportional to weight magnitude, so high-weight channels with negative pre-activations can unexpectedly activate post-quantization, causing accuracy degradation.

## Architecture

### Quantization Strategies

**Fast-R-PRAQ v3 Algorithm:**
1. Computes pre-activation statistics (mean, std) per channel
2. Estimates quantization noise impact: `noise = input_mag × weight_mag × noise_factor`
3. Calculates risk-adjusted upper bound: `z_risk_upper = (mean + 3×std) + noise`
4. Computes probability of activation: `P(active) = sigmoid(β × (z_risk_upper - τ))`
5. Combines with utility magnitude: `importance = P(active) × magnitude`
6. Groups channels by hardware block size (default: 32) for efficient quantization

**Hybrid Approach (quantize_minicpm_PRAQ.py):**
- MLP layers: Use post-activation importance (ReLU-family activations drop negatives)
- Attention layers: Use AWQ-style pre-activation magnitude

### File Purposes

**convert_to_safetensors.py** - Model format converter:
- Converts models from PyTorch bin format to safetensors format
- Required for AWQ quantization (AutoAWQ only accepts safetensors)
- Downloads MiniCPM-2B and saves to `./models/MiniCPM-2B-sft-bf16-safetensors`
- Run this first if you get "model.safetensors not found" error

**awq_vs_fprpa.py** - Synthetic benchmark demonstrating the "loud silence" problem where naive methods fail:
- Generates synthetic data with Group A (useful, active) and Group B (dangerous, saturated high-energy)
- Shows AWQ accidentally handles Group B correctly by keeping high magnitudes
- Demonstrates Fast-R-PRAQ v3 correctly identifies both utility and risk

**quantize_minicpm_awq.py** - AWQ baseline quantization:
- Uses AutoAWQ library for standard 4-bit quantization
- Simple importance metric: `E[|XW + b|]` (output magnitude)
- Saves to `./quantized_models/minicpm_awq`

**quantize_minicpm_PRAQ.py** - Fast-R-PRAQ hybrid quantization implementation:
- Automatically detects layer types (MLP vs Attention) via naming heuristics
- Implements risk-aware importance scoring for MLP layers
- Simulates INT4 quantization with noise injection
- Saves to `./quantized_models/minicpm_praq_hybrid`

**compare_models.py** - Evaluation framework:
- Loads original FP16, AWQ-quantized, and PRAQ-quantized models
- Evaluates perplexity on WikiText-2 validation set (2000 samples)
- Reports model size, throughput, and perplexity degradation

**visualize_preactivations.py** - Multi-layer hypothesis testing:
- Hooks all linear layers to capture pre-activation distributions
- Identifies "critical channels" (dead + risky + high weight)
- Creates per-layer visualizations showing risk regions
- Generates summary statistics across all layers

**visualize_layer_focused.py** - Single-layer deep analysis:
- Focused analysis on a specific layer (default: layer 28)
- Computes both AWQ and PRAQ importance scores for direct comparison
- Analyzes correlation between methods (Spearman rank correlation)
- Visualizes individual channel distributions for negative channels
- Exports detailed per-channel statistics to CSV

**check_mse_layer.py** - Layer-level MSE comparison:
- Direct MSE comparison of AWQ vs FastRPRAQ on a specific layer's output (default: layer 28)
- Uses calibration data (WikiText-2 train) to compute importance scores
- Evaluates on validation data (WikiText-2 validation) to measure MSE
- Simulates mixed-precision quantization (50% channels in FP16, 50% in INT4)
- Reports per-channel MSE analysis showing which channels benefit most from each method
- Provides empirical evidence for which importance scoring method produces lower reconstruction error

## Commands

### Running Quantization

```bash
# IMPORTANT: AWQ requires safetensors format
# If you get "model.safetensors not found" error, run conversion first:
python convert_to_safetensors.py
# Then update model_name in quantize_minicpm_awq.py to use the local path

# Quantize with AWQ (baseline)
python quantize_minicpm_awq.py

# Quantize with Fast-R-PRAQ hybrid approach (works with PyTorch bins)
python quantize_minicpm_PRAQ.py

# Run synthetic benchmark
python awq_vs_fprpa.py
```

### Evaluation

```bash
# Compare all three models (requires both quantized models)
python compare_models.py
```

### Analysis & Visualization

```bash
# Analyze pre-activation distributions across all layers
python visualize_preactivations.py

# Deep dive into specific layer (modify target_layer_id in main())
python visualize_layer_focused.py

# Compare MSE on layer 28 between AWQ and FastRPRAQ
python check_mse_layer.py
```

## Key Parameters

**Quantization Parameters (in quantize_minicpm_PRAQ.py):**
- `beta=3.0` - Temperature for probability calculation (higher = sharper transitions)
- `tau=-3.0` - Activation threshold for SiLU (where activation starts)
- `noise_factor=0.2` - Estimated INT4 quantization noise ratio (20% of weight magnitude)
- `group_size=32` - Hardware block size for grouped quantization
- `keep_ratio=0.5` - Fraction of channels kept in higher precision (50%)
- `bits=4` - Quantization bit width

**Model Configuration:**
- Default model: `openbmb/MiniCPM-2B-sft-bf16`
- Calibration dataset: WikiText-2 train split
- Calibration samples: 500
- Evaluation samples: 2000
- Max sequence length: 512 tokens

## Layer Naming Heuristics

The automatic layer type detection uses these patterns:

**MLP Layers:** Keywords in layer name (case-insensitive):
- `mlp`, `fc`, `gate`, `up_proj`, `down_proj`, `ffn`

**Attention Layers:** Keywords in layer name:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`, `qkv`, `out_proj`, `attention`

**Default:** If uncertain, treated as MLP (safer assumption)

## Expected Outputs

**Quantized Models:** Saved to `./quantized_models/` with subdirectories:
- `minicpm_awq/` - AWQ quantized model
- `minicpm_praq_hybrid/` - Fast-R-PRAQ quantized model

**Visualizations:** Saved to `./visualizations/` with subdirectories:
- `preactivation_analysis/` - Multi-layer analysis plots
- `layer_X_focused/` - Single-layer deep dive (X = layer number)
  - `channel_statistics.csv` - Per-channel statistics
  - `*.png` - Various visualization plots

## Hardware Requirements

- CUDA-compatible GPU recommended (tested on CUDA)
- Minimum 16GB GPU memory for MiniCPM-2B quantization
- Falls back to CPU if CUDA unavailable (significantly slower)

## Dependencies

Core libraries used:
- `torch` - PyTorch for model operations
- `transformers` - HuggingFace Transformers for model loading
- `awq` - AutoAWQ library for AWQ quantization baseline
- `datasets` - HuggingFace Datasets for WikiText-2
- `matplotlib`, `seaborn` - Visualization
- `numpy`, `pandas`, `scipy` - Data analysis
- `tqdm` - Progress bars

## Research Context

This project tests the hypothesis that standard quantization methods (like AWQ) fail to account for the risk of quantization noise "resurrecting" dead neurons. The critical insight is:

**Standard approaches see:**
- Channel has mean = -10 → "Dead neuron, safe to quantize aggressively"

**Fast-R-PRAQ sees:**
- Channel has mean = -10, std = 4, weight_mag = 5
- Quantization noise ~ 5 × 0.2 = 1.0
- Risk-adjusted upper bound: -10 + 3×4 + 1.0 = 3.0 (above SiLU threshold!)
- Conclusion: "Risky channel, must preserve precision"

The visualization tools empirically test whether such critical channels exist in real models (MiniCPM-2B).
