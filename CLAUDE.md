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

**quantize_minicpm_awq.py** - AWQ baseline quantization (library-based):
- Uses AutoAWQ library for standard 4-bit quantization
- Simple importance metric: `E[|XW + b|]` (output magnitude)
- Requires safetensors format (run convert_to_safetensors.py first)
- Saves to `./quantized_models/minicpm_awq`

**quantize_minicpm_AWQ.py** - AWQ custom implementation (library-free):
- Pure PyTorch implementation of AWQ algorithm
- Same importance metric as library version: `E[|XW^T + b|]`
- Works with PyTorch bin files (no safetensors required)
- Mirrors the structure of quantize_minicpm_PRAQ.py for fair comparison
- Saves to `./quantized_models/minicpm_awq_custom`

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
# Quantize with AWQ - Custom Implementation (RECOMMENDED)
# Pure PyTorch, works with PyTorch bins, no dependencies
python quantize_minicpm_AWQ.py

# Quantize with AWQ - Library Version (requires safetensors)
# If you get "model.safetensors not found" error, run conversion first:
# python convert_to_safetensors.py
python quantize_minicpm_awq.py

# Quantize with Fast-R-PRAQ hybrid approach
python quantize_minicpm_PRAQ.py

# Run synthetic benchmark comparing AWQ vs FastRPRAQ
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

## Common Issues & Solutions

When writing new analysis or quantization scripts for this project, be aware of these recurring issues:

### 1. Model Cache Compatibility Error

**Error:**
```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'. Did you mean: 'get_seq_length'?
```

**Cause:** Version mismatch between transformers library and MiniCPM model code.

**Solution:** Always disable caching when running model forward passes:
```python
# BAD - will cause error
outputs = model(**inputs)

# GOOD - disables problematic caching
outputs = model(**inputs, use_cache=False)
```

### 2. Variable Sequence Length Concatenation Error

**Error:**
```
RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 13 but got size 184 for tensor number 1 in the list.
```

**Cause:** Different text samples have different sequence lengths after tokenization. Attempting to concatenate tensors with shapes `[seq_len_1, hidden_dim]` and `[seq_len_2, hidden_dim]` fails.

**Solution:** Reshape each tensor to `[-1, hidden_dim]` before concatenating:
```python
# BAD - fails with variable sequence lengths
all_activations = torch.cat(activation_list, dim=0)

# GOOD - reshape first to handle variable lengths
reshaped = [x.reshape(-1, x.shape[-1]) for x in activation_list]
all_activations = torch.cat(reshaped, dim=0)
```

### 3. BFloat16 NumPy Incompatibility

**Error:**
```
TypeError: Got unsupported ScalarType BFloat16
```

**Cause:** The MiniCPM model uses bfloat16 precision, but NumPy doesn't support this dtype.

**Solution:** Convert to float32 before calling `.numpy()`:
```python
# BAD - fails if tensor is bfloat16
array = tensor.numpy()

# GOOD - convert to float32 first
array = tensor.float().numpy()
```

### 4. Understanding Pre/Post Activation for SiLU Layers

**CRITICAL:** When capturing pre/post activation for gate_proj layers with SiLU, understand the flow:

```
Input (X) → gate_proj (Linear) → Output (XW) → SiLU → SiLU(XW)
            [in, hidden]         [in, intermediate]    [in, intermediate]
```

**Common mistakes:**
```python
# ❌ WRONG - This captures INPUT and OUTPUT of gate_proj
def hook_fn(module, input, output):
    pre_act = input[0]   # This is X (input to linear layer), NOT pre-activation
    post_act = output    # This is XW (output of linear layer), NOT post-activation
```

**Correct approaches:**

**Option 1: Manual SiLU application (simpler)**
```python
def hook_fn(module, input, output):
    pre_act = output.detach().cpu()          # XW (pre-activation for SiLU)
    post_act = silu(pre_act)                 # SiLU(XW) (post-activation)
    self.pre_activation.append(pre_act)
    self.post_activation.append(post_act)

def silu(x):
    return x * torch.sigmoid(x)
```

**Option 2: Hook both gate_proj and act_fn modules (more accurate)**
```python
# Hook gate_proj for pre-activation
def gate_hook(module, input, output):
    pre_activations.append(output.detach().cpu())

# Hook the separate SiLU module for post-activation
def silu_hook(module, input, output):
    post_activations.append(output.detach().cpu())

hook1 = model.layers[i].mlp.gate_proj.register_forward_hook(gate_hook)
hook2 = model.layers[i].mlp.act_fn.register_forward_hook(silu_hook)
```

### 5. Activation Hook Best Practices

When writing hooks to capture activations:

```python
class ActivationCapture:
    def __init__(self):
        self.activations = []

    def hook_fn(self, module, input, output):
        # Always detach and move to CPU immediately to save GPU memory
        act = output.detach().cpu()
        self.activations.append(act)

    def get_concatenated(self):
        # Handle variable sequence lengths
        reshaped = [x.reshape(-1, x.shape[-1]) for x in self.activations]
        # Convert to float32 for numpy compatibility
        result = torch.cat(reshaped, dim=0).float()
        return result
```

**Note:** For AWQ quantization, you need the INPUT to linear layers (not pre/post activation):
```python
def awq_hook(module, input, output):
    # Capture INPUT for computing activation salience
    X = input[0].detach().cpu()
    self.inputs.append(X)
```

### 6. Model Forward Pass Template

Standard template for running MiniCPM model on calibration data:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM-2B-sft-bf16",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-2B-sft-bf16", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

with torch.no_grad():
    for sample in dataset:
        text = sample['text']
        if len(text.strip()) == 0:
            continue

        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=False
        ).to(device)

        # CRITICAL: use_cache=False to avoid compatibility issues
        outputs = model(**inputs, use_cache=False)
```

These patterns should be followed in all new scripts to avoid repeating these common errors.
