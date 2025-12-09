# AutoAWQ Library vs Custom Implementation Comparison Guide

This guide explains how to quantize MiniCPM-2B using both the AutoAWQ library and the custom `gw_awq_asym_l2.py` implementation, then compare their performance.

## Overview

**AutoAWQ Library (`quantize_autoawq_library.py`)**
- Official AWQ implementation from MIT Han Lab
- Symmetric quantization (centered around zero)
- Standard AWQ importance: `E[|XW|]`
- Optimized CUDA kernels for fast INT4 inference
- Group-wise quantization (default group_size=128)

**Custom Implementation (`gw_awq_asym_l2.py`)**
- Asymmetric quantization using full [0, 15] range
- L2-based importance: `E[X²]` (better MSE alignment)
- Custom grid search for optimal scaling factor α
- Group-wise quantization with per-group zero points

## Step-by-Step Usage

### Step 1: Quantize with AutoAWQ Library

```bash
# Basic usage (uses default settings)
python quantize_autoawq_library.py

# With custom settings
python quantize_autoawq_library.py \
  --model-path openbmb/MiniCPM-2B-sft-bf16 \
  --output-dir ./quantized_models/minicpm_autoawq \
  --w-bit 4 \
  --q-group-size 128 \
  --zero-point \
  --calib-samples 128
```

**Parameters:**
- `--model-path`: HuggingFace model ID or local path
- `--output-dir`: Where to save quantized model
- `--w-bit`: Weight quantization bits (default: 4)
- `--q-group-size`: Group size for quantization (default: 128)
- `--zero-point`: Enable asymmetric quantization
- `--calib-samples`: Number of calibration samples from WikiText-2

**Expected Output:**
```
Model size before quantization: 4862.45 MB
Model size after quantization: 1521.78 MB
Compression ratio: 3.19x
Quantized model saved to: ./quantized_models/minicpm_autoawq
```

### Step 2: Quantize with Custom Implementation

```bash
# Basic usage
python gw_awq_asym_l2.py

# With custom settings
python gw_awq_asym_l2.py \
  --n-calib 128 \
  --n-grid 20 \
  --group-size 128 \
  --output-dir ./quantized_models/minicpm_gw_awq_asym_l2
```

**Parameters:**
- `--n-calib`: Number of calibration samples (default: 128)
- `--n-grid`: Grid search points for α (default: 20)
- `--group-size`: Group size for quantization (default: 128)
- `--output-dir`: Where to save quantized model

**Expected Output:**
```
Model size before quantization: 4862.45 MB
Model size after quantization: 4862.45 MB (still FP16 dequantized)
Compression ratio: 1.00x (model is dequantized after quantization)
Quantized model saved to: ./quantized_models/minicpm_gw_awq_asym_l2
```

**Note:** The custom implementation applies quantization noise but stores weights in FP16 format (dequantized). This is for research purposes to evaluate quantization quality.

### Step 3: Compare Both Models

```bash
# Full comparison (includes original FP16 model)
python compare_autoawq_vs_custom.py

# Quick comparison (skip original model and speed tests)
python compare_autoawq_vs_custom.py \
  --skip-original \
  --skip-speed \
  --n-samples 50

# Custom model paths
python compare_autoawq_vs_custom.py \
  --original-model openbmb/MiniCPM-2B-sft-bf16 \
  --autoawq-model ./quantized_models/minicpm_autoawq \
  --custom-model ./quantized_models/minicpm_gw_awq_asym_l2 \
  --n-samples 100 \
  --output-json ./comparison_results.json
```

**Parameters:**
- `--original-model`: Path to original FP16 model
- `--autoawq-model`: Path to AutoAWQ quantized model
- `--custom-model`: Path to custom quantized model
- `--n-samples`: Number of validation samples for perplexity (default: 100)
- `--output-json`: Output file for results (default: ./comparison_results.json)
- `--skip-original`: Skip evaluation of original model (saves time)
- `--skip-speed`: Skip throughput measurement (saves time)

**Expected Output:**
```
COMPARISON SUMMARY
================================================================================

Model                          Size (MB)    PPL        Throughput      Compression
--------------------------------------------------------------------------------
original_fp16                  4862.45      12.3456    1234.56         1.00x
autoawq_library                1521.78      12.8901    2345.67         3.19x
custom_gw_awq_asym_l2          4862.45      12.7654    1256.78         1.00x

AutoAWQ vs Custom Comparison
================================================================================
Perplexity difference: -0.1247 (-0.97%)
✅ Custom implementation achieves LOWER perplexity (better)

Throughput difference: -1088.89 tokens/sec (-46.41%)
⚠️  Custom implementation is SLOWER
```

## Expected Results & Interpretation

### Perplexity (Lower is Better)
- **Original FP16**: Baseline perplexity (typically 12-13 on WikiText-2)
- **AutoAWQ Library**: Small degradation (+0.3-0.5 PPL is typical for 4-bit)
- **Custom gw_awq_asym_l2**: Should be competitive or better due to:
  - Asymmetric quantization using full [0,15] range
  - L2 salience better aligns with MSE objective

### Model Size
- **AutoAWQ Library**: ~4x compression (actual INT4 storage)
- **Custom Implementation**: No compression (stores dequantized FP16 weights)
  - This is intentional for research/evaluation purposes
  - To get true compression, weights would need to be stored in INT4 format

### Throughput
- **AutoAWQ Library**: Much faster due to optimized CUDA kernels for INT4
- **Custom Implementation**: Slower (uses standard FP16 operations)
  - Would need custom kernels for true INT4 inference speed

## Troubleshooting

### Error: "model.safetensors not found"
AutoAWQ requires models in safetensors format. Convert your model:
```bash
python convert_to_safetensors.py  # If you have this script
# Or download pre-converted model from HuggingFace
```

### Error: "CUDA out of memory"
Reduce calibration samples:
```bash
python quantize_autoawq_library.py --calib-samples 64
python gw_awq_asym_l2.py --n-calib 64
```

### Error: "AutoAWQ model not found"
Ensure you ran the quantization script first:
```bash
ls ./quantized_models/minicpm_autoawq/
# Should contain: config.json, model.safetensors, etc.
```

### Comparison script fails to load models
Check that output directories match:
```bash
# List quantized models
ls -la ./quantized_models/

# Verify paths in comparison script match actual directories
python compare_autoawq_vs_custom.py \
  --autoawq-model ./quantized_models/YOUR_ACTUAL_PATH \
  --custom-model ./quantized_models/YOUR_ACTUAL_PATH
```

## Research Questions to Explore

1. **Does L2 salience improve quality over L1?**
   - Compare perplexity degradation: L2-based vs standard AWQ

2. **Does asymmetric quantization help?**
   - Compare asymmetric [0,15] vs symmetric [-7,7]

3. **How does rounding behavior differ?**
   - Use `gw_awq_asym_l2_stats.py` to analyze rounding statistics

4. **Layer-wise analysis**
   - Which layers benefit most from asymmetric quantization?
   - Which layers have highest rounding UP/DOWN ratios?

## Output Files

After running all scripts, you should have:
```
./quantized_models/
  ├── minicpm_autoawq/              # AutoAWQ library quantized model
  │   ├── config.json
  │   ├── model.safetensors
  │   └── tokenizer files
  └── minicpm_gw_awq_asym_l2/       # Custom quantized model
      ├── config.json
      ├── pytorch_model.bin
      └── tokenizer files

./comparison_results.json             # Detailed comparison metrics
./rounding_stats.csv                  # Optional: from gw_awq_asym_l2_stats.py
```

## Next Steps

1. **Run perplexity evaluation on more datasets**
   - Try LAMBADA, C4, or domain-specific datasets

2. **Layer-wise MSE comparison**
   - Use `check_mse_layer.py` to compare per-layer reconstruction error

3. **Analyze critical channels**
   - Use `visualize_layer_focused.py` to identify which channels benefit most

4. **Mixed-precision quantization**
   - Keep high-importance channels in FP16, quantize rest to INT4
   - Compare different keep_ratio values (0.3, 0.5, 0.7)

## References

- **AutoAWQ Paper**: https://arxiv.org/abs/2306.00978
- **AutoAWQ Library**: https://github.com/casper-hansen/AutoAWQ
- **MiniCPM Model**: https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16
