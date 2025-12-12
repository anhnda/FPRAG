# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project comparing quantization methods for Large Language Model compression, focusing on the MiniCPM-2B model. The project implements and evaluates multiple approaches to 4-bit weight quantization:

1. **Group-Wise AWQ with L2 Salience** - Asymmetric quantization using E[X²] activation importance
2. **Heuristic-Guided Quantization** - Global greedy rounding correction using E[Xs] statistics
3. **Cross-dataset validation** - Evaluation across WikiText-2, C4, and AG News

### Core Research Questions

1. **L2 vs L1 Salience**: Does E[X²] improve upon E[|X|] for identifying important channels?
2. **Heuristic Rounding**: Does global greedy rounding correction reduce quantization error?
3. **Asymmetric vs Symmetric**: Does asymmetric [0,15] quantization outperform symmetric [-7,7]?
4. **Group-wise Quantization**: What is the optimal group size for hardware efficiency vs quality?

## Architecture

### Quantization Pipeline

**Group-Wise AWQ (gw_awq_asym_l2.py):**
1. Capture activations from calibration data (WikiText-2 train)
2. Compute per-input-channel L2 salience: `s[j] = E[X[:,j]²]`
3. Grid search for optimal scaling exponent α ∈ [0, 1]
4. Scale weight columns: `W[:,j] *= s[j]^α`
5. Group-wise asymmetric INT4 quantization [0, 15]
   - Per group (default: 128 channels): `scale = (max - min) / 15`
   - Zero point: `z = round(-min / scale)`
6. Divide by input scales: `W_final = Q(W×s) / s`

**Heuristic-Guided Quantization (awq_op_ref.py):**
- All AWQ steps above, PLUS:
- Global greedy rounding correction:
  1. Compute initial quantization error: `e = x·(w - w_quant)`
  2. Identify flip candidates (exclude outliers)
  3. Sort candidates globally by rounding cost
  4. Find optimal K flips to minimize |error|

## Key Files

### Quantization Implementations

**gw_awq_asym_l2.py** - Group-Wise AWQ with Asymmetric Quantization:
- L2 salience metric: `E[X²]` (better MSE alignment than `E[|X|]`)
- Asymmetric quantization using full [0,15] range
- Group-wise scales with per-group zero points
- Grid search for optimal α (default: 20 points from 0.0 to 1.0)
- Output: `./quantized_models/minicpm_gw_awq_asym_l2/`

**awq_op_ref.py** - Heuristic-Guided Quantization:
- Extends gw_awq_asym_l2.py with global greedy rounding
- Uses E[Xs] statistics to guide rounding decisions
- Outlier masking (default: top 5% activations ignored)
- Output: `./quantized_models/minicpm_awq_op_ref/`

**heuristic_quantize.py** - Heuristic Quantizer Class:
- Corrected implementation matching global greedy logic
- Vectorized for performance
- Includes outlier protection
- Used as reference for awq_op_ref.py

### Evaluation & Comparison

**compare_awq_heuristic.py** - Cross-Dataset Validation:
- Compares Standard AWQ vs Heuristic-Guided AWQ
- Evaluates on 3 datasets:
  - WikiText-2 validation (in-distribution)
  - C4 validation (cross-dataset, web crawl)
  - AG News test (cross-dataset, news)
- Computes perplexity, loss, and statistical significance
- Saves detailed results to JSON

**final_cross_validation.py** - Final Production Validation:
- V1 GWH-PRAQ vs GW-AWQ comparison
- Same 3-dataset evaluation as compare_awq_heuristic.py
- Production-ready evaluation pipeline

**heuristic_verification.py** - Synthetic Verification:
- Reference implementation of quantization methods
- Demonstrates global greedy logic on synthetic data
- Shows improvements over nearest rounding:
  - Non-group global: 0.004118 error
  - Group-wise nearest: 0.001875 error
  - Group-wise global greedy: 0.001396 error (best)

### Analysis & Visualization

**visualize_importances.py** - Channel Importance Visualization:
- Compares E[X] vs E[X²] importance distributions
- Shows sorted vs original channel ordering
- Target: Layer 3 gate_proj
- Output: Importance distribution plots

**stats_pre_vs_post_act.py** - Pre/Post Activation Analysis:
- Captures XW (pre-activation) and SiLU(XW) (post-activation)
- Identifies risky channels (negative pre-activation, high variance)
- Demonstrates activation function effects on distributions

**stats_rounding_error.py** - Rounding Error Statistics:
- Analyzes quantization error decomposition
- Compares rounding strategies (nearest, floor, ceil, heuristic)
- Per-layer error analysis

**stats_scaling_awq.py** - AWQ Scaling Analysis:
- Analyzes effect of α on weight scaling
- Compares scaled vs unscaled weight distributions
- Grid search convergence analysis

**stats_error_parts.py** - Error Decomposition:
- Decomposes total quantization error into components
- Analyzes contribution of different error sources
- Per-channel error attribution

**export_data.py** - Data Export for External Analysis:
- Exports E[X[:,j]] (mean activation per channel)
- Exports W[:,k] (weight column for specific output channel)
- CSV format for external analysis tools
- Usage: `python export_data.py --layer-id 3 --out-channel-id 0`

**analyze_saliency_gradients.py** - Gradient-based Saliency:
- Computes importance via gradients
- Compares gradient-based vs activation-based importance

**analyze_saliency_tail.py** - Tail Distribution Analysis:
- Analyzes outlier channels in importance distributions
- Heavy-tail vs normal distribution analysis

### Automation

**run_awq_comparison.sh** - Complete Comparison Pipeline:
- Automated workflow from conversion to comparison
- Steps:
  1. Convert model to safetensors (if needed)
  2. Quantize with AutoAWQ library
  3. Quantize with custom implementation
  4. Run comparison and generate reports
- Configuration variables at top of script

**quantize_autoawq_library.py** - AutoAWQ Library Baseline:
- Uses official AutoAWQ implementation
- Requires safetensors format
- Standard symmetric quantization
- Output: `./quantized_models/minicpm_autoawq/`

## Commands

### Running Quantization

```bash
# Standard Group-Wise AWQ with L2 Salience
python gw_awq_asym_l2.py --n-calib 128 --n-grid 20 --group-size 128

# Heuristic-Guided Quantization
python awq_op_ref.py --n-calib 128 --n-grid 20 --group-size 128

# AutoAWQ Library Baseline (requires safetensors)
python quantize_autoawq_library.py --calib-samples 128 --w-bit 4 --q-group-size 128

# Custom group sizes
python gw_awq_asym_l2.py --group-size 64  # Faster, lower quality
python gw_awq_asym_l2.py --group-size 256 # Slower, higher quality
```

### Evaluation & Comparison

```bash
# Cross-dataset validation (Standard AWQ vs Heuristic AWQ)
python compare_awq_heuristic.py --n-samples 2000

# Final production validation
python final_cross_validation.py --n-samples 2000

# Quick comparison (fewer samples)
python compare_awq_heuristic.py --n-samples 500
```

### Analysis & Visualization

```bash
# Visualize channel importance distributions
python visualize_importances.py --layer-id 3 --n-calib 128

# Pre vs post activation analysis
python stats_pre_vs_post_act.py --layer-id 3 --n-calib 128

# Rounding error analysis
python stats_rounding_error.py --layer-id 3

# AWQ scaling effect analysis
python stats_scaling_awq.py --layer-id 3

# Export data for external analysis
python export_data.py --layer-id 3 --out-channel-id 0
```

### Automated Pipeline

```bash
# Run complete comparison pipeline
bash run_awq_comparison.sh

# Edit configuration in script:
# CALIB_SAMPLES, N_GRID, GROUP_SIZE, etc.
```

## Key Parameters

### Quantization Parameters

**Common across all methods:**
- `--n-calib`: Calibration samples (default: 128, range: 50-500)
- `--n-grid`: Grid search points for α (default: 20, range: 10-30)
- `--group-size`: Channels per quantization group (default: 128, options: 32, 64, 128, 256)
- `--bits`: Quantization bit width (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)

**Heuristic-specific (awq_op_ref.py):**
- `--outlier-percent`: Top X% activations to ignore (default: 0.05)
- `--use-heuristic`: Enable heuristic rounding (default: True)

**AutoAWQ library (quantize_autoawq_library.py):**
- `--w-bit`: Weight bit width (default: 4)
- `--q-group-size`: Group size (default: 128)
- `--zero-point`: Enable asymmetric quantization (flag)

### Evaluation Parameters

**Cross-dataset validation:**
- `--n-samples`: Samples per dataset (default: 2000)
- `--seed`: Random seed (default: 42)
- `--output-json`: Results file (default: ./comparison_results.json)

## Expected Outputs

### Quantized Models

```
./quantized_models/
├── minicpm_gw_awq_asym_l2/     # Standard GW-AWQ with L2 salience
│   ├── config.json
│   ├── pytorch_model.bin        # FP16 (dequantized for research)
│   └── tokenizer files
├── minicpm_awq_op_ref/          # Heuristic-guided quantization
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── minicpm_autoawq/             # AutoAWQ library baseline
    ├── config.json
    ├── model.safetensors        # True INT4 storage
    └── tokenizer files
```

### Visualizations

```
./visualizations/
├── importance_distributions/    # Channel importance plots
├── pre_post_activation/         # Activation analysis
├── rounding_error/              # Error statistics
└── scaling_analysis/            # AWQ scaling effects
```

### Data Exports

```
./data10.csv                     # Exported activation/weight data
./comparison_results.json        # Cross-dataset evaluation results
./rounding_stats.csv            # Per-layer rounding statistics
```

## Hardware Requirements

- **GPU:** CUDA-compatible, 16GB+ VRAM recommended
  - MiniCPM-2B in bfloat16: ~5GB
  - Calibration activations: ~3-5GB
  - Grid search: ~2-4GB peak
- **CPU fallback:** Supported but 10-20× slower
- **Storage:** ~15GB for models and cached datasets
  - Original model: ~5GB
  - Quantized models: ~5GB each (FP16 storage)
  - Datasets (cached): ~2GB

## Group Size Selection Guide

| Group Size | Hardware Efficiency | Quality | Memory Access | Use Case |
|-----------|-------------------|---------|---------------|----------|
| 32 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Optimal | Edge devices, maximum speed |
| 64 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Excellent | Balanced deployment |
| 128 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Good | **Recommended default** |
| 256 | ⭐⭐ | ⭐⭐⭐⭐⭐ | Fair | Quality-focused research |

**Recommendation:** Use `group_size=128` for best quality-efficiency trade-off.

## Common Issues & Solutions

### 1. Model Cache Compatibility Error

**Error:**
```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'
```

**Solution:** Always disable caching in forward passes:
```python
outputs = model(**inputs, use_cache=False)  # CRITICAL
```

### 2. Variable Sequence Length Concatenation

**Error:**
```
RuntimeError: Sizes of tensors must match except in dimension 0
```

**Solution:** Reshape before concatenating:
```python
reshaped = [x.reshape(-1, x.shape[-1]) for x in activation_list]
all_activations = torch.cat(reshaped, dim=0)
```

### 3. BFloat16 NumPy Incompatibility

**Error:**
```
TypeError: Got unsupported ScalarType BFloat16
```

**Solution:** Convert to float32 first:
```python
array = tensor.float().numpy()  # Not: tensor.numpy()
```

### 4. CUDA Out of Memory

**Solutions:**
```bash
# Reduce calibration samples
python gw_awq_asym_l2.py --n-calib 64

# Reduce grid search points
python gw_awq_asym_l2.py --n-grid 10

# Use CPU (slower)
CUDA_VISIBLE_DEVICES="" python gw_awq_asym_l2.py
```

### 5. Activation Hook Best Practices

**Correct pattern for capturing activations:**
```python
class ActivationCapture:
    def __init__(self):
        self.activations = []

    def hook_fn(self, module, input, output):
        # Always detach and move to CPU immediately
        act = output.detach().cpu()
        self.activations.append(act)

    def get_concatenated(self):
        # Handle variable sequence lengths
        reshaped = [x.reshape(-1, x.shape[-1]) for x in self.activations]
        # Convert to float32 for numpy compatibility
        return torch.cat(reshaped, dim=0).float()
```

**For AWQ quantization:** Capture INPUT to linear layers:
```python
def awq_hook(module, input, output):
    X = input[0].detach().cpu()  # INPUT, not output
    self.inputs.append(X)
```

### 6. Model Loading Template

**Standard pattern for MiniCPM-2B:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM-2B-sft-bf16",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-2B-sft-bf16",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load calibration data
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Forward pass
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

        # CRITICAL: use_cache=False
        outputs = model(**inputs, use_cache=False)
```

## Research Context

### Key Insights

**L2 Salience Advantage:**
- Quantization MSE = `E[(δW × X)²]` ∝ `E[X²]`
- L2 (`E[X²]`) emphasizes channels with spikes/outliers (quadratic weighting)
- L1 (`E[|X|]`) treats all channels linearly
- L2 directly matches the MSE objective

**Asymmetric Quantization Benefits:**
- Symmetric [-7,7]: Wastes range for non-negative activations
- Asymmetric [0,15]: Uses full range, adds zero-point parameter
- Group-wise zero-points enable per-group range adaptation

**Heuristic Rounding Correction:**
- Standard rounding: Minimize per-weight error independently
- Heuristic (global greedy): Minimize total dot product error `x·w`
- Outlier masking prevents unstable flips
- Empirically reduces error by ~25% (see heuristic_verification.py)

### Validation Methodology

**Cross-dataset approach:**
1. WikiText-2: In-distribution validation (model calibrated on WikiText-2 train)
2. C4: Cross-dataset (web crawl, diverse domains)
3. AG News: Cross-dataset (news, different style)

**Why multiple datasets:**
- Prevents overfitting to calibration distribution
- Tests robustness across domains
- More realistic deployment scenario

## Dependencies

```bash
pip install torch transformers datasets
pip install matplotlib seaborn numpy pandas scipy tqdm
pip install autoawq  # Optional: for AutoAWQ baseline comparison
```

**Versions tested:**
- Python: 3.8+
- PyTorch: 2.0+
- Transformers: 4.30+
- CUDA: 11.8+ (if using GPU)

## File Naming Conventions

- `gw_*.py`: Group-wise quantization implementations
- `stats_*.py`: Statistical analysis scripts
- `analyze_*.py`: Deep analysis tools
- `visualize_*.py`: Visualization scripts
- `compare_*.py`: Comparison/evaluation scripts
- `quantize_*.py`: Main quantization pipelines

## Output File Conventions

- Quantized models: `./quantized_models/minicpm_{method}/`
- Visualizations: `./visualizations/{analysis_type}/`
- Data exports: `./{name}.csv` or `./{name}.json` in root directory
