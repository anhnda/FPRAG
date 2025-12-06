# Fast-R-PRAQ vs AWQ: LLM Quantization Research

This is a research project comparing **Fast-R-PRAQ** (Fast Risk-aware Post-activation Quantization) against **AWQ** (Activation-aware Weight Quantization) for compressing Large Language Models to 4-bit precision while preserving model quality.

## Project Overview

This repository implements and evaluates two quantization strategies for the MiniCPM-2B model:

### 1. FullAWQ (Activation-aware Weight Quantization)

**Key Idea:** Protect important channels using pre-activation magnitude

- **Importance Metric:** `s[j] = E[|X[:, j]|]` (average input magnitude per channel)
- **Algorithm:**
  1. Measure pre-activation magnitude for each input channel
  2. Grid search for optimal scaling exponent α ∈ [0, 1]
  3. Scale weight columns: `W[:, j] *= s[j]^α`
  4. Quantize ALL weights to uniform INT4
  5. At inference: compensate with inverse scales on activations

- **Philosophy:** Channels with large input magnitudes are important and should be protected during quantization

### 2. FullPRAQ (Post-activation Risk-aware Quantization)

**Key Idea:** Account for "risky dead neurons" using post-activation importance

- **Importance Metric (MLP layers):**
  1. Compute post-activation output: `Y = SiLU(X @ W^T + b)`
  2. Measure output importance: `importance_out[k] = E[|Y[:, k]|]`
  3. Backpropagate to input channels: `importance_in[j] = Σ_k(importance_out[k] × |W[k,j]|)`

- **Algorithm:**
  1. For MLP layers: Use post-activation importance (accounts for activation function)
  2. For Attention layers: Use AWQ-style pre-activation importance
  3. Grid search for optimal scaling exponent α ∈ [0, 1]
  4. Scale weight columns: `W[:, j] *= s[j]^α`
  5. Quantize ALL weights to uniform INT4
  6. At inference: compensate with inverse scales

- **Philosophy:** A channel with large negative pre-activation may appear inactive but can be "resurrected" by quantization noise. PRAQ measures actual post-activation output to identify truly important channels.

### The "Risky Dead Neuron" Problem

Standard quantization methods (like AWQ) can fail when encountering channels with:
- Large negative pre-activation mean (e.g., mean = -10)
- High variance (e.g., std = 4)
- Large weight magnitudes (e.g., |W| = 5)

**AWQ sees:** Channel has mean = -10 → "Dead neuron, safe to quantize aggressively"

**PRAQ sees:**
- Quantization noise ≈ weight_mag × noise_factor ≈ 5 × 0.2 = 1.0
- Risk-adjusted upper bound: -10 + 3×4 + 1.0 = 3.0 (above SiLU threshold!)
- Conclusion: "Risky channel, must preserve precision"

## Key Distinction: Simulation vs Hardware

### ⚠️ IMPORTANT: This is a simulation study focusing on ACCURACY ONLY

**What this project DOES:**
- ✅ Simulate INT4 quantization effects on model accuracy
- ✅ Compare perplexity degradation between FullAWQ and FullPRAQ
- ✅ Evaluate which importance metric better preserves model quality
- ✅ Implement proper per-channel scaling algorithms
- ✅ Use uniform INT4 quantization (all channels same bit-width)

**What this project DOES NOT do:**
- ❌ Real hardware acceleration (weights stored as FP16, not packed INT4)
- ❌ Optimized inference kernels
- ❌ Memory compression at runtime
- ❌ Actual INT4 arithmetic operations
- ❌ CUDA/GPU kernel optimization

**Storage Format:** While the quantization simulates INT4 precision (values rounded to INT4 range), weights are still stored as FP16 tensors for compatibility with PyTorch. A real deployment would pack these into 4-bit integers for 4× memory savings.

**Expected Compression:** In a real hardware implementation, you would achieve ~4× memory reduction (16-bit → 4-bit) and potential inference speedup. This project measures **quality retention** only.

## Project Structure

### Core Quantization Scripts

- **`quantize_minicpm_full_awq.py`** - FullAWQ implementation (pre-activation importance)
- **`quantize_minicpm_full_praq.py`** - FullPRAQ implementation (post-activation importance)
- **`compare_full_quantization.py`** - Evaluation framework comparing both methods

### Analysis & Visualization

- **`visualize_preactivations.py`** - Multi-layer distribution analysis
- **`visualize_layer_focused.py`** - Single-layer deep dive with AWQ vs PRAQ comparison
- **`check_mse_layer.py`** - Layer-level MSE comparison between methods
- **`awq_vs_fprpa.py`** - Synthetic benchmark demonstrating the "loud silence" problem

### Utilities

- **`convert_to_safetensors.py`** - Convert model to safetensors format (for AutoAWQ library)
- **`quantize_minicpm_awq.py`** - AWQ baseline using AutoAWQ library
- **`quantize_minicpm_AWQ.py`** - Custom AWQ implementation (library-free)
- **`quantize_minicpm_PRAQ.py`** - Hybrid PRAQ (mixed-precision variant)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended, 16GB+ VRAM for MiniCPM-2B)
- PyTorch with CUDA support

### Dependencies

```bash
pip install torch transformers datasets
pip install matplotlib seaborn numpy pandas scipy tqdm
pip install autoawq  # Optional: for AWQ baseline
```

## Quick Start

### Step 1: Run Quantization

Quantize the model using both methods (run these in sequence or parallel):

```bash
# Quantize with FullAWQ (pre-activation importance)
python quantize_minicpm_full_awq.py

# Quantize with FullPRAQ (post-activation importance)
python quantize_minicpm_full_praq.py
```

**Optional parameters:**
```bash
# Custom calibration samples and grid search points
python quantize_minicpm_full_awq.py --n-calib 500 --n-grid 20

# Custom output directory
python quantize_minicpm_full_praq.py --output-dir ./my_models/praq
```

**What happens:**
- Downloads MiniCPM-2B model (if not cached)
- Loads 500 WikiText-2 training samples for calibration
- Collects activation statistics by running forward passes
- Grid search for optimal scaling exponent α (21 points: 0.0, 0.05, ..., 1.0)
- Quantizes all linear layers to simulated INT4
- Saves quantized model to `./quantized_models/`

**Expected time:** 30-60 minutes per method on a modern GPU

### Step 2: Run Comparison

After both models are quantized, evaluate them:

```bash
# Compare Original FP16, FullAWQ, and FullPRAQ
python compare_full_quantization.py

# With visualization
python compare_full_quantization.py --visualize

# Custom evaluation samples
python compare_full_quantization.py --n-eval 2000
```

**What happens:**
- Loads Original FP16 (baseline)
- Loads FullAWQ quantized model
- Loads FullPRAQ quantized model
- Evaluates perplexity on WikiText-2 validation set (2000 samples)
- Generates comparison table and analysis
- (Optional) Creates visualization plots

**Expected time:** 15-30 minutes on a modern GPU

## Understanding the Results

### Comparison Table

```
Metric                         Original             Full-AWQ             Full-PRAQ
------------------------------------------------------------------------------------
Perplexity                        12.34                13.21                12.98
Avg Loss                          2.51                 2.58                 2.56
Model Size (MB)                 4824.32              4824.32              4824.32
Throughput (tok/s)               1234.56              1245.67              1238.90
```

**Note:** Model size appears the same because weights are stored as FP16 for simulation. Real INT4 deployment would show ~4× size reduction.

### Analysis Metrics

- **Perplexity:** Lower is better (measures prediction quality)
- **Delta (Δ):** Perplexity increase from FP16 baseline
  - Δ < 5%: ✅ Excellent quality retention
  - Δ < 10%: 🟢 Good quality retention
  - Δ < 20%: 🟡 Acceptable quality retention
  - Δ ≥ 20%: ❌ Poor quality retention

### Key Questions Answered

1. **Does FullPRAQ outperform FullAWQ?**
   - If yes: Post-activation importance is more accurate
   - If no: Pre-activation magnitude is sufficient

2. **How much quality is lost with uniform INT4?**
   - Typical range: 5-15% perplexity increase
   - Acceptable for many deployment scenarios

3. **Are "risky dead neurons" a real problem?**
   - Check `visualize_layer_focused.py` output for empirical evidence
   - Look for channels with negative mean but high risk scores

## Advanced Usage

### Analyze Specific Layers

```bash
# Deep dive into layer 28 (modify target_layer_id in main())
python visualize_layer_focused.py

# Compare AWQ vs FastRPRAQ importance scores
# Output: Spearman correlation, per-channel statistics, visualizations
```

### Multi-layer Analysis

```bash
# Analyze pre-activation distributions across all layers
python visualize_preactivations.py

# Identifies "critical channels" (dead + risky + high weight)
# Output: Per-layer visualizations in ./visualizations/preactivation_analysis/
```

### Layer-level MSE Comparison

```bash
# Direct MSE comparison on layer 28
python check_mse_layer.py

# Shows which importance metric produces lower reconstruction error
```

### Synthetic Benchmark

```bash
# Demonstrate the "loud silence" problem on synthetic data
python awq_vs_fprpa.py

# Shows how AWQ accidentally handles Group B (dangerous channels)
# Shows how FastRPRAQ correctly identifies both utility and risk
```

## Configuration Parameters

### Quantization Parameters

Located in `quantize_minicpm_full_praq.py` and `quantize_minicpm_full_awq.py`:

```python
--n-calib 500      # Calibration samples (more = better but slower)
--n-grid 20        # Grid search points (more = finer but slower)
--bits 4           # Target bit width (default: 4)
--seed 42          # Random seed for reproducibility
```

### PRAQ-specific Parameters

Located in `quantize_minicpm_full_praq.py`:

```python
--beta 3.0           # Temperature for probability calculation
--tau -3.0           # Activation threshold for SiLU
--noise-factor 0.2   # Estimated INT4 quantization noise ratio (20%)
```

### Evaluation Parameters

Located in `compare_full_quantization.py`:

```python
--n-eval 2000                                    # Evaluation samples
--original-model openbmb/MiniCPM-2B-sft-bf16    # Baseline model
--full-awq-path ./quantized_models/minicpm_full_awq
--full-praq-path ./quantized_models/minicpm_full_praq
--visualize                                      # Generate plots
```

## Hardware Requirements

- **GPU:** CUDA-compatible, 16GB+ VRAM recommended
  - MiniCPM-2B in FP16 requires ~5GB
  - Calibration activations require additional memory
- **CPU fallback:** Supported but significantly slower
- **Storage:** ~10GB for model weights and cached datasets

## Expected Outputs

### Quantized Models
```
./quantized_models/
├── minicpm_full_awq/      # FullAWQ quantized model
│   ├── config.json
│   ├── model.safetensors  # Weights (FP16 storage, INT4 precision)
│   └── tokenizer files
└── minicpm_full_praq/     # FullPRAQ quantized model
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

### Visualizations
```
./visualizations/
├── full_quantization/
│   └── full_quantization_comparison.png  # Perplexity comparison
├── preactivation_analysis/
│   ├── layer_0_analysis.png
│   ├── layer_1_analysis.png
│   └── ...
└── layer_X_focused/
    ├── channel_statistics.csv
    ├── correlation_awq_praq.png
    └── negative_channels_detailed.png
```

## Research Context

### Hypothesis

Standard quantization methods (like AWQ) fail to account for the risk of quantization noise "resurrecting" dead neurons. Channels with:
- Large negative pre-activation (appear "dead")
- High variance (risky, can cross activation threshold)
- Large weights (quantization noise proportional to weight magnitude)

...are dangerous and should be preserved in higher precision.

### Key Insight

**Quantization noise is proportional to weight magnitude:**
```
noise ≈ input_magnitude × weight_magnitude × noise_factor
```

So a channel with `mean = -10, std = 4, |W| = 5` has:
```
risk_upper = mean + 3×std + noise
           = -10 + 12 + 1.0
           = 3.0  (above SiLU threshold of 0!)
```

AWQ would see this as "dead" (mean = -10). PRAQ identifies it as "risky" (can activate post-quantization).

### Validation

Use the analysis tools to empirically test:
1. Do such "risky dead neurons" exist in MiniCPM-2B? (check `visualize_preactivations.py`)
2. Does PRAQ's post-activation importance correlate with reconstruction error? (check `check_mse_layer.py`)
3. Does PRAQ outperform AWQ in perplexity? (check `compare_full_quantization.py`)

## Comparison to Other Methods

### FullAWQ vs Mixed-Precision AWQ

- **FullAWQ:** ALL weights in INT4, protected via per-channel scaling
- **Mixed-Precision AWQ:** Top-k channels in FP16, rest in INT4
- **Tradeoff:** Full uniform quantization is simpler for hardware but may sacrifice quality

### FullPRAQ vs Hybrid PRAQ

- **FullPRAQ:** Uniform INT4 with post-activation importance
- **Hybrid PRAQ:** Mixed-precision with risk-aware channel selection
- **This project:** Focuses on FullPRAQ for fair comparison with FullAWQ

## Citation

If you use this research, please cite:

```bibtex
@misc{fpraq2024,
  title={Fast Risk-aware Post-activation Quantization for LLM Compression},
  author={Your Name},
  year={2024},
  note={Research implementation comparing PRAQ vs AWQ}
}
```

## License

[Specify your license here]

## Contributing

Contributions welcome! Areas for improvement:
- Real INT4 kernel implementation for hardware acceleration
- Support for more model architectures (LLaMA, GPT, etc.)
- Additional quantization bit-widths (2-bit, 3-bit)
- Group quantization variants
- Calibration dataset experiments

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce calibration samples
python quantize_minicpm_full_awq.py --n-calib 250

# Use CPU (slower)
# Set device="cpu" in the scripts
```

### Model Not Found

```bash
# For AutoAWQ baseline, convert to safetensors first:
python convert_to_safetensors.py
```

### Perplexity is Infinity

- Check for NaN values in weights after quantization
- Reduce grid search range or use smaller α values
- Verify calibration data is not empty

## Contact

For questions or issues, please open a GitHub issue.

---

**Disclaimer:** This is a research project for evaluating quantization quality. For production deployment, use optimized libraries like AutoAWQ, GPTQ, or llama.cpp with proper INT4 kernel implementations.
