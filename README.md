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
python gw_awq_asym_l2.py

# Quantize with FullPRAQ (post-activation importance)
python awq_op_ref.py
```

**Optional parameters:**
```bash
# Custom calibration samples and grid search points
python compare_awq_heuristics.py

# Custom output directory
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

### Step 3 (Optional): Robust-PRAQ with Noise Augmentation

For improved robustness with limited calibration data:

```bash
# Quantize with Robust-PRAQ (noise-augmented importance)
python quantize_minicpm_robust_praq.py

# With custom noise parameters
python quantize_minicpm_robust_praq.py --noise-std 0.01 --n-noise-samples 3

# Compare Robust-PRAQ vs Full-PRAQ
python compare_robust_praq.py --visualize
```

**Noise Augmentation Benefits:**
- More stable importance estimates with small calibration sets
- Reduces overfitting to specific calibration samples
- Better generalization to unseen data
- Particularly useful when calibration budget is limited (< 500 samples)

**What happens:**
- Same as FullPRAQ, but with noise augmentation:
  - Adds Gaussian noise to inputs during importance computation
  - Averages over multiple noise samples (default: 3)
  - Evaluates reconstruction error with noisy inputs
- Saves to `./quantized_models/minicpm_robust_praq`

**Expected time:** 40-70 minutes per method (slightly slower due to noise sampling)

**When to use Robust-PRAQ:**
- ✅ Limited calibration data (< 500 samples)
- ✅ Calibration set may not be representative
- ✅ Need maximum robustness
- ❌ Large calibration sets (> 1000 samples) - FullPRAQ is sufficient

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

### Robust-PRAQ Parameters

Located in `quantize_minicpm_robust_praq.py` (includes all PRAQ parameters plus):

```python
--noise-std 0.01          # Gaussian noise std (relative to input std)
                          # Typical range: 0.005 - 0.05
                          # Higher = more regularization, lower = closer to FullPRAQ

--n-noise-samples 3       # Number of noise samples to average
                          # Typical range: 2 - 5
                          # Higher = more stable but slower
```

**Noise Parameter Guidelines:**

| Calibration Size | Recommended noise_std | Recommended n_noise_samples |
|------------------|----------------------|----------------------------|
| < 200 samples    | 0.02 - 0.03          | 4 - 5                      |
| 200-500 samples  | 0.01 - 0.02          | 3 - 4                      |
| 500-1000 samples | 0.005 - 0.01         | 2 - 3                      |
| > 1000 samples   | Use FullPRAQ         | N/A                        |

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
├── minicpm_full_praq/     # FullPRAQ quantized model
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
└── minicpm_robust_praq/   # Robust-PRAQ quantized model
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

### Visualizations
```
./visualizations/
├── full_quantization/
│   └── full_quantization_comparison.png  # AWQ vs PRAQ comparison
├── robust_praq/
│   └── robust_praq_comparison.png        # Robust-PRAQ vs PRAQ comparison
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

### Robust-PRAQ vs FullPRAQ

- **FullPRAQ:** Standard post-activation importance computation
- **Robust-PRAQ:** Noise-augmented importance with multi-sample averaging
- **Key Difference:** Robust-PRAQ adds Gaussian noise during importance estimation
- **Tradeoff:** Robust-PRAQ is slightly slower but more stable with limited calibration data
- **Use Case:** Robust-PRAQ is particularly beneficial when:
  - Calibration budget is tight (< 500 samples)
  - Calibration set may not be representative of deployment distribution
  - Maximum robustness is required

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
