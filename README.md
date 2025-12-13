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

Quantize the model using both methods (batched sequential quantization for optimal memory/speed):

```bash
# Standard Group-Wise AWQ with L2 Salience (default: C4 calibration, 50 layers/batch)
python gw_awq_asym_l2.py --n-calib 128 --layer-batch-size 50

# Heuristic AWQ with Global Greedy Rounding (adds heuristic correction on top)
python awq_sh.py --n-calib 128 --layer-batch-size 50
```

**What happens:**
- Downloads MiniCPM-2B model (if not cached)
- Loads 128 C4 samples for calibration (default dataset)
- **Batched sequential processing:** Processes 50 layers at a time (~14GB memory)
- Collects activation statistics via forward passes for each batch
- Grid search for optimal scaling exponent α (20 points: 0.0, 0.05, ..., 1.0)
- Quantizes all 281 linear layers to INT4 (asymmetric, group-wise)
- Saves quantized model to `./quantized_models/`

**Memory usage:** ~14GB peak (configurable via `--layer-batch-size`)
**Expected time:** 6-12 minutes per method on modern GPU

**Optional parameters:**
```bash
# Fast iteration (lower memory, WikiText-2 Simple)
python gw_awq_asym_l2.py --calib-dataset wikitext2-simple --n-calib 128 --layer-batch-size 50

# Limited memory (8GB system)
python gw_awq_asym_l2.py --calib-dataset wikitext2-simple --n-calib 64 --layer-batch-size 20

# Maximum quality (30GB+ system)
python gw_awq_asym_l2.py --n-calib 256 --layer-batch-size 100
```

### Step 2: Run Comparison

After both models are quantized, evaluate them across multiple datasets:

```bash
# Compare Standard AWQ vs Heuristic AWQ
python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/minicpm_awq_sh \
    --standard-path ./quantized_models/minicpm_gw_awq_asym_l2 \
    --n-samples 2000
```

**What happens:**
- Loads both quantized models
- Evaluates perplexity on 3 datasets:
  - WikiText-2 **test** (in-distribution, Wikipedia)
  - C4 validation (cross-dataset, web crawl)
  - AG News test (cross-dataset, news)
- Generates comparison table with statistical analysis
- Determines winner based on cross-dataset performance
- Saves results to JSON with detailed metrics

**Expected time:** 10-20 minutes on modern GPU

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

The comparison script evaluates both models on 3 datasets:

```
Dataset       Heuristic AWQ    Standard AWQ     Delta       Winner
------------------------------------------------------------------------
WikiText-2         13.45            13.52       -0.5%      Heuristic
C4                 14.21            14.18       +0.2%      Standard
AG News            15.67            15.89       -1.4%      Heuristic
------------------------------------------------------------------------
Average            14.44            14.53       -0.6%      Heuristic
```

### Analysis Metrics

- **Perplexity:** Lower is better (measures prediction quality)
- **Delta:** Percentage difference between methods
  - Negative (−): Heuristic AWQ is better
  - Positive (+): Standard AWQ is better
  - Close to 0: Methods are equivalent
- **Winner:** Determined by cross-dataset performance

### What the Results Tell You

**1. Does Heuristic Rounding Improve Quality?**
- If Heuristic wins on 2+ datasets: Global greedy rounding is effective
- If Standard wins: Simple nearest rounding is sufficient
- If tied: Heuristic adds complexity without clear benefit

**2. Cross-Dataset Robustness**
- **WikiText-2 test:** In-distribution performance
- **C4:** Generalization to web content
- **AG News:** Generalization to news domain

**3. Quality vs Complexity Trade-off**
- Standard AWQ: O(n), simple and fast
- Heuristic AWQ: O(n²), slower but may be more accurate

### Key Questions Answered

1. **Is the heuristic worth the complexity?**
   - Check the average delta and win count
   - If improvement < 1%: Standard AWQ is sufficient
   - If improvement > 2%: Heuristic AWQ is beneficial

2. **How much quality is lost with INT4?**
   - Compare against FP16 baseline (typically ~5-15% increase)
   - Both methods should be within acceptable range

3. **Does the heuristic generalize?**
   - Check if it wins consistently across all 3 datasets
   - Or only on specific domains (e.g., WikiText-2)

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

**Common parameters (gw_awq_asym_l2.py, awq_sh.py):**

```bash
--n-calib 128              # Calibration samples (default: 128, range: 50-500)
--n-grid 20                # Grid search points for α (default: 20)
--group-size 128           # Quantization group size (default: 128)
--bits 4                   # Target bit width (default: 4)
--seed 42                  # Random seed for reproducibility
--calib-dataset c4         # Calibration dataset (default: c4)
                          # Choices: c4, wikitext2, wikitext2-simple
--layer-batch-size 50      # Layers per batch (default: 50, ~14GB memory)
```

**Memory formula:** `batch_size × 280 MB`
- Example: 50 layers = ~14 GB peak memory

**Calibration dataset options:**
- `c4`: High quality, cross-dataset robustness (~10GB) **[DEFAULT]**
- `wikitext2-simple`: Fast, memory-efficient (~6GB)
- `wikitext2`: Balanced, chunked sequences

### Heuristic-specific Parameters (awq_sh.py)

```bash
--use-heuristic           # Enable heuristic rounding (default: True)
--no-heuristic            # Disable heuristic (standard AWQ mode)
--outlier-percent 0.05    # Top X% activations to ignore (default: 0.05)
--max-tokens-per-sample 512  # Token subsampling for memory
```

### Evaluation Parameters (compare_awq_heuristic.py)

```bash
--heuristic-path ./quantized_models/minicpm_awq_sh
--standard-path ./quantized_models/minicpm_gw_awq_asym_l2
--n-samples 2000          # Samples per dataset (default: 2000)
--seed 42                 # Random seed
--save-results            # Save detailed results to JSON
```

## Hardware Requirements

### Batched Sequential Quantization (Recommended)

- **GPU:** CUDA-compatible (configurable memory usage)
  - **16GB+ VRAM:** Default settings (50 layers/batch, C4)
  - **8-16GB VRAM:** Use `--layer-batch-size 20 --calib-dataset wikitext2-simple`
  - **8GB VRAM:** Use `--layer-batch-size 10 --calib-dataset wikitext2-simple --n-calib 64`

**Memory breakdown:**
- MiniCPM-2B model: ~5GB
- Calibration activations: `batch_size × 280MB` (e.g., 50 layers = 14GB)
- Total peak: ~19GB (with default settings)

**Memory optimization:**
1. Use `--calib-dataset wikitext2-simple` (saves ~4GB vs C4)
2. Reduce `--layer-batch-size` (most effective control)
3. Reduce `--n-calib` samples
4. Reduce `--n-grid` points

- **CPU fallback:** Supported but 10-20× slower
- **Storage:** ~15GB for models and cached datasets

## Expected Outputs

### Quantized Models
```
./quantized_models/
├── minicpm_gw_awq_asym_l2/    # Standard Group-Wise AWQ with L2 Salience
│   ├── config.json
│   ├── pytorch_model.bin       # Weights (FP16 storage for research)
│   └── tokenizer files
└── minicpm_awq_sh/             # Heuristic AWQ with Global Greedy Rounding
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

### Evaluation Results
```
./results/
├── awq_heuristic_validation_YYYYMMDD_HHMMSS.json  # Detailed results
└── AWQ_HEURISTIC_SUMMARY.md                        # Human-readable summary
```

### Visualizations
```
./visualizations/
├── importance_distributions/      # Channel importance analysis
├── pre_post_activation/          # Activation analysis
├── rounding_error/               # Quantization error statistics
└── scaling_analysis/             # AWQ scaling effects
```

## Research Context

### Project Focus

This project compares quantization methods for LLM compression, specifically:

1. **Standard Group-Wise AWQ** (gw_awq_asym_l2.py)
   - Uses L2 salience (E[X²]) for activation-aware scaling
   - Asymmetric INT4 quantization [0,15] per group
   - Standard nearest rounding

2. **Heuristic-Guided AWQ** (awq_sh.py)
   - Same base as Standard AWQ
   - Adds global greedy rounding correction using E[Xs] statistics
   - Outlier masking for stability

### Core Research Questions

1. **L2 vs L1 Salience:** Does E[X²] improve upon E[|X|] for identifying important channels?
2. **Heuristic Rounding:** Does global greedy rounding reduce quantization error?
3. **Asymmetric vs Symmetric:** Does asymmetric [0,15] outperform symmetric [-7,7]?
4. **Group-wise Quantization:** What's the optimal group size for hardware efficiency vs quality?

### Key Innovation: Batched Sequential Quantization

**Problem:** Processing all 281 layers simultaneously requires ~75GB memory

**Solution:** Process layers in batches (default: 50 at a time)
- Memory: ~14GB per batch (vs 75GB for all layers)
- Speed: 6 calibration runs (vs 281 for pure sequential)
- Error propagation aware: Layer N+1 sees quantized outputs from layer N

### Validation Methodology

**Cross-dataset evaluation** on 3 datasets:
1. **WikiText-2 test:** In-distribution (Wikipedia, formal)
2. **C4 validation:** Cross-dataset (web crawl, diverse)
3. **AG News test:** Cross-dataset (news, journalistic)

This prevents overfitting to calibration distribution and tests robustness across domains.

## Comparison to Other Methods

### Standard AWQ vs Heuristic AWQ

**Standard AWQ (gw_awq_asym_l2.py):**
- Salience: E[X²] for activation-aware scaling
- Quantization: Min/max asymmetric per group
- Rounding: Nearest (standard approach)
- Complexity: O(n) - simple and fast
- Speed: 6-12 minutes on modern GPU

**Heuristic AWQ (awq_sh.py):**
- Salience: E[X²] for activation-aware scaling (same as Standard)
- Quantization: E[Xs]-guided greedy refinement
- Rounding: Global greedy with outlier masking
- Complexity: O(n²) - iterative flip selection
- Speed: Similar (6-12 minutes) - heuristic overhead is minimal
- Innovation: Minimizes output error dot(Xs, W-W_quant)

### Batched Sequential vs Batch Quantization

**Old Batch Approach:**
- Register hooks on ALL 281 layers
- Run calibration → Store activations for ALL layers (~75GB)
- Quantize each layer
- Result: OOM on most systems

**Batched Sequential (Current):**
- Process 50 layers at a time
- Memory: ~14GB per batch (constant)
- Speed: 6 calibration runs (fast enough)
- Error propagation aware (more realistic)
- Result: Practical on consumer hardware

### This Implementation vs AutoAWQ

**AutoAWQ (official library):**
- Production-optimized, fast kernels
- Symmetric quantization
- Requires safetensors format
- True INT4 storage

**This Implementation (research):**
- Asymmetric quantization [0,15]
- Batched sequential for memory efficiency
- FP16 storage for easy analysis
- Heuristic rounding experimentation
- Focus: Quality comparison and research

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

**Solution 1: Use WikiText-2 Simple (instead of default C4)**
```bash
python gw_awq_asym_l2.py --calib-dataset wikitext2-simple --n-calib 128
```
Saves ~4GB by using variable-length sequences.

**Solution 2: Reduce layer batch size**
```bash
# 6GB instead of 14GB
python gw_awq_asym_l2.py --layer-batch-size 20

# 3GB for very limited memory
python gw_awq_asym_l2.py --layer-batch-size 10
```

**Solution 3: Reduce calibration samples**
```bash
python gw_awq_asym_l2.py --n-calib 64
```

**Solution 4: Combine all optimizations**
```bash
python gw_awq_asym_l2.py \
    --calib-dataset wikitext2-simple \
    --n-calib 64 \
    --layer-batch-size 10 \
    --n-grid 10
```
Uses only ~3GB memory.

**Solution 5: Use CPU (10-20× slower)**
```bash
CUDA_VISIBLE_DEVICES="" python gw_awq_asym_l2.py
```

### All α Values are 0.0 (Sequential Quantization Bug)

**Symptom:** Grid search shows α=0.0 for all layers, "no activations captured" warnings

**Cause:** Dtype mismatch after quantizing previous layers (weights became float32 but model is float16)

**Solution:** This is fixed in the current version. The code now preserves original dtype:
```python
original_dtype = W.dtype
W_final = (W_quant / scales).to(original_dtype)
```

If you still encounter this, ensure you're using the latest version of the scripts.

### Model Not Found

**For AutoAWQ baseline:**
```bash
python convert_to_safetensors.py  # If needed
python quantize_autoawq_library.py
```

### Perplexity is Infinity

- Check for NaN values in weights after quantization
- Reduce grid search range or use smaller α values
- Verify calibration data is not empty
- Ensure `use_cache=False` in all forward passes

## Contact

For questions or issues, please open a GitHub issue.

---

**Disclaimer:** This is a research project for evaluating quantization quality. For production deployment, use optimized libraries like AutoAWQ, GPTQ, or llama.cpp with proper INT4 kernel implementations.
