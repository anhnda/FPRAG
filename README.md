# AWQ Quantization: Standard vs Heuristic Comparison

Research project comparing quantization methods for LLM compression on MiniCPM-2B (2.7B parameters).

**Key Innovation:** Batched sequential quantization - processes layers in batches for optimal memory/speed balance.

## Results Summary

Cross-dataset validation shows **Heuristic AWQ consistently outperforms Standard AWQ**:

```
Dataset         Heuristic AWQ   Standard AWQ    Delta        Winner
--------------------------------------------------------------------------------
WikiText-2      15.0546         15.2585         -1.336%      Heuristic
C4              13.3848         13.5698         -1.364%      Heuristic
AG News         25.4520         26.1080         -2.512%      Heuristic
--------------------------------------------------------------------------------
Average         17.9638         18.3121         -1.737%      Heuristic ✅
```

**Conclusion:** Global greedy rounding correction provides measurable quality improvement across all tested datasets.

## Quick Start

### Prerequisites

```bash
pip install torch transformers datasets
pip install matplotlib seaborn numpy pandas scipy tqdm
```

**Hardware:** 16GB+ GPU VRAM recommended (configurable down to 8GB)

### Step 1: Run Quantization

Run both quantization methods:

```bash
# Standard Group-Wise AWQ with L2 Salience
python gw_awq_asym_l2.py --n-calib 128 --layer-batch-size 50

# Heuristic AWQ with Global Greedy Rounding
python awq_sh.py --n-calib 128 --layer-batch-size 50
```

**What happens:**
- Downloads MiniCPM-2B model (~5GB)
- Loads 128 C4 samples for calibration
- Processes 50 layers at a time (~14GB peak memory)
- Quantizes all 281 layers to INT4 asymmetric [0,15]
- Saves to `./quantized_models/`

**Time:** ~6-12 minutes per method on modern GPU
**Memory:** ~14GB peak (default settings)

### Step 2: Run Comparison

Evaluate both models across 3 datasets:

```bash
python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/minicpm_awq_sh \
    --standard-path ./quantized_models/minicpm_gw_awq_asym_l2 \
    --n-samples 2000
```

**Datasets evaluated:**
- WikiText-2 test (Wikipedia, formal)
- C4 validation (web crawl, diverse)
- AG News test (news, journalistic)

**Time:** ~10-20 minutes
**Output:** Comparison table + JSON results in `./results/`

## Method Comparison

### Standard AWQ (gw_awq_asym_l2.py)

**Approach:**
1. Compute L2 salience: `s[j] = E[X[:,j]²]`
2. Grid search for optimal scaling α
3. Scale weights: `W[:,j] *= s[j]^α`
4. Group-wise asymmetric INT4 quantization [0,15]
5. **Standard nearest rounding** (round to nearest quantized value)

**Characteristics:**
- Simple, fast, standard approach
- O(n) complexity
- Perplexity: 15.26 (WikiText-2), 13.57 (C4), 26.11 (AG News)

### Heuristic AWQ (awq_sh.py)

**Approach:**
1. Same as Standard AWQ (steps 1-4)
2. **Global greedy rounding correction:**
   - Compute E[Xs] statistics
   - Identify flip candidates (exclude outliers)
   - Sort globally by rounding cost
   - Find optimal K flips to minimize output error

**Characteristics:**
- Advanced rounding with outlier masking
- O(n²) complexity (still fast in practice)
- Perplexity: 15.05 (WikiText-2), 13.38 (C4), 25.45 (AG News)
- **1-2.5% improvement** over Standard AWQ

## Batched Sequential Quantization

**Problem:** Processing all 281 layers simultaneously requires ~75GB memory → OOM

**Solution:** Process layers in batches

```
For each batch of 50 layers:
  1. Register hooks on these 50 layers
  2. Run calibration → Store activations (~14GB)
  3. Quantize all 50 layers
  4. Clear activations, move to next batch
```

**Benefits:**
- **Memory:** 14GB vs 75GB (5.4× reduction)
- **Speed:** 6 calibration runs vs 281 (47× faster than pure sequential)
- **Quality:** Error propagation aware (layer N+1 sees quantized outputs from N)

## Configuration

### Memory Configurations

| Available RAM | Command | Memory Usage | Time |
|--------------|---------|--------------|------|
| **16GB+ (recommended)** | `--layer-batch-size 50` | ~14GB | 6-12 min |
| **8-16GB** | `--layer-batch-size 20 --calib-dataset wikitext2-simple` | ~6GB | 10-15 min |
| **8GB** | `--layer-batch-size 10 --calib-dataset wikitext2-simple --n-calib 64` | ~3GB | 15-20 min |

### Key Parameters

```bash
--n-calib 128              # Calibration samples (default: 128)
--n-grid 20                # Grid search points for α (default: 20)
--group-size 128           # Quantization group size (default: 128)
--layer-batch-size 50      # Layers per batch (default: 50, ~14GB)
--calib-dataset c4         # Dataset: c4 (default), wikitext2-simple, wikitext2
--seed 42                  # Random seed for reproducibility
```

**Memory formula:** `model_size (5GB) + batch_size × 280MB`

### Calibration Datasets

- **c4** (default): High quality, cross-dataset robustness, ~10GB
- **wikitext2-simple**: Fast, memory-efficient, ~6GB
- **wikitext2**: Balanced, chunked sequences

## Understanding the Results

### Perplexity Metrics

Lower perplexity = better quality

```
Method           WikiText-2    C4        AG News    Average
------------------------------------------------------------------------
Heuristic AWQ    15.05        13.38      25.45      17.96  ✅ Winner
Standard AWQ     15.26        13.57      26.11      18.31
Improvement      -1.34%       -1.36%     -2.51%     -1.74%
```

### Cross-Dataset Validation

Testing on 3 different datasets ensures:
- Not overfitting to calibration distribution (C4)
- Robustness across domains (Wikipedia, web, news)
- Generalization to real-world deployment

### Is Heuristic Worth It?

**Yes, based on empirical results:**
- Consistent improvement across all 3 datasets
- Average improvement: 1.74%
- Minimal speed overhead (~same 6-12 min runtime)
- Complexity justified by measurable quality gains

## Project Structure

```
.
├── gw_awq_asym_l2.py              # Standard AWQ quantization
├── awq_sh.py                      # Heuristic AWQ quantization
├── compare_awq_heuristic.py       # Cross-dataset comparison
├── calibration_utils.py           # Optimized data loading
├── quantized_models/
│   ├── minicpm_gw_awq_asym_l2/   # Standard AWQ output
│   └── minicpm_awq_sh/            # Heuristic AWQ output
└── results/
    ├── awq_heuristic_validation_*.json
    └── AWQ_HEURISTIC_SUMMARY.md
```

## Troubleshooting

### CUDA Out of Memory

**Solution 1:** Use WikiText-2 Simple (saves ~4GB)
```bash
python gw_awq_asym_l2.py --calib-dataset wikitext2-simple
```

**Solution 2:** Reduce batch size
```bash
python gw_awq_asym_l2.py --layer-batch-size 20  # 6GB instead of 14GB
```

**Solution 3:** Combine optimizations
```bash
python gw_awq_asym_l2.py \
    --calib-dataset wikitext2-simple \
    --n-calib 64 \
    --layer-batch-size 10  # ~3GB total
```

### All α Values are 0.0

**Symptom:** Grid search shows α=0.0 for all layers

**Cause:** Dtype mismatch (fixed in current version)

**Solution:** Update to latest code - dtype preservation is now implemented

### Slow Performance

Check:
- Using GPU? (10-20× faster than CPU)
- Appropriate batch size? (larger = faster but more memory)
- Network speed for C4 download?

## Technical Details

### Quantization Pipeline

1. **Calibration:** Capture activations from 128 C4 samples
2. **Salience:** Compute per-channel L2 importance: `E[X²]`
3. **Grid Search:** Find optimal α ∈ [0, 1] (20 points)
4. **Scaling:** Apply per-channel scaling: `W[:,j] *= s[j]^α`
5. **Quantization:** Group-wise asymmetric INT4 [0,15]
   - Per-group scale: `(max - min) / 15`
   - Per-group zero-point: `round(-min / scale)`
6. **Heuristic (awq_sh.py only):** Global greedy rounding correction
7. **Unscaling:** `W_final = Q(W×s) / s`

### Implementation Features

- **Batched sequential processing** for memory efficiency
- **Dtype preservation** to prevent forward pass failures
- **Optimized C4 loading** with fast filtering and random slicing
- **Periodic cache clearing** to prevent memory accumulation
- **Error propagation awareness** for realistic calibration

## Research Context

### Core Research Questions

1. **L2 vs L1 Salience:** Does E[X²] improve upon E[|X|]? → Yes
2. **Heuristic Rounding:** Does global greedy reduce error? → **Yes (1-2.5%)**
3. **Asymmetric vs Symmetric:** Does [0,15] outperform [-7,7]? → Yes
4. **Group Size:** What's optimal for hardware vs quality? → 128 (default)

### Key Findings

**Heuristic rounding is effective:**
- Consistent quality improvement across all datasets
- Minimal computational overhead
- Benefits from E[Xs] statistics and outlier masking

**Batched sequential is practical:**
- Enables 128+ calibration samples on consumer hardware
- Memory-efficient without sacrificing quality
- Error propagation aware (more realistic than batch)

## Citation

```bibtex
@misc{awq-heuristic-2025,
  title={Heuristic-Guided AWQ: Global Greedy Rounding for LLM Quantization},
  author={AWQ Quantization Research},
  year={2025},
  note={Batched sequential quantization with cross-dataset validation}
}
```

## Additional Documentation

- `SEQUENTIAL_QUANTIZATION.md` - Batched sequential strategy details
- `CALIBRATION_OPTIMIZATION.md` - C4 and WikiText-2 loading optimizations
- `AWQ_SH_BATCHED_SEQUENTIAL.md` - awq_sh.py implementation guide
- `CLAUDE.md` - Complete project guide for Claude Code

## License

[Specify your license here]

---

**Recommendation:** Use Heuristic AWQ (awq_sh.py) for production deployment - it provides measurable quality improvements with minimal overhead.
