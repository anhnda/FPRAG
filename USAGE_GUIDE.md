# Quantization Methods Comparison Guide

This guide explains the three quantization approaches and how to compare them.

## Overview of Methods

### 1. Real-AWQ (Baseline)
**File**: `quantize_minicpm_real_awq.py`

**Algorithm**:
- Computes activation salience: `s = E[|X|]` per input feature
- Grid searches for optimal scaling exponent `α ∈ [0, 1]`
- Scales weights: `W_scaled = W * (s^α)`
- Applies **uniform INT4 quantization** to ALL channels
- Minimizes reconstruction error: `||Q(W*s)*(s^-1*X) - WX||²`

**Characteristics**:
- ✅ High compression (~2.4x)
- ✅ Simple, no layer-type awareness
- ❌ Runtime scaling overhead
- ❌ No risk-awareness

---

### 2. PRAQ (Mixed-Precision)
**File**: `quantize_minicpm_PRAQ.py`

**Algorithm**:
- Computes risk-aware importance: `P(activation) × magnitude`
  - For MLP layers: Accounts for risky dead neurons
  - For Attention layers: Uses standard magnitude
- Selects top-k% channels based on importance
- **Mixed-precision**: Top-k in FP16, rest in INT4
- No optimization/grid search

**Characteristics**:
- ✅ Risk-aware (accounts for quantization noise)
- ✅ Layer-type aware (MLP vs Attention)
- ✅ No runtime overhead
- ❌ Lower compression (~1.5x)
- ❌ No optimization

---

### 3. Real-PRAQ (Best of Both Worlds)
**File**: `quantize_minicpm_real_praq.py`

**Algorithm**:
- Computes risk-aware salience:
  - MLP layers: `s = P(activation) × magnitude` (accounts for risky neurons)
  - Attention layers: `s = E[|X|]` (standard AWQ)
- Grid searches for optimal scaling exponent `α ∈ [0, 1]`
- Scales weights: `W_scaled = W * (s^α)`
- Applies **uniform INT4 quantization** with optimal scaling
- Minimizes reconstruction error with risk-aware importance

**Characteristics**:
- ✅ High compression (~2.4x)
- ✅ Risk-aware + layer-type aware
- ✅ Optimized via grid search
- ❌ Runtime scaling overhead
- ❌ Slower to quantize (grid search)

---

## Complete Workflow

### Step 1: Quantize Models

```bash
# 1. Real-AWQ (Baseline)
python quantize_minicpm_real_awq.py \
  --n-calib 500 \
  --n-grid 20 \
  --output-dir ./quantized_models/minicpm_real_awq

# 2. PRAQ (Mixed-Precision)
python quantize_minicpm_PRAQ.py \
  --keep-ratio 0.2 \
  --n-calib 500 \
  --output-dir ./quantized_models/minicpm_praq_hybrid

# 3. Real-PRAQ (Hybrid)
python quantize_minicpm_real_praq.py \
  --n-calib 500 \
  --n-grid 20 \
  --beta 3.0 \
  --tau -3.0 \
  --noise-factor 0.2 \
  --output-dir ./quantized_models/minicpm_real_praq
```

**Time Estimates**:
- Real-AWQ: ~2-3 hours (grid search)
- PRAQ: ~30-45 minutes (no optimization)
- Real-PRAQ: ~3-4 hours (grid search + risk computation)

---

### Step 2: Compare All Methods

```bash
# Full comparison with all three methods
python compare_praq_vs_real_awq.py \
  --original-model openbmb/MiniCPM-2B-sft-bf16 \
  --real-awq-path ./quantized_models/minicpm_real_awq \
  --praq-path ./quantized_models/minicpm_praq_hybrid \
  --real-praq-path ./quantized_models/minicpm_real_praq \
  --n-eval 2000 \
  --visualize
```

**Output**:
```
================================================================================
COMPARISON TABLE
================================================================================

Metric                         Original (FP16)      Real-AWQ             PRAQ (Mixed)         Real-PRAQ
------------------------------------------------------------------------------------------------------------
Perplexity                              15.8358             16.7234             17.4325             16.5123
Avg Loss                                 2.7623              2.8154              2.8589              2.8045
Model Size (MB)                        5277.30            2200.45            3200.23            2200.45
Throughput (tok/s)                     3164.63            3850.24            3181.01            3820.15
```

```
================================================================================
METHOD COMPARISON
================================================================================

🏆 OVERALL WINNER: Real-PRAQ
   Perplexity: 16.5123

Pairwise Comparisons:

  Real-AWQ vs Real-PRAQ (uniform INT4 comparison):
    ✅ Real-PRAQ wins by 1.26%
    → Risk-aware scaling beats activation-only scaling

  PRAQ (Mixed) vs Real-PRAQ (mixed vs uniform):
    ✅ Real-PRAQ wins by 5.30%
    → Uniform INT4 + scaling beats mixed-precision

  Real-AWQ vs PRAQ (Mixed) (paradigm comparison):
    ✅ Real-AWQ wins by 4.07%
```

---

## Expected Results & Interpretation

### Scenario 1: Real-PRAQ Wins
**Implication**: Risk-awareness + optimization is the winning combination
- Real-PRAQ combines PRAQ's intelligence with AWQ's optimization
- Grid search allows risk-aware importance to shine
- Best approach overall

**Next Steps**:
- Use Real-PRAQ for deployment
- Paper focus: "Optimized risk-aware quantization"
- Ablation: Test different β, τ, noise_factor values

---

### Scenario 2: Real-AWQ Wins
**Implication**: Simple activation salience is sufficient
- Risk-awareness doesn't provide value in optimized setting
- PRAQ's complexity not justified
- Simpler is better

**Next Steps**:
- Analyze why risk-awareness fails with optimization
- Check if grid search already accounts for risky neurons
- Consider: Does scaling implicitly handle risk?

---

### Scenario 3: PRAQ (Mixed) Wins
**Implication**: Mixed-precision > Uniform quantization
- Preserving critical channels in FP16 beats scaling
- Lower compression acceptable for better quality
- Risk-awareness validated

**Next Steps**:
- Optimize PRAQ: Add grid search for keep_ratio
- Test variance-penalized PRAQ
- Focus on mixed-precision optimization

---

## Parameter Tuning

### Real-AWQ Parameters
```bash
--n-grid 20      # More = better α, but slower (try: 10, 20, 30)
```

### PRAQ Parameters
```bash
--keep-ratio 0.2  # Higher = more FP16 = better quality (try: 0.1, 0.2, 0.3)
--beta 3.0        # Sharpness of activation probability (try: 1.0, 3.0, 5.0)
--tau -3.0        # SiLU activation threshold (fixed for SiLU)
--noise-factor 0.2 # Quantization noise estimate (try: 0.1, 0.2, 0.3)
```

### Real-PRAQ Parameters
```bash
--n-grid 20       # Grid search resolution
--beta 3.0        # Risk-awareness sharpness
--tau -3.0        # Activation threshold
--noise-factor 0.2 # Noise estimate
```

---

## Troubleshooting

### "Out of Memory" Errors
```bash
# Reduce calibration samples
--n-calib 100

# Reduce grid search points
--n-grid 10

# For Real-PRAQ, the salience computation is memory-intensive
# Edit the file to reduce max_samples in get_risk_aware_salience_mlp()
```

### Slow Quantization
```bash
# Use fewer calibration samples
--n-calib 100

# Use coarser grid search
--n-grid 10

# Run on GPU
# (Should auto-detect CUDA)
```

### Poor Results
```bash
# Try different PRAQ parameters
--beta 5.0 --noise-factor 0.3

# Try different keep ratios
--keep-ratio 0.3

# Use more calibration data
--n-calib 1000
```

---

## Research Paper Positioning

### If Real-PRAQ Wins:
**Title**: "Risk-Aware Activation Scaling for LLM Quantization"

**Contribution**:
- Novel risk-aware salience metric
- Integration with AWQ optimization framework
- Beats both AWQ and mixed-precision baselines

**Baselines**:
- Real-AWQ (activation-only)
- PRAQ (mixed-precision, no optimization)
- Show Real-PRAQ combines best of both

---

### If PRAQ (Mixed) Wins:
**Title**: "Risk-Aware Mixed-Precision Quantization for LLMs"

**Contribution**:
- Novel risk-aware importance metric
- Mixed-precision channel selection
- Outperforms uniform quantization approaches

**Baselines**:
- Real-AWQ (uniform INT4)
- Real-PRAQ (uniform INT4 + risk)
- Show mixed-precision is key advantage

---

## Quick Reference

| Method | Compression | Quality | Speed | Complexity |
|--------|-------------|---------|-------|------------|
| **Real-AWQ** | ~2.4x | Baseline | Fast | Low |
| **PRAQ (Mixed)** | ~1.5x | ? | Fast | Medium |
| **Real-PRAQ** | ~2.4x | ? | Fast | High |

**Legend**:
- Compression: Model size reduction
- Quality: Perplexity (? = depends on results)
- Speed: Inference speed
- Complexity: Implementation complexity

---

## Files Reference

- `quantize_minicpm_real_awq.py` - Real AWQ implementation
- `quantize_minicpm_PRAQ.py` - PRAQ mixed-precision implementation
- `quantize_minicpm_real_praq.py` - Real-PRAQ hybrid implementation
- `compare_praq_vs_real_awq.py` - Comprehensive comparison script
- `quantize_minicpm_PRAQ_var.py` - Variance-penalized PRAQ (optional)

---

**Good luck with your experiments! 🚀**
