# awq_op_x.py - Fixed Version Documentation

## Overview

`awq_op_x.py` is a corrected version of `awq_op_ref.py` that fixes three bugs causing perplexity differences from `gw_awq_asym_l2.py`.

## Changes Applied

### 1. Fixed Salience Bias (Lines 296-300)

**Before (awq_op_ref.py):**
```python
# Line 296
activation_salience = activation_salience + 1e-6  # ← BUG!

# Line 300
scales = activation_salience.pow(alpha)
```

**After (awq_op_x.py):**
```python
# Line 296 REMOVED

# Line 306
scales = activation_salience.pow(alpha).clamp(min=1e-5)
```

**Why:** The `+1e-6` bias systematically shifts all scale values, causing different quantization results. Using `clamp(min=1e-5)` handles division by zero correctly without introducing bias.

---

### 2. Fixed Bias Handling in Grid Search (Lines 287-312)

**Before (awq_op_ref.py):**
```python
# Line 287-288
W = module.weight.data
Y_orig = torch.matmul(X_search, W.t())  # ← Missing bias!

# Line 312
Y_quant = torch.matmul(X_search, W_recon.t())  # ← Missing bias!
```

**After (awq_op_x.py):**
```python
# Line 293-298
W = module.weight.data
b = module.bias.data if module.bias is not None else None
if b is not None:
    Y_orig = torch.matmul(X_search, W.t()) + b
else:
    Y_orig = torch.matmul(X_search, W.t())

# Line 317-321
W_recon = W_quant / scales.unsqueeze(0)
if b is not None:
    Y_quant = torch.matmul(X_search, W_recon.t()) + b
else:
    Y_quant = torch.matmul(X_search, W_recon.t())
```

**Why:** Grid search must include bias to accurately compute reconstruction error for layers with bias (most MLP layers). Ignoring bias leads to suboptimal alpha selection.

---

### 3. Added Safety Check for Outlier Masking (Line 201)

**Before (awq_op_ref.py):**
```python
# Line 180-181
k_outliers = int(padded_in_features * self.outlier_percent)
if k_outliers > 0:
```

**After (awq_op_x.py):**
```python
# Line 201
if k_outliers > 0 and k_outliers < padded_in_features:
```

**Why:** Prevents edge case where `outlier_percent >= 1.0` would try to mask all channels.

---

### 4. Improved Command-Line Interface

**Added:**
- `--use-heuristic` flag (instead of hardcoded in code)
- Warning for high `outlier_percent` values (> 0.5)
- Better help messages
- Status output showing heuristic state

---

## Usage

### Basic Usage (Matches gw_awq_asym_l2.py)

```bash
# Quantize without heuristic (should match gw_awq_asym_l2.py)
python awq_op_x.py \
  --n-calib 128 \
  --n-grid 20 \
  --group-size 128 \
  --output-dir ./quantized_models/awq_op_x_baseline

# With heuristic enabled
python awq_op_x.py \
  --n-calib 128 \
  --use-heuristic \
  --outlier-percent 0.05 \
  --output-dir ./quantized_models/awq_op_x_heuristic
```

### Verification Test

```bash
# 1. Quantize with fixed version (no heuristic)
python awq_op_x.py \
  --n-calib 128 \
  --output-dir ./quantized_models/awq_op_x_test

# 2. Quantize with gw_awq_asym_l2.py (reference)
python gw_awq_asym_l2.py \
  --n-calib 128 \
  --output-dir ./quantized_models/gw_awq_test

# 3. Compare perplexity
python compare_awq_heuristic.py \
  --heuristic-path ./quantized_models/awq_op_x_test \
  --standard-path ./quantized_models/gw_awq_test \
  --n-samples 2000
```

**Expected result:** Perplexity difference should be < 0.1 (< 0.5%)

---

## Parameter Recommendations

### outlier_percent
- **Recommended:** `0.01` to `0.10`
- **Default:** `0.05` (5%)
- **Purpose:** Exclude top X% activations to prevent overfitting
- **Warning:** Values > 0.5 will trigger a warning

### use_heuristic
- **False (default):** Standard AWQ quantization (nearest rounding)
- **True:** Enable heuristic rounding correction
- **Note:** Only use `True` if you understand the implications

### n_calib
- **Recommended:** `128` to `500`
- **Trade-off:** More samples = better quality but slower calibration
- **Default:** `128`

### n_grid
- **Recommended:** `10` to `30`
- **Trade-off:** More points = finer search but slower
- **Default:** `20`

### group_size
- **Recommended:** `128` (best quality-efficiency balance)
- **Options:** `32`, `64`, `128`, `256`
- **Trade-off:** Smaller = faster but lower quality

---

## Expected Performance

After fixes, with `--use-heuristic` disabled:

| Configuration | Perplexity | Notes |
|--------------|-----------|-------|
| awq_op_x.py (no heuristic) | ~15.35-15.40 | Should match gw_awq_asym_l2.py |
| gw_awq_asym_l2.py | ~15.35 | Reference baseline |
| awq_op_ref.py (no heuristic) | ~15.66 | OLD - Has bugs |

**Improvement:** ~0.3 perplexity reduction (2% better)

---

## Differences from awq_op_ref.py

| Feature | awq_op_ref.py | awq_op_x.py |
|---------|--------------|------------|
| Salience bias | +1e-6 before pow | clamp(min=1e-5) after pow ✓ |
| Bias in grid search | Ignored ✗ | Included ✓ |
| Outlier safety check | Missing | Added ✓ |
| use_heuristic flag | Hardcoded | CLI argument ✓ |
| Outlier warning | None | Added ✓ |
| Documentation | Minimal | Comprehensive ✓ |

---

## Testing Checklist

- [ ] Quantize with `awq_op_x.py` (no heuristic)
- [ ] Quantize with `gw_awq_asym_l2.py`
- [ ] Compare perplexity (should be < 0.1 difference)
- [ ] Run `diagnose_model_difference.py` to verify weights
- [ ] Test with heuristic enabled (optional)
- [ ] Verify outlier warning appears for high values

---

## Troubleshooting

### Perplexity still differs by > 0.1

**Check:**
1. Same calibration samples used? (`--n-calib` and `--seed`)
2. Same grid search points? (`--n-grid`)
3. Same group size? (`--group-size`)
4. Models loaded with correct dtype? (should be bfloat16)

**Debug:**
```bash
python diagnose_model_difference.py
```

### Out of memory

**Solutions:**
```bash
# Reduce calibration samples
python awq_op_x.py --n-calib 64

# Reduce grid search points
python awq_op_x.py --n-grid 10

# Use CPU (slower)
CUDA_VISIBLE_DEVICES="" python awq_op_x.py
```

### Outlier warning

If you see:
```
⚠️  WARNING: outlier_percent=0.999 is very high!
```

**Fix:** Use recommended range `0.01-0.10`:
```bash
python awq_op_x.py --outlier-percent 0.05
```

---

## Summary

`awq_op_x.py` fixes critical bugs in `awq_op_ref.py`:
1. ✅ Removes salience bias
2. ✅ Includes bias in grid search
3. ✅ Adds safety checks
4. ✅ Improves usability

With `--use-heuristic` disabled, it should produce nearly identical results to `gw_awq_asym_l2.py` (< 0.1 perplexity difference).
