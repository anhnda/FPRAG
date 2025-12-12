# Outlier Masking Analysis: Why outlier_percent=0.999 ≠ apply_heuristic=False

## Issue Summary

When running `awq_op_ref.py` with `outlier_percent=0.999`, results differ from running with `apply_heuristic=False`, even though 99.9% of channels are masked. This document explains why.

## Root Cause

The heuristic algorithm is **NOT disabled** by high outlier percentages. It's just **restricted** to a bad set of candidates.

### Key Differences:

| Configuration | Behavior | Channels that can flip |
|--------------|----------|----------------------|
| `apply_heuristic=False` | Heuristic **completely disabled** | 0 channels (no flips) |
| `outlier_percent=0.999` | Heuristic **still active** | Bottom 0.1% (smallest activations) |

## How Outlier Masking Works

From `awq_op_ref.py` lines 180-188:

```python
k_outliers = int(padded_in_features * self.outlier_percent)
if k_outliers > 0:
    _, outlier_indices = torch.topk(act_padded.abs(), k_outliers)
    is_outlier = torch.zeros(padded_in_features, dtype=torch.bool, device=device)
    is_outlier[outlier_indices] = True
    valid_mask = valid_mask & (~is_outlier).unsqueeze(0)
```

**What this does:**
1. Computes `k_outliers = floor(padded_in_features × outlier_percent)`
2. Selects the **TOP k_outliers** channels by **absolute activation magnitude**
3. Marks these as outliers (cannot be flipped)
4. Remaining channels are candidates for flipping

**With outlier_percent=0.999:**
- If `padded_in_features = 4096`, then `k_outliers = 4091`
- Masks the 4091 **largest** activations
- Leaves only 5 **smallest** activations as flip candidates

## Why This Causes Different Results

### Multi-Output Channel Behavior

The quantization operates on a weight matrix `W[out_features, in_features]`:

1. **Error is computed PER output channel** (line 151):
   ```python
   current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1)  # [out_features]
   ```

2. **Each output channel independently optimizes** its own error by selecting which input channels to flip

3. **The outlier mask is SHARED** across all output channels:
   - Same input channels are masked for all outputs
   - But different outputs may make different flipping decisions

4. **Even with 99.9% outliers**, some output channels may still find it beneficial to flip the remaining 0.1% smallest channels

### Empirical Test Results

From `test_multioutput_outlier.py` (10 outputs × 100 inputs):

| Configuration | Output Error | Weight L1 | Flips |
|--------------|-------------|-----------|-------|
| No heuristic | 0.731467 | 86.405136 | 0 |
| Heuristic outlier=0.05 | 0.046292 | 88.791962 | 61 |
| Heuristic outlier=0.50 | 0.025861 | 99.568184 | 115 |
| Heuristic outlier=0.90 | 0.694471 | 93.122719 | 43 |
| **Heuristic outlier=0.999** | **0.731405** | **86.753220** | **3** |

**Key observation:** With outlier=0.999, the algorithm still applies **3 flips**, causing:
- Weight L1 difference: 0.348 from baseline
- Weight MSE difference: 0.000110 from baseline

Even though these differences are small, they demonstrate that the heuristic is **still active** and modifying the weights.

## Why This Is Problematic

### 1. **Flipping Wrong Channels**

The smallest activations contribute least to the output `y = Wx`:
- Large activations dominate the dot product
- Small activations have minimal impact
- Flipping small-activation channels doesn't meaningfully reduce error

### 2. **Counterproductive Optimization**

With 99.9% masking:
- The algorithm can only flip channels that don't matter
- It may flip multiple small channels to "minimize" error mathematically
- But this doesn't translate to actual quality improvement
- May even introduce noise

### 3. **Misaligned Expectations**

Users might expect:
```python
outlier_percent=0.999  ≈  apply_heuristic=False  # WRONG!
```

But actually:
```python
outlier_percent=0.999 → heuristic active, but broken
apply_heuristic=False → heuristic completely disabled
```

## Design Intent of Outlier Masking

The outlier masking feature is designed to **prevent overfitting** to calibration data:

- **Purpose:** Exclude the largest activations (which may be outliers) to avoid overweighting them
- **Typical value:** `outlier_percent=0.05` (exclude top 5%)
- **Rationale:** Top 5% might be anomalies; optimize based on the remaining 95%

**This makes sense at low percentages:**
- `outlier_percent=0.05`: Exclude top 5%, use 95% for optimization ✓
- `outlier_percent=0.10`: Exclude top 10%, use 90% for optimization ✓
- `outlier_percent=0.50`: Exclude top 50%, use 50% for optimization (questionable)

**This breaks down at high percentages:**
- `outlier_percent=0.90`: Exclude top 90%, use bottom 10% (broken)
- `outlier_percent=0.999`: Exclude top 99.9%, use bottom 0.1% (completely broken)

## Recommended Fix Options

### Option 1: Add Safety Check (Conservative)

Prevent unreasonable outlier percentages:

```python
def __init__(self, ..., outlier_percent=0.05):
    if outlier_percent > 0.50:
        raise ValueError(
            f"outlier_percent={outlier_percent} is too high. "
            "Recommended range: 0.01-0.10. "
            "Use apply_heuristic=False to disable heuristic instead."
        )
    self.outlier_percent = outlier_percent
```

### Option 2: Auto-Disable Heuristic (Aggressive)

Automatically disable heuristic if outlier percent is too high:

```python
def quantize_weight_heuristic_groupwise(self, W, group_activation_means, apply_heuristic=True):
    # Auto-disable if outlier masking would break the algorithm
    if apply_heuristic and self.outlier_percent > 0.90:
        print(f"⚠️  outlier_percent={self.outlier_percent} too high, disabling heuristic")
        apply_heuristic = False

    if not apply_heuristic:
        # ... return early ...
```

### Option 3: Warning Only (Minimal)

Just warn the user:

```python
def __init__(self, ..., outlier_percent=0.05):
    self.outlier_percent = outlier_percent
    if outlier_percent > 0.50:
        print(f"⚠️  WARNING: outlier_percent={outlier_percent} is very high.")
        print(f"   This may cause suboptimal quantization.")
        print(f"   Recommended range: 0.01-0.10")
        print(f"   Use apply_heuristic=False to fully disable heuristic.")
```

## Correct Usage Guidelines

### ✅ **Correct Usage**

```python
# Standard use case: exclude top 5% outliers
quantizer = HeuristicGroupWiseAWQQuantizer(
    model=model,
    outlier_percent=0.05,  # Good default
    ...
)

# Disable heuristic entirely
quantizer = HeuristicGroupWiseAWQQuantizer(
    model=model,
    use_heuristic=False,  # Explicitly disable
    ...
)
```

### ❌ **Incorrect Usage**

```python
# DON'T: Use high outlier_percent expecting it to disable heuristic
quantizer = HeuristicGroupWiseAWQQuantizer(
    model=model,
    outlier_percent=0.999,  # WRONG! Heuristic still active
    ...
)
```

### ⚠️ **Questionable Usage**

```python
# Borderline: May work but reduces effectiveness
quantizer = HeuristicGroupWiseAWQQuantizer(
    model=model,
    outlier_percent=0.20,  # Questionable
    ...
)
```

## Recommended Parameters

| Parameter | Recommended Range | Purpose |
|-----------|------------------|---------|
| `outlier_percent` | 0.01 - 0.10 | Balance between outlier robustness and optimization power |
| `outlier_percent` | **0.05** (default) | **Good default for most cases** |
| `use_heuristic` | True/False | Enable or completely disable heuristic |

## Conclusion

**The key insight:** `outlier_percent` controls which channels the heuristic **can flip**, not whether the heuristic is **enabled**. To disable the heuristic entirely, use `apply_heuristic=False` (or `use_heuristic=False` in the class constructor).

**The bug (if any):** The code allows unreasonable `outlier_percent` values that break the intended optimization behavior. This should either be prevented, warned about, or auto-corrected.

**Answer to original question:** Results differ with `outlier_percent=0.999` because the heuristic is still active and flipping the smallest 0.1% of channels, even though these flips don't meaningfully improve (and may harm) the quantization quality.
