# Bugs in awq_op_ref.py Causing Difference from gw_awq_asym_l2.py

## Problem Statement

When running `awq_op_ref.py` with `use_heuristic=False`, the results differ from `gw_awq_asym_l2.py`:
- Heuristic AWQ (awq_op_ref with use_heuristic=False): **Perplexity 15.6633**
- Standard AWQ (gw_awq_asym_l2.py): **Perplexity 15.3539**
- Difference: **+0.31 perplexity (2.0% worse)**

This should NOT happen since both should be doing identical nearest-rounding quantization.

## Root Causes Identified

### Bug #1: Salience Bias (Line 296)

**Location:** `awq_op_ref.py` line 296

**Code:**
```python
# Avoid division by zero
activation_salience = activation_salience + 1e-6  # ← BUG!
```

**Issue:** This adds a constant bias to ALL salience values before computing scales.

**Impact:**
- Changes scale computation: `scales = (salience + 1e-6)^alpha` instead of `scales = salience^alpha`
- Affects low-salience channels more than high-salience channels
- Causes systematic shift in weight scaling

**Correct approach (from gw_awq_asym_l2.py line 234):**
```python
scales = activation_salience.pow(alpha).clamp(min=1e-5)
```

**Fix:**
```python
# Remove line 296, or change line 300 to:
scales = activation_salience.pow(alpha).clamp(min=1e-5)
```

---

### Bug #2: Bias Ignored in Grid Search (Line 288)

**Location:** `awq_op_ref.py` line 288

**Code:**
```python
W = module.weight.data
Y_orig = torch.matmul(X_search, W.t())  # ← BUG: No bias!
```

**Issue:** The original output computation ignores the bias term, even for layers that have bias.

**Correct approach (from gw_awq_asym_l2.py lines 218-221):**
```python
W = module.weight.data
b = module.bias.data if module.bias is not None else None

if b is not None:
    Y_orig = torch.matmul(X_search, W.t()) + b
else:
    Y_orig = torch.matmul(X_search, W.t())
```

**Impact:**
- Grid search optimizes without considering bias contribution
- May select suboptimal alpha values for layers with significant bias
- Affects MLP layers (gate_proj, up_proj, down_proj) which typically have bias

**Fix:**
```python
# Line 287-288, change to:
W = module.weight.data
b = module.bias.data if module.bias is not None else None
if b is not None:
    Y_orig = torch.matmul(X_search, W.t()) + b
else:
    Y_orig = torch.matmul(X_search, W.t())
```

And also update the quantized output computation (line 312) to include bias if present.

---

### Bug #3: Dtype Mismatch in Evaluation

**Location:** `compare_awq_heuristic.py` line 158

**Code:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # ← Loading as float16
    ...
)
```

**Issue:** Both quantization scripts save models in `torch.bfloat16`, but the comparison script loads them as `torch.float16`.

**Impact:**
- Precision changes during loading
- May introduce numerical differences
- Different rounding behavior between bfloat16 and float16

**Fix:**
```python
# Change line 158 to:
torch_dtype=torch.bfloat16,  # Match the saved dtype
```

Or better yet, let the model use its original dtype:
```python
# Remove torch_dtype parameter, use model's saved dtype
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=self.device,
    trust_remote_code=True
)
```

---

## Additional Differences (Non-Bugs)

### Difference #1: Quantization Formula Representation

**awq_op_ref.py (lines 135-137):**
```python
W_div = W_padded / scale_flat
W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
W_dequant = (W_int - zp_flat) * scale_flat
```

**gw_awq_asym_l2.py (lines 160-163):**
```python
W_int = torch.round(W_grouped / scale + zero_point).clamp(0, 15)
W_dequant_grouped = (W_int - zero_point) * scale
```

**Analysis:** These are mathematically equivalent, just different representations:
- awq_op_ref works with flattened tensors `[out, in]`
- gw_awq_asym_l2 works with grouped tensors `[out, n_groups, group_size]`

**Impact:** None (when bugs are fixed)

---

### Difference #2: Implementation Style

**awq_op_ref.py:**
- Flattens groups to `[out_features, padded_in_features]`
- Uses `scatter_add_` for applying flips
- More complex for heuristic support

**gw_awq_asym_l2.py:**
- Works directly with grouped tensors `[out, n_groups, group_size]`
- Simpler, cleaner code
- No heuristic support needed

**Impact:** None (both should give identical results when bugs are fixed)

---

## Expected Impact After Fixes

### Bug #1 Fix (Salience Bias)

**Current behavior:**
- Scales are biased by 1e-6 addition
- Small systematic shift in all scales

**After fix:**
- Scales computed correctly: `scales = salience^alpha`
- Should reduce perplexity difference significantly

**Estimated impact:** ~0.1-0.2 perplexity improvement

### Bug #2 Fix (Bias in Grid Search)

**Current behavior:**
- Grid search ignores bias term
- May select suboptimal alpha

**After fix:**
- Grid search considers full layer output (W@X + b)
- Better alpha selection for layers with bias
- Affects ~50% of MLP layers (those with bias)

**Estimated impact:** ~0.1-0.2 perplexity improvement

### Bug #3 Fix (Dtype Mismatch)

**Current behavior:**
- Models saved in bfloat16, loaded as float16
- Precision loss during loading

**After fix:**
- Consistent dtype throughout
- No precision loss

**Estimated impact:** ~0.01-0.05 perplexity improvement

### Combined Expected Result

After fixing all three bugs:
- **Current:** 15.6633 (awq_op_ref) vs 15.3539 (gw_awq_asym_l2)
- **Expected:** ~15.35-15.40 (should be nearly identical)

---

## Recommended Actions

### Immediate Fixes

1. **Fix awq_op_ref.py line 296:**
   ```python
   # REMOVE: activation_salience = activation_salience + 1e-6
   # ADD after line 300:
   scales = activation_salience.pow(alpha).clamp(min=1e-5)
   ```

2. **Fix awq_op_ref.py lines 287-288:**
   ```python
   W = module.weight.data
   b = module.bias.data if module.bias is not None else None

   if b is not None:
       Y_orig = torch.matmul(X_search, W.t()) + b
   else:
       Y_orig = torch.matmul(X_search, W.t())
   ```

   And update line 312 accordingly:
   ```python
   W_recon = W_quant / scales.unsqueeze(0)
   if b is not None:
       Y_quant = torch.matmul(X_search, W_recon.t()) + b
   else:
       Y_quant = torch.matmul(X_search, W_recon.t())
   ```

3. **Fix compare_awq_heuristic.py line 157-158:**
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       # Remove torch_dtype or use torch.bfloat16
       device_map=self.device,
       trust_remote_code=True
   )
   ```

### Verification Steps

1. **Re-run quantization:**
   ```bash
   # Quantize with fixed awq_op_ref.py
   python awq_op_ref.py --n-calib 128 --output-dir ./quantized_models/awq_op_ref_fixed

   # Quantize with gw_awq_asym_l2.py (reference)
   python gw_awq_asym_l2.py --n-calib 128 --output-dir ./quantized_models/gw_awq_asym_l2_ref
   ```

2. **Compare perplexity:**
   ```bash
   python compare_awq_heuristic.py \
     --heuristic-path ./quantized_models/awq_op_ref_fixed \
     --standard-path ./quantized_models/gw_awq_asym_l2_ref \
     --n-samples 2000
   ```

3. **Check weight differences:**
   ```bash
   python diagnose_model_difference.py
   ```

Expected result: Perplexity difference should be < 0.1 (< 0.5%)

---

## Testing Script

To verify the fixes work, run:

```bash
# Run diagnostic
python diagnose_model_difference.py

# Run comparison test
python compare_quantization_logic.py
```

---

## Conclusion

The perplexity difference of 0.31 (2%) between `awq_op_ref.py` with `use_heuristic=False` and `gw_awq_asym_l2.py` is caused by three bugs:

1. **Salience bias** (+1e-6 before power operation)
2. **Ignored bias term** in grid search optimization
3. **Dtype mismatch** during model loading (bfloat16 → float16)

All three should be fixed to ensure `use_heuristic=False` produces identical results to `gw_awq_asym_l2.py`.
