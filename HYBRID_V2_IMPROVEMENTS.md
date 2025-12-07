# Hybrid PRAQ V2 - Improvements & Usage

## Problem with V1

The original hybrid (GWH-PRAQ) performed **worse** than both pure methods on C4:
- GW-AWQ: 13.5323 (best)
- GW-PRAQ: 13.5748 (+0.31%)
- **GWH-PRAQ: 13.7577 (+1.67%)** ❌

## Root Causes Identified

### 1. **Insufficient Calibration Data**
- V1: Only 128 samples
- Pure methods: 150 samples
- **Impact:** Overfitting to limited WikiText-2 patterns

### 2. **Aggressive Normalization**
```python
# V1 - Loses magnitude information
awq_salience_norm = awq_salience / (awq_salience.mean() + 1e-8)
```
- Dividing by mean can create extreme values
- Loses natural magnitude relationships

### 3. **Conflicting Signals**
- Uses AWQ (input importance) for scaling
- Uses PRAQ (output importance) for error weighting
- No blending → methods work against each other

### 4. **Over-focused Error Weighting**
- PRAQ weighting can become too aggressive
- Over-optimizes for few channels
- Poor generalization to diverse C4 data

### 5. **Poor Attention Layer Handling**
- Attention layers got uniform PRAQ importance
- Wasted the potential of hybrid approach

## Key Improvements in V2

### ✅ 1. More Calibration Data
```python
def calibrate(self, calibration_data, n_samples=150):  # Was 128
```
**Benefit:** Matches pure methods, reduces overfitting

### ✅ 2. Adaptive Blending for Scaling
```python
blended_salience = (1 - β) * awq_salience + β * praq_salience
```
- **β = 0.0:** Pure AWQ (robust, input-based)
- **β = 0.5:** Balanced (default)
- **β = 1.0:** Pure PRAQ (intelligent, output-based)

**Benefit:** Combines strengths of both methods instead of conflict

### ✅ 3. Temperature-Controlled Error Weighting
```python
praq_weights = softmax(praq_importance / τ)
```
- **τ > 1:** Softer, more uniform weighting → better generalization
- **τ = 1:** Standard softmax
- **τ < 1:** Sharper, focused weighting

**Benefit:** Prevents over-focusing on few channels

### ✅ 4. Softer Normalization
```python
# V2 - Preserves magnitude, prevents extremes
def soft_normalize(x):
    return 0.1 + 9.9 * (x - x_min) / (x_max - x_min)  # Map to [0.1, 10]
```
**Benefit:** Stable, preserves natural importance relationships

### ✅ 5. Better Attention Layer Handling
- Computes E[|XW|] for attention layers (no activation function)
- Uses output magnitude for error weighting
- No longer uses uniform importance

**Benefit:** Leverages hybrid approach for all layer types

## Usage

### Quick Start (Balanced Configuration)
```bash
# Use default balanced settings
python gwh_praq_v2.py --n-calib 150
```

### Hyperparameter Tuning
```bash
# Make script executable
chmod +x tune_hybrid.sh

# Run all configurations
./tune_hybrid.sh
```

This tests 4 configurations:
1. **Balanced** (β=0.5, τ=2.0) - Equal mix
2. **AWQ-leaning** (β=0.3, τ=2.5) - More robust
3. **PRAQ-leaning** (β=0.7, τ=1.5) - More intelligent
4. **Conservative** (β=0.4, τ=3.0) - Safest generalization

### Manual Configuration
```bash
python gwh_praq_v2.py \
    --n-calib 150 \
    --blend-beta 0.5 \    # Blending: 0=AWQ, 1=PRAQ
    --temp-tau 2.0 \      # Temperature: higher=softer
    --n-grid 20 \
    --group-size 128 \
    --output-dir ./quantized_models/custom_hybrid
```

### Evaluate on C4
```bash
# Update comparison script to use V2 model
python compare_gw_quantize_cross.py \
    --gwh-praq-path ./quantized_models/minicpm_gwh_v2_balanced \
    --visualize --save-csv
```

## Expected Improvements

### Target Performance on C4
- **Goal:** Beat or match best pure method (GW-AWQ: 13.5323)
- **Minimum:** No worse than 0.5% behind best pure method

### Why V2 Should Work Better

1. **More data** = Less overfitting
2. **Blending** = Combines strengths, not conflicts
3. **Temperature** = Better generalization
4. **Softer norm** = Stability
5. **Better attention** = Utilizes all layers

## Hyperparameter Recommendations

### For Best C4 Performance (Generalization)
- **β = 0.3-0.4** (AWQ-leaning): AWQ is more robust on diverse data
- **τ = 2.0-3.0** (Soft): Prevents overfitting to calibration set

### For Best WikiText-2 Performance (In-Distribution)
- **β = 0.5-0.7** (Balanced to PRAQ): PRAQ better on similar data
- **τ = 1.5-2.0** (Moderate): Can be more focused

### Recommended Starting Point
```bash
python gwh_praq_v2.py \
    --blend-beta 0.4 \   # Slightly AWQ-leaning for robustness
    --temp-tau 2.5       # Moderately soft for generalization
```

## Next Steps

1. **Run single config:**
   ```bash
   python gwh_praq_v2.py --blend-beta 0.4 --temp-tau 2.5
   ```

2. **Evaluate on C4:**
   ```bash
   python compare_gw_quantize_cross.py \
       --gwh-praq-path ./quantized_models/minicpm_gwh_praq_v2
   ```

3. **If still underperforming, try tuning:**
   ```bash
   ./tune_hybrid.sh
   ```

4. **Compare all configs:**
   Create evaluation loop for all 4 configurations

## Theory: Why Hybrid Can Work

**Pure AWQ:**
- ✅ Robust scaling (E[|X|])
- ❌ Ignores activation functions

**Pure PRAQ:**
- ✅ Intelligent (E[|activation(XW)|])
- ❌ May create uneven groups for group-wise quantization

**Improved Hybrid:**
- ✅ Robust scaling (AWQ + PRAQ blend)
- ✅ Intelligent optimization (temperature-controlled PRAQ)
- ✅ Balanced trade-off
- ✅ Better generalization

The key insight: **Use AWQ for structural stability, PRAQ for optimization intelligence, with temperature control to prevent overfitting.**
