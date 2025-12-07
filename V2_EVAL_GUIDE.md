# V2 Hyperparameter Search & Comparison Guide

## Quick Start

Run the automated search and comparison:

```bash
python find_best_hybrid.py
```

This will:
1. ✅ Quantize 4 V2 configurations (β, τ combinations)
2. ✅ Evaluate all on C4 (2000 samples)
3. ✅ Compare with V1 and GW-AWQ
4. ✅ Find best configuration
5. ✅ Save results to JSON
6. ✅ Provide recommendation

## What Gets Tested

### V2 Configurations (all with 128 calibration samples)

1. **Balanced** (β=0.5, τ=2.0)
   - Equal AWQ+PRAQ blend
   - Moderate temperature
   - Balanced approach

2. **AWQ-Leaning** (β=0.3, τ=2.5)
   - More AWQ (30% PRAQ, 70% AWQ)
   - Softer weighting
   - Best for diverse/noisy data

3. **PRAQ-Leaning** (β=0.7, τ=1.5)
   - More PRAQ (70% PRAQ, 30% AWQ)
   - Sharper weighting
   - Best for in-distribution data

4. **Conservative** (β=0.4, τ=3.0)
   - Slightly AWQ-leaning
   - Very soft weighting
   - Safest for generalization

### Baselines

- **V1-Original**: Current gwh_praq.py (if exists)
- **GW-AWQ**: Pure AWQ baseline (if exists)

## Expected Runtime

- **Quantization**: ~15-20 min per config × 4 = 60-80 min
- **Evaluation**: ~10 min per model × 6 = 60 min
- **Total**: ~2-2.5 hours

## Output

### Console Output

```
================================================================================
RESULTS SUMMARY
================================================================================

Perplexity Ranking (lower is better):
Rank   Model                Perplexity   β        τ
----------------------------------------------------------------------
🏆 1    V2-Conservative      13.7400      0.40     3.00
   2    V2-AWQ-Leaning       13.7455      0.30     2.50
   3    V1-Original          13.7577      N/A      N/A
   4    V2-Balanced          13.7590      0.50     2.00
   5    GW-AWQ               13.7670      0.00     N/A
   6    V2-PRAQ-Leaning      13.7720      0.70     1.50

================================================================================
BEST V2 CONFIGURATION
================================================================================
Winner: V2-Conservative
  Perplexity: 13.7400
  β (blend): 0.40
  τ (temp): 3.00
  Description: Slightly AWQ-leaning, very soft weighting

V2 vs V1:
  V1:      13.7577
  V2:      13.7400
  Delta:   -0.0177 (-0.129%)
  → ✅ V2 WINS by 0.129%!

Best V2 vs GW-AWQ:
  GW-AWQ:  13.7670
  Best V2: 13.7400
  Delta:   -0.0270 (-0.196%)
  → ✅ V2 WINS by 0.196%!

================================================================================
RECOMMENDATION
================================================================================
✅ Use V2 with Conservative configuration
   Path: ./quantized_models/minicpm_gwh_v2_conservative
```

### Saved Results

Results saved to: `./results/v2_hyperparameter_search.json`

```json
{
  "V2-Conservative": {
    "perplexity": 13.7400,
    "avg_loss": 2.6200,
    "blend_beta": 0.4,
    "temp_tau": 3.0,
    "description": "Slightly AWQ-leaning, very soft weighting"
  },
  ...
}
```

## What to Look For

### 🏆 Success Criteria

1. **Best V2 < V1**: V2 improves over current implementation
2. **Best V2 < GW-AWQ**: Hybrid beats pure AWQ
3. **Margin > 0.1%**: Improvement is meaningful

### Decision Guide

**If Best V2 wins by >0.1%:**
→ ✅ Use the winning V2 configuration

**If V2 ≈ V1 (within 0.05%):**
→ Stick with V1 (simpler, fewer hyperparameters)

**If V1 or GW-AWQ wins:**
→ V2 improvements don't help, use simpler method

## Manual Testing (If Needed)

Test a specific configuration:

```bash
python gwh_praq_v2.py \
    --n-calib 128 \
    --blend-beta 0.4 \
    --temp-tau 2.5 \
    --output-dir ./quantized_models/custom_v2

python compare_gw_quantize_cross.py \
    --gwh-praq-path ./quantized_models/custom_v2 \
    --visualize
```

## Understanding Parameters

### β (Blend Beta) - Scaling Strategy

- **β = 0.0**: Pure AWQ - E[|X|] only (robust, simple)
- **β = 0.3**: AWQ-leaning - 30% PRAQ, 70% AWQ (good for C4)
- **β = 0.5**: Balanced - Equal AWQ + PRAQ
- **β = 0.7**: PRAQ-leaning - 70% PRAQ, 30% AWQ
- **β = 1.0**: Pure PRAQ - Backprop importance only (intelligent)

**Rule of thumb:**
- More diverse data (C4) → Lower β (AWQ-leaning)
- In-distribution data (WikiText-2) → Higher β (PRAQ-leaning)

### τ (Temperature Tau) - Error Weighting

- **τ = 1.0**: Standard softmax (sharp weighting)
- **τ = 2.0**: Moderate softening (balanced)
- **τ = 3.0**: Very soft weighting (best generalization)
- **τ > 3.0**: Nearly uniform (too soft)

**Rule of thumb:**
- Cross-dataset validation → Higher τ (softer)
- Same-dataset validation → Lower τ (sharper)

## Troubleshooting

### Out of Memory
Reduce evaluation samples:
```python
# In find_best_hybrid.py, line 275
n_eval_samples = 1000  # Instead of 2000
```

### Quantization Fails
Check individual config:
```bash
python gwh_praq_v2.py --help
```

### Models Don't Exist
The script will skip missing models and compare what's available.

## Next Steps After Finding Best Config

1. **Document the winner**:
   ```bash
   # Update your main quantization script to use winning params
   echo "Best config: β=0.4, τ=3.0" >> RESULTS.md
   ```

2. **Test on WikiText-2** (in-distribution):
   ```bash
   python compare_groupwise.py \
       --gwh-praq-path ./quantized_models/minicpm_gwh_v2_[winner]
   ```

3. **Deploy the best model**:
   ```bash
   cp -r ./quantized_models/minicpm_gwh_v2_[winner] ./production/
   ```

4. **Update documentation** with optimal hyperparameters

## Summary

This script automates the entire V2 hyperparameter search process:
- ✅ No manual intervention needed
- ✅ Fair comparison (all use 128 samples)
- ✅ Comprehensive evaluation (4 configs + baselines)
- ✅ Clear recommendation
- ✅ Saved results for future reference

**Just run `python find_best_hybrid.py` and wait for results!**
