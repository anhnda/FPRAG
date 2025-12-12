"""
Diagnose why awq_op_ref.py with use_heuristic=False differs from gw_awq_asym_l2.py

This script checks:
1. Are the models actually quantized?
2. Do they have the same number of parameters?
3. Are the weights actually different?
4. What's the magnitude of weight differences?
"""

import torch
from transformers import AutoModelForCausalLM
import os

def load_and_analyze_model(model_path, name):
    """Load model and analyze its weights"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found!")
        return None

    # Check what files exist
    print(f"\nFiles in directory:")
    for file in os.listdir(model_path):
        if file.endswith(('.bin', '.safetensors', '.json')):
            file_path = os.path.join(model_path, file)
            size_mb = os.path.getsize(file_path) / (1024**2)
            print(f"  {file}: {size_mb:.2f} MB")

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Analyze weights
        print(f"\nModel architecture:")
        print(f"  Config: {model.config}")
        print(f"  Dtype: {model.dtype}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nParameter count: {total_params:,}")

        # Sample some weights
        print(f"\nSample weights from first layer:")
        first_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = (name, module)
                break

        if first_layer:
            name, module = first_layer
            W = module.weight.data
            print(f"  Layer: {name}")
            print(f"  Shape: {W.shape}")
            print(f"  Dtype: {W.dtype}")
            print(f"  Mean: {W.mean():.6f}")
            print(f"  Std: {W.std():.6f}")
            print(f"  Min: {W.min():.6f}")
            print(f"  Max: {W.max():.6f}")
            print(f"  Unique values (first 1000): {W.flatten()[:1000].unique().shape[0]}")

        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def compare_models(model1, model2, name1, name2):
    """Compare two models layer by layer"""
    print(f"\n{'='*80}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'='*80}")

    if model1 is None or model2 is None:
        print("Cannot compare - one or both models failed to load")
        return

    # Get state dicts
    state1 = model1.state_dict()
    state2 = model2.state_dict()

    # Check keys match
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())

    if keys1 != keys2:
        print("❌ Models have different keys!")
        print(f"  Only in {name1}: {keys1 - keys2}")
        print(f"  Only in {name2}: {keys2 - keys1}")
        return

    print("✅ Models have identical keys")

    # Compare weights
    print(f"\nComparing {len(keys1)} parameters...")

    max_diff = 0
    max_diff_key = None
    total_diff = 0
    num_weights = 0
    num_identical = 0

    for key in sorted(keys1):
        w1 = state1[key]
        w2 = state2[key]

        if w1.shape != w2.shape:
            print(f"❌ Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
            continue

        diff = (w1 - w2).abs()
        max_layer_diff = diff.max().item()
        mean_layer_diff = diff.mean().item()

        total_diff += mean_layer_diff
        num_weights += 1

        if max_layer_diff > max_diff:
            max_diff = max_layer_diff
            max_diff_key = key

        if max_layer_diff < 1e-6:
            num_identical += 1

    print(f"\nDifference statistics:")
    print(f"  Layers compared: {num_weights}")
    print(f"  Identical layers (diff < 1e-6): {num_identical} ({100*num_identical/num_weights:.1f}%)")
    print(f"  Average difference: {total_diff/num_weights:.6f}")
    print(f"  Maximum difference: {max_diff:.6f}")
    print(f"  Layer with max diff: {max_diff_key}")

    if max_diff < 1e-5:
        print("\n✅ Models are VIRTUALLY IDENTICAL (max diff < 1e-5)")
    elif max_diff < 1e-3:
        print("\n⚠️  Models are SLIGHTLY DIFFERENT (1e-5 < max diff < 1e-3)")
    else:
        print(f"\n❌ Models are SIGNIFICANTLY DIFFERENT (max diff = {max_diff:.6f})")

        # Show top 10 different layers
        print(f"\nTop 10 layers by difference:")
        diffs = []
        for key in keys1:
            w1 = state1[key]
            w2 = state2[key]
            if w1.shape == w2.shape:
                diff = (w1 - w2).abs().mean().item()
                diffs.append((key, diff))

        diffs.sort(key=lambda x: x[1], reverse=True)
        for i, (key, diff) in enumerate(diffs[:10]):
            print(f"  {i+1}. {key}: {diff:.6f}")


def main():
    print("="*80)
    print("DIAGNOSTIC: Comparing awq_op_ref.py vs gw_awq_asym_l2.py outputs")
    print("="*80)

    # Paths
    heuristic_path = "./quantized_models/awq_heuristic"
    standard_path = "./quantized_models/minicpm_gw_awq_asym_l2"

    # Alternative paths that might exist
    alt_heuristic = "./quantized_models/minicpm_awq_op_ref"

    print(f"\nChecking standard paths...")
    print(f"  Heuristic: {heuristic_path} - Exists: {os.path.exists(heuristic_path)}")
    print(f"  Standard:  {standard_path} - Exists: {os.path.exists(standard_path)}")
    print(f"  Alt Heuristic: {alt_heuristic} - Exists: {os.path.exists(alt_heuristic)}")

    # Load models
    model_heuristic = load_and_analyze_model(
        heuristic_path if os.path.exists(heuristic_path) else alt_heuristic,
        "Heuristic AWQ (awq_op_ref.py)"
    )

    model_standard = load_and_analyze_model(
        standard_path,
        "Standard AWQ (gw_awq_asym_l2.py)"
    )

    # Compare
    if model_heuristic and model_standard:
        compare_models(
            model_heuristic,
            model_standard,
            "Heuristic AWQ",
            "Standard AWQ"
        )

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("""
If models are IDENTICAL or VIRTUALLY IDENTICAL:
  → The perplexity difference must come from evaluation differences
  → Check: dtype loading (bfloat16 vs float16), random seeds, or data preprocessing

If models are SLIGHTLY DIFFERENT:
  → Check the +1e-6 bias in awq_op_ref.py line 296
  → Check bias handling in grid search

If models are SIGNIFICANTLY DIFFERENT:
  → One model may not have been quantized correctly
  → Check if use_heuristic parameter was set correctly
  → Check if different calibration data was used
  → Check if different random seeds were used

KNOWN ISSUES IN awq_op_ref.py:
1. Line 296: activation_salience + 1e-6 (should use clamp instead)
2. Line 288: Y_orig ignores bias (should include bias for layers that have it)
3. Line 158 in compare_awq_heuristic.py: Loads as float16 but models saved as bfloat16
""")

if __name__ == "__main__":
    main()
