"""
Test to understand why outlier_percent=0.999 differs from apply_heuristic=False

The key insight: With outlier_percent=0.999, the heuristic is still ACTIVE,
but it can only flip the SMALLEST 0.1% of activations. This introduces
unnecessary changes because:
1. Small activations contribute little to the output x·w
2. Flipping them doesn't reduce error effectively
3. It still differs from apply_heuristic=False (which flips NOTHING)
"""

import torch
import numpy as np

def quantize_with_heuristic(w, x, outlier_percent=0.05, apply_heuristic=True):
    """Simplified version of the quantization logic"""

    # 1. Initial quantization (nearest rounding)
    scale = w.abs().max() / 7  # 4-bit: [-7, 7]
    w_div = w / scale
    w_int = torch.round(w_div).clamp(-7, 7)
    w_quant = w_int * scale

    if not apply_heuristic:
        print(f"  No heuristic: returning initial quantization")
        return w_quant, 0  # 0 flips

    # 2. Compute error
    current_error = torch.dot(x, w - w_quant)

    if abs(current_error) < 1e-6:
        return w_quant, 0

    # 3. Identify flip candidates
    target_sign = torch.sign(current_error)
    flip_dir = torch.sign(w_div - w_int)
    flip_dir[flip_dir == 0] = 1.0
    flip_impacts = x * flip_dir * scale

    # 4. Outlier masking (THIS IS THE KEY)
    d = len(x)
    k_outliers = int(d * outlier_percent)
    outlier_mask = torch.zeros(d, dtype=torch.bool)

    if k_outliers > 0:
        # Mask the TOP k_outliers by activation magnitude
        outlier_mask[torch.topk(x.abs(), k_outliers).indices] = True

    # Valid mask: not outlier AND sign matches AND in range
    valid_mask = (~outlier_mask) & (torch.sign(flip_impacts) == target_sign)
    w_int_prop = w_int + flip_dir
    in_range = (w_int_prop >= -7) & (w_int_prop <= 7)
    valid_mask = valid_mask & in_range

    print(f"  Outlier percent: {outlier_percent*100:.1f}%")
    print(f"  Channels masked as outliers: {outlier_mask.sum().item()}/{d}")
    print(f"  Channels available for flipping: {valid_mask.sum().item()}/{d}")

    if not valid_mask.any():
        print(f"  No valid flip candidates!")
        return w_quant, 0

    # 5. Sort and optimize
    valid_indices = torch.nonzero(valid_mask).squeeze()
    if valid_indices.ndim == 0:
        valid_indices = valid_indices.unsqueeze(0)

    candidate_costs = (w_div - w_int).abs()[valid_indices]
    sorted_indices_local = torch.argsort(candidate_costs, descending=True)
    sorted_indices = valid_indices[sorted_indices_local]
    sorted_impacts = flip_impacts[sorted_indices]

    cumsum = torch.cumsum(sorted_impacts, dim=0)
    best_k = torch.argmin(torch.abs(current_error - cumsum))

    print(f"  Best K flips: {best_k.item()}")

    if best_k == 0:
        return w_quant, 0

    # 6. Apply flips
    indices_to_flip = sorted_indices[:best_k]
    w_int[indices_to_flip] += flip_dir[indices_to_flip].long()

    return w_int * scale, best_k.item()


def main():
    torch.manual_seed(42)
    d = 100

    # Create synthetic data
    # x: activations with some large outliers
    x = torch.randn(d) * 0.5
    x[0:5] = torch.randn(5) * 5.0  # Add some large outliers

    # w: weights
    w = torch.randn(d)

    print("=" * 80)
    print("COMPARING DIFFERENT OUTLIER PERCENTAGES")
    print("=" * 80)
    print(f"\nActivation statistics:")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}")
    print(f"  Top 5 magnitudes: {torch.topk(x.abs(), 5).values.tolist()}")
    print(f"  Bottom 5 magnitudes: {torch.topk(x.abs(), 5, largest=False).values.tolist()}")

    # Test different configurations
    configs = [
        ("No heuristic", False, 0.05),
        ("Heuristic with outlier_percent=0.05", True, 0.05),
        ("Heuristic with outlier_percent=0.50", True, 0.50),
        ("Heuristic with outlier_percent=0.90", True, 0.90),
        ("Heuristic with outlier_percent=0.999", True, 0.999),
    ]

    original_output = torch.dot(x, w)
    print(f"\nOriginal output (x·w): {original_output:.6f}")
    print()

    results = []

    for name, use_heuristic, outlier_pct in configs:
        print(f"\n{name}:")
        print("-" * 80)

        w_quant, num_flips = quantize_with_heuristic(
            w.clone(), x,
            outlier_percent=outlier_pct,
            apply_heuristic=use_heuristic
        )

        quant_output = torch.dot(x, w_quant)
        error = original_output - quant_output
        mse = (w - w_quant).pow(2).mean()

        print(f"  Quantized output: {quant_output:.6f}")
        print(f"  Error: {error:.6f}")
        print(f"  Weight MSE: {mse:.6f}")
        print(f"  Total flips applied: {num_flips}")

        results.append({
            'name': name,
            'error': error.item(),
            'mse': mse.item(),
            'num_flips': num_flips
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<50} {'Error':>12} {'MSE':>12} {'Flips':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<50} {r['error']:>12.6f} {r['mse']:>12.6f} {r['num_flips']:>8}")

    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("""
With outlier_percent=0.999, the heuristic is still ACTIVE but restricted to
flipping only the SMALLEST 0.1% of activations. This is counterproductive because:

1. Small activations contribute little to the output (x·w)
2. Flipping them has minimal impact on error reduction
3. The algorithm still introduces changes (flips channels)
4. These changes don't provide the intended error reduction benefit
5. Results differ from apply_heuristic=False (which flips NOTHING)

The outlier masking is designed to EXCLUDE the largest activations (which matter most)
to prevent overfitting. But with 0.999, we exclude almost everything, leaving only
the irrelevant channels that don't help optimization.

EXPECTED BEHAVIOR:
- outlier_percent=0.05: Exclude top 5%, flip from remaining 95% (normal operation)
- outlier_percent=0.999: Exclude top 99.9%, flip from bottom 0.1% (broken operation)
- apply_heuristic=False: No flips at all (baseline)

The 0.999 case is DIFFERENT from the baseline because it still flips channels,
just the WRONG ones that don't matter.
""")

if __name__ == "__main__":
    main()
