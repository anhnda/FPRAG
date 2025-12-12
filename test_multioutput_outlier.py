"""
Test multi-output case to see why outlier_percent=0.999 differs from apply_heuristic=False

The key difference: In the actual code, we have MULTIPLE output channels,
and each output channel independently decides which input channels to flip.
"""

import torch

def quantize_weight_matrix(W, x, outlier_percent=0.05, apply_heuristic=True):
    """Simplified version matching awq_op_ref.py structure"""
    out_features, in_features = W.shape

    # 1. Asymmetric quantization setup (simplified)
    scale = (W.max(dim=1, keepdim=True)[0] - W.min(dim=1, keepdim=True)[0]) / 15
    scale = scale.clamp(min=1e-8)
    w_min = W.min(dim=1, keepdim=True)[0]
    zp = torch.round(-w_min / scale).clamp(0, 15)

    # 2. Initial quantization
    W_div = W / scale
    W_int = torch.round(W_div + zp).clamp(0, 15)
    W_quant = (W_int - zp) * scale

    if not apply_heuristic:
        return W_quant, 0

    # 3. Compute error PER OUTPUT CHANNEL
    W_diff = W - W_quant
    current_error = (W_diff * x.unsqueeze(0)).sum(dim=1)  # [out_features]

    # 4. Identify flip candidates
    flip_dir = torch.sign(W_div + zp - W_int)
    flip_dir[flip_dir == 0] = 1.0
    flip_impacts = x.unsqueeze(0) * flip_dir * scale  # [out, in]

    # 5. CRITICAL: Outlier masking (same mask for ALL output channels)
    k_outliers = int(in_features * outlier_percent)
    is_outlier = torch.zeros(in_features, dtype=torch.bool)
    if k_outliers > 0:
        _, outlier_indices = torch.topk(x.abs(), k_outliers)
        is_outlier[outlier_indices] = True

    # 6. Validity mask
    target_sign = torch.sign(current_error).unsqueeze(1)  # [out, 1]
    valid_mask = (torch.sign(flip_impacts) == target_sign)
    valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

    # Range check
    w_int_prop = W_int + flip_dir
    in_range = (w_int_prop >= 0) & (w_int_prop <= 15)
    valid_mask = valid_mask & in_range

    # 7. Sort and optimize PER OUTPUT CHANNEL
    rounding_costs = (W_div + zp - W_int).abs()
    rounding_costs_masked = rounding_costs.clone()
    rounding_costs_masked[~valid_mask] = -1.0

    sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
    sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
    sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
    sorted_impacts = sorted_impacts * sorted_validity

    cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
    residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
    error_unsqueezed = torch.abs(current_error).unsqueeze(1)
    all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
    best_k = torch.argmin(all_residuals, dim=1)

    # 8. Apply flips
    total_flips = 0
    idx_range = torch.arange(in_features).unsqueeze(0)
    flip_mask_sorted = idx_range < best_k.unsqueeze(1)
    final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())

    sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
    sorted_flip_dir[~final_flips_sorted] = 0.0
    W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)

    total_flips = (sorted_flip_dir != 0).sum().item()

    W_dequant = (W_int - zp) * scale
    return W_dequant, total_flips


def main():
    torch.manual_seed(42)
    out_features = 10
    in_features = 100

    # Create activations with some large outliers
    x = torch.randn(in_features) * 0.5
    x[0:5] = torch.randn(5) * 5.0  # Large outliers

    # Create weight matrix
    W = torch.randn(out_features, in_features)

    print("=" * 80)
    print("MULTI-OUTPUT QUANTIZATION TEST")
    print("=" * 80)
    print(f"\nMatrix shape: {out_features} output × {in_features} input channels")
    print(f"\nActivation statistics:")
    print(f"  Top 5 magnitudes: {torch.topk(x.abs(), 5).values.tolist()}")
    print(f"  Bottom 5 magnitudes: {torch.topk(x.abs(), 5, largest=False).values.tolist()}")

    # Original output
    Y_orig = torch.matmul(W, x)
    print(f"\nOriginal output shape: {Y_orig.shape}")
    print(f"Original output mean: {Y_orig.mean():.6f}, std: {Y_orig.std():.6f}")

    configs = [
        ("No heuristic", False, 0.05),
        ("Heuristic outlier=0.05", True, 0.05),
        ("Heuristic outlier=0.50", True, 0.50),
        ("Heuristic outlier=0.90", True, 0.90),
        ("Heuristic outlier=0.999", True, 0.999),
    ]

    results = []

    for name, use_heuristic, outlier_pct in configs:
        print(f"\n{name}:")
        print("-" * 80)

        W_quant, num_flips = quantize_weight_matrix(
            W.clone(), x,
            outlier_percent=outlier_pct,
            apply_heuristic=use_heuristic
        )

        Y_quant = torch.matmul(W_quant, x)
        error = (Y_orig - Y_quant).abs().mean()
        weight_mse = (W - W_quant).pow(2).mean()
        weight_diff = (W - W_quant).abs().sum()

        print(f"  Output error (MAE): {error:.6f}")
        print(f"  Weight MSE: {weight_mse:.6f}")
        print(f"  Weight diff (L1): {weight_diff:.6f}")
        print(f"  Total flips: {num_flips}")

        results.append({
            'name': name,
            'error': error.item(),
            'mse': weight_mse.item(),
            'diff': weight_diff.item(),
            'flips': num_flips
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} {'Output Error':>15} {'Weight MSE':>15} {'Weight L1':>15} {'Flips':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<35} {r['error']:>15.6f} {r['mse']:>15.6f} {r['diff']:>15.6f} {r['flips']:>10}")

    # Compare specific cases
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)

    baseline = results[0]
    outlier999 = results[4]

    print(f"\nNo heuristic vs Heuristic outlier=0.999:")
    print(f"  Output error difference: {abs(baseline['error'] - outlier999['error']):.6f}")
    print(f"  Weight MSE difference: {abs(baseline['mse'] - outlier999['mse']):.6f}")
    print(f"  Weight L1 difference: {abs(baseline['diff'] - outlier999['diff']):.6f}")
    print(f"  Flip count difference: {abs(baseline['flips'] - outlier999['flips'])}")

    if abs(baseline['diff'] - outlier999['diff']) < 1e-6:
        print("\n✅ Results are IDENTICAL (as expected if no flips occur)")
    else:
        print("\n⚠️  Results are DIFFERENT (heuristic made changes even with 99.9% outliers)")
        print("\nREASON: Even with 99.9% outliers, some output channels found it beneficial")
        print("to flip the remaining 0.1% smallest input channels. This changes the weights")
        print("even though it may not improve (or may worsen) the actual output quality.")

    print("\n" + "=" * 80)
    print("EXPLANATION")
    print("=" * 80)
    print("""
The heuristic algorithm operates PER OUTPUT CHANNEL:
1. Each output channel has its own error to minimize
2. Each output channel independently decides which input channels to flip
3. The outlier mask is SHARED across all output channels
4. With outlier_percent=0.999, only the smallest 0.1% input channels can be flipped

Even if these small channels contribute little to the output, the algorithm
may still decide to flip them if it mathematically reduces the per-output error.
This is why results differ from apply_heuristic=False, which flips nothing.

The key insight: The heuristic is NOT disabled by outlier_percent=0.999.
It's just restricted to a bad set of candidates (smallest activations).
""")

if __name__ == "__main__":
    main()
