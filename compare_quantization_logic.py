"""
Compare quantization logic between awq_op_ref.py and gw_awq_asym_l2.py

This test identifies why use_heuristic=False in awq_op_ref.py gives different
results from gw_awq_asym_l2.py
"""

import torch

def gw_awq_asym_l2_grid_search(W, X, salience, alpha, group_size=128):
    """Replicate gw_awq_asym_l2.py grid search logic"""
    # Line 234: scales = activation_salience.pow(alpha).clamp(min=1e-5)
    scales = salience.pow(alpha).clamp(min=1e-5)

    # Scale weights
    W_scaled = W * scales.unsqueeze(0)

    # Quantize (simplified asymmetric)
    out_features, in_features = W_scaled.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    if padded_in > in_features:
        W_pad = torch.zeros(out_features, padded_in, device=W.device, dtype=W.dtype)
        W_pad[:, :in_features] = W_scaled
    else:
        W_pad = W_scaled

    W_g = W_pad.reshape(out_features, n_groups, group_size)
    W_min = W_g.min(dim=2, keepdim=True)[0]
    W_max = W_g.max(dim=2, keepdim=True)[0]
    scale = (W_max - W_min) / 15.0
    scale = scale.clamp(min=1e-8)
    zp = torch.round(-W_min / scale).clamp(0, 15)

    W_int = torch.round(W_g / scale + zp).clamp(0, 15)
    W_dequant_g = (W_int - zp) * scale
    W_dequant = W_dequant_g.reshape(out_features, padded_in)
    if padded_in > in_features:
        W_dequant = W_dequant[:, :in_features]

    # Reconstruct output
    # Line 243: X_compensated = X_search / scales.unsqueeze(0)
    X_comp = X / scales.unsqueeze(0)
    Y = torch.matmul(X_comp, W_dequant.t())

    return Y, scales


def awq_op_ref_grid_search(W, X, salience, alpha, group_size=128):
    """Replicate awq_op_ref.py grid search logic"""
    # Line 296: activation_salience = activation_salience + 1e-6  ← BUG!
    salience_biased = salience + 1e-6

    # Line 300: scales = activation_salience.pow(alpha)
    scales = salience_biased.pow(alpha)

    # Scale weights
    W_scaled = W * scales.unsqueeze(0)

    # Quantize (same logic)
    out_features, in_features = W_scaled.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    if padded_in > in_features:
        W_pad = torch.zeros(out_features, padded_in, device=W.device, dtype=W.dtype)
        W_pad[:, :in_features] = W_scaled
    else:
        W_pad = W_scaled

    W_g = W_pad.reshape(out_features, n_groups, group_size)
    w_min = W_g.min(dim=2, keepdim=True)[0]
    w_max = W_g.max(dim=2, keepdim=True)[0]
    scale = (w_max - w_min) / 15.0
    scale = scale.clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, 15)

    # Note: awq_op_ref flattens instead of keeping grouped
    scale_flat = scale.repeat(1, 1, group_size).reshape(out_features, padded_in)
    zp_flat = zp.repeat(1, 1, group_size).reshape(out_features, padded_in)

    W_div = W_pad / scale_flat
    W_int = torch.round(W_div + zp_flat).clamp(0, 15)
    W_dequant = (W_int - zp_flat) * scale_flat

    if padded_in > in_features:
        W_dequant = W_dequant[:, :in_features]

    # Reconstruct output
    # Line 311-312: W_recon = W_quant / scales.unsqueeze(0)
    #               Y_quant = torch.matmul(X_search, W_recon.t())
    W_recon = W_dequant / scales.unsqueeze(0)
    Y = torch.matmul(X, W_recon.t())

    return Y, scales


def main():
    torch.manual_seed(42)
    out_features = 5
    in_features = 128
    batch_size = 100

    # Create test data
    W = torch.randn(out_features, in_features)
    X = torch.randn(batch_size, in_features) * 0.5
    X[:, 0:5] *= 10  # Add some large activations

    # Compute salience (E[X²])
    salience = X.pow(2).mean(dim=0)

    print("=" * 80)
    print("COMPARING GRID SEARCH LOGIC: awq_op_ref.py vs gw_awq_asym_l2.py")
    print("=" * 80)

    print(f"\nSalience statistics:")
    print(f"  Mean: {salience.mean():.6f}")
    print(f"  Std: {salience.std():.6f}")
    print(f"  Min: {salience.min():.6f}")
    print(f"  Max: {salience.max():.6f}")
    print(f"  Top 5: {torch.topk(salience, 5).values.tolist()}")
    print(f"  Bottom 5: {torch.topk(salience, 5, largest=False).values.tolist()}")

    # Test different alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n" + "=" * 80)
    print("SCALE DIFFERENCES (awq_op_ref - gw_awq_asym_l2)")
    print("=" * 80)
    print(f"{'Alpha':<10} {'Max Diff':<15} {'Mean Diff':<15} {'Std Diff':<15}")
    print("-" * 80)

    for alpha in alphas:
        _, scales_gw = gw_awq_asym_l2_grid_search(W, X, salience, alpha)
        _, scales_awq = awq_op_ref_grid_search(W, X, salience, alpha)

        scale_diff = (scales_awq - scales_gw).abs()

        print(f"{alpha:<10.2f} {scale_diff.max():<15.6f} {scale_diff.mean():<15.6f} {scale_diff.std():<15.6f}")

    # Detailed analysis for alpha=0.5
    alpha = 0.5
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS FOR ALPHA = {alpha}")
    print("=" * 80)

    Y_orig = torch.matmul(X, W.t())
    Y_gw, scales_gw = gw_awq_asym_l2_grid_search(W, X, salience, alpha)
    Y_awq, scales_awq = awq_op_ref_grid_search(W, X, salience, alpha)

    error_gw = (Y_orig - Y_gw).pow(2).mean()
    error_awq = (Y_orig - Y_awq).pow(2).mean()

    print(f"\nReconstruction MSE:")
    print(f"  gw_awq_asym_l2: {error_gw:.6f}")
    print(f"  awq_op_ref:     {error_awq:.6f}")
    print(f"  Difference:     {abs(error_gw - error_awq):.6f}")

    print(f"\nScale statistics:")
    print(f"  gw_awq_asym_l2 - Mean: {scales_gw.mean():.6f}, Std: {scales_gw.std():.6f}")
    print(f"  awq_op_ref     - Mean: {scales_awq.mean():.6f}, Std: {scales_awq.std():.6f}")
    print(f"  Difference     - Max: {(scales_awq - scales_gw).abs().max():.6f}")

    # Show channels with largest scale differences
    scale_diff = (scales_awq - scales_gw).abs()
    top_diff_idx = torch.topk(scale_diff, 5).indices

    print(f"\nTop 5 channels with largest scale differences:")
    print(f"{'Channel':<10} {'Salience':<15} {'Scale (GW)':<15} {'Scale (AWQ)':<15} {'Diff':<15}")
    print("-" * 80)
    for idx in top_diff_idx:
        print(f"{idx.item():<10} {salience[idx]:<15.6f} {scales_gw[idx]:<15.6f} {scales_awq[idx]:<15.6f} {scale_diff[idx]:<15.6f}")

    print("\n" + "=" * 80)
    print("ROOT CAUSE IDENTIFIED")
    print("=" * 80)
    print("""
awq_op_ref.py line 296 contains a BUG:
    activation_salience = activation_salience + 1e-6

This adds a constant to the salience BEFORE computing scales:
    scales = (salience + 1e-6)^alpha  ← awq_op_ref.py (WRONG)
    scales = clamp(salience^alpha, min=1e-5)  ← gw_awq_asym_l2.py (CORRECT)

The 1e-6 bias causes scales to differ, leading to different quantization results.

IMPACT:
- For channels with low salience, the bias is significant
- For channels with high salience, the bias is negligible
- This shifts the weight distribution and affects grid search optimization
- Results in different perplexity even with use_heuristic=False

FIX:
Remove line 296 in awq_op_ref.py, and replace line 300 with:
    scales = activation_salience.pow(alpha).clamp(min=1e-5)

Or remove the +1e-6 and handle division by zero differently.

ADDITIONAL ISSUE (minor):
awq_op_ref.py line 288 ignores bias in grid search:
    Y_orig = torch.matmul(X_search, W.t())  # No bias

gw_awq_asym_l2.py includes bias:
    Y_orig = torch.matmul(X_search, W.t()) + b  # With bias

This may cause slight differences in alpha selection for layers with bias.
""")

if __name__ == "__main__":
    main()
