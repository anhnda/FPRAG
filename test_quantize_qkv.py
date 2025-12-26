"""
Test INT4 Weight-Only Quantization on GQA Attention (Group 0)

This script:
1. Loads James-Stein means as input X (activation vector)
2. Loads Q, K weights for group 0 (4 query heads sharing 1 KV head)
3. Quantizes weights to INT4 with group size 128 (nearest rounding)
4. Computes attention scores: (X @ Wq^T) · (X @ Wk^T) for each head
5. Compares original vs quantized attention scores
6. Reports quantization scales and errors

Usage:
    python test_quantize_qkv.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def quantize_weight_groupwise_int4(W, group_size=128):
    """
    Quantize weights to INT4 using group-wise asymmetric quantization [0, 15].

    Args:
        W: Weight matrix of shape [..., in_features]
        group_size: Size of each quantization group (default: 128)

    Returns:
        W_quant: Dequantized weights (same shape as W)
        scales: Per-group scales, shape [..., n_groups]
        zp: Per-group zero points, shape [..., n_groups]
        W_int: Integer weights, shape [..., in_features]
    """
    original_shape = W.shape

    # Flatten to 2D if needed
    if W.ndim > 2:
        # Reshape keeping last dimension
        W_flat = W.reshape(-1, W.shape[-1])
    else:
        W_flat = W.copy()

    out_features, in_features = W_flat.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    # Pad if needed
    if padded_in > in_features:
        W_padded = np.zeros((out_features, padded_in), dtype=W.dtype)
        W_padded[:, :in_features] = W_flat
    else:
        W_padded = W_flat

    # Reshape to groups: [out_features, n_groups, group_size]
    W_grouped = W_padded.reshape(out_features, n_groups, group_size)

    # Asymmetric quantization [0, 15] (INT4)
    w_min = W_grouped.min(axis=2, keepdims=True)
    w_max = W_grouped.max(axis=2, keepdims=True)
    max_int = 15

    # Compute scale and zero point
    scale = (w_max - w_min) / max_int
    scale = np.maximum(scale, 1e-8)  # Avoid division by zero
    zp = np.round(-w_min / scale).clip(0, max_int)

    # Quantize with NEAREST rounding
    W_div = W_grouped / scale
    W_int = np.round(W_div + zp).clip(0, max_int)

    # Dequantize
    W_quant_grouped = (W_int - zp) * scale

    # Reshape back to flat
    W_quant_flat = W_quant_grouped.reshape(out_features, padded_in)
    W_int_flat = W_int.reshape(out_features, padded_in)

    # Remove padding
    if padded_in > in_features:
        W_quant_flat = W_quant_flat[:, :in_features]
        W_int_flat = W_int_flat[:, :in_features]

    # Reshape back to original shape
    W_quant = W_quant_flat.reshape(original_shape)
    W_int_final = W_int_flat.reshape(original_shape)

    # Flatten scales and zp for output
    scales_flat = scale.reshape(out_features, n_groups)
    zp_flat = zp.reshape(out_features, n_groups)

    # If original was 3D, reshape scales/zp accordingly
    if len(original_shape) == 3:
        scales_out = scales_flat.reshape(original_shape[0], n_groups)
        zp_out = zp_flat.reshape(original_shape[0], n_groups)
    else:
        scales_out = scales_flat
        zp_out = zp_flat

    return W_quant, scales_out, zp_out, W_int_final


def compute_quantization_error(W_orig, W_quant):
    """Compute quantization error metrics."""
    diff = W_quant - W_orig
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))
    rel_error = mae / (np.mean(np.abs(W_orig)) + 1e-10) * 100

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'rel_error_pct': rel_error
    }


def main():
    print("="*70)
    print("INT4 Weight Quantization for GQA Attention (Group 0)")
    print("="*70)

    # Load data
    print("\n[1] Loading data...")
    try:
        js_means = np.load('js_means.npy')  # [4096]
        Wq = np.load('Wq_group0.npy')  # [4, 128, 4096]
        Wk = np.load('Wk_group0.npy')  # [128, 4096]
        Wv = np.load('Wv_group0.npy')  # [128, 4096]
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Please ensure you've run xspot.py and have the following files:")
        print("  - js_means.npy")
        print("  - Wq_group0.npy")
        print("  - Wk_group0.npy")
        print("  - Wv_group0.npy")
        return

    print(f"  JS means: {js_means.shape}")
    print(f"  Wq (group 0): {Wq.shape} [num_heads, head_dim, hidden_size]")
    print(f"  Wk (group 0): {Wk.shape} [head_dim, hidden_size]")
    print(f"  Wv (group 0): {Wv.shape} [head_dim, hidden_size]")

    # Use JS means as input X (single activation vector)
    X = js_means  # [4096]
    print(f"\n  Using JS means as input X: {X.shape}")
    print(f"  X statistics: min={X.min():.6f}, max={X.max():.6f}, "
          f"mean={X.mean():.6f}, std={X.std():.6f}")

    # Quantize weights
    print("\n[2] Quantizing weights to INT4 (group_size=128, nearest rounding)...")

    # Quantize Wq (4 query heads)
    print("  Quantizing Wq (4 heads)...")
    Wq_quant, Wq_scales, Wq_zp, Wq_int = quantize_weight_groupwise_int4(Wq, group_size=128)

    # Quantize Wk (1 KV head)
    print("  Quantizing Wk (1 head)...")
    Wk_quant, Wk_scales, Wk_zp, Wk_int = quantize_weight_groupwise_int4(Wk, group_size=128)

    print(f"\n  Wq scales: {Wq_scales.shape} [num_heads, n_groups]")
    print(f"  Wk scales: {Wk_scales.shape} [1, n_groups]")

    # Compute weight quantization errors
    print("\n[3] Weight quantization errors:")
    for head_idx in range(4):
        err = compute_quantization_error(Wq[head_idx], Wq_quant[head_idx])
        print(f"  Wq head {head_idx}: MAE={err['mae']:.6f}, "
              f"Max={err['max_error']:.6f}, Rel={err['rel_error_pct']:.4f}%")

    err_k = compute_quantization_error(Wk, Wk_quant)
    print(f"  Wk: MAE={err_k['mae']:.6f}, "
          f"Max={err_k['max_error']:.6f}, Rel={err_k['rel_error_pct']:.4f}%")

    # Compute attention scores
    print("\n[4] Computing attention scores: (X @ Wq^T) · (X @ Wk^T)")
    print("="*70)

    num_heads = 4
    results = []

    for head_idx in range(num_heads):
        print(f"\n--- Query Head {head_idx} ---")

        # Original computation
        # Q = X @ Wq^T: [4096] @ [4096, 128]^T = [128]
        Q_orig = X @ Wq[head_idx].T  # [128]

        # K = X @ Wk^T: [4096] @ [4096, 128]^T = [128]
        K_orig = X @ Wk.T  # [128] (shared across all heads)

        # Attention score = Q · K (dot product)
        score_orig = Q_orig @ K_orig  # scalar

        # Quantized computation
        Q_quant = X @ Wq_quant[head_idx].T
        K_quant = X @ Wk_quant.T
        score_quant = Q_quant @ K_quant

        # Error
        error = score_quant - score_orig
        rel_error = error / (np.abs(score_orig) + 1e-10) * 100

        print(f"Original score:     {score_orig:15.6f}")
        print(f"Quantized score:    {score_quant:15.6f}")
        print(f"Absolute error:     {error:15.6f}")
        print(f"Relative error:     {rel_error:15.4f}%")

        # Scale statistics for this head
        head_scales = Wq_scales[head_idx]  # [n_groups]
        print(f"\nWq scales for head {head_idx} ({len(head_scales)} groups):")
        print(f"  Mean:   {head_scales.mean():.8f}")
        print(f"  Median: {np.median(head_scales):.8f}")
        print(f"  Min:    {head_scales.min():.8f}")
        print(f"  Max:    {head_scales.max():.8f}")
        print(f"  Std:    {head_scales.std():.8f}")

        results.append({
            'head': head_idx,
            'score_orig': score_orig,
            'score_quant': score_quant,
            'error': error,
            'rel_error': rel_error,
            'wq_scales': head_scales,
            'wq_scale_mean': head_scales.mean(),
            'Q_orig': Q_orig,
            'Q_quant': Q_quant,
            'K_orig': K_orig,
            'K_quant': K_quant
        })

    # K scales (shared across all heads)
    print(f"\n{'='*70}")
    print(f"Wk scales (shared by all 4 query heads, {len(Wk_scales[0])} groups):")
    print(f"  Mean:   {Wk_scales.mean():.8f}")
    print(f"  Median: {np.median(Wk_scales):.8f}")
    print(f"  Min:    {Wk_scales.min():.8f}")
    print(f"  Max:    {Wk_scales.max():.8f}")
    print(f"  Std:    {Wk_scales.std():.8f}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Attention Score Quantization Impact")
    print("="*70)
    print(f"{'Head':<6} {'Original':<15} {'Quantized':<15} {'Abs Error':<15} {'Rel Error %':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['head']:<6} {r['score_orig']:<15.6f} {r['score_quant']:<15.6f} "
              f"{r['error']:<15.6f} {r['rel_error']:<15.4f}")

    errors = [r['error'] for r in results]
    rel_errors = [r['rel_error'] for r in results]
    scale_means = [r['wq_scale_mean'] for r in results]

    print("-"*70)
    print(f"Mean absolute error:    {np.mean(np.abs(errors)):.6f}")
    print(f"Mean relative error:    {np.mean(np.abs(rel_errors)):.4f}%")
    print(f"Max absolute error:     {np.max(np.abs(errors)):.6f}")
    print(f"Max relative error:     {np.max(np.abs(rel_errors)):.4f}%")
    print(f"\nMean Wq scale per head: {scale_means}")
    print(f"Overall Wq scale mean:  {np.mean(scale_means):.8f}")
    print(f"Overall Wk scale mean:  {Wk_scales.mean():.8f}")
    print("="*70)

    # Visualization
    print("\n[5] Generating visualizations...")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Attention scores comparison
    ax1 = fig.add_subplot(gs[0, 0])
    heads = [r['head'] for r in results]
    scores_orig = [r['score_orig'] for r in results]
    scores_quant = [r['score_quant'] for r in results]

    x = np.arange(len(heads))
    width = 0.35
    ax1.bar(x - width/2, scores_orig, width, label='Original', alpha=0.8, color='blue')
    ax1.bar(x + width/2, scores_quant, width, label='Quantized', alpha=0.8, color='orange')
    ax1.set_xlabel('Query Head')
    ax1.set_ylabel('Attention Score (Q·K)')
    ax1.set_title('Original vs Quantized Attention Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'H{i}' for i in heads])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Absolute errors
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(heads, errors, alpha=0.7, color='red')
    ax2.set_xlabel('Query Head')
    ax2.set_ylabel('Error (Quantized - Original)')
    ax2.set_title('Absolute Quantization Error per Head')
    ax2.set_xticks(heads)
    ax2.set_xticklabels([f'H{i}' for i in heads])
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Relative errors
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(heads, rel_errors, alpha=0.7, color='purple')
    ax3.set_xlabel('Query Head')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Relative Quantization Error per Head')
    ax3.set_xticks(heads)
    ax3.set_xticklabels([f'H{i}' for i in heads])
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Mean scale per head
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(heads, scale_means, alpha=0.7, color='green')
    ax4.set_xlabel('Query Head')
    ax4.set_ylabel('Mean Scale Value')
    ax4.set_title('Mean Wq Quantization Scale per Head')
    ax4.set_xticks(heads)
    ax4.set_xticklabels([f'H{i}' for i in heads])
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Scale distributions (boxplot)
    ax5 = fig.add_subplot(gs[1, 1])
    scale_data = [results[i]['wq_scales'] for i in range(num_heads)]
    bp = ax5.boxplot(scale_data, labels=[f'H{i}' for i in heads], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax5.set_xlabel('Query Head')
    ax5.set_ylabel('Scale Value')
    ax5.set_title('Wq Scale Distribution per Head')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Wk scale distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(Wk_scales.flatten(), bins=30, alpha=0.7, color='teal', edgecolor='black')
    ax6.axvline(Wk_scales.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {Wk_scales.mean():.6f}')
    ax6.set_xlabel('Scale Value')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Wk Scale Distribution ({len(Wk_scales.flatten())} groups)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Q vector comparison (head 0)
    ax7 = fig.add_subplot(gs[2, 0])
    Q0_orig = results[0]['Q_orig']
    Q0_quant = results[0]['Q_quant']
    ax7.plot(Q0_orig, label='Original', alpha=0.7, linewidth=1.5)
    ax7.plot(Q0_quant, label='Quantized', alpha=0.7, linewidth=1.5)
    ax7.set_xlabel('Dimension')
    ax7.set_ylabel('Value')
    ax7.set_title('Query Vector Q (Head 0): Original vs Quantized')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. K vector comparison (shared)
    ax8 = fig.add_subplot(gs[2, 1])
    K_orig = results[0]['K_orig']  # Same for all heads
    K_quant = results[0]['K_quant']
    ax8.plot(K_orig, label='Original', alpha=0.7, linewidth=1.5)
    ax8.plot(K_quant, label='Quantized', alpha=0.7, linewidth=1.5)
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('Value')
    ax8.set_title('Key Vector K (Shared): Original vs Quantized')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Q-K error scatter (head 0)
    ax9 = fig.add_subplot(gs[2, 2])
    Q_error = Q0_quant - Q0_orig
    K_error = K_quant - K_orig
    ax9.scatter(Q_error, K_error, alpha=0.5, s=10)
    ax9.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax9.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Q Error (Head 0)')
    ax9.set_ylabel('K Error')
    ax9.set_title('Q vs K Quantization Errors')
    ax9.grid(True, alpha=0.3)

    plt.savefig('attention_quantization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: attention_quantization_analysis.png")

    # Save detailed results
    print("\n[6] Saving detailed results...")
    np.savez('quantization_results.npz',
             # Input
             X=X,
             js_means=js_means,
             # Original weights
             Wq_orig=Wq,
             Wk_orig=Wk,
             # Quantized weights
             Wq_quant=Wq_quant,
             Wk_quant=Wk_quant,
             Wq_int=Wq_int,
             Wk_int=Wk_int,
             # Scales and zero points
             Wq_scales=Wq_scales,
             Wk_scales=Wk_scales,
             Wq_zp=Wq_zp,
             Wk_zp=Wk_zp,
             # Attention scores
             attention_scores_orig=np.array(scores_orig),
             attention_scores_quant=np.array(scores_quant),
             # Errors
             errors=np.array(errors),
             rel_errors=np.array(rel_errors),
             # Query and Key vectors
             Q_orig=[r['Q_orig'] for r in results],
             Q_quant=[r['Q_quant'] for r in results],
             K_orig=K_orig,
             K_quant=K_quant)
    print(f"  Saved: quantization_results.npz")

    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)
    print("\nKey findings:")
    print(f"  1. Mean attention score error: {np.mean(np.abs(errors)):.6f} "
          f"({np.mean(np.abs(rel_errors)):.4f}%)")
    print(f"  2. Mean Wq scale across 4 heads: {np.mean(scale_means):.8f}")
    print(f"  3. Mean Wk scale: {Wk_scales.mean():.8f}")
    print(f"  4. INT4 quantization (group_size=128) introduces "
          f"{np.mean(np.abs(rel_errors)):.2f}% error on average")


if __name__ == '__main__':
    main()
