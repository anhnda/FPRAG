"""
Comparison Script: E[X²] vs E[1+X²] Salience for AWQ Quantization

This script compares two activation salience metrics for Group-Wise AWQ:
1. Standard L2: E[X²]
2. Baseline-shifted L2: E[1+X²]

Comparison Metrics:
- Per-layer optimal α values
- Per-layer reconstruction errors
- Final perplexity on WikiText-2
- Salience distribution statistics
- Model size and compression ratio
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import argparse
from scipy import stats

# Import both quantizers
from gw_awq_asym_l2 import GroupWiseAWQAsymmetricL2Quantizer, load_wikitext2
from gw_awq_asym_1l2 import GroupWiseAWQAsymmetric1L2Quantizer


def compute_perplexity(model, tokenizer, texts, device, max_samples=500):
    """
    Compute perplexity on a given set of texts.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        texts: List of text strings
        device: Device to run on
        max_samples: Maximum number of samples to evaluate

    Returns:
        perplexity: Float
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    print(f"\nComputing perplexity on {min(max_samples, len(texts))} samples...")

    with torch.no_grad():
        for text in tqdm(texts[:max_samples], desc="Evaluating perplexity"):
            if len(text.strip()) == 0:
                continue

            try:
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=False
                ).to(device)

                if inputs.input_ids.shape[1] < 2:
                    continue

                outputs = model(**inputs, use_cache=False, labels=inputs.input_ids)
                loss = outputs.loss

                total_loss += loss.item() * inputs.input_ids.shape[1]
                total_tokens += inputs.input_ids.shape[1]

            except Exception as e:
                continue

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def compare_salience_distributions(quantizer_l2, quantizer_1l2, output_dir):
    """
    Compare the salience distributions between E[X²] and E[1+X²].

    Creates visualizations showing:
    - Histogram of salience values
    - Ratio of E[1+X²] / E[X²]
    - Channels with zero salience in E[X²]
    """
    print("\nComparing salience distributions...")
    os.makedirs(output_dir, exist_ok=True)

    # Collect salience values from both methods
    saliences_l2 = []
    saliences_1l2 = []
    layer_names = []

    for name in quantizer_l2.activation_data.keys():
        sal_l2 = quantizer_l2.get_activation_salience_l2(name)
        sal_1l2 = quantizer_1l2.get_activation_salience_1l2(name)

        if sal_l2 is not None and sal_1l2 is not None:
            saliences_l2.append(sal_l2.numpy())
            saliences_1l2.append(sal_1l2.numpy())
            layer_names.append(name)

    if len(saliences_l2) == 0:
        print("⚠️  No salience data to compare")
        return

    # Concatenate all salience values
    all_sal_l2 = np.concatenate(saliences_l2)
    all_sal_1l2 = np.concatenate(saliences_1l2)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram comparison
    axes[0, 0].hist(all_sal_l2, bins=100, alpha=0.6, label='E[X²]', color='blue')
    axes[0, 0].hist(all_sal_1l2, bins=100, alpha=0.6, label='E[1+X²]', color='red')
    axes[0, 0].set_xlabel('Salience Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Salience Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # Plot 2: Log-scale histogram
    axes[0, 1].hist(np.log10(all_sal_l2 + 1e-10), bins=100, alpha=0.6, label='log10(E[X²])', color='blue')
    axes[0, 1].hist(np.log10(all_sal_1l2), bins=100, alpha=0.6, label='log10(E[1+X²])', color='red')
    axes[0, 1].set_xlabel('log10(Salience Value)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Log-Scale Salience Distribution')
    axes[0, 1].legend()

    # Plot 3: Ratio E[1+X²] / E[X²]
    ratio = all_sal_1l2 / (all_sal_l2 + 1e-10)
    axes[1, 0].hist(ratio, bins=100, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Ratio: E[1+X²] / E[X²]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Salience Ratio Distribution')
    axes[1, 0].axvline(ratio.mean(), color='red', linestyle='--', label=f'Mean: {ratio.mean():.2f}')
    axes[1, 0].legend()

    # Plot 4: Zero salience analysis
    zero_threshold = 1e-6
    zero_count_l2 = np.sum(all_sal_l2 < zero_threshold)
    nonzero_count_l2 = np.sum(all_sal_l2 >= zero_threshold)
    zero_count_1l2 = np.sum(all_sal_1l2 < 1.0)  # Should be 0
    nonzero_count_1l2 = np.sum(all_sal_1l2 >= 1.0)  # Should be all

    x = ['E[X²]', 'E[1+X²]']
    zero_counts = [zero_count_l2, zero_count_1l2]
    nonzero_counts = [nonzero_count_l2, nonzero_count_1l2]

    axes[1, 1].bar(x, zero_counts, label='Near-zero (<1e-6)', color='red', alpha=0.7)
    axes[1, 1].bar(x, nonzero_counts, bottom=zero_counts, label='Non-zero', color='green', alpha=0.7)
    axes[1, 1].set_ylabel('Number of Channels')
    axes[1, 1].set_title('Zero vs Non-Zero Salience Channels')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'salience_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved salience comparison plot to {output_dir}/salience_comparison.png")

    # Print statistics
    print("\n" + "=" * 80)
    print("SALIENCE STATISTICS")
    print("=" * 80)
    print(f"\nE[X²] (Standard L2):")
    print(f"  Mean: {all_sal_l2.mean():.6f}")
    print(f"  Median: {np.median(all_sal_l2):.6f}")
    print(f"  Std: {all_sal_l2.std():.6f}")
    print(f"  Min: {all_sal_l2.min():.6e}")
    print(f"  Max: {all_sal_l2.max():.6f}")
    print(f"  Near-zero channels (< 1e-6): {zero_count_l2} / {len(all_sal_l2)} ({100*zero_count_l2/len(all_sal_l2):.2f}%)")

    print(f"\nE[1+X²] (Baseline-Shifted L2):")
    print(f"  Mean: {all_sal_1l2.mean():.6f}")
    print(f"  Median: {np.median(all_sal_1l2):.6f}")
    print(f"  Std: {all_sal_1l2.std():.6f}")
    print(f"  Min: {all_sal_1l2.min():.6f}")
    print(f"  Max: {all_sal_1l2.max():.6f}")
    print(f"  Near-zero channels (< 1.0): {zero_count_1l2} / {len(all_sal_1l2)} ({100*zero_count_1l2/len(all_sal_1l2):.2f}%)")

    print(f"\nRatio E[1+X²] / E[X²]:")
    print(f"  Mean: {ratio.mean():.6f}")
    print(f"  Median: {np.median(ratio):.6f}")
    print(f"  Min: {ratio.min():.6f}")
    print(f"  Max: {ratio.max():.6f}")
    print("=" * 80)


def compare_quantization_results(layer_scales_l2, layer_scales_1l2, output_dir):
    """
    Compare the quantization results (optimal α and errors) between methods.
    """
    print("\nComparing quantization results...")
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for common layers
    common_layers = set(layer_scales_l2.keys()) & set(layer_scales_1l2.keys())

    if len(common_layers) == 0:
        print("⚠️  No common layers to compare")
        return

    alphas_l2 = []
    alphas_1l2 = []
    errors_l2 = []
    errors_1l2 = []
    layer_names = []

    for layer in sorted(common_layers):
        alphas_l2.append(layer_scales_l2[layer]['alpha'])
        alphas_1l2.append(layer_scales_1l2[layer]['alpha'])
        errors_l2.append(layer_scales_l2[layer]['error'])
        errors_1l2.append(layer_scales_1l2[layer]['error'])
        layer_names.append(layer)

    alphas_l2 = np.array(alphas_l2)
    alphas_1l2 = np.array(alphas_1l2)
    errors_l2 = np.array(errors_l2)
    errors_1l2 = np.array(errors_1l2)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Alpha comparison
    axes[0, 0].scatter(alphas_l2, alphas_1l2, alpha=0.5)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 0].set_xlabel('α (E[X²])')
    axes[0, 0].set_ylabel('α (E[1+X²])')
    axes[0, 0].set_title('Optimal α Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Alpha distribution
    axes[0, 1].hist(alphas_l2, bins=20, alpha=0.6, label='E[X²]', color='blue')
    axes[0, 1].hist(alphas_1l2, bins=20, alpha=0.6, label='E[1+X²]', color='red')
    axes[0, 1].set_xlabel('Optimal α')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Optimal α Distribution')
    axes[0, 1].legend()

    # Plot 3: Error comparison (log scale)
    axes[1, 0].scatter(np.log10(errors_l2 + 1e-10), np.log10(errors_1l2 + 1e-10), alpha=0.5)
    min_val = min(np.log10(errors_l2 + 1e-10).min(), np.log10(errors_1l2 + 1e-10).min())
    max_val = max(np.log10(errors_l2 + 1e-10).max(), np.log10(errors_1l2 + 1e-10).max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    axes[1, 0].set_xlabel('log10(Error) [E[X²]]')
    axes[1, 0].set_ylabel('log10(Error) [E[1+X²]]')
    axes[1, 0].set_title('Reconstruction Error Comparison (log scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Error improvement
    error_ratio = errors_1l2 / (errors_l2 + 1e-10)
    axes[1, 1].hist(error_ratio, bins=50, color='purple', alpha=0.7)
    axes[1, 1].axvline(1.0, color='red', linestyle='--', label='No change')
    axes[1, 1].axvline(error_ratio.mean(), color='green', linestyle='--',
                       label=f'Mean: {error_ratio.mean():.3f}')
    axes[1, 1].set_xlabel('Error Ratio: E[1+X²] / E[X²]')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Improvement (< 1.0 = better)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantization_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved quantization comparison plot to {output_dir}/quantization_comparison.png")

    # Statistical tests
    alpha_diff = alphas_1l2 - alphas_l2
    error_ratio = errors_1l2 / (errors_l2 + 1e-10)

    print("\n" + "=" * 80)
    print("QUANTIZATION RESULTS COMPARISON")
    print("=" * 80)

    print(f"\nOptimal α Statistics:")
    print(f"  E[X²]     - Mean: {alphas_l2.mean():.3f}, Median: {np.median(alphas_l2):.3f}, Std: {alphas_l2.std():.3f}")
    print(f"  E[1+X²]   - Mean: {alphas_1l2.mean():.3f}, Median: {np.median(alphas_1l2):.3f}, Std: {alphas_1l2.std():.3f}")
    print(f"  Difference: Mean: {alpha_diff.mean():.3f}, Median: {np.median(alpha_diff):.3f}")

    # Paired t-test for alpha
    t_stat_alpha, p_val_alpha = stats.ttest_rel(alphas_l2, alphas_1l2)
    print(f"  Paired t-test: t={t_stat_alpha:.3f}, p={p_val_alpha:.6f}")
    if p_val_alpha < 0.05:
        print(f"  → Statistically significant difference (p < 0.05)")
    else:
        print(f"  → No significant difference (p >= 0.05)")

    print(f"\nReconstruction Error Statistics:")
    print(f"  E[X²]     - Mean: {errors_l2.mean():.6e}, Median: {np.median(errors_l2):.6e}")
    print(f"  E[1+X²]   - Mean: {errors_1l2.mean():.6e}, Median: {np.median(errors_1l2):.6e}")
    print(f"  Error Ratio (E[1+X²] / E[X²]):")
    print(f"    Mean: {error_ratio.mean():.3f}")
    print(f"    Median: {np.median(error_ratio):.3f}")

    better_count = np.sum(error_ratio < 1.0)
    worse_count = np.sum(error_ratio > 1.0)
    same_count = np.sum(error_ratio == 1.0)

    print(f"  E[1+X²] is better: {better_count}/{len(error_ratio)} layers ({100*better_count/len(error_ratio):.1f}%)")
    print(f"  E[1+X²] is worse:  {worse_count}/{len(error_ratio)} layers ({100*worse_count/len(error_ratio):.1f}%)")
    print(f"  Same:              {same_count}/{len(error_ratio)} layers ({100*same_count/len(error_ratio):.1f}%)")

    # Wilcoxon signed-rank test for errors
    w_stat, p_val_wilcoxon = stats.wilcoxon(errors_l2, errors_1l2)
    print(f"  Wilcoxon signed-rank test: W={w_stat:.1f}, p={p_val_wilcoxon:.6f}")
    if p_val_wilcoxon < 0.05:
        print(f"  → Statistically significant difference (p < 0.05)")
    else:
        print(f"  → No significant difference (p >= 0.05)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare E[X²] vs E[1+X²] salience for AWQ quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-eval", type=int, default=500, help="Evaluation samples for perplexity")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./comparison_results_l2_vs_1l2",
                       help="Output directory for comparison results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("COMPARISON: E[X²] vs E[1+X²] Salience for AWQ Quantization")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Evaluation samples: {args.n_eval}")
    print(f"Grid search points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load calibration and evaluation data
    print("\nLoading datasets...")
    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)
    eval_texts = load_wikitext2(split="validation", n_samples=args.n_eval)

    # Load tokenizer (shared)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ========== Method 1: E[X²] ==========
    print("\n" + "=" * 80)
    print("METHOD 1: E[X²] (Standard L2 Salience)")
    print("=" * 80)

    print("\nLoading model for E[X²] method...")
    model_l2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    quantizer_l2 = GroupWiseAWQAsymmetricL2Quantizer(
        model=model_l2,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size
    )

    quantizer_l2.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer_l2.quantize_model()

    print("\nEvaluating E[X²] quantized model...")
    ppl_l2 = compute_perplexity(model_l2, tokenizer, eval_texts, device, max_samples=args.n_eval)
    print(f"Perplexity (E[X²]): {ppl_l2:.2f}")

    # ========== Method 2: E[1+X²] ==========
    print("\n" + "=" * 80)
    print("METHOD 2: E[1+X²] (Baseline-Shifted L2 Salience)")
    print("=" * 80)

    print("\nLoading model for E[1+X²] method...")
    model_1l2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    quantizer_1l2 = GroupWiseAWQAsymmetric1L2Quantizer(
        model=model_1l2,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size
    )

    quantizer_1l2.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer_1l2.quantize_model()

    print("\nEvaluating E[1+X²] quantized model...")
    ppl_1l2 = compute_perplexity(model_1l2, tokenizer, eval_texts, device, max_samples=args.n_eval)
    print(f"Perplexity (E[1+X²]): {ppl_1l2:.2f}")

    # ========== Comparison ==========
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(f"\nPerplexity Results:")
    print(f"  E[X²]:     {ppl_l2:.2f}")
    print(f"  E[1+X²]:   {ppl_1l2:.2f}")
    print(f"  Difference: {ppl_1l2 - ppl_l2:+.2f}")

    if ppl_1l2 < ppl_l2:
        improvement = (ppl_l2 - ppl_1l2) / ppl_l2 * 100
        print(f"  → E[1+X²] is BETTER by {improvement:.2f}%")
    elif ppl_1l2 > ppl_l2:
        degradation = (ppl_1l2 - ppl_l2) / ppl_l2 * 100
        print(f"  → E[1+X²] is WORSE by {degradation:.2f}%")
    else:
        print(f"  → Both methods perform equally")

    # Create comparison visualizations
    os.makedirs(args.output_dir, exist_ok=True)

    # Before quantization, we need to recalibrate to get salience data
    print("\nRecalibrating models for salience comparison...")
    quantizer_l2.activation_data = {}
    quantizer_l2.calibrate(calib_texts[:50], n_samples=50)  # Quick recalibration

    quantizer_1l2.activation_data = {}
    quantizer_1l2.calibrate(calib_texts[:50], n_samples=50)  # Quick recalibration

    compare_salience_distributions(quantizer_l2, quantizer_1l2, args.output_dir)
    compare_quantization_results(quantizer_l2.layer_scales, quantizer_1l2.layer_scales, args.output_dir)

    # Save summary report
    summary_path = os.path.join(args.output_dir, "comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPARISON SUMMARY: E[X²] vs E[1+X²]\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Model: {model_name}\n")
        f.write(f"  Calibration samples: {args.n_calib}\n")
        f.write(f"  Evaluation samples: {args.n_eval}\n")
        f.write(f"  Group size: {args.group_size}\n")
        f.write(f"  Grid search points: {args.n_grid + 1}\n\n")

        f.write(f"Perplexity Results:\n")
        f.write(f"  E[X²]:     {ppl_l2:.2f}\n")
        f.write(f"  E[1+X²]:   {ppl_1l2:.2f}\n")
        f.write(f"  Difference: {ppl_1l2 - ppl_l2:+.2f}\n\n")

        if ppl_1l2 < ppl_l2:
            improvement = (ppl_l2 - ppl_1l2) / ppl_l2 * 100
            f.write(f"Winner: E[1+X²] (better by {improvement:.2f}%)\n")
        elif ppl_1l2 > ppl_l2:
            degradation = (ppl_1l2 - ppl_l2) / ppl_l2 * 100
            f.write(f"Winner: E[X²] (E[1+X²] worse by {degradation:.2f}%)\n")
        else:
            f.write(f"Result: Tie (both methods perform equally)\n")

    print(f"\nComparison summary saved to {summary_path}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("  - salience_comparison.png: Salience distribution analysis")
    print("  - quantization_comparison.png: Optimal α and error analysis")
    print("  - comparison_summary.txt: Summary report")
    print("=" * 80)


if __name__ == "__main__":
    main()
