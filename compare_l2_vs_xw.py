"""
Comparison Script: E[X²] vs ||X||²||W||² Salience for AWQ

This script compares two channel importance metrics:
1. E[X²] - Pure activation-based (gw_awq_asym_l2.py)
2. ||X||²||W||² - Joint activation-weight energy (awq_l2_xw.py)

Comparison Metrics:
- Layer-wise reconstruction MSE
- Perplexity on WikiText-2
- Channel importance correlation
- Optimal alpha distributions
- Quantization quality metrics
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, pearsonr

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Warning: seaborn not installed, using default matplotlib style")
    sns = None

# Import both quantizers
import sys
sys.path.append(os.path.dirname(__file__))
from gw_awq_asym_l2 import GroupWiseAWQAsymmetricL2Quantizer
from awq_l2_xw import GroupWiseAWQAsymmetricL2XW


def load_wikitext2(split="train", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device, max_samples=100):
    """Compute perplexity on a set of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts[:max_samples], desc="Computing perplexity"):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Skip if input is too short
            if inputs["input_ids"].shape[1] < 2:
                continue

            outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
            loss = outputs.loss

            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

        except Exception:
            continue

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


@torch.no_grad()
def compare_layer_importance(module, X_data, method='l2'):
    """
    Compute channel importance for a single layer using specified method.

    Args:
        module: Linear module
        X_data: List of activation tensors
        method: 'l2' for E[X²] or 'xw' for ||X||²||W||²

    Returns:
        importance: Tensor of shape [in_features]
    """
    if len(X_data) == 0:
        return None

    W = module.weight.data
    in_features = W.shape[1]

    # Compute ||X[:, j]||² for each channel
    X_l2_squared = torch.zeros(in_features, device=W.device)

    for x in X_data:
        x_flat = x.reshape(-1, x.shape[-1]).to(W.device)
        X_l2_squared += x_flat.pow(2).sum(dim=0)

    if method == 'l2':
        # E[X²] - Average over samples
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_data)
        importance = X_l2_squared / total_samples

    elif method == 'xw':
        # ||X||²||W||² - Joint energy
        W_l2_squared = W.pow(2).sum(dim=0)
        importance = X_l2_squared * W_l2_squared

    else:
        raise ValueError(f"Unknown method: {method}")

    return importance


def analyze_importance_correlation(model, tokenizer, calib_data, device, n_samples=128, n_layers=5):
    """
    Analyze correlation between E[X²] and ||X||²||W||² importance scores.
    """
    print("\n" + "=" * 80)
    print("IMPORTANCE CORRELATION ANALYSIS")
    print("=" * 80)

    # Register hooks to capture activations
    activation_data = {}
    hooks = []

    def get_hook(name):
        def hook(module, inp, output):
            if name not in activation_data:
                activation_data[name] = []
            if isinstance(inp, tuple):
                inp_data = inp[0].detach().cpu()
            else:
                inp_data = inp.detach().cpu()
            activation_data[name].append(inp_data)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(get_hook(name))
            hooks.append(handle)

    # Collect activations
    model.eval()
    print(f"\nCollecting activations from {n_samples} samples...")
    for text in tqdm(calib_data[:n_samples], desc="Calibration"):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = model(**inputs, use_cache=False)
        except:
            continue

    # Remove hooks
    for handle in hooks:
        handle.remove()

    # Analyze correlations
    results = []
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    # Sample layers evenly
    if len(layer_names) > n_layers:
        indices = np.linspace(0, len(layer_names) - 1, n_layers, dtype=int)
        layer_names = [layer_names[i] for i in indices]

    print(f"\nAnalyzing {len(layer_names)} layers...")

    for name in tqdm(layer_names, desc="Computing correlations"):
        module = dict(model.named_modules())[name]

        if name not in activation_data or len(activation_data[name]) == 0:
            continue

        # Compute both importance scores
        importance_l2 = compare_layer_importance(module, activation_data[name], method='l2')
        importance_xw = compare_layer_importance(module, activation_data[name], method='xw')

        if importance_l2 is None or importance_xw is None:
            continue

        # Move to CPU for correlation computation
        imp_l2_cpu = importance_l2.cpu().numpy()
        imp_xw_cpu = importance_xw.cpu().numpy()

        # Compute correlations
        spearman_corr, _ = spearmanr(imp_l2_cpu, imp_xw_cpu)
        pearson_corr, _ = pearsonr(imp_l2_cpu, imp_xw_cpu)

        # Compute rank differences
        rank_l2 = np.argsort(np.argsort(-imp_l2_cpu))  # Descending rank
        rank_xw = np.argsort(np.argsort(-imp_xw_cpu))
        rank_diff = np.abs(rank_l2 - rank_xw).mean()

        results.append({
            'layer': name,
            'spearman_r': spearman_corr,
            'pearson_r': pearson_corr,
            'avg_rank_diff': rank_diff,
            'n_channels': len(imp_l2_cpu)
        })

        # Clear data
        del activation_data[name]

    # Create DataFrame
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY")
    print("=" * 80)
    print(f"Average Spearman correlation: {df['spearman_r'].mean():.4f} ± {df['spearman_r'].std():.4f}")
    print(f"Average Pearson correlation: {df['pearson_r'].mean():.4f} ± {df['pearson_r'].std():.4f}")
    print(f"Average rank difference: {df['avg_rank_diff'].mean():.2f} ± {df['avg_rank_diff'].std():.2f}")
    print("\nInterpretation:")
    if df['spearman_r'].mean() > 0.9:
        print("  → Very high correlation - methods are very similar")
    elif df['spearman_r'].mean() > 0.7:
        print("  → High correlation - methods mostly agree")
    elif df['spearman_r'].mean() > 0.5:
        print("  → Moderate correlation - some differences")
    else:
        print("  → Low correlation - methods differ significantly")

    return df


def compare_quantization_methods(model_name, calib_data, eval_data, device,
                                 n_calib=128, n_eval=100, n_grid=20, group_size=128):
    """
    Compare E[X²] vs ||X||²||W||² quantization methods.
    """
    print("\n" + "=" * 80)
    print("QUANTIZATION METHOD COMPARISON")
    print("=" * 80)
    print(f"Method 1: E[X²] - Pure activation-based salience")
    print(f"Method 2: ||X||²||W||² - Joint activation-weight salience")
    print("=" * 80)

    results = {}

    # Method 1: E[X²] (gw_awq_asym_l2)
    print("\n" + "-" * 80)
    print("METHOD 1: E[X²] Salience")
    print("-" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
        n_grid=n_grid,
        group_size=group_size
    )

    quantizer_l2.calibrate(calib_data, n_samples=n_calib)
    quantizer_l2.quantize_model()

    ppl_l2 = compute_perplexity(model_l2, tokenizer, eval_data, device, max_samples=n_eval)
    results['l2'] = {
        'perplexity': ppl_l2,
        'alphas': [info['alpha'] for info in quantizer_l2.layer_scales.values()],
        'errors': [info['error'] for info in quantizer_l2.layer_scales.values()]
    }

    print(f"\nMethod 1 Results:")
    print(f"  Perplexity: {ppl_l2:.4f}")
    print(f"  Mean α: {np.mean(results['l2']['alphas']):.3f}")
    print(f"  Median α: {np.median(results['l2']['alphas']):.3f}")

    # Method 2: ||X||²||W||² (awq_l2_xw)
    print("\n" + "-" * 80)
    print("METHOD 2: ||X||²||W||² Salience")
    print("-" * 80)

    model_xw = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    quantizer_xw = GroupWiseAWQAsymmetricL2XW(
        model=model_xw,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=n_grid,
        group_size=group_size
    )

    quantizer_xw.calibrate(calib_data, n_samples=n_calib)
    quantizer_xw.quantize_model()

    ppl_xw = compute_perplexity(model_xw, tokenizer, eval_data, device, max_samples=n_eval)
    results['xw'] = {
        'perplexity': ppl_xw,
        'alphas': [info['alpha'] for info in quantizer_xw.layer_scales.values()],
        'errors': [info['error'] for info in quantizer_xw.layer_scales.values()]
    }

    print(f"\nMethod 2 Results:")
    print(f"  Perplexity: {ppl_xw:.4f}")
    print(f"  Mean α: {np.mean(results['xw']['alphas']):.3f}")
    print(f"  Median α: {np.median(results['xw']['alphas']):.3f}")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Perplexity:")
    print(f"  E[X²]:        {ppl_l2:.4f}")
    print(f"  ||X||²||W||²: {ppl_xw:.4f}")
    print(f"  Improvement:  {((ppl_l2 - ppl_xw) / ppl_l2 * 100):.2f}%")
    print(f"\nOptimal α:")
    print(f"  E[X²]:        {np.mean(results['l2']['alphas']):.3f} ± {np.std(results['l2']['alphas']):.3f}")
    print(f"  ||X||²||W||²: {np.mean(results['xw']['alphas']):.3f} ± {np.std(results['xw']['alphas']):.3f}")
    print(f"\nReconstruction Error:")
    print(f"  E[X²]:        {np.mean(results['l2']['errors']):.6f}")
    print(f"  ||X||²||W||²: {np.mean(results['xw']['errors']):.6f}")

    return results


def plot_comparison(results, output_dir="./visualizations/l2_vs_xw"):
    """Create visualization plots comparing the two methods."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Alpha distribution comparison
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Alpha histograms
    axes[0, 0].hist(results['l2']['alphas'], bins=20, alpha=0.6, label='E[X²]', color='blue', edgecolor='black')
    axes[0, 0].hist(results['xw']['alphas'], bins=20, alpha=0.6, label='||X||²||W||²', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Optimal α')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Optimal α Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_l2 = np.sort(results['l2']['alphas'])
    sorted_xw = np.sort(results['xw']['alphas'])
    axes[0, 1].plot(sorted_l2, np.linspace(0, 1, len(sorted_l2)), label='E[X²]', linewidth=2)
    axes[0, 1].plot(sorted_xw, np.linspace(0, 1, len(sorted_xw)), label='||X||²||W||²', linewidth=2)
    axes[0, 1].set_xlabel('Optimal α')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution of α')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Error comparison
    axes[1, 0].scatter(results['l2']['errors'], results['xw']['errors'], alpha=0.5, s=30)
    max_error = max(max(results['l2']['errors']), max(results['xw']['errors']))
    axes[1, 0].plot([0, max_error], [0, max_error], 'r--', label='y=x', linewidth=2)
    axes[1, 0].set_xlabel('E[X²] Reconstruction Error')
    axes[1, 0].set_ylabel('||X||²||W||² Reconstruction Error')
    axes[1, 0].set_title('Layer-wise Reconstruction Error Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')

    # Summary metrics
    metrics_data = {
        'Metric': ['Perplexity', 'Mean α', 'Mean Error'],
        'E[X²]': [
            results['l2']['perplexity'],
            np.mean(results['l2']['alphas']),
            np.mean(results['l2']['errors'])
        ],
        '||X||²||W||²': [
            results['xw']['perplexity'],
            np.mean(results['xw']['alphas']),
            np.mean(results['xw']['errors'])
        ]
    }

    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(
        cellText=[[metrics_data['Metric'][i],
                   f"{metrics_data['E[X²]'][i]:.4f}",
                   f"{metrics_data['||X||²||W||²'][i]:.4f}"]
                  for i in range(len(metrics_data['Metric']))],
        colLabels=['Metric', 'E[X²]', '||X||²||W||²'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Summary Statistics', pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_summary.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_dir}/comparison_summary.png")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare E[X²] vs ||X||²||W||² salience methods")
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-eval", type=int, default=100, help="Evaluation samples for perplexity")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./visualizations/l2_vs_xw", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="openbmb/MiniCPM-2B-sft-bf16", help="Model name")
    args = parser.parse_args()

    # Set seeds
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("COMPARISON: E[X²] vs ||X||²||W||² Salience Methods")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Evaluation samples: {args.n_eval}")
    print(f"Grid search points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
    print("=" * 80)

    # Load data
    calib_data = load_wikitext2(split="train", n_samples=args.n_calib)
    eval_data = load_wikitext2(split="validation", n_samples=args.n_eval)

    # First, analyze importance correlation (on original model)
    print("\nStep 1: Analyzing importance score correlation...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    corr_df = analyze_importance_correlation(model, tokenizer, calib_data, device,
                                             n_samples=min(128, args.n_calib), n_layers=10)

    # Save correlation results
    os.makedirs(args.output_dir, exist_ok=True)
    corr_df.to_csv(f"{args.output_dir}/importance_correlation.csv", index=False)
    print(f"\nSaved correlation analysis to {args.output_dir}/importance_correlation.csv")

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Compare quantization methods
    print("\nStep 2: Comparing quantization methods...")
    results = compare_quantization_methods(
        model_name=args.model,
        calib_data=calib_data,
        eval_data=eval_data,
        device=device,
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        n_grid=args.n_grid,
        group_size=args.group_size
    )

    # Plot comparison
    print("\nStep 3: Creating visualization...")
    plot_comparison(results, output_dir=args.output_dir)

    # Save results
    import json
    results_serializable = {
        'l2': {
            'perplexity': float(results['l2']['perplexity']),
            'mean_alpha': float(np.mean(results['l2']['alphas'])),
            'std_alpha': float(np.std(results['l2']['alphas'])),
            'mean_error': float(np.mean(results['l2']['errors']))
        },
        'xw': {
            'perplexity': float(results['xw']['perplexity']),
            'mean_alpha': float(np.mean(results['xw']['alphas'])),
            'std_alpha': float(np.std(results['xw']['alphas'])),
            'mean_error': float(np.mean(results['xw']['errors']))
        }
    }

    with open(f"{args.output_dir}/comparison_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nSaved results to {args.output_dir}/comparison_results.json")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
