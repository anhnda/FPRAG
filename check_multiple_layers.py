"""
Test AWQ vs FastRPRAQ on multiple layers to see if the pattern holds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_wikitext2(split="train", n_samples=None, seed=42):
    """Load WikiText-2 dataset with optional sampling."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    if n_samples and split == "validation":
        random.seed(seed)
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)
    elif n_samples:
        texts = texts[:n_samples]

    return texts


class LayerTester:
    """Test importance scoring methods on a single layer."""

    def __init__(self, model, tokenizer, layer_name, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        self.device = device

        # Find target module
        self.target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.target_module = module
                break

        if not isinstance(self.target_module, nn.Linear):
            raise ValueError(f"Layer {layer_name} is not a Linear layer!")

        self.input_activations = []
        self.output_ground_truth = []
        self.hook = None

    def register_hook(self, store_output=False):
        """Register hook to collect activation data."""
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                inp = input[0].detach()
            else:
                inp = input.detach()

            inp_flat = inp.reshape(-1, inp.shape[-1])
            self.input_activations.append(inp_flat.cpu().float())

            if store_output:
                if isinstance(output, tuple):
                    out = output[0].detach()
                else:
                    out = output.detach()
                out_flat = out.reshape(-1, out.shape[-1])
                self.output_ground_truth.append(out_flat.cpu().float())

        self.hook = self.target_module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    @torch.no_grad()
    def collect_data(self, texts, n_samples, store_output=False):
        """Collect activation data."""
        self.input_activations = []
        self.output_ground_truth = []
        self.model.eval()

        for text in tqdm(texts[:n_samples], desc=f"Collecting data", leave=False, disable=True):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                self.model(**inputs, use_cache=False)
            except:
                continue

    @torch.no_grad()
    def compute_awq_importance(self, X):
        """
        Compute AWQ importance using original AWQ metric: E[|XW^T + b|]
        """
        W = self.target_module.weight.data.cpu().float()
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None

        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b

        # AWQ importance: E[|Z|] per channel (original AWQ paper)
        return Z.abs().mean(dim=0)

    @torch.no_grad()
    def compute_praq_importance(self, X, beta=3.0, tau=-3.0, noise_factor=0.2):
        """Compute FastRPRAQ importance."""
        W = self.target_module.weight.data.cpu().float()
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None

        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b

        z_mean = Z.mean(dim=0)
        z_std = Z.std(dim=0) + 1e-8
        z_upper = z_mean + 3 * z_std

        x_mag = X.abs().mean()
        w_mag = W.abs().mean(dim=1)
        estimated_noise_impact = x_mag * w_mag * noise_factor

        z_risk_upper = z_upper + estimated_noise_impact
        prob_active = torch.sigmoid(beta * (z_risk_upper - tau))
        magnitude = Z.abs().mean(dim=0) + z_std

        return prob_active * magnitude

    @torch.no_grad()
    def apply_quantization(self, X, importance_scores, keep_ratio=0.5):
        """Apply quantization and compute output."""
        W = self.target_module.weight.data.cpu().float().clone()
        b = self.target_module.bias.data.cpu().float().clone() if self.target_module.bias is not None else None

        out_features = W.shape[0]
        k = max(1, int(out_features * keep_ratio))
        top_k_indices = torch.topk(importance_scores, k).indices

        mask_keep = torch.zeros(out_features, dtype=torch.bool)
        mask_keep[top_k_indices] = True

        W_quantized = W.clone()
        for c in range(out_features):
            if not mask_keep[c]:
                w_channel = W[c, :]
                w_range = w_channel.abs().max()
                if w_range > 0:
                    scale = w_range / 7.0
                    w_quant = torch.round(w_channel / scale).clamp(-8, 7)
                    W_quantized[c, :] = w_quant * scale
                    noise = torch.randn_like(w_channel) * scale * 0.1
                    W_quantized[c, :] += noise

        Z_quantized = torch.matmul(X, W_quantized.t())
        if b is not None:
            Z_quantized = Z_quantized + b

        return Z_quantized

    def test_layer(self, calib_texts, val_texts, n_calib=500, n_val=2000, keep_ratio=0.5):
        """Test both methods on this layer with detailed error distribution analysis."""
        # Collect calibration data
        self.register_hook(store_output=False)
        self.collect_data(calib_texts, n_calib, store_output=False)
        self.remove_hook()
        calib_inputs = self.input_activations.copy()

        # Collect validation data
        self.register_hook(store_output=True)
        self.collect_data(val_texts, n_val, store_output=True)
        self.remove_hook()
        val_inputs = self.input_activations.copy()
        val_outputs = self.output_ground_truth.copy()

        # Compute importance from calibration data
        X_calib = torch.cat(calib_inputs, dim=0)
        awq_importance = self.compute_awq_importance(X_calib)
        praq_importance = self.compute_praq_importance(X_calib)

        # Evaluate on validation data
        X_val = torch.cat(val_inputs, dim=0)
        Y_gt = torch.cat(val_outputs, dim=0)

        Y_awq = self.apply_quantization(X_val, awq_importance, keep_ratio)
        Y_praq = self.apply_quantization(X_val, praq_importance, keep_ratio)

        # Compute element-wise squared errors
        squared_errors_awq = (Y_awq - Y_gt).pow(2)
        squared_errors_praq = (Y_praq - Y_gt).pow(2)

        # Compute absolute errors
        abs_errors_awq = (Y_awq - Y_gt).abs()
        abs_errors_praq = (Y_praq - Y_gt).abs()

        # For large tensors, sample for percentile computation to avoid memory issues
        max_samples_for_quantile = 1_000_000
        se_awq_flat = squared_errors_awq.flatten()
        se_praq_flat = squared_errors_praq.flatten()

        if se_awq_flat.numel() > max_samples_for_quantile:
            # Sample uniformly for percentile computation
            indices = torch.linspace(0, se_awq_flat.numel() - 1, max_samples_for_quantile, dtype=torch.long)
            se_awq_sample = se_awq_flat[indices]
            se_praq_sample = se_praq_flat[indices]
        else:
            se_awq_sample = se_awq_flat
            se_praq_sample = se_praq_flat

        # Detailed statistics
        return {
            # Mean squared error
            'mse_awq': squared_errors_awq.mean().item(),
            'mse_praq': squared_errors_praq.mean().item(),

            # Mean absolute error
            'mae_awq': abs_errors_awq.mean().item(),
            'mae_praq': abs_errors_praq.mean().item(),

            # Error distribution statistics
            'mse_std_awq': squared_errors_awq.std().item(),
            'mse_std_praq': squared_errors_praq.std().item(),

            # Percentiles of squared errors (computed on sample if needed)
            'mse_p50_awq': squared_errors_awq.median().item(),
            'mse_p50_praq': squared_errors_praq.median().item(),
            'mse_p95_awq': torch.quantile(se_awq_sample, 0.95).item(),
            'mse_p95_praq': torch.quantile(se_praq_sample, 0.95).item(),
            'mse_p99_awq': torch.quantile(se_awq_sample, 0.99).item(),
            'mse_p99_praq': torch.quantile(se_praq_sample, 0.99).item(),

            # Maximum errors
            'mse_max_awq': squared_errors_awq.max().item(),
            'mse_max_praq': squared_errors_praq.max().item(),
            'mae_max_awq': abs_errors_awq.max().item(),
            'mae_max_praq': abs_errors_praq.max().item(),

            # Per-channel statistics (mean across samples, then stats across channels)
            'mse_per_channel_awq': squared_errors_awq.mean(dim=0),  # Keep tensor for later analysis
            'mse_per_channel_praq': squared_errors_praq.mean(dim=0),

            # Improvement metric
            'improvement': ((squared_errors_awq.mean() - squared_errors_praq.mean()) / squared_errors_awq.mean() * 100).item()
        }


def visualize_error_distributions(results, output_dir="./visualizations/error_distribution"):
    """Create visualizations comparing error distributions between AWQ and PRAQ."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Summary plot: Mean vs P95 vs P99 vs Max for all layers
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ('mse_awq', 'mse_praq', 'Mean MSE', axes[0, 0]),
        ('mse_p95_awq', 'mse_p95_praq', '95th Percentile MSE', axes[0, 1]),
        ('mse_p99_awq', 'mse_p99_praq', '99th Percentile MSE', axes[1, 0]),
        ('mse_max_awq', 'mse_max_praq', 'Max MSE', axes[1, 1])
    ]

    for awq_key, praq_key, title, ax in metrics:
        awq_vals = [r[awq_key] for r in results]
        praq_vals = [r[praq_key] for r in results]
        layer_names = [r['layer'].split('.')[-2] + '.' + r['layer'].split('.')[-1] for r in results]

        x = np.arange(len(layer_names))
        width = 0.35

        ax.bar(x - width/2, awq_vals, width, label='AWQ', alpha=0.8)
        ax.bar(x + width/2, praq_vals, width, label='PRAQ', alpha=0.8)

        ax.set_xlabel('Layer')
        ax.set_ylabel('MSE')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution_summary.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_dir}/error_distribution_summary.png")
    plt.close()

    # 2. Scatter plot: Mean MSE vs Tail behavior
    fig, ax = plt.subplots(figsize=(12, 8))

    for r in results:
        mean_improvement = r['improvement']
        # Compute tail degradation (positive means PRAQ has worse tail)
        tail_degradation = ((r['mse_p95_praq'] - r['mse_p95_awq']) / r['mse_p95_awq'] * 100)

        # Color based on whether it's a "paradox case" (lower mean, worse tail)
        if mean_improvement > 0 and tail_degradation > 0:
            color = 'red'
            marker = 'o'
            label = 'PRAQ: Lower mean, worse tail'
        elif mean_improvement > 0 and tail_degradation <= 0:
            color = 'green'
            marker = 's'
            label = 'PRAQ: Lower mean, better tail'
        elif mean_improvement <= 0 and tail_degradation > 0:
            color = 'orange'
            marker = '^'
            label = 'PRAQ: Higher mean, worse tail'
        else:
            color = 'blue'
            marker = 'v'
            label = 'PRAQ: Higher mean, better tail'

        ax.scatter(mean_improvement, tail_degradation, c=color, marker=marker, s=100, alpha=0.7)

    # Add quadrant lines
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

    # Add labels
    ax.set_xlabel('Mean MSE Improvement (%) [PRAQ vs AWQ]', fontsize=12)
    ax.set_ylabel('95th Percentile Degradation (%) [PRAQ vs AWQ]', fontsize=12)
    ax.set_title('Mean vs Tail Behavior Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Lower mean, worse tail (PROBLEM)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Lower mean, better tail (IDEAL)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='Higher mean, worse tail'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue', markersize=10, label='Higher mean, better tail')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_vs_tail_tradeoff.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/mean_vs_tail_tradeoff.png")
    plt.close()

    # 3. Per-channel MSE distributions for selected layers
    selected_layers = [results[0], results[len(results)//2], results[-1]] if len(results) >= 3 else results

    for r in selected_layers:
        layer_name = r['layer'].split('.')[-2] + '.' + r['layer'].split('.')[-1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of per-channel MSE
        mse_awq = r['mse_per_channel_awq'].numpy()
        mse_praq = r['mse_per_channel_praq'].numpy()

        axes[0].hist(mse_awq, bins=50, alpha=0.6, label='AWQ', color='blue', edgecolor='black')
        axes[0].hist(mse_praq, bins=50, alpha=0.6, label='PRAQ', color='red', edgecolor='black')
        axes[0].set_xlabel('Per-channel MSE')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Per-channel MSE Distribution: {layer_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # CDF comparison
        mse_awq_sorted = np.sort(mse_awq)
        mse_praq_sorted = np.sort(mse_praq)
        cdf = np.arange(1, len(mse_awq_sorted) + 1) / len(mse_awq_sorted)

        axes[1].plot(mse_awq_sorted, cdf, label='AWQ', linewidth=2)
        axes[1].plot(mse_praq_sorted, cdf, label='PRAQ', linewidth=2)
        axes[1].set_xlabel('Per-channel MSE')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title(f'CDF of Per-channel MSE: {layer_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')

        # Highlight tail region
        axes[1].axvline(np.percentile(mse_awq, 95), color='blue', linestyle='--', alpha=0.5, label='95th (AWQ)')
        axes[1].axvline(np.percentile(mse_praq, 95), color='red', linestyle='--', alpha=0.5, label='95th (PRAQ)')

        plt.tight_layout()
        safe_name = layer_name.replace('.', '_')
        plt.savefig(f"{output_dir}/per_channel_dist_{safe_name}.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/per_channel_dist_{safe_name}.png")
        plt.close()


def find_mlp_layers(model, num_blocks=3):
    """
    Find MLP layers to test. Tests all 3 components (gate_proj, up_proj, down_proj)
    from multiple transformer blocks.

    Args:
        num_blocks: Number of transformer blocks to sample

    Returns:
        List of layer names covering all MLP components
    """
    # Group by transformer block
    blocks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name_lower = name.lower()
            if 'mlp' in name_lower:
                # Extract block number (e.g., "model.layers.5.mlp.gate_proj" -> 5)
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        block_id = int(parts[i + 1])
                        if block_id not in blocks:
                            blocks[block_id] = []
                        blocks[block_id].append(name)
                        break

    # Sample blocks evenly across model depth
    sorted_blocks = sorted(blocks.keys())
    if len(sorted_blocks) > num_blocks:
        step = len(sorted_blocks) // num_blocks
        selected_blocks = [sorted_blocks[i * step] for i in range(num_blocks)]
    else:
        selected_blocks = sorted_blocks

    # Collect all layers from selected blocks
    selected_layers = []
    for block_id in selected_blocks:
        # Sort to get consistent order: gate_proj, up_proj, down_proj
        block_layers = sorted(blocks[block_id])
        selected_layers.extend(block_layers)

    return selected_layers


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Multi-layer MSE comparison: AWQ vs FastRPRAQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.2,
        help="Fraction of channels to keep in FP16"
    )
    parser.add_argument(
        "--n-calib",
        type=int,
        default=500,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=2000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of transformer blocks to sample (3 blocks × 3 layers = 9 layers)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_calib = args.n_calib
    n_val = args.n_val
    keep_ratio = args.keep_ratio

    print("=" * 80)
    print("MULTI-LAYER MSE COMPARISON: AWQ vs FastRPRAQ")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Calibration samples: {n_calib}")
    print(f"Validation samples: {n_val}")
    print(f"Keep ratio: {keep_ratio} ({int(keep_ratio*100)}% FP16, {int((1-keep_ratio)*100)}% INT4)")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Find layers to test
    mlp_layers = find_mlp_layers(model, num_blocks=args.num_blocks)
    print(f"\nTesting {len(mlp_layers)} MLP layers from {args.num_blocks} blocks:")
    print(f"(Each block has 3 layers: gate_proj, up_proj, down_proj)")
    for layer in mlp_layers:
        print(f"  - {layer}")

    # Load data
    print("\nLoading datasets...")
    calib_texts = load_wikitext2("train", n_samples=n_calib, seed=args.seed)
    val_texts = load_wikitext2("validation", n_samples=n_val, seed=args.seed)

    # Test each layer
    results = []
    print("\n" + "=" * 80)
    print("TESTING LAYERS")
    print("=" * 80)

    for layer_name in mlp_layers:
        print(f"\nTesting: {layer_name}")
        try:
            tester = LayerTester(model, tokenizer, layer_name, device)
            result = tester.test_layer(calib_texts, val_texts, n_calib, n_val, keep_ratio)
            result['layer'] = layer_name

            # Convert tensors to scalars for storage
            result['mse_per_channel_awq'] = result['mse_per_channel_awq'].cpu()
            result['mse_per_channel_praq'] = result['mse_per_channel_praq'].cpu()
            results.append(result)

            print(f"  Mean MSE:    AWQ={result['mse_awq']:.6f}  PRAQ={result['mse_praq']:.6f}  ({result['improvement']:+.2f}%)")
            print(f"  Median MSE:  AWQ={result['mse_p50_awq']:.6f}  PRAQ={result['mse_p50_praq']:.6f}")
            print(f"  95th %ile:   AWQ={result['mse_p95_awq']:.6f}  PRAQ={result['mse_p95_praq']:.6f}")
            print(f"  99th %ile:   AWQ={result['mse_p99_awq']:.6f}  PRAQ={result['mse_p99_praq']:.6f}")
            print(f"  Max Error:   AWQ={result['mse_max_awq']:.6f}  PRAQ={result['mse_max_praq']:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        n = len(results)

        # Aggregate statistics
        avg_improvement = sum(r['improvement'] for r in results) / n
        praq_wins_mean = sum(1 for r in results if r['improvement'] > 0)

        # Count wins at different percentiles
        praq_wins_p95 = sum(1 for r in results if r['mse_p95_praq'] < r['mse_p95_awq'])
        praq_wins_p99 = sum(1 for r in results if r['mse_p99_praq'] < r['mse_p99_awq'])
        praq_wins_max = sum(1 for r in results if r['mse_max_praq'] < r['mse_max_awq'])

        print(f"\nLayers tested: {n}")
        print(f"Keep ratio: {keep_ratio} ({int(keep_ratio*100)}% FP16, {int((1-keep_ratio)*100)}% INT4)")

        print("\n" + "-" * 80)
        print("MEAN MSE COMPARISON")
        print("-" * 80)
        print(f"PRAQ wins (lower mean MSE): {praq_wins_mean}/{n} ({100*praq_wins_mean/n:.0f}%)")
        print(f"Average improvement: {avg_improvement:+.2f}%")

        print("\n" + "-" * 80)
        print("ERROR DISTRIBUTION ANALYSIS")
        print("-" * 80)
        print(f"PRAQ wins at 95th percentile: {praq_wins_p95}/{n} ({100*praq_wins_p95/n:.0f}%)")
        print(f"PRAQ wins at 99th percentile: {praq_wins_p99}/{n} ({100*praq_wins_p99/n:.0f}%)")
        print(f"PRAQ wins at max error:       {praq_wins_max}/{n} ({100*praq_wins_max/n:.0f}%)")

        # Compute average percentiles across layers
        avg_mse_p95_awq = sum(r['mse_p95_awq'] for r in results) / n
        avg_mse_p95_praq = sum(r['mse_p95_praq'] for r in results) / n
        avg_mse_p99_awq = sum(r['mse_p99_awq'] for r in results) / n
        avg_mse_p99_praq = sum(r['mse_p99_praq'] for r in results) / n
        avg_mse_max_awq = sum(r['mse_max_awq'] for r in results) / n
        avg_mse_max_praq = sum(r['mse_max_praq'] for r in results) / n

        print(f"\nAverage 95th percentile: AWQ={avg_mse_p95_awq:.6f}  PRAQ={avg_mse_p95_praq:.6f}")
        print(f"Average 99th percentile: AWQ={avg_mse_p99_awq:.6f}  PRAQ={avg_mse_p99_praq:.6f}")
        print(f"Average max error:       AWQ={avg_mse_max_awq:.6f}  PRAQ={avg_mse_max_praq:.6f}")

        # KEY INSIGHT: Check if PRAQ has lower mean but higher tail
        tail_problem_count = sum(1 for r in results
                                if r['improvement'] > 0  # PRAQ wins on mean
                                and (r['mse_p95_praq'] > r['mse_p95_awq'] or
                                     r['mse_p99_praq'] > r['mse_p99_awq'] or
                                     r['mse_max_praq'] > r['mse_max_awq']))

        if tail_problem_count > 0:
            print("\n" + "!" * 80)
            print(f"⚠ TAIL BEHAVIOR PROBLEM DETECTED:")
            print(f"  {tail_problem_count}/{praq_wins_mean} layers where PRAQ has lower mean BUT worse tail errors")
            print(f"  This explains why PRAQ has lower MSE but worse perplexity!")
            print("!" * 80)

        print("\n" + "-" * 80)
        print(f"{'Layer':<40} {'Mean':<8} {'P95':<8} {'P99':<8} {'Max':<8}")
        print("-" * 80)
        for r in results:
            layer_short = r['layer'].split('.')[-2] + '.' + r['layer'].split('.')[-1]

            # Mark which metrics PRAQ wins
            mean_mark = "✓" if r['improvement'] > 0 else "✗"
            p95_mark = "✓" if r['mse_p95_praq'] < r['mse_p95_awq'] else "✗"
            p99_mark = "✓" if r['mse_p99_praq'] < r['mse_p99_awq'] else "✗"
            max_mark = "✓" if r['mse_max_praq'] < r['mse_max_awq'] else "✗"

            print(f"{layer_short:<40} {mean_mark:<8} {p95_mark:<8} {p99_mark:<8} {max_mark:<8}")
        print("-" * 80)

        # Show statistics by layer type
        gate_results = [r for r in results if 'gate' in r['layer']]
        up_results = [r for r in results if 'up' in r['layer']]
        down_results = [r for r in results if 'down' in r['layer']]

        print("\nBreakdown by layer type:")
        if gate_results:
            gate_avg = sum(r['improvement'] for r in gate_results) / len(gate_results)
            gate_wins = sum(1 for r in gate_results if r['improvement'] > 0)
            print(f"  gate_proj: {gate_wins}/{len(gate_results)} wins on mean, avg {gate_avg:+.2f}%")
        if up_results:
            up_avg = sum(r['improvement'] for r in up_results) / len(up_results)
            up_wins = sum(1 for r in up_results if r['improvement'] > 0)
            print(f"  up_proj:   {up_wins}/{len(up_results)} wins on mean, avg {up_avg:+.2f}%")
        if down_results:
            down_avg = sum(r['improvement'] for r in down_results) / len(down_results)
            down_wins = sum(1 for r in down_results if r['improvement'] > 0)
            print(f"  down_proj: {down_wins}/{len(down_results)} wins on mean, avg {down_avg:+.2f}%")

        # Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        visualize_error_distributions(results)
        print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
