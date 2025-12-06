"""
Compare which channel types AWQ protects vs PRAQ protects.

This script analyzes the differences in channel selection between AWQ and PRAQ
to understand why PRAQ has worse error propagation despite better per-layer MSE.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


class ChannelSelectionAnalyzer:
    """Analyze which channels AWQ vs PRAQ protect."""

    def __init__(self, model, tokenizer, device="cuda", beta=3.0, tau=-3.0, noise_factor=0.2):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.tau = tau
        self.noise_factor = noise_factor

        # Storage for activations
        self.activation_data = {}
        self.hooks = []

        # Detect layer types
        self.layer_types = self._detect_layer_types()

    def _detect_layer_types(self):
        """Detect which layers are MLP vs attention."""
        layer_types = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                name_lower = name.lower()
                if any(kw in name_lower for kw in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'out_proj', 'attention']):
                    layer_types[name] = 'attention'
                elif any(kw in name_lower for kw in ['mlp', 'fc', 'gate', 'up_proj', 'down_proj', 'ffn']):
                    layer_types[name] = 'mlp'
                else:
                    layer_types[name] = 'mlp'
        return layer_types

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                if isinstance(input, tuple):
                    inp = input[0].detach().cpu()
                else:
                    inp = input.detach().cpu()
                self.activation_data[name].append(inp)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def calibrate(self, calibration_data, n_samples=500):
        """Collect activations from calibration data."""
        print(f"Calibrating with {min(n_samples, len(calibration_data))} samples...")
        self.model.eval()
        self.register_hooks()

        for i, text in enumerate(tqdm(calibration_data[:n_samples], desc="Calibration")):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    _ = self.model(**inputs, use_cache=False, return_dict=True)
            except Exception:
                if i % 100 == 0 and i > 0:
                    print(f"\nNote: Skipped {i} samples with errors")
                continue

        self.remove_hooks()
        print("Calibration complete!")

    @torch.no_grad()
    def compute_awq_importance(self, name, module):
        """Compute AWQ importance: E[|X|] · |W| (salient weight magnitude)"""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return torch.ones(module.out_features)

        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Compute activation salience: E[|X|] per input feature
        activation_salience = X.abs().mean(dim=0)  # Shape: [in_features]

        # Get weight matrix
        W = module.weight.data  # [out_features, in_features]

        # AWQ importance: E[|X|] · |W| per output channel
        importance = torch.matmul(W.abs().cpu(), activation_salience.cpu())

        return importance

    @torch.no_grad()
    def compute_praq_importance(self, name, module):
        """Compute PRAQ importance: P(activation) × magnitude"""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return torch.ones(module.out_features)

        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        batch_size = 1024
        n_samples = X.shape[0]
        W = module.weight.data
        b = module.bias.data if module.bias is not None else torch.zeros(module.out_features, device=self.device)

        z_sum = torch.zeros(module.out_features, device=self.device)
        z_sq_sum = torch.zeros(module.out_features, device=self.device)
        z_abs_sum = torch.zeros(module.out_features, device=self.device)

        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size].to(self.device)
            Z = torch.matmul(batch_X, W.t()) + b
            z_sum += Z.sum(dim=0)
            z_sq_sum += (Z ** 2).sum(dim=0)
            z_abs_sum += Z.abs().sum(dim=0)
            del batch_X, Z

        z_mean = z_sum / n_samples
        z_variance = (z_sq_sum / n_samples) - (z_mean ** 2)
        z_std = torch.sqrt(z_variance.clamp(min=0)) + 1e-8
        z_upper = z_mean + 3 * z_std

        # Estimate quantization noise
        X_cpu = X.to(self.device) if X.device != self.device else X
        x_mag = X_cpu.abs().mean().item()
        del X_cpu

        w_mag = W.abs().mean(dim=1)
        estimated_noise_impact = x_mag * w_mag * self.noise_factor

        z_risk_upper = z_upper + estimated_noise_impact
        prob_active = torch.sigmoid(self.beta * (z_risk_upper - self.tau))
        magnitude = z_abs_sum / n_samples + z_std

        return (prob_active * magnitude).cpu()

    @torch.no_grad()
    def get_channel_statistics(self, name, module):
        """Get detailed statistics for each channel."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        batch_size = 1024
        n_samples = X.shape[0]
        W = module.weight.data
        b = module.bias.data if module.bias is not None else torch.zeros(module.out_features, device=self.device)

        z_sum = torch.zeros(module.out_features, device=self.device)
        z_sq_sum = torch.zeros(module.out_features, device=self.device)
        z_abs_sum = torch.zeros(module.out_features, device=self.device)

        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size].to(self.device)
            Z = torch.matmul(batch_X, W.t()) + b
            z_sum += Z.sum(dim=0)
            z_sq_sum += (Z ** 2).sum(dim=0)
            z_abs_sum += Z.abs().sum(dim=0)
            del batch_X, Z

        z_mean = z_sum / n_samples
        z_variance = (z_sq_sum / n_samples) - (z_mean ** 2)
        z_std = torch.sqrt(z_variance.clamp(min=0)) + 1e-8

        # Compute all relevant statistics
        X_cpu = X.to(self.device) if X.device != self.device else X
        x_mag = X_cpu.abs().mean().item()
        del X_cpu

        w_mag = W.abs().mean(dim=1)
        estimated_noise = x_mag * w_mag * self.noise_factor
        z_risk_upper = z_mean + 3 * z_std + estimated_noise

        return {
            'mean': z_mean.cpu(),
            'std': z_std.cpu(),
            'magnitude': (z_abs_sum / n_samples).cpu(),
            'weight_mag': w_mag.cpu(),
            'noise_estimate': estimated_noise.cpu(),
            'risk_upper': z_risk_upper.cpu()
        }

    def analyze_layer(self, name, module, keep_ratio=0.2):
        """Analyze channel selection differences for a single layer."""
        layer_type = self.layer_types.get(name, 'mlp')

        # Compute importance scores
        awq_scores = self.compute_awq_importance(name, module)

        if layer_type == 'mlp':
            praq_scores = self.compute_praq_importance(name, module)
        else:
            # For attention layers, PRAQ uses AWQ-style importance
            praq_scores = awq_scores.clone()

        # Get channel statistics
        stats = self.get_channel_statistics(name, module)
        if stats is None:
            return None

        # Determine which channels each method protects
        out_features = module.out_features
        k = max(1, int(out_features * keep_ratio))

        awq_protected = set(torch.topk(awq_scores, k).indices.tolist())
        praq_protected = set(torch.topk(praq_scores, k).indices.tolist())

        # Categorize channels
        both_protected = awq_protected & praq_protected
        awq_only = awq_protected - praq_protected
        praq_only = praq_protected - awq_protected
        neither = set(range(out_features)) - awq_protected - praq_protected

        return {
            'name': name,
            'layer_type': layer_type,
            'out_features': out_features,
            'awq_scores': awq_scores,
            'praq_scores': praq_scores,
            'stats': stats,
            'both_protected': both_protected,
            'awq_only': awq_only,
            'praq_only': praq_only,
            'neither': neither,
            'overlap_ratio': len(both_protected) / k if k > 0 else 0
        }

    def analyze_all_layers(self, keep_ratio=0.2, target_layers=None):
        """Analyze all layers and aggregate results."""
        print(f"\n{'='*80}")
        print(f"Analyzing channel selection (keep_ratio={keep_ratio})")
        print(f"{'='*80}\n")

        all_results = []

        for name, module in tqdm(list(self.model.named_modules()), desc="Analyzing"):
            if isinstance(module, nn.Linear):
                # Skip if target_layers specified and this layer not in list
                if target_layers and name not in target_layers:
                    continue

                result = self.analyze_layer(name, module, keep_ratio)
                if result is not None:
                    all_results.append(result)

        return all_results


def visualize_selection_differences(results, output_dir="./visualizations/channel_selection"):
    """Create visualizations comparing AWQ and PRAQ channel selection."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    # Filter to MLP layers only (where PRAQ differs from AWQ)
    mlp_results = [r for r in results if r['layer_type'] == 'mlp']

    if not mlp_results:
        print("No MLP layers found for analysis!")
        return

    # 1. Overlap analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Overlap ratio distribution
    overlap_ratios = [r['overlap_ratio'] for r in mlp_results]
    axes[0, 0].hist(overlap_ratios, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(overlap_ratios), color='red', linestyle='--',
                       label=f'Mean: {np.mean(overlap_ratios):.2%}')
    axes[0, 0].set_xlabel('Overlap Ratio (Both Protected / Total Protected)')
    axes[0, 0].set_ylabel('Number of Layers')
    axes[0, 0].set_title('Channel Selection Overlap: AWQ vs PRAQ')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Channel category sizes
    awq_only_sizes = [len(r['awq_only']) for r in mlp_results]
    praq_only_sizes = [len(r['praq_only']) for r in mlp_results]
    both_sizes = [len(r['both_protected']) for r in mlp_results]

    x = np.arange(len(mlp_results))
    width = 0.25

    axes[0, 1].bar(x - width, both_sizes, width, label='Both Protect', alpha=0.8)
    axes[0, 1].bar(x, awq_only_sizes, width, label='AWQ Only', alpha=0.8)
    axes[0, 1].bar(x + width, praq_only_sizes, width, label='PRAQ Only', alpha=0.8)
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Number of Channels')
    axes[0, 1].set_title('Protected Channel Distribution by Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Aggregate statistics for channel categories
    categories = []
    means = []
    stds = []
    weight_mags = []

    for category_name, get_indices in [
        ('Both Protected', lambda r: r['both_protected']),
        ('AWQ Only', lambda r: r['awq_only']),
        ('PRAQ Only', lambda r: r['praq_only'])
    ]:
        cat_means = []
        cat_stds = []
        cat_wmags = []

        for r in mlp_results:
            indices = list(get_indices(r))
            if indices:
                cat_means.extend(r['stats']['mean'][indices].tolist())
                cat_stds.extend(r['stats']['std'][indices].tolist())
                cat_wmags.extend(r['stats']['weight_mag'][indices].tolist())

        if cat_means:
            categories.append(category_name)
            means.append(np.mean(cat_means))
            stds.append(np.mean(cat_stds))
            weight_mags.append(np.mean(cat_wmags))

    x_pos = np.arange(len(categories))
    axes[1, 0].bar(x_pos, means, alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categories, rotation=15, ha='right')
    axes[1, 0].set_ylabel('Mean Pre-activation Value')
    axes[1, 0].set_title('Average Pre-activation Mean by Channel Category')
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Weight magnitude comparison
    axes[1, 1].bar(x_pos, weight_mags, alpha=0.8, color='orange')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(categories, rotation=15, ha='right')
    axes[1, 1].set_ylabel('Average Weight Magnitude')
    axes[1, 1].set_title('Average Weight Magnitude by Channel Category')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'selection_overview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

    # 2. Detailed scatter plots for channel characteristics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect data for all MLP layers
    all_channels = {
        'Both Protected': {'mean': [], 'std': [], 'weight_mag': [], 'risk_upper': []},
        'AWQ Only': {'mean': [], 'std': [], 'weight_mag': [], 'risk_upper': []},
        'PRAQ Only': {'mean': [], 'std': [], 'weight_mag': [], 'risk_upper': []}
    }

    for r in mlp_results:
        for category, indices_fn in [
            ('Both Protected', lambda: r['both_protected']),
            ('AWQ Only', lambda: r['awq_only']),
            ('PRAQ Only', lambda: r['praq_only'])
        ]:
            indices = list(indices_fn())
            if indices:
                all_channels[category]['mean'].extend(r['stats']['mean'][indices].tolist())
                all_channels[category]['std'].extend(r['stats']['std'][indices].tolist())
                all_channels[category]['weight_mag'].extend(r['stats']['weight_mag'][indices].tolist())
                all_channels[category]['risk_upper'].extend(r['stats']['risk_upper'][indices].tolist())

    # Plot scatter: mean vs std
    colors = {'Both Protected': 'green', 'AWQ Only': 'blue', 'PRAQ Only': 'red'}
    for category in all_channels.keys():
        if all_channels[category]['mean']:
            axes[0, 0].scatter(
                all_channels[category]['mean'],
                all_channels[category]['std'],
                alpha=0.3, s=10, label=category, color=colors[category]
            )
    axes[0, 0].set_xlabel('Pre-activation Mean')
    axes[0, 0].set_ylabel('Pre-activation Std')
    axes[0, 0].set_title('Mean vs Std by Channel Category')
    axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot scatter: mean vs weight magnitude
    for category in all_channels.keys():
        if all_channels[category]['mean']:
            axes[0, 1].scatter(
                all_channels[category]['mean'],
                all_channels[category]['weight_mag'],
                alpha=0.3, s=10, label=category, color=colors[category]
            )
    axes[0, 1].set_xlabel('Pre-activation Mean')
    axes[0, 1].set_ylabel('Weight Magnitude')
    axes[0, 1].set_title('Mean vs Weight Magnitude by Channel Category')
    axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot scatter: risk_upper vs weight magnitude
    for category in all_channels.keys():
        if all_channels[category]['risk_upper']:
            axes[1, 0].scatter(
                all_channels[category]['risk_upper'],
                all_channels[category]['weight_mag'],
                alpha=0.3, s=10, label=category, color=colors[category]
            )
    axes[1, 0].set_xlabel('Risk-adjusted Upper Bound')
    axes[1, 0].set_ylabel('Weight Magnitude')
    axes[1, 0].set_title('Risk Upper Bound vs Weight Magnitude')
    axes[1, 0].axvline(-3.0, color='purple', linestyle='--', linewidth=1.5, label='SiLU threshold (τ=-3)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot histogram: risk_upper distribution
    bins = np.linspace(-10, 10, 50)
    for category in all_channels.keys():
        if all_channels[category]['risk_upper']:
            axes[1, 1].hist(
                all_channels[category]['risk_upper'],
                bins=bins, alpha=0.5, label=category, color=colors[category]
            )
    axes[1, 1].axvline(-3.0, color='purple', linestyle='--', linewidth=1.5, label='SiLU threshold (τ=-3)')
    axes[1, 1].set_xlabel('Risk-adjusted Upper Bound')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Risk Upper Bound Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'channel_characteristics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def print_summary_statistics(results):
    """Print summary statistics about channel selection."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    mlp_results = [r for r in results if r['layer_type'] == 'mlp']

    if not mlp_results:
        print("No MLP layers to analyze!")
        return

    # Overall overlap
    overlap_ratios = [r['overlap_ratio'] for r in mlp_results]
    print(f"Average overlap ratio: {np.mean(overlap_ratios):.2%}")
    print(f"  (What fraction of protected channels are agreed upon by both methods)")
    print(f"  Min: {np.min(overlap_ratios):.2%}, Max: {np.max(overlap_ratios):.2%}")

    # Channel category statistics
    print(f"\nChannel Category Statistics (across {len(mlp_results)} MLP layers):")

    for category_name, get_indices in [
        ('Both Protected (Agreed)', lambda r: r['both_protected']),
        ('AWQ Only (AWQ prefers, PRAQ skips)', lambda r: r['awq_only']),
        ('PRAQ Only (PRAQ prefers, AWQ skips)', lambda r: r['praq_only'])
    ]:
        cat_means = []
        cat_stds = []
        cat_wmags = []
        cat_risk_uppers = []

        for r in mlp_results:
            indices = list(get_indices(r))
            if indices:
                cat_means.extend(r['stats']['mean'][indices].tolist())
                cat_stds.extend(r['stats']['std'][indices].tolist())
                cat_wmags.extend(r['stats']['weight_mag'][indices].tolist())
                cat_risk_uppers.extend(r['stats']['risk_upper'][indices].tolist())

        if cat_means:
            print(f"\n  {category_name}:")
            print(f"    Count: {len(cat_means)}")
            print(f"    Mean pre-activation: {np.mean(cat_means):.4f} ± {np.std(cat_means):.4f}")
            print(f"    Std pre-activation: {np.mean(cat_stds):.4f} ± {np.std(cat_stds):.4f}")
            print(f"    Weight magnitude: {np.mean(cat_wmags):.4f} ± {np.std(cat_wmags):.4f}")
            print(f"    Risk upper bound: {np.mean(cat_risk_uppers):.4f} ± {np.std(cat_risk_uppers):.4f}")

            # Count risky channels (risk_upper > -3)
            risky_count = sum(1 for x in cat_risk_uppers if x > -3.0)
            print(f"    Risky channels (risk_upper > τ=-3): {risky_count}/{len(cat_risk_uppers)} ({100*risky_count/len(cat_risk_uppers):.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare AWQ vs PRAQ channel selection")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Fraction of channels to protect")
    parser.add_argument("--n-calib", type=int, default=500, help="Calibration samples")
    parser.add_argument("--beta", type=float, default=3.0, help="PRAQ beta parameter")
    parser.add_argument("--tau", type=float, default=-3.0, help="PRAQ tau parameter")
    parser.add_argument("--noise-factor", type=float, default=0.2, help="PRAQ noise factor")
    parser.add_argument("--output-dir", type=str, default="./visualizations/channel_selection",
                       help="Output directory")
    args = parser.parse_args()

    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("Channel Selection Comparison: AWQ vs PRAQ")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Keep ratio: {args.keep_ratio}")
    print(f"Calibration samples: {args.n_calib}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)

    # Load calibration data
    print("Loading calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0][:args.n_calib]

    # Initialize analyzer
    analyzer = ChannelSelectionAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beta=args.beta,
        tau=args.tau,
        noise_factor=args.noise_factor
    )

    # Calibrate
    analyzer.calibrate(texts, n_samples=args.n_calib)

    # Analyze all layers
    results = analyzer.analyze_all_layers(keep_ratio=args.keep_ratio)

    # Print summary
    print_summary_statistics(results)

    # Visualize
    visualize_selection_differences(results, output_dir=args.output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
