"""
Rounding Error Analysis for Weight Quantization

This script analyzes the distribution of rounding errors when quantizing weights
to INT4 precision. It compares errors between important and non-important channels.

Key Analysis:
1. Compute importance scores using AWQ L2 salience
2. Simulate INT4 quantization and track rounding errors (with sign)
3. Separate channels into important vs non-important groups
4. Compare error distributions between groups
5. Visualize and report statistics

Output:
- Error distribution plots (important vs non-important channels)
- Mean/std statistics for each group
- Per-channel error analysis
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class RoundingErrorAnalyzer:
    """Analyzes rounding errors during weight quantization"""

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size

        # Storage
        self.activation_data = {}
        self.hooks = []

        print(f"\n[Rounding Error Analyzer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Device: {device}")

    def load_calibration_data(self, n_samples=128, max_length=512):
        """Load calibration data from WikiText-2"""
        print(f"\nLoading {n_samples} calibration samples from WikiText-2...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        calibration_data = []
        for example in tqdm(dataset, desc="Tokenizing"):
            text = example['text']
            if len(text.strip()) == 0:
                continue

            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=False
            )
            calibration_data.append(inputs['input_ids'].to(self.device))

            if len(calibration_data) >= n_samples:
                break

        return calibration_data

    def register_hooks_for_layer(self, layer_idx):
        """Register hook for a specific layer"""
        target_module = self.model.model.layers[layer_idx].mlp.gate_proj
        layer_name = f"model.layers.{layer_idx}.mlp.gate_proj"

        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                inp = input[0].detach().cpu()
                self.activation_data[name].append(inp)
            return hook

        handle = target_module.register_forward_hook(get_hook(layer_name))
        self.hooks.append(handle)

        print(f"Registered hook for: {layer_name}")
        print(f"Weight shape: {target_module.weight.shape}")

        return target_module, layer_name

    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @torch.no_grad()
    def collect_activations(self, calibration_data):
        """Run forward passes to collect activations"""
        print("\nCollecting activations from calibration data...")
        self.model.eval()

        for input_ids in tqdm(calibration_data, desc="Forward passes"):
            _ = self.model(input_ids, use_cache=False)

    @torch.no_grad()
    def compute_importance_l2(self, layer_name):
        """
        Compute per-input-channel importance using L2 salience: E[X[:, j]²]

        Returns:
            importance: Tensor of shape [in_features]
        """
        if layer_name not in self.activation_data:
            return None

        X_list = self.activation_data[layer_name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate L2 salience
        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            salience_sum += x_flat.pow(2).sum(dim=0)

        importance = salience_sum / total_samples
        return importance

    @torch.no_grad()
    def compute_rounding_errors(self, W, importance_scores, top_k_ratio=0.5):
        """
        Compute rounding errors for each weight during quantization.

        Args:
            W: Weight tensor [out_features, in_features]
            importance_scores: Per-input-channel importance [in_features]
            top_k_ratio: Ratio of channels to consider as "important"

        Returns:
            Dictionary with error statistics for important vs non-important channels
        """
        out_features, in_features = W.shape

        # Determine important channels
        k = int(in_features * top_k_ratio)
        top_k_indices = torch.topk(importance_scores, k).indices
        important_mask = torch.zeros(in_features, dtype=torch.bool)
        important_mask[top_k_indices] = True

        # Pad weights for group-wise quantization
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            # Extend importance mask
            importance_mask_padded = torch.zeros(padded_in_features, dtype=torch.bool)
            importance_mask_padded[:in_features] = important_mask
        else:
            W_padded = W
            importance_mask_padded = important_mask

        # Reshape to groups: [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Compute asymmetric quantization parameters per group
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Scale and zero point
        n_levels = 2 ** self.bits
        scale = (W_max - W_min) / (n_levels - 1)
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, n_levels - 1)

        # Continuous quantized values (before rounding)
        W_continuous = W_grouped / scale + zero_point

        # Apply rounding
        W_int = torch.round(W_continuous).clamp(0, n_levels - 1)

        # Rounding error in quantized space (signed)
        rounding_error_quant = W_int - W_continuous

        # Convert rounding error back to original weight space
        rounding_error_weight = rounding_error_quant * scale

        # Flatten to [out_features * padded_in_features]
        rounding_error_flat = rounding_error_weight.reshape(out_features, padded_in_features)

        # Remove padding
        if padded_in_features > in_features:
            rounding_error_flat = rounding_error_flat[:, :in_features]
            importance_mask_padded = importance_mask_padded[:in_features]

        # Separate errors by importance
        # Shape: [out_features, in_features]
        important_errors = rounding_error_flat[:, important_mask]
        non_important_errors = rounding_error_flat[:, ~important_mask]

        # Flatten to get all errors (convert to float32 for numpy compatibility)
        important_errors_all = important_errors.flatten().float().cpu().numpy()
        non_important_errors_all = non_important_errors.flatten().float().cpu().numpy()

        # Per-channel statistics (average over output features)
        per_channel_error_mean = rounding_error_flat.mean(dim=0).float().cpu().numpy()
        per_channel_error_std = rounding_error_flat.std(dim=0).float().cpu().numpy()
        per_channel_error_abs_mean = rounding_error_flat.abs().mean(dim=0).float().cpu().numpy()

        results = {
            'important_errors': important_errors_all,
            'non_important_errors': non_important_errors_all,
            'per_channel_error_mean': per_channel_error_mean,
            'per_channel_error_std': per_channel_error_std,
            'per_channel_error_abs_mean': per_channel_error_abs_mean,
            'importance_scores': importance_scores.float().cpu().numpy(),
            'important_mask': important_mask.cpu().numpy(),
            'n_important': important_mask.sum().item(),
            'n_non_important': (~important_mask).sum().item()
        }

        return results

    def visualize_rounding_errors(self, results, layer_name, output_dir):
        """Create comprehensive visualization of rounding errors"""

        important_errors = results['important_errors']
        non_important_errors = results['non_important_errors']
        per_channel_error_mean = results['per_channel_error_mean']
        per_channel_error_abs_mean = results['per_channel_error_abs_mean']
        importance_scores = results['importance_scores']
        important_mask = results['important_mask']

        # Create figure
        fig = plt.figure(figsize=(20, 12))

        # 1. Overall error distribution (signed)
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(important_errors, bins=100, alpha=0.6, label=f'Important channels (n={len(important_errors)})',
                 color='red', density=True)
        ax1.hist(non_important_errors, bins=100, alpha=0.6, label=f'Non-important channels (n={len(non_important_errors)})',
                 color='blue', density=True)
        ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Rounding Error (signed)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{layer_name}\nRounding Error Distribution (Signed)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. Absolute error distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(np.abs(important_errors), bins=100, alpha=0.6, label='Important channels',
                 color='red', density=True)
        ax2.hist(np.abs(non_important_errors), bins=100, alpha=0.6, label='Non-important channels',
                 color='blue', density=True)
        ax2.set_xlabel('|Rounding Error|')
        ax2.set_ylabel('Density')
        ax2.set_title('Absolute Rounding Error Distribution')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative distribution of absolute errors
        ax3 = plt.subplot(2, 3, 3)
        sorted_imp = np.sort(np.abs(important_errors))
        sorted_non = np.sort(np.abs(non_important_errors))
        ax3.plot(sorted_imp, np.linspace(0, 1, len(sorted_imp)), label='Important', linewidth=2, color='red')
        ax3.plot(sorted_non, np.linspace(0, 1, len(sorted_non)), label='Non-important', linewidth=2, color='blue')
        ax3.set_xlabel('|Rounding Error|')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('CDF of Absolute Rounding Error')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)

        # 4. Per-channel mean error vs importance
        ax4 = plt.subplot(2, 3, 4)
        colors = ['red' if m else 'blue' for m in important_mask]
        ax4.scatter(importance_scores, per_channel_error_mean, alpha=0.5, s=10, c=colors)
        ax4.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Channel Importance (L2 salience)')
        ax4.set_ylabel('Mean Rounding Error (per channel)')
        ax4.set_title('Mean Error vs Importance')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Important'),
            Patch(facecolor='blue', alpha=0.5, label='Non-important')
        ]
        ax4.legend(handles=legend_elements, loc='upper right')

        # 5. Per-channel absolute mean error vs importance
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(importance_scores, per_channel_error_abs_mean, alpha=0.5, s=10, c=colors)
        ax5.set_xlabel('Channel Importance (L2 salience)')
        ax5.set_ylabel('Mean |Rounding Error| (per channel)')
        ax5.set_title('Mean Absolute Error vs Importance')
        ax5.set_xscale('log')
        ax5.grid(True, alpha=0.3)
        ax5.legend(handles=legend_elements, loc='upper right')

        # 6. Box plot comparison
        ax6 = plt.subplot(2, 3, 6)
        data_to_plot = [important_errors, non_important_errors]
        bp = ax6.boxplot(data_to_plot, labels=['Important', 'Non-important'],
                         patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('blue')
        bp['boxes'][1].set_alpha(0.6)
        ax6.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax6.set_ylabel('Rounding Error (signed)')
        ax6.set_title('Error Distribution Comparison (Box Plot)')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{layer_name.replace(".", "_")}_rounding_errors.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
        plt.close()

        # Print statistics
        self.print_statistics(results, layer_name)

    def print_statistics(self, results, layer_name):
        """Print detailed statistics"""
        important_errors = results['important_errors']
        non_important_errors = results['non_important_errors']

        print(f"\n{'='*80}")
        print(f"Rounding Error Statistics for {layer_name}")
        print(f"{'='*80}")

        print(f"\nChannel Groups:")
        print(f"  Important channels: {results['n_important']}")
        print(f"  Non-important channels: {results['n_non_important']}")

        print(f"\nImportant Channels (signed errors):")
        print(f"  Mean error: {important_errors.mean():.6f}")
        print(f"  Std error: {important_errors.std():.6f}")
        print(f"  Min error: {important_errors.min():.6f}")
        print(f"  Max error: {important_errors.max():.6f}")
        print(f"  Mean |error|: {np.abs(important_errors).mean():.6f}")

        print(f"\nNon-important Channels (signed errors):")
        print(f"  Mean error: {non_important_errors.mean():.6f}")
        print(f"  Std error: {non_important_errors.std():.6f}")
        print(f"  Min error: {non_important_errors.min():.6f}")
        print(f"  Max error: {non_important_errors.max():.6f}")
        print(f"  Mean |error|: {np.abs(non_important_errors).mean():.6f}")

        print(f"\nComparison:")
        mean_diff = important_errors.mean() - non_important_errors.mean()
        abs_mean_diff = np.abs(important_errors).mean() - np.abs(non_important_errors).mean()
        print(f"  Difference in mean error (Important - Non-important): {mean_diff:.6f}")
        print(f"  Difference in mean |error|: {abs_mean_diff:.6f}")

        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(important_errors, non_important_errors)
        print(f"\n  T-test (means differ?): t={t_stat:.4f}, p={p_value:.4e}")

        ks_stat, ks_p = stats.ks_2samp(np.abs(important_errors), np.abs(non_important_errors))
        print(f"  KS test (distributions differ?): KS={ks_stat:.4f}, p={ks_p:.4e}")

        print(f"\n{'='*80}")


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    target_layer_id = 3  # Layer to analyze
    n_samples = 128
    max_seq_len = 512
    bits = 4
    group_size = 128
    top_k_ratio = 0.5  # Top 50% channels are "important"
    output_dir = "./visualizations/rounding_error_analysis"

    print(f"Model: {model_name}")
    print(f"Target layer: {target_layer_id}")
    print(f"Calibration samples: {n_samples}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create analyzer
    analyzer = RoundingErrorAnalyzer(model, tokenizer, device, bits=bits, group_size=group_size)

    # Load calibration data
    calibration_data = analyzer.load_calibration_data(n_samples, max_seq_len)

    # Register hooks for target layer
    target_module, layer_name = analyzer.register_hooks_for_layer(target_layer_id)

    # Collect activations
    analyzer.collect_activations(calibration_data)

    # Remove hooks
    analyzer.remove_hooks()

    # Compute importance scores
    print(f"\nComputing importance scores using L2 salience...")
    importance_scores = analyzer.compute_importance_l2(layer_name)

    if importance_scores is None:
        print("Error: No activation data collected!")
        return

    print(f"Importance scores shape: {importance_scores.shape}")
    print(f"Importance range: [{importance_scores.min():.4f}, {importance_scores.max():.4f}]")

    # Compute rounding errors
    print(f"\nComputing rounding errors (top {top_k_ratio*100:.0f}% as important)...")
    W = target_module.weight.data
    results = analyzer.compute_rounding_errors(W, importance_scores, top_k_ratio=top_k_ratio)

    # Visualize and report
    analyzer.visualize_rounding_errors(results, layer_name, output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
