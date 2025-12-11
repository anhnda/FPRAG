"""
Visualize Channel Importance Distributions

This script visualizes two types of channel importance metrics:
1. E[X[:,j]] - Mean activation per channel (L1-style)
2. E[X[:,j]²] - L2 salience per channel (AWQ-style)

Shows both original order and sorted order to understand importance distribution.

Target: Layer 3 gate_proj (XW before SiLU)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)


class ImportanceVisualizer:
    """Visualize channel importance distributions"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activation_data = {}
        self.hooks = []

        print(f"\n[Importance Visualizer Initialized]")
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
    def compute_importances(self, layer_name):
        """
        Compute two types of importance:
        1. E[X[:,j]] - Mean activation
        2. E[X[:,j]²] - L2 salience

        Returns:
            dict with 'mean' and 'l2' importance scores
        """
        if layer_name not in self.activation_data:
            return None

        X_list = self.activation_data[layer_name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        print(f"\nComputing importance scores...")
        print(f"  Total samples: {total_samples}")
        print(f"  Input features: {in_features}")

        # Accumulate statistics
        mean_sum = torch.zeros(in_features)
        l2_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            mean_sum += x_flat.sum(dim=0)
            l2_sum += x_flat.pow(2).sum(dim=0)

        # Compute importance scores
        importance_mean = mean_sum / total_samples  # E[X[:,j]]
        importance_l2 = l2_sum / total_samples      # E[X[:,j]²]

        return {
            'mean': importance_mean.float().cpu().numpy(),
            'l2': importance_l2.float().cpu().numpy(),
            'n_channels': in_features,
            'n_samples': total_samples
        }

    def visualize_importances(self, importances, layer_name, output_dir):
        """Create comprehensive visualization of importance distributions"""

        mean_importance = importances['mean']
        l2_importance = importances['l2']
        n_channels = importances['n_channels']

        # Sort by importance
        mean_sorted_idx = np.argsort(mean_importance)[::-1]  # Descending
        l2_sorted_idx = np.argsort(l2_importance)[::-1]

        mean_sorted = mean_importance[mean_sorted_idx]
        l2_sorted = l2_importance[l2_sorted_idx]

        # Create explicit x-axis arrays
        x_original = np.arange(n_channels)
        x_sorted = np.arange(n_channels)

        # Create figure with 3x3 layout
        fig = plt.figure(figsize=(24, 18))

        # === E[X[:,j]] VISUALIZATIONS ===

        # 1. Original order - E[X[:,j]]
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(x_original, mean_importance, linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Channel Index (original order)')
        ax1.set_ylabel('E[X[:,j]]')
        ax1.set_title(f'{layer_name}\nMean Activation - Original Order')
        ax1.set_xlim(0, n_channels - 1)
        ax1.grid(True, alpha=0.3)

        # 2. Sorted order - E[X[:,j]]
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(x_sorted, mean_sorted, linewidth=0.8, color='blue')
        ax2.set_xlabel('Channel Rank (sorted by E[X[:,j]])')
        ax2.set_ylabel('E[X[:,j]]')
        ax2.set_title('Mean Activation - Sorted (Descending)')
        ax2.set_xlim(0, n_channels - 1)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. Distribution histogram - E[X[:,j]]
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(mean_importance, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('E[X[:,j]]')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Mean Activations')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xscale('log')

        # === E[X[:,j]²] VISUALIZATIONS ===

        # 4. Original order - E[X[:,j]²]
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(x_original, l2_importance, linewidth=0.5, alpha=0.7, color='red')
        ax4.set_xlabel('Channel Index (original order)')
        ax4.set_ylabel('E[X[:,j]²]')
        ax4.set_title('L2 Salience - Original Order')
        ax4.set_xlim(0, n_channels - 1)
        ax4.grid(True, alpha=0.3)

        # 5. Sorted order - E[X[:,j]²]
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(x_sorted, l2_sorted, linewidth=0.8, color='red')
        ax5.set_xlabel('Channel Rank (sorted by E[X[:,j]²])')
        ax5.set_ylabel('E[X[:,j]²]')
        ax5.set_title('L2 Salience - Sorted (Descending)')
        ax5.set_xlim(0, n_channels - 1)
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')

        # 6. Distribution histogram - E[X[:,j]²]
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(l2_importance, bins=100, alpha=0.7, color='red', edgecolor='black')
        ax6.set_xlabel('E[X[:,j]²]')
        ax6.set_ylabel('Count')
        ax6.set_title('Distribution of L2 Salience')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xscale('log')

        # === COMPARISON VISUALIZATIONS ===

        # 7. Cumulative importance (normalized)
        ax7 = plt.subplot(3, 3, 7)
        mean_cumsum = np.cumsum(mean_sorted) / np.sum(mean_sorted)
        l2_cumsum = np.cumsum(l2_sorted) / np.sum(l2_sorted)

        x_percent = np.arange(n_channels) / n_channels * 100
        ax7.plot(x_percent, mean_cumsum * 100, label='E[X[:,j]]', linewidth=2, color='blue')
        ax7.plot(x_percent, l2_cumsum * 100, label='E[X[:,j]²]', linewidth=2, color='red')
        ax7.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50%')
        ax7.axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='80%')
        ax7.axhline(95, color='lightgray', linestyle='--', linewidth=1, alpha=0.5, label='95%')
        ax7.set_xlabel('Top % of Channels')
        ax7.set_ylabel('Cumulative % of Total Importance')
        ax7.set_title('Cumulative Importance Distribution')
        ax7.legend(loc='lower right')
        ax7.grid(True, alpha=0.3)

        # 8. Top-k importance concentration
        ax8 = plt.subplot(3, 3, 8)
        k_values = [10, 50, 100, 200, 500, 1000]
        mean_top_k = []
        l2_top_k = []

        for k in k_values:
            if k <= n_channels:
                mean_top_k.append(np.sum(mean_sorted[:k]) / np.sum(mean_sorted) * 100)
                l2_top_k.append(np.sum(l2_sorted[:k]) / np.sum(l2_sorted) * 100)
            else:
                mean_top_k.append(100)
                l2_top_k.append(100)

        x_pos = np.arange(len(k_values))
        width = 0.35

        ax8.bar(x_pos - width/2, mean_top_k, width, label='E[X[:,j]]', color='blue', alpha=0.7)
        ax8.bar(x_pos + width/2, l2_top_k, width, label='E[X[:,j]²]', color='red', alpha=0.7)
        ax8.set_xlabel('Top-k Channels')
        ax8.set_ylabel('% of Total Importance')
        ax8.set_title('Importance Concentration in Top-k Channels')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels([f'Top-{k}' for k in k_values], rotation=45)
        ax8.legend(loc='upper left')
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. Correlation between E[X] and E[X²]
        ax9 = plt.subplot(3, 3, 9)
        ax9.scatter(mean_importance, l2_importance, alpha=0.3, s=5)
        ax9.set_xlabel('E[X[:,j]]')
        ax9.set_ylabel('E[X[:,j]²]')
        ax9.set_title('Correlation: Mean vs L2 Salience')
        ax9.set_xscale('log')
        ax9.set_yscale('log')
        ax9.grid(True, alpha=0.3)

        # Add correlation coefficient
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(mean_importance, l2_importance)
        ax9.text(0.05, 0.95, f'Spearman ρ = {corr:.4f}\np = {p_value:.2e}',
                transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{layer_name.replace(".", "_")}_importance_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
        plt.close()

        # Print statistics
        self.print_statistics(importances, mean_sorted, l2_sorted)

    def print_statistics(self, importances, mean_sorted, l2_sorted):
        """Print detailed statistics"""

        mean_importance = importances['mean']
        l2_importance = importances['l2']
        n_channels = importances['n_channels']

        print(f"\n{'='*80}")
        print(f"Channel Importance Statistics")
        print(f"{'='*80}")

        print(f"\nTotal Channels: {n_channels}")
        print(f"Total Samples: {importances['n_samples']}")

        print(f"\n--- E[X[:,j]] (Mean Activation) ---")
        print(f"  Mean: {mean_importance.mean():.6e}")
        print(f"  Std: {mean_importance.std():.6e}")
        print(f"  Min: {mean_importance.min():.6e}")
        print(f"  Max: {mean_importance.max():.6e}")
        print(f"  Median: {np.median(mean_importance):.6e}")

        print(f"\n--- E[X[:,j]²] (L2 Salience) ---")
        print(f"  Mean: {l2_importance.mean():.6e}")
        print(f"  Std: {l2_importance.std():.6e}")
        print(f"  Min: {l2_importance.min():.6e}")
        print(f"  Max: {l2_importance.max():.6e}")
        print(f"  Median: {np.median(l2_importance):.6e}")

        print(f"\n--- Concentration Analysis ---")

        # For E[X]
        total_mean = np.sum(mean_sorted)
        top1_mean = mean_sorted[0] / total_mean * 100
        top10_mean = np.sum(mean_sorted[:10]) / total_mean * 100
        top100_mean = np.sum(mean_sorted[:100]) / total_mean * 100
        top50pct_mean = np.sum(mean_sorted[:n_channels//2]) / total_mean * 100

        print(f"\nE[X[:,j]] Concentration:")
        print(f"  Top-1 channel: {top1_mean:.2f}% of total")
        print(f"  Top-10 channels: {top10_mean:.2f}% of total")
        print(f"  Top-100 channels: {top100_mean:.2f}% of total")
        print(f"  Top-50% channels: {top50pct_mean:.2f}% of total")

        # For E[X²]
        total_l2 = np.sum(l2_sorted)
        top1_l2 = l2_sorted[0] / total_l2 * 100
        top10_l2 = np.sum(l2_sorted[:10]) / total_l2 * 100
        top100_l2 = np.sum(l2_sorted[:100]) / total_l2 * 100
        top50pct_l2 = np.sum(l2_sorted[:n_channels//2]) / total_l2 * 100

        print(f"\nE[X[:,j]²] Concentration:")
        print(f"  Top-1 channel: {top1_l2:.2f}% of total")
        print(f"  Top-10 channels: {top10_l2:.2f}% of total")
        print(f"  Top-100 channels: {top100_l2:.2f}% of total")
        print(f"  Top-50% channels: {top50pct_l2:.2f}% of total")

        # Ratio analysis
        print(f"\n--- Ratio Analysis ---")
        print(f"  Max/Min ratio (E[X]): {mean_sorted[0] / (mean_sorted[-1] + 1e-10):.2e}")
        print(f"  Max/Min ratio (E[X²]): {l2_sorted[0] / (l2_sorted[-1] + 1e-10):.2e}")
        print(f"  Max/Median ratio (E[X]): {mean_sorted[0] / (np.median(mean_sorted) + 1e-10):.2f}")
        print(f"  Max/Median ratio (E[X²]): {l2_sorted[0] / (np.median(l2_sorted) + 1e-10):.2f}")

        # Correlation
        from scipy.stats import spearmanr, pearsonr
        spearman_corr, spearman_p = spearmanr(mean_importance, l2_importance)
        pearson_corr, pearson_p = pearsonr(mean_importance, l2_importance)

        print(f"\n--- Correlation between E[X] and E[X²] ---")
        print(f"  Spearman ρ: {spearman_corr:.4f} (p={spearman_p:.2e})")
        print(f"  Pearson r: {pearson_corr:.4f} (p={pearson_p:.2e})")

        print(f"\n{'='*80}")


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    target_layer_id = 3  # Layer to analyze
    n_samples = 128
    max_seq_len = 512
    output_dir = "./visualizations/importance_distributions"

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

    # Create visualizer
    visualizer = ImportanceVisualizer(model, tokenizer, device)

    # Load calibration data
    calibration_data = visualizer.load_calibration_data(n_samples, max_seq_len)

    # Register hooks for target layer
    target_module, layer_name = visualizer.register_hooks_for_layer(target_layer_id)

    # Collect activations
    visualizer.collect_activations(calibration_data)

    # Remove hooks
    visualizer.remove_hooks()

    # Compute importance scores
    importances = visualizer.compute_importances(layer_name)

    if importances is None:
        print("Error: No activation data collected!")
        return

    # Visualize
    visualizer.visualize_importances(importances, layer_name, output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
