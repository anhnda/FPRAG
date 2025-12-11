"""
MSE Error Analysis by Channel Importance

This script analyzes the reconstruction error (MSE) of XW when quantizing weights,
comparing important vs non-important channels based on L2 salience scores.

Key Analysis:
1. Compute importance scores using L2 salience: E[X²]
2. Split channels into two halves: top 50% (important) vs bottom 50% (non-important)
3. Quantize weights to INT4
4. Compute MSE of XW for each group: E[(XW_original - XW_quantized)²]
5. Compare error distributions and statistics

Output:
- MSE distribution plots (important vs non-important channels)
- Mean/std MSE statistics for each group
- Per-channel MSE analysis
- Cumulative error plots
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


class MSEErrorAnalyzer:
    """Analyzes MSE reconstruction errors by channel importance"""

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size

        # Storage
        self.activation_data = {}
        self.hooks = []

        print(f"\n[MSE Error Analyzer Initialized]")
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
        Compute per-input-channel importance using L2 salience: E[X²]

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
    def quantize_weight_groupwise(self, W):
        """
        Group-wise asymmetric INT4 quantization

        Args:
            W: Weight tensor [out_features, in_features]

        Returns:
            W_quant: Quantized and dequantized weights
        """
        out_features, in_features = W.shape

        # Pad weights for group-wise quantization
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

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

        # Quantize
        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, n_levels - 1)

        # Dequantize
        W_dequant_grouped = (W_int - zero_point) * scale

        # Reshape back
        W_dequant = W_dequant_grouped.reshape(out_features, padded_in_features)

        # Remove padding
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    @torch.no_grad()
    def compute_mse_by_importance(self, layer_name, W_original, W_quantized, importance_scores):
        """
        Compute MSE of XW for important vs non-important channels

        Args:
            layer_name: Name of the layer
            W_original: Original weights [out_features, in_features]
            W_quantized: Quantized weights [out_features, in_features]
            importance_scores: Per-input-channel importance [in_features]

        Returns:
            Dictionary with MSE statistics
        """
        # Get activation data
        X_list = self.activation_data[layer_name]

        # Concatenate all activations
        X_all = []
        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])  # [n_tokens, in_features]
            X_all.append(x_flat)
        X = torch.cat(X_all, dim=0)  # [total_tokens, in_features]

        print(f"Total tokens: {X.shape[0]}")
        print(f"Input features: {X.shape[1]}")

        # Determine important vs non-important channels
        in_features = importance_scores.shape[0]
        k = in_features // 2  # Split in half

        top_k_indices = torch.topk(importance_scores, k).indices
        important_mask = torch.zeros(in_features, dtype=torch.bool)
        important_mask[top_k_indices] = True

        # Move to device for computation
        X_device = X.to(self.device)
        W_original_device = W_original.to(self.device)
        W_quantized_device = W_quantized.to(self.device)

        # Compute outputs: XW^T
        # X: [n_tokens, in_features]
        # W: [out_features, in_features]
        # XW^T: [n_tokens, out_features]
        output_original = torch.matmul(X_device, W_original_device.T)
        output_quantized = torch.matmul(X_device, W_quantized_device.T)

        # Compute squared error per output element
        squared_error = (output_original - output_quantized).pow(2)  # [n_tokens, out_features]

        # Now we need to attribute errors to input channels
        # For each output channel, compute which input channels contributed
        # Since output[i,j] = sum_k(X[i,k] * W[j,k]), the error contribution from input channel k is complex
        #
        # Instead, let's compute MSE by analyzing the weight error contribution
        # Error in output: X @ (W_orig - W_quant)^T
        # For channel group analysis, we can mask the weight matrix

        W_error = W_original_device - W_quantized_device  # [out_features, in_features]

        # Compute contribution to output error from important channels
        W_error_important = W_error.clone()
        W_error_important[:, ~important_mask] = 0  # Zero out non-important channels
        output_error_important = torch.matmul(X_device, W_error_important.T)
        mse_important = output_error_important.pow(2).mean().item()

        # Compute contribution to output error from non-important channels
        W_error_non_important = W_error.clone()
        W_error_non_important[:, important_mask] = 0  # Zero out important channels
        output_error_non_important = torch.matmul(X_device, W_error_non_important.T)
        mse_non_important = output_error_non_important.pow(2).mean().item()

        # Per-channel MSE contribution
        # For each input channel, compute its contribution to output MSE
        per_channel_mse = torch.zeros(in_features)

        for ch_idx in tqdm(range(in_features), desc="Computing per-channel MSE"):
            W_error_ch = W_error.clone()
            W_error_ch[:, torch.arange(in_features) != ch_idx] = 0
            output_error_ch = torch.matmul(X_device, W_error_ch.T)
            per_channel_mse[ch_idx] = output_error_ch.pow(2).mean().item()

        # Overall MSE
        total_mse = squared_error.mean().item()

        # Separate per-channel MSE by importance
        per_channel_mse_important = per_channel_mse[important_mask].float().cpu().numpy()
        per_channel_mse_non_important = per_channel_mse[~important_mask].float().cpu().numpy()

        results = {
            'mse_important': mse_important,
            'mse_non_important': mse_non_important,
            'total_mse': total_mse,
            'per_channel_mse': per_channel_mse.float().cpu().numpy(),
            'per_channel_mse_important': per_channel_mse_important,
            'per_channel_mse_non_important': per_channel_mse_non_important,
            'importance_scores': importance_scores.float().cpu().numpy(),
            'important_mask': important_mask.cpu().numpy(),
            'n_important': important_mask.sum().item(),
            'n_non_important': (~important_mask).sum().item()
        }

        return results

    def visualize_mse_errors(self, results, layer_name, output_dir):
        """Create comprehensive visualization of MSE errors"""

        per_channel_mse_imp = results['per_channel_mse_important']
        per_channel_mse_non = results['per_channel_mse_non_important']
        per_channel_mse_all = results['per_channel_mse']
        importance_scores = results['importance_scores']
        important_mask = results['important_mask']

        # Create figure
        fig = plt.figure(figsize=(20, 12))

        # 1. Per-channel MSE distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(per_channel_mse_imp, bins=50, alpha=0.6,
                 label=f'Important channels (top 50%, n={len(per_channel_mse_imp)})',
                 color='red', density=True)
        ax1.hist(per_channel_mse_non, bins=50, alpha=0.6,
                 label=f'Non-important channels (bottom 50%, n={len(per_channel_mse_non)})',
                 color='blue', density=True)
        ax1.set_xlabel('Per-Channel MSE Contribution')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{layer_name}\nPer-Channel MSE Distribution')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. Log-scale MSE distribution
        ax2 = plt.subplot(2, 3, 2)
        # Filter out zeros for log scale
        imp_nonzero = per_channel_mse_imp[per_channel_mse_imp > 0]
        non_nonzero = per_channel_mse_non[per_channel_mse_non > 0]

        ax2.hist(np.log10(imp_nonzero + 1e-10), bins=50, alpha=0.6,
                 label='Important channels', color='red', density=True)
        ax2.hist(np.log10(non_nonzero + 1e-10), bins=50, alpha=0.6,
                 label='Non-important channels', color='blue', density=True)
        ax2.set_xlabel('log10(Per-Channel MSE)')
        ax2.set_ylabel('Density')
        ax2.set_title('MSE Distribution (Log Scale)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative MSE distribution
        ax3 = plt.subplot(2, 3, 3)
        sorted_imp = np.sort(per_channel_mse_imp)
        sorted_non = np.sort(per_channel_mse_non)
        ax3.plot(sorted_imp, np.linspace(0, 1, len(sorted_imp)),
                 label='Important', linewidth=2, color='red')
        ax3.plot(sorted_non, np.linspace(0, 1, len(sorted_non)),
                 label='Non-important', linewidth=2, color='blue')
        ax3.set_xlabel('Per-Channel MSE Contribution')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('CDF of Per-Channel MSE')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)

        # 4. MSE vs Importance scatter
        ax4 = plt.subplot(2, 3, 4)
        colors = ['red' if m else 'blue' for m in important_mask]
        ax4.scatter(importance_scores, per_channel_mse_all, alpha=0.5, s=10, c=colors)
        ax4.set_xlabel('Channel Importance (L2 salience)')
        ax4.set_ylabel('MSE Contribution')
        ax4.set_title('MSE vs Importance')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Important (top 50%)'),
            Patch(facecolor='blue', alpha=0.5, label='Non-important (bottom 50%)')
        ]
        ax4.legend(handles=legend_elements, loc='upper left')

        # 5. Bar chart comparison
        ax5 = plt.subplot(2, 3, 5)
        categories = ['Important\nChannels', 'Non-important\nChannels']
        mse_values = [results['mse_important'], results['mse_non_important']]
        colors_bar = ['red', 'blue']

        bars = ax5.bar(categories, mse_values, color=colors_bar, alpha=0.6, edgecolor='black')
        ax5.set_ylabel('Mean Squared Error')
        ax5.set_title('Total MSE Contribution by Channel Group')
        ax5.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, mse_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.6f}', ha='center', va='bottom', fontsize=10)

        # 6. Box plot comparison
        ax6 = plt.subplot(2, 3, 6)
        data_to_plot = [per_channel_mse_imp, per_channel_mse_non]
        bp = ax6.boxplot(data_to_plot, labels=['Important', 'Non-important'],
                         patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('blue')
        bp['boxes'][1].set_alpha(0.6)
        ax6.set_ylabel('Per-Channel MSE Contribution')
        ax6.set_title('MSE Distribution Comparison (Box Plot)')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{layer_name.replace(".", "_")}_mse_errors.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
        plt.close()

        # Print statistics
        self.print_statistics(results, layer_name)

    def print_statistics(self, results, layer_name):
        """Print detailed statistics"""

        print(f"\n{'='*80}")
        print(f"MSE Error Statistics for {layer_name}")
        print(f"{'='*80}")

        print(f"\nChannel Groups:")
        print(f"  Important channels (top 50%): {results['n_important']}")
        print(f"  Non-important channels (bottom 50%): {results['n_non_important']}")

        print(f"\nTotal MSE Contribution:")
        print(f"  Important channels: {results['mse_important']:.6e}")
        print(f"  Non-important channels: {results['mse_non_important']:.6e}")
        print(f"  Total: {results['total_mse']:.6e}")

        print(f"\nPer-Channel MSE Statistics:")
        print(f"\nImportant Channels:")
        imp_mse = results['per_channel_mse_important']
        print(f"  Mean MSE: {imp_mse.mean():.6e}")
        print(f"  Std MSE: {imp_mse.std():.6e}")
        print(f"  Min MSE: {imp_mse.min():.6e}")
        print(f"  Max MSE: {imp_mse.max():.6e}")
        print(f"  Median MSE: {np.median(imp_mse):.6e}")

        print(f"\nNon-important Channels:")
        non_mse = results['per_channel_mse_non_important']
        print(f"  Mean MSE: {non_mse.mean():.6e}")
        print(f"  Std MSE: {non_mse.std():.6e}")
        print(f"  Min MSE: {non_mse.min():.6e}")
        print(f"  Max MSE: {non_mse.max():.6e}")
        print(f"  Median MSE: {np.median(non_mse):.6e}")

        print(f"\nComparison:")
        ratio = results['mse_important'] / (results['mse_non_important'] + 1e-10)
        print(f"  MSE ratio (Important / Non-important): {ratio:.4f}")

        per_channel_ratio = imp_mse.mean() / (non_mse.mean() + 1e-10)
        print(f"  Per-channel mean MSE ratio: {per_channel_ratio:.4f}")

        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(imp_mse, non_mse)
        print(f"\n  T-test (means differ?): t={t_stat:.4f}, p={p_value:.4e}")

        ks_stat, ks_p = stats.ks_2samp(imp_mse, non_mse)
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
    output_dir = "./visualizations/mse_error_parts"

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
    analyzer = MSEErrorAnalyzer(model, tokenizer, device, bits=bits, group_size=group_size)

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
    print(f"Importance range: [{importance_scores.min():.4e}, {importance_scores.max():.4e}]")

    # Quantize weights
    print(f"\nQuantizing weights to INT{bits}...")
    W_original = target_module.weight.data
    W_quantized = analyzer.quantize_weight_groupwise(W_original)

    # Compute MSE errors
    print(f"\nComputing MSE errors (top 50% vs bottom 50%)...")
    results = analyzer.compute_mse_by_importance(layer_name, W_original, W_quantized, importance_scores)

    # Visualize and report
    analyzer.visualize_mse_errors(results, layer_name, output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
