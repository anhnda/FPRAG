"""
AWQ Scaling Visualization

Visualizes how AWQ per-input-channel scaling affects:
1. E[X[:,i]] - Mean activation per input channel (before/after dividing by scale)
2. E[X[:,i]²] - L2 salience per input channel (before/after dividing by scale)
3. W[j,i] - Weight distribution (before/after multiplying by scale)

Based on gw_awq_asym_l2_stats.py but focused on visualizing the scaling effect.

AWQ Scaling:
- Compute salience: s[i] = E[X[:,i]²]
- Find optimal α via grid search
- Scale factors: scale[i] = s[i]^α
- Apply: W_scaled[:, i] = W[:, i] * scale[i]
        X_scaled[:, i] = X[:, i] / scale[i]
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


class AWQScalingVisualizer:
    """Visualize AWQ scaling effects on activations and weights"""

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.activation_data = {}
        self.hooks = []

        print(f"\n[AWQ Scaling Visualizer Initialized]")
        print(f"  Bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")

    def load_calibration_data(self, n_samples=128, max_length=512):
        """Load calibration data from WikiText-2"""
        print(f"\nLoading {n_samples} calibration samples...")
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
        """Register hook for target layer"""
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
        print("\nCollecting activations...")
        self.model.eval()

        for input_ids in tqdm(calibration_data, desc="Forward passes"):
            _ = self.model(input_ids, use_cache=False)

    @torch.no_grad()
    def get_activation_statistics(self, layer_name):
        """
        Compute activation statistics: E[X[:,i]] and E[X[:,i]²]

        Returns:
            dict with 'mean', 'l2', and concatenated X tensor
        """
        if layer_name not in self.activation_data:
            return None

        X_list = self.activation_data[layer_name]

        # Concatenate all activations
        X_all = []
        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            X_all.append(x_flat)
        X = torch.cat(X_all, dim=0)  # [total_tokens, in_features]

        total_samples = X.shape[0]
        in_features = X.shape[1]

        print(f"\nActivation statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Input features: {in_features}")

        # Compute statistics
        mean_activation = X.mean(dim=0)  # E[X[:,i]]
        l2_salience = X.pow(2).mean(dim=0)  # E[X[:,i]²]

        return {
            'X': X,  # Keep for later use
            'mean': mean_activation.float().cpu().numpy(),
            'l2': l2_salience.float().cpu().numpy(),
            'n_samples': total_samples,
            'n_features': in_features
        }

    @torch.no_grad()
    def quantize_weight_groupwise(self, W):
        """Group-wise INT4 quantization"""
        out_features, in_features = W.shape
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in = n_groups * self.group_size

        if padded_in > in_features:
            W_padded = torch.zeros(out_features, padded_in, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Quantization parameters
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # Quantize
        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, 15)
        W_dequant = (W_int - zero_point) * scale

        W_dequant = W_dequant.reshape(out_features, padded_in)
        if padded_in > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    @torch.no_grad()
    def search_best_scale(self, X, W, salience):
        """
        Grid search for optimal α

        Args:
            X: Activations [n_samples, in_features]
            W: Weights [out_features, in_features]
            salience: Per-input-channel L2 salience [in_features]

        Returns:
            best_alpha, best_mse, scales
        """
        print("\nGrid search for optimal scaling...")

        # Move to device and convert to float32 for computation
        X_device = X.float().to(self.device)
        W_device = W.float().to(self.device)
        salience_device = salience.float().to(self.device)

        # Original output
        output_orig = torch.matmul(X_device, W_device.T)

        best_mse = float('inf')
        best_alpha = 0.0

        alphas = np.linspace(0, 1, self.n_grid)

        for alpha in tqdm(alphas, desc="Grid search"):
            # Compute scales: s[i]^alpha
            scales = torch.pow(salience_device, alpha)
            scales = scales.clamp(min=1e-8)

            # Scale weights: W[:, i] *= scales[i]
            W_scaled = W_device * scales.unsqueeze(0)

            # Quantize scaled weights
            W_quant = self.quantize_weight_groupwise(W_scaled)

            # Scale back: W_final[:, i] = W_quant[:, i] / scales[i]
            W_final = W_quant / scales.unsqueeze(0)

            # Compute output with quantized weights
            output_quant = torch.matmul(X_device, W_final.T)

            # MSE
            mse = torch.mean((output_orig - output_quant).pow(2)).item()

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        print(f"\nBest alpha: {best_alpha:.4f}")
        print(f"Best MSE: {best_mse:.6e}")

        # Compute best scales
        best_scales = torch.pow(salience_device, best_alpha)
        best_scales = best_scales.clamp(min=1e-8)

        return best_alpha, best_mse, best_scales

    def visualize_scaling_effects(self, stats, W, best_alpha, scales, layer_name, output_dir):
        """
        Visualize how AWQ scaling affects activations and weights

        Args:
            stats: Activation statistics dict
            W: Original weights
            best_alpha: Optimal alpha value
            scales: Scaling factors [in_features]
            layer_name: Layer name
            output_dir: Output directory
        """
        mean_orig = stats['mean']
        l2_orig = stats['l2']
        n_features = stats['n_features']

        # Compute scaled statistics
        scales_np = scales.cpu().numpy()
        mean_scaled = mean_orig / scales_np  # X_scaled = X / scale
        l2_scaled = l2_orig / (scales_np ** 2)  # E[(X/s)²] = E[X²]/s²

        # Get weight statistics (sample a few output channels)
        W_np = W.float().cpu().numpy()
        sample_out_channels = np.linspace(0, W_np.shape[0]-1, 5, dtype=int)

        # Create figure
        fig = plt.figure(figsize=(24, 20))

        # === E[X[:,i]] VISUALIZATION ===

        # 1. E[X[:,i]] - original vs scaled
        ax1 = plt.subplot(4, 3, 1)
        x_idx = np.arange(n_features)
        ax1.plot(x_idx, mean_orig, linewidth=0.5, alpha=0.7, label='Original', color='blue')
        ax1.plot(x_idx, mean_scaled, linewidth=0.5, alpha=0.7, label='Scaled (÷ scale)', color='red')
        ax1.set_xlabel('Input Channel Index')
        ax1.set_ylabel('E[X[:,i]]')
        ax1.set_title(f'{layer_name}\nE[X[:,i]] Before/After Scaling (α={best_alpha:.3f})')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. E[X[:,i]] distribution histogram
        ax2 = plt.subplot(4, 3, 2)
        ax2.hist(np.abs(mean_orig), bins=50, alpha=0.6, label='Original', color='blue', density=True)
        ax2.hist(np.abs(mean_scaled), bins=50, alpha=0.6, label='Scaled', color='red', density=True)
        ax2.set_xlabel('|E[X[:,i]]|')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of |E[X[:,i]]|')
        ax2.set_xscale('log')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. Ratio: scaled / original
        ax3 = plt.subplot(4, 3, 3)
        ratio_mean = mean_scaled / (mean_orig + 1e-10)
        ax3.scatter(x_idx, ratio_mean, s=1, alpha=0.5)
        ax3.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='No change')
        ax3.set_xlabel('Input Channel Index')
        ax3.set_ylabel('E[X_scaled] / E[X_orig]')
        ax3.set_title('Mean Activation Ratio (Scaled / Original)')
        ax3.set_yscale('log')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # === E[X[:,i]²] VISUALIZATION ===

        # 4. E[X[:,i]²] - original vs scaled
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(x_idx, l2_orig, linewidth=0.5, alpha=0.7, label='Original', color='blue')
        ax4.plot(x_idx, l2_scaled, linewidth=0.5, alpha=0.7, label='Scaled (÷ scale²)', color='red')
        ax4.set_xlabel('Input Channel Index')
        ax4.set_ylabel('E[X[:,i]²]')
        ax4.set_title('E[X[:,i]²] (L2 Salience) Before/After Scaling')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        # 5. E[X[:,i]²] distribution histogram
        ax5 = plt.subplot(4, 3, 5)
        ax5.hist(l2_orig, bins=50, alpha=0.6, label='Original', color='blue', density=True)
        ax5.hist(l2_scaled, bins=50, alpha=0.6, label='Scaled', color='red', density=True)
        ax5.set_xlabel('E[X[:,i]²]')
        ax5.set_ylabel('Density')
        ax5.set_title('Distribution of E[X[:,i]²]')
        ax5.set_xscale('log')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)

        # 6. Ratio: scaled / original
        ax6 = plt.subplot(4, 3, 6)
        ratio_l2 = l2_scaled / (l2_orig + 1e-10)
        ax6.scatter(x_idx, ratio_l2, s=1, alpha=0.5)
        ax6.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='No change')
        ax6.set_xlabel('Input Channel Index')
        ax6.set_ylabel('E[X²_scaled] / E[X²_orig]')
        ax6.set_title('L2 Salience Ratio (Scaled / Original)')
        ax6.set_yscale('log')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)

        # === SCALING FACTORS ===

        # 7. Scaling factors distribution
        ax7 = plt.subplot(4, 3, 7)
        ax7.plot(x_idx, scales_np, linewidth=0.5, alpha=0.7)
        ax7.set_xlabel('Input Channel Index')
        ax7.set_ylabel('Scale Factor')
        ax7.set_title(f'AWQ Scaling Factors (s^α where α={best_alpha:.3f})')
        ax7.set_yscale('log')
        ax7.grid(True, alpha=0.3)

        # 8. Scale vs L2 salience
        ax8 = plt.subplot(4, 3, 8)
        ax8.scatter(l2_orig, scales_np, s=2, alpha=0.5)
        ax8.set_xlabel('E[X[:,i]²] (L2 Salience)')
        ax8.set_ylabel('Scale Factor')
        ax8.set_title('Scaling Factor vs L2 Salience')
        ax8.set_xscale('log')
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)

        # 9. Scale distribution histogram
        ax9 = plt.subplot(4, 3, 9)
        ax9.hist(scales_np, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax9.set_xlabel('Scale Factor')
        ax9.set_ylabel('Count')
        ax9.set_title('Distribution of Scaling Factors')
        ax9.set_xscale('log')
        ax9.grid(True, alpha=0.3, axis='y')

        # === WEIGHT DISTRIBUTION ===

        # 10-12. Weight distributions for sample output channels
        for plot_idx, out_ch in enumerate(sample_out_channels[:3]):
            ax = plt.subplot(4, 3, 10 + plot_idx)

            W_orig_ch = W_np[out_ch, :]
            W_scaled_ch = W_orig_ch * scales_np

            ax.hist(W_orig_ch, bins=50, alpha=0.6, label='Original', color='blue', density=True)
            ax.hist(W_scaled_ch, bins=50, alpha=0.6, label='Scaled (× scale)', color='red', density=True)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Weight Distribution: Output Channel {out_ch}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{layer_name.replace(".", "_")}_awq_scaling.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
        plt.close()

        # Print statistics
        self.print_statistics(mean_orig, mean_scaled, l2_orig, l2_scaled, scales_np, W_np, best_alpha)

    def print_statistics(self, mean_orig, mean_scaled, l2_orig, l2_scaled, scales, W, best_alpha):
        """Print detailed statistics"""

        print(f"\n{'='*80}")
        print(f"AWQ Scaling Statistics (α={best_alpha:.4f})")
        print(f"{'='*80}")

        print(f"\n--- Scaling Factors ---")
        print(f"  Mean scale: {scales.mean():.6e}")
        print(f"  Std scale: {scales.std():.6e}")
        print(f"  Min scale: {scales.min():.6e}")
        print(f"  Max scale: {scales.max():.6e}")
        print(f"  Max/Min ratio: {scales.max() / scales.min():.2f}")

        print(f"\n--- E[X[:,i]] Statistics ---")
        print(f"  Original - Mean: {np.abs(mean_orig).mean():.6e}")
        print(f"  Original - Std: {np.abs(mean_orig).std():.6e}")
        print(f"  Scaled - Mean: {np.abs(mean_scaled).mean():.6e}")
        print(f"  Scaled - Std: {np.abs(mean_scaled).std():.6e}")
        print(f"  Mean ratio (scaled/orig): {np.abs(mean_scaled).mean() / np.abs(mean_orig).mean():.4f}")

        print(f"\n--- E[X[:,i]²] Statistics ---")
        print(f"  Original - Mean: {l2_orig.mean():.6e}")
        print(f"  Original - Std: {l2_orig.std():.6e}")
        print(f"  Scaled - Mean: {l2_scaled.mean():.6e}")
        print(f"  Scaled - Std: {l2_scaled.std():.6e}")
        print(f"  Mean ratio (scaled/orig): {l2_scaled.mean() / l2_orig.mean():.4f}")

        print(f"\n--- Weight Statistics ---")
        print(f"  Original - Mean |W|: {np.abs(W).mean():.6e}")
        print(f"  Original - Std W: {W.std():.6e}")
        W_scaled = W * scales[np.newaxis, :]
        print(f"  Scaled - Mean |W|: {np.abs(W_scaled).mean():.6e}")
        print(f"  Scaled - Std W: {W_scaled.std():.6e}")

        print(f"\n{'='*80}")


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    target_layer_id = 3
    n_samples = 128
    max_seq_len = 512
    bits = 4
    n_grid = 20
    group_size = 128
    output_dir = "./visualizations/awq_scaling_analysis"

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
    visualizer = AWQScalingVisualizer(model, tokenizer, device, bits=bits, n_grid=n_grid, group_size=group_size)

    # Load calibration data
    calibration_data = visualizer.load_calibration_data(n_samples, max_seq_len)

    # Register hooks
    target_module, layer_name = visualizer.register_hooks_for_layer(target_layer_id)

    # Collect activations
    visualizer.collect_activations(calibration_data)

    # Remove hooks
    visualizer.remove_hooks()

    # Get activation statistics
    stats = visualizer.get_activation_statistics(layer_name)

    if stats is None:
        print("Error: No activation data collected!")
        return

    # Get weights
    W = target_module.weight.data

    # Convert statistics to tensors
    salience = torch.from_numpy(stats['l2']).to(device)

    # Grid search for best alpha
    best_alpha, best_mse, best_scales = visualizer.search_best_scale(stats['X'], W, salience)

    # Visualize
    visualizer.visualize_scaling_effects(stats, W, best_alpha, best_scales, layer_name, output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
