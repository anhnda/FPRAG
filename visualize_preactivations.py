import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os


class PreActivationAnalyzer:
    """
    Analyzer to capture and visualize pre-activation distributions
    to test the Fast-R-PRAQ hypothesis.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.preactivation_data = defaultdict(list)
        self.weight_data = {}
        self.hooks = []

    def register_hooks(self, layer_names=None):
        """
        Register forward hooks to capture pre-activations.

        Args:
            layer_names: List of specific layer names to hook, or None for all linear layers
        """
        def get_hook(name):
            def hook(module, input, output):
                # Capture input to the linear layer (pre-activation)
                if isinstance(input, tuple):
                    inp = input[0].detach()
                else:
                    inp = input.detach()

                # Compute pre-activation: Z = X @ W^T + b
                W = module.weight.data
                b = module.bias.data if module.bias is not None else None

                # Reshape input to [batch * seq, hidden]
                inp_flat = inp.reshape(-1, inp.shape[-1])

                # Compute pre-activation
                Z = torch.matmul(inp_flat, W.t())
                if b is not None:
                    Z = Z + b

                # Store pre-activations (move to CPU to save memory)
                self.preactivation_data[name].append(Z.cpu())

            return hook

        # Register hooks for specified layers or all linear layers
        hooked_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if layer_names is None or name in layer_names:
                    handle = module.register_forward_hook(get_hook(name))
                    self.hooks.append(handle)
                    # Store weight magnitude for analysis
                    self.weight_data[name] = module.weight.data.abs().mean(dim=1).cpu()
                    hooked_count += 1

        print(f"Registered hooks on {hooked_count} layers")

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @torch.no_grad()
    def collect_data(self, texts, max_samples=100):
        """
        Run calibration data through the model to collect pre-activations.

        Args:
            texts: List of text samples
            max_samples: Maximum number of samples to process
        """
        print(f"Collecting pre-activation data from {min(max_samples, len(texts))} samples...")
        self.model.eval()

        for i, text in enumerate(tqdm(texts[:max_samples], desc="Processing")):
            if i >= max_samples:
                break

            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass to collect activations
                self.model(**inputs)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        print("Data collection complete!")

    def compute_statistics(self):
        """
        Compute statistics for each layer.

        Returns:
            Dictionary with layer statistics
        """
        stats = {}

        for layer_name, preacts_list in self.preactivation_data.items():
            if len(preacts_list) == 0:
                continue

            # Concatenate all pre-activations
            Z = torch.cat(preacts_list, dim=0)  # [total_tokens, out_features]

            # Compute per-channel statistics
            z_mean = Z.mean(dim=0)  # [out_features]
            z_std = Z.std(dim=0)    # [out_features]
            z_min = Z.min(dim=0)[0]
            z_max = Z.max(dim=0)[0]

            # Compute percentiles
            z_p05 = torch.quantile(Z, 0.05, dim=0)
            z_p95 = torch.quantile(Z, 0.95, dim=0)

            # Risk metric: channels with large negative mean but high variance
            # These are the "dangerous" channels that Fast-R-PRAQ cares about
            z_risk_upper = z_mean + 3 * z_std  # 3-sigma upper bound

            stats[layer_name] = {
                'mean': z_mean.numpy(),
                'std': z_std.numpy(),
                'min': z_min.numpy(),
                'max': z_max.numpy(),
                'p05': z_p05.numpy(),
                'p95': z_p95.numpy(),
                'risk_upper': z_risk_upper.numpy(),
                'weight_mag': self.weight_data[layer_name].numpy(),
                'raw_data': Z.numpy()
            }

        return stats

    def analyze_hypothesis(self, stats, activation_threshold=-3.0):
        """
        Test the Fast-R-PRAQ hypothesis: Are there channels with large negative
        pre-activations that are risky due to quantization noise?

        Args:
            stats: Statistics dictionary from compute_statistics()
            activation_threshold: Threshold below which activations are "dead" (e.g., -3.0 for SiLU)

        Returns:
            Analysis results
        """
        analysis = {}

        for layer_name, layer_stats in stats.items():
            z_mean = layer_stats['mean']
            z_std = layer_stats['std']
            z_risk_upper = layer_stats['risk_upper']
            w_mag = layer_stats['weight_mag']

            # Identify "dead" channels (mean below threshold)
            dead_mask = z_mean < activation_threshold

            # Identify "risky dead" channels (dead but might activate with noise)
            # Risk condition: mean is negative, but mean + noise could cross threshold
            risky_mask = (z_mean < activation_threshold) & (z_risk_upper > activation_threshold)

            # Identify channels with high weight magnitude
            high_weight_mask = w_mag > np.percentile(w_mag, 75)

            # The critical case: Dead + Risky + High Weight
            critical_mask = risky_mask & high_weight_mask

            analysis[layer_name] = {
                'total_channels': len(z_mean),
                'dead_channels': dead_mask.sum(),
                'risky_channels': risky_mask.sum(),
                'high_weight_channels': high_weight_mask.sum(),
                'critical_channels': critical_mask.sum(),
                'critical_percentage': (critical_mask.sum() / len(z_mean)) * 100,
                'dead_mask': dead_mask,
                'risky_mask': risky_mask,
                'critical_mask': critical_mask
            }

        return analysis


def visualize_layer(layer_name, stats, analysis, save_dir="./visualizations"):
    """
    Create comprehensive visualizations for a single layer.

    Args:
        layer_name: Name of the layer
        stats: Statistics for the layer
        analysis: Analysis results for the layer
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    z_mean = stats['mean']
    z_std = stats['std']
    z_risk_upper = stats['risk_upper']
    w_mag = stats['weight_mag']

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # Sanitize layer name for filename
    safe_name = layer_name.replace('/', '_').replace('.', '_')

    # 1. Distribution of mean pre-activations
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(z_mean, bins=100, alpha=0.7, edgecolor='black')
    ax1.axvline(-3.0, color='red', linestyle='--', label='SiLU threshold (~-3)')
    ax1.set_xlabel('Mean Pre-activation')
    ax1.set_ylabel('Number of Channels')
    ax1.set_title(f'Distribution of Mean Pre-activations\n{layer_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of std pre-activations
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(z_std, bins=100, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Std Pre-activation')
    ax2.set_ylabel('Number of Channels')
    ax2.set_title('Distribution of Std Pre-activations')
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: Mean vs Std (with risk regions)
    ax3 = plt.subplot(2, 3, 3)

    # Color code by risk level
    dead_mask = analysis['dead_mask']
    risky_mask = analysis['risky_mask']
    critical_mask = analysis['critical_mask']

    # Plot normal channels
    normal_mask = ~dead_mask
    ax3.scatter(z_mean[normal_mask], z_std[normal_mask],
               alpha=0.3, s=10, c='blue', label='Active')

    # Plot dead but safe channels
    dead_safe_mask = dead_mask & ~risky_mask
    ax3.scatter(z_mean[dead_safe_mask], z_std[dead_safe_mask],
               alpha=0.5, s=10, c='gray', label='Dead (Safe)')

    # Plot risky channels
    risky_noncritical_mask = risky_mask & ~critical_mask
    ax3.scatter(z_mean[risky_noncritical_mask], z_std[risky_noncritical_mask],
               alpha=0.6, s=20, c='orange', label='Risky (Low Weight)')

    # Plot critical channels (dead + risky + high weight)
    ax3.scatter(z_mean[critical_mask], z_std[critical_mask],
               alpha=0.8, s=50, c='red', marker='X', label='CRITICAL (High Risk)')

    ax3.axvline(-3.0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Mean Pre-activation')
    ax3.set_ylabel('Std Pre-activation')
    ax3.set_title(f'Mean vs Std (Risk Analysis)\nCritical: {analysis["critical_channels"]} / {analysis["total_channels"]} ({analysis["critical_percentage"]:.2f}%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Weight magnitude vs Mean pre-activation
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(z_mean, w_mag, c=z_std, cmap='viridis', alpha=0.5, s=10)
    ax4.axvline(-3.0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Mean Pre-activation')
    ax4.set_ylabel('Weight Magnitude')
    ax4.set_title('Weight Magnitude vs Mean Pre-activation\n(color = std)')
    plt.colorbar(scatter, ax=ax4, label='Std')
    ax4.grid(True, alpha=0.3)

    # 5. Risk-adjusted upper bound distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(z_risk_upper, bins=100, alpha=0.7, color='purple', edgecolor='black')
    ax5.axvline(-3.0, color='red', linestyle='--', label='SiLU threshold')
    ax5.set_xlabel('Risk-Adjusted Upper Bound (mean + 3*std)')
    ax5.set_ylabel('Number of Channels')
    ax5.set_title('Distribution of Risk-Adjusted Upper Bounds')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Example pre-activation histograms for critical channels
    ax6 = plt.subplot(2, 3, 6)

    if critical_mask.sum() > 0:
        # Get indices of critical channels
        critical_indices = np.where(critical_mask)[0]

        # Plot histograms for up to 3 critical channels
        for i, idx in enumerate(critical_indices[:3]):
            channel_data = stats['raw_data'][:, idx]
            ax6.hist(channel_data, bins=50, alpha=0.5,
                    label=f'Ch {idx}: μ={z_mean[idx]:.2f}, σ={z_std[idx]:.2f}')

        ax6.axvline(-3.0, color='red', linestyle='--', alpha=0.5, label='SiLU threshold')
        ax6.set_xlabel('Pre-activation Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Pre-activation Distributions of Critical Channels')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No critical channels found',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Pre-activation Distributions of Critical Channels')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization for {layer_name}")


def create_summary_visualization(analysis, save_dir="./visualizations"):
    """
    Create a summary visualization across all layers.

    Args:
        analysis: Analysis results dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract data for all layers
    layer_names = list(analysis.keys())
    critical_percentages = [analysis[name]['critical_percentage'] for name in layer_names]
    risky_percentages = [analysis[name]['risky_channels'] / analysis[name]['total_channels'] * 100
                         for name in layer_names]
    dead_percentages = [analysis[name]['dead_channels'] / analysis[name]['total_channels'] * 100
                        for name in layer_names]

    # Create summary plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    x_pos = np.arange(len(layer_names))

    # Plot 1: Critical channel percentage
    axes[0].bar(x_pos, critical_percentages, alpha=0.7, color='red')
    axes[0].set_ylabel('Critical Channels (%)')
    axes[0].set_title('Percentage of Critical Channels per Layer\n(Dead + Risky + High Weight)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Risky channel percentage
    axes[1].bar(x_pos, risky_percentages, alpha=0.7, color='orange')
    axes[1].set_ylabel('Risky Channels (%)')
    axes[1].set_title('Percentage of Risky Channels per Layer\n(Dead but might activate with noise)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Dead channel percentage
    axes[2].bar(x_pos, dead_percentages, alpha=0.7, color='gray')
    axes[2].set_ylabel('Dead Channels (%)')
    axes[2].set_title('Percentage of Dead Channels per Layer\n(Mean pre-activation < -3)')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right', fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/summary_all_layers.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved summary visualization")


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = 100
    save_dir = "./visualizations/preactivation_analysis"

    print("=" * 80)
    print("Pre-activation Distribution Analysis for Fast-R-PRAQ Hypothesis Testing")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Calibration samples: {n_samples}")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Load calibration data
    print("\nLoading calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    # Initialize analyzer
    analyzer = PreActivationAnalyzer(model, tokenizer, device)

    # Register hooks (analyze a subset of layers for efficiency)
    # Focus on MLP layers and attention projections
    print("\nRegistering hooks on linear layers...")
    analyzer.register_hooks()  # Hook all linear layers

    # Collect data
    analyzer.collect_data(texts, max_samples=n_samples)

    # Remove hooks
    analyzer.remove_hooks()

    # Compute statistics
    print("\nComputing statistics...")
    stats = analyzer.compute_statistics()

    # Analyze hypothesis
    print("\nAnalyzing Fast-R-PRAQ hypothesis...")
    analysis = analyzer.analyze_hypothesis(stats, activation_threshold=-3.0)

    # Print summary
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 80)
    print(f"\nTesting: Do 'critical' channels exist?")
    print("Critical = Dead (mean < -3) + Risky (upper bound > -3) + High Weight\n")

    total_critical = 0
    total_channels = 0

    for layer_name, layer_analysis in analysis.items():
        total_critical += layer_analysis['critical_channels']
        total_channels += layer_analysis['total_channels']

        if layer_analysis['critical_percentage'] > 0:
            print(f"{layer_name}:")
            print(f"  Total channels: {layer_analysis['total_channels']}")
            print(f"  Dead channels: {layer_analysis['dead_channels']} ({layer_analysis['dead_channels']/layer_analysis['total_channels']*100:.2f}%)")
            print(f"  Risky channels: {layer_analysis['risky_channels']} ({layer_analysis['risky_channels']/layer_analysis['total_channels']*100:.2f}%)")
            print(f"  CRITICAL channels: {layer_analysis['critical_channels']} ({layer_analysis['critical_percentage']:.2f}%)")
            print()

    print("=" * 80)
    print(f"OVERALL: {total_critical} / {total_channels} ({total_critical/total_channels*100:.2f}%) channels are CRITICAL")
    print("=" * 80)

    if total_critical > 0:
        print("\n✅ HYPOTHESIS CONFIRMED:")
        print("   Critical channels exist in real models!")
        print("   Fast-R-PRAQ's approach is necessary to handle these risky channels.")
    else:
        print("\n❌ HYPOTHESIS REJECTED:")
        print("   No critical channels found.")
        print("   Fast-R-PRAQ may be over-engineering for this model.")

    # Create visualizations
    print("\n\nCreating visualizations...")

    # Visualize a few representative layers
    layers_to_visualize = list(stats.keys())[:5]  # First 5 layers

    for layer_name in tqdm(layers_to_visualize, desc="Visualizing layers"):
        visualize_layer(layer_name, stats[layer_name], analysis[layer_name], save_dir)

    # Create summary visualization
    create_summary_visualization(analysis, save_dir)

    print(f"\n✅ Visualizations saved to: {save_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
