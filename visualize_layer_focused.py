import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


class FocusedLayerAnalyzer:
    """
    Focused analyzer for a single layer to understand pre-activation distributions.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.preactivation_data = []
        self.hook = None
        self.target_layer_name = None

    def register_hook(self, layer_name):
        """
        Register hook on a specific layer.

        Args:
            layer_name: Exact name of the layer to hook
        """
        self.target_layer_name = layer_name

        # Find the target module
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Layer {layer_name} not found in model!")

        if not isinstance(target_module, nn.Linear):
            raise ValueError(f"Layer {layer_name} is not a Linear layer!")

        def hook_fn(module, input, output):
            # Capture input to the linear layer
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
            self.preactivation_data.append(Z.cpu().float())

        self.hook = target_module.register_forward_hook(hook_fn)
        print(f"Registered hook on layer: {layer_name}")

    def remove_hook(self):
        """Remove the hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

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

        successful_samples = 0

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
                self.model(**inputs, use_cache=False)
                successful_samples += 1

            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                continue

        print(f"\nData collection complete! Processed {successful_samples} samples")

    def compute_channel_statistics(self):
        """
        Compute detailed statistics for each channel.

        Returns:
            pandas DataFrame with per-channel statistics
        """
        if len(self.preactivation_data) == 0:
            raise ValueError("No data collected! Call collect_data() first.")

        # Concatenate all pre-activations
        Z = torch.cat(self.preactivation_data, dim=0)  # [total_tokens, out_features]
        print(f"\nTotal activations collected: {Z.shape[0]} tokens x {Z.shape[1]} channels")

        # Compute per-channel statistics
        channel_stats = []

        for ch in range(Z.shape[1]):
            ch_data = Z[:, ch].numpy()

            stats = {
                'channel': ch,
                'mean': np.mean(ch_data),
                'std': np.std(ch_data),
                'min': np.min(ch_data),
                'max': np.max(ch_data),
                'p05': np.percentile(ch_data, 5),
                'p25': np.percentile(ch_data, 25),
                'p50': np.percentile(ch_data, 50),
                'p75': np.percentile(ch_data, 75),
                'p95': np.percentile(ch_data, 95),
                'negative_ratio': (ch_data < 0).mean(),
                'dead_ratio': (ch_data < -3.0).mean(),  # Ratio below SiLU threshold
            }

            channel_stats.append(stats)

        df = pd.DataFrame(channel_stats)

        # Add derived columns
        df['abs_mean'] = df['mean'].abs()
        df['range'] = df['max'] - df['min']

        return df, Z


def visualize_focused_layer(df, Z, layer_name, save_dir="./visualizations/focused"):
    """
    Create comprehensive visualizations for the focused layer.

    Args:
        df: DataFrame with channel statistics
        Z: Raw pre-activation tensor [tokens, channels]
        layer_name: Name of the layer
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Channel Statistics Overview
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Mean distribution
    axes[0, 0].hist(df['mean'], bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(-3.0, color='red', linestyle='--', label='SiLU threshold')
    axes[0, 0].axvline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0, 0].set_xlabel('Mean Pre-activation')
    axes[0, 0].set_ylabel('Number of Channels')
    axes[0, 0].set_title(f'Distribution of Mean Pre-activations\n{layer_name}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Std distribution
    axes[0, 1].hist(df['std'], bins=100, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Std Pre-activation')
    axes[0, 1].set_ylabel('Number of Channels')
    axes[0, 1].set_title('Distribution of Std Pre-activations')
    axes[0, 1].grid(True, alpha=0.3)

    # Min distribution
    axes[0, 2].hist(df['min'], bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].axvline(-3.0, color='red', linestyle='--', label='SiLU threshold')
    axes[0, 2].set_xlabel('Min Pre-activation')
    axes[0, 2].set_ylabel('Number of Channels')
    axes[0, 2].set_title('Distribution of Min Pre-activations')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Max distribution
    axes[1, 0].hist(df['max'], bins=100, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Max Pre-activation')
    axes[1, 0].set_ylabel('Number of Channels')
    axes[1, 0].set_title('Distribution of Max Pre-activations')
    axes[1, 0].grid(True, alpha=0.3)

    # Dead ratio distribution
    axes[1, 1].hist(df['dead_ratio'], bins=100, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Dead Ratio (< -3.0)')
    axes[1, 1].set_ylabel('Number of Channels')
    axes[1, 1].set_title('Distribution of Dead Ratios')
    axes[1, 1].grid(True, alpha=0.3)

    # Mean vs Std scatter
    axes[1, 2].scatter(df['mean'], df['std'], alpha=0.5, s=10)
    axes[1, 2].axvline(-3.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 2].set_xlabel('Mean Pre-activation')
    axes[1, 2].set_ylabel('Std Pre-activation')
    axes[1, 2].set_title('Mean vs Std')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/channel_statistics_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_dir}/channel_statistics_overview.png")

    # 2. Negative Channels Analysis
    negative_channels = df[df['mean'] < -3.0].sort_values('mean')

    if len(negative_channels) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Select top 10 most negative channels
        top_negative = negative_channels.head(10)

        # Plot distributions for most negative channels
        ax = axes[0, 0]
        for idx, row in top_negative.iterrows():
            ch_id = row['channel']
            ch_data = Z[:, ch_id].numpy()
            ax.hist(ch_data, bins=50, alpha=0.3, label=f"Ch {ch_id}: μ={row['mean']:.2f}")

        ax.axvline(-3.0, color='red', linestyle='--', label='SiLU threshold')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Pre-activation Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top 10 Most Negative Channels\n(mean < -3.0)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Box plot of negative channels
        ax = axes[0, 1]
        negative_data = [Z[:, row['channel']].numpy() for _, row in top_negative.iterrows()]
        bp = ax.boxplot(negative_data, labels=[f"Ch{row['channel']}" for _, row in top_negative.iterrows()])
        ax.axhline(-3.0, color='red', linestyle='--', label='SiLU threshold')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Pre-activation Value')
        ax.set_title('Box Plot of Most Negative Channels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Heatmap of percentiles for negative channels
        ax = axes[1, 0]
        percentile_data = negative_channels.head(20)[['p05', 'p25', 'p50', 'p75', 'p95']].values
        im = ax.imshow(percentile_data, aspect='auto', cmap='coolwarm', vmin=-10, vmax=5)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Channel Index')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['p05', 'p25', 'p50', 'p75', 'p95'])
        ax.set_yticks(range(len(percentile_data)))
        ax.set_yticklabels([f"Ch{row['channel']}" for _, row in negative_channels.head(20).iterrows()])
        ax.set_title('Percentile Heatmap (Top 20 Most Negative)')
        plt.colorbar(im, ax=ax)

        # Statistics table for negative channels
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        table_data = negative_channels.head(10)[['channel', 'mean', 'std', 'min', 'max', 'dead_ratio']].round(3)
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        ax.set_title('Statistics of Most Negative Channels', pad=20)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/negative_channels_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_dir}/negative_channels_analysis.png")

    # 3. Individual channel plots for top negative channels
    if len(negative_channels) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (_, row) in enumerate(negative_channels.head(9).iterrows()):
            ch_id = row['channel']
            ch_data = Z[:, ch_id].numpy()

            ax = axes[i]
            ax.hist(ch_data, bins=100, alpha=0.7, edgecolor='black')
            ax.axvline(-3.0, color='red', linestyle='--', label='SiLU threshold')
            ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(row['mean'], color='blue', linestyle='--', label=f"Mean={row['mean']:.2f}")
            ax.set_xlabel('Pre-activation Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f"Channel {ch_id}\nμ={row['mean']:.2f}, σ={row['std']:.2f}, dead={row['dead_ratio']:.2%}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/negative_channels_individual.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_dir}/negative_channels_individual.png")


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    target_layer_id = 28  # Layer number to analyze
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = 100
    save_dir = f"./visualizations/layer_{target_layer_id}_focused"

    print("=" * 80)
    print(f"Focused Layer Analysis for MiniCPM-2B Layer {target_layer_id}")
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

    # Find layer name
    print("\nSearching for target layer...")
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    # Try to find layer by ID (common patterns)
    target_layer_name = None
    for name in layer_names:
        # Look for pattern like "model.layers.16." or ".16."
        if f".{target_layer_id}." in name or f"layers.{target_layer_id}." in name:
            # Prefer MLP gate/up projection layers
            if 'gate' in name.lower() or 'up' in name.lower() or 'mlp' in name.lower():
                target_layer_name = name
                break

    # If not found, try other patterns
    if target_layer_name is None:
        for name in layer_names:
            if f".{target_layer_id}." in name:
                target_layer_name = name
                break

    if target_layer_name is None:
        print(f"\nERROR: Could not find layer {target_layer_id}")
        print("\nAvailable layers:")
        for i, name in enumerate(layer_names[:20]):
            print(f"  {i}: {name}")
        return

    print(f"\nTarget layer found: {target_layer_name}")

    # Load calibration data
    print("\nLoading calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    # Initialize analyzer
    analyzer = FocusedLayerAnalyzer(model, tokenizer, device)

    # Register hook on target layer
    analyzer.register_hook(target_layer_name)

    # Collect data
    analyzer.collect_data(texts, max_samples=n_samples)

    # Remove hook
    analyzer.remove_hook()

    # Compute statistics
    print("\nComputing channel statistics...")
    df, Z = analyzer.compute_channel_statistics()

    # Save statistics to CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/channel_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nChannel statistics saved to: {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("CHANNEL STATISTICS SUMMARY")
    print("=" * 80)
    print(f"\nTotal channels: {len(df)}")
    print(f"\nChannels with negative mean: {(df['mean'] < 0).sum()} ({(df['mean'] < 0).mean()*100:.2f}%)")
    print(f"Channels with mean < -3.0: {(df['mean'] < -3.0).sum()} ({(df['mean'] < -3.0).mean()*100:.2f}%)")
    print(f"\nMean statistics:")
    print(f"  Min mean: {df['mean'].min():.4f}")
    print(f"  Max mean: {df['mean'].max():.4f}")
    print(f"  Avg mean: {df['mean'].mean():.4f}")
    print(f"  Std of means: {df['mean'].std():.4f}")

    print("\n" + "=" * 80)
    print("TOP 10 MOST NEGATIVE CHANNELS")
    print("=" * 80)
    negative_channels = df[df['mean'] < -3.0].sort_values('mean')
    if len(negative_channels) > 0:
        print(negative_channels.head(10)[['channel', 'mean', 'std', 'min', 'max', 'dead_ratio']].to_string(index=False))
    else:
        print("No channels with mean < -3.0 found!")

    # Create visualizations
    print("\n\nCreating visualizations...")
    visualize_focused_layer(df, Z, target_layer_name, save_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {save_dir}")
    print(f"  - channel_statistics.csv: Per-channel statistics")
    print(f"  - channel_statistics_overview.png: Overview visualizations")
    print(f"  - negative_channels_analysis.png: Negative channels analysis")
    print(f"  - negative_channels_individual.png: Individual channel distributions")
    print("=" * 80)


if __name__ == "__main__":
    main()
