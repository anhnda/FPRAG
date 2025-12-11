"""
Visualize pre-activation vs post-activation distributions for layers with SiLU.

This script captures the distribution of XW (pre-activation) before and after
applying SiLU activation to demonstrate how activation changes the distribution
and identify potentially risky channels (negative pre-activation with high variance).
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def silu(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)

class ActivationCapture:
    """Hook to capture pre and post activation values"""
    def __init__(self):
        self.pre_activation = []
        self.post_activation = []

    def hook_fn(self, module, input, output):
        """Capture both pre-activation (input) and post-activation (output)"""
        # input is a tuple, we want the first element
        pre_act = input[0].detach().cpu()
        post_act = output.detach().cpu()

        self.pre_activation.append(pre_act)
        self.post_activation.append(post_act)

    def get_concatenated(self):
        """Concatenate all captured activations"""
        pre = torch.cat(self.pre_activation, dim=0)
        post = torch.cat(self.post_activation, dim=0)
        return pre, post

    def clear(self):
        """Clear stored activations"""
        self.pre_activation = []
        self.post_activation = []


def find_silu_layers(model):
    """Find layers that use SiLU activation (typically MLP gate/up projections)"""
    silu_layers = []

    for name, module in model.named_modules():
        # Check if this is followed by SiLU in MiniCPM architecture
        # Typically: model.layers.X.mlp.gate_proj or up_proj
        if 'mlp.gate_proj' in name or 'mlp.up_proj' in name:
            silu_layers.append((name, module))

    return silu_layers


def visualize_pre_post_activation(pre_activation, post_activation, layer_name, output_dir):
    """Create comprehensive visualization of pre vs post activation distributions"""

    # Flatten to get all values
    pre_flat = pre_activation.reshape(-1).numpy()
    post_flat = post_activation.reshape(-1).numpy()

    # Per-channel statistics
    # Assuming shape is [batch*seq_len, hidden_dim]
    pre_channel_mean = pre_activation.mean(dim=0).numpy()
    pre_channel_std = pre_activation.std(dim=0).numpy()
    post_channel_mean = post_activation.mean(dim=0).numpy()
    post_channel_std = post_activation.std(dim=0).numpy()

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Overall distribution comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(pre_flat, bins=100, alpha=0.6, label='Pre-activation (XW)', color='blue', density=True)
    ax1.hist(post_flat, bins=100, alpha=0.6, label='Post-activation (SiLU(XW))', color='red', density=True)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
    ax1.set_xlabel('Activation Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{layer_name}\nOverall Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Log-scale histogram to see tails
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(pre_flat, bins=100, alpha=0.6, label='Pre-activation', color='blue', log=True)
    ax2.hist(post_flat, bins=100, alpha=0.6, label='Post-activation', color='red', log=True)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Activation Value')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('Distribution (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative distribution
    ax3 = plt.subplot(2, 3, 3)
    sorted_pre = np.sort(pre_flat)
    sorted_post = np.sort(post_flat)
    ax3.plot(sorted_pre, np.linspace(0, 1, len(sorted_pre)), label='Pre-activation', linewidth=2, alpha=0.8)
    ax3.plot(sorted_post, np.linspace(0, 1, len(sorted_post)), label='Post-activation', linewidth=2, alpha=0.8)
    ax3.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Activation Value')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Per-channel mean comparison
    ax4 = plt.subplot(2, 3, 4)
    channels = np.arange(len(pre_channel_mean))
    ax4.scatter(pre_channel_mean, post_channel_mean, alpha=0.5, s=10)

    # Add identity line
    min_val = min(pre_channel_mean.min(), post_channel_mean.min())
    max_val = max(pre_channel_mean.max(), post_channel_mean.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.5, label='y=x')

    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax4.set_xlabel('Pre-activation Mean')
    ax4.set_ylabel('Post-activation Mean')
    ax4.set_title('Per-Channel Mean: Pre vs Post')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Effect of SiLU on negative values
    ax5 = plt.subplot(2, 3, 5)
    # Sample points
    x_range = np.linspace(-10, 10, 1000)
    y_silu = x_range * (1 / (1 + np.exp(-x_range)))  # SiLU formula

    ax5.plot(x_range, x_range, 'b--', linewidth=2, alpha=0.5, label='Identity (no activation)')
    ax5.plot(x_range, y_silu, 'r-', linewidth=2, label='SiLU activation')
    ax5.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax5.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax5.axvline(-3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (τ=-3)')
    ax5.set_xlabel('Pre-activation (x)')
    ax5.set_ylabel('Post-activation (SiLU(x))')
    ax5.set_title('SiLU Activation Function')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-10, 10)
    ax5.set_ylim(-2, 10)

    # 6. Negative channel analysis
    ax6 = plt.subplot(2, 3, 6)
    negative_channels = pre_channel_mean < 0
    if negative_channels.sum() > 0:
        neg_pre_mean = pre_channel_mean[negative_channels]
        neg_pre_std = pre_channel_std[negative_channels]
        neg_post_mean = post_channel_mean[negative_channels]

        # Show how much activation is suppressed
        suppression_ratio = neg_post_mean / neg_pre_mean  # Should be close to 0 for very negative

        ax6.scatter(neg_pre_mean, suppression_ratio, alpha=0.6, s=20, c=neg_pre_std, cmap='viridis')
        cbar = plt.colorbar(ax6.collections[0], ax=ax6, label='Pre-activation Std')
        ax6.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No suppression')
        ax6.set_xlabel('Pre-activation Mean (negative channels)')
        ax6.set_ylabel('Post/Pre Ratio (suppression)')
        ax6.set_title('SiLU Suppression Effect on Negative Channels')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{layer_name.replace(".", "_")}_pre_post_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()

    # Print statistics
    print(f"\n{'='*80}")
    print(f"Statistics for {layer_name}")
    print(f"{'='*80}")
    print(f"\nPre-activation (XW):")
    print(f"  Mean: {pre_flat.mean():.4f}")
    print(f"  Std:  {pre_flat.std():.4f}")
    print(f"  Min:  {pre_flat.min():.4f}")
    print(f"  Max:  {pre_flat.max():.4f}")
    print(f"  Fraction < 0: {(pre_flat < 0).mean()*100:.2f}%")
    print(f"  Fraction < -3: {(pre_flat < -3).mean()*100:.2f}%")

    print(f"\nPost-activation (SiLU(XW)):")
    print(f"  Mean: {post_flat.mean():.4f}")
    print(f"  Std:  {post_flat.std():.4f}")
    print(f"  Min:  {post_flat.min():.4f}")
    print(f"  Max:  {post_flat.max():.4f}")
    print(f"  Fraction < 0: {(post_flat < 0).mean()*100:.2f}%")

    print(f"\nPer-channel statistics:")
    print(f"  Total channels: {len(pre_channel_mean)}")
    print(f"  Negative mean channels (pre-act): {(pre_channel_mean < 0).sum()}")
    print(f"  Negative mean channels (post-act): {(post_channel_mean < 0).sum()}")

    # Identify risky channels
    risky_channels = (pre_channel_mean < 0) & (pre_channel_std > 2)
    print(f"  Risky channels (mean < 0, std > 2): {risky_channels.sum()}")

    if risky_channels.sum() > 0:
        print(f"\n  Example risky channels (first 5):")
        risky_indices = np.where(risky_channels)[0][:5]
        for idx in risky_indices:
            print(f"    Channel {idx}: pre_mean={pre_channel_mean[idx]:.3f}, pre_std={pre_channel_std[idx]:.3f}, "
                  f"post_mean={post_channel_mean[idx]:.3f}")


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    target_layer_id = 3  # Layer 3 as mentioned by user
    n_samples = 500  # Number of calibration samples
    max_seq_len = 512
    output_dir = "./visualizations/pre_post_activation_analysis"

    print(f"Loading model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
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

    # Find SiLU layers
    print("\nFinding layers with SiLU activation...")
    silu_layers = find_silu_layers(model)
    print(f"Found {len(silu_layers)} layers with SiLU activation")

    # Print available layers
    print("\nAvailable SiLU layers:")
    for i, (name, _) in enumerate(silu_layers):
        layer_num = int(name.split('.')[2]) if 'layers.' in name else -1
        print(f"  [{i}] {name} (layer {layer_num})")

    # Find target layer
    target_layer_name = None
    target_module = None
    for name, module in silu_layers:
        if f'layers.{target_layer_id}.' in name and 'gate_proj' in name:
            target_layer_name = name
            target_module = module
            break

    if target_module is None:
        print(f"\nError: Could not find gate_proj for layer {target_layer_id}")
        print("Available layers:")
        for name, _ in silu_layers:
            print(f"  {name}")
        return

    print(f"\nTarget layer: {target_layer_name}")

    # Register hook
    capture = ActivationCapture()
    hook = target_module.register_forward_hook(capture.hook_fn)

    # Load calibration data
    print(f"\nLoading WikiText-2 calibration data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Collect activations
    print(f"\nCollecting activations from {n_samples} samples...")
    n_collected = 0

    with torch.no_grad():
        for sample in tqdm(dataset, total=n_samples):
            if n_collected >= n_samples:
                break

            text = sample['text']
            if len(text.strip()) == 0:
                continue

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=max_seq_len,
                truncation=True,
                padding=False
            ).to(device)

            # Forward pass (disable cache to avoid compatibility issues)
            _ = model(**inputs, use_cache=False)
            n_collected += 1

    # Remove hook
    hook.remove()

    # Get concatenated activations
    print("\nProcessing collected activations...")
    pre_activation, post_activation = capture.get_concatenated()

    print(f"Pre-activation shape: {pre_activation.shape}")
    print(f"Post-activation shape: {post_activation.shape}")

    # Visualize
    print("\nGenerating visualizations...")
    visualize_pre_post_activation(pre_activation, post_activation, target_layer_name, output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
