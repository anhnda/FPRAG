"""
Analyze saliency and squared gradient magnitudes for XW layer followed by SiLU activation.

This script:
1. Computes saliency scores for all channels in a target layer
2. Identifies top 10 channels with largest saliency
3. For these channels, visualizes:
   - Sorted [X*,j]^2 (squared input magnitudes) with knee point detection
   - grad(w*,j)^2 (squared weight gradients from SiLU output) with knee point detection
4. Uses Kneedle algorithm to detect knee points on the first half of sorted data
   for both E[X²] and grad² curves
5. Compares weight ranges:
   - Top important weights (up to knee_point + k_offset) vs all weights
   - Separate comparisons for X²-based and grad²-based importance rankings
   - Box plots, statistics, and distribution histograms
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import pandas as pd
from kneed import KneeLocator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class SaliencyGradientAnalyzer:
    """Analyzer for saliency and gradient statistics in linear layers with SiLU."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    def get_calibration_data(self, n_samples=128, max_length=512):
        """Load calibration data from WikiText-2."""
        print(f"Loading {n_samples} calibration samples from WikiText-2...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        texts = []
        for example in dataset:
            if example['text'].strip():
                texts.append(example['text'])
                if len(texts) >= n_samples * 2:  # Get extra in case some are too short
                    break

        calibration_data = []
        for text in tqdm(texts[:n_samples], desc="Tokenizing"):
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            calibration_data.append(inputs['input_ids'].to(self.device))

        return calibration_data

    def find_target_layer(self, layer_idx=3):
        """
        Find the target linear layer in the specified transformer layer.
        We'll target the gate_proj or up_proj in MLP which is followed by SiLU.
        """
        # MiniCPM structure: model.layers[i].mlp.gate_proj (followed by SiLU)
        target_module = self.model.model.layers[layer_idx].mlp.gate_proj
        print(f"Target layer: model.layers[{layer_idx}].mlp.gate_proj")
        print(f"Weight shape: {target_module.weight.shape}")
        return target_module, layer_idx

    def compute_saliency_and_gradients(self, target_module, calibration_data, layer_idx):
        """
        Compute saliency scores and collect gradients for the target layer.

        Saliency is computed as: E[|X * W^T|] (AWQ-style importance)
        Gradients are computed w.r.t. SiLU output
        """
        # Storage for analysis
        pre_activations = []  # XW (before SiLU)
        post_activations = []  # SiLU(XW)
        input_activations = []  # X (inputs to the layer)

        # Hook to capture activations
        def forward_hook(module, input, output):
            # input[0] is the input tensor X (batch_size, seq_len, in_features)
            # output is XW^T (batch_size, seq_len, out_features)
            input_activations.append(input[0].detach().cpu())
            pre_activations.append(output.detach().cpu())

        # Hook to capture post-SiLU activations
        def silu_hook(module, input, output):
            post_activations.append(output.detach().cpu())

        # Register hooks
        hook1 = target_module.register_forward_hook(forward_hook)
        # Find the SiLU activation after gate_proj
        silu_module = self.model.model.layers[layer_idx].mlp.act_fn
        hook2 = silu_module.register_forward_hook(silu_hook)

        # Run forward passes to collect data
        print("Collecting activations from calibration data...")
        self.model.eval()
        with torch.no_grad():
            for input_ids in tqdm(calibration_data, desc="Forward passes"):
                # Disable cache to avoid compatibility issues
                self.model(input_ids, use_cache=False)

        # Remove hooks
        hook1.remove()
        hook2.remove()

        # Concatenate all collected data
        X = torch.cat(input_activations, dim=0)  # (total_tokens, in_features)
        X = X.view(-1, X.shape[-1]).float()  # Flatten batch and sequence dimensions, convert to float32

        pre_act = torch.cat(pre_activations, dim=0)  # (total_tokens, out_features)
        pre_act = pre_act.view(-1, pre_act.shape[-1]).float()

        post_act = torch.cat(post_activations, dim=0)  # (total_tokens, out_features)
        post_act = post_act.view(-1, post_act.shape[-1]).float()

        print(f"Collected {X.shape[0]} tokens")
        print(f"Input shape: {X.shape}, Pre-activation shape: {pre_act.shape}")

        # Compute saliency: E[|XW^T|] per channel
        saliency = torch.mean(torch.abs(pre_act), dim=0)  # (out_features,)

        # Compute gradients of weights w.r.t. post-SiLU output
        print("Computing weight gradients...")
        weight_gradients = self.compute_weight_gradients(
            X, pre_act, post_act, target_module.weight
        )

        return {
            'saliency': saliency,
            'weight_gradients': weight_gradients,
            'input_activations': X,
            'pre_activations': pre_act,
            'post_activations': post_act
        }

    def compute_weight_gradients(self, X, pre_act, post_act, weight):
        """
        Compute gradients of weights w.r.t. SiLU output.

        For y = SiLU(XW^T), we want ∂y/∂W
        Using chain rule: ∂y/∂W = ∂y/∂(XW^T) * ∂(XW^T)/∂W
                                = SiLU'(XW^T) * X

        SiLU'(z) = σ(z) * (1 + z * (1 - σ(z))) where σ is sigmoid
        """
        # Move to GPU for computation
        X_gpu = X.to(self.device)
        pre_act_gpu = pre_act.to(self.device)

        # Compute SiLU derivative: σ(z) * (1 + z * (1 - σ(z)))
        sigmoid_z = torch.sigmoid(pre_act_gpu)
        silu_grad = sigmoid_z * (1 + pre_act_gpu * (1 - sigmoid_z))  # (tokens, out_features)

        # For each output channel j: grad_w[i,j] = mean over tokens of (silu_grad[t,j] * X[t,i])
        # Weight is (out_features, in_features), so gradient is same shape

        # We'll compute mean squared gradient for each weight
        # grad^2 = E[(silu_grad[j] * X[i])^2]

        num_tokens = X_gpu.shape[0]
        out_features, in_features = weight.shape

        # Compute per-channel gradient magnitude
        weight_grad_squared = torch.zeros(out_features, in_features, device=self.device)

        batch_size = 1000  # Process in batches to save memory
        for start_idx in tqdm(range(0, num_tokens, batch_size), desc="Computing gradients"):
            end_idx = min(start_idx + batch_size, num_tokens)

            X_batch = X_gpu[start_idx:end_idx]  # (batch, in_features)
            silu_grad_batch = silu_grad[start_idx:end_idx]  # (batch, out_features)

            # For each output channel j
            for j in range(out_features):
                # grad_w[:,j] = silu_grad[:,j][:, None] * X  (batch, in_features)
                grad_batch = silu_grad_batch[:, j:j+1] * X_batch  # (batch, in_features)
                weight_grad_squared[j] += torch.sum(grad_batch ** 2, dim=0)

        # Average over tokens
        weight_grad_squared = weight_grad_squared / num_tokens

        return weight_grad_squared.cpu()

    def analyze_top_channels(self, results, top_k=10, output_dir='./visualizations/saliency_gradient_analysis', target_module=None, k_offset=0):
        """
        Analyze top-k channels by saliency and create visualizations.

        Args:
            k_offset: Offset to add to knee point when selecting top weights for range analysis (default: 0)
        """
        os.makedirs(output_dir, exist_ok=True)

        saliency = results['saliency']
        weight_gradients = results['weight_gradients']
        X = results['input_activations']
        pre_act = results['pre_activations']

        # Get top-k channels by saliency
        top_k_indices = torch.argsort(saliency, descending=True)[:top_k]

        print(f"\nTop {top_k} channels by saliency:")
        for rank, ch_idx in enumerate(top_k_indices):
            print(f"  Rank {rank+1}: Channel {ch_idx.item()}, Saliency = {saliency[ch_idx]:.4f}")

        # Store knee point info for summary
        knee_points_info = []

        # Create visualizations for each top channel
        for rank, ch_idx in enumerate(top_k_indices):
            ch_idx = ch_idx.item()
            knee_info = self.visualize_channel(
                ch_idx, rank + 1, X, pre_act, weight_gradients,
                saliency[ch_idx].item(), output_dir, target_module, k_offset
            )
            knee_points_info.append(knee_info)

        # Create summary statistics
        self.create_summary(results, top_k_indices, output_dir, knee_points_info)

        # Create overall distribution plots
        self.plot_saliency_distribution(saliency, top_k_indices, output_dir)

    def visualize_channel(self, ch_idx, rank, X, pre_act, weight_gradients, saliency_score, output_dir, target_module=None, k_offset=0):
        """
        Create visualization for a single channel showing:
        1. Sorted [X*,j]^2 (squared input contributions) with knee point
        2. Corresponding X values in sorted order
        3. Sorted grad(w*,j)^2 (squared weight gradients) with knee point
        4. Corresponding weight values in sorted order
        5. Weight range comparison: top important weights (up to knee + k_offset) vs all weights

        Args:
            k_offset: Offset to add to knee point when selecting top weights for range analysis (default: 0)

        Returns:
            dict: Dictionary containing channel index and knee point indices for X² and grad²
        """
        # For channel j, we want to analyze how each input feature i contributes
        # [X*,j] means the contribution of input feature * to output channel j
        # Actually, the pre_activation[j] = sum_i(X[i] * W[j, i])
        # So we want to look at X[i]^2 for all input features

        # Get the pre-activation for this channel across all tokens
        pre_act_channel = pre_act[:, ch_idx]  # (total_tokens,)

        # For this channel, compute X[token, feature]^2 contribution
        # We'll average over tokens to get per-feature importance
        X_squared = X ** 2  # (tokens, in_features)
        X_squared_mean = torch.mean(X_squared, dim=0)  # (in_features,)
        X_mean = torch.mean(X, dim=0)  # (in_features,) - mean of actual X values

        # Get weight gradients for this channel
        grad_squared = weight_gradients[ch_idx]  # (in_features,)

        # Get actual weights for this channel if available
        if target_module is not None:
            weights = target_module.weight[ch_idx].detach().cpu().float()  # (in_features,) - convert to float32
        else:
            weights = None

        # Sort by X_squared_mean
        X_sorted_values, X_sorted_indices = torch.sort(X_squared_mean, descending=True)

        # Sort by grad_squared
        grad_sorted_values, grad_sorted_indices = torch.sort(grad_squared, descending=True)

        # Initialize knee point tracking
        knee_idx_x2 = None
        knee_idx_grad2 = None

        # Create figure with 3 rows, 3 columns
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # Plot 1 (Row 1, Col 1): Sorted X^2
        ax = axes[0, 0]
        n_features = len(X_sorted_values)
        ax.plot(range(n_features), X_sorted_values.numpy(), linewidth=1.5, alpha=0.8)

        # Detect knee point on first half of data
        half_n = n_features // 2
        try:
            # Use KneeLocator on first half
            x_range_half = np.arange(half_n)
            y_values_half = X_sorted_values[:half_n].numpy()

            # Since values are sorted descending and on log scale, use 'convex' and 'decreasing'
            knee_locator_x2 = KneeLocator(
                x_range_half, y_values_half,
                curve='convex', direction='decreasing',
                S=1.0  # Sensitivity parameter
            )

            if knee_locator_x2.knee is not None:
                knee_idx_x2 = int(knee_locator_x2.knee)
                ax.axvline(knee_idx_x2, color='red', linestyle='--', linewidth=2,
                          label=f'Knee point: {knee_idx_x2}')
                ax.plot(knee_idx_x2, X_sorted_values[knee_idx_x2].numpy(),
                       'ro', markersize=10, label=f'Knee value: {X_sorted_values[knee_idx_x2].item():.2e}')
                ax.legend(fontsize=9)
        except Exception as e:
            print(f"    Warning: Could not detect knee point for X²: {e}")

        ax.set_xlabel('Feature Rank (sorted by X²)', fontsize=11)
        ax.set_ylabel('Mean Squared Input (X²)', fontsize=11)
        ax.set_title(f'Rank {rank}: Channel {ch_idx}\nSorted Input Magnitudes (X²)', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Plot 2 (Row 2, Col 1): Corresponding X values in sorted X² order
        ax = axes[1, 0]
        X_mean_sorted_by_X2 = X_mean[X_sorted_indices]
        ax.plot(range(n_features), X_mean_sorted_by_X2.numpy(), linewidth=1.5, alpha=0.8, color='steelblue')
        ax.set_xlabel('Feature Rank (sorted by X²)', fontsize=11)
        ax.set_ylabel('Mean Input Value (X)', fontsize=11)
        ax.set_title(f'Corresponding X Values\n(in X² sorted order)', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 3 (Row 1, Col 2): Sorted grad^2
        ax = axes[0, 1]
        ax.plot(range(n_features), grad_sorted_values.numpy(), linewidth=1.5, alpha=0.8, color='orangered')

        # Detect knee point on first half of data for grad²
        try:
            # Use KneeLocator on first half
            x_range_half = np.arange(half_n)
            y_values_half_grad = grad_sorted_values[:half_n].numpy()

            # Since values are sorted descending and on log scale, use 'convex' and 'decreasing'
            knee_locator_grad2 = KneeLocator(
                x_range_half, y_values_half_grad,
                curve='convex', direction='decreasing',
                S=1.0  # Sensitivity parameter
            )

            if knee_locator_grad2.knee is not None:
                knee_idx_grad2 = int(knee_locator_grad2.knee)
                ax.axvline(knee_idx_grad2, color='red', linestyle='--', linewidth=2,
                          label=f'Knee point: {knee_idx_grad2}')
                ax.plot(knee_idx_grad2, grad_sorted_values[knee_idx_grad2].numpy(),
                       'ro', markersize=10, label=f'Knee value: {grad_sorted_values[knee_idx_grad2].item():.2e}')
                ax.legend(fontsize=9)
        except Exception as e:
            print(f"    Warning: Could not detect knee point for grad²: {e}")

        ax.set_xlabel('Weight Rank (sorted by grad²)', fontsize=11)
        ax.set_ylabel('Squared Weight Gradient', fontsize=11)
        ax.set_title(f'Rank {rank}: Channel {ch_idx}\nSorted Weight Gradients (grad²)', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Plot 4 (Row 2, Col 2): Corresponding weight values in sorted grad² order
        ax = axes[1, 1]
        if weights is not None:
            weights_sorted_by_grad2 = weights[grad_sorted_indices]
            ax.plot(range(n_features), weights_sorted_by_grad2.numpy(), linewidth=1.5, alpha=0.8, color='darkred')
            ax.set_xlabel('Weight Rank (sorted by grad²)', fontsize=11)
            ax.set_ylabel('Weight Value (W)', fontsize=11)
            ax.set_title(f'Corresponding Weight Values\n(in grad² sorted order)', fontsize=12)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Weights not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')

        # Plot 5 (Row 1, Col 3): Joint scatter plot
        ax = axes[0, 2]
        # Sample points for clearer visualization
        sample_indices = np.random.choice(n_features, size=min(2000, n_features), replace=False)
        ax.scatter(
            X_squared_mean[sample_indices].numpy(),
            grad_squared[sample_indices].numpy(),
            alpha=0.4, s=10, c='purple'
        )
        ax.set_xlabel('Mean Squared Input (X²)', fontsize=11)
        ax.set_ylabel('Squared Weight Gradient', fontsize=11)
        ax.set_title(f'Rank {rank}: Channel {ch_idx}\nX² vs grad² Correlation', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        from scipy.stats import spearmanr
        corr, _ = spearmanr(X_squared_mean.numpy(), grad_squared.numpy())
        ax.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 6 (Row 2, Col 3): X vs Weight correlation
        ax = axes[1, 2]
        if weights is not None:
            ax.scatter(
                X_mean[sample_indices].numpy(),
                weights[sample_indices].numpy(),
                alpha=0.4, s=10, c='green'
            )
            ax.set_xlabel('Mean Input Value (X)', fontsize=11)
            ax.set_ylabel('Weight Value (W)', fontsize=11)
            ax.set_title(f'X vs W Correlation', fontsize=12)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            corr_xw, _ = spearmanr(X_mean.numpy(), weights.numpy())
            ax.text(0.05, 0.95, f'Spearman ρ = {corr_xw:.3f}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Weights not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')

        # Row 3: Weight range comparisons
        # Plot 7 (Row 3, Col 1): Weight range for top-k based on X² importance
        ax = axes[2, 0]
        if weights is not None and knee_idx_x2 is not None:
            top_k_x2 = min(knee_idx_x2 + k_offset, n_features)
            top_weights_x2 = weights[X_sorted_indices[:top_k_x2]]
            all_weights = weights

            # Compute statistics
            data_to_plot = [all_weights.numpy(), top_weights_x2.numpy()]
            labels = [f'All Weights\n(n={len(all_weights)})', f'Top by X²\n(n={top_k_x2})']

            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)

            # Color the boxes
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')

            ax.set_ylabel('Weight Value', fontsize=11)
            ax.set_title(f'Weight Range Comparison (X² knee={knee_idx_x2}, offset={k_offset})', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Add statistics text
            stats_text = f'Top weights: μ={top_weights_x2.mean():.3f}, σ={top_weights_x2.std():.3f}\n'
            stats_text += f'All weights: μ={all_weights.mean():.3f}, σ={all_weights.std():.3f}\n'
            stats_text += f'Range ratio: {(top_weights_x2.max() - top_weights_x2.min()) / (all_weights.max() - all_weights.min()):.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Knee point not detected\nor weights unavailable',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.axis('off')

        # Plot 8 (Row 3, Col 2): Weight range for top-k based on grad² importance
        ax = axes[2, 1]
        if weights is not None and knee_idx_grad2 is not None:
            top_k_grad2 = min(knee_idx_grad2 + k_offset, n_features)
            top_weights_grad2 = weights[grad_sorted_indices[:top_k_grad2]]
            all_weights = weights

            # Compute statistics
            data_to_plot = [all_weights.numpy(), top_weights_grad2.numpy()]
            labels = [f'All Weights\n(n={len(all_weights)})', f'Top by grad²\n(n={top_k_grad2})']

            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)

            # Color the boxes
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('orange')

            ax.set_ylabel('Weight Value', fontsize=11)
            ax.set_title(f'Weight Range Comparison (grad² knee={knee_idx_grad2}, offset={k_offset})', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Add statistics text
            stats_text = f'Top weights: μ={top_weights_grad2.mean():.3f}, σ={top_weights_grad2.std():.3f}\n'
            stats_text += f'All weights: μ={all_weights.mean():.3f}, σ={all_weights.std():.3f}\n'
            stats_text += f'Range ratio: {(top_weights_grad2.max() - top_weights_grad2.min()) / (all_weights.max() - all_weights.min()):.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Knee point not detected\nor weights unavailable',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.axis('off')

        # Plot 9 (Row 3, Col 3): Combined weight distribution histogram
        ax = axes[2, 2]
        if weights is not None and knee_idx_x2 is not None and knee_idx_grad2 is not None:
            top_k_x2 = min(knee_idx_x2 + k_offset, n_features)
            top_k_grad2 = min(knee_idx_grad2 + k_offset, n_features)
            top_weights_x2 = weights[X_sorted_indices[:top_k_x2]]
            top_weights_grad2 = weights[grad_sorted_indices[:top_k_grad2]]

            # Create overlapping histograms
            ax.hist(all_weights.numpy(), bins=50, alpha=0.3, label='All weights', color='gray', density=True)
            ax.hist(top_weights_x2.numpy(), bins=30, alpha=0.5, label=f'Top by X² (n={top_k_x2})',
                   color='red', density=True)
            ax.hist(top_weights_grad2.numpy(), bins=30, alpha=0.5, label=f'Top by grad² (n={top_k_grad2})',
                   color='orange', density=True)

            ax.set_xlabel('Weight Value', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title('Weight Distribution Comparison', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'Knee points not detected\nor weights unavailable',
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.axis('off')

        plt.suptitle(f'Channel {ch_idx} Analysis (Saliency = {saliency_score:.4f})',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/channel_{ch_idx}_rank_{rank}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization for channel {ch_idx} (rank {rank})")

        # Return knee point information and weight range statistics
        weight_range_info = {}
        if weights is not None:
            if knee_idx_x2 is not None:
                top_k_x2 = min(knee_idx_x2 + k_offset, n_features)
                top_weights_x2 = weights[X_sorted_indices[:top_k_x2]]
                weight_range_info['top_x2_mean'] = top_weights_x2.mean().item()
                weight_range_info['top_x2_std'] = top_weights_x2.std().item()
                weight_range_info['top_x2_range'] = (top_weights_x2.max() - top_weights_x2.min()).item()
            if knee_idx_grad2 is not None:
                top_k_grad2 = min(knee_idx_grad2 + k_offset, n_features)
                top_weights_grad2 = weights[grad_sorted_indices[:top_k_grad2]]
                weight_range_info['top_grad2_mean'] = top_weights_grad2.mean().item()
                weight_range_info['top_grad2_std'] = top_weights_grad2.std().item()
                weight_range_info['top_grad2_range'] = (top_weights_grad2.max() - top_weights_grad2.min()).item()

            weight_range_info['all_mean'] = weights.mean().item()
            weight_range_info['all_std'] = weights.std().item()
            weight_range_info['all_range'] = (weights.max() - weights.min()).item()

        return {
            'channel_idx': ch_idx,
            'knee_x2': knee_idx_x2,
            'knee_grad2': knee_idx_grad2,
            **weight_range_info
        }

    def create_summary(self, results, top_k_indices, output_dir, knee_points_info):
        """Create summary statistics for top channels."""
        saliency = results['saliency']
        weight_gradients = results['weight_gradients']
        X = results['input_activations']

        summary_data = []
        for rank, ch_idx in enumerate(top_k_indices):
            ch_idx = ch_idx.item()

            X_squared_mean = torch.mean(X ** 2, dim=0)
            grad_squared = weight_gradients[ch_idx]

            from scipy.stats import spearmanr
            corr, _ = spearmanr(X_squared_mean.numpy(), grad_squared.numpy())

            # Get knee point info for this channel
            knee_info = knee_points_info[rank]

            summary_entry = {
                'Rank': rank + 1,
                'Channel_Index': ch_idx,
                'Saliency': saliency[ch_idx].item(),
                'Mean_X_Squared': X_squared_mean.mean().item(),
                'Max_X_Squared': X_squared_mean.max().item(),
                'Mean_Grad_Squared': grad_squared.mean().item(),
                'Max_Grad_Squared': grad_squared.max().item(),
                'Spearman_Correlation': corr,
                'Knee_X2_Index': knee_info['knee_x2'] if knee_info['knee_x2'] is not None else -1,
                'Knee_Grad2_Index': knee_info['knee_grad2'] if knee_info['knee_grad2'] is not None else -1
            }

            # Add weight range statistics if available
            if 'all_mean' in knee_info:
                summary_entry['All_Weights_Mean'] = knee_info['all_mean']
                summary_entry['All_Weights_Std'] = knee_info['all_std']
                summary_entry['All_Weights_Range'] = knee_info['all_range']
            if 'top_x2_mean' in knee_info:
                summary_entry['Top_X2_Mean'] = knee_info['top_x2_mean']
                summary_entry['Top_X2_Std'] = knee_info['top_x2_std']
                summary_entry['Top_X2_Range'] = knee_info['top_x2_range']
            if 'top_grad2_mean' in knee_info:
                summary_entry['Top_Grad2_Mean'] = knee_info['top_grad2_mean']
                summary_entry['Top_Grad2_Std'] = knee_info['top_grad2_std']
                summary_entry['Top_Grad2_Range'] = knee_info['top_grad2_range']

            summary_data.append(summary_entry)

        df = pd.DataFrame(summary_data)
        df.to_csv(f'{output_dir}/top_channels_summary.csv', index=False)
        print(f"\nSaved summary to {output_dir}/top_channels_summary.csv")
        print("\nSummary Statistics:")
        print(df.to_string(index=False))

    def plot_saliency_distribution(self, saliency, top_k_indices, output_dir):
        """Plot overall saliency distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Histogram of all saliency scores
        ax = axes[0]
        ax.hist(saliency.numpy(), bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Saliency Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Saliency Scores (All Channels)', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Mark top-k threshold
        threshold = saliency[top_k_indices[-1]].item()
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Top-{len(top_k_indices)} threshold')
        ax.legend()

        # Plot 2: Sorted saliency scores
        ax = axes[1]
        sorted_saliency, _ = torch.sort(saliency, descending=True)
        ax.plot(range(len(sorted_saliency)), sorted_saliency.numpy(), linewidth=1.5)
        ax.set_xlabel('Channel Rank', fontsize=12)
        ax.set_ylabel('Saliency Score', fontsize=12)
        ax.set_title('Sorted Saliency Scores', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Highlight top-k
        ax.axvline(len(top_k_indices), color='red', linestyle='--', linewidth=2,
                   label=f'Top-{len(top_k_indices)} cutoff')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/saliency_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved saliency distribution plot")


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    layer_idx = 3  # Analyze layer 3
    n_calibration_samples = 128
    top_k = 10
    k_offset = 0  # Offset to add to knee point for weight range analysis
    output_dir = './visualizations/saliency_gradient_analysis'

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map='auto' if device == 'cuda' else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create analyzer
    analyzer = SaliencyGradientAnalyzer(model, tokenizer, device)

    # Get calibration data
    calibration_data = analyzer.get_calibration_data(n_samples=n_calibration_samples)

    # Find target layer
    target_module, layer_idx = analyzer.find_target_layer(layer_idx=layer_idx)

    # Compute saliency and gradients
    print("\n" + "="*60)
    print(f"Analyzing Layer {layer_idx}")
    print("="*60)
    results = analyzer.compute_saliency_and_gradients(
        target_module, calibration_data, layer_idx
    )

    # Analyze top channels
    analyzer.analyze_top_channels(results, top_k=top_k, output_dir=output_dir, target_module=target_module, k_offset=k_offset)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
