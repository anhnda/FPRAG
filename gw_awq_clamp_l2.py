"""
Group-Wise AWQ Implementation with ASYMMETRIC Quantization + L2 Salience + Knee-Point Weight Clamping

Key Difference from gw_awq_asym_l2.py:
- gw_awq_asym_l2.py: No weight clamping, uses full weight range
- gw_awq_clamp_l2.py: Uses Kneedle algorithm to find knee point in sorted E[X²] importance,
                       then clamps weights to the range of top-k+k_offset important weights

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm)
2. For each output channel:
   a. Compute per-weight importance: w_imp[i,j] = |W[i,j]| × s[j]
   b. Sort weights by importance and find knee point k on first half (Kneedle)
   c. Compute k_offset = int(k_offset_ratio × num_weights) (default: 5%)
   d. Get weight range: [min(W_top), max(W_top)] where W_top = top (k+k_offset) weights
   e. Clamp all weights in channel to this range
3. Grid search for optimal α ∈ [0, 1]
4. Scale weight COLUMNS: W[:, j] *= s[j]^α
5. Quantize with GROUP-WISE ASYMMETRIC scales
   - Per group: scale = (max - min) / 15, zero_point = round(-min / scale)
6. Divide by input scales: W_final = Q(W*s) / s
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
from kneed import KneeLocator


class GroupWiseAWQClampL2Quantizer:
    """
    Group-Wise AWQ with Asymmetric Quantization, L2 Salience, and Knee-Point Weight Clamping.

    Key Features:
    - Per-input-channel scaling based on E[X²] (L2 salience)
    - Knee-point detection for each channel using Kneedle algorithm
    - Weight clamping to range of top-k+k_offset important weights
    - Grid search for optimal scaling exponent α
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128, k_offset_ratio=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.k_offset_ratio = k_offset_ratio

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}
        self.layer_clamp_stats = {}

        print(f"\n[Group-Wise AWQ ASYMMETRIC L2 + Knee-Point Clamping Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  K-offset ratio: {k_offset_ratio * 100:.1f}% (knee point offset)")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric: E[X²] (L2 norm)")
        print(f"  Weight clamping: Based on knee-point analysis")

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

    @torch.no_grad()
    def get_activation_salience_l2(self, name):
        """
        Compute per-input-channel activation salience using L2 norm: E[X[:, j]²]

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate L2 salience on CPU
        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            salience_sum += x_flat.pow(2).sum(dim=0)

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def compute_weight_clamp_ranges(self, W, activation_salience):
        """
        Compute per-channel weight clamping ranges using knee-point detection.

        For each output channel i:
        1. Compute weight importance: w_imp[i,j] = |W[i,j]| × s[j]
        2. Sort by importance, find knee point k on first half using Kneedle
        3. k_offset = int(k_offset_ratio × in_features)
        4. Get range from top (k + k_offset) weights
        5. Return [min_clamp[i], max_clamp[i]]

        Args:
            W: Weight tensor [out_features, in_features]
            activation_salience: Tensor [in_features] with E[X²] per input channel

        Returns:
            min_clamp: Tensor [out_features] - minimum value per channel
            max_clamp: Tensor [out_features] - maximum value per channel
            clamp_stats: Dict with statistics
        """
        out_features, in_features = W.shape
        min_clamp = torch.zeros(out_features, device=W.device)
        max_clamp = torch.zeros(out_features, device=W.device)

        k_offset = max(1, int(self.k_offset_ratio * in_features))
        knee_points = []
        clamp_ratios = []

        for i in range(out_features):
            # Compute weight importance for this channel
            weight_importance = torch.abs(W[i, :]) * activation_salience

            # Sort by importance
            sorted_importance, sorted_indices = torch.sort(weight_importance, descending=True)

            # Find knee point on first half
            half_n = in_features // 2
            try:
                x_range = np.arange(half_n)
                y_values = sorted_importance[:half_n].cpu().numpy()

                knee_locator = KneeLocator(
                    x_range, y_values,
                    curve='convex', direction='decreasing',
                    S=1.0
                )

                if knee_locator.knee is not None:
                    knee_k = int(knee_locator.knee)
                else:
                    # Fallback: use 10% of features
                    knee_k = max(1, in_features // 10)
            except:
                # Fallback if Kneedle fails
                knee_k = max(1, in_features // 10)

            knee_points.append(knee_k)

            # Get top k+k_offset weights
            top_k = min(knee_k + k_offset, in_features)
            top_indices = sorted_indices[:top_k]
            top_weights = W[i, top_indices]

            # Compute clamp range
            min_clamp[i] = top_weights.min()
            max_clamp[i] = top_weights.max()

            # Track statistics
            original_range = W[i, :].max() - W[i, :].min()
            clamped_range = max_clamp[i] - min_clamp[i]
            clamp_ratios.append((clamped_range / (original_range + 1e-8)).item())

        clamp_stats = {
            'mean_knee': np.mean(knee_points),
            'median_knee': np.median(knee_points),
            'mean_clamp_ratio': np.mean(clamp_ratios),
            'k_offset': k_offset
        }

        return min_clamp, max_clamp, clamp_stats

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric_clamped(self, W, min_clamp, max_clamp):
        """
        Group-wise ASYMMETRIC quantization with per-channel weight clamping.

        Args:
            W: Weight tensor [out_features, in_features]
            min_clamp: Tensor [out_features] - minimum clamp value per channel
            max_clamp: Tensor [out_features] - maximum clamp value per channel

        Returns:
            W_quant: Quantized and dequantized weights
        """
        out_features, in_features = W.shape

        # Apply per-channel clamping
        W_clamped = W.clone()
        for i in range(out_features):
            W_clamped[i, :] = torch.clamp(W_clamped[i, :], min=min_clamp[i], max=max_clamp[i])

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W_clamped
        else:
            W_padded = W_clamped

        # Reshape to [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Compute min and max per group
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization parameters
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # Quantize to [0, 15]
        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, 15)

        # Dequantize
        W_dequant_grouped = (W_int - zero_point) * scale

        # Reshape back
        W_dequant = W_dequant_grouped.reshape(out_features, padded_in_features)

        # Remove padding if added
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Grid search for optimal per-input-channel scaling factor using L2 salience.
        Includes weight clamping based on knee-point analysis.

        Returns:
            best_scales, best_alpha, best_error, clamp_stats
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0, {}

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0, {}

        # Prepare calibration data
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        max_samples = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        # Compute original output
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)
        best_clamp_stats = {}

        activation_salience = activation_salience.to(self.device)

        # Compute weight clamp ranges (done once, independent of alpha)
        min_clamp, max_clamp, clamp_stats = self.compute_weight_clamp_ranges(W, activation_salience)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize with GROUP-WISE ASYMMETRIC quantization + CLAMPING
            W_quant = self.quantize_weight_groupwise_asymmetric_clamped(W_scaled, min_clamp, max_clamp)

            # Compensate input
            X_compensated = X_search / scales.unsqueeze(0)

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Compute reconstruction error (MSE)
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()
                best_clamp_stats = clamp_stats

        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error, best_clamp_stats

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Group-Wise AWQ with Asymmetric Quantization, L2 Salience, and Weight Clamping.

        Steps:
        1. Grid search for best per-input-channel scales (L2-based)
        2. Compute weight clamp ranges using knee-point detection
        3. Scale weight columns: W[:, j] *= scales[j]
        4. Clamp weights per channel based on knee-point analysis
        5. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
        6. Divide by scales: W_final = Q(W*s) / s
        """
        best_scales, best_alpha, best_error, clamp_stats = self.search_best_scale(name, module)

        W = module.weight.data

        # Get activation salience for clamping
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            activation_salience = torch.ones(W.shape[1], device=self.device)
        else:
            activation_salience = activation_salience.to(self.device)

        # Compute clamp ranges
        min_clamp, max_clamp, _ = self.compute_weight_clamp_ranges(W, activation_salience)

        # Scale weight COLUMNS
        W_scaled = W * best_scales.unsqueeze(0)

        # Quantize with GROUP-WISE ASYMMETRIC quantization + CLAMPING
        W_quant = self.quantize_weight_groupwise_asymmetric_clamped(W_scaled, min_clamp, max_clamp)

        # Divide by scales to restore original magnitude
        W_final = W_quant / best_scales.unsqueeze(0)

        # Update module weights
        module.weight.data = W_final

        # Store metadata
        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error
        }

        self.layer_clamp_stats[name] = clamp_stats

    def calibrate(self, calibration_data, n_samples=500):
        """Run calibration on the dataset to collect activations."""
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
        self.model.eval()
        self.register_hooks()

        successful = 0
        for i, text in enumerate(tqdm(calibration_data[:n_samples], desc="Calibration")):
            try:
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    _ = self.model(**inputs, use_cache=False, return_dict=True)

                successful += 1

            except Exception as e:
                if i % 100 == 0 and i > 0:
                    print(f"\nNote: Some samples skipped due to errors")
                continue

        self.remove_hooks()
        print(f"Calibration complete! Successfully processed {successful}/{n_samples} samples")

    def quantize_model(self):
        """Quantize all linear layers using Group-Wise AWQ with L2 Salience and Weight Clamping."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise AWQ ASYMMETRIC + L2 Salience + Knee-Point Clamping")
        print("=" * 80)
        print("Method:")
        print("  1. Compute per-input-channel L2 salience: s[j] = E[X[:, j]²]")
        print("  2. For each output channel:")
        print("     a. Compute weight importance: w_imp[i,j] = |W[i,j]| × s[j]")
        print("     b. Sort by importance, find knee point k on first half (Kneedle)")
        print(f"     c. k_offset = {self.k_offset_ratio * 100:.1f}% × num_weights")
        print("     d. Clamp to range of top (k + k_offset) weights")
        print("  3. Grid search for optimal α ∈ [0, 1]")
        print("  4. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  5. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={self.group_size})")
        print("  6. Divide by input scales: W_final = Q(W*s) / s")
        print("\nKey Innovation: Knee-point based weight clamping for better quantization")
        print("=" * 80)

        quantized_count = 0
        skipped_count = 0

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        for name, module in tqdm(layer_names, desc="Quantizing layers"):
            try:
                self.quantize_layer(name, module)

                if quantized_count % 10 == 0 and quantized_count > 0:
                    if name in self.layer_scales:
                        info = self.layer_scales[name]
                        clamp_info = self.layer_clamp_stats.get(name, {})
                        print(f"\n  Layer {name}:")
                        print(f"    α={info['alpha']:.3f}, error={info['error']:.6f}")
                        if clamp_info:
                            print(f"    knee={clamp_info.get('mean_knee', 0):.1f}, "
                                  f"clamp_ratio={clamp_info.get('mean_clamp_ratio', 0):.3f}")

                quantized_count += 1

                # Clear activation data
                if name in self.activation_data:
                    del self.activation_data[name]

                if quantized_count % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Error quantizing layer {name}: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                continue

        print(f"\n✅ Quantization complete!")
        print(f"   Total linear layers quantized: {quantized_count}")
        if skipped_count > 0:
            print(f"   ⚠️  Skipped {skipped_count} layers due to errors")

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")
            print(f"  Min: {np.min(alphas):.3f}")
            print(f"  Max: {np.max(alphas):.3f}")

        if self.layer_clamp_stats:
            mean_knees = [info.get('mean_knee', 0) for info in self.layer_clamp_stats.values()]
            clamp_ratios = [info.get('mean_clamp_ratio', 0) for info in self.layer_clamp_stats.values()]
            print(f"\nWeight clamping statistics:")
            print(f"  Mean knee point: {np.mean(mean_knees):.1f}")
            print(f"  Mean clamp ratio: {np.mean(clamp_ratios):.3f} (clamped/original range)")

        self.activation_data = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_wikitext2(split="train", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Group-Wise AWQ with ASYMMETRIC quantization, L2 Salience, and Knee-Point Clamping for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--k-offset-ratio", type=float, default=0.05,
                       help="K-offset ratio for knee point (default: 5%% of weights)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_awq_clamp_l2",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Group-Wise AWQ with ASYMMETRIC Quantization + L2 Salience + Knee-Point Clamping")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-input-channel L2 salience: s[j] = E[X[:, j]²]")
    print("  2. Knee-point detection per channel:")
    print("     - Compute w_imp[i,j] = |W[i,j]| × s[j]")
    print("     - Sort and find knee k on first half (Kneedle)")
    print(f"     - k_offset = {args.k_offset_ratio * 100:.1f}% × num_weights")
    print("     - Clamp to range of top (k + k_offset) weights")
    print("  3. Grid search optimal α ∈ [0, 1]")
    print("  4. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  5. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("  6. Descaling: W_final = Q(W*s) / s")
    print("\nKey Innovation: Knee-point based range limiting reduces quantization outliers")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid search points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
    print(f"K-offset ratio: {args.k_offset_ratio * 100:.1f}%")
    print(f"Output directory: {args.output_dir}")
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

    # Get model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_before = (param_size + buffer_size) / 1024**2
    print(f"Model size before quantization: {size_mb_before:.2f} MB")

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)

    # Initialize quantizer
    quantizer = GroupWiseAWQClampL2Quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        k_offset_ratio=args.k_offset_ratio
    )

    # Calibrate and quantize
    quantizer.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer.quantize_model()

    # Get model size after
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_after = (param_size + buffer_size) / 1024**2
    print(f"\nModel size after quantization: {size_mb_after:.2f} MB")
    print(f"Compression ratio: {size_mb_before / size_mb_after:.2f}x")

    # Save model
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)
    print(f"Quantized model saved to: {args.output_dir}")
    print("\nGroup-Wise AWQ ASYMMETRIC + L2 + Knee-Point Clamping Approach:")
    print("  ✓ L2 salience: E[X²] (MSE-aligned)")
    print("  ✓ Knee-point detection (Kneedle algorithm)")
    print(f"  ✓ Weight clamping: top (k + {args.k_offset_ratio * 100:.1f}%) weights")
    print("  ✓ Grid search for optimal α")
    print("  ✓ Column-wise weight scaling")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("  ✓ Reduced quantization outliers through range limiting")
    print("=" * 80)


if __name__ == "__main__":
    main()
