"""
Group-Wise AWQ with KNEE-point L2 (KL2) Importance-Aware Zero-Point

Key Innovation:
- Uses Kneedle algorithm to identify truly important channels
- Adjusts zero_point in asymmetric quantization to minimize error on important weights
- Prevents wasting precision on unimportant channels

Algorithm:
1. Compute per-channel L2 salience: s[j] = E[X[:, j]²]
2. Sort salience values
3. Apply Kneedle algorithm on first half to find knee point
4. Create importance mask: important[j] = (salience[j] >= knee_threshold)
5. Grid search for optimal α using standard MSE
6. During quantization: adjust zero_point to minimize error on important weight columns

Why This Works:
- Kneedle finds the "elbow" where salience drops significantly
- Channels above knee are high-impact, below knee are negligible
- Zero-point optimization focuses quantization range on important weights
- Matches intuition: allocate quantization budget to critical weights
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

try:
    from kneed import KneeLocator
except ImportError:
    print("⚠️  Warning: kneed library not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "kneed"])
    from kneed import KneeLocator


class GroupWiseAWQKneeL2Quantizer:
    """
    Group-Wise AWQ with Knee-point L2 (KL2) Importance-Aware Zero-Point.

    Key Features:
    - Kneedle algorithm to identify critical channels
    - Importance-aware zero_point: adjusts quantization range for important weights
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
    - Prevents wasting quantization budget on unimportant channels
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}
        self.layer_importance_stats = {}  # Store knee point stats

        print(f"\n[Group-Wise AWQ KNEE-L2 Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Innovation: Kneedle-based importance identification")
        print(f"  Zero-point adjusted to minimize error on important weights")

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

    def find_knee_threshold(self, salience):
        """
        Use Kneedle algorithm to find knee point on first half of sorted salience.

        Algorithm:
        1. Sort salience in descending order
        2. Take first half (most important channels)
        3. Apply Kneedle to find where importance drops sharply
        4. Return threshold: channels above = important, below = unimportant

        Args:
            salience: Tensor of shape [in_features]

        Returns:
            knee_threshold: float, channels with salience >= threshold are important
            important_mask: bool tensor, True for important channels
            stats: dict with knee point statistics
        """
        salience_np = salience.cpu().numpy()
        n_channels = len(salience_np)

        # Clean data: replace NaN/inf with 0
        salience_np = np.nan_to_num(salience_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Sort in descending order
        sorted_salience = np.sort(salience_np)[::-1]

        # Take first half for knee detection
        half_point = n_channels // 2
        first_half = sorted_salience[:half_point]

        # Validate data for Kneedle
        # Check if all values are the same or very close
        if len(first_half) < 3 or np.ptp(first_half) < 1e-10:
            # No clear knee, use median fallback
            knee_idx = len(first_half) // 2
            knee_threshold = first_half[knee_idx]
        else:
            # Prepare data for Kneedle
            x = np.arange(len(first_half))
            y = first_half

            # Find knee point using Kneedle algorithm
            # curve='convex' because salience decreases
            # direction='decreasing' because we're going from high to low
            try:
                knee = KneeLocator(
                    x, y,
                    curve='convex',
                    direction='decreasing',
                    online=False
                )

                if knee.knee is not None:
                    knee_idx = knee.knee
                    knee_threshold = first_half[knee_idx]
                else:
                    # Fallback: use median of first half
                    knee_idx = len(first_half) // 2
                    knee_threshold = first_half[knee_idx]

            except Exception as e:
                # Fallback: use median of first half
                knee_idx = len(first_half) // 2
                knee_threshold = first_half[knee_idx]

        # Create importance mask
        important_mask = salience >= knee_threshold

        # Statistics
        n_important = important_mask.sum().item()
        importance_ratio = n_important / n_channels

        stats = {
            'knee_threshold': float(knee_threshold),
            'knee_index': int(knee_idx),
            'n_important': n_important,
            'n_total': n_channels,
            'importance_ratio': importance_ratio,
            'max_salience': float(sorted_salience[0]),
            'min_salience': float(sorted_salience[-1]),
        }

        return knee_threshold, important_mask, stats

    @torch.no_grad()
    def _compute_importance_aware_zero_point(self, W_grouped, scale, importance_grouped):
        """
        Compute zero_point that minimizes MSE on important weight columns.

        For each group, searches over z ∈ [0, 15] to find the zero_point that
        minimizes quantization error on weight columns corresponding to important
        input channels (where importance_grouped[g, j] = True).

        Args:
            W_grouped: [out_features, n_groups, group_size]
            scale: [out_features, n_groups, 1]
            importance_grouped: [n_groups, group_size] boolean mask

        Returns:
            zero_point: [out_features, n_groups, 1]
        """
        out_features, n_groups, group_size = W_grouped.shape

        # Initialize with standard zero_point
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # For each group, optimize zero_point based on important columns
        for g in range(n_groups):
            importance_mask_g = importance_grouped[g]  # [group_size]

            # If no important columns in this group, use standard zero_point
            if not importance_mask_g.any():
                continue

            W_group_g = W_grouped[:, g, :]  # [out_features, group_size]
            scale_g = scale[:, g, :]  # [out_features, 1]

            # Grid search over z ∈ [0, 15]
            best_z = zero_point[:, g, 0].clone()
            best_error = float('inf')

            for z_candidate in range(16):
                # Quantize with this zero_point
                W_int = torch.round(W_group_g / scale_g + z_candidate).clamp(0, 15)
                W_dequant = (W_int - z_candidate) * scale_g

                # Compute MSE only on important columns
                error_per_col = (W_group_g - W_dequant).pow(2)  # [out_features, group_size]
                error_important = error_per_col[:, importance_mask_g].mean()  # scalar

                if error_important < best_error:
                    best_error = error_important
                    best_z = z_candidate

            # Apply same zero_point to all output features for this group
            zero_point[:, g, 0] = best_z

        return zero_point

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W, important_mask=None):
        """
        Group-wise ASYMMETRIC quantization with importance-aware zero_point.
        Uses full INT4 range [0, 15] with computed zero_point.

        When important_mask is provided, adjusts zero_point to minimize
        quantization error on weight columns corresponding to important
        input channels.

        Args:
            W: Weight tensor [out_features, in_features]
            important_mask: Optional boolean tensor [in_features] indicating important channels

        Returns:
            W_quant: Quantized and dequantized weights
        """
        out_features, in_features = W.shape

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W

            # Pad importance mask too
            if important_mask is not None:
                important_mask_padded = torch.zeros(padded_in_features, dtype=torch.bool, device=important_mask.device)
                important_mask_padded[:in_features] = important_mask
            else:
                important_mask_padded = None
        else:
            W_padded = W
            important_mask_padded = important_mask

        # Reshape to [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Reshape importance mask to [n_groups, group_size]
        if important_mask_padded is not None:
            importance_grouped = important_mask_padded.reshape(n_groups, self.group_size)
        else:
            importance_grouped = None

        # Compute min and max per group (standard, use full range)
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Standard scale computation
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)

        # Compute zero_point
        if importance_grouped is not None:
            # Importance-aware zero_point: minimize error on important columns
            zero_point = self._compute_importance_aware_zero_point(
                W_grouped, scale, importance_grouped
            )
        else:
            # Standard zero_point
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
        Grid search for optimal per-input-channel scaling factor.

        KEY INNOVATION: Identifies important channels via Kneedle, then passes
        importance mask to quantization function to adjust zero_point.

        Algorithm:
        1. Compute L2 salience for all channels
        2. Find knee point → identify important channels
        3. For α in [0, 0.05, ..., 1.0]:
           a. Compute scales: s[j] = salience[j]^α
           b. Quantize with these scales (importance mask used internally)
           c. Compute standard MSE (on all channels)
           d. Track best α
        4. Return scales and importance mask

        Returns:
            best_scales, best_alpha, best_error, important_mask
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0, None

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0, None

        # Find knee point and create importance mask
        knee_threshold, important_mask, stats = self.find_knee_threshold(activation_salience)

        # Store statistics
        self.layer_importance_stats[name] = stats

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

        activation_salience = activation_salience.to(self.device)
        important_mask = important_mask.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize with GROUP-WISE ASYMMETRIC quantization
            # Note: We don't pass important_mask here during search (for speed)
            # It will be used during final quantization in quantize_layer
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

            # Compensate input
            X_compensated = X_search / scales.unsqueeze(0)

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Compute standard MSE on all outputs
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error, important_mask

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Group-Wise AWQ with Knee-L2 Importance-Aware Zero-Point.

        Steps:
        1. Compute L2 salience
        2. Find knee point → identify important channels
        3. Grid search for best scales (standard MSE)
        4. Quantize with best scales AND importance mask (adjusts zero_point)
        """
        best_scales, best_alpha, best_error, important_mask = self.search_best_scale(name, module)

        W = module.weight.data

        # Scale weight COLUMNS
        W_scaled = W * best_scales.unsqueeze(0)

        # Quantize with GROUP-WISE ASYMMETRIC quantization
        # Pass importance mask to adjust zero_point for important weights
        W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled, important_mask)

        # Divide by scales to restore original magnitude
        W_final = W_quant / best_scales.unsqueeze(0)

        # Update module weights
        module.weight.data = W_final

        # Store metadata (including importance stats)
        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'importance_stats': self.layer_importance_stats.get(name, {})
        }

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
        """Quantize all linear layers using Group-Wise AWQ with Knee-L2."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise AWQ + Knee-L2 Importance-Aware Zero-Point")
        print("=" * 80)
        print("Method:")
        print("  1. Compute per-channel L2 salience: s[j] = E[X[:, j]²]")
        print("  2. Sort salience, apply Kneedle on first half")
        print("  3. Create importance mask: important = (salience >= knee)")
        print("  4. Grid search for optimal α ∈ [0, 1]")
        print("     → Standard MSE on all channels")
        print("  5. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  6. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={self.group_size})")
        print("     → Zero-point adjusted to minimize error on important weights")
        print("  7. Descaling: W_final = Q(W*s) / s")
        print("\nKey Innovation: Zero-point optimization focuses on critical weights")
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
                        stats = info.get('importance_stats', {})
                        print(f"\n  Layer {name}:")
                        print(f"    α={info['alpha']:.3f}, error={info['error']:.6f}")
                        if stats:
                            print(f"    Important channels: {stats['n_important']}/{stats['n_total']} "
                                  f"({stats['importance_ratio']*100:.1f}%)")
                            print(f"    Knee threshold: {stats['knee_threshold']:.6f}")

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
            importance_ratios = [
                info['importance_stats']['importance_ratio']
                for info in self.layer_scales.values()
                if 'importance_stats' in info and info['importance_stats']
            ]

            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")
            print(f"  Min: {np.min(alphas):.3f}")
            print(f"  Max: {np.max(alphas):.3f}")

            if importance_ratios:
                print(f"\nImportance ratio statistics:")
                print(f"  Mean: {np.mean(importance_ratios)*100:.1f}%")
                print(f"  Median: {np.median(importance_ratios)*100:.1f}%")
                print(f"  Min: {np.min(importance_ratios)*100:.1f}%")
                print(f"  Max: {np.max(importance_ratios)*100:.1f}%")

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
        description="Group-Wise AWQ with Knee-L2 Importance Weighting for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_awq_kl2",
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
    print("Group-Wise AWQ with Knee-L2 Importance-Aware Zero-Point")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-channel L2 salience: s[j] = E[X[:, j]²]")
    print("  2. Sort salience → Kneedle algorithm → Find knee point")
    print("  3. Mask: important = (salience >= knee threshold)")
    print("  4. Grid search with standard MSE")
    print("  5. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  6. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("     → Zero-point adjusted to minimize error on important weights")
    print("  7. Descaling: W_final = Q(W*s) / s")
    print("\nInnovation: Zero-point optimization focuses on critical weights")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid search points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
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
    quantizer = GroupWiseAWQKneeL2Quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size
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
    print("\nGroup-Wise AWQ Knee-L2 Approach:")
    print("  ✓ Kneedle algorithm identifies critical channels")
    print("  ✓ Zero-point adjusted to minimize error on important weights")
    print("  ✓ Focuses quantization budget on high-impact channels")
    print("  ✓ Prevents wasting precision on unimportant channels")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("=" * 80)


if __name__ == "__main__":
    main()
