"""
Group-Wise AWQ with KNEE-point L2 (KL2) Importance-Weighted Loss

Key Innovation:
- Uses Kneedle algorithm to identify truly important channels
- Focuses MSE optimization only on high-salience channels
- Prevents wasting capacity on unimportant channels

Algorithm:
1. Compute per-channel L2 salience: s[j] = E[X[:, j]²]
2. Sort salience values
3. Apply Kneedle algorithm on first half to find knee point
4. Create importance mask: important[j] = (salience[j] >= knee_threshold)
5. Grid search for optimal α, but compute MSE ONLY on important channels
6. This focuses quantization precision on truly critical weights

Why This Works:
- Kneedle finds the "elbow" where salience drops significantly
- Channels above knee are high-impact, below knee are negligible
- Optimizing for high-impact channels prevents overfitting to noise
- Matches intuition: protect important channels, aggressive on others
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
    Group-Wise AWQ with Knee-point L2 (KL2) Importance Weighting.

    Key Features:
    - Kneedle algorithm to identify critical channels
    - Importance-weighted MSE: focus on high-salience channels only
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
    - Prevents optimization overfitting to unimportant channels
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
        print(f"  Innovation: Kneedle-based importance masking")
        print(f"  MSE computed ONLY on high-salience channels")

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

        # Sort in descending order
        sorted_salience = np.sort(salience_np)[::-1]

        # Take first half for knee detection
        half_point = n_channels // 2
        first_half = sorted_salience[:half_point]

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
                print(f"    ⚠️  Kneedle failed, using median fallback")

        except Exception as e:
            # Fallback: use median of first half
            knee_idx = len(first_half) // 2
            knee_threshold = first_half[knee_idx]
            print(f"    ⚠️  Kneedle error ({e}), using median fallback")

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
    def quantize_weight_groupwise_asymmetric(self, W):
        """
        Group-wise ASYMMETRIC quantization.
        Uses full INT4 range [0, 15] with computed zero_point.

        Args:
            W: Weight tensor [out_features, in_features]

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
        else:
            W_padded = W

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
        Grid search for optimal per-input-channel scaling factor.

        KEY INNOVATION: MSE computed ONLY on important channels (above knee point)

        Algorithm:
        1. Compute L2 salience for all channels
        2. Find knee point → identify important channels
        3. For α in [0, 0.05, ..., 1.0]:
           a. Compute scales: s[j] = salience[j]^α
           b. Quantize with these scales
           c. Compute MSE ONLY on important channels (not all!)
           d. Track best α
        4. Return scales that minimize important-channel MSE

        Returns:
            best_scales, best_alpha, best_error
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

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

        # Expand mask for broadcasting: [1, in_features] for inputs, affects all outputs
        # When computing Y = X @ W.t(), important input channels affect all output dims
        # So we mask the input: X_important = X[:, important_mask]

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize with GROUP-WISE ASYMMETRIC quantization
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

            # Compensate input
            X_compensated = X_search / scales.unsqueeze(0)

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # KEY INNOVATION: Compute MSE ONLY on important input channels
            # Since Y = X @ W.t(), we can compute per-channel contributions to output
            # and weight the error by channel importance

            # Method: Compute error on outputs from important channels only
            # Y = X @ W.t() = sum_j (X[:, j] * W[:, j])
            # Focus on: sum over j where important_mask[j] = True

            # Create masked versions for error computation
            # Zero out unimportant channels in both orig and quant
            X_important = X_search.clone()
            X_important[:, ~important_mask] = 0  # Zero out unimportant channels

            # Recompute outputs with only important channels
            if b is not None:
                Y_orig_important = torch.matmul(X_important, W.t()) + b
                Y_quant_important = torch.matmul(X_important / scales.unsqueeze(0), W_quant.t()) + b
            else:
                Y_orig_important = torch.matmul(X_important, W.t())
                Y_quant_important = torch.matmul(X_important / scales.unsqueeze(0), W_quant.t())

            # Compute MSE on important-channel outputs only
            error = (Y_orig_important - Y_quant_important).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Group-Wise AWQ with Knee-L2 Importance Weighting.

        Steps:
        1. Compute L2 salience
        2. Find knee point → identify important channels
        3. Grid search for best scales (MSE on important channels only)
        4. Quantize with best scales
        """
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        # Scale weight COLUMNS
        W_scaled = W * best_scales.unsqueeze(0)

        # Quantize with GROUP-WISE ASYMMETRIC quantization
        W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

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
        print("Quantizing with Group-Wise AWQ + Knee-L2 Importance Weighting")
        print("=" * 80)
        print("Method:")
        print("  1. Compute per-channel L2 salience: s[j] = E[X[:, j]²]")
        print("  2. Sort salience, apply Kneedle on first half")
        print("  3. Create importance mask: important = (salience >= knee)")
        print("  4. Grid search for optimal α ∈ [0, 1]")
        print("     → MSE computed focusing on important channels")
        print("  5. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  6. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={self.group_size})")
        print("  7. Descaling: W_final = Q(W*s) / s")
        print("\nKey Innovation: Knee-point identifies truly critical channels")
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
    print("Group-Wise AWQ with Knee-L2 Importance Weighting")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-channel L2 salience: s[j] = E[X[:, j]²]")
    print("  2. Sort salience → Kneedle algorithm → Find knee point")
    print("  3. Mask: important = (salience >= knee threshold)")
    print("  4. Grid search with importance-weighted MSE")
    print("  5. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  6. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("  7. Descaling: W_final = Q(W*s) / s")
    print("\nInnovation: Focus optimization on truly important channels")
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
    print("  ✓ Importance-weighted MSE optimization")
    print("  ✓ Focuses precision on high-impact channels")
    print("  ✓ Prevents overfitting to unimportant channels")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("=" * 80)


if __name__ == "__main__":
    main()
