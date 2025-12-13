"""
AWQ with Heuristic Option (awq_ho.py)

Strategy: Copy EXACT code from gw_awq_asym_l2.py, then add heuristic option.
When heuristic is OFF, behavior is IDENTICAL to gw_awq_asym_l2.py.
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


class AWQHeuristicOptionQuantizer:
    """
    AWQ Quantizer with Optional Heuristic-Guided Rounding.

    When use_heuristic=False: EXACT same as GroupWiseAWQAsymmetricL2Quantizer
    When use_heuristic=True: Adds global greedy rounding correction
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, use_heuristic=False, outlier_percent=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_heuristic = use_heuristic
        self.outlier_percent = outlier_percent

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[AWQ Heuristic Option Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Heuristic mode: {'ENABLED' if use_heuristic else 'DISABLED'}")
        if use_heuristic:
            print(f"  Outlier protection: Top {outlier_percent*100:.1f}% ignored")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Salience metric: E[X²] (L2 norm)")

    # ========== EXACT COPY FROM gw_awq_asym_l2.py ==========

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
        EXACT COPY from gw_awq_asym_l2.py
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
    def quantize_weight_groupwise_asymmetric(self, W):
        """
        Group-wise ASYMMETRIC quantization.
        EXACT COPY from gw_awq_asym_l2.py
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
        EXACT COPY from gw_awq_asym_l2.py when use_heuristic=False
        Modified to use heuristic quantization when use_heuristic=True
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

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

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize - use heuristic if enabled
            if self.use_heuristic:
                # Compute activation mean for heuristic
                mean_sum = torch.zeros(W.shape[1], device='cpu')
                for x in self.activation_data[name]:
                    x_flat = x.reshape(-1, x.shape[-1])
                    mean_sum += x_flat.sum(dim=0)
                total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in self.activation_data[name])
                act_mean = (mean_sum / total_samples).to(self.device)
                scaled_act_mean = act_mean / scales

                W_quant = self.quantize_weight_heuristic_groupwise(W_scaled, scaled_act_mean)
            else:
                # EXACT same as gw_awq_asym_l2.py
                W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

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

        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply quantization.
        EXACT COPY from gw_awq_asym_l2.py when use_heuristic=False
        """
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        # Scale weight COLUMNS
        W_scaled = W * best_scales.unsqueeze(0)

        # Quantize
        if self.use_heuristic:
            # Compute activation mean for heuristic
            mean_sum = torch.zeros(W.shape[1], device='cpu')
            for x in self.activation_data[name]:
                x_flat = x.reshape(-1, x.shape[-1])
                mean_sum += x_flat.sum(dim=0)
            total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in self.activation_data[name])
            act_mean = (mean_sum / total_samples).to(self.device).to(W.dtype)
            scaled_act_mean = act_mean / best_scales

            W_quant = self.quantize_weight_heuristic_groupwise(W_scaled, scaled_act_mean)
        else:
            # EXACT same as gw_awq_asym_l2.py
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

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

    # ========== HEURISTIC QUANTIZATION (only used when use_heuristic=True) ==========

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, group_activation_means):
        """
        Group-wise asymmetric quantization with global greedy rounding correction.
        Only called when use_heuristic=True.
        """
        out_features, in_features = W.shape
        device = W.device

        # Padding
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        # Reshape to groups
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Asymmetric Quantization Setup
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]
        max_int = 15

        scale = (W_max - W_min) / max_int
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, max_int)

        # Expand to full size for heuristic computation
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zero_point.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        # Initial Quantization
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_quant = (W_int - zp_flat) * scale_flat

        # Global Greedy Heuristic
        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1)

        flip_dir = torch.sign(W_div + zp_flat - W_int)
        flip_dir[flip_dir == 0] = 1.0

        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat

        # Validity Masks
        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        w_int_proposed = W_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # Outlier Masking
        k_outliers = int(padded_in_features * self.outlier_percent)
        if k_outliers > 0:
            _, outlier_indices = torch.topk(act_padded.abs(), k_outliers)
            is_outlier = torch.zeros(padded_in_features, dtype=torch.bool, device=device)
            is_outlier[outlier_indices] = True
            valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        # Sorting & Optimization
        rounding_costs = (W_div + zp_flat - W_int).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)

        error_unsqueezed = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        # Apply Flips
        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())

        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0
        W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)

        # Dequantize
        W_dequant = (W_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    # ========== EXACT COPY FROM gw_awq_asym_l2.py ==========

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

            except Exception:
                if i % 100 == 0 and i > 0:
                    print(f"\nNote: Some samples skipped due to errors")
                continue

        self.remove_hooks()
        print(f"Calibration complete! Successfully processed {successful}/{n_samples} samples")

    def quantize_model(self):
        """Quantize all linear layers."""
        print("\n" + "=" * 80)
        mode_str = "WITH Heuristic" if self.use_heuristic else "WITHOUT Heuristic (EXACT as gw_awq_asym_l2.py)"
        print(f"Quantizing with AWQ {mode_str}")
        print("=" * 80)
        print("Method:")
        print("  1. Compute per-input-channel L2 salience: s[j] = E[X[:, j]²]")
        print("  2. Grid search for optimal α ∈ [0, 1]")
        print("  3. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]")
        if self.use_heuristic:
            print("     → WITH global greedy rounding correction")
            print(f"     → WITH outlier masking (top {self.outlier_percent*100:.1f}%)")
        else:
            print("     → Standard nearest rounding (IDENTICAL to gw_awq_asym_l2.py)")
        print("  5. Divide by input scales: W_final = Q(W*s) / s")
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
                        print(f"\n  Layer {name}:")
                        print(f"    α={info['alpha']:.3f}, error={info['error']:.6f}")

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
        description="AWQ with Heuristic Option for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--use-heuristic", action="store_true", help="Enable heuristic rounding")
    parser.add_argument("--outlier-percent", type=float, default=0.05,
                       help="Percent of outliers to ignore (only used if heuristic enabled)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (auto-set based on heuristic mode if not provided)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Auto-set output directory based on mode
    if args.output_dir is None:
        if args.use_heuristic:
            args.output_dir = "./quantized_models/minicpm_awq_ho_heuristic"
        else:
            args.output_dir = "./quantized_models/minicpm_awq_ho_standard"

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("AWQ with Heuristic Option (awq_ho.py)")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-input-channel L2 salience: s[j] = E[X[:, j]²]")
    print("  2. Grid search optimal α ∈ [0, 1]")
    print("  3. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    if args.use_heuristic:
        print("     → WITH heuristic: Global greedy rounding correction")
        print(f"     → WITH outlier masking: Top {args.outlier_percent*100:.1f}% ignored")
    else:
        print("     → WITHOUT heuristic: EXACT copy of gw_awq_asym_l2.py")
    print("  5. Descaling: W_final = Q(W*s) / s")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid search points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
    print(f"Heuristic mode: {'ENABLED' if args.use_heuristic else 'DISABLED'}")
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
    quantizer = AWQHeuristicOptionQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        outlier_percent=args.outlier_percent
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
    print("\nAWQ Heuristic Option Approach:")
    print("  ✓ L2 salience: E[X²] (MSE-aligned)")
    print("  ✓ Grid search for optimal α")
    print("  ✓ Column-wise weight scaling")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    if args.use_heuristic:
        print("  ✓ WITH heuristic: Global greedy rounding")
        print(f"  ✓ WITH outlier masking: Top {args.outlier_percent*100:.1f}%")
    else:
        print("  ✓ WITHOUT heuristic: IDENTICAL to gw_awq_asym_l2.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
