"""
Group-Wise AWQ with Heuristic-Guided Asymmetric Quantization

Key Innovation: Uses activation-aware heuristic from heuristic_quantize.py
Combines:
1. AWQ's activation-aware scaling: scales = E[X²]^α
2. Heuristic quantization: Uses E[Xs[:,j]] to guide rounding decisions

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm)
2. Grid search for optimal α ∈ [0, 1]
3. Scale weight COLUMNS: W[:, j] *= s[j]^α
4. Compute scaled activations: Xs = X / s
5. Quantize with HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC:
   - For each group: compute group_activation = E[Xs[:, group]]
   - Use group_activation to guide quantization (minimize output error)
   - Use asymmetric quantization [0, 15] with computed zero_point
6. Divide by input scales: W_final = Q(W*s) / s

Motivation:
- Standard quantization: Round to nearest, minimize weight error
- Heuristic quantization: Consider activation impact, minimize output error
- Key insight: error = dot(X, W - W_quant) depends on BOTH weights and activations
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


class HeuristicGroupWiseAWQQuantizer:
    """
    Group-Wise AWQ with Heuristic-Guided Asymmetric Quantization.

    Key Features:
    - Per-input-channel scaling based on E[X²] (L2 salience)
    - Grid search for optimal scaling exponent α
    - HEURISTIC-GUIDED GROUP-WISE quantization using E[Xs]
    - Minimizes output error rather than just weight error
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128, use_heuristic=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_heuristic = use_heuristic

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Heuristic Group-Wise AWQ Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric: E[X²] (L2 norm)")
        print(f"  Heuristic guidance: E[Xs] per group (signed mean)")
        print(f"  Use heuristic: {use_heuristic} (optimized with sampling)")

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
        This is used for AWQ scaling.
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
    def get_scaled_activation_mean(self, name, scales):
        """
        Compute mean of scaled activations: E[Xs[:, j]] where Xs = X / scales
        This is used for heuristic quantization guidance.

        Returns:
            Tensor of shape [in_features] with SIGNED mean values
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Move scales to CPU for computation (activations are stored on CPU)
        scales_cpu = scales.cpu() if scales.device.type != 'cpu' else scales

        # Accumulate scaled activation mean on CPU
        scaled_mean_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            # Scale the activations (both on CPU now)
            xs_flat = x_flat / scales_cpu.unsqueeze(0)
            scaled_mean_sum += xs_flat.sum(dim=0)

        scaled_mean = scaled_mean_sum / total_samples
        return scaled_mean

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, group_activation_means, apply_heuristic=True):
        """
        Heuristic-guided group-wise ASYMMETRIC quantization (OPTIMIZED).

        Optimizations:
        - Fast path: Skip heuristic for small layers
        - Vectorized: Process multiple groups in parallel where possible
        - Early exit: Skip if error is already small

        Args:
            W: Weight tensor [out_features, in_features]
            group_activation_means: Mean scaled activation per channel [in_features]
            apply_heuristic: If False, use standard min/max quantization

        Returns:
            W_quant: Quantized and dequantized weights
        """
        out_features, in_features = W.shape

        # Fast path: skip heuristic for small layers
        if not apply_heuristic or out_features * in_features < 100000:
            apply_heuristic = False

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W

            # Pad activation means with zeros
            group_activation_padded = torch.zeros(padded_in_features, device=W.device, dtype=W.dtype)
            group_activation_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            group_activation_padded = group_activation_means

        # Reshape to [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)
        group_act_grouped = group_activation_padded.reshape(n_groups, self.group_size)

        # Compute min and max per group for asymmetric quantization
        W_min = W_grouped.min(dim=2, keepdim=True)[0]  # [out_features, n_groups, 1]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]  # [out_features, n_groups, 1]

        # Asymmetric quantization parameters
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # Initial quantization (nearest rounding)
        W_div = W_grouped / scale
        W_int = torch.round(W_div + zero_point).clamp(0, 15)

        # === HEURISTIC REFINEMENT (OPTIMIZED) ===
        if apply_heuristic:
            # Process only every Nth output channel to speed up (sample-based)
            stride = max(1, out_features // 100)  # Process at most 100 output channels

            for out_idx in range(0, out_features, stride):
                for group_idx in range(n_groups):
                    w_group = W_grouped[out_idx, group_idx]  # [group_size]
                    w_int_group = W_int[out_idx, group_idx]  # [group_size]
                    group_scale = scale[out_idx, group_idx, 0]
                    group_zp = zero_point[out_idx, group_idx, 0]

                    # Get activation mean for this group
                    x_group = group_act_grouped[group_idx]  # [group_size]

                    # Current quantized weights
                    w_quant_group = (w_int_group - group_zp) * group_scale

                    # Compute current error: error = dot(x, w - w_quant)
                    current_error = torch.dot(x_group, w_group - w_quant_group)

                    # Early exit if error is small
                    if abs(current_error) < 1e-5:
                        continue

                    # Calculate flip direction (up or down from current quantized value)
                    w_div_group = W_div[out_idx, group_idx] + group_zp
                    flip_direction = torch.sign(w_div_group - w_int_group)
                    flip_direction[flip_direction == 0] = 1.0

                    # Calculate impact of flipping each weight
                    flip_impacts = x_group * flip_direction * group_scale

                    # Filter: impact must have same sign as error (to reduce it)
                    target_sign = torch.sign(current_error)
                    valid_mask = (torch.sign(flip_impacts) == target_sign)

                    # Ensure bits don't overflow [0, 15]
                    w_int_proposed = w_int_group + flip_direction
                    in_range = (w_int_proposed >= 0) & (w_int_proposed <= 15)
                    valid_mask = valid_mask & in_range

                    if not valid_mask.any():
                        continue

                    valid_indices = torch.nonzero(valid_mask).squeeze()
                    if valid_indices.ndim == 0:
                        valid_indices = valid_indices.unsqueeze(0)

                    candidate_impacts = flip_impacts[valid_indices]

                    # Simplified: just take top-k by impact (no sorting by rounding error)
                    k = min(3, len(candidate_impacts))  # Limit flips per group
                    if k > 0:
                        top_k_vals, top_k_idx = torch.topk(candidate_impacts.abs(), k)
                        indices_to_flip = valid_indices[top_k_idx]
                        W_int[out_idx, group_idx, indices_to_flip] += flip_direction[indices_to_flip].long()

        # Dequantize with adjusted integer values
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

            # Compute scaled activation means for heuristic
            scaled_act_mean = self.get_scaled_activation_mean(name, scales)
            scaled_act_mean_gpu = scaled_act_mean.to(self.device)

            # Quantize with HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC quantization
            W_quant = self.quantize_weight_heuristic_groupwise(W_scaled, scaled_act_mean_gpu, apply_heuristic=self.use_heuristic)

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
        Apply Heuristic Group-Wise AWQ Quantization.
        """
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        # Scale weight COLUMNS
        W_scaled = W * best_scales.unsqueeze(0)

        # Compute scaled activation means for heuristic
        scaled_act_mean = self.get_scaled_activation_mean(name, best_scales)
        if scaled_act_mean is None:
            # Fallback: use standard group-wise quantization
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device)
        else:
            scaled_act_mean = scaled_act_mean.to(W.device)

        # Quantize with HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC quantization
        W_quant = self.quantize_weight_heuristic_groupwise(W_scaled, scaled_act_mean, apply_heuristic=self.use_heuristic)

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
        """Quantize all linear layers using Heuristic Group-Wise AWQ."""
        print("\n" + "=" * 80)
        print("Quantizing with Heuristic-Guided Group-Wise AWQ")
        print("=" * 80)
        print("Method:")
        print("  1. Compute per-input-channel L2 salience: s[j] = E[X[:, j]²]")
        print("  2. Grid search for optimal α ∈ [0, 1]")
        print("  3. Scale weight columns: W[:, j] *= s[j]^α")
        print("  4. Compute scaled activation means: E[Xs[:,j]] where Xs = X / s")
        print(f"  5. HEURISTIC-GUIDED GROUP-WISE quantization [0, 15] (group_size={self.group_size})")
        print("     - Per group: use E[Xs[group]] to guide rounding")
        print("     - Minimize output error: dot(Xs, W - W_quant)")
        print("     - Greedy flip selection to reduce residual")
        print("  6. Divide by input scales: W_final = Q(W*s) / s")
        print("\nKey Innovation: Activation-aware heuristic minimizes output error")
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
        description="Heuristic Group-Wise AWQ for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--use-heuristic", action="store_true", default=True, help="Use heuristic guidance (default: True)")
    parser.add_argument("--no-heuristic", action="store_false", dest="use_heuristic", help="Disable heuristic guidance")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_awq_heuristic",
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
    print("Heuristic-Guided Group-Wise AWQ")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-input-channel L2 salience: s[j] = E[X[:, j]²]")
    print("  2. Grid search optimal α ∈ [0, 1]")
    print("  3. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print("  4. Compute scaled activation means: E[Xs] where Xs = X / s")
    print(f"  5. HEURISTIC-GUIDED GROUP-WISE INT4 [0, 15] (group_size={args.group_size})")
    print("     - Uses E[Xs[group]] to minimize output error per group")
    print("  6. Descaling: W_final = Q(W*s) / s")
    print("\nKey Innovation: Heuristic uses activation impact to guide quantization")
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
    quantizer = HeuristicGroupWiseAWQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic
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
    print("\nHeuristic-Guided Group-Wise AWQ Approach:")
    print("  ✓ L2 salience: E[X²] for scaling")
    print("  ✓ Scaled activation guidance: E[Xs] per group")
    print("  ✓ Heuristic minimizes output error")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("  ✓ Activation-aware rounding decisions")
    print("=" * 80)


if __name__ == "__main__":
    main()
