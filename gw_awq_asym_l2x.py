"""
Group-Wise AWQ Implementation with ASYMMETRIC Quantization + L2 Salience (Enhanced)

This is gw_awq_asym_l2.py with winning features from awq_op_ref.py:
- float64 precision for activation stats (not float32)
- Sequential sampling in grid search (not random)
- bfloat16 model loading (not float16)
- activation_salience + 1e-6 (not .clamp(min=1e-5))

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm) in FLOAT64
2. Grid search for optimal α ∈ [0, 1] using SEQUENTIAL sampling
3. Scale weight COLUMNS: W[:, j] *= s[j]^α
4. Quantize with GROUP-WISE ASYMMETRIC scales
   - Per group: scale = (max - min) / 15, zero_point = round(-min / scale)
5. Divide by input scales: W_final = Q(W*s) / s
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


class GroupWiseAWQAsymmetricL2XQuantizer:
    """
    Group-Wise AWQ with Asymmetric Quantization and L2 Salience.

    Key Features:
    - Per-input-channel scaling based on E[X²] (L2 salience)
    - Grid search for optimal scaling exponent α
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
    - Better MSE alignment through L2 norm
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

        print(f"\n[Group-Wise AWQ ASYMMETRIC L2X Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric: E[X²] (L2 norm) - Better MSE alignment")
        print(f"  Enhanced features: float64 stats, sequential sampling, bfloat16 model")

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

        Key Difference from L1:
        - L1: E[|X|] - Linear weighting, average magnitude
        - L2: E[X²] - Quadratic weighting, emphasizes spikes

        Why L2 is Better:
        - Quantization error is quadratic: (δW × X)²
        - MSE = E[(δW × X)²] = E[δW²] × E[X²] (if independent)
        - L2 directly matches the MSE objective

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate L2 salience on CPU with FLOAT64 precision (from awq_op_ref.py)
        salience_sum = torch.zeros(in_features, dtype=torch.float64)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).double()
            # Use pow(2) for L2 norm
            salience_sum += x_flat.pow(2).sum(dim=0)

        salience = (salience_sum / total_samples).float()
        return salience

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
        Grid search for optimal per-input-channel scaling factor using L2 salience.

        Algorithm with L2:
        1. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           a. Compute scales: s[j] = (E[X[:, j]²])^α for each input channel j
           b. Scale weight COLUMNS: W_scaled[:, j] = W[:, j] * s[j]
           c. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
           d. Compute compensated output: Y_q = W_q @ (X / s)
           e. Measure error: ||Y_q - Y_orig||²
        2. Return α and scales that minimize error

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

        # Prepare calibration data with SEQUENTIAL sampling (from awq_op_ref.py)
        X_list = self.activation_data[name]
        X_cpu = []
        curr_len = 0
        for x in X_list:
            x_f = x.reshape(-1, x.shape[-1])
            X_cpu.append(x_f)
            curr_len += x_f.shape[0]
            if curr_len >= 2048:
                break

        X_search = torch.cat(X_cpu, dim=0)[:2048].to(self.device)

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

        # Add 1e-6 to avoid division by zero (from awq_op_ref.py)
        activation_salience = activation_salience + 1e-6

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            # Use pow(alpha) without clamping (from awq_op_ref.py)
            scales = activation_salience.pow(alpha)

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
        Apply Group-Wise AWQ with Asymmetric Quantization and L2 Salience.

        Steps:
        1. Grid search for best per-input-channel scales (L2-based)
        2. Scale weight columns: W[:, j] *= scales[j]
        3. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
        4. Divide by scales: W_final = Q(W*s) / s
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
        """Quantize all linear layers using Group-Wise AWQ with L2 Salience."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise AWQ ASYMMETRIC + L2 Salience")
        print("=" * 80)
        print("Method:")
        print("  1. Compute per-input-channel L2 salience: s[j] = E[X[:, j]²]")
        print("     → Emphasizes high-energy channels (better MSE alignment)")
        print("  2. Grid search for optimal α ∈ [0, 1]")
        print("  3. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={self.group_size})")
        print("     - Per group: scale = (max - min) / 15")
        print("     - Per group: zero_point = round(-min / scale)")
        print("  5. Divide by input scales: W_final = Q(W*s) / s")
        print("\nKey Improvement: L2 norm better matches MSE objective")
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

            # Debug: Print first 5 layers for comparison
            print(f"\nDEBUG - First 5 layer alphas:")
            for i, (name, info) in enumerate(list(self.layer_scales.items())[:5]):
                print(f"  {name}: α={info['alpha']:.4f}, error={info['error']:.8f}")

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
        description="Group-Wise AWQ with ASYMMETRIC quantization and L2 Salience for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_awq_asym_l2x",
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
    print("Group-Wise AWQ with ASYMMETRIC Quantization + L2 Salience (Enhanced)")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-input-channel L2 salience: s[j] = E[X[:, j]²] in FLOAT64")
    print("     → Better MSE alignment (quadratic weighting)")
    print("  2. Grid search optimal α ∈ [0, 1] with SEQUENTIAL sampling")
    print("  3. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("     - scale = (max - min) / 15 per group")
    print("     - zero_point = round(-min / scale) per group")
    print("  5. Descaling: W_final = Q(W*s) / s")
    print("\nEnhancements from awq_op_ref.py:")
    print("  ✓ float64 precision for activation statistics")
    print("  ✓ Sequential sampling (not random)")
    print("  ✓ bfloat16 model loading")
    print("  ✓ activation_salience + 1e-6 (not .clamp)")
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
    # Use bfloat16 instead of float16 (from awq_op_ref.py)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
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
    quantizer = GroupWiseAWQAsymmetricL2XQuantizer(
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
    print("\nGroup-Wise AWQ ASYMMETRIC + L2X Approach:")
    print("  ✓ L2 salience: E[X²] (MSE-aligned)")
    print("  ✓ Grid search for optimal α")
    print("  ✓ Column-wise weight scaling")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("\nEnhancements Applied:")
    print("  ✓ float64 precision for stats")
    print("  ✓ Sequential sampling")
    print("  ✓ bfloat16 model")
    print("  ✓ activation_salience + 1e-6")
    print("=" * 80)


if __name__ == "__main__":
    main()
