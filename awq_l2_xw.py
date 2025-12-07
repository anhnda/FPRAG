"""
Group-Wise AWQ Implementation with ASYMMETRIC Quantization + ||X||²||W||² Salience

Key Difference from gw_awq_asym_l2.py:
- gw_awq_asym_l2.py: Uses E[X²] (L2 norm) for activation salience only
- awq_l2_xw.py: Uses ||X||²||W||² (combined activation and weight energy) for channel salience

Why ||X||²||W||² is Better:
- Quantization MSE ∝ E[(δW × X)²]
- Since δW ∝ ||W||, we have MSE ∝ ||W||² × E[X²]
- ||X||² captures total activation energy per channel
- ||W||² captures total weight energy per channel
- Their product identifies channels that are both high-activation AND high-weight

Algorithm:
1. Compute per-input-channel salience: s[j] = ||X[:, j]||² × ||W[:, j]||²
   - ||X[:, j]||² = sum of X[:, j]² over all calibration samples
   - ||W[:, j]||² = sum of W[:, j]² over all output features
2. Grid search for optimal α ∈ [0, 1]
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


class GroupWiseAWQAsymmetricL2XW:
    """
    Group-Wise AWQ with Asymmetric Quantization and ||X||²||W||² Salience.

    Key Features:
    - Per-input-channel scaling based on ||X[:, j]||² × ||W[:, j]||²
    - Combines activation energy and weight energy
    - Grid search for optimal scaling exponent α
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
    - Better MSE alignment through joint X-W energy metric
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

        print(f"\n[Group-Wise AWQ ASYMMETRIC ||X||²||W||² Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric: ||X||²||W||² - Joint activation-weight energy")

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
    def get_activation_weight_salience(self, name, W):
        """
        Compute per-input-channel salience using ||X[:, j]||² × ||W[:, j]||²

        Key Innovation:
        - Combines activation energy (||X||²) and weight energy (||W||²)
        - Identifies channels that are both high-activation AND high-weight
        - Better MSE predictor: MSE ∝ ||W||² × E[X²] ∝ ||W||² × ||X||²

        Args:
            name: Layer name
            W: Weight tensor [out_features, in_features]

        Returns:
            Tensor of shape [in_features] containing ||X[:, j]||² × ||W[:, j]||²
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        in_features = X_list[0].shape[-1]

        # Compute ||X[:, j]||² for each input channel j
        X_l2_squared = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])  # [batch*seq, in_features]
            # Sum of squares across all samples for each channel
            X_l2_squared += x_flat.pow(2).sum(dim=0)

        # Compute ||W[:, j]||² for each input channel j
        # W is [out_features, in_features], so W[:, j] is all weights for input channel j
        W_l2_squared = W.pow(2).sum(dim=0)  # [in_features]

        # Combine: ||X[:, j]||² × ||W[:, j]||²
        salience = X_l2_squared.to(W.device) * W_l2_squared

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
        Grid search for optimal per-input-channel scaling factor using ||X||²||W||² salience.

        Algorithm:
        1. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           a. Compute scales: s[j] = (||X[:, j]||² × ||W[:, j]||²)^α for each input channel j
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

        W = module.weight.data

        # Get ||X||²||W||² activation-weight salience
        activation_weight_salience = self.get_activation_weight_salience(name, W)
        if activation_weight_salience is None:
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

        b = module.bias.data if module.bias is not None else None

        # Compute original output
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_weight_salience = activation_weight_salience.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from ||X||²||W||² salience
            scales = activation_weight_salience.pow(alpha).clamp(min=1e-5)

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
        Apply Group-Wise AWQ with Asymmetric Quantization and ||X||²||W||² Salience.

        Steps:
        1. Grid search for best per-input-channel scales (||X||²||W||²-based)
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
        """Quantize all linear layers using Group-Wise AWQ with ||X||²||W||² Salience."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise AWQ ASYMMETRIC + ||X||²||W||² Salience")
        print("=" * 80)
        print("Method:")
        print("  1. Compute per-input-channel ||X||²||W||² salience:")
        print("     s[j] = ||X[:, j]||² × ||W[:, j]||²")
        print("     → Combines activation energy and weight energy")
        print("  2. Grid search for optimal α ∈ [0, 1]")
        print("  3. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={self.group_size})")
        print("     - Per group: scale = (max - min) / 15")
        print("     - Per group: zero_point = round(-min / scale)")
        print("  5. Divide by input scales: W_final = Q(W*s) / s")
        print("\nKey Innovation: Joint X-W energy metric for better MSE prediction")
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
        description="Group-Wise AWQ with ASYMMETRIC quantization and ||X||²||W||² Salience for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_awq_l2_xw",
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
    print("Group-Wise AWQ with ASYMMETRIC Quantization + ||X||²||W||² Salience")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-input-channel ||X||²||W||² salience:")
    print("     s[j] = ||X[:, j]||² × ||W[:, j]||²")
    print("     → Combines activation energy and weight energy")
    print("  2. Grid search optimal α ∈ [0, 1]")
    print("  3. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("     - scale = (max - min) / 15 per group")
    print("     - zero_point = round(-min / scale) per group")
    print("  5. Descaling: W_final = Q(W*s) / s")
    print("\nKey Innovation: Joint X-W energy better predicts MSE")
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
    quantizer = GroupWiseAWQAsymmetricL2XW(
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
    print("\nGroup-Wise AWQ ASYMMETRIC + ||X||²||W||² Approach:")
    print("  ✓ ||X||²||W||² salience (joint X-W energy)")
    print("  ✓ Grid search for optimal α")
    print("  ✓ Column-wise weight scaling")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("  ✓ Better MSE prediction through joint metric")
    print("=" * 80)


if __name__ == "__main__":
    main()
