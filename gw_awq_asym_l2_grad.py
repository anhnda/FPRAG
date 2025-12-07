"""
Group-Wise AWQ Implementation with ASYMMETRIC Quantization + L2×Gradient Salience

Key Difference from gw_awq_asym_l2.py:
- gw_awq_asym_l2.py: Uses E[X²] (L2 norm) for ALL layers
- gw_awq_asym_l2_grad.py: HYBRID approach
  * MLP layers (with SiLU): E[X² * |∇SiLU|] (gradient-weighted)
  * Other layers (no SiLU): E[X²] (standard L2 fallback)

Why Hybrid L2×Gradient is Better:
- For MLP with SiLU: Combines input magnitude with gradient sensitivity
- E[X² * |∇Y/∇X|] captures both: "how large?" and "how sensitive?"
- For attention/other layers: Standard L2 (no activation to compute gradient)
- SiLU gradient: ∇SiLU(z) = sigmoid(z) + z·sigmoid(z)·(1-sigmoid(z))
- No backward pass needed! Gradient computed analytically from reconstructed Z = W^T·X + b
- No SiLU hooks needed! We only hook linear layer inputs, then reconstruct everything

Algorithm:
1. Hook linear layers to capture inputs X (no SiLU hooks needed!)
2. Detect if layer has SiLU activation following it (mapping only)
3. FOR MLP LAYERS (with SiLU):
   a. Reconstruct pre-activation from captured data: Z = W^T·X + b
   b. Compute SiLU gradient analytically: ∇SiLU(Z) = sigmoid(Z) + Z·sigmoid(Z)·(1-sigmoid(Z))
   c. Compute input gradient: ∇Y/∇X = ∇SiLU(Z) @ W
   d. Compute per-channel salience: s[j] = E[X[:, j]² * |∇Y/∇X[:, j]|]
4. FOR OTHER LAYERS (no SiLU):
   a. Compute per-channel salience: s[j] = E[X[:, j]²]
5. Grid search for optimal α ∈ [0, 1] with GRADIENT-WEIGHTED MSE:
   a. For MLP: Error = E[(Y_q - Y_orig)² * |∇SiLU(Y_orig)|]
   b. For others: Error = E[(Y_q - Y_orig)²]
6. Scale weight COLUMNS: W[:, j] *= s[j]^α
7. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
8. Divide by input scales: W_final = Q(W*s) / s
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


class GroupWiseAWQAsymmetricL2GradQuantizer:
    """
    Group-Wise AWQ with Asymmetric Quantization and HYBRID L2×Gradient Salience.

    Key Features:
    - HYBRID salience scoring:
      * MLP layers (with SiLU): E[X² * |∇Y/∇X|] (gradient-weighted)
      * Other layers (no SiLU): E[X²] (standard L2)
    - Analytical SiLU gradient (no backward pass)
    - Grid search for optimal scaling exponent α
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
    - Better sensitivity-aware importance scoring where applicable
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size

        # Storage for activations
        self.activation_data = {}  # Linear layer inputs (X)
        self.linear_to_silu = {}  # Map linear layer names to their SiLU successors
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Group-Wise AWQ ASYMMETRIC HYBRID L2×Gradient Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric (HYBRID):")
        print(f"    - MLP layers (with SiLU): E[X² * |∇SiLU|] (gradient-weighted)")
        print(f"    - Other layers (no SiLU): E[X²] (standard L2)")

    def _find_silu_layers(self):
        """
        Build mapping from linear layers to their subsequent SiLU activations.
        Common patterns:
        - model.layers.X.mlp.gate_proj → model.layers.X.mlp.act_fn (SiLU)
        - model.layers.X.mlp.up_proj → model.layers.X.mlp.act_fn (SiLU)
        """
        linear_layers = {}
        silu_layers = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
            elif isinstance(module, nn.SiLU) or (hasattr(module, '__class__') and 'SiLU' in module.__class__.__name__):
                silu_layers[name] = module

        # Match linear layers to SiLU by shared prefix
        for linear_name in linear_layers:
            # Get parent path (e.g., "model.layers.0.mlp" from "model.layers.0.mlp.gate_proj")
            parts = linear_name.split('.')
            for i in range(len(parts), 0, -1):
                parent_path = '.'.join(parts[:i])
                # Look for SiLU in same parent
                for silu_name in silu_layers:
                    if silu_name.startswith(parent_path):
                        self.linear_to_silu[linear_name] = silu_name
                        break
                if linear_name in self.linear_to_silu:
                    break

        print(f"\n[Layer Mapping]")
        print(f"  Found {len(linear_layers)} linear layers")
        print(f"  Found {len(silu_layers)} SiLU layers")
        print(f"  Mapped {len(self.linear_to_silu)} linear→SiLU connections")

    def register_hooks(self):
        """Register forward hooks to capture linear layer inputs."""
        # First, find SiLU layers for mapping (don't need to hook them)
        self._find_silu_layers()

        # Hook for linear layer inputs only
        def get_linear_hook(name):
            def hook(_module, input, _output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                if isinstance(input, tuple):
                    inp = input[0].detach()
                else:
                    inp = input.detach()
                self.activation_data[name].append(inp)
            return hook

        # Register hooks only for linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_linear_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @torch.no_grad()
    def compute_silu_gradient(self, z):
        """
        Compute SiLU gradient analytically.

        SiLU(z) = z * sigmoid(z)
        d(SiLU)/dz = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
                   = sigmoid(z) * (1 + z * (1 - sigmoid(z)))

        Args:
            z: Pre-activation values [batch, features]

        Returns:
            grad_silu: Gradient d(SiLU)/dz [batch, features]
        """
        sigmoid_z = torch.sigmoid(z)
        grad_silu = sigmoid_z * (1 + z * (1 - sigmoid_z))
        return grad_silu

    @torch.no_grad()
    def get_activation_salience_l2(self, name):
        """
        FALLBACK: Compute per-input-channel activation salience using standard L2 norm: E[X[:, j]²]

        Used for layers WITHOUT SiLU activation (e.g., attention layers).

        Args:
            name: Layer name

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
            # Standard L2: E[X²]
            salience_sum += x_flat.pow(2).sum(dim=0).cpu()

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def get_activation_salience_l2_grad(self, name, module):
        """
        Compute per-input-channel activation salience using L2×Gradient:
        s[j] = E[X[:, j]² * |∇Y/∇X[:, j]|]

        Steps:
        1. Get linear layer inputs X
        2. Reconstruct pre-activation Z = W^T·X + b
        3. Compute SiLU gradient: ∇SiLU(Z)
        4. Compute input gradient: ∇Y/∇X = ∇SiLU(Z) @ W
        5. Compute L2×Grad salience: E[X² * |∇Y/∇X|]

        Args:
            name: Layer name
            module: Linear module

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else None

        X_list = self.activation_data[name]
        in_features = X_list[0].shape[-1]

        # Accumulate salience on CPU
        salience_sum = torch.zeros(in_features)
        total_samples = 0

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])  # [N, in_features]
            batch_size = x_flat.shape[0]

            # Move to device for computation
            x_device = x_flat.to(self.device)

            # Compute pre-activation: Z = X @ W^T + b
            if b is not None:
                z = torch.matmul(x_device, W.t()) + b  # [N, out_features]
            else:
                z = torch.matmul(x_device, W.t())

            # Compute SiLU gradient analytically
            grad_silu = self.compute_silu_gradient(z)  # [N, out_features]

            # Compute input gradient: ∇Y/∇X = ∇SiLU @ W
            # grad_input[i, j] = sum_k (grad_silu[i, k] * W[k, j])
            grad_input = torch.matmul(grad_silu, W)  # [N, in_features]

            # Compute L2×Gradient salience per channel
            # s[j] = sum_i (X[i, j]² * |grad_input[i, j]|)
            x2 = x_device.pow(2)  # [N, in_features]
            grad_abs = grad_input.abs()  # [N, in_features]
            channel_salience = (x2 * grad_abs).sum(dim=0).cpu()  # [in_features]

            salience_sum += channel_salience
            total_samples += batch_size

        # Average over all samples
        salience = salience_sum / total_samples
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
        Grid search for optimal per-input-channel scaling factor using HYBRID salience.

        HYBRID Algorithm:
        - IF layer has SiLU (MLP): Use L2×Gradient salience s[j] = E[X[:, j]² * |∇Y/∇X[:, j]|]
        - ELSE (no SiLU): Use L2 salience s[j] = E[X[:, j]²]

        1. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           a. Compute scales: s[j] = salience[j]^α (using appropriate salience metric)
           b. Scale weight COLUMNS: W_scaled[:, j] = W[:, j] * s[j]
           c. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
           d. Compute compensated output: Y_q = W_q @ (X / s)
           e. Measure error:
              - MLP (SiLU): Gradient-weighted MSE = E[(Y_q - Y_orig)² * |∇SiLU(Y_orig)|]
              - Other:      Standard MSE = E[(Y_q - Y_orig)²]
        2. Return α and scales that minimize error

        Key Innovation: Gradient-weighted MSE penalizes errors on high-sensitivity outputs more!

        Returns:
            best_scales, best_alpha, best_error, salience_type
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # HYBRID: Choose salience metric based on layer type
        has_silu = name in self.linear_to_silu

        if has_silu:
            # MLP layer with SiLU: Use L2×Gradient salience
            activation_salience = self.get_activation_salience_l2_grad(name, module)
            salience_type = "L2×Grad"
        else:
            # Other layer (attention, etc.): Use standard L2 salience
            activation_salience = self.get_activation_salience_l2(name)
            salience_type = "L2"

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

        # Compute gradient weights for MLP layers (if applicable)
        if has_silu:
            # Compute SiLU gradient from original output for weighting
            grad_silu_orig = self.compute_silu_gradient(Y_orig)  # [N, out_features]
            grad_weight = grad_silu_orig.abs()  # Use absolute gradient as weight
        else:
            grad_weight = None

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from salience (L2×Grad or L2)
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

            # Compute reconstruction error
            if has_silu:
                # GRADIENT-WEIGHTED MSE for MLP layers
                # Weight errors by gradient magnitude - high-gradient outputs penalized more
                squared_error = (Y_orig - Y_quant).pow(2)  # [N, out_features]
                weighted_error = squared_error * grad_weight  # Element-wise weighting
                error = weighted_error.mean().item()
            else:
                # STANDARD MSE for other layers
                error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        if 'grad_weight' in locals() and grad_weight is not None:
            del grad_weight
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error, salience_type

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Group-Wise AWQ with Asymmetric Quantization and HYBRID Salience.

        Steps:
        1. Grid search for best per-input-channel scales (HYBRID: L2×Grad or L2)
        2. Scale weight columns: W[:, j] *= scales[j]
        3. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
        4. Divide by scales: W_final = Q(W*s) / s
        """
        best_scales, best_alpha, best_error, salience_type = self.search_best_scale(name, module)

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
            'error': best_error,
            'salience_type': salience_type
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

            except Exception:
                if i % 100 == 0 and i > 0:
                    print(f"\nNote: Some samples skipped due to errors")
                continue

        self.remove_hooks()
        print(f"Calibration complete! Successfully processed {successful}/{n_samples} samples")

    def quantize_model(self):
        """Quantize all linear layers using Group-Wise AWQ with HYBRID Salience."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise AWQ ASYMMETRIC + HYBRID Salience")
        print("=" * 80)
        print("HYBRID Method:")
        print("  1. Capture inputs X at linear layers (no SiLU hooks!)")
        print("  2. Detect layer type (MLP with SiLU vs. other)")
        print("  3. FOR MLP LAYERS (with SiLU):")
        print("     a. Reconstruct pre-activation: Z = W^T·X + b")
        print("     b. Compute SiLU gradient analytically: ∇SiLU(Z)")
        print("     c. Compute input gradient: ∇Y/∇X = ∇SiLU @ W")
        print("     d. Compute L2×Grad salience: s[j] = E[X[:, j]² * |∇Y/∇X[:, j]|]")
        print("        → Combines magnitude (X²) with sensitivity (|∇|)")
        print("  4. FOR OTHER LAYERS (no SiLU):")
        print("     a. Compute L2 salience: s[j] = E[X[:, j]²]")
        print("  5. Grid search for optimal α ∈ [0, 1] with GRADIENT-WEIGHTED MSE:")
        print("     a. MLP layers: Error = E[(Y_q - Y)² * |∇SiLU(Y)|]")
        print("     b. Other layers: Error = E[(Y_q - Y)²]")
        print("  6. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  7. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={self.group_size})")
        print("     - Per group: scale = (max - min) / 15")
        print("     - Per group: zero_point = round(-min / scale)")
        print("  8. Divide by input scales: W_final = Q(W*s) / s")
        print("\nKey Improvement: Gradient-weighted MSE → penalize errors on sensitive outputs")
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
                        print(f"    Method: {info['salience_type']}, α={info['alpha']:.3f}, error={info['error']:.6f}")

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
            # Salience type statistics
            salience_types = [info['salience_type'] for info in self.layer_scales.values()]
            l2grad_count = salience_types.count('L2×Grad')
            l2_count = salience_types.count('L2')
            print(f"\n📊 Salience Method Distribution:")
            print(f"  L2×Grad (MLP with SiLU): {l2grad_count} layers")
            print(f"  L2 (other layers): {l2_count} layers")

            # Alpha statistics
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
        description="Group-Wise AWQ with ASYMMETRIC quantization and HYBRID Salience (L2×Grad for MLP, L2 for others) for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_awq_asym_l2_grad",
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
    print("Group-Wise AWQ with ASYMMETRIC Quantization + HYBRID Salience")
    print("=" * 80)
    print("HYBRID Algorithm:")
    print("  1. Hook linear layers only (no SiLU hooks needed!)")
    print("  2. Detect layer type (MLP with SiLU vs. other)")
    print("  3. FOR MLP LAYERS (with SiLU):")
    print("     → Reconstruct Z = W^T·X + b from captured data")
    print("     → L2×Gradient salience: s[j] = E[X[:, j]² * |∇Y/∇X[:, j]|]")
    print("     → Combines input magnitude with output sensitivity")
    print("     → SiLU gradient computed analytically (no backward!)")
    print("  4. FOR OTHER LAYERS (attention, etc.):")
    print("     → L2 salience: s[j] = E[X[:, j]²]")
    print("  5. Grid search optimal α ∈ [0, 1]")
    print("  6. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  7. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("     - scale = (max - min) / 15 per group")
    print("     - zero_point = round(-min / scale) per group")
    print("  8. Descaling: W_final = Q(W*s) / s")
    print("\nKey Improvement: Efficient hybrid → reconstruct Z, no extra hooks")
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
    quantizer = GroupWiseAWQAsymmetricL2GradQuantizer(
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
    print("\nGroup-Wise AWQ ASYMMETRIC + HYBRID Salience Approach:")
    print("  ✓ HYBRID salience scoring:")
    print("    - MLP layers (with SiLU): E[X² * |∇SiLU|]")
    print("    - Other layers: E[X²]")
    print("  ✓ Analytical SiLU gradient (no backward)")
    print("  ✓ Grid search for optimal α")
    print("  ✓ Column-wise weight scaling")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("  ✓ Sensitivity-aware for MLP, standard L2 for others")
    print("=" * 80)


if __name__ == "__main__":
    main()
