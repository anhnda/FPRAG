"""
Group-Wise AWQ Implementation with Gradient-Squared (G²) Saliency

Key Difference from gw_awq_asym_l2.py:
- gw_awq_asym_l2.py: Uses E[X²] (L2 norm) for activation salience
- gw_awq_g2.py: Uses gradient-based saliency for SiLU-followed layers, AWQ for others

Algorithm:
For layers followed by SiLU:
1. Compute gradients g of SiLU output w.r.t. weight W
2. Normalize: g_norm = ln(1+g²)/max(ln(1+g²))
3. Saliency per channel: s[j] = E[1 + g_norm[j]]

For layers NOT followed by SiLU:
1. Use AWQ-style: s[j] = E[X[:, j]²] (L2 activation magnitude)

Then apply standard AWQ quantization:
1. Grid search for optimal α ∈ [0, 1]
2. Scale weight COLUMNS: W[:, j] *= s[j]^α
3. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
4. Divide by input scales: W_final = Q(W*s) / s
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


class GroupWiseAWQGradientSquaredQuantizer:
    """
    Group-Wise AWQ with Gradient-Squared (G²) Saliency for SiLU-followed layers.

    Key Features:
    - Gradient-based saliency for layers followed by SiLU: E[1 + ln(1+g²)/max(ln(1+g²))]
    - AWQ-style saliency for other layers: E[X²] (L2)
    - Grid search for optimal scaling exponent α
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
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

        # Detect which layers are followed by SiLU
        self.silu_followed_layers = self._detect_silu_layers()

        print(f"\n[Group-Wise AWQ G² Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric:")
        print(f"    - SiLU-followed layers: E[1 + ln(1+g²)/max(ln(1+g²))] (gradient-based)")
        print(f"    - Other layers: E[X²] (AWQ L2-style)")
        print(f"  Detected {len(self.silu_followed_layers)} SiLU-followed layers")

    def _detect_silu_layers(self):
        """
        Detect which linear layers are followed by SiLU activation.

        Heuristics:
        - MLP layers typically have SiLU/GELU activations
        - Keywords: 'gate', 'up_proj', 'fc1', 'mlp'

        Returns:
            Set of layer names that are followed by SiLU
        """
        silu_layers = set()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if layer name suggests it's followed by activation
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in ['gate', 'up_proj', 'fc1', 'w1', 'w3']):
                    silu_layers.add(name)
                # Also check for explicit SiLU/GELU patterns in MLP blocks
                elif 'mlp' in name_lower and any(x in name_lower for x in ['gate', 'up']):
                    silu_layers.add(name)

        return silu_layers

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                if isinstance(input, tuple):
                    inp = input[0].detach()
                else:
                    inp = input.detach()

                # Store on CPU to save GPU memory
                self.activation_data[name].append(inp.cpu())

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
    def get_activation_salience_awq(self, name):
        """
        Compute AWQ-style activation salience: E[X[:, j]²] (L2 norm)

        Used for layers NOT followed by SiLU.

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

    def compute_gradient_salience(self, name, module):
        """
        Compute gradient-based saliency for SiLU-followed layers.

        Algorithm:
        1. For each calibration sample:
           a. Compute forward: Z = X @ W^T + b
           b. Apply SiLU: Y = SiLU(Z) = Z * sigmoid(Z)
           c. Compute gradient: g = dY/dW for each weight
           d. Accumulate g² per channel
        2. Normalize: g_norm = ln(1+g²)/max(ln(1+g²))
        3. Saliency: s[j] = E[1 + g_norm[j]]

        Args:
            name: Layer name
            module: The linear module (to get W and b)

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        in_features = module.weight.shape[1]

        # Get weight and bias from module (no need to store separately!)
        W = module.weight.data.cpu()  # [out_features, in_features]
        b = module.bias.data.cpu() if module.bias is not None else None

        # Accumulate squared gradients per input channel ON CPU
        grad_squared_sum = torch.zeros(in_features)
        total_samples = 0

        for X in tqdm(X_list, desc=f"Computing gradients for {name}", leave=False):
            # X is on CPU and might be [batch, seq_len, hidden] - flatten it
            X_flat = X.reshape(-1, X.shape[-1])  # [batch * seq_len, in_features]

            # Process in small batches on GPU to avoid OOM
            batch_size = min(32, X_flat.shape[0])
            for start_idx in range(0, X_flat.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X_flat.shape[0])
                X_batch = X_flat[start_idx:end_idx].to(self.device)

                # Forward pass
                W_gpu = W.to(self.device)
                Z = torch.matmul(X_batch, W_gpu.t())
                if b is not None:
                    Z = Z + b.to(self.device)

                # Apply SiLU activation
                sigmoid_Z = torch.sigmoid(Z)
                d_silu = sigmoid_Z + Z * sigmoid_Z * (1 - sigmoid_Z)  # [batch, out_features]

                # Gradient magnitude per input channel
                # For each sample, compute contribution to each input channel
                X_abs = X_batch.abs()  # [batch, in_features]
                d_silu_sum = d_silu.abs().sum(dim=1, keepdim=True)  # [batch, 1]

                # grad_per_input[j] = sum_i |x_i[j] * d_silu[i]|
                grad_per_input = (X_abs * d_silu_sum).sum(dim=0)  # [in_features]

                # Move to CPU and accumulate
                grad_squared_sum += grad_per_input.pow(2).cpu()
                total_samples += X_batch.shape[0]

                # Clear GPU memory
                del X_batch, W_gpu, Z, sigmoid_Z, d_silu, X_abs, d_silu_sum, grad_per_input
                torch.cuda.empty_cache()

        # Average over samples
        grad_squared_avg = grad_squared_sum / max(total_samples, 1)

        # Apply logarithmic transformation: ln(1 + g²)
        log_grad_squared = torch.log1p(grad_squared_avg)  # log1p(x) = ln(1+x)

        # Normalize by max: ln(1+g²)/max(ln(1+g²))
        max_val = log_grad_squared.max()
        if max_val > 0:
            grad_norm = log_grad_squared / max_val
        else:
            grad_norm = log_grad_squared

        # Saliency: E[1 + ln(1+g²)/max(ln(1+g²))]
        salience = 1.0 + grad_norm

        # Debug: Check if gradient importance is actually varying
        print(f"    G² debug - min: {salience.min():.4f}, max: {salience.max():.4f}, "
              f"std: {salience.std():.4f}, range: {(salience.max()-salience.min()):.4f}")
        print(f"    Raw g² - min: {grad_squared_avg.min():.6f}, max: {grad_squared_avg.max():.6f}, "
              f"ratio: {grad_squared_avg.max()/grad_squared_avg.min():.2f}x")

        return salience

    def get_salience(self, name, module):
        """
        Get saliency for a layer based on whether it's followed by SiLU.

        Args:
            name: Layer name
            module: The linear module

        Returns:
            Tensor of shape [in_features]
        """
        if name in self.silu_followed_layers:
            # Use gradient-based saliency
            print(f"  Computing G² saliency for SiLU-followed layer: {name}")
            return self.compute_gradient_salience(name, module)
        else:
            # Use AWQ-style activation saliency
            return self.get_activation_salience_awq(name)

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

        Algorithm:
        1. Get saliency (gradient-based for SiLU layers, AWQ for others)
        2. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           a. Compute scales: s[j] = salience[j]^α for each input channel j
           b. Scale weight COLUMNS: W_scaled[:, j] = W[:, j] * s[j]
           c. Quantize with GROUP-WISE ASYMMETRIC scales [0, 15]
           d. Compute compensated output: Y_q = W_q @ (X / s)
           e. Measure error: ||Y_q - Y_orig||²
        3. Return α and scales that minimize error

        Returns:
            best_scales, best_alpha, best_error
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get salience (either gradient-based or AWQ-style)
        salience = self.get_salience(name, module)
        if salience is None:
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

        salience = salience.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from salience
            scales = salience.pow(alpha).clamp(min=1e-5)

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
        Apply Group-Wise AWQ with appropriate saliency metric.

        Steps:
        1. Grid search for best per-input-channel scales (gradient or AWQ-based)
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
            'error': best_error,
            'silu_followed': name in self.silu_followed_layers
        }

        # Clear intermediate tensors
        del W_scaled, W_quant, W_final
        torch.cuda.empty_cache()

    def calibrate(self, calibration_data, n_samples=500):
        """Run calibration on the dataset to collect activations."""
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
        print("Collecting activations for all layers...")
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
        print(f"  - {len(self.silu_followed_layers)} layers will use G² saliency")
        print(f"  - {len([n for n, m in self.model.named_modules() if isinstance(m, nn.Linear)]) - len(self.silu_followed_layers)} layers will use AWQ L2 saliency")

    def quantize_model(self):
        """Quantize all linear layers using Group-Wise AWQ with G² Saliency."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise AWQ + Gradient-Squared (G²) Saliency")
        print("=" * 80)
        print("Method:")
        print("  For SiLU-followed layers:")
        print("    1. Compute gradients g of SiLU output w.r.t. weight W")
        print("    2. Normalize: g_norm = ln(1+g²)/max(ln(1+g²))")
        print("    3. Saliency: s[j] = E[1 + g_norm[j]]")
        print("  For other layers:")
        print("    1. AWQ-style saliency: s[j] = E[X[:, j]²]")
        print("  Then for all layers:")
        print("    2. Grid search for optimal α ∈ [0, 1]")
        print("    3. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"    4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={self.group_size})")
        print("    5. Divide by input scales: W_final = Q(W*s) / s")
        print(f"\nSiLU-followed layers: {len(self.silu_followed_layers)}")
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
                        method = "G²" if info['silu_followed'] else "AWQ"
                        print(f"\n  Layer {name} [{method}]:")
                        print(f"    α={info['alpha']:.3f}, error={info['error']:.6f}")

                quantized_count += 1

                # Clear activation data immediately after quantizing
                if name in self.activation_data:
                    del self.activation_data[name]

                # More frequent cache clearing
                if quantized_count % 5 == 0 and torch.cuda.is_available():
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
            silu_alphas = [info['alpha'] for info in self.layer_scales.values() if info['silu_followed']]
            awq_alphas = [info['alpha'] for info in self.layer_scales.values() if not info['silu_followed']]

            print(f"\nOptimal α statistics:")
            print(f"  Overall - Mean: {np.mean(alphas):.3f}, Median: {np.median(alphas):.3f}")
            if silu_alphas:
                print(f"  SiLU layers (G²) - Mean: {np.mean(silu_alphas):.3f}, Median: {np.median(silu_alphas):.3f}")
            if awq_alphas:
                print(f"  Other layers (AWQ) - Mean: {np.mean(awq_alphas):.3f}, Median: {np.median(awq_alphas):.3f}")

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
        description="Group-Wise AWQ with Gradient-Squared (G²) Saliency for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_awq_g2",
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
    print("Group-Wise AWQ with Gradient-Squared (G²) Saliency")
    print("=" * 80)
    print("Algorithm:")
    print("  SiLU-followed layers:")
    print("    1. Compute gradients g of SiLU(XW) w.r.t. W")
    print("    2. Normalize: ln(1+g²)/max(ln(1+g²))")
    print("    3. Saliency: s[j] = E[1 + ln(1+g²)/max(ln(1+g²))]")
    print("  Other layers:")
    print("    1. AWQ-style: s[j] = E[X[:, j]²]")
    print("  All layers:")
    print("    2. Grid search optimal α ∈ [0, 1]")
    print("    3. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"    4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("    5. Descaling: W_final = Q(W*s) / s")
    print("\nMemory Optimization:")
    print("  - No duplicate storage: reuses activation_data for gradient computation")
    print("  - Gradient computation done in small GPU batches")
    print("  - Aggressive memory clearing after each layer")
    print("  - If OOM still occurs, set: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
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
    quantizer = GroupWiseAWQGradientSquaredQuantizer(
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
    print("\nGroup-Wise AWQ G² Approach:")
    print("  ✓ Gradient-based saliency for SiLU-followed layers")
    print("  ✓ AWQ-style saliency for other layers")
    print("  ✓ Grid search for optimal α")
    print("  ✓ Column-wise weight scaling")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("=" * 80)


if __name__ == "__main__":
    main()
