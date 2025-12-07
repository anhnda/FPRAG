"""
Group-Wise PRAQ Implementation

This implements PRAQ with GROUP-WISE quantization instead of per-channel quantization.

Key Differences from Full PRAQ:
- Full PRAQ: Per-output-channel quantization (one scale per row of W)
- Group-Wise PRAQ: Per-group quantization along INPUT channels (one scale per group of columns)

Group-Wise Quantization:
- Divide input channels into groups of size `group_size` (default: 128)
- For each group, compute a single quantization scale
- This is more hardware-efficient and closer to real INT4 deployment

Algorithm:
1. Compute risk-aware salience:
   - MLP: Post-activation importance (accounts for activation function)
   - Attention: Standard AWQ-style (pre-activation)
2. Grid search for optimal α ∈ [0, 1]
3. Scale weight COLUMNS: W[:, j] *= s[j]^α
4. Quantize with GROUP-WISE scales (one scale per group of input channels)
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


class GroupWisePRAQQuantizer:
    """
    Group-Wise PRAQ: PRAQ algorithm with group-wise quantization.

    Key Features:
    - Risk-aware importance (post-activation for MLP layers)
    - Grid search for optimal scaling exponent α
    - GROUP-WISE INT4 quantization (one scale per group of input channels)
    - More hardware-efficient than per-channel quantization
    """

    def __init__(self, model, tokenizer, device="cuda", beta=3.0, tau=-3.0,
                 noise_factor=0.2, bits=4, n_grid=20, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.tau = tau
        self.noise_factor = noise_factor
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size

        # Storage for activations
        self.activation_data = {}
        self.hooks = []

        # Storage for scales
        self.layer_scales = {}

        # Detect layer types
        self.layer_types = self._detect_layer_types()

        mlp_count = sum(1 for t in self.layer_types.values() if t == 'mlp')
        attn_count = sum(1 for t in self.layer_types.values() if t == 'attention')

        print(f"\n[Group-Wise PRAQ Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: GROUP-WISE (one scale per {group_size} input channels)")
        print(f"  MLP layers (risk-aware): {mlp_count}")
        print(f"  Attention layers (AWQ-style): {attn_count}")
        print(f"  Risk parameters: beta={beta}, tau={tau}, noise_factor={noise_factor}")

    def _detect_layer_types(self):
        """Detect which layers are MLP vs attention."""
        layer_types = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                name_lower = name.lower()
                if any(kw in name_lower for kw in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'out_proj', 'attention']):
                    layer_types[name] = 'attention'
                elif any(kw in name_lower for kw in ['mlp', 'fc', 'gate', 'up_proj', 'down_proj', 'ffn']):
                    layer_types[name] = 'mlp'
                else:
                    layer_types[name] = 'mlp'
        return layer_types

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
    def get_risk_aware_salience_mlp(self, name, module):
        """
        Compute PRAQ salience for MLP layers using POST-ACTIVATION importance.

        Key PRAQ Insight: Measure importance based on OUTPUT after activation,
        not input before activation. This accounts for channels killed by ReLU/SiLU.

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        # For MiniCPM, MLPs use SiLU activation
        activation_fn = torch.nn.functional.silu

        X_list = self.activation_data[name]
        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else None

        in_features = W.shape[1]
        out_features = W.shape[0]

        # Accumulate post-activation importance on CPU
        output_importance_sum = torch.zeros(out_features)
        total_samples = 0

        # Process in batches to avoid OOM
        for x_batch in X_list:
            x_flat = x_batch.reshape(-1, x_batch.shape[-1])
            batch_size = x_flat.shape[0]

            # Compute pre-activation on GPU
            x_gpu = x_flat.to(self.device)
            z = torch.matmul(x_gpu, W.t())
            if b is not None:
                z = z + b

            # Apply activation (PRAQ KEY STEP!)
            y = activation_fn(z)

            # Measure post-activation output magnitude
            output_importance_sum += y.abs().sum(dim=0).cpu()
            total_samples += batch_size

            del x_gpu, z, y

        # Average post-activation output magnitude
        output_importance = output_importance_sum / total_samples

        # Backprop importance to input channels via weight magnitudes
        W_abs = W.abs().cpu().float()
        input_importance = torch.matmul(output_importance, W_abs)

        return input_importance

    @torch.no_grad()
    def get_activation_salience_awq(self, name):
        """
        Compute standard AWQ activation salience: E[|X|] per input feature.

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            salience_sum += x_flat.abs().sum(dim=0)

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def get_salience(self, name, module):
        """Get salience scores based on layer type."""
        layer_type = self.layer_types.get(name, 'mlp')

        if layer_type == 'mlp':
            return self.get_risk_aware_salience_mlp(name, module)
        else:
            return self.get_activation_salience_awq(name)

    @torch.no_grad()
    def quantize_weight_groupwise(self, W):
        """
        Group-wise quantization: Quantize weights with one scale per group of input channels.

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

        # Compute scale per group: [out_features, n_groups, 1]
        W_abs_max = W_grouped.abs().max(dim=2, keepdim=True)[0]
        W_abs_max = W_abs_max.clamp(min=1e-8)

        # INT4 range: [-8, 7]
        scale = W_abs_max / 7.0

        # Quantize
        W_int = torch.round(W_grouped / scale).clamp(-8, 7)

        # Dequantize
        W_dequant_grouped = W_int * scale

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

        Group-Wise PRAQ Algorithm:
        1. Compute risk-aware salience (post-activation for MLP)
        2. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           a. Compute scales: s[j] = (salience[j])^α
           b. Scale weight COLUMNS: W[:, j] *= s[j]
           c. Quantize with GROUP-WISE scales
           d. Compute compensated output: Y_q = W_q @ (X / s)
           e. Measure error: ||Y_q - Y_orig||²
        3. Return α and scales that minimize error

        Returns:
            best_scales: Optimal per-input-channel scales
            best_alpha: Best alpha value
            best_error: Minimum reconstruction error
        """
        salience = self.get_salience(name, module)

        if salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Check for invalid values
        if torch.isnan(salience).any() or torch.isinf(salience).any():
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        if salience.max() < 1e-10:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Normalize salience to [0.1, 10] range
        salience_min = salience.min()
        salience_max = salience.max()
        if salience_max > salience_min:
            salience = 0.1 + 9.9 * (salience - salience_min) / (salience_max - salience_min)
        else:
            salience = torch.ones_like(salience)

        # Get activations
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Limit samples
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

        # Grid search
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            scales = salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize with GROUP-WISE quantization
            W_quant = self.quantize_weight_groupwise(W_scaled)

            # Compensate input
            X_compensated = X_search / scales.unsqueeze(0)

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Compute error
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        # Clean up
        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Group-Wise PRAQ quantization to a layer.

        Steps:
        1. Grid search for best per-input-channel scales (risk-aware for MLP)
        2. Scale weight columns: W[:, j] *= scales[j]
        3. Quantize with GROUP-WISE scales (one scale per group)
        4. Divide by scales: W_final = Q(W*s) / s
        """
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        # Scale weight COLUMNS
        W_scaled = W * best_scales.unsqueeze(0)

        # Quantize with GROUP-WISE quantization
        W_quant = self.quantize_weight_groupwise(W_scaled)

        # Divide by scales
        W_final = W_quant / best_scales.unsqueeze(0)

        # Update weights
        module.weight.data = W_final

        # Store metadata
        layer_type = self.layer_types.get(name, 'mlp')
        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'type': layer_type
        }

        del W_scaled, W_quant, W_final
        torch.cuda.empty_cache()

    def calibrate(self, calibration_data, n_samples=500):
        """Run calibration to collect activations."""
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
                    print(f"\nNote: Some samples skipped")
                continue

        self.remove_hooks()
        print(f"Calibration complete! Successfully processed {successful}/{n_samples} samples")

    def quantize_model(self):
        """Quantize all linear layers using Group-Wise PRAQ."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise PRAQ")
        print("=" * 80)
        print("Method:")
        print("  1. MLP: Risk-aware post-activation importance")
        print("  2. Attention: AWQ-style pre-activation importance")
        print("  3. Grid search for optimal α ∈ [0, 1]")
        print("  4. Scale weight columns: W[:, j] *= s[j]^α")
        print(f"  5. GROUP-WISE INT4 quantization (group_size={self.group_size})")
        print("  6. Descaling: W_final = Q(W*s) / s")
        print("=" * 80)

        quantized_count = 0
        skipped_count = 0

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        for name, module in tqdm(layer_names, desc="Quantizing layers"):
            try:
                self.quantize_layer(name, module)
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

        # Print statistics
        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            mlp_alphas = [info['alpha'] for info in self.layer_scales.values() if info['type'] == 'mlp']
            attn_alphas = [info['alpha'] for info in self.layer_scales.values() if info['type'] == 'attention']

            print(f"\nOptimal α statistics (all layers):")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")
            print(f"  Min: {np.min(alphas):.3f}")
            print(f"  Max: {np.max(alphas):.3f}")

            if mlp_alphas:
                print(f"\nMLP layers (risk-aware):")
                print(f"  Mean α: {np.mean(mlp_alphas):.3f}")
                print(f"  Median α: {np.median(mlp_alphas):.3f}")

            if attn_alphas:
                print(f"\nAttention layers (AWQ-style):")
                print(f"  Mean α: {np.mean(attn_alphas):.3f}")
                print(f"  Median α: {np.median(attn_alphas):.3f}")

        self.activation_data = {}
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
        description="Group-Wise PRAQ quantization for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--beta", type=float, default=3.0, help="PRAQ beta parameter")
    parser.add_argument("--tau", type=float, default=-3.0, help="PRAQ tau parameter")
    parser.add_argument("--noise-factor", type=float, default=0.2, help="PRAQ noise factor")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_praq",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Group-Wise PRAQ: Risk-Aware Quantization with Group-Wise Scaling")
    print("=" * 80)
    print("Algorithm:")
    print("  1. MLP: Post-activation importance (E[|SiLU(XW+b)|])")
    print("  2. Attention: Pre-activation importance (E[|X|])")
    print("  3. Grid search for optimal α ∈ [0, 1]")
    print("  4. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  5. GROUP-WISE INT4 quantization (group_size={args.group_size})")
    print("  6. Descaling: W_final = Q(W*s) / s")
    print("\nKey Features:")
    print("  - Risk-aware importance for MLP layers")
    print("  - Group-wise quantization for hardware efficiency")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
    print(f"Risk params: beta={args.beta}, tau={args.tau}, noise={args.noise_factor}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Load model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_before = (param_size + buffer_size) / 1024**2
    print(f"Model size before quantization: {size_mb_before:.2f} MB")

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)

    # Initialize quantizer
    quantizer = GroupWisePRAQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beta=args.beta,
        tau=args.tau,
        noise_factor=args.noise_factor,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size
    )

    # Calibrate and quantize
    quantizer.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer.quantize_model()

    # Model size after
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
    print("\nGroup-Wise PRAQ Approach:")
    print("  ✓ MLP layers: POST-activation importance (risk-aware)")
    print("  ✓ Attention layers: Standard AWQ (pre-activation)")
    print("  ✓ Grid search optimization for scaling exponent")
    print(f"  ✓ GROUP-WISE quantization (group_size={args.group_size})")
    print("  ✓ Hardware-efficient quantization strategy")
    print("  ✓ Accounts for activation function effects!")
    print("=" * 80)


if __name__ == "__main__":
    main()
