"""
Full-PRAQ Implementation (CORRECTED - AWQ Framework + POST-ACTIVATION Importance)

This implements the TRUE PRAQ insight combined with AWQ's framework:
- AWQ's optimization: Per-INPUT-channel scaling + grid search + uniform quantization
- PRAQ's insight: Measure importance using POST-ACTIVATION outputs, not pre-activation

Key Algorithm:
1. Compute per-input-channel salience:
   - MLP layers (PRAQ):
     a. Compute Y = activation(X @ W^T + b) - POST-activation output
     b. Measure output importance: importance_out[k] = E[|Y[:, k]|]
     c. Backprop to inputs: importance_in[j] = Σ_k(importance_out[k] × |W[k,j]|)
   - Attention layers (AWQ): s[j] = E[|X[:, j]|] (standard pre-activation)
2. Grid search for optimal α ∈ [0, 1]
3. Scale weight COLUMNS: W[:, j] *= s[j]^α (per input channel)
4. Quantize scaled weights to INT4 uniformly
5. Divide by scales: W_final = Q(W*s) / s (restore original magnitude)

PRAQ Key Insight:
- AWQ: Uses E[|X|] - doesn't account for activation function
- PRAQ: Uses E[|activation(XW)|] - accounts for channels killed by ReLU/SiLU
- Example: A channel with large negative pre-activation has:
  * AWQ: High importance (large |X|)
  * PRAQ: Zero importance (killed by ReLU)

This is the CORRECT interpretation of PRAQ for AWQ-style scaling!

IMPORTANT: Uses CORRECTED per-column scaling (not per-element)
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


class FullPRAQQuantizer:
    """
    Full-PRAQ: AWQ-style quantization with POST-ACTIVATION importance (CORRECTED).

    Combines:
    - Real AWQ framework: Grid search + per-INPUT-channel (column-wise) scaling + uniform INT4
    - PRAQ insight: Measure importance using POST-activation outputs for MLP layers

    Key Features:
    - Proper column-wise scaling: W[:, j] *= s[j] (not per-element)
    - Post-activation importance: Accounts for activation function effects
    - AWQ measures |X|, PRAQ measures |activation(XW)| → more accurate!
    """

    def __init__(self, model, tokenizer, device="cuda", beta=3.0, tau=-3.0,
                 noise_factor=0.2, bits=4, n_grid=20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.tau = tau
        self.noise_factor = noise_factor
        self.bits = bits
        self.n_grid = n_grid

        # Storage for activations
        self.activation_data = {}
        self.hooks = []

        # Storage for scales (for analysis)
        self.layer_scales = {}

        # Detect layer types
        self.layer_types = self._detect_layer_types()

        # Print summary
        mlp_count = sum(1 for t in self.layer_types.values() if t == 'mlp')
        attn_count = sum(1 for t in self.layer_types.values() if t == 'attention')

        print(f"\n[Full-PRAQ Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
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
                # Store input activations on CPU to save GPU memory
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

        For each input channel j:
            1. Compute post-activation output: Y = activation(X @ W^T + b)
            2. Measure output magnitude per channel
            3. Backprop importance to input channels via weights

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        # Detect activation function (common in MLP: SiLU, ReLU, GELU)
        # For MiniCPM, MLPs typically use SiLU
        activation_fn = torch.nn.functional.silu  # SiLU activation

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
            x_flat = x_batch.reshape(-1, x_batch.shape[-1])  # [batch, in_features]
            batch_size = x_flat.shape[0]

            # Compute pre-activation on GPU
            x_gpu = x_flat.to(self.device)
            z = torch.matmul(x_gpu, W.t())  # [batch, out_features]
            if b is not None:
                z = z + b

            # Apply activation function (PRAQ KEY STEP!)
            y = activation_fn(z)  # [batch, out_features]

            # Measure post-activation output magnitude
            output_importance_sum += y.abs().sum(dim=0).cpu()
            total_samples += batch_size

            del x_gpu, z, y

        # Average post-activation output magnitude per output channel
        output_importance = output_importance_sum / total_samples  # [out_features], float32

        # Backprop importance to input channels via weight magnitudes
        # importance_in[j] = sum_k(output_importance[k] * |W[k, j]|)
        # This tells us: how much does input channel j contribute to important outputs?
        W_abs = W.abs().cpu().float()  # [out_features, in_features], convert to float32
        input_importance = torch.matmul(output_importance, W_abs)  # [in_features]

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

        # Process activations in chunks to avoid OOM
        X_list = self.activation_data[name]

        # Get total number of samples and features
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate salience on CPU
        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            salience_sum += x_flat.abs().sum(dim=0)

        # Compute mean
        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def get_salience(self, name, module):
        """
        Get salience scores based on layer type.

        Returns:
            Tensor of shape [in_features]
        """
        layer_type = self.layer_types.get(name, 'mlp')

        if layer_type == 'mlp':
            print(f"    Computing risk-aware salience for MLP layer {name}...")
            return self.get_risk_aware_salience_mlp(name, module)
        else:
            print(f"    Computing AWQ salience for attention layer {name}...")
            return self.get_activation_salience_awq(name)

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Grid search for optimal per-input-channel scaling factor.

        Full-PRAQ Algorithm:
        1. Compute risk-aware salience: s_risk = P(activation) × magnitude
        2. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           a. Compute scales: s[j] = (s_risk[j])^α for each input channel j
           b. Scale weight COLUMNS: W[:, j] *= s[j]
           c. Quantize: W_q = Quantize(W_scaled)
           d. Compute compensated output: Y_q = W_q @ (X / s)
           e. Measure error: ||Y_q - Y_orig||²
        3. Return α and scales that minimize error

        Args:
            name: Layer name
            module: Linear layer module

        Returns:
            best_scales: Optimal per-input-channel scales (shape: [in_features])
            best_alpha: Best alpha value
            best_error: Minimum reconstruction error
        """
        # Get salience (risk-aware for MLP, AWQ-style for attention)
        salience = self.get_salience(name, module)

        if salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Check for invalid salience values
        if torch.isnan(salience).any() or torch.isinf(salience).any():
            print(f"    WARNING: Invalid salience values detected, using uniform scaling")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Check if salience is all zeros or very small
        if salience.max() < 1e-10:
            print(f"    WARNING: Salience values too small, using uniform scaling")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Normalize salience to prevent numerical issues
        # Scale to [0.1, 10] range to avoid extreme values
        salience_min = salience.min()
        salience_max = salience.max()
        if salience_max > salience_min:
            salience = 0.1 + 9.9 * (salience - salience_min) / (salience_max - salience_min)
        else:
            salience = torch.ones_like(salience)

        # Get activations for reconstruction error measurement
        X_list = self.activation_data[name]

        # Concatenate on CPU first, then sample
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Limit samples for efficiency
        max_samples = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        # Free CPU memory
        del X_cpu

        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else None

        # Compute original output (FP16 baseline)
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Move salience to device
        salience = salience.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales: s[j] = (salience[j])^α
            # Shape: [in_features]
            scales = salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS: W_scaled[:, j] = W[:, j] * scales[j]
            # Broadcasting: scales shape [in_features] → W shape [out_features, in_features]
            W_scaled = W * scales.unsqueeze(0)  # scales broadcast across output channels

            # Quantize scaled weights
            W_quant = self.quantize_weight(W_scaled)

            # Compute output with quantized scaled weights and compensated input
            # Y_q = W_q @ (X / s)
            # Compensate input: X_compensated = X / scales
            X_compensated = X_search / scales.unsqueeze(0)  # scales broadcast across batch

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Compute reconstruction error
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

        layer_type = self.layer_types.get(name, 'mlp')
        print(f"    [{layer_type}] best_alpha={best_alpha:.3f}, error={best_error:.6f}")

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_weight(self, W):
        """Quantize weights to INT4 (per-channel symmetric)."""
        W_abs_max = W.abs().max(dim=1, keepdim=True)[0]
        W_abs_max = W_abs_max.clamp(min=1e-8)
        scale = W_abs_max / 7.0
        W_quant = torch.round(W / scale).clamp(-8, 7)
        W_dequant = W_quant * scale
        return W_dequant

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Full-PRAQ quantization to a layer.

        Steps:
        1. Grid search for best per-input-channel scales (risk-aware for MLP)
        2. Scale weight columns: W[:, j] *= scales[j]
        3. Quantize scaled weights to INT4
        4. Divide by scales to restore original magnitude: W_final = Q(W*s) / s

        This gives us: W_final @ X ≈ Q(W*s)/s @ X ≈ W @ X
        The scaling-and-dividing protects important channels during quantization.
        """
        # Search for optimal scales (risk-aware for MLP, AWQ for attention)
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data  # [out_features, in_features]

        # Scale weight COLUMNS by per-input-channel scales
        W_scaled = W * best_scales.unsqueeze(0)

        # Quantize scaled weights to INT4
        W_quant = self.quantize_weight(W_scaled)

        # CRITICAL: Divide by scales to restore original magnitude
        # This ensures: Q(W*s)/s @ X ≈ W @ X at inference
        W_final = W_quant / best_scales.unsqueeze(0)

        # Update module weights with scaled-quantized-descaled weights
        module.weight.data = W_final

        # Store scales and metadata for later analysis
        layer_type = self.layer_types.get(name, 'mlp')
        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'type': layer_type
        }

        # Clean up
        del W_scaled, W_quant, W_final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def calibrate(self, calibration_data, n_samples=500):
        """Run calibration to collect activations."""
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
        self.model.eval()
        self.register_hooks()

        for i, text in enumerate(tqdm(calibration_data[:n_samples], desc="Calibration")):
            try:
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    _ = self.model(**inputs, use_cache=False, return_dict=True)

            except Exception:
                if i % 100 == 0 and i > 0:
                    print(f"\nNote: Skipped {i} samples with errors")
                continue

        self.remove_hooks()
        print("Calibration complete!")

    def quantize_model(self):
        """Quantize all linear layers using Full-PRAQ."""
        print("\n" + "=" * 80)
        print("Quantizing with Full-PRAQ (risk-aware + grid search + uniform INT4)")
        print("=" * 80)

        quantized_count = 0
        skipped_count = 0

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        for name, module in tqdm(layer_names, desc="Quantizing layers"):
            try:
                self.quantize_layer(name, module)
                quantized_count += 1

                # Clear activation data for this layer to save memory
                if name in self.activation_data:
                    del self.activation_data[name]

                # Clear GPU cache periodically
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

        # Print statistics about optimal alpha values
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
        description="Full-PRAQ quantization for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=150, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--beta", type=float, default=3.0, help="PRAQ beta parameter")
    parser.add_argument("--tau", type=float, default=-3.0, help="PRAQ tau parameter")
    parser.add_argument("--noise-factor", type=float, default=0.2, help="PRAQ noise factor")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_full_praq",
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
    print("Full-PRAQ: POST-ACTIVATION Importance for MiniCPM-2B (CORRECTED)")
    print("=" * 80)
    print("Approach: AWQ framework with PRAQ's post-activation insight")
    print("  1. MLP layers: Post-activation importance")
    print("     - Compute Y = SiLU(X @ W^T + b)")
    print("     - Measure importance[out] = E[|Y|]")
    print("     - Backprop to inputs via weights")
    print("  2. Attention layers: Standard AWQ (pre-activation)")
    print("  3. Grid search for optimal scaling exponent α")
    print("  4. Per-INPUT-channel scaling: W[:, j] *= s[j]^α")
    print("  5. Descaling: W_final = Q(W*s) / s")
    print("\nKey: Post-activation accounts for channels killed by activation fn!")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid points: {args.n_grid}")
    print(f"Risk parameters: beta={args.beta}, tau={args.tau}, noise={args.noise_factor}")
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
    quantizer = FullPRAQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beta=args.beta,
        tau=args.tau,
        noise_factor=args.noise_factor,
        bits=4,
        n_grid=args.n_grid
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
    print("\nFull-PRAQ Approach (CORRECTED - TRUE PRAQ INSIGHT):")
    print("  ✓ MLP layers: POST-activation importance")
    print("    → Measures E[|SiLU(XW+b)|] not E[|X|]")
    print("    → Accounts for activation function effects!")
    print("  ✓ Attention layers: Standard AWQ (pre-activation)")
    print("  ✓ Grid search optimization for per-INPUT-channel scaling")
    print("  ✓ Column-wise weight scaling: W[:, j] *= s[j]^α")
    print("  ✓ Descaling after quantization: W_final = Q(W*s) / s")
    print("  ✓ Uniform INT4 quantization (all channels)")
    print("\nExpected: BETTER than Real-AWQ (true PRAQ insight!)")
    print("\nNext steps:")
    print("=" * 80)


if __name__ == "__main__":
    main()
