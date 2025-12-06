"""
Real-PRAQ Implementation (AWQ Framework + PRAQ Risk-Awareness)

This combines the best of both approaches:
- AWQ's optimization: Per-channel scaling + grid search + uniform quantization
- PRAQ's intelligence: Risk-aware importance accounting for risky dead neurons

Key Innovation:
Instead of scaling by (E[|X|])^α like Real AWQ, we scale by:
    (P(activation) × magnitude)^α

where P(activation) accounts for:
- Pre-activation mean and variance
- Quantization noise impact
- Probability of neuron resurrection after quantization

This should combine AWQ's compression efficiency with PRAQ's risk-awareness.
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


class RealPRAQQuantizer:
    """
    Real-PRAQ: AWQ-style quantization with risk-aware importance.

    Combines:
    - Real AWQ framework: Grid search + per-channel scaling + uniform INT4
    - PRAQ risk-awareness: P(activation) × magnitude importance scoring
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

        # Detect layer types
        self.layer_types = self._detect_layer_types()

        # Print summary
        mlp_count = sum(1 for t in self.layer_types.values() if t == 'mlp')
        attn_count = sum(1 for t in self.layer_types.values() if t == 'attention')

        print(f"\n[Real-PRAQ Quantizer Initialized]")
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
        Compute PRAQ risk-aware salience for MLP layers.

        For each input feature, computes:
            salience[i] = P(activation) × magnitude

        where P(activation) accounts for quantization noise risk.

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Limit samples for efficiency
        max_samples = min(4096, X.shape[0])
        if X.shape[0] > max_samples:
            indices = torch.randperm(X.shape[0])[:max_samples]
            X = X[indices]

        n_samples = X.shape[0]
        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else torch.zeros(module.out_features, device=self.device)

        # Process in batches
        batch_size = 1024
        in_features = W.shape[1]

        # Accumulators for per-INPUT-feature statistics
        # We need to track how each input feature contributes across all output channels
        input_salience = torch.zeros(in_features)

        # For each input feature, compute aggregated risk-aware importance
        for input_idx in range(in_features):
            # Get weight column for this input feature across all output channels
            w_col = W[:, input_idx]  # [out_features]

            # Get activation values for this input feature
            x_col = X[:, input_idx]  # [n_samples]

            # Compute statistics for this input feature's contribution
            # across all output channels
            z_sum = 0.0
            z_sq_sum = 0.0
            z_abs_sum = 0.0

            for i in range(0, n_samples, batch_size):
                batch_x = x_col[i:i+batch_size].to(self.device)

                # Pre-activation contribution from this input across all outputs
                # z = x_i * W[:,i] (broadcasting)
                z = batch_x.unsqueeze(1) * w_col.unsqueeze(0)  # [batch, out_features]

                z_sum += z.sum(dim=0).sum().item()
                z_sq_sum += (z ** 2).sum(dim=0).sum().item()
                z_abs_sum += z.abs().sum(dim=0).sum().item()

                del batch_x, z

            # Compute aggregated statistics for this input feature
            total_samples = n_samples * module.out_features

            # Add numerical stability checks
            if total_samples == 0:
                input_salience[input_idx] = 0.0
                continue

            z_mean = z_sum / total_samples
            z_variance = (z_sq_sum / total_samples) - (z_mean ** 2)
            z_variance = max(0, z_variance)  # Ensure non-negative
            z_std = np.sqrt(z_variance) + 1e-8
            z_upper = z_mean + 3 * z_std

            # Estimate quantization noise impact
            x_mag = x_col.abs().mean().item()
            w_mag = w_col.abs().mean().item()
            estimated_noise = x_mag * w_mag * self.noise_factor

            # Risk-adjusted upper bound
            z_risk_upper = z_upper + estimated_noise

            # Probability of activation (clip to avoid numerical issues)
            logit = self.beta * (z_risk_upper - self.tau)
            logit = np.clip(logit, -20, 20)  # Prevent overflow in sigmoid
            prob_active = 1.0 / (1.0 + np.exp(-logit))

            # Magnitude (use absolute value sum normalized)
            magnitude = z_abs_sum / total_samples + z_std

            # Check for NaN/Inf and set to safe values
            if np.isnan(prob_active) or np.isinf(prob_active):
                prob_active = 0.0
            if np.isnan(magnitude) or np.isinf(magnitude):
                magnitude = 0.0

            # Risk-aware salience for this input feature
            input_salience[input_idx] = prob_active * magnitude

        # Debug: Print statistics
        valid_salience = input_salience[input_salience > 0]
        if len(valid_salience) > 0:
            print(f"      Salience stats: min={input_salience.min():.6f}, max={input_salience.max():.6f}, "
                  f"mean={input_salience.mean():.6f}, nonzero={len(valid_salience)}/{len(input_salience)}")

        return input_salience

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
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Activation salience
        salience = X.abs().mean(dim=0)
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
        Grid search for optimal per-channel scaling factor.

        Real-PRAQ Algorithm:
        1. Compute risk-aware salience: s_risk = P(activation) × magnitude
        2. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           - Compute scales: s = (s_risk)^α
           - Scale weights: W_scaled = W * s
           - Quantize: Q(W_scaled)
           - Measure reconstruction error
        3. Return α that minimizes error

        Args:
            name: Layer name
            module: Linear layer module

        Returns:
            best_scales: Optimal per-channel scales
        """
        # Get salience (risk-aware for MLP, AWQ-style for attention)
        salience = self.get_salience(name, module)

        if salience is None:
            return torch.ones(module.out_features)

        # Check for invalid salience values
        if torch.isnan(salience).any() or torch.isinf(salience).any():
            print(f"    WARNING: Invalid salience values detected, using uniform scaling")
            return torch.ones(module.out_features)

        # Check if salience is all zeros or very small
        if salience.max() < 1e-10:
            print(f"    WARNING: Salience values too small, using uniform scaling")
            return torch.ones(module.out_features)

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
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Limit samples for efficiency
        max_samples = min(2048, X.shape[0])
        if X.shape[0] > max_samples:
            indices = torch.randperm(X.shape[0])[:max_samples]
            X_search = X[indices]
        else:
            X_search = X

        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        # Compute original output
        X_search_gpu = X_search.to(self.device)
        if b is not None:
            original_out = torch.matmul(X_search_gpu, W.t()) + b
        else:
            original_out = torch.matmul(X_search_gpu, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[0], device=self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute scales: s = (salience)^α
            scales_per_input = salience.pow(alpha).clamp(min=1e-4)

            # Broadcast to weight shape
            scales = scales_per_input.unsqueeze(0).expand(W.shape[0], -1).to(self.device)

            # Scale weights
            W_scaled = W * scales

            # Quantize scaled weights
            W_quant = self.quantize_weight(W_scaled)

            # Compute output: Q(W*s) * (s^-1 * X)
            X_scaled = X_search_gpu / scales_per_input.to(self.device)

            if b is not None:
                quant_out = torch.matmul(X_scaled, W_quant.t()) + b
            else:
                quant_out = torch.matmul(X_scaled, W_quant.t())

            # Compute reconstruction error
            error = (original_out - quant_out).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales_per_input.clone()

        # Clean up
        del X_search_gpu, original_out
        if 'quant_out' in locals():
            del quant_out

        layer_type = self.layer_types.get(name, 'mlp')
        print(f"    [{layer_type}] best_alpha={best_alpha:.3f}, error={best_error:.6f}")

        return best_scales

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
        """Apply Real-PRAQ quantization to a layer."""
        # Search for optimal scales
        best_scales = self.search_best_scale(name, module)

        W = module.weight.data

        # Broadcast scales to weight shape
        scales = best_scales.unsqueeze(0).expand(W.shape[0], -1).to(W.device)

        # Scale weights
        W_scaled = W * scales

        # Quantize scaled weights
        W_quant = self.quantize_weight(W_scaled)

        # Update module weights
        module.weight.data = W_quant

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
        """Quantize all linear layers using Real-PRAQ."""
        print("\n" + "=" * 80)
        print("Quantizing with Real-PRAQ (risk-aware + grid search + uniform INT4)")
        print("=" * 80)

        quantized_count = 0
        skipped_count = 0

        for name, module in tqdm(list(self.model.named_modules()), desc="Quantizing"):
            if isinstance(module, nn.Linear):
                try:
                    self.quantize_layer(name, module)
                    quantized_count += 1

                    if quantized_count % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️  Error quantizing layer {name}: {e}")
                    skipped_count += 1
                    continue

        print(f"\n✅ Quantization complete!")
        print(f"   Total linear layers quantized: {quantized_count}")
        if skipped_count > 0:
            print(f"   ⚠️  Skipped {skipped_count} layers due to errors")

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
        description="Real-PRAQ quantization for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=500, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--beta", type=float, default=3.0, help="PRAQ beta parameter")
    parser.add_argument("--tau", type=float, default=-3.0, help="PRAQ tau parameter")
    parser.add_argument("--noise-factor", type=float, default=0.2, help="PRAQ noise factor")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_real_praq",
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
    print("Real-PRAQ: Risk-Aware AWQ for MiniCPM-2B")
    print("=" * 80)
    print("Innovation: Combines AWQ optimization with PRAQ risk-awareness")
    print("  1. MLP layers: Risk-aware salience = P(activation) × magnitude")
    print("  2. Attention layers: Standard AWQ salience = E[|X|]")
    print("  3. Grid search for optimal scaling exponent α")
    print("  4. Uniform INT4 quantization with optimal per-channel scaling")
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
    quantizer = RealPRAQQuantizer(
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
    print("\nReal-PRAQ Approach:")
    print("  - MLP layers: Risk-aware salience (accounts for risky dead neurons)")
    print("  - Attention layers: AWQ-style salience (activation magnitude)")
    print("  - Grid search optimization for per-channel scaling")
    print("  - Uniform INT4 quantization (all channels)")
    print("  - Expected: Better than Real-AWQ due to risk-awareness")
    print("=" * 80)


if __name__ == "__main__":
    main()
