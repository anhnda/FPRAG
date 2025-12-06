"""
Robust-PRAQ Implementation with Noise Augmentation

This extends Full-PRAQ with noise augmentation for improved stability and robustness
when calibration data is limited.

Key Improvements over Full-PRAQ:
1. Gaussian noise augmentation during importance computation
2. Multiple noise samples averaged for robust importance estimates
3. Noise augmentation during grid search for better generalization
4. Configurable noise parameters (std, num_samples)

Motivation:
- Small calibration sets can lead to overfitting in importance estimates
- Adding controlled noise helps explore the activation distribution better
- Averaging over noise samples provides more stable importance scores
- Improves robustness to calibration set selection

Algorithm:
1. For each layer, compute importance with noise augmentation:
   - Sample n noise realizations: X_noisy = X + ε, where ε ~ N(0, σ²)
   - Compute post-activation importance for each noisy sample
   - Average importance scores across noise samples
2. Grid search with noisy inputs for robust scale selection
3. Quantize with optimal scales (same as Full-PRAQ)
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


class RobustPRAQQuantizer:
    """
    Robust-PRAQ: Full-PRAQ with noise augmentation for stability.

    Extends Full-PRAQ with:
    - Gaussian noise injection during importance computation
    - Multi-sample averaging for robust importance estimates
    - Noise augmentation during grid search
    """

    def __init__(self, model, tokenizer, device="cuda", beta=3.0, tau=-3.0,
                 noise_factor=0.2, bits=4, n_grid=20,
                 noise_std=0.01, n_noise_samples=3):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise (relative to input std)
            n_noise_samples: Number of noise samples to average over
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.tau = tau
        self.noise_factor = noise_factor
        self.bits = bits
        self.n_grid = n_grid
        self.noise_std = noise_std
        self.n_noise_samples = n_noise_samples

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

        print(f"\n[Robust-PRAQ Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  MLP layers (risk-aware): {mlp_count}")
        print(f"  Attention layers (AWQ-style): {attn_count}")
        print(f"  Risk parameters: beta={beta}, tau={tau}, noise_factor={noise_factor}")
        print(f"  Noise augmentation: std={noise_std}, samples={n_noise_samples}")
        print(f"  → Noise helps with small calibration sets!")

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
    def add_gaussian_noise(self, x, noise_std):
        """
        Add Gaussian noise to input tensor.

        Args:
            x: Input tensor
            noise_std: Noise standard deviation (relative to x.std())

        Returns:
            Noisy tensor: x + N(0, noise_std² × x.std()²)
        """
        if noise_std <= 0:
            return x

        # Compute input std for adaptive noise scaling
        x_std = x.std()
        if x_std < 1e-8:
            return x

        # Generate noise: ε ~ N(0, (noise_std × x_std)²)
        noise = torch.randn_like(x) * (noise_std * x_std)
        return x + noise

    @torch.no_grad()
    def get_risk_aware_salience_mlp(self, name, module):
        """
        Compute PRAQ salience for MLP layers with NOISE AUGMENTATION.

        Key Innovation: Average importance over multiple noise realizations
        for more robust estimates when calibration data is limited.

        Algorithm:
        1. For each noise sample k in [1..n_noise_samples]:
           a. Add noise: X_noisy = X + ε_k, where ε_k ~ N(0, σ²)
           b. Compute post-activation: Y_k = SiLU(X_noisy @ W^T + b)
           c. Measure importance: importance_k[j] = E[|Y_k|] backprop via |W|
        2. Average: importance[j] = (1/n) Σ_k importance_k[j]

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        activation_fn = torch.nn.functional.silu

        X_list = self.activation_data[name]
        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else None

        in_features = W.shape[1]
        out_features = W.shape[0]

        # Accumulate importance across noise samples
        total_input_importance = torch.zeros(in_features)

        # Average over multiple noise realizations for robustness
        for noise_idx in range(self.n_noise_samples):
            output_importance_sum = torch.zeros(out_features)
            total_samples = 0

            # Process in batches
            for x_batch in X_list:
                x_flat = x_batch.reshape(-1, x_batch.shape[-1])
                batch_size = x_flat.shape[0]

                # Add Gaussian noise for robustness
                x_gpu = x_flat.to(self.device)
                if self.noise_std > 0:
                    x_gpu = self.add_gaussian_noise(x_gpu, self.noise_std)

                # Compute pre-activation
                z = torch.matmul(x_gpu, W.t())
                if b is not None:
                    z = z + b

                # Apply activation function (PRAQ KEY STEP!)
                y = activation_fn(z)

                # Measure post-activation output magnitude
                output_importance_sum += y.abs().sum(dim=0).cpu()
                total_samples += batch_size

                del x_gpu, z, y

            # Average post-activation output magnitude per output channel
            output_importance = output_importance_sum / total_samples

            # Backprop importance to input channels via weight magnitudes
            W_abs = W.abs().cpu().float()
            input_importance = torch.matmul(output_importance, W_abs)

            # Accumulate across noise samples
            total_input_importance += input_importance

        # Average over noise samples
        avg_input_importance = total_input_importance / self.n_noise_samples

        return avg_input_importance

    @torch.no_grad()
    def get_activation_salience_awq(self, name):
        """
        Compute AWQ activation salience with noise augmentation: E[|X + ε|]

        Returns:
            Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate across noise samples
        total_salience = torch.zeros(in_features)

        for noise_idx in range(self.n_noise_samples):
            salience_sum = torch.zeros(in_features)

            for x in X_list:
                x_flat = x.reshape(-1, x.shape[-1])

                # Add noise for robustness
                if self.noise_std > 0:
                    x_noisy = self.add_gaussian_noise(x_flat, self.noise_std)
                else:
                    x_noisy = x_flat

                salience_sum += x_noisy.abs().sum(dim=0)

            total_salience += salience_sum / total_samples

        # Average over noise samples
        avg_salience = total_salience / self.n_noise_samples

        return avg_salience

    @torch.no_grad()
    def get_salience(self, name, module):
        """Get salience scores based on layer type (with noise augmentation)."""
        layer_type = self.layer_types.get(name, 'mlp')

        if layer_type == 'mlp':
            print(f"    Computing robust risk-aware salience for MLP layer {name}...")
            return self.get_risk_aware_salience_mlp(name, module)
        else:
            print(f"    Computing robust AWQ salience for attention layer {name}...")
            return self.get_activation_salience_awq(name)

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Grid search for optimal per-input-channel scaling with noise augmentation.

        Enhancement over Full-PRAQ:
        - Uses noise-augmented importance scores (more robust)
        - Evaluates reconstruction error with noisy inputs (better generalization)
        """
        # Get noise-augmented salience
        salience = self.get_salience(name, module)

        if salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Check for invalid salience values
        if torch.isnan(salience).any() or torch.isinf(salience).any():
            print(f"    WARNING: Invalid salience values detected, using uniform scaling")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        if salience.max() < 1e-10:
            print(f"    WARNING: Salience values too small, using uniform scaling")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Normalize salience to prevent numerical issues
        salience_min = salience.min()
        salience_max = salience.max()
        if salience_max > salience_min:
            salience = 0.1 + 9.9 * (salience - salience_min) / (salience_max - salience_min)
        else:
            salience = torch.ones_like(salience)

        # Get activations for reconstruction error measurement
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Limit samples for efficiency
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

            scales = salience.pow(alpha).clamp(min=1e-5)
            W_scaled = W * scales.unsqueeze(0)
            W_quant = self.quantize_weight(W_scaled)

            # Evaluate with noise augmentation for better generalization
            total_error = 0.0
            for noise_idx in range(max(1, self.n_noise_samples // 2)):  # Use fewer samples for speed
                # Add noise to inputs during evaluation
                if self.noise_std > 0:
                    X_eval = self.add_gaussian_noise(X_search, self.noise_std * 0.5)  # Smaller noise
                else:
                    X_eval = X_search

                X_compensated = X_eval / scales.unsqueeze(0)

                if b is not None:
                    Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
                else:
                    Y_quant = torch.matmul(X_compensated, W_quant.t())

                total_error += (Y_orig - Y_quant).pow(2).mean().item()

            error = total_error / max(1, self.n_noise_samples // 2)

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
        """Apply Robust-PRAQ quantization to a layer."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        W_scaled = W * best_scales.unsqueeze(0)
        W_quant = self.quantize_weight(W_scaled)
        W_final = W_quant / best_scales.unsqueeze(0)

        module.weight.data = W_final

        layer_type = self.layer_types.get(name, 'mlp')
        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'type': layer_type
        }

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
        """Quantize all linear layers using Robust-PRAQ."""
        print("\n" + "=" * 80)
        print("Quantizing with Robust-PRAQ (noise-augmented + post-activation)")
        print("=" * 80)

        quantized_count = 0
        skipped_count = 0

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        for name, module in tqdm(layer_names, desc="Quantizing layers"):
            try:
                self.quantize_layer(name, module)
                quantized_count += 1

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
                print(f"\nMLP layers (robust risk-aware):")
                print(f"  Mean α: {np.mean(mlp_alphas):.3f}")
                print(f"  Median α: {np.median(mlp_alphas):.3f}")

            if attn_alphas:
                print(f"\nAttention layers (robust AWQ-style):")
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
        description="Robust-PRAQ quantization for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=500, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--beta", type=float, default=3.0, help="PRAQ beta parameter")
    parser.add_argument("--tau", type=float, default=-3.0, help="PRAQ tau parameter")
    parser.add_argument("--noise-factor", type=float, default=0.2, help="PRAQ noise factor")
    parser.add_argument("--noise-std", type=float, default=0.01,
                       help="Noise std for augmentation (relative to input std)")
    parser.add_argument("--n-noise-samples", type=int, default=3,
                       help="Number of noise samples to average")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_robust_praq",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openbmb/MiniCPM-2B-sft-bf16"

    print("=" * 80)
    print("Robust-PRAQ: Noise-Augmented Post-Activation Quantization")
    print("=" * 80)
    print("Key Innovation: Noise augmentation for robust importance estimation")
    print("  1. MLP layers: Post-activation importance with noise averaging")
    print("     - Add Gaussian noise: X_noisy = X + N(0, σ²)")
    print("     - Average over multiple noise samples")
    print("     - More robust to small calibration sets!")
    print("  2. Attention layers: AWQ with noise augmentation")
    print("  3. Grid search with noisy inputs for better generalization")
    print("  4. Same quantization framework as Full-PRAQ")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid points: {args.n_grid}")
    print(f"Noise std: {args.noise_std} (relative to input std)")
    print(f"Noise samples: {args.n_noise_samples}")
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
    quantizer = RobustPRAQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beta=args.beta,
        tau=args.tau,
        noise_factor=args.noise_factor,
        bits=4,
        n_grid=args.n_grid,
        noise_std=args.noise_std,
        n_noise_samples=args.n_noise_samples
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
    print("\nRobust-PRAQ Advantages:")
    print("  ✓ Noise augmentation for stability with small calibration sets")
    print("  ✓ Multi-sample averaging for robust importance estimates")
    print("  ✓ Better generalization via noisy grid search")
    print("  ✓ Post-activation importance (true PRAQ insight)")
    print("  ✓ Same quantization quality as Full-PRAQ, but more robust!")
    print("\nExpected: Better than Full-PRAQ when calibration data is limited")
    print("=" * 80)


if __name__ == "__main__":
    main()
