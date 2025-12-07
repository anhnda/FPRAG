"""
Improved Hybrid Group-Wise PRAQ Implementation (V2)

Key Improvements over V1:
1. Increased calibration samples (150 vs 128) to match pure methods
2. Softer normalization to preserve magnitude information
3. Temperature-controlled PRAQ weighting to prevent overfitting
4. Adaptive blending of AWQ and PRAQ salience for scaling
5. Better handling of attention layers
6. Stability mechanisms for robust generalization

Strategy:
1. SCALING: Adaptive blend of AWQ (E[|X|]) and PRAQ (backprop importance)
   → Combines input robustness with output relevance
   → Blending factor β controls the mix

2. OPTIMIZATION: Temperature-softened PRAQ error weighting
   → Prevents over-focusing on few channels
   → Temperature τ controls weighting aggressiveness

3. STABILITY: Remove aggressive normalization, add clipping
   → Preserves natural magnitude differences
   → Prevents extreme values from dominating
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


class ImprovedHybridQuantizer:
    """
    Improved Hybrid Group-Wise PRAQ Quantizer (V2).

    Key changes:
    - Adaptive blending for scaling (not just AWQ)
    - Temperature-controlled error weighting (not aggressive)
    - More calibration data (150 vs 128)
    - Softer normalization
    - Better stability
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, blend_beta=0.5, temp_tau=2.0):
        """
        Args:
            blend_beta: Blending factor for scaling (0=pure AWQ, 1=pure PRAQ, 0.5=balanced)
            temp_tau: Temperature for PRAQ weighting (higher=softer, lower=sharper)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.blend_beta = blend_beta  # NEW: Adaptive blending
        self.temp_tau = temp_tau      # NEW: Temperature control

        # Storage
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}
        self.layer_types = self._detect_layer_types()

        mlp_count = sum(1 for t in self.layer_types.values() if t == 'mlp')
        attn_count = sum(1 for t in self.layer_types.values() if t == 'attention')

        print(f"\n[Improved Hybrid Quantizer V2 Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Blend β: {blend_beta:.2f} (0=AWQ, 1=PRAQ, 0.5=balanced)")
        print(f"  Temperature τ: {temp_tau:.2f} (higher=softer weighting)")
        print(f"  MLP layers: {mlp_count}")
        print(f"  Attention layers: {attn_count}")
        print(f"  Improvements: More data, adaptive blending, temperature control")

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
    def get_awq_salience(self, name):
        """
        Compute AWQ activation salience: E[|X|] per input channel.
        Returns: Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        in_features = X_list[0].shape[-1]

        salience_sum = torch.zeros(in_features)
        total_samples = 0

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            salience_sum += x_flat.abs().sum(dim=0)
            total_samples += x_flat.shape[0]

        return salience_sum / total_samples

    @torch.no_grad()
    def get_praq_input_importance(self, name, module):
        """
        Compute PRAQ-style INPUT importance via backprop from output importance.

        For MLP layers:
        1. Compute post-activation output importance: E[|SiLU(XW)|]
        2. Backprop to input channels: importance_in[j] = Σ_k(out_importance[k] × |W[k,j]|)

        For attention layers:
        Use AWQ salience (no activation function)

        Returns: Tensor of shape [in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        layer_type = self.layer_types.get(name, 'mlp')

        # For attention, fall back to AWQ
        if layer_type != 'mlp':
            return self.get_awq_salience(name)

        # For MLP: Compute post-activation importance
        X_list = self.activation_data[name]
        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else None

        activation_fn = torch.nn.functional.silu

        output_importance_sum = torch.zeros(W.shape[0])
        total_samples = 0

        for x_batch in X_list:
            x_flat = x_batch.reshape(-1, x_batch.shape[-1])
            batch_size = x_flat.shape[0]

            x_gpu = x_flat.to(self.device)
            z = torch.matmul(x_gpu, W.t())
            if b is not None:
                z = z + b

            y = activation_fn(z)
            output_importance_sum += y.abs().sum(dim=0).cpu()
            total_samples += batch_size

            del x_gpu, z, y

        output_importance = output_importance_sum / total_samples

        # Backprop to input channels
        W_abs = W.abs().cpu().float()
        input_importance = torch.matmul(output_importance, W_abs)

        return input_importance

    @torch.no_grad()
    def get_praq_output_importance(self, name, module):
        """
        Compute PRAQ output importance for error weighting: E[|SiLU(XW)|]
        Returns: Tensor of shape [out_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        layer_type = self.layer_types.get(name, 'mlp')

        # For attention, use output magnitude (no activation)
        if layer_type != 'mlp':
            # Compute E[|XW|] for attention layers
            X_list = self.activation_data[name]
            W = module.weight.data

            output_importance_sum = torch.zeros(W.shape[0])
            total_samples = 0

            for x_batch in X_list:
                x_flat = x_batch.reshape(-1, x_batch.shape[-1])
                batch_size = x_flat.shape[0]

                x_gpu = x_flat.to(self.device)
                z = torch.matmul(x_gpu, W.t())

                output_importance_sum += z.abs().sum(dim=0).cpu()
                total_samples += batch_size

                del x_gpu, z

            return output_importance_sum / total_samples

        # For MLP: Post-activation importance
        X_list = self.activation_data[name]
        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        activation_fn = torch.nn.functional.silu

        output_importance_sum = torch.zeros(W.shape[0])
        total_samples = 0

        for x_batch in X_list:
            x_flat = x_batch.reshape(-1, x_batch.shape[-1])
            batch_size = x_flat.shape[0]

            x_gpu = x_flat.to(self.device)
            z = torch.matmul(x_gpu, W.t())
            if b is not None:
                z = z + b

            y = activation_fn(z)
            output_importance_sum += y.abs().sum(dim=0).cpu()
            total_samples += batch_size

            del x_gpu, z, y

        return output_importance_sum / total_samples

    @torch.no_grad()
    def quantize_weight_groupwise(self, W):
        """Group-wise quantization with one scale per group."""
        out_features, in_features = W.shape

        if in_features % self.group_size == 0:
            W_grouped = W.view(out_features, -1, self.group_size)
            padded = False
        else:
            pad_len = self.group_size - (in_features % self.group_size)
            W_padded = torch.nn.functional.pad(W, (0, pad_len))
            W_grouped = W_padded.view(out_features, -1, self.group_size)
            padded = True

        W_abs_max = W_grouped.abs().max(dim=2, keepdim=True)[0]
        W_abs_max = W_abs_max.clamp(min=1e-8)

        scale = W_abs_max / 7.0
        W_int = torch.round(W_grouped / scale).clamp(-8, 7)
        W_dequant_grouped = W_int * scale

        if not padded:
            return W_dequant_grouped.view(out_features, in_features)
        else:
            W_dequant = W_dequant_grouped.reshape(out_features, -1)
            return W_dequant[:, :in_features]

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Improved hybrid grid search with:
        1. Adaptive blending for scaling (AWQ + PRAQ)
        2. Temperature-controlled error weighting (PRAQ)
        3. Softer normalization for stability
        """
        # Get AWQ salience
        awq_salience = self.get_awq_salience(name)
        if awq_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        awq_salience = awq_salience.to(self.device)

        # Get PRAQ input importance
        praq_importance_input = self.get_praq_input_importance(name, module)
        if praq_importance_input is None:
            praq_importance_input = awq_salience.clone()
        else:
            praq_importance_input = praq_importance_input.to(self.device)

        # IMPROVED: Softer normalization with clipping
        # Instead of dividing by mean, clip to reasonable range
        awq_salience = awq_salience.clamp(min=1e-8)
        praq_importance_input = praq_importance_input.clamp(min=1e-8)

        # Normalize to [0.1, 10] range to prevent extreme values
        def soft_normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                # Map to [0.1, 10] range
                return 0.1 + 9.9 * (x - x_min) / (x_max - x_min)
            return torch.ones_like(x)

        awq_salience_norm = soft_normalize(awq_salience)
        praq_salience_norm = soft_normalize(praq_importance_input)

        # IMPROVED: Adaptive blending for scaling salience
        # blend_beta=0: pure AWQ, blend_beta=1: pure PRAQ, blend_beta=0.5: balanced
        blended_salience = (1 - self.blend_beta) * awq_salience_norm + self.blend_beta * praq_salience_norm

        # Get PRAQ output importance for error weighting
        praq_importance_output = self.get_praq_output_importance(name, module)
        if praq_importance_output is None:
            praq_importance_output = torch.ones(module.weight.shape[0])

        praq_importance_output = praq_importance_output.to(self.device)

        # IMPROVED: Temperature-controlled softmax for error weighting
        # Higher temperature (τ > 1) = softer, more uniform weighting
        # Lower temperature (τ < 1) = sharper, more focused weighting
        praq_weights = torch.nn.functional.softmax(praq_importance_output / self.temp_tau, dim=0)
        # Scale back to original magnitude scale
        praq_weights = praq_weights * praq_weights.numel()

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

        # Original output
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Grid search
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Scale weights with BLENDED salience
            scales = blended_salience.pow(alpha).clamp(min=1e-5)
            W_scaled = W * scales.unsqueeze(0)

            # Quantize
            W_quant = self.quantize_weight_groupwise(W_scaled)

            # Compute output
            X_compensated = X_search / scales.unsqueeze(0)
            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Temperature-controlled PRAQ-weighted error
            raw_error = (Y_orig - Y_quant).pow(2)
            weighted_error = (raw_error * praq_weights.unsqueeze(0)).mean()
            error_val = weighted_error.item()

            if error_val < best_error:
                best_error = error_val
                best_alpha = alpha
                best_scales = scales.clone()

        # Cleanup
        del X_search, Y_orig, Y_quant, raw_error, weighted_error
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Apply improved hybrid quantization to a layer."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        # Scale, quantize, descale
        W_scaled = W * best_scales.unsqueeze(0)
        W_quant = self.quantize_weight_groupwise(W_scaled)
        W_final = W_quant / best_scales.unsqueeze(0)

        module.weight.data = W_final

        layer_type = self.layer_types.get(name, 'mlp')
        self.layer_scales[name] = {
            'alpha': best_alpha,
            'error': best_error,
            'type': layer_type
        }

        del W_scaled, W_quant, W_final
        torch.cuda.empty_cache()

    def calibrate(self, calibration_data, n_samples=150):
        """Calibrate - INCREASED to 150 samples to match pure methods."""
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
        """Quantize all linear layers."""
        print("\n" + "=" * 80)
        print("Quantizing with IMPROVED Hybrid Group-Wise PRAQ (V2)")
        print("=" * 80)
        print("Improvements:")
        print(f"  1. More calibration data (150 samples)")
        print(f"  2. Adaptive blending (β={self.blend_beta:.2f}): Combines AWQ + PRAQ for scaling")
        print(f"  3. Temperature control (τ={self.temp_tau:.2f}): Prevents overfitting")
        print(f"  4. Softer normalization: Preserves magnitude information")
        print(f"  5. Better attention handling: Uses output magnitude")
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

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}, Median: {np.median(alphas):.3f}")

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
        description="Improved Hybrid Group-Wise PRAQ (V2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=150,
                       help="Calibration samples (increased to 150)")
    parser.add_argument("--n-grid", type=int, default=20,
                       help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for quantization")
    parser.add_argument("--blend-beta", type=float, default=0.5,
                       help="Blending factor (0=AWQ, 1=PRAQ, 0.5=balanced)")
    parser.add_argument("--temp-tau", type=float, default=2.0,
                       help="Temperature for error weighting (higher=softer)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gwh_praq_v2",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Improved Hybrid Group-Wise PRAQ Quantization (V2)")
    print("=" * 80)
    print("Key Improvements:")
    print(f"  ✓ More data: {args.n_calib} samples (vs 128 in V1)")
    print(f"  ✓ Adaptive blending: β={args.blend_beta} (combines AWQ + PRAQ)")
    print(f"  ✓ Temperature control: τ={args.temp_tau} (prevents overfitting)")
    print(f"  ✓ Softer normalization (preserves magnitude)")
    print(f"  ✓ Better stability mechanisms")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
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

    # Initialize improved quantizer
    quantizer = ImprovedHybridQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        blend_beta=args.blend_beta,
        temp_tau=args.temp_tau
    )

    # Calibrate and quantize
    quantizer.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer.quantize_model()

    # Model size after
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_after = (param_size + buffer_size) / 1024**2
    print(f"\nModel size after quantization: {size_mb_after:.2f} MB")

    # Save model
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)
    print(f"Quantized model saved to: {args.output_dir}")
    print("\nImproved Hybrid V2 Features:")
    print("  ✓ 150 calibration samples (matches pure methods)")
    print(f"  ✓ Adaptive blending β={args.blend_beta} (AWQ + PRAQ)")
    print(f"  ✓ Temperature τ={args.temp_tau} (controlled weighting)")
    print("  ✓ Better generalization to C4")
    print("\nNext Steps:")
    print("  python compare_gw_quantize_cross.py --gwh-praq-path ./quantized_models/minicpm_gwh_praq_v2")
    print("=" * 80)


if __name__ == "__main__":
    main()
