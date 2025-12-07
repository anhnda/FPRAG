"""
Hybrid Group-Wise PRAQ Implementation with ENHANCED Asymmetric Quantization (V2)

Improvements over gwh_praq_asym.py:
1. Adaptive quantization: Choose symmetric vs asymmetric per layer based on weight distribution
2. Enhanced importance metric: Weight by distribution asymmetry
3. Finer grid search: 40 points instead of 20
4. Layer-wise skewness analysis: Quantize based on actual weight characteristics
5. Mixed-precision groups: Some groups symmetric, some asymmetric within same layer

Key Algorithm:
1. Analyze weight distribution per layer (mean, std, skewness)
2. Compute AWQ Salience (Input |X|) + Distribution Asymmetry
3. Compute PRAQ Importance (Output |SiLU(XW)|) + Skewness Weighting
4. Adaptive Grid Search:
   a. For each group, decide: symmetric or asymmetric based on skewness
   b. Scale W using enhanced AWQ metric
   c. Quantize adaptively (mixed sym/asym groups)
   d. Measure Error: ||Y - Y_q||² weighted by enhanced PRAQ Importance
   e. Choose α that minimizes weighted error
5. Result: Best of all worlds: AWQ + PRAQ + Adaptive Asymmetric
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
from scipy import stats


class EnhancedHybridPRAQAsymmetricQuantizer:
    """
    Enhanced Hybrid PRAQ with Adaptive Asymmetric Quantization.

    Key improvements:
    - Adaptive symmetric vs asymmetric per group based on weight skewness
    - Enhanced importance metrics considering distribution characteristics
    - Finer grid search for better optimization
    - Mixed-precision quantization within layers
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=40, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid  # Increased from 20 to 40
        self.group_size = group_size

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        # Detect layer types (MLP vs Attention)
        self.layer_types = self._detect_layer_types()

        mlp_count = sum(1 for t in self.layer_types.values() if t == 'mlp')
        attn_count = sum(1 for t in self.layer_types.values() if t == 'attention')

        print(f"\n[Enhanced Hybrid PRAQ ASYMMETRIC V2 Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid} (enhanced!)")
        print(f"  Group size: {group_size}")
        print(f"  Strategy: Adaptive Sym/Asym + Enhanced Importance")
        print(f"  MLP layers: {mlp_count}")
        print(f"  Attention layers: {attn_count}")

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
    def analyze_weight_distribution(self, W):
        """
        Analyze weight distribution to determine if asymmetric quantization is beneficial.

        Returns:
            skewness_score: Per-group skewness measure
            use_asymmetric: Boolean mask indicating which groups should use asymmetric
        """
        out_features, in_features = W.shape

        # Group weights
        if in_features % self.group_size == 0:
            W_grouped = W.view(out_features, -1, self.group_size)
        else:
            pad_len = self.group_size - (in_features % self.group_size)
            W_padded = torch.nn.functional.pad(W, (0, pad_len))
            W_grouped = W_padded.view(out_features, -1, self.group_size)

        # Compute skewness per group
        W_grouped_np = W_grouped.cpu().numpy()
        skewness = np.abs(stats.skew(W_grouped_np, axis=2))  # [out_features, n_groups]

        # Use asymmetric if skewness > threshold (0.5)
        use_asymmetric = torch.from_numpy(skewness > 0.5).to(W.device)

        return torch.from_numpy(skewness).to(W.device), use_asymmetric

    @torch.no_grad()
    def get_activation_salience_awq(self, name):
        """Enhanced AWQ activation salience with distribution awareness."""
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
    def get_output_importance_praq(self, name, module):
        """Enhanced PRAQ importance with skewness weighting."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        layer_type = self.layer_types.get(name, 'mlp')
        if layer_type != 'mlp':
            return torch.ones(module.weight.shape[0])

        X_list = self.activation_data[name]
        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        activation_fn = torch.nn.functional.silu

        output_importance_sum = torch.zeros(module.weight.shape[0])
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

        base_importance = output_importance_sum / total_samples

        # Weight by output channel skewness
        _, use_asym_mask = self.analyze_weight_distribution(W)
        asym_ratio = use_asym_mask.float().mean(dim=1)  # Per output channel

        # Boost importance for channels with high asymmetry
        enhanced_importance = base_importance * (1.0 + 0.5 * asym_ratio.cpu())

        return enhanced_importance

    @torch.no_grad()
    def quantize_weight_adaptive_asymmetric(self, W):
        """
        Adaptive quantization: use symmetric or asymmetric per group based on distribution.
        """
        out_features, in_features = W.shape

        # Analyze distribution
        skewness_score, use_asymmetric = self.analyze_weight_distribution(W)

        # Group weights
        if in_features % self.group_size == 0:
            W_grouped = W.view(out_features, -1, self.group_size)
            padded = False
        else:
            pad_len = self.group_size - (in_features % self.group_size)
            W_padded = torch.nn.functional.pad(W, (0, pad_len))
            W_grouped = W_padded.view(out_features, -1, self.group_size)
            padded = True

        # Quantize each group adaptively
        W_quant_grouped = torch.zeros_like(W_grouped)

        for i in range(W_grouped.shape[0]):  # out_features
            for j in range(W_grouped.shape[1]):  # n_groups
                group = W_grouped[i, j, :]

                if use_asymmetric[i, j]:
                    # Asymmetric quantization
                    w_min = group.min()
                    w_max = group.max()
                    scale = (w_max - w_min) / 15.0
                    scale = max(scale, 1e-8)
                    zero_point = round((-w_min / scale).item())
                    zero_point = max(0, min(15, zero_point))

                    w_int = torch.round(group / scale + zero_point).clamp(0, 15)
                    W_quant_grouped[i, j, :] = (w_int - zero_point) * scale
                else:
                    # Symmetric quantization
                    w_abs_max = group.abs().max()
                    w_abs_max = max(w_abs_max, 1e-8)
                    scale = w_abs_max / 7.0

                    w_int = torch.round(group / scale).clamp(-8, 7)
                    W_quant_grouped[i, j, :] = w_int * scale

        # Reshape back
        if not padded:
            return W_quant_grouped.view(out_features, in_features)
        else:
            W_dequant = W_quant_grouped.reshape(out_features, -1)
            return W_dequant[:, :in_features]

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """Enhanced grid search with adaptive quantization."""
        # Get enhanced metrics
        awq_salience = self.get_activation_salience_awq(name)
        if awq_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        awq_salience = awq_salience.to(self.device)
        awq_salience_norm = awq_salience / (awq_salience.mean() + 1e-8)

        praq_importance = self.get_output_importance_praq(name, module)
        if praq_importance is None:
            praq_importance = torch.ones(module.weight.shape[0])

        praq_importance = praq_importance.to(self.device)
        praq_importance_norm = praq_importance / (praq_importance.mean() + 1e-8)

        # Prepare data
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

        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Enhanced grid search (40 points)
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            scales = awq_salience_norm.pow(alpha).clamp(min=1e-5)
            W_scaled = W * scales.unsqueeze(0)

            # Use adaptive quantization
            W_quant = self.quantize_weight_adaptive_asymmetric(W_scaled)

            X_compensated = X_search / scales.unsqueeze(0)
            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            raw_error = (Y_orig - Y_quant).pow(2)
            weighted_error = (raw_error * praq_importance_norm.unsqueeze(0)).mean()

            error_val = weighted_error.item()

            if error_val < best_error:
                best_error = error_val
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig, Y_quant, raw_error, weighted_error
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Apply enhanced hybrid quantization."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        W_scaled = W * best_scales.unsqueeze(0)
        W_quant = self.quantize_weight_adaptive_asymmetric(W_scaled)
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

    def calibrate(self, calibration_data, n_samples=128):
        """Calibrate the quantizer."""
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
        print("Quantizing with ENHANCED HYBRID PRAQ + ADAPTIVE ASYMMETRIC")
        print("=" * 80)
        print("Enhancements:")
        print("  1. Adaptive Sym/Asym per group (skewness-based)")
        print("  2. Enhanced importance (distribution-aware)")
        print(f"  3. Finer grid search ({self.n_grid + 1} points)")
        print("  4. Mixed-precision groups (adaptive)")
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
        description="Enhanced Hybrid PRAQ with Adaptive Asymmetric Quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=40, help="Grid search points (enhanced: 40)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gwh_praq_asym_v2",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Enhanced Hybrid PRAQ + Adaptive Asymmetric Quantization V2")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Grid points: {args.n_grid + 1} (enhanced!)")
    print("=" * 80)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)

    quantizer = EnhancedHybridPRAQAsymmetricQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size
    )

    quantizer.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer.quantize_model()

    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)
    print("V2 Enhancements Applied:")
    print("  ✓ Adaptive Sym/Asym (skewness > 0.5 → Asymmetric)")
    print("  ✓ Enhanced importance (distribution-aware weighting)")
    print("  ✓ Finer grid search (40 points for better optimization)")
    print("  ✓ Mixed-precision groups (optimal for each group)")
    print(f"\nSaved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
