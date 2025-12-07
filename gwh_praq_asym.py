"""
Hybrid Group-Wise PRAQ Implementation with ASYMMETRIC Quantization

This implements a HYBRID strategy with ASYMMETRIC quantization.

Key Difference from gwh_praq.py:
- gwh_praq.py: Symmetric quantization, INT4 range [-8, 7], zero_point=0
- gwh_praq_asym.py: Asymmetric quantization, INT4 range [0, 15], zero_point computed

STRATEGY:
1. SCALING (AWQ): Use input magnitude E[|X|] to scale weights
   → Keeps weight groups smooth and prevents dead channels from disrupting neighbors
   → Robust for group-wise quantization

2. OPTIMIZATION (PRAQ): Use output importance E[|SiLU(XW)|] to weight reconstruction error
   → Minimizes error specifically on channels that survive activation
   → Smart objective function that ignores dead neurons

3. ASYMMETRIC QUANTIZATION: Use full INT4 range [0, 15]
   → Better for asymmetric weight distributions
   → Computes scale and zero_point per group

Key Algorithm:
1. Compute AWQ Salience (Input |X|) → Used for Weight Scaling
2. Compute PRAQ Importance (Output |SiLU(XW)|) → Used for Error Weighting
3. Grid Search for α:
   a. Scale W using AWQ metric: W' = W × (s_awq)^α
   b. Quantize W' using Group-Wise ASYMMETRIC Quantization [0, 15]
   c. Measure Error: ||Y - Y_q||² weighted by PRAQ Importance
   d. Choose α that minimizes PRAQ-weighted error
4. Result: Robust grouping (AWQ) + Intelligent optimization (PRAQ) + Full range (Asymmetric)
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


class HybridGroupWisePRAQAsymmetricQuantizer:
    """
    Hybrid Group-Wise PRAQ Quantizer with Asymmetric Quantization.

    Combines:
    - AWQ's robustness: Smooth scaling for group-wise quantization
    - PRAQ's intelligence: Error weighting based on post-activation importance
    - Asymmetric quantization: Full INT4 range [0, 15] for better representation
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

        # Detect layer types (MLP vs Attention)
        self.layer_types = self._detect_layer_types()

        mlp_count = sum(1 for t in self.layer_types.values() if t == 'mlp')
        attn_count = sum(1 for t in self.layer_types.values() if t == 'attention')

        print(f"\n[Hybrid Group-Wise PRAQ ASYMMETRIC Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Strategy: AWQ Scaling + PRAQ Error Weighting + Asymmetric Quant")
        print(f"  MLP layers: {mlp_count}")
        print(f"  Attention layers: {attn_count}")
        print(f"  Hypothesis: Asymmetric should improve upon symmetric quantization")

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
                    layer_types[name] = 'mlp'  # Default to MLP
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
    def get_activation_salience_awq(self, name):
        """
        Compute standard AWQ activation salience: E[|X|] per input feature.

        This is used for SCALING weights to keep groups smooth.
        AWQ's input magnitude provides robust, stable scaling that works well
        with group-wise quantization.

        Returns:
            Tensor of shape [in_features]
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
    def get_output_importance_praq(self, name, module):
        """
        Compute PRAQ Output Importance: E[|SiLU(XW)|] per output channel.

        This is used for WEIGHTING THE ERROR during grid search.
        PRAQ's post-activation importance tells us which output channels
        actually matter (ignoring those killed by the activation function).

        Returns:
            Tensor of shape [out_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        # For attention layers, use uniform importance (no activation function)
        layer_type = self.layer_types.get(name, 'mlp')
        if layer_type != 'mlp':
            return torch.ones(module.weight.shape[0])

        X_list = self.activation_data[name]
        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else None

        # MiniCPM uses SiLU for MLPs
        activation_fn = torch.nn.functional.silu

        output_importance_sum = torch.zeros(module.weight.shape[0])
        total_samples = 0

        for x_batch in X_list:
            x_flat = x_batch.reshape(-1, x_batch.shape[-1])
            batch_size = x_flat.shape[0]

            # Compute Post-Activation Output on GPU
            x_gpu = x_flat.to(self.device)
            z = torch.matmul(x_gpu, W.t())
            if b is not None:
                z = z + b

            # PRAQ Key: Measure importance AFTER activation
            y = activation_fn(z)

            output_importance_sum += y.abs().sum(dim=0).cpu()
            total_samples += batch_size

            del x_gpu, z, y

        return output_importance_sum / total_samples

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W):
        """
        Group-wise ASYMMETRIC quantization.

        Quantizes weights with one scale and zero_point per group of input channels.
        Uses full INT4 range [0, 15].

        Args:
            W: Weight tensor [out_features, in_features]

        Returns:
            W_quant: Quantized and dequantized weights
        """
        out_features, in_features = W.shape

        # Optimize: Check if padding is needed
        if in_features % self.group_size == 0:
            W_grouped = W.view(out_features, -1, self.group_size)
            padded = False
        else:
            pad_len = self.group_size - (in_features % self.group_size)
            W_padded = torch.nn.functional.pad(W, (0, pad_len))
            W_grouped = W_padded.view(out_features, -1, self.group_size)
            padded = True

        # Compute min and max per group: [out_features, n_groups, 1]
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization parameters
        # Scale: (max - min) / (2^bits - 1)
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)  # Avoid division by zero

        # Zero point: maps 0.0 in FP to integer value
        # zero_point = round(-min / scale)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # Quantize to [0, 15]
        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, 15)

        # Dequantize
        W_dequant_grouped = (W_int - zero_point) * scale

        # Reshape back
        if not padded:
            return W_dequant_grouped.view(out_features, in_features)
        else:
            W_dequant = W_dequant_grouped.reshape(out_features, -1)
            return W_dequant[:, :in_features]

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Hybrid Grid Search Strategy with Asymmetric Quantization:

        1. Scale weights with AWQ metric (input magnitude)
           → Creates smooth, well-behaved groups for quantization

        2. Minimize error weighted by PRAQ metric (output importance)
           → Focuses optimization on channels that actually matter

        3. Use asymmetric quantization [0, 15]
           → Better representation of asymmetric weight distributions

        This combines AWQ's structural robustness with PRAQ's intelligent
        objective function and asymmetric quantization's full range.

        Returns:
            best_scales: Optimal per-input-channel scales
            best_alpha: Best scaling exponent
            best_error: Minimum PRAQ-weighted error
        """
        # === STEP 1: Get AWQ Salience for Scaling ===
        awq_salience = self.get_activation_salience_awq(name)
        if awq_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Normalize AWQ salience
        awq_salience = awq_salience.to(self.device)
        awq_salience_norm = awq_salience / (awq_salience.mean() + 1e-8)

        # === STEP 2: Get PRAQ Importance for Error Weighting ===
        praq_importance = self.get_output_importance_praq(name, module)
        if praq_importance is None:
            praq_importance = torch.ones(module.weight.shape[0])

        praq_importance = praq_importance.to(self.device)
        # Normalize to avoid scaling the loss magnitude
        praq_importance_norm = praq_importance / (praq_importance.mean() + 1e-8)

        # === STEP 3: Prepare Calibration Data ===
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        # Subsample for efficiency
        max_samples = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)
        del X_cpu

        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        # Compute original output (FP16 baseline)
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # === STEP 4: Grid Search ===
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # A. Scale weights with AWQ metric (smooth groups)
            scales = awq_salience_norm.pow(alpha).clamp(min=1e-5)
            W_scaled = W * scales.unsqueeze(0)

            # B. Quantize with group-wise ASYMMETRIC quantization [0, 15]
            W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

            # C. Compute output with compensated input
            X_compensated = X_search / scales.unsqueeze(0)
            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # D. Compute PRAQ-weighted error (intelligent objective)
            # Raw squared error: [batch, out_features]
            raw_error = (Y_orig - Y_quant).pow(2)

            # Weight by PRAQ importance: emphasize important output channels
            # praq_importance_norm is [out_features], broadcast over batch
            weighted_error = (raw_error * praq_importance_norm.unsqueeze(0)).mean()

            error_val = weighted_error.item()

            if error_val < best_error:
                best_error = error_val
                best_alpha = alpha
                best_scales = scales.clone()

        # Cleanup
        del X_search, Y_orig, Y_quant, raw_error, weighted_error
        torch.cuda.empty_cache()

        layer_type = self.layer_types.get(name, 'mlp')
        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Hybrid Quantization with Asymmetric Quantization to a layer.

        Steps:
        1. Grid search with hybrid strategy (AWQ scaling + PRAQ error weighting)
        2. Scale weight columns with optimal AWQ-based scales
        3. Quantize with group-wise ASYMMETRIC quantization [0, 15]
        4. Descale to restore original magnitude
        """
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        # 1. Scale weights (AWQ metric)
        W_scaled = W * best_scales.unsqueeze(0)

        # 2. Quantize (Group-wise Asymmetric)
        W_quant = self.quantize_weight_groupwise_asymmetric(W_scaled)

        # 3. Descale
        W_final = W_quant / best_scales.unsqueeze(0)

        # Update module weights
        module.weight.data = W_final

        # Store metadata
        layer_type = self.layer_types.get(name, 'mlp')
        self.layer_scales[name] = {
            'alpha': best_alpha,
            'error': best_error,
            'type': layer_type
        }

        del W_scaled, W_quant, W_final
        torch.cuda.empty_cache()

    def calibrate(self, calibration_data, n_samples=128):
        """
        Calibrate the quantizer on calibration data.

        Industry standard is 128 samples with 512 tokens each.
        """
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
        """Quantize all linear layers using Hybrid strategy with Asymmetric Quantization."""
        print("\n" + "=" * 80)
        print("Quantizing with HYBRID Group-Wise PRAQ + ASYMMETRIC")
        print("=" * 80)
        print("Strategy:")
        print("  1. Scale Weights: AWQ Input Magnitude E[|X|]")
        print("     → Creates smooth, well-behaved groups")
        print("  2. Minimize Error: PRAQ Output Importance E[|SiLU(XW)|]")
        print("     → Focuses on channels that survive activation")
        print("  3. Asymmetric Quantization [0, 15]")
        print("     → Full INT4 range for better representation")
        print("  4. Group-Wise Quantization (group_size={})".format(self.group_size))
        print("     → Hardware-efficient INT4")
        print("\nHypothesis: Asymmetric should improve upon symmetric!")
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
            mlp_alphas = [info['alpha'] for info in self.layer_scales.values() if info.get('type') == 'mlp']
            attn_alphas = [info['alpha'] for info in self.layer_scales.values() if info.get('type') == 'attention']

            print(f"\nOptimal α statistics:")
            print(f"  Overall - Mean: {np.mean(alphas):.3f}, Median: {np.median(alphas):.3f}")
            if mlp_alphas:
                print(f"  MLP layers - Mean: {np.mean(mlp_alphas):.3f}, Median: {np.median(mlp_alphas):.3f}")
            if attn_alphas:
                print(f"  Attention layers - Mean: {np.mean(attn_alphas):.3f}, Median: {np.median(attn_alphas):.3f}")

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
        description="Hybrid Group-Wise PRAQ with ASYMMETRIC quantization for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128,
                       help="Calibration samples (128 is industry standard)")
    parser.add_argument("--n-grid", type=int, default=20,
                       help="Grid search points for α")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gwh_praq_asym",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
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
    print("Hybrid Group-Wise PRAQ + ASYMMETRIC Quantization")
    print("=" * 80)
    print("Strategy:")
    print("  1. AWQ for Scaling: E[|X|] creates smooth groups")
    print("  2. PRAQ for Optimization: E[|SiLU(XW)|] weights error")
    print("  3. Asymmetric Quantization: Full INT4 range [0, 15]")
    print("  4. Group-Wise Quantization for hardware efficiency")
    print("\nCombines:")
    print("  ✓ AWQ's robustness (smooth, stable scaling)")
    print("  ✓ PRAQ's intelligence (smart error weighting)")
    print("  ✓ Asymmetric's full range (better representation)")
    print("  ✓ Group-wise efficiency (practical deployment)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
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
    quantizer = HybridGroupWisePRAQAsymmetricQuantizer(
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
    print("\nHybrid Group-Wise PRAQ + ASYMMETRIC Approach:")
    print("  ✓ AWQ Scaling: Smooth, robust weight groups")
    print("  ✓ PRAQ Error Weighting: Smart optimization objective")
    print("  ✓ Asymmetric Quantization: Full INT4 range [0, 15]")
    print("  ✓ Group-Wise Quantization: Hardware-efficient deployment")
    print("\nExpected Performance:")
    print("  - Should outperform symmetric GWH-PRAQ (better range utilization)")
    print("  - Should maintain hybrid advantages (AWQ + PRAQ)")
    print("\nNext Steps:")
    print("  Run: python compare_symmetric_vs_asymmetric.py")
    print("       to compare symmetric vs asymmetric quantization")
    print("=" * 80)


if __name__ == "__main__":
    main()
