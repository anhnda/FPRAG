"""
Group-Wise AWQ Implementation with ASYMMETRIC Quantization + L2 Salience

Key Difference from gw_awq_asym.py:
- gw_awq_asym.py: Uses E[|X|] (L1 norm) for activation salience
- gw_awq_asym_l2.py: Uses E[X²] (L2 norm) for activation salience

Why L2 is Better:
- Quantization MSE = E[(δW × X)²] ∝ E[X²]
- L2 emphasizes channels with spikes/outliers (quadratic weighting)
- Matches the squared error objective directly

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm)
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
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  Warning: psutil not installed. Memory monitoring disabled.")
    print("   Install with: pip install psutil")

from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data


class GroupWiseAWQAsymmetricL2Quantizer:
    """
    Group-Wise AWQ with Asymmetric Quantization and L2 Salience.

    Key Features:
    - Per-input-channel scaling based on E[X²] (L2 salience)
    - Grid search for optimal scaling exponent α
    - GROUP-WISE ASYMMETRIC INT4 quantization [0, 15]
    - Better MSE alignment through L2 norm
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128, max_tokens_per_sample=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.max_tokens_per_sample = max_tokens_per_sample  # Subsample to save memory

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Group-Wise AWQ ASYMMETRIC L2 Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample (memory optimization)")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric: E[X²] (L2 norm) - Better MSE alignment")


    @torch.no_grad()
    def get_activation_salience_l2(self, name):
        """
        Compute per-input-channel activation salience using L2 norm: E[X[:, j]²]

        Key Difference from L1:
        - L1: E[|X|] - Linear weighting, average magnitude
        - L2: E[X²] - Quadratic weighting, emphasizes spikes

        Why L2 is Better:
        - Quantization error is quadratic: (δW × X)²
        - MSE = E[(δW × X)²] = E[δW²] × E[X²] (if independent)
        - L2 directly matches the MSE objective

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
            # KEY CHANGE: Use pow(2) instead of abs()
            salience_sum += x_flat.pow(2).sum(dim=0)

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
        Grid search for optimal per-input-channel scaling factor using L2 salience.

        Algorithm with L2:
        1. For α in [0, 0.05, 0.1, ..., 0.95, 1.0]:
           a. Compute scales: s[j] = (E[X[:, j]²])^α for each input channel j
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

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
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

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
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
        Apply Group-Wise AWQ with Asymmetric Quantization and L2 Salience.

        Steps:
        1. Grid search for best per-input-channel scales (L2-based)
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

    def calibrate_single_layer(self, layer_name, calibration_data, n_samples=500):
        """
        Calibrate a SINGLE layer (memory efficient approach).

        Only stores activations for ONE layer, then quantizes it immediately.
        This uses constant memory regardless of number of layers.
        """
        # Clear any previous activation data
        self.activation_data = {}

        # Register hook for THIS layer only
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                handle = module.register_forward_hook(self.get_hook(layer_name))

                # Run calibration data through model
                with torch.no_grad():
                    for i, text in enumerate(calibration_data[:n_samples]):
                        try:
                            inputs = self.tokenizer(text, return_tensors="pt",
                                                   truncation=True, max_length=512)
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}

                            _ = self.model(**inputs, use_cache=False, return_dict=True)
                            del inputs

                            # Cleanup every 16 samples
                            if (i + 1) % 16 == 0:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()
                        except:
                            continue

                # Remove hook
                handle.remove()
                break

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if sequence is too long
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len, device=inp.device)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            # Store activation (keep as float16 to save memory)
            inp_stored = inp.detach().cpu().clone()
            self.activation_data[name].append(inp_stored)
            del inp
        return hook

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """
        SEQUENTIAL LAYER-BY-LAYER QUANTIZATION (Memory Efficient).

        Strategy:
        1. For each layer:
           - Calibrate ONLY this layer (store activations for 1 layer)
           - Quantize this layer immediately
           - Clear activations
           - Move to next layer

        Benefits:
        - Constant memory usage (~280MB per layer vs ~75GB for all layers)
        - Better accuracy (accounts for error propagation)
        - Can handle 128+ samples with <10GB RAM
        """
        print("\n" + "=" * 80)
        print("SEQUENTIAL Layer-by-Layer Quantization (Memory Efficient)")
        print("=" * 80)
        print("Strategy:")
        print("  1. Calibrate ONE layer at a time")
        print("  2. Quantize that layer immediately")
        print("  3. Clear activations and move to next layer")
        print("  4. Constant memory usage regardless of total layers")
        print("\nBenefits:")
        print("  ✓ Memory: ~280MB per layer (vs ~75GB for all layers)")
        print("  ✓ Can handle 128+ samples with <10GB RAM")
        print("  ✓ Better accuracy (accounts for error propagation)")
        print("=" * 80)

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            initial_gb = psutil.virtual_memory().used / (1024**3)
            print(f"\nInitial RAM: {initial_ram:.1f}% ({initial_gb:.1f} GB)")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        print(f"\nFound {len(layer_names)} linear layers to quantize")

        quantized_count = 0
        skipped_count = 0

        for name, module in tqdm(layer_names, desc="Sequential Quantization"):
            try:
                # STEP 1: Calibrate THIS layer only
                self.calibrate_single_layer(name, calibration_data, n_samples)

                # STEP 2: Quantize THIS layer
                self.quantize_layer(name, module)

                # STEP 3: Clear activations for this layer
                if name in self.activation_data:
                    del self.activation_data[name]
                self.activation_data = {}

                quantized_count += 1

                # Show progress every 10 layers
                if quantized_count % 10 == 0:
                    if HAS_PSUTIL:
                        ram_pct = psutil.virtual_memory().percent
                        ram_gb = psutil.virtual_memory().used / (1024**3)
                        print(f"\n  [{quantized_count}/{len(layer_names)}] RAM: {ram_pct:.1f}% ({ram_gb:.1f} GB)")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"\n⚠️  Error quantizing layer {name}: {e}")
                skipped_count += 1
                continue

        print(f"\n✅ Sequential Quantization Complete!")
        print(f"   Total layers quantized: {quantized_count}/{len(layer_names)}")
        if skipped_count > 0:
            print(f"   ⚠️  Skipped {skipped_count} layers due to errors")

        if HAS_PSUTIL:
            final_ram = psutil.virtual_memory().percent
            final_gb = psutil.virtual_memory().used / (1024**3)
            print(f"\nFinal RAM: {final_ram:.1f}% ({final_gb:.1f} GB)")
            print(f"Memory increase: {final_gb - initial_gb:.1f} GB")

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")
            print(f"  Min: {np.min(alphas):.3f}")
            print(f"  Max: {np.max(alphas):.3f}")

        # Final cleanup
        self.activation_data = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_c4_calibration(tokenizer, n_samples=128, seq_len=2048, seed=42):
    """
    DEPRECATED: Use calibration_utils.get_c4_calibration_data() instead.

    This wrapper is kept for backward compatibility.
    """
    return get_c4_calibration_data(tokenizer, n_samples, seq_len, seed)


def load_wikitext2_simple(n_samples=128):
    """
    Simple WikiText-2 loader (MEMORY EFFICIENT - original approach).

    Use this if you're experiencing memory issues with the new loaders.
    This is the original approach that was fast and memory-efficient.
    """
    from datasets import load_dataset
    print(f"Loading WikiText-2 (simple/fast approach)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    print(f"  ✓ Loaded {len(texts[:n_samples])} samples from WikiText-2")
    return texts[:n_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Group-Wise AWQ with ASYMMETRIC quantization and L2 Salience for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample (subsampling for memory, default: 512)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_awq_asym_l2",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset (default: c4). Use 'wikitext2-simple' for lowest memory usage")
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
    print("Group-Wise AWQ with ASYMMETRIC Quantization + L2 Salience")
    print("=" * 80)
    print("Algorithm:")
    print("  1. Per-input-channel L2 salience: s[j] = E[X[:, j]²]")
    print("     → Better MSE alignment (quadratic weighting)")
    print("  2. Grid search optimal α ∈ [0, 1]")
    print("  3. Column-wise weight scaling: W[:, j] *= s[j]^α")
    print(f"  4. GROUP-WISE ASYMMETRIC INT4 quantization [0, 15] (group_size={args.group_size})")
    print("     - scale = (max - min) / 15 per group")
    print("     - zero_point = round(-min / scale) per group")
    print("  5. Descaling: W_final = Q(W*s) / s")
    print("\nKey Improvement: L2 norm emphasizes spike channels")
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
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        # MEMORY OPTIMIZATION: Match seqlen with calibration max_length (512)
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)
    elif args.calib_dataset == "wikitext2-simple":
        # MEMORY EFFICIENT: Original simple approach (fastest, lowest memory)
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        # MEMORY OPTIMIZATION: Match seqlen with calibration max_length (512)
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=512, seed=args.seed)

    # Initialize quantizer
    quantizer = GroupWiseAWQAsymmetricL2Quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        max_tokens_per_sample=args.max_tokens_per_sample
    )

    # Sequential layer-by-layer quantization (memory efficient)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

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
    print("\nGroup-Wise AWQ ASYMMETRIC + L2 Approach:")
    print("  ✓ L2 salience: E[X²] (MSE-aligned)")
    print("  ✓ Grid search for optimal α")
    print("  ✓ Column-wise weight scaling")
    print(f"  ✓ GROUP-WISE ASYMMETRIC quantization [0, 15]")
    print("  ✓ Better representation through L2 norm")
    print("=" * 80)


if __name__ == "__main__":
    main()
