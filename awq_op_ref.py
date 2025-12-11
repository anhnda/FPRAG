"""
Group-Wise AWQ with Heuristic-Guided Asymmetric Quantization (Optimized)

Key Innovation: Uses activation-aware heuristic from heuristic_quantize.py
Combines:
1. AWQ's activation-aware scaling: scales = E[X²]^α
2. Heuristic quantization: Uses E[Xs[:,j]] to guide rounding decisions

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm)
2. Grid search for optimal α ∈ [0, 1]
3. Scale weight COLUMNS: W[:, j] *= s[j]^α
4. Compute scaled activations: Xs = X / s
5. Quantize with HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC:
   - For each group: compute group_activation = E[Xs[:, group]]
   - Use group_activation to guide quantization (minimize output error)
   - Use asymmetric quantization [0, 15] with computed zero_point
6. Divide by input scales: W_final = Q(W*s) / s

Motivation:
- Standard quantization: Round to nearest, minimize weight error
- Heuristic quantization: Consider activation impact, minimize output error
- Key insight: error = dot(X, W - W_quant) depends on BOTH weights and activations
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

class HeuristicGroupWiseAWQQuantizer:
    """
    Group-Wise AWQ with Heuristic-Guided Asymmetric Quantization.

    Key Features:
    - Per-input-channel scaling based on E[X²] (L2 salience)
    - Grid search for optimal scaling exponent α
    - HEURISTIC-GUIDED GROUP-WISE quantization using E[Xs]
    - Minimizes output error rather than just weight error
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128, use_heuristic=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_heuristic = use_heuristic

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Heuristic Group-Wise AWQ Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Quantization: HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC [0, 15]")
        print(f"  Salience metric: E[X²] (L2 norm)")
        print(f"  Heuristic guidance: E[Xs] per group (signed mean)")
        print(f"  Use heuristic: {use_heuristic} (Vectorized Implementation)")

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                # Detach and move to CPU immediately to save VRAM
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
    def get_activation_stats(self, name):
        """
        Compute both L2 salience (E[X²]) and raw mean (E[X]) in ONE PASS.
        Uses float64 accumulation to prevent precision loss.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None, None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate both statistics on CPU in one pass using Double Precision
        l2_sum = torch.zeros(in_features, dtype=torch.float64)
        mean_sum = torch.zeros(in_features, dtype=torch.float64)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).double() # Convert to double for accumulation
            l2_sum += x_flat.pow(2).sum(dim=0)
            mean_sum += x_flat.sum(dim=0)

        salience = (l2_sum / total_samples).float()  # E[X²]
        raw_mean = (mean_sum / total_samples).float()  # E[X]

        return salience, raw_mean

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, group_activation_means, apply_heuristic=True):
        """
        Group-wise GLOBAL GREEDY heuristic quantization (asymmetric).
        
        **OPTIMIZED VECTORIZED IMPLEMENTATION**
        Replaces the per-channel Python loop with batched tensor operations.
        """
        out_features, in_features = W.shape
        device = W.device

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        # Reshape to [out_features, n_groups, group_size]
        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        # Compute min/max per group
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization parameters
        max_int = 2**self.bits - 1
        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        # Flatten scale and zp: [out_features, padded_in_features]
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        # Initial quantization (nearest rounding)
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_quant = (W_int - zp_flat) * scale_flat

        # === VECTORIZED GLOBAL GREEDY HEURISTIC ===
        if apply_heuristic:
            # 1. Calculate Error: dot(x, w - w_quant)
            # act_padded is [padded_in], W_diff is [out, padded_in]
            W_diff = W_padded - W_quant
            current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1) # [out_features]

            # 2. Identify Flip Directions and Impacts
            # Direction to move W_int to reduce quantization error (towards W_div)
            flip_dir = torch.sign(W_div - W_int)
            flip_dir[flip_dir == 0] = 1.0
            
            # Impact on the output if we flip: x * direction * scale
            # Broadcast act_padded: [1, padded_in] * [out, padded_in] * [out, padded_in]
            flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat

            # 3. Determine Valid Flips
            # We want to flip if the impact reduces the magnitude of the current error.
            # Sign of impact must match sign of error.
            target_sign = torch.sign(current_error).unsqueeze(1) # [out, 1]
            valid_mask = (torch.sign(flip_impacts) == target_sign)

            # Range check: don't flip out of [0, max_int]
            w_int_proposed = W_int + flip_dir
            in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
            valid_mask = valid_mask & in_range
            
            # Mask out invalid impacts (set to 0 so they don't affect cumsum significantly)
            # We use a large penalty for sorting later, effectively moving them to the end
            
            # 4. Sort by Rounding Cost (distance to boundary)
            rounding_costs = (W_div - W_int).abs()
            # Set cost of invalid flips to infinity so they are sorted last
            rounding_costs_masked = rounding_costs.clone()
            rounding_costs_masked[~valid_mask] = float('inf')
            
            # Sort: [out, padded_in]
            sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
            
            # Reorder impacts based on cost
            sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
            
            # Zero out impacts that were invalid (they are now at the tail)
            # (We need to re-gather the mask validity to zero them out strictly)
            sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
            sorted_impacts = sorted_impacts * sorted_validity

            # 5. Find Optimal K (Cumulative Sum)
            cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
            
            # We want to minimize |error - cumsum|
            # residuals: [out, padded_in]
            residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
            
            # Add the "0 flips" case (original error)
            # Concatenate [out, 1] with [out, padded_in] -> [out, padded_in + 1]
            error_unsqueezed = torch.abs(current_error).unsqueeze(1)
            all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
            
            # Best K index: [out_features]
            best_k = torch.argmin(all_residuals, dim=1)
            
            # 6. Apply Flips
            # Create a mask for the top K elements
            # Create a range tensor [0, 1, ..., padded_in-1]
            idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
            # Mask: where index < best_k (note: best_k indices align with sorted array)
            flip_mask_sorted = idx_range < best_k.unsqueeze(1)
            
            # Map back to original indices
            # We need to construct the update matrix.
            # Initialize zero update
            update_matrix = torch.zeros_like(W_int)
            
            # We have sorted_indices. The elements to flip are sorted_indices where flip_mask_sorted is True
            # Since scatter requires `dim` index, we use scatter_add or standard scatter
            
            # Prepare values to scatter: flip_dir
            # But we need to align flip_dir to the sorted positions first? No.
            # We have the boolean mask in SORTED order. We need to unsort it.
            
            # scatter expects `index` to be the same size as `src`? Or we can just iterate?
            # Since strict vectorization of "unsort" boolean mask is tricky with scatter, 
            # we simply apply changes where necessary.
            
            # Gather flip_dir in sorted order
            sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
            
            # Zero out non-selected flips
            sorted_flip_dir[~flip_mask_sorted] = 0.0
            
            # Scatter back to original positions
            # W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
            # Note: scatter_add_ with float works, W_int is float here until final cast
            W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)

        # Dequantize with updated integer values
        W_dequant = (W_int - zp_flat) * scale_flat

        # Remove padding
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        # Ensure output dtype matches input dtype
        if W_dequant.dtype != W.dtype:
            W_dequant = W_dequant.to(W.dtype)

        return W_dequant

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Grid search for optimal per-input-channel scaling factor.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get stats ONCE
        activation_salience, raw_mean = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Move to device
        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        raw_mean = raw_mean.to(self.device).to(module.weight.dtype)

        # Prepare calibration data subsample (limit to 2048 to save memory/time)
        X_list = self.activation_data[name]
        X_cpu = []
        curr_len = 0
        for x in X_list:
            x_f = x.reshape(-1, x.shape[-1])
            X_cpu.append(x_f)
            curr_len += x_f.shape[0]
            if curr_len >= 2048:
                break
        
        # Concatenate and move to GPU
        X_search = torch.cat(X_cpu, dim=0)[:2048].to(self.device)
        if X_search.dtype != module.weight.dtype:
            X_search = X_search.to(module.weight.dtype)

        W = module.weight.data
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weights
            W_scaled = W * scales.unsqueeze(0)

            # Scaled Mean Activation: E[X/s] = E[X] / s
            scaled_act_mean = raw_mean / scales

            # Quantize (Vectorized)
            W_quant = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic
            )

            # Descale: W_recon = W_quant / s
            W_recon = W_quant / scales.unsqueeze(0)

            # Error calculation
            Y_quant = torch.matmul(X_search, W_recon.t())
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()
            
            # Cleanup inside loop to prevent OOM on small GPUs
            del W_scaled, W_quant, W_recon, Y_quant, scales
        
        # Final cleanup
        del X_search, Y_orig
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Apply Heuristic Group-Wise AWQ Quantization."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        W_scaled = W * best_scales.unsqueeze(0)

        _, raw_mean = self.get_activation_stats(name)
        if raw_mean is not None:
            scaled_act_mean = (raw_mean.to(self.device).to(W.dtype) / best_scales)
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

        W_quant = self.quantize_weight_heuristic_groupwise(
            W_scaled,
            scaled_act_mean,
            apply_heuristic=self.use_heuristic
        )

        W_final = W_quant / best_scales.unsqueeze(0)
        module.weight.data = W_final

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error
        }

        # Aggressive cleanup
        del best_scales, scaled_act_mean, W_scaled, W_quant, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate(self, calibration_data, n_samples=500):
        """Run calibration to collect activations."""
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
        self.model.eval()
        self.register_hooks()

        successful = 0
        with torch.no_grad():
            for i, text in enumerate(tqdm(calibration_data[:n_samples], desc="Calibration")):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1
                except Exception:
                    continue

        self.remove_hooks()
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Calibration complete! Processed {successful}/{n_samples}")

    def quantize_model(self):
        """Quantize all linear layers."""
        print("\n" + "=" * 80)
        print("Quantizing with Heuristic-Guided Group-Wise AWQ (Vectorized)")
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
                        print(f"  Layer {name}: α={info['alpha']:.3f}, error={info['error']:.6f}")
                
                quantized_count += 1

            except Exception as e:
                print(f"\n⚠️  Error quantizing {name}: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                if name in self.activation_data:
                    del self.activation_data[name]
                torch.cuda.empty_cache()
                continue

        print(f"\n✅ Quantization complete! Layers: {quantized_count}, Skipped: {skipped_count}")

def load_wikitext2(split="train", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    # Handle the fact that wikitext usually has a 'text' column
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts

def main():
    parser = argparse.ArgumentParser(description="Heuristic Group-Wise AWQ")
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument("--use-heuristic", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="./quantized_models/awq_heuristic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device} | Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, 
                                                device_map=device, trust_remote_code=True)

    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)

    quantizer = HeuristicGroupWiseAWQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic
    )

    quantizer.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer.quantize_model()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()