"""
AWQ-GQA: Combined AWQ Quantization with GQA-Specific ReFlip Refinement

Pipeline:
1. Apply AWQ quantization to all linear layers (from awq_js_xl.py)
2. Detect Group-Query Attention layers (Q_proj, K_proj, V_proj)
3. Apply ReFlip refinement to GQA layers (from fast_quantize_qkv.py)

This provides:
- General AWQ quantization for all layers
- Specialized attention-aware refinement for GQA layers (Llama 3+, etc.)

Usage:
    python awq_gqa_xl.py \
        --model-path ./models/Llama-3-8B \
        --output-dir ./quantized_models/llama3_awq_gqa \
        --n-calib 128 \
        --apply-gqa-reflip  # Enable GQA ReFlip refinement
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import os
import numpy as np
from tqdm import tqdm
import gc

# Import AWQ quantizer from awq_js_xl
from awq_js_xl import JamesSteinHeuristicAWQQuantizerXL, compute_james_stein_mean


def is_gqa_layer(layer_name):
    """Detect if a layer is part of Group-Query Attention."""
    gqa_keywords = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']
    return any(keyword in layer_name.lower() for keyword in gqa_keywords)


def get_layer_group(layer_name):
    """Extract layer group from name (e.g., 'model.layers.0.self_attn')."""
    parts = layer_name.split('.')

    layer_idx = None
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                break
            except ValueError:
                continue

    if layer_idx is None:
        return None

    if 'self_attn' in parts:
        attn_idx = parts.index('self_attn')
        attn_group = '.'.join(parts[:attn_idx + 1])
        return (layer_idx, attn_group)

    return None


class AWQGQAQuantizer(JamesSteinHeuristicAWQQuantizerXL):
    """Extended AWQ Quantizer with GQA ReFlip refinement."""

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, use_heuristic=True, knee_tolerance=0.1,
                 max_tokens_per_sample=512, layer_batch_size=16, lmhead_chunks=8,
                 max_flip_percent=0.05, use_james_stein=True,
                 apply_gqa_reflip=False, gqa_critical_dim_pct=0.15,
                 gqa_max_flip_pct=0.05):

        super().__init__(
            model=model, tokenizer=tokenizer, device=device, bits=bits, n_grid=n_grid,
            group_size=group_size, use_heuristic=use_heuristic, knee_tolerance=knee_tolerance,
            max_tokens_per_sample=max_tokens_per_sample, layer_batch_size=layer_batch_size,
            lmhead_chunks=lmhead_chunks, max_flip_percent=max_flip_percent,
            use_james_stein=use_james_stein
        )

        self.apply_gqa_reflip = apply_gqa_reflip
        self.gqa_critical_dim_pct = gqa_critical_dim_pct
        self.gqa_max_flip_pct = gqa_max_flip_pct
        self.original_state_dict = None
        self.gqa_js_means = {}

    def quantize_layer(self, name, module):
        """Override to store James-Stein means for GQA layers."""
        if self.apply_gqa_reflip and is_gqa_layer(name):
            _, js_mean = self.get_activation_stats(name)
            if js_mean is not None:
                self.gqa_js_means[name] = js_mean.cpu().float()
        super().quantize_layer(name, module)

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """Override to add GQA ReFlip refinement after AWQ quantization."""
        if not self.apply_gqa_reflip:
            super().quantize_model_sequential(calibration_data, n_samples)
            return

        print("\n" + "=" * 80)
        print("AWQ-GQA: Combined Quantization Pipeline")
        print("=" * 80)
        print(f"  Step 1: Save original model state")
        print(f"  Step 2: AWQ quantization for all layers")
        print(f"  Step 3: GQA ReFlip refinement")
        print("=" * 80)

        # Step 1: Save original weights
        print("\n[Step 1] Saving original GQA layer weights...")
        self.original_state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                self.original_state_dict[name + '.weight'] = module.weight.data.clone().cpu()
        print(f"  ✓ Saved {len(self.original_state_dict)} GQA layer weights")

        # Step 2: AWQ quantization
        print("\n[Step 2] Running AWQ quantization...")
        super().quantize_model_sequential(calibration_data, n_samples)

        # Step 3: GQA ReFlip refinement
        print("\n[Step 3] Applying GQA ReFlip refinement...")
        self.apply_gqa_reflip_refinement()

        self.original_state_dict = None
        torch.cuda.empty_cache()
        gc.collect()

    def apply_gqa_reflip_refinement(self):
        """Apply ReFlip refinement to GQA layers."""
        print("\n" + "=" * 80)
        print("GQA ReFlip Refinement")
        print("=" * 80)

        # Group GQA layers by attention block
        attn_groups = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                group_info = get_layer_group(name)
                if group_info:
                    _, attn_group = group_info
                    if attn_group not in attn_groups:
                        attn_groups[attn_group] = {}

                    if 'q_proj' in name.lower() or 'query' in name.lower():
                        attn_groups[attn_group]['q_proj'] = (name, module)
                    elif 'k_proj' in name.lower() or 'key' in name.lower():
                        attn_groups[attn_group]['k_proj'] = (name, module)
                    elif 'v_proj' in name.lower() or 'value' in name.lower():
                        attn_groups[attn_group]['v_proj'] = (name, module)

        print(f"  Found {len(attn_groups)} attention groups")

        refined_count = 0
        for attn_group, projs in tqdm(attn_groups.items(), desc="  Refining GQA layers"):
            if 'q_proj' not in projs or 'k_proj' not in projs:
                print(f"    ⚠️  Skipping {attn_group}: Missing Q or K projection")
                continue

            try:
                self.refine_attention_group(attn_group, projs)
                refined_count += 1
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"    ⚠️  Error refining {attn_group}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n  ✓ Refined {refined_count}/{len(attn_groups)} attention groups")
        print("=" * 80)

        self.gqa_js_means.clear()

    def infer_head_dim(self, k_out):
        """
        Infer head dimension from K output features.
        FIXED: Checks largest dimensions first (256 before 64) to avoid incorrect splitting.

        Why this matters:
        - For Llama-3 (head_dim=128, k_out=1024):
          - WRONG (ascending): Checks 64 first → 1024%64==0 → Splits into 16 fake heads
          - RIGHT (descending): Checks 128 first → 1024%128==0 → Correct 8 heads
        """
        # Try to get from model config first
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'head_dim'):
            return self.model.config.head_dim

        # CRITICAL: Check in descending order to avoid incorrect head splitting
        possible_head_dims = [256, 128, 96, 80, 64]  # FIXED: Descending order is crucial
        for hd in possible_head_dims:
            if k_out % hd == 0:
                return hd

        return k_out

    def quantize_qkv_reflip_batched(self, Wq_orig_4d, Wk_orig_4d,
                                      Wq_int_heuristic_4d, Wq_scales_4d, Wq_zp_4d,
                                      Q_orig_all, Q_heuristic_all, K_orig_all, X, group_size):
        """
        Fully Vectorized ReFlip following fast_quantize_qkv.py logic.

        FIXED:
        - Uses Kneedle algorithm to find moderate dimensions (not TopK)
        - Redistributes score error proportionally to moderate dimensions
        - Computes targets based on redistributed corrections
        """
        num_k_heads, gqa_ratio, head_dim, hidden_dim = Wq_orig_4d.shape
        total_heads = num_k_heads * gqa_ratio

        # ===== 1. Flatten to [Total_Heads, Head_Dim, Hidden_Dim] =====
        Wq_int_flat = Wq_int_heuristic_4d.contiguous().view(total_heads, head_dim, hidden_dim).float()
        Wq_scales_flat = Wq_scales_4d.contiguous().view(total_heads, head_dim, -1)
        Wq_zp_flat = Wq_zp_4d.contiguous().view(total_heads, head_dim, -1)

        # ===== 2. Compute Q/K Values and Errors (Vectorized) =====
        # K_values: [total_heads, head_dim] - Each K head repeated gqa_ratio times
        # K_orig_all shape: [num_k_heads, s=1, head_dim] → expand → [num_k_heads, gqa_ratio, head_dim]
        K_values = K_orig_all.expand(num_k_heads, gqa_ratio, head_dim).reshape(total_heads, head_dim)

        Q_target = Q_orig_all.contiguous().view(total_heads, head_dim)
        Q_current = Q_heuristic_all.contiguous().view(total_heads, head_dim)

        # Attention scores: Q @ K (element-wise multiply and sum)
        scores_target = (Q_target * K_values).sum(dim=1)  # [total_heads]
        scores_current = (Q_current * K_values).sum(dim=1)  # [total_heads]

        # Error = Target - Current
        # Positive error means we need to INCREASE the score
        score_errors = scores_target - scores_current  # [total_heads]

        # ===== 3. FIXED: Identify MODERATE Dimensions using Kneedle Algorithm =====
        # Import kneedle function
        from utils_qkv import find_knee_point

        # Process each head to find moderate dimensions
        all_moderate_indices = []
        all_dim_corrections = []

        for head_idx in range(total_heads):
            Q_orig = Q_target[head_idx].cpu().numpy()
            score_error = score_errors[head_idx].item()

            # Sort by |Q_orig| magnitude (descending)
            sorted_indices_desc = np.argsort(np.abs(Q_orig))[::-1]
            sorted_magnitudes = np.abs(Q_orig[sorted_indices_desc])

            # Apply Kneedle to find knee point (transition from high to moderate)
            first_half = sorted_magnitudes[:head_dim // 2]
            knee_idx = find_knee_point(first_half[::-1], tolerance_offset=self.knee_tolerance)
            knee_idx = len(first_half) - knee_idx - 1  # Convert back to descending index

            # Select MODERATE dimensions: starting from knee, take next critical_dim_pct %
            num_moderate = max(int(self.gqa_critical_dim_pct * head_dim), 1)
            moderate_start = knee_idx
            moderate_end = min(moderate_start + num_moderate, head_dim)
            moderate_indices = sorted_indices_desc[moderate_start:moderate_end]

            all_moderate_indices.append(moderate_indices)

            # ===== 4. FIXED: Proportional Error Redistribution =====
            if len(moderate_indices) > 0:
                Q_moderate = Q_orig[moderate_indices]
                Q_moderate_abs = np.abs(Q_moderate)
                Q_moderate_sum = Q_moderate_abs.sum()

                if Q_moderate_sum > 1e-10:
                    # Proportional redistribution: correction[i] = score_error * (|Q[i]| / sum(|Q[moderate]|))
                    dim_corrections = score_error * (Q_moderate_abs / Q_moderate_sum)
                else:
                    # Uniform distribution if all values are near zero
                    dim_corrections = np.full(len(moderate_indices), score_error / len(moderate_indices))
            else:
                dim_corrections = np.array([])

            all_dim_corrections.append(dim_corrections)

        # ===== 5. FIXED: Apply Integer Flips Based on Redistributed Corrections =====
        # Expand scales and zp to full hidden_dim
        scales_expanded = Wq_scales_flat.repeat_interleave(group_size, dim=2)[:, :, :hidden_dim]
        zp_expanded = Wq_zp_flat.repeat_interleave(group_size, dim=2)[:, :, :hidden_dim]

        X_expanded = X.view(1, 1, -1)  # [1, 1, hidden_dim]
        total_flips = 0

        # Process each head and its moderate dimensions
        for head_idx in range(total_heads):
            moderate_indices = all_moderate_indices[head_idx]
            dim_corrections = all_dim_corrections[head_idx]
            score_error = score_errors[head_idx].item()

            if len(moderate_indices) == 0:
                continue

            # Process each moderate dimension
            for i, dim_idx in enumerate(moderate_indices):
                correction = dim_corrections[i]
                if abs(correction) < 1e-10:
                    continue

                # Get K value for this dimension
                K_value = K_values[head_idx, dim_idx].item()
                if abs(K_value) < 1e-10:
                    continue

                # ===== 5a. FIXED: Compute Target Q Based on Redistributed Correction =====
                # Get scales and zero points for this dimension
                scales_row = scales_expanded[head_idx, dim_idx, :]  # [hidden_dim]
                zp_row = zp_expanded[head_idx, dim_idx, :]  # [hidden_dim]

                # Compute current Q value
                W_current = (Wq_int_flat[head_idx, dim_idx, :] - zp_row) * scales_row
                Q_current = (X_expanded.squeeze() * W_current).sum().item()

                # Target: dim_corrections[i] is the change in (Q[i] * K[i]) we want
                # Therefore: delta_Q[i] = dim_corrections[i] / K[i]
                delta_Q_target = correction / K_value
                Q_target_dim = Q_current + delta_Q_target

                # Current error from target
                error_current = Q_current - Q_target_dim

                # ===== 5b. Determine Flip Direction (Following fast_quantize_qkv.py lines 393-396) =====
                # To correct: we need delta_score = score_error
                # Since score = Q @ K, we need: delta_Q[i] * K[i] to contribute to delta_score
                delta_score_needed = score_error

                if delta_score_needed > 0:  # Need to increase score
                    flip_direction = 1 if K_value > 0 else -1
                else:  # Need to decrease score
                    flip_direction = -1 if K_value > 0 else 1

                # ===== 5c. Greedy Flip Selection (Following fast_quantize_qkv.py lines 424-473) =====
                current_qvals = Wq_int_flat[head_idx, dim_idx, :].cpu().numpy()
                new_qvals = current_qvals + flip_direction

                # Validity check
                valid_flips = (new_qvals >= 0) & (new_qvals <= 15)

                # Calculate impact of each flip
                X_np = X.cpu().numpy()
                scales_np = scales_row.cpu().numpy()
                flip_impacts = flip_direction * scales_np * X_np

                # Filter to beneficial flips (reduce |error|)
                target_sign = -np.sign(error_current)
                beneficial_flips = (np.sign(flip_impacts) == target_sign) & valid_flips

                if not beneficial_flips.any():
                    continue

                # Sort beneficial flips by magnitude
                flip_scores = np.abs(flip_impacts) * beneficial_flips
                sorted_indices = np.argsort(-flip_scores)  # Descending

                # Greedy selection: find optimal K flips that minimize residual error
                cumsum_impacts = np.zeros(hidden_dim + 1)
                for k in range(1, hidden_dim + 1):
                    if beneficial_flips[sorted_indices[k-1]]:
                        cumsum_impacts[k] = cumsum_impacts[k-1] + flip_impacts[sorted_indices[k-1]]

                # Find K that minimizes |error_current + cumsum_impacts[K]|
                residuals = np.abs(error_current + cumsum_impacts)
                best_k = np.argmin(residuals)

                # Cap at max_flip_pct
                max_flips = int(hidden_dim * self.gqa_max_flip_pct)
                best_k = min(best_k, max_flips)

                # Apply the optimal flips
                if best_k > 0:
                    for k in range(best_k):
                        j = sorted_indices[k]
                        if beneficial_flips[j]:
                            Wq_int_flat[head_idx, dim_idx, j] += flip_direction
                            total_flips += 1

        # Clamp to valid range
        Wq_int_flat.clamp_(0, 15)

        # ===== 6. Dequantize and Reshape =====
        Wq_dequant_flat = (Wq_int_flat - zp_expanded) * scales_expanded
        Wq_refined_4d = Wq_dequant_flat.view(num_k_heads, gqa_ratio, head_dim, hidden_dim)

        # Stats
        total_moderate_dims = sum(len(indices) for indices in all_moderate_indices)
        avg_flip_rate = total_flips / (total_heads * head_dim * hidden_dim) * 100 if total_heads > 0 else 0

        stats = {
            'total_flips': total_flips,
            'avg_flip_rate': avg_flip_rate,
            'total_moderate_dims': total_moderate_dims,
            'moderate_dims_per_head': [len(indices) for indices in all_moderate_indices]
        }

        return Wq_refined_4d, stats

    def refine_attention_group(self, attn_group, projs):
        """Optimized GQA refinement using 4D tensor batching."""
        q_name, q_module = projs['q_proj']
        k_name, k_module = projs['k_proj']

        # Validate we have required data
        q_weight_key = q_name + '.weight'
        k_weight_key = k_name + '.weight'

        if (q_weight_key not in self.original_state_dict or
            k_weight_key not in self.original_state_dict or
            q_name not in self.gqa_js_means or
            k_name not in self.gqa_js_means):
            print(f"      ⚠️  Missing data for {attn_group}, skipping")
            return

        # ===== 1. Setup GPU Tensors =====
        device = q_module.weight.device
        dtype = torch.float32

        Wq_orig = self.original_state_dict[q_weight_key].to(device).to(dtype)
        Wk_orig = self.original_state_dict[k_weight_key].to(device).to(dtype)
        X = self.gqa_js_means[q_name].to(device).to(dtype)

        # ===== 2. Detect GQA Structure =====
        q_out, hidden_dim = Wq_orig.shape
        k_out = Wk_orig.shape[0]

        head_dim = self.infer_head_dim(k_out)
        num_k_heads = k_out // head_dim
        num_q_heads = q_out // head_dim
        gqa_ratio = num_q_heads // num_k_heads

        if num_q_heads % num_k_heads != 0:
            print(f"      ⚠️  Q heads ({num_q_heads}) not divisible by K heads ({num_k_heads})")
            return

        print(f"      GQA: {num_q_heads} Q heads, {num_k_heads} K heads, ratio {gqa_ratio}:1, head_dim={head_dim}")

        # ===== 3. Prepare AWQ-Quantized Weights =====
        Wq_awq = q_module.weight.data.to(dtype)

        from utils_qkv import quantize_weight_groupwise_int4

        Wq_heuristic_np, Wq_scales_np, Wq_zp_np, Wq_int_np = \
            quantize_weight_groupwise_int4(Wq_awq.cpu().numpy(), group_size=self.group_size, method='nearest')

        Wq_int = torch.from_numpy(Wq_int_np).to(device)
        Wq_scales = torch.from_numpy(Wq_scales_np).to(device).to(dtype)
        Wq_zp = torch.from_numpy(Wq_zp_np).to(device).to(dtype)

        # ===== 4. Reshape to 4D Batched Format =====
        Wq_orig_4d = Wq_orig.view(num_k_heads, gqa_ratio, head_dim, hidden_dim)
        Wk_orig_4d = Wk_orig.view(num_k_heads, 1, head_dim, hidden_dim)

        Wq_int_4d = Wq_int.view(num_k_heads, gqa_ratio, head_dim, hidden_dim)
        Wq_scales_4d = Wq_scales.view(num_k_heads, gqa_ratio, head_dim, -1)
        Wq_zp_4d = Wq_zp.view(num_k_heads, gqa_ratio, head_dim, -1)

        # ===== 5. Vectorized Q/K Projections (GPU) =====
        Q_orig_all = torch.einsum('bghd,d->bgh', Wq_orig_4d, X)  # [num_k, gqa_ratio, head_dim]

        # Expand scales and zp to full hidden_dim BEFORE operations
        scales_expanded = Wq_scales_4d.repeat_interleave(self.group_size, dim=3)[:,:,:,:hidden_dim]
        zp_expanded = Wq_zp_4d.repeat_interleave(self.group_size, dim=3)[:,:,:,:hidden_dim]

        # Dequantize: W = (W_int - zp) * scale
        Wq_dequant_4d = (Wq_int_4d.float() - zp_expanded) * scales_expanded
        Q_heuristic_all = torch.einsum('bghd,d->bgh', Wq_dequant_4d, X)

        K_orig_all = torch.einsum('bshd,d->bsh', Wk_orig_4d, X)  # [num_k, s=1, head_dim]

        # ===== 6. Apply Batched ReFlip =====
        try:
            Wq_refined_4d, stats = self.quantize_qkv_reflip_batched(
                Wq_orig_4d=Wq_orig_4d,
                Wk_orig_4d=Wk_orig_4d,
                Wq_int_heuristic_4d=Wq_int_4d,
                Wq_scales_4d=Wq_scales_4d,
                Wq_zp_4d=Wq_zp_4d,
                Q_orig_all=Q_orig_all,
                Q_heuristic_all=Q_heuristic_all,
                K_orig_all=K_orig_all,
                X=X,
                group_size=self.group_size
            )

            # ===== 7. Update Model Weights =====
            Wq_refined_2d = Wq_refined_4d.view(q_out, hidden_dim)
            q_module.weight.data.copy_(Wq_refined_2d.to(q_module.weight.dtype))

            print(f"      ✓ Refined {attn_group} ({num_k_heads} GQA groups)")
            print(f"        Total moderate dims: {stats['total_moderate_dims']}")
            print(f"        Total flips: {stats['total_flips']}")
            print(f"        Avg flip rate: {stats['avg_flip_rate']:.3f}%")

        except Exception as e:
            print(f"      ⚠️  ReFlip failed for {attn_group}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='AWQ-GQA: Combined Quantization')

    # All arguments from awq_js_xl.py
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--use-heuristic", action="store_true", default=True)
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false")
    parser.add_argument("--use-james-stein", action="store_true", default=True)
    parser.add_argument("--no-james-stein", dest="use_james_stein", action="store_false")
    parser.add_argument("--knee-tolerance", type=float, default=0.000)
    parser.add_argument("--max-flip-percent", type=float, default=0.05)
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./quantized_models/llama3_awq_gqa")
    parser.add_argument("--model-path", type=str, default="./models/Llama-3-8B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")

    # GQA ReFlip options
    parser.add_argument('--apply-gqa-reflip', action='store_true',
                        help='Apply ReFlip refinement to GQA layers')
    parser.add_argument('--gqa-critical-dim-pct', type=float, default=0.15)
    parser.add_argument('--gqa-max-flip-pct', type=float, default=0.05)

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AWQ-GQA: Combined Quantization with GQA ReFlip Refinement")
    print("=" * 80)
    print(f"  Model: {args.model_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Calibration samples: {args.n_calib}")
    print(f"  GQA ReFlip: {'Enabled' if args.apply_gqa_reflip else 'Disabled'}")
    if args.apply_gqa_reflip:
        print(f"    - Critical dim %: {args.gqa_critical_dim_pct}")
        print(f"    - Max flip %: {args.gqa_max_flip_pct}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"\nLoading model and tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Create quantizer with GQA support
    quantizer = AWQGQAQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        knee_tolerance=args.knee_tolerance,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        max_flip_percent=args.max_flip_percent,
        use_james_stein=args.use_james_stein,
        apply_gqa_reflip=args.apply_gqa_reflip,
        gqa_critical_dim_pct=args.gqa_critical_dim_pct,
        gqa_max_flip_pct=args.gqa_max_flip_pct
    )

    # Load calibration data
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(quantizer.tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(quantizer.tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    # Quantize model
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    # Save quantized model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
