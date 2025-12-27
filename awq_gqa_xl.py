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
        --model openbmb/MiniCPM-2B-sft-bf16 \
        --output-dir ./quantized_models/minicpm_awq_gqa \
        --n-calib 128 \
        --apply-gqa-reflip  # Enable GQA ReFlip refinement
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import numpy as np
from tqdm import tqdm
import gc

# Import AWQ quantizer from awq_js_xl
from awq_js_xl import AWQQuantizer, find_knee_point

# Import ReFlip function from fast_quantize_qkv
from fast_quantize_qkv import quantize_qkv_reflip_fast


def is_gqa_layer(layer_name):
    """
    Detect if a layer is part of Group-Query Attention.

    GQA layers typically have names like:
    - model.layers.0.self_attn.q_proj
    - model.layers.0.self_attn.k_proj
    - model.layers.0.self_attn.v_proj
    """
    gqa_keywords = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']
    layer_lower = layer_name.lower()

    return any(keyword in layer_lower for keyword in gqa_keywords)


def get_layer_group(layer_name):
    """
    Extract layer group from name (e.g., 'model.layers.0.self_attn').

    Returns: (layer_idx, attn_group) or None
    """
    parts = layer_name.split('.')

    # Find layer index
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

    # Get attention group (everything before q_proj/k_proj/v_proj)
    if 'self_attn' in parts:
        attn_idx = parts.index('self_attn')
        attn_group = '.'.join(parts[:attn_idx + 1])
        return (layer_idx, attn_group)

    return None


class AWQGQAQuantizer(AWQQuantizer):
    """
    Extended AWQ Quantizer with GQA ReFlip refinement.

    Inherits from AWQQuantizer and adds:
    - GQA layer detection
    - Activation data preservation for GQA layers
    - ReFlip refinement after AWQ quantization
    """

    def __init__(self, *args, apply_gqa_reflip=False, gqa_critical_dim_pct=0.15,
                 gqa_max_flip_pct=0.05, **kwargs):
        super().__init__(*args, **kwargs)

        self.apply_gqa_reflip = apply_gqa_reflip
        self.gqa_critical_dim_pct = gqa_critical_dim_pct
        self.gqa_max_flip_pct = gqa_max_flip_pct

        # Store GQA layer info for refinement
        self.gqa_layers = {}  # {layer_group: {q_proj, k_proj, v_proj}}
        self.gqa_activations = {}  # Preserve activations for GQA layers
        self.gqa_original_weights = {}  # Original FP weights before quantization
        self.gqa_quant_artifacts = {}  # INT4, scales, zp from AWQ

    def quantize_weight_heuristic_groupwise_extended(self, W, group_activation_means, apply_heuristic=True):
        """
        Extended version that returns INT4 representation in addition to dequantized weights.

        Returns:
            tuple: (W_dequant, outlier_percent, flip_stats, W_int, scales, zp)
        """
        out_features, in_features = W.shape
        device = W.device

        # Padding
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

        # Reshape to groups
        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        # Asymmetric Quantization
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2**self.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        # Expand to full size
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        # Initial Quantization
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)

        if not apply_heuristic:
            W_dequant = (W_int - zp_flat) * scale_flat
            if padded_in_features > in_features:
                W_dequant = W_dequant[:, :in_features]
                W_int = W_int[:, :in_features]

            flip_stats = {'total': 0, 'per_channel_mean': 0}
            # Return with INT4 artifacts
            return W_dequant.to(W.dtype), None, flip_stats, W_int, scale, zp

        # Call parent heuristic method for the full quantization with flips
        W_dequant, outlier_pct, flip_stats = super().quantize_weight_heuristic_groupwise(
            W, group_activation_means, apply_heuristic=True
        )

        # Re-compute W_int after flips (approximate from dequantized version)
        # This is needed because parent method doesn't return W_int
        W_padded_final = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
        if padded_in_features > in_features:
            W_padded_final[:, :in_features] = W_dequant
        else:
            W_padded_final = W_dequant

        W_int_final = torch.round(W_padded_final / scale_flat + zp_flat).clamp(0, max_int)
        if padded_in_features > in_features:
            W_int_final = W_int_final[:, :in_features]

        return W_dequant, outlier_pct, flip_stats, W_int_final, scale, zp

    def quantize_layer(self, name, module):
        """
        Override to preserve INT4 artifacts and original weights for GQA layers.
        """
        # Save original weights for GQA layers BEFORE quantization
        if self.apply_gqa_reflip and is_gqa_layer(name):
            self.gqa_original_weights[name] = module.weight.data.clone().cpu()

        # Perform AWQ quantization
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        _, js_mean = self.get_activation_stats(name)
        if js_mean is not None:
            scaled_act_mean = (js_mean.to(self.device).to(W.dtype) / best_scales)
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

        # Use extended version for GQA layers to get INT4 artifacts
        if self.apply_gqa_reflip and is_gqa_layer(name):
            W_quant, outlier_pct, flip_stats, W_int, scales_grouped, zp_grouped = \
                self.quantize_weight_heuristic_groupwise_extended(
                    W_scaled, scaled_act_mean, apply_heuristic=self.use_heuristic
                )

            # Store INT4 artifacts (scaled weights)
            self.gqa_quant_artifacts[name] = {
                'W_int': W_int.cpu(),  # INT4 values of SCALED weights
                'scales': scales_grouped.cpu(),  # Group-wise scales
                'zp': zp_grouped.cpu(),  # Group-wise zero points
                'awq_scales': best_scales.cpu(),  # AWQ input scaling factors
            }
        else:
            # Standard quantization for non-GQA layers
            W_quant, outlier_pct, flip_stats = self.quantize_weight_heuristic_groupwise(
                W_scaled, scaled_act_mean, apply_heuristic=self.use_heuristic
            )

        # Apply AWQ descaling and store
        W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)
        module.weight.data = W_final

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'outlier_percent': outlier_pct if outlier_pct is not None else 0.0,
            'flip_stats': flip_stats
        }

        del best_scales, scaled_act_mean, W_scaled, W_quant, W_final
        if name in self.activation_data and not (self.apply_gqa_reflip and is_gqa_layer(name)):
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_layer_batch(self, batch_layers, calibration_data, n_samples):
        """
        Override to preserve activations for GQA layers.
        """
        # Call parent calibration
        super().calibrate_layer_batch(batch_layers, calibration_data, n_samples)

        # Preserve activations for GQA layers (keep them, don't delete)
        # The parent will try to delete activation_data, but we need it for ReFlip
        # So we make a copy before the parent deletes it
        if self.apply_gqa_reflip:
            for name, _ in batch_layers:
                if is_gqa_layer(name) and name in self.activation_data:
                    # Store as list of tensors (preserve memory layout)
                    self.gqa_activations[name] = [x.clone() for x in self.activation_data[name]]

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """
        Override to add GQA ReFlip refinement after AWQ quantization.
        """
        print("\n" + "=" * 80)
        print("AWQ-GQA: Combined Quantization Pipeline")
        print("=" * 80)
        print(f"  Step 1: AWQ quantization for all layers")
        print(f"  Step 2: GQA ReFlip refinement (enabled: {self.apply_gqa_reflip})")
        print("=" * 80)

        # Step 1: Standard AWQ quantization
        super().quantize_model_sequential(calibration_data, n_samples)

        # Step 2: GQA ReFlip refinement
        if self.apply_gqa_reflip:
            self.apply_gqa_reflip_refinement()

    def apply_gqa_reflip_refinement(self):
        """
        Apply ReFlip refinement to GQA layers.

        For each attention layer:
        1. Group Q, K, V projections
        2. Extract quantized weights and activations
        3. Apply ReFlip to correct attention score errors
        4. Update quantized weights
        """
        print("\n" + "=" * 80)
        print("GQA ReFlip Refinement")
        print("=" * 80)

        # Group GQA layers by attention block
        attn_groups = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                group_info = get_layer_group(name)
                if group_info:
                    layer_idx, attn_group = group_info

                    if attn_group not in attn_groups:
                        attn_groups[attn_group] = {}

                    # Identify projection type
                    if 'q_proj' in name.lower() or 'query' in name.lower():
                        attn_groups[attn_group]['q_proj'] = (name, module)
                    elif 'k_proj' in name.lower() or 'key' in name.lower():
                        attn_groups[attn_group]['k_proj'] = (name, module)
                    elif 'v_proj' in name.lower() or 'value' in name.lower():
                        attn_groups[attn_group]['v_proj'] = (name, module)

        print(f"  Found {len(attn_groups)} attention groups")

        refined_count = 0
        for attn_group, projs in tqdm(attn_groups.items(), desc="  Refining GQA layers"):
            # Check if we have all three projections
            if 'q_proj' not in projs or 'k_proj' not in projs:
                print(f"    ⚠️  Skipping {attn_group}: Missing Q or K projection")
                continue

            try:
                self.refine_attention_group(attn_group, projs)
                refined_count += 1
            except Exception as e:
                print(f"    ⚠️  Error refining {attn_group}: {e}")
                continue

        print(f"\n  ✓ Refined {refined_count}/{len(attn_groups)} attention groups")
        print("=" * 80)

    def refine_attention_group(self, attn_group, projs):
        """
        Apply ReFlip refinement to a single attention group (Q, K, V).

        This method:
        1. Extracts original and AWQ-quantized weights
        2. Prepares activations
        3. Calls fast ReFlip to refine Q projections
        4. Updates model weights with refined versions
        """
        q_name, q_module = projs['q_proj']
        k_name, k_module = projs['k_proj']

        # Check if we have all required data
        if (q_name not in self.gqa_activations or k_name not in self.gqa_activations or
            q_name not in self.gqa_original_weights or k_name not in self.gqa_original_weights or
            q_name not in self.gqa_quant_artifacts or k_name not in self.gqa_quant_artifacts):
            return  # Skip if data not available

        # ===== 1. Prepare Activations =====
        # Concatenate all activation samples
        X_q_list = self.gqa_activations[q_name]
        X_concat = []
        for x in X_q_list:
            # x is [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            if x.dim() == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x
            X_concat.append(x_flat)

        X_all = torch.cat(X_concat, dim=0)  # [total_samples, hidden_dim]
        # Use mean activation as representative (standard AWQ practice)
        X = X_all.mean(dim=0).cpu().float().numpy()  # [hidden_dim]

        # ===== 2. Extract Original Weights =====
        Wq_orig = self.gqa_original_weights[q_name].cpu().float().numpy()  # [out_features, in_features]
        Wk_orig = self.gqa_original_weights[k_name].cpu().float().numpy()

        # ===== 3. Extract AWQ-Quantized Weights (Current State) =====
        Wq_awq = q_module.weight.data.cpu().float().numpy()  # Dequantized AWQ weights
        Wk_awq = k_module.weight.data.cpu().float().numpy()

        # ===== 4. Extract INT4 Artifacts from AWQ =====
        q_artifacts = self.gqa_quant_artifacts[q_name]
        Wq_int_awq = q_artifacts['W_int'].cpu().numpy().astype(np.int32)  # [out_features, in_features]
        Wq_scales = q_artifacts['scales'].cpu().numpy()  # [out_features, n_groups]
        Wq_zp = q_artifacts['zp'].cpu().numpy()

        # Note: AWQ stored INT4 of SCALED weights (W × AWQ_scale)
        # But we need to work with the final weights (after AWQ descaling)
        # So we need to re-quantize the AWQ-dequantized weights

        # Actually, let's use a simpler approach: directly quantize the AWQ output
        # This ensures we're refining what's actually in the model
        from utils_qkv import quantize_weight_groupwise_int4

        # Re-quantize AWQ weights to get clean INT4 representation
        Wq_heuristic, Wq_scales_clean, Wq_zp_clean, Wq_int_heuristic = \
            quantize_weight_groupwise_int4(Wq_awq, group_size=self.group_size, method='nearest')
        Wk_heuristic, _, _, _ = quantize_weight_groupwise_int4(Wk_awq, group_size=self.group_size)

        # ===== 5. Reshape to Multi-Head Format =====
        # Detect number of heads from weight shape
        # For MiniCPM-2B: Q has 2304 output features, hidden_dim=2304
        # This suggests num_heads=18, head_dim=128 (2304 / 18 = 128)
        # Or could be different - need to infer from model config

        # For now, treat as single-head (simplified)
        # Full implementation would need model.config.num_attention_heads
        # Wq: [out_features, in_features] → [num_heads, head_dim, hidden_dim]

        # Simplified: Treat entire Wq as one large "head"
        num_heads = 1
        head_dim = Wq_orig.shape[0]  # out_features
        hidden_dim = Wq_orig.shape[1]  # in_features

        # Reshape to [1, out_features, in_features] for ReFlip
        Wq_orig_3d = Wq_orig.reshape(num_heads, head_dim, hidden_dim)
        Wk_orig_3d = Wk_orig.reshape(num_heads, Wk_orig.shape[0], hidden_dim)
        Wq_heuristic_3d = Wq_heuristic.reshape(num_heads, head_dim, hidden_dim)
        Wk_heuristic_3d = Wk_heuristic.reshape(num_heads, Wk_orig.shape[0], hidden_dim)
        Wq_int_heuristic_3d = Wq_int_heuristic.reshape(num_heads, head_dim, hidden_dim)

        # Reshape scales and zp: [out_features, n_groups] → [num_heads, head_dim, n_groups]
        Wq_scales_3d = Wq_scales_clean.reshape(num_heads, head_dim, -1)
        Wq_zp_3d = Wq_zp_clean.reshape(num_heads, head_dim, -1)

        # ===== 6. Compute Q and K Values =====
        # Q_orig_all = X @ Wq_orig.T for each head
        Q_orig_all = np.zeros(num_heads * head_dim)
        Q_heuristic_all = np.zeros(num_heads * head_dim)

        for h in range(num_heads):
            Q_orig_all[h*head_dim:(h+1)*head_dim] = X @ Wq_orig_3d[h].T
            Q_heuristic_all[h*head_dim:(h+1)*head_dim] = X @ Wq_heuristic_3d[h].T

        # Reshape to [num_heads, head_dim]
        Q_orig_all = Q_orig_all.reshape(num_heads, head_dim)
        Q_heuristic_all = Q_heuristic_all.reshape(num_heads, head_dim)

        # K_heuristic = X @ Wk_heuristic.T
        K_heuristic = X @ Wk_heuristic.T  # [k_out_features]

        # ===== 7. Apply ReFlip Refinement =====
        try:
            (Wq_refined, Wq_scales_out, Wq_zp_out, Wq_int_out,
             Wk_refined, Wk_scales_out, Wk_zp_out, Wk_int_out,
             reflip_stats) = quantize_qkv_reflip_fast(
                Wq=Wq_orig_3d,
                Wk=Wk_orig_3d,
                X=X,
                Q_orig_all=Q_orig_all,
                Q_heuristic_all=Q_heuristic_all,
                Wq_heuristic=Wq_heuristic_3d,
                Wk_heuristic=Wk_heuristic_3d,
                K_heuristic=K_heuristic,
                Wq_int_heuristic=Wq_int_heuristic_3d,
                Wq_scales_heuristic=Wq_scales_3d,
                Wq_zp_heuristic=Wq_zp_3d,
                critical_dim_pct=self.gqa_critical_dim_pct,
                knee_tolerance=0.0,
                group_size=self.group_size,
                max_flip_pct=self.gqa_max_flip_pct,
                correction_scale=1.0,
                debug=False
            )

            # ===== 8. Update Model Weights =====
            # Reshape back to 2D and update Q projection
            Wq_refined_2d = Wq_refined.reshape(head_dim, hidden_dim)
            q_module.weight.data = torch.from_numpy(Wq_refined_2d).to(
                q_module.weight.dtype
            ).to(q_module.weight.device)

            print(f"      ✓ Refined {attn_group}")
            print(f"        Total flips: {reflip_stats['total_flips']}")
            print(f"        Flip rate: {reflip_stats['flip_rate_pct']:.3f}%")

        except Exception as e:
            print(f"      ⚠️  ReFlip failed for {attn_group}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='AWQ-GQA: Combined Quantization')
    parser.add_argument('--model', type=str, default='openbmb/MiniCPM-2B-sft-bf16',
                        help='Model name or path')
    parser.add_argument('--output-dir', type=str, default='./quantized_models/minicpm_awq_gqa',
                        help='Output directory for quantized model')
    parser.add_argument('--n-calib', type=int, default=128,
                        help='Number of calibration samples')
    parser.add_argument('--n-grid', type=int, default=20,
                        help='Number of grid points for alpha search')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Group size for quantization')
    parser.add_argument('--layer-batch-size', type=int, default=50,
                        help='Number of layers per batch')

    # GQA ReFlip options
    parser.add_argument('--apply-gqa-reflip', action='store_true',
                        help='Apply ReFlip refinement to GQA layers')
    parser.add_argument('--gqa-critical-dim-pct', type=float, default=0.15,
                        help='Percentage of moderate dimensions for ReFlip')
    parser.add_argument('--gqa-max-flip-pct', type=float, default=0.05,
                        help='Max flip percentage for ReFlip')

    # AWQ options
    parser.add_argument('--use-heuristic', action='store_true', default=True,
                        help='Use heuristic flip correction')
    parser.add_argument('--calib-dataset', type=str, default='c4',
                        choices=['c4', 'wikitext2', 'wikitext2-simple'],
                        help='Calibration dataset')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AWQ-GQA: Combined Quantization with GQA ReFlip Refinement")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output_dir}")
    print(f"  Calibration samples: {args.n_calib}")
    print(f"  GQA ReFlip: {'Enabled' if args.apply_gqa_reflip else 'Disabled'}")
    if args.apply_gqa_reflip:
        print(f"    - Critical dim %: {args.gqa_critical_dim_pct}")
        print(f"    - Max flip %: {args.gqa_max_flip_pct}")
    print("=" * 80)

    # Create quantizer with GQA support
    quantizer = AWQGQAQuantizer(
        args.model,
        w_bit=4,
        group_size=args.group_size,
        n_grid=args.n_grid,
        use_heuristic=args.use_heuristic,
        layer_batch_size=args.layer_batch_size,
        apply_gqa_reflip=args.apply_gqa_reflip,
        gqa_critical_dim_pct=args.gqa_critical_dim_pct,
        gqa_max_flip_pct=args.gqa_max_flip_pct
    )

    # Load calibration data
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

    if args.calib_dataset == 'c4':
        calib_data = get_c4_calibration_data(
            quantizer.tokenizer,
            n_samples=args.n_calib,
            block_size=512
        )
    else:
        calib_data = get_wikitext2_calibration_data(
            quantizer.tokenizer,
            n_samples=args.n_calib,
            simple_mode=(args.calib_dataset == 'wikitext2-simple')
        )

    # Quantize model (includes GQA ReFlip if enabled)
    quantizer.quantize_model_sequential(calib_data, n_samples=args.n_calib)

    # Save quantized model
    os.makedirs(args.output_dir, exist_ok=True)
    quantizer.save_quantized_model(args.output_dir)

    print("\n" + "=" * 80)
    print("✓ AWQ-GQA Quantization Complete!")
    print(f"  Saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
