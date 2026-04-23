"""
Unified Expectation-Guided Bias Correction (U-EGBC)

Extends EGBC's James-Stein Heuristic AWQ (XL version) to a unified
bias-correction framework that handles:

  (A) Linear projections  y = W x          (the original EGBC instance)
  (B) Attention scores    s = x^T W_Q^T W_K x   (the new QK instance)

Both instances are solved by the same batched subset-sum flip selection,
parameterized by a layer-specific "effective activation" statistic:

  Instance A:  mu_tilde_j     = E[x_j]                   (shape [d])
  Instance B:  mu_tilde_{ij}  = E[x_j * k^{fp}_{g,i}(x)] (shape [h, d])

Under GQA with ratio r = H_q/H_k, mu_tilde^{(g)} is shared across the
r query heads of key group g -- computed once per key group, reused.

Key algorithmic components retained from EGBC (Document 12):
  * James-Stein shrinkage on first-moment estimates
  * Kneedle-based outlier masking (now generalised to 2D mu_tilde)
  * Prefix-subset-sum greedy selection ordered by rounding deviation
  * Per-bias-unit flip budget

Usage:
    # EGBC-only (reproduces Document 12 behaviour):
    python ubc.py --model-path /models/Llama-3-8B \
                        --output-dir ./models/Llama-3-8B_ubc

    # EGBC + QK instance on top (new unified method):
    python ubc.py --model-path /models/Llama-3-8B \
                        --output-dir ./models/Llama-3-8B_uegbc_full \
                        --apply-qk-correction \
                        --qk-max-flip-pct 0.01
"""

import argparse
import gc
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from awq_js_xl import (
    JamesSteinHeuristicAWQQuantizerXL,
    compute_james_stein_mean,
    find_knee_point,
)

try:
    from calibration_utils import (
        get_c4_calibration_data,
        get_wikitext2_calibration_data,
    )
except ImportError:
    print("⚠ calibration_utils not found; relying on in-script dataset loaders.")


# --------------------------------------------------------------------------- #
# GQA utilities                                                               #
# --------------------------------------------------------------------------- #

def is_gqa_layer(name):
    lname = name.lower()
    keywords = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']
    return any(k in lname for k in keywords)


def get_attention_group(name):
    """Extract (layer_idx, attn_group) from a module name like
    'model.layers.7.self_attn.q_proj' → (7, 'model.layers.7.self_attn')."""
    parts = name.split('.')
    layer_idx = None
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                break
            except ValueError:
                continue
    if layer_idx is None or 'self_attn' not in parts:
        return None
    attn_idx = parts.index('self_attn')
    return layer_idx, '.'.join(parts[:attn_idx + 1])


# --------------------------------------------------------------------------- #
# The unified quantizer                                                       #
# --------------------------------------------------------------------------- #

class UnifiedEGBCQuantizer(JamesSteinHeuristicAWQQuantizerXL):
    """
    U-EGBC = EGBC (linear instance) + QK instance on top.

    The QK instance runs AFTER the linear EGBC pass (which modifies Q and K
    via ordinary first-moment correction), using the already-quantised W_K
    as the fixed "context" for the bilinear score.
    """

    def __init__(self, model, tokenizer, device="cuda",
                 bits=4, n_grid=20, group_size=128,
                 use_heuristic=True, knee_tolerance=0.1,
                 max_tokens_per_sample=512, layer_batch_size=16,
                 lmhead_chunks=4, max_flip_percent=0.05,
                 use_james_stein=True,
                 # --- QK instance parameters ---
                 apply_qk_correction=False,
                 qk_max_flip_pct=0.01,
                 qk_calib_n_tokens=2048,
                 qk_knee_tolerance=0.0):
        super().__init__(
            model=model, tokenizer=tokenizer, device=device,
            bits=bits, n_grid=n_grid, group_size=group_size,
            use_heuristic=use_heuristic, knee_tolerance=knee_tolerance,
            max_tokens_per_sample=max_tokens_per_sample,
            layer_batch_size=layer_batch_size, lmhead_chunks=lmhead_chunks,
            max_flip_percent=max_flip_percent,
            use_james_stein=use_james_stein,
        )
        self.apply_qk_correction = apply_qk_correction
        self.qk_max_flip_pct = qk_max_flip_pct
        self.qk_calib_n_tokens = qk_calib_n_tokens
        self.qk_knee_tolerance = qk_knee_tolerance

        # State needed by the QK instance, populated by the linear pass:
        self._fp_q_weights = {}       # q_name -> FP weight (CPU)
        self._fp_k_weights = {}       # k_name -> FP weight (CPU)
        self._q_int_weights = {}      # q_name -> heuristic int weights (CPU, uint8)
        self._q_scales = {}           # q_name -> group quant scales (CPU)
        self._q_zps = {}              # q_name -> group zero points (CPU)
        self._q_awq_scales = {}       # q_name -> AWQ per-channel scales (CPU)
        self._k_awq_scales = {}       # k_name -> AWQ per-channel scales (CPU)

        # QK-instance effective activation: mu_tilde^{(g)}_{ij} = E[x_j * k^fp_{g,i}(x)]
        # Keyed by k_name → tensor [H_k * h, d] (flattened; un-scaled domain)
        self._qk_mu_tilde = {}

        print(f"  U-EGBC: QK correction: "
              f"{'ENABLED' if apply_qk_correction else 'DISABLED'}")
        if apply_qk_correction:
            print(f"    QK max flip %: {qk_max_flip_pct*100:.2f}%")
            print(f"    QK calibration tokens: {qk_calib_n_tokens}")
            print(f"    QK knee tolerance: {qk_knee_tolerance}")

    # ==================================================================== #
    # Override quantize_layer to cache what the QK instance will need.      #
    # For non-GQA layers, behaviour is identical to parent class.           #
    # For GQA layers, we cache the integer weights + scales + AWQ scales.   #
    # ==================================================================== #

    @torch.no_grad()
    def quantize_layer(self, name, module):
        if not (self.apply_qk_correction and is_gqa_layer(name)):
            # Plain EGBC path (unchanged).
            super().quantize_layer(name, module)
            return

        # GQA layer: run EGBC as usual, but also stash the pieces the QK
        # instance will need. We cannot read them back after the parent has
        # written the dequantised weights back to module.weight, so we do
        # the computation ourselves in parallel with the parent's logic.

        best_scales, best_alpha, best_error = self.search_best_scale(name, module)
        W = module.weight.data
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        _, js_mean = self.get_activation_stats(name)
        if js_mean is not None:
            scaled_act_mean = js_mean.to(self.device).to(W.dtype) / best_scales
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device,
                                          dtype=W.dtype)

        # Quantise + apply EGBC heuristic flips AND return the int weights.
        W_dequant_scaled, scales, zps, W_int = \
            self._quantize_with_int_output(W_scaled, scaled_act_mean,
                                           apply_heuristic=self.use_heuristic)

        # Cache everything the QK instance will need (CPU).
        # Note: FP weights are the *original* module weights before anything
        # we did here touched them. We already read W = module.weight.data
        # at the top; it's still the FP value at this point.
        key = name + '.weight'
        if 'q_proj' in name.lower() or 'query' in name.lower():
            self._fp_q_weights[name] = W.detach().clone().cpu().float()
            self._q_int_weights[name] = W_int.cpu()
            self._q_scales[name] = scales.cpu()
            self._q_zps[name] = zps.cpu()
            self._q_awq_scales[name] = best_scales.cpu()
        elif 'k_proj' in name.lower() or 'key' in name.lower():
            self._fp_k_weights[name] = W.detach().clone().cpu().float()
            self._k_awq_scales[name] = best_scales.cpu()
        # v_proj: we don't need it for QK correction.

        # Write the EGBC-quantised result back to the module (same as parent).
        W_final = (W_dequant_scaled / best_scales.unsqueeze(0)).to(original_dtype)
        module.weight.data = W_final

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
        }

        # Cleanup.
        del best_scales, scaled_act_mean, W_scaled, W_dequant_scaled, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def _quantize_with_int_output(self, W, group_activation_means, apply_heuristic=True):
        """
        Mirrors parent.quantize_weight_heuristic_groupwise but also returns
        the integer-quantised weights, per-group scale, per-group zero-point.

        Returns: (W_dequant [out, in], scale [out, n_groups],
                  zp [out, n_groups], W_int [out, in] uint8)
        """
        out_features, in_features = W.shape
        device = W.device

        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in = n_groups * self.group_size

        if padded_in > in_features:
            W_pad = torch.zeros(out_features, padded_in, device=device, dtype=W.dtype)
            W_pad[:, :in_features] = W
            act_pad = torch.zeros(padded_in, device=device, dtype=W.dtype)
            act_pad[:in_features] = group_activation_means
        else:
            W_pad = W
            act_pad = group_activation_means

        W_g = W_pad.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2 ** self.bits - 1

        scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in)

        W_div = W_pad / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)

        if apply_heuristic:
            W_quant = (W_int - zp_flat) * scale_flat
            W_diff = W_pad - W_quant
            current_error = (W_diff * act_pad.unsqueeze(0)).sum(dim=1)

            flip_dir = torch.sign(W_div + zp_flat - W_int)
            flip_dir[flip_dir == 0] = 1.0
            flip_impacts = act_pad.unsqueeze(0) * flip_dir * scale_flat

            target_sign = torch.sign(current_error).unsqueeze(1)
            valid = (torch.sign(flip_impacts) == target_sign)

            w_int_prop = W_int + flip_dir
            valid = valid & (w_int_prop >= 0) & (w_int_prop <= max_int)

            outlier_thresh, _ = self.compute_dynamic_outlier_threshold(act_pad)
            is_outlier = act_pad.abs() > outlier_thresh
            valid = valid & (~is_outlier).unsqueeze(0)

            rounding_costs = (W_div + zp_flat - W_int).abs()
            rounding_costs_masked = rounding_costs.clone()
            rounding_costs_masked[~valid] = -1.0

            sorted_idx = torch.argsort(rounding_costs_masked, dim=1, descending=True)
            sorted_imp = torch.gather(flip_impacts, 1, sorted_idx)
            sorted_val = torch.gather(valid.long(), 1, sorted_idx)
            sorted_imp = sorted_imp * sorted_val

            cumsum_imp = torch.cumsum(sorted_imp, dim=1)
            residuals = torch.abs(current_error.unsqueeze(1) - cumsum_imp)
            residuals = torch.cat([torch.abs(current_error).unsqueeze(1), residuals], dim=1)
            best_k = torch.argmin(residuals, dim=1)

            idx_range = torch.arange(padded_in, device=device).unsqueeze(0)
            flip_mask_sorted = idx_range < best_k.unsqueeze(1)
            final_flips = flip_mask_sorted & sorted_val.bool()

            sorted_flip_dir = torch.gather(flip_dir, 1, sorted_idx)
            sorted_flip_dir[~final_flips] = 0.0

            max_flips = int(self.max_flip_percent * in_features)
            cumcnt = final_flips.long().cumsum(dim=1)
            sorted_flip_dir[cumcnt > max_flips] = 0.0

            W_int.scatter_add_(1, sorted_idx, sorted_flip_dir)
            W_int.clamp_(0, max_int)

            # --- Linear EGBC error-reduction debug ---
            W_quant_after = (W_int - zp_flat) * scale_flat
            error_after = ((W_pad - W_quant_after) * act_pad.unsqueeze(0)).sum(dim=1)
            abs_before_lin = current_error.abs()
            abs_after_lin  = error_after.abs()
            mean_b = abs_before_lin.mean().item()
            mean_a = abs_after_lin.mean().item()
            pct = (1.0 - mean_a / max(mean_b, 1e-12)) * 100.0
            rows_improved = (abs_after_lin < abs_before_lin).sum().item()
            n_flips_lin = sorted_flip_dir.ne(0).sum().item()
            print(f"      [Lin] rows={out_features}  flips={n_flips_lin}  "
                  f"|err|: {mean_b:.4e} → {mean_a:.4e} ({pct:+.1f}%)  "
                  f"rows_improved={rows_improved}/{out_features}  "
                  f"max|err|: {abs_before_lin.max().item():.4e} → "
                  f"{abs_after_lin.max().item():.4e}")

        W_dequant = (W_int - zp_flat) * scale_flat
        if padded_in > in_features:
            W_dequant = W_dequant[:, :in_features]
            W_int = W_int[:, :in_features]

        return (W_dequant.to(W.dtype),
                scale.squeeze(-1),   # [out, n_groups]
                zp.squeeze(-1),      # [out, n_groups]
                W_int.to(torch.uint8))  # [out, in]

    # ==================================================================== #
    # Pipeline: EGBC first (per-layer), QK instance as a post-processing.   #
    # ==================================================================== #

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        # Step 1: standard EGBC pass over every linear layer.
        super().quantize_model_sequential(calibration_data, n_samples)

        if not self.apply_qk_correction:
            return

        # Step 2: accumulate QK effective-activation statistic over calibration.
        self._accumulate_qk_effective_activation(
            calibration_data, n_tokens=self.qk_calib_n_tokens,
        )

        # Step 3: apply QK instance of U-EGBC to every Q projection.
        self._apply_qk_correction()

        # Cleanup.
        self._fp_q_weights.clear()
        self._fp_k_weights.clear()
        self._q_int_weights.clear()
        self._q_scales.clear()
        self._q_zps.clear()
        self._q_awq_scales.clear()
        self._k_awq_scales.clear()
        self._qk_mu_tilde.clear()
        torch.cuda.empty_cache()
        gc.collect()

    # ==================================================================== #
    # QK instance: effective-activation accumulation.                       #
    # ==================================================================== #

    @torch.no_grad()
    def _accumulate_qk_effective_activation(self, calibration_data, n_tokens):
        """
        Compute mu_tilde^{(g)}_{ij} = E[x_j * k^{fp}_{g,i}(x)] for every
        attention block, by streaming the calibration set once.

        The accumulator is (sum_n x_j * u_i) / N  where u_i = <W_K^fp_i, x>.
        Implemented as  u.T @ x  summed online.

        Stored as self._qk_mu_tilde[k_name] of shape [H_k * h, d], un-scaled.
        """
        print("\n" + "=" * 80)
        print("U-EGBC Step 2: Accumulating QK effective activation")
        print(f"  Target tokens: ~{n_tokens}")
        print("=" * 80)

        # Find all k_proj modules and their matching FP weights.
        k_modules = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if 'k_proj' in name.lower() or 'key' in name.lower():
                if name in self._fp_k_weights:
                    k_modules[name] = module

        if not k_modules:
            print("  No k_proj modules found with cached FP weights. Skipping.")
            return

        # Pre-allocate accumulators on each module's device.
        accum = {}
        token_count = {}
        for name, module in k_modules.items():
            fp = self._fp_k_weights[name]  # [k_out, d] fp32
            accum[name] = torch.zeros_like(fp, device=module.weight.device)
            token_count[name] = 0

        # Hook: for each token batch x, compute u = x @ W_K^fp.T, then
        # update accum[name] += u.T @ x (shape [k_out, d]).
        handles = []

        def make_hook(name):
            W_fp = self._fp_k_weights[name]

            def hook(module, inputs, output):
                x = inputs[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                x = x.to(torch.float32).to(module.weight.device)
                W_fp_dev = W_fp.to(module.weight.device)  # fp32
                u = x @ W_fp_dev.T  # [N, k_out]
                # accum[name] += u.T @ x  == sum_n u_i * x_j per (i, j)
                accum[name].add_(u.T @ x)
                token_count[name] += x.shape[0]

            return hook

        for name, module in k_modules.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        # Stream calibration data through the model.
        self.model.eval()
        tokens_seen = 0
        try:
            for sample in calibration_data:
                if tokens_seen >= n_tokens:
                    break
                if isinstance(sample, str):
                    enc = self.tokenizer(sample, return_tensors='pt',
                                         truncation=True,
                                         max_length=self.max_tokens_per_sample)
                    input_ids = enc['input_ids'].to(self.device)
                else:
                    input_ids = sample.to(self.device)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                self.model(input_ids, use_cache=False)
                tokens_seen += input_ids.numel()
        finally:
            for h in handles:
                h.remove()

        # Normalise and move to CPU.
        for name in list(accum.keys()):
            N = max(token_count[name], 1)
            self._qk_mu_tilde[name] = (accum[name] / N).cpu()

        print(f"  ✓ Accumulated mu_tilde for {len(self._qk_mu_tilde)} "
              f"attention blocks (~{tokens_seen} tokens)")

    # ==================================================================== #
    # QK instance: apply to each attention block.                           #
    # ==================================================================== #

    def _apply_qk_correction(self):
        """Iterate over attention blocks, apply QK correction to each Q."""
        print("\n" + "=" * 80)
        print("U-EGBC Step 3: Applying QK correction to Q projections")
        print("=" * 80)

        # Group Q, K by attention block.
        groups = {}
        for name, module in self.model.named_modules():
            if not (isinstance(module, nn.Linear) and is_gqa_layer(name)):
                continue
            info = get_attention_group(name)
            if info is None:
                continue
            _, gkey = info
            groups.setdefault(gkey, {})
            lname = name.lower()
            if 'q_proj' in lname or 'query' in lname:
                groups[gkey]['q'] = (name, module)
            elif 'k_proj' in lname or 'key' in lname:
                groups[gkey]['k'] = (name, module)

        refined = 0
        for gkey, projs in tqdm(groups.items(), desc="  QK blocks"):
            if 'q' not in projs or 'k' not in projs:
                continue
            try:
                self._correct_q_for_block(projs['q'], projs['k'])
                refined += 1
            except Exception as e:
                print(f"    ! {gkey}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

        print(f"  ✓ Refined {refined}/{len(groups)} attention blocks")
    # ==================================================================== #
    # Head-dim inference for GQA.                                           #
    # ==================================================================== #

    def infer_head_dim(self, k_out):
        """
        Infer head_dim from the K-projection output size.

        Preference order:
          1. model.config.head_dim (LLaMA-3 and newer expose this)
          2. model.config.hidden_size / model.config.num_attention_heads
             (works for most transformer configs)
          3. Largest-divisor heuristic over common head_dim values, checked
             in descending order. Descending is important: for LLaMA-3 K-out
             of 1024, head_dim=128 gives 8 K-heads (correct) while checking
             64 first would give 16 "fake" heads.

        The model config is the source of truth whenever it's available;
        the divisor heuristic is only a fallback.
        """
        cfg = getattr(self.model, "config", None)

        if cfg is not None and getattr(cfg, "head_dim", None) is not None:
            return int(cfg.head_dim)

        if cfg is not None:
            hidden = getattr(cfg, "hidden_size", None)
            n_heads = getattr(cfg, "num_attention_heads", None)
            if hidden is not None and n_heads is not None and n_heads > 0:
                cand = hidden // n_heads
                if cand > 0 and cand <= k_out and k_out % cand == 0:
                    return int(cand)

        # Fallback: check common head_dim values in descending order.
        for hd in (256, 192, 160, 128, 96, 80, 72, 64, 48, 32):
            if k_out % hd == 0:
                return int(hd)

        return int(k_out)
    @torch.no_grad()
    def _correct_q_for_block(self, q_entry, k_entry):
        """Apply the QK instance of U-EGBC to the Q projection of one block."""
        q_name, q_module = q_entry
        k_name, k_module = k_entry

        # Required cached state.
        for d, key, desc in [(self._fp_q_weights, q_name, 'FP Q'),
                              (self._fp_k_weights, k_name, 'FP K'),
                              (self._q_int_weights, q_name, 'int Q'),
                              (self._q_scales, q_name, 'scale Q'),
                              (self._q_zps, q_name, 'zp Q'),
                              (self._q_awq_scales, q_name, 'AWQ scale Q'),
                              (self._qk_mu_tilde, k_name, 'mu_tilde')]:
            if key not in d:
                print(f"    ! {q_name}: missing {desc}, skipping")
                return

        device = q_module.weight.device
        dtype = torch.float32

        # Load everything we need onto the device.
        Wq_fp = self._fp_q_weights[q_name].to(device, dtype)   # [q_out, d]
        Wq_int = self._q_int_weights[q_name].to(device, dtype) # [q_out, d]
        q_scale = self._q_scales[q_name].to(device, dtype)     # [q_out, n_g]
        q_zp = self._q_zps[q_name].to(device, dtype)           # [q_out, n_g]
        sQ = self._q_awq_scales[q_name].to(device, dtype)      # [d]
        mu_tilde = self._qk_mu_tilde[k_name].to(device, dtype) # [k_out, d]

        q_out, d = Wq_fp.shape
        k_out = mu_tilde.shape[0]

        # Infer GQA structure.
        h = self.infer_head_dim(k_out)
        H_k = k_out // h
        H_q = q_out // h
        if H_q % H_k != 0:
            print(f"    ! H_q={H_q} not divisible by H_k={H_k}, skipping")
            return
        r = H_q // H_k

        # ------------- Scaled-domain bookkeeping ------------------------ #
        # EGBC has already applied AWQ scaling: the stored integer weights
        # Wq_int correspond to (Wq_fp * sQ), quantised group-wise in this
        # scaled domain. The activation in that domain is x/sQ, so:
        #     mu_tilde_scaled_{ij}
        #       = E[(x_j / sQ_j) * u_i(x)]       (u_i is FP key coord)
        #       = mu_tilde_{ij} / sQ_j
        mu_tilde_scaled = mu_tilde / sQ.unsqueeze(0)  # [k_out, d]

        # Reshape to [H_k, h, d], then broadcast across the r query heads.
        mu_tilde_per_head = mu_tilde_scaled.view(H_k, h, d)   # [H_k, h, d]

        # Expand per-group scales/zps to [q_out, d] then to [H_k, r, h, d].
        q_scale_full = q_scale.repeat_interleave(self.group_size, dim=-1)[:, :d]
        q_zp_full = q_zp.repeat_interleave(self.group_size, dim=-1)[:, :d]
        q_scale_4d = q_scale_full.view(H_k, r, h, d)
        q_zp_4d = q_zp_full.view(H_k, r, h, d)

        # Integer Q, scaled-domain dequantised Q, and scaled FP Q.
        Wq_int_4d = Wq_int.view(H_k, r, h, d)
        Wq_dequant_scaled = (Wq_int_4d - q_zp_4d) * q_scale_4d    # scaled domain
        Wq_fp_scaled = (Wq_fp * sQ.unsqueeze(0)).view(H_k, r, h, d)

        # Delta (in scaled domain) -- this is what EGBC's flip logic works in.
        delta_scaled = Wq_dequant_scaled - Wq_fp_scaled            # [H_k, r, h, d]

        # ------------- Per-head bias b_{g,ell} -------------------------- #
        # b_{g,ell} = sum_{i,j} Delta(W_Q)_{ij} * mu_tilde_scaled_{g,i,j}
        # Note: mu_tilde depends only on g (key group), not ell.
        b = torch.einsum('grhd,ghd->gr', delta_scaled,
                         mu_tilde_per_head)                        # [H_k, r]

        # ------------- Per-flip impact v_{(g,r,h,d)} -------------------- #
        # A flip of the integer weight at position (i, j) of Q head (g, ell)
        # by delta_{ij} ∈ {-1, +1} changes the scaled-dequantised weight by
        # delta_{ij} * q_scale_4d, and therefore the head bias by
        # delta_{ij} * q_scale_4d * mu_tilde_per_head (broadcast across r).
        mu_tilde_for_v = mu_tilde_per_head.unsqueeze(1).expand(H_k, r, h, d)
        per_flip_impact = q_scale_4d * mu_tilde_for_v              # [H_k, r, h, d]

        # ------------- Call the unified subset-sum core ----------------- #
        # Flatten bias units to [M = H_k*r], candidates to [M, h*d].
        M = H_k * r
        num_cand = h * d
        b_flat = b.reshape(M)                                       # [M]
        impact_flat = per_flip_impact.reshape(M, num_cand)           # [M, hd]
        Wq_int_flat_cand = Wq_int_4d.reshape(M, num_cand)            # [M, hd]

        # Rounding deviation, i.e. distance from the integer to the
        # noisy unrounded value  (Wq_fp / q_scale + q_zp). Larger deviation
        # → flipping incurs smaller local reconstruction penalty.
        Wq_noisy_scaled = Wq_fp_scaled / q_scale_4d + q_zp_4d        # [H_k, r, h, d]
        rounding_dev = (Wq_noisy_scaled - Wq_int_4d).abs().reshape(M, num_cand)

        # Flip direction (which way to move each integer if we flip it):
        # toward the unrounded point, i.e. sign(Wq_noisy - Wq_int). This is
        # the direction that would result from "un-nearest-rounding", i.e.
        # the move with smallest local distortion.
        flip_dir_nearest = torch.sign(Wq_noisy_scaled - Wq_int_4d)
        flip_dir_nearest[flip_dir_nearest == 0] = 1.0
        flip_dir_nearest = flip_dir_nearest.reshape(M, num_cand)

        # Effective-activation outlier mask (Kneedle over |mu_tilde|).
        mu_tilde_abs_flat = mu_tilde_for_v.reshape(M, num_cand).abs()
        outlier_mask = self._qk_outlier_mask(mu_tilde_abs_flat)       # [M, hd]

        # Integer range bounds.
        max_int = 2 ** self.bits - 1

        # Dispatch into the shared primitive.
        flips_flat = self._subset_sum_flip_selection(
            W_int=Wq_int_flat_cand,
            b=b_flat,
            flip_dir_nearest=flip_dir_nearest,
            per_flip_impact=impact_flat,
            rounding_dev=rounding_dev,
            outlier_mask=outlier_mask,
            max_int=max_int,
            flip_budget_frac=self.qk_max_flip_pct,
        )

        # Apply flips to the integer grid, then dequantise, then un-AWQ-scale.
        Wq_int_new_flat = (Wq_int_flat_cand + flips_flat).clamp(0, max_int)
        Wq_int_new = Wq_int_new_flat.view(H_k, r, h, d)
        Wq_dequant_new = (Wq_int_new - q_zp_4d) * q_scale_4d        # scaled
        Wq_out_2d = Wq_dequant_new.reshape(q_out, d) / sQ.unsqueeze(0)  # un-scaled

        n_flips = (flips_flat != 0).sum().item()
        print(f"    {q_name}: H_q={H_q}, H_k={H_k}, r={r}, h={h}  "
              f"flips={n_flips} / {M * num_cand} "
              f"(avg {n_flips / max(M, 1):.1f}/head, cap "
              f"{int(self.qk_max_flip_pct * num_cand)})")

        q_module.weight.data.copy_(Wq_out_2d.to(q_module.weight.dtype))

    # ==================================================================== #
    # The unified subset-sum primitive (used by both instances).            #
    # This is called per-layer for the QK instance; EGBC's linear case is   #
    # handled by the parent's quantize_weight_heuristic_groupwise, which    #
    # is morally the same algorithm restricted to row-independent biases.   #
    # ==================================================================== #

    @torch.no_grad()
    def _subset_sum_flip_selection(
        self,
        W_int,              # [M, N]   current integer weights (fp for arithmetic)
        b,                  # [M]      per-bias-unit residual bias to cancel
        flip_dir_nearest,   # [M, N]   direction of nearest-boundary flip, ±1
        per_flip_impact,    # [M, N]   signed impact on b of flipping by +1
        rounding_dev,       # [M, N]   |frac(W_noisy)| → sort key (desc)
        outlier_mask,       # [M, N]   True where outlier (to be excluded)
        max_int,            # int
        flip_budget_frac,   # float, per-bias-unit budget
    ):
        """
        Unified flip-selection primitive.

        For each bias unit m, chooses a subset of candidate positions to
        flip so as to minimise |b_m - sum_{in subset} v_m|, subject to:
          (i)   sign agreement: flip must reduce |b_m|
          (ii)  integer-range feasibility: W_int + delta in [0, max_int]
          (iii) non-outlier effective activation
          (iv)  per-m budget of floor(flip_budget_frac * N) flips

        Returns: flips [M, N] in {-1, 0, +1}, to be added to W_int.
        """
        M, N = W_int.shape
        device = W_int.device

        # Effective per-flip impact, signed by the nearest-boundary direction.
        v = flip_dir_nearest * per_flip_impact                   # [M, N]

        # (i) sign agreement: flipping reduces |b| iff sign(v) == sign(b).
        target_sign = torch.sign(b).unsqueeze(1)                 # [M, 1]
        sign_ok = torch.sign(v) == target_sign

        # (ii) integer-range feasibility.
        w_prop = W_int + flip_dir_nearest
        range_ok = (w_prop >= 0) & (w_prop <= max_int)

        # (iii) non-outlier.
        non_out = ~outlier_mask

        valid = sign_ok & range_ok & non_out                     # [M, N]

        # Sort candidates by rounding deviation, descending.
        rdev_masked = rounding_dev.clone()
        rdev_masked[~valid] = -1.0
        sorted_idx = torch.argsort(rdev_masked, dim=1, descending=True)

        # Gather impacts and validity in sorted order.
        v_sorted = torch.gather(v, 1, sorted_idx)
        valid_sorted = torch.gather(valid.long(), 1, sorted_idx)
        v_sorted = v_sorted * valid_sorted                        # zeroes invalid

        # Cumulative impact along the sorted candidate list.
        cumsum_v = torch.cumsum(v_sorted, dim=1)                  # [M, N]

        # Residual |b - cumsum| after taking the first K flips, for K = 0..N.
        #   K = 0 corresponds to "take no flips" → residual = |b|.
        residuals = torch.abs(b.unsqueeze(1) - cumsum_v)          # [M, N]
        residuals_aug = torch.cat([torch.abs(b).unsqueeze(1), residuals], dim=1)
        best_K = torch.argmin(residuals_aug, dim=1)                # [M]

        # Which sorted positions are in the chosen prefix?
        idx_range = torch.arange(N, device=device).unsqueeze(0)
        take_sorted = idx_range < best_K.unsqueeze(1)              # [M, N]
        take_sorted = take_sorted & valid_sorted.bool()

        # Apply per-bias-unit budget cap.
        budget = int(flip_budget_frac * N)
        if budget < 1:
            budget = 1
        cumcnt = take_sorted.long().cumsum(dim=1)
        take_sorted = take_sorted & (cumcnt <= budget)

        # Recover flip directions in sorted order, then scatter back.
        flip_dir_sorted = torch.gather(flip_dir_nearest, 1, sorted_idx)
        flip_dir_sorted[~take_sorted] = 0.0

        flips = torch.zeros(M, N, dtype=flip_dir_nearest.dtype, device=device)
        flips.scatter_(1, sorted_idx, flip_dir_sorted)
        return flips

    @torch.no_grad()
    def _qk_outlier_mask(self, mu_tilde_abs):
        """
        Per-row Kneedle mask on the flattened |mu_tilde| of shape [M, N].

        Fully vectorized: no Python-level loop over rows. Semantically
        identical to the original per-row implementation that called
        find_knee_point on each row's first half.
        """
        M, N = mu_tilde_abs.shape
        device = mu_tilde_abs.device
        half = N // 2

        # Batched descending sort (the caller's semantics).
        sorted_vals, sorted_i = torch.sort(mu_tilde_abs, dim=1, descending=True)

        # Global branch: first_half too short for Kneedle (same for every row
        # since it only depends on N).
        if half < 3:
            k_default = max(int(0.05 * N), 1)
            ks = torch.full((M,), k_default, dtype=torch.long, device=device)
        else:
            # first_half is [M, n] with n = half. Cast to fp32 to match the
            # original (which did .cpu().float().numpy()).
            first_half = sorted_vals[:, :half].to(torch.float32)            # [M, n]
            n = half

            # Per-row min / max (keepdim for broadcasting).
            y_min = first_half.min(dim=1, keepdim=True).values              # [M, 1]
            y_max = first_half.max(dim=1, keepdim=True).values              # [M, 1]
            span = y_max - y_min                                            # [M, 1]

            # Rows where span < 1e-10 are treated as "flat" -> knee_idx = n // 2,
            # and the offset branch is skipped (original returns n // 2 directly).
            flat = span.squeeze(1) < 1e-10                                  # [M]

            # Safe normalization (avoid div-by-zero on flat rows; we'll mask
            # them out at the end anyway).
            span_safe = span.clamp_min(1e-10)
            y_norm = (first_half - y_min) / span_safe                       # [M, n]

            # Reference line from (0, y_norm[:, 0]) to (1, y_norm[:, -1]).
            x_norm = torch.linspace(0.0, 1.0, n, device=device,
                                    dtype=torch.float32)                    # [n]
            y0 = y_norm[:, :1]                                              # [M, 1]
            y1 = y_norm[:, -1:]                                             # [M, 1]
            y_line = y0 + (y1 - y0) * x_norm.unsqueeze(0)                   # [M, n]

            distances = (y_norm - y_line).abs()                             # [M, n]
            knee_idx = distances.argmax(dim=1)                              # [M]

            # Offset branch. Original: if knee_idx < n - 1, add
            # int(tolerance_offset * n) and clamp to [0, n-1]. If knee_idx
            # equals n-1, it's left alone.
            offset_indices = int(self.qk_knee_tolerance * n)
            shifted = torch.clamp(knee_idx + offset_indices, 0, n - 1)
            at_end = knee_idx >= n - 1
            ks = torch.where(at_end, knee_idx, shifted)

            # Flat rows: return n // 2 (matches the early-return in the original).
            flat_k = torch.full_like(ks, n // 2)
            ks = torch.where(flat, flat_k, ks)

        # Build the mask: mark the first ks[m] entries of the sort permutation.
        idx_range = torch.arange(N, device=device).unsqueeze(0)             # [1, N]
        sorted_mask = idx_range < ks.unsqueeze(1)                            # [M, N]

        mask = torch.zeros(M, N, dtype=torch.bool, device=device)
        mask.scatter_(1, sorted_i, sorted_mask)
        return mask


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Unified Expectation-Guided Bias Correction "
                    "(EGBC linear instance + QK instance)")

    # Parent-class / EGBC args (same as Document 12).
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--use-heuristic", action="store_true", default=True)
    parser.add_argument("--no-heuristic", dest="use_heuristic",
                        action="store_false")
    parser.add_argument("--use-james-stein", action="store_true", default=True)
    parser.add_argument("--no-james-stein", dest="use_james_stein",
                        action="store_false")
    parser.add_argument("--knee-tolerance", type=float, default=0.000)
    parser.add_argument("--max-flip-percent", type=float, default=0.05)
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=8)
    parser.add_argument("--output-dir", type=str,
                        default="./quantized_models/model_uegbc")
    parser.add_argument("--model-path", type=str,
                        default="./models/Mistral-7B-v0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str,
                        default="./calibration_cache")

    # QK-instance args.
    parser.add_argument("--apply-qk-correction", action="store_true",
                        help="Enable the U-EGBC QK instance on Q projections.")
    parser.add_argument("--qk-max-flip-pct", type=float, default=0.01,
                        help="Per-head flip budget, as fraction of h*d. "
                             "Typical: 0.005–0.02.")
    parser.add_argument("--qk-calib-n-tokens", type=int, default=2048,
                        help="Number of calibration tokens for mu_tilde "
                             "accumulation. Set to 0 to use the full set.")
    parser.add_argument("--qk-knee-tolerance", type=float, default=0.000,
                        help="Kneedle tolerance offset for the QK outlier "
                             "mask.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Unified EGBC (U-EGBC): Linear instance + optional QK instance")
    print(f"  Model:       {args.model_path}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Bits:        {args.bits}")
    print(f"  Group size:  {args.group_size}")
    print(f"  Calibration: {args.n_calib} samples from {args.calib_dataset}")
    print(f"  QK instance: "
          f"{'enabled' if args.apply_qk_correction else 'disabled'}")
    print("=" * 80)

    # Load model + tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Calibration data.
    if args.calib_dataset == "c4":
        calib = get_c4_calibration_data(tokenizer, n_samples=args.n_calib,
                                        seqlen=2048, seed=args.seed,
                                        cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib = [it['text'] for it in ds
                 if len(it['text'].strip()) > 100][:args.n_calib]
    else:
        calib = get_wikitext2_calibration_data(tokenizer,
                                               n_samples=args.n_calib,
                                               seqlen=2048, seed=args.seed,
                                               cache_dir=args.cache_dir)

    # Expand "0" for qk-calib-n-tokens to full budget.
    qk_tokens = (args.qk_calib_n_tokens if args.qk_calib_n_tokens > 0
                 else args.n_calib * args.max_tokens_per_sample)

    quantizer = UnifiedEGBCQuantizer(
        model=model, tokenizer=tokenizer, device=device,
        bits=args.bits, n_grid=args.n_grid, group_size=args.group_size,
        use_heuristic=args.use_heuristic, knee_tolerance=args.knee_tolerance,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        max_flip_percent=args.max_flip_percent,
        use_james_stein=args.use_james_stein,
        apply_qk_correction=args.apply_qk_correction,
        qk_max_flip_pct=args.qk_max_flip_pct,
        qk_calib_n_tokens=qk_tokens,
        qk_knee_tolerance=args.qk_knee_tolerance,
    )

    quantizer.quantize_model_sequential(calib, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✓ Saved U-EGBC model to {args.output_dir}")


if __name__ == "__main__":
    main()