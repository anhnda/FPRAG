"""
AWQ-SNC: AWQ Quantization with Signal-to-Noise Cancellation

Pipeline (analogous to awq_gqa_xl.py):
  1. Linear-SNC: AWQ + standalone SNC correction (M = I case of the SNC
     paper's bilinear framework) on every linear layer. Default behavior.
  2. (Optional, --qk-snc) Bilinear SNC refinement on GQA Q/K projections,
     using partner-weighted alpha_sig / alpha_noi:
       - Q-flips: alpha_sig_j = ((W_K^T mu)_j)^2,
                  alpha_noi_j = (W_K^T Sigma W_K)_{jj}
       - K-flips under GQA: above SUMMED over the gqa_ratio Q heads sharing
                  the K projection (predicted ~sqrt(|H_g|) over-allocation).

SNC vs the CLC/JS-Heuristic this replaces:
  - Scoring (Eq. 9): SNR = |mu_i| sqrt(alpha_sig_j) sqrt(s_j)
                          / sqrt(mu_i^2 + lam alpha_noi_j (Sigma_ii + beta r_i))
  - Flip direction (Eq. 7): d = -sign(g) where
        g_{j,i} = 2 alpha_noi_j (Sigma e_j)_i + 2 alpha_sig_j b_j mu_i
    (CLC used d = -sign(e_{j,i}); the two coincide when the Sigma diagonal
    dominates, but g can flip when the off-diagonal Sigma or the bilinear
    partner term wins.)
  - Benefit filter: keep candidates with |g| s > h s^2, where
        h_{j,i} = alpha_noi_j Sigma_ii + alpha_sig_j mu_i^2.
  - Selection: rank by SNR descending, take top ceil(p * |feasible|);
    within each row apply value-greedy on b_j to enforce G >= 0
    (Proposition 1 of the SNC paper).

Linear-side and QK-side budgets/flip caps are independent parameters.
By default, lm_head is NOT quantized (--skip-lmhead is on).

Usage:
    python awq_snc.py \\
        --model-path ./models/Llama-3-8B \\
        --output-dir ./quantized_models/llama3_awq_snc \\
        --qk-snc       # enable bilinear refinement on Q/K projections
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
import sys

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    def get_c4_calibration_data(*a, **k):
        raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*a, **k):
        raise NotImplementedError("Please provide calibration_utils.py")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_james_stein_mean(raw_means, variance_estimate=None):
    """James-Stein shrinkage estimator for activation means (Eq. 13 of SNC)."""
    p = len(raw_means)
    if p < 3:
        return raw_means
    grand_mean = raw_means.mean()
    deviations = raw_means - grand_mean
    sum_sq_dev = (deviations ** 2).sum()
    if sum_sq_dev < 1e-10:
        return raw_means
    if variance_estimate is None:
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp(min=1e-8)
    shrinkage_factor = ((p - 2) * variance_estimate) / sum_sq_dev
    shrinkage_factor = shrinkage_factor.clamp(0, 1)
    return grand_mean + (1 - shrinkage_factor) * deviations


def is_gqa_layer(layer_name):
    """Detect if a layer is a Q/K/V projection of attention."""
    kw = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']
    return any(k in layer_name.lower() for k in kw)


def is_lmhead_layer(layer_name):
    return 'lm_head' in layer_name.lower() or layer_name.endswith('lm_head')


def get_layer_group(layer_name):
    """Return (layer_idx, attn_group_name) for an attention projection layer."""
    parts = layer_name.split('.')
    layer_idx = None
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                break
            except ValueError:
                continue
    if layer_idx is None:
        return None
    if 'self_attn' in parts:
        ai = parts.index('self_attn')
        return (layer_idx, '.'.join(parts[:ai + 1]))
    return None


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class SNCAWQQuantizer:
    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        bits=4,
        n_grid=20,
        group_size=128,
        use_snc=True,
        max_tokens_per_sample=512,
        layer_batch_size=16,
        lmhead_chunks=4,
        skip_lmhead=True,
        # Linear-side SNC knobs
        max_flip_percent=0.05,
        snr_lambda=1.0,
        snr_beta=1.0,
        snr_budget_p=0.05,
        use_safety_cap=False,
        use_james_stein=True,
        # QK-bilinear SNC (optional)
        apply_qk_snc=False,
        qk_max_flip_percent=0.05,
        qk_snr_budget_p=0.05,
        qk_refine_k=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_snc = use_snc
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.lmhead_chunks = lmhead_chunks
        self.skip_lmhead = skip_lmhead
        self.max_flip_percent = max_flip_percent
        self.snr_lambda = snr_lambda
        self.snr_beta = snr_beta
        self.snr_budget_p = snr_budget_p
        self.use_safety_cap = use_safety_cap
        self.use_james_stein = use_james_stein

        self.apply_qk_snc = apply_qk_snc
        self.qk_max_flip_percent = qk_max_flip_percent
        self.qk_snr_budget_p = qk_snr_budget_p
        self.qk_refine_k = qk_refine_k

        self.activation_data = {}
        self.layer_scales = {}

        # Caches populated during the linear pass when QK refinement is on,
        # then consumed by the bilinear pass. Mirrors awq_gqa_xl.py.
        self.original_state_dict = None
        self.qk_activation_stats = {}
        self.qk_awq_scales = {}

        print("\n[SNC AWQ Quantizer Initialized]")
        print(f"  Bits: {bits},  group_size: {group_size}")
        print(f"  Token subsample/sample: {max_tokens_per_sample}")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Use SNC (linear, M=I): {use_snc}")
        print(f"  Use James-Stein mu: {use_james_stein}")
        print(f"  Skip lm_head: {skip_lmhead}")
        if use_snc:
            print(f"  [linear-SNC] lambda={snr_lambda} beta={snr_beta} "
                  f"p={snr_budget_p} max_flip={max_flip_percent*100:.2f}%")
            print(f"  [linear-SNC] safety cap (Cor. 1(ii)): {use_safety_cap}")
        print(f"  QK bilinear SNC: {apply_qk_snc}")
        if apply_qk_snc:
            print(f"  [qk-SNC]   p={qk_snr_budget_p} "
                  f"max_flip={qk_max_flip_percent*100:.2f}% refine_K={qk_refine_k}")
        if not skip_lmhead:
            print(f"  lm_head chunks: {lmhead_chunks}")

    # ----------------------------------------------------- activation hooks

    def get_hook(self, name):
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                idx = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                idx = idx.sort()[0]
                inp = inp[:, idx, :]
            # fp32 on CPU so cross-products stay accurate.
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_activation_stats(self, name):
        """
        Compute (salience, mu_JS, Sigma, sigma_diag, sigma_rowsum_abs) for one
        linear layer. Sigma is the full d x d activation covariance (needed
        for the gradient indicator g_{j,i} = 2 alpha_noi (Sigma e)_i + ...).
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None, None, None, None, None

        X_list = self.activation_data[name]
        total = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_f = X_list[0].shape[-1]

        sum_x = torch.zeros(in_f, dtype=torch.float32)
        sum_xx = torch.zeros(in_f, in_f, dtype=torch.float32)
        sum_sq = torch.zeros(in_f, dtype=torch.float32)
        for x in X_list:
            xf = x.reshape(-1, x.shape[-1]).float()
            sum_x += xf.sum(dim=0)
            sum_sq += xf.pow(2).sum(dim=0)
            sum_xx += xf.t() @ xf

        N = float(total)
        mu = sum_x / N
        salience = sum_sq / N
        Sigma = sum_xx / N - torch.outer(mu, mu)

        if self.use_james_stein:
            mu = compute_james_stein_mean(mu)

        sigma_diag = torch.diagonal(Sigma).clone().clamp(min=0.0)
        Sigma_abs = Sigma.abs()
        sigma_rowsum = Sigma_abs.sum(dim=1) - torch.diagonal(Sigma_abs)

        return salience, mu, Sigma, sigma_diag, sigma_rowsum

    # ------------------------------------------- core SNC group-wise rounding

    @torch.no_grad()
    def quantize_weight_snc_groupwise(
        self,
        W,
        mu,
        Sigma,
        sigma_diag,
        sigma_rowsum,
        alpha_sig=None,
        alpha_noi=None,
        apply_snc=True,
        max_flip_percent=None,
        snr_budget_p=None,
        debug=False,
    ):
        """
        SNC inner step. Per-channel asymmetric RTN base, then the SNC
        adjacent-level flip pass.

        max_flip_percent / snr_budget_p override per-instance defaults so
        linear vs QK passes can carry different budgets.
        """
        if max_flip_percent is None:
            max_flip_percent = self.max_flip_percent
        if snr_budget_p is None:
            snr_budget_p = self.snr_budget_p

        out_f, in_f = W.shape
        device = W.device
        dtype = W.dtype

        n_groups = (in_f + self.group_size - 1) // self.group_size
        padded_in = n_groups * self.group_size

        if padded_in > in_f:
            W_padded = torch.zeros(out_f, padded_in, device=device, dtype=dtype)
            W_padded[:, :in_f] = W
            mu_padded = torch.zeros(padded_in, device=device, dtype=dtype)
            mu_padded[:in_f] = mu
            sd_padded = torch.zeros(padded_in, device=device, dtype=dtype)
            sd_padded[:in_f] = sigma_diag
            sr_padded = torch.zeros(padded_in, device=device, dtype=dtype)
            sr_padded[:in_f] = sigma_rowsum
        else:
            W_padded = W
            mu_padded = mu
            sd_padded = sigma_diag
            sr_padded = sigma_rowsum

        W_g = W_padded.reshape(out_f, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2 ** self.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_f, padded_in)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_f, padded_in)

        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_quant = (W_int - zp_flat) * scale_flat

        empty_stats = {
            'total': 0, 'per_channel_mean': 0.0, 'per_channel_median': 0.0,
            'per_channel_min': 0, 'per_channel_max': 0, 'per_channel_std': 0.0,
            'per_channel_p25': 0.0, 'per_channel_p75': 0.0, 'per_channel_p90': 0.0,
            'per_channel_p95': 0.0, 'per_channel_p99': 0.0,
            'per_channel_zero_pct': 100.0, 'sign_disagree_pct': 0.0,
        }

        if not apply_snc:
            W_dq = W_quant
            if padded_in > in_f:
                W_dq = W_dq[:, :in_f]
            return W_dq.to(dtype), None, empty_stats

        # ---- SNC ingredients --------------------------------------------
        e_padded = W_quant - W_padded
        if alpha_sig is None:
            alpha_sig = torch.ones(out_f, device=device, dtype=dtype)
        if alpha_noi is None:
            alpha_noi = torch.ones(out_f, device=device, dtype=dtype)
        alpha_sig = alpha_sig.to(device=device, dtype=dtype)
        alpha_noi = alpha_noi.to(device=device, dtype=dtype)

        b = (e_padded * mu_padded.unsqueeze(0)).sum(dim=1)        # [out]

        # Sigma e: defined in original input space (no padding).
        e_orig = e_padded[:, :in_f]
        Sigma_dev = Sigma.to(device=device, dtype=torch.float32)
        Sigma_e = (e_orig.float() @ Sigma_dev).to(dtype)          # Sigma symmetric
        if padded_in > in_f:
            Sigma_e_pad = torch.zeros(out_f, padded_in, device=device, dtype=dtype)
            Sigma_e_pad[:, :in_f] = Sigma_e
        else:
            Sigma_e_pad = Sigma_e

        # Gradient indicator (Eq. 7):
        g = (
            2.0 * alpha_noi.unsqueeze(1) * Sigma_e_pad
            + 2.0 * alpha_sig.unsqueeze(1) * b.unsqueeze(1) * mu_padded.unsqueeze(0)
        )
        d = -torch.sign(g)
        d[d == 0] = 1.0

        W_int_proposed = W_int + d
        in_range = (W_int_proposed >= 0) & (W_int_proposed <= max_int)

        # Benefit filter: linear gain |g|*s vs quadratic self-cost h * s^2.
        h = (
            alpha_noi.unsqueeze(1) * sd_padded.unsqueeze(0)
            + alpha_sig.unsqueeze(1) * mu_padded.unsqueeze(0).pow(2)
        )
        s_flat = scale_flat
        benefit_pass = (g.abs() * s_flat) > (h * s_flat.pow(2))
        feasible = in_range & benefit_pass

        # SNR score (Eq. 9):
        mu_abs = mu_padded.abs()
        sqrt_alpha_sig = alpha_sig.sqrt()
        sqrt_s = s_flat.sqrt()
        snr_num = sqrt_alpha_sig.unsqueeze(1) * mu_abs.unsqueeze(0) * sqrt_s
        denom_inside = (
            mu_padded.unsqueeze(0).pow(2)
            + self.snr_lambda * alpha_noi.unsqueeze(1)
              * (sd_padded.unsqueeze(0) + self.snr_beta * sr_padded.unsqueeze(0))
        )
        denom_inside = denom_inside.clamp(min=1e-12)
        snr = snr_num / denom_inside.sqrt()
        snr_masked = torch.where(feasible, snr, torch.full_like(snr, float('-inf')))

        # Per-row budgets
        n_feasible = feasible.sum(dim=1)
        budget_p_row = torch.ceil(snr_budget_p * n_feasible.float()).long()
        hard_cap = int(max_flip_percent * in_f)
        budget_row = torch.clamp(budget_p_row, max=hard_cap)

        # Sort each row by SNR descending
        sorted_idx = torch.argsort(snr_masked, dim=1, descending=True)
        d_sorted = torch.gather(d, 1, sorted_idx)
        feas_sorted = torch.gather(feasible.long(), 1, sorted_idx)
        scale_sorted = torch.gather(scale_flat, 1, sorted_idx)
        mu_sorted = mu_padded.unsqueeze(0).expand(out_f, -1).gather(1, sorted_idx)

        # v_{j,i} = mu_i * d_{j,i} * s_j (signal-side contribution of one flip)
        v_sorted = mu_sorted * d_sorted * scale_sorted * feas_sorted.to(dtype)
        cumsum_v = torch.cumsum(v_sorted, dim=1)

        # Value-greedy: argmin over k in [0, budget_row] of |b - cumsum_k|.
        b_u = b.unsqueeze(1)
        residuals = torch.abs(b_u - cumsum_v)
        residuals_full = torch.cat([b.abs().unsqueeze(1), residuals], dim=1)

        idx_range = torch.arange(padded_in + 1, device=device).unsqueeze(0)
        beyond = idx_range > budget_row.unsqueeze(1)
        residuals_full = residuals_full.masked_fill(beyond, float('inf'))
        k_star = torch.argmin(residuals_full, dim=1)

        flip_mask_sorted = idx_range[:, :-1] < k_star.unsqueeze(1)
        flip_mask_sorted = flip_mask_sorted & (feas_sorted.bool())
        d_apply_sorted = d_sorted * flip_mask_sorted.to(dtype)

        # Optional per-channel safety cap from Corollary 1(ii).
        if self.use_safety_cap:
            with torch.no_grad():
                e_sorted = torch.gather(e_padded, 1, sorted_idx)
                sd_sorted = sd_padded.unsqueeze(0).expand(out_f, -1).gather(1, sorted_idx)
                gamma_terms = scale_sorted * sd_sorted * e_sorted.abs()
                gamma_terms_m = gamma_terms.masked_fill(~flip_mask_sorted, float('inf'))
                gamma_j = gamma_terms_m.min(dim=1).values
                rho_Sigma = sr_padded.max()
                Sig_op_proxy = sd_padded.max()
                s_max_sq = scale_sorted.max(dim=1).values.pow(2)
                Phi_j = -2.0 * gamma_j + s_max_sq * (rho_Sigma + Sig_op_proxy)
                kstar_m = (k_star - 1).clamp(min=0)
                b_prime = torch.gather(cumsum_v, 1, kstar_m.unsqueeze(1)).squeeze(1)
                b_prime = torch.where(k_star == 0, torch.zeros_like(b_prime), b_prime)
                G_j = alpha_sig * (b.pow(2) - (b - b_prime).pow(2))
                bad = Phi_j > 0
                if bad.any():
                    realized_count = flip_mask_sorted.sum(dim=1).clamp(min=1)
                    cap = torch.where(bad, (G_j / Phi_j.clamp(min=1e-12)).floor().long(),
                                      torch.full_like(realized_count, padded_in))
                    cap = cap.clamp(min=0)
                    new_count = torch.minimum(realized_count, cap)
                    flip_mask_sorted = idx_range[:, :-1] < new_count.unsqueeze(1)
                    flip_mask_sorted = flip_mask_sorted & (feas_sorted.bool())
                    d_apply_sorted = d_sorted * flip_mask_sorted.to(dtype)

        W_int.scatter_add_(1, sorted_idx, d_apply_sorted)
        W_int.clamp_(0, max_int)

        # Statistics
        flip_mask_unsorted = torch.zeros_like(d, dtype=torch.bool)
        flip_mask_unsorted.scatter_(1, sorted_idx, flip_mask_sorted)
        flips_per_input = flip_mask_unsorted.sum(dim=0).float()
        if padded_in > in_f:
            flips_per_input = flips_per_input[:in_f]
        n_flips = int(flip_mask_unsorted.sum().item())

        if n_flips > 0:
            disagree = ((torch.sign(g) != torch.sign(-e_padded)) & flip_mask_unsorted).sum().item()
            sign_dis = 100.0 * disagree / max(n_flips, 1)
        else:
            sign_dis = 0.0

        stats = {
            'total': n_flips,
            'per_channel_mean': flips_per_input.mean().item(),
            'per_channel_median': flips_per_input.median().item(),
            'per_channel_min': int(flips_per_input.min().item()),
            'per_channel_max': int(flips_per_input.max().item()),
            'per_channel_std': flips_per_input.std().item(),
            'per_channel_p25': torch.quantile(flips_per_input, 0.25).item(),
            'per_channel_p75': torch.quantile(flips_per_input, 0.75).item(),
            'per_channel_p90': torch.quantile(flips_per_input, 0.90).item(),
            'per_channel_p95': torch.quantile(flips_per_input, 0.95).item(),
            'per_channel_p99': torch.quantile(flips_per_input, 0.99).item(),
            'per_channel_zero_pct': (flips_per_input == 0).float().mean().item() * 100.0,
            'sign_disagree_pct': sign_dis,
        }

        if debug:
            print(f"    [SNC] feasible/row mean={n_feasible.float().mean():.1f}/{padded_in}")
            print(f"    [SNC] budget/row mean={budget_row.float().mean():.1f}, "
                  f"realized flips={n_flips}, sign-disagree vs CLC = {sign_dis:.2f}%")

        W_dq = (W_int - zp_flat) * scale_flat
        if padded_in > in_f:
            W_dq = W_dq[:, :in_f]
        feas_in = feasible.sum(dim=0)
        outlier_frac = (feas_in == 0).float().mean().item()

        return W_dq.to(dtype), outlier_frac, stats

    # ----------------------------------------------- AWQ grid search for alpha

    @torch.no_grad()
    def _rescale_sigma_for_awq(self, Sigma, scales):
        """Sigma_scaled[i,j] = Sigma[i,j] / (s_i s_j)."""
        inv_s = (1.0 / scales).detach().cpu().to(Sigma.dtype)
        return Sigma * inv_s.unsqueeze(0) * inv_s.unsqueeze(1)

    @torch.no_grad()
    def search_best_scale(self, name, module):
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_f = module.weight.shape[1]
            return torch.ones(in_f).to(self.device), 0.0, 0.0

        salience, mu, Sigma, sigma_diag, sigma_rowsum = self.get_activation_stats(name)
        if salience is None:
            in_f = module.weight.shape[1]
            return torch.ones(in_f).to(self.device), 0.0, 0.0

        dtype = module.weight.dtype
        salience = salience.to(self.device).to(dtype)
        mu = mu.to(self.device).to(dtype)

        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)
        max_s = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_s:
            idx = torch.randperm(X_cpu.shape[0])[:max_s]
            X_search = X_cpu[idx].to(self.device)
        else:
            X_search = X_cpu.to(self.device)
        del X_cpu
        if X_search.dtype != dtype:
            X_search = X_search.to(dtype)

        W = module.weight.data
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)
        salience = salience.clamp(min=1e-5)

        for gi in range(self.n_grid + 1):
            alpha = gi / self.n_grid
            scales = salience.pow(alpha)
            W_scaled = W * scales.unsqueeze(0)
            scaled_mu = mu / scales
            Sigma_scaled = self._rescale_sigma_for_awq(Sigma, scales)
            Sigma_abs = Sigma_scaled.abs()
            scaled_sd = torch.diagonal(Sigma_scaled).clamp(min=0.0).to(self.device).to(dtype)
            scaled_sr = (Sigma_abs.sum(dim=1) - torch.diagonal(Sigma_abs)).to(self.device).to(dtype)

            W_q, _, _ = self.quantize_weight_snc_groupwise(
                W_scaled, scaled_mu, Sigma_scaled, scaled_sd, scaled_sr,
                alpha_sig=None, alpha_noi=None, apply_snc=self.use_snc,
            )
            W_recon = W_q / scales.unsqueeze(0)
            Y_q = torch.matmul(X_search, W_recon.t())
            err = (Y_orig - Y_q).pow(2).mean().item()
            if err < best_error:
                best_error = err
                best_alpha = alpha
                best_scales = scales.clone()
            del W_scaled, W_q, W_recon, Y_q, scales, Sigma_scaled

        del X_search, Y_orig
        torch.cuda.empty_cache()
        return best_scales, best_alpha, best_error

    # ----------------------------------------------- quantize_layer (linear-SNC)

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Quantize a single linear layer with AWQ + standalone-linear SNC.
        If QK refinement is enabled and this layer is a Q/K/V projection,
        cache the FP weights, activation stats, and AWQ scales for use by
        the bilinear pass.
        """
        cache_for_qk = self.apply_qk_snc and is_gqa_layer(name)

        # Capture FP weights BEFORE quantization if we'll need them for QK.
        if cache_for_qk:
            if self.original_state_dict is None:
                self.original_state_dict = {}
            self.original_state_dict[name + '.weight'] = module.weight.data.detach().cpu().clone()

        best_scales, best_alpha, best_error = self.search_best_scale(name, module)
        W = module.weight.data
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        _, mu, Sigma, sigma_diag, sigma_rowsum = self.get_activation_stats(name)
        if mu is None:
            in_f = W.shape[1]
            mu = torch.zeros(in_f, dtype=torch.float32)
            Sigma = torch.zeros(in_f, in_f, dtype=torch.float32)
            sigma_diag = torch.zeros(in_f, dtype=torch.float32)
            sigma_rowsum = torch.zeros(in_f, dtype=torch.float32)

        mu_dev = (mu.to(self.device).to(original_dtype) / best_scales)
        Sigma_scaled = self._rescale_sigma_for_awq(Sigma, best_scales)
        Sigma_abs = Sigma_scaled.abs()
        sd_dev = torch.diagonal(Sigma_scaled).clamp(min=0.0).to(self.device).to(original_dtype)
        sr_dev = (Sigma_abs.sum(dim=1) - torch.diagonal(Sigma_abs)).to(self.device).to(original_dtype)

        W_q, outlier, stats = self.quantize_weight_snc_groupwise(
            W_scaled, mu_dev, Sigma_scaled, sd_dev, sr_dev,
            alpha_sig=None, alpha_noi=None, apply_snc=self.use_snc,
        )

        if cache_for_qk:
            # Save original-input-space mu and Sigma for the bilinear pass.
            self.qk_activation_stats[name] = (
                mu.detach().cpu(),
                Sigma.detach().cpu(),
                sigma_diag.detach().cpu(),
                sigma_rowsum.detach().cpu(),
            )
            self.qk_awq_scales[name + '.weight'] = best_scales.detach().cpu()

        W_final = (W_q / best_scales.unsqueeze(0)).to(original_dtype)
        module.weight.data = W_final

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'outlier_percent': outlier if outlier is not None else 0.0,
            'flip_stats': stats,
        }

        del best_scales, mu_dev, Sigma_scaled, W_scaled, W_q, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------ lm_head chunked path

    @torch.no_grad()
    def search_best_scale_lmhead_chunk(self, name, module, out_start, out_end, debug=False):
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_f = module.weight.shape[1]
            return torch.ones(in_f).to(self.device), 0.0, 0.0

        salience, mu, Sigma, sigma_diag, sigma_rowsum = self.get_activation_stats(name)
        if salience is None:
            in_f = module.weight.shape[1]
            return torch.ones(in_f).to(self.device), 0.0, 0.0

        dtype = module.weight.dtype
        salience = salience.to(self.device).to(dtype)
        mu = mu.to(self.device).to(dtype)

        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)
        max_s = min(1024, X_cpu.shape[0])
        if X_cpu.shape[0] > max_s:
            idx = torch.randperm(X_cpu.shape[0])[:max_s]
            X_search = X_cpu[idx].to(self.device)
        else:
            X_search = X_cpu.to(self.device)
        del X_cpu
        if X_search.dtype != dtype:
            X_search = X_search.to(dtype)

        W_full = module.weight.data
        W = W_full[out_start:out_end, :]
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)
        salience = salience.clamp(min=1e-5)

        for gi in range(self.n_grid + 1):
            alpha = gi / self.n_grid
            scales = salience.pow(alpha)
            W_scaled = W * scales.unsqueeze(0)
            scaled_mu = mu / scales
            Sigma_scaled = self._rescale_sigma_for_awq(Sigma, scales)
            Sigma_abs = Sigma_scaled.abs()
            scaled_sd = torch.diagonal(Sigma_scaled).clamp(min=0.0).to(self.device).to(dtype)
            scaled_sr = (Sigma_abs.sum(dim=1) - torch.diagonal(Sigma_abs)).to(self.device).to(dtype)

            W_q, _, _ = self.quantize_weight_snc_groupwise(
                W_scaled, scaled_mu, Sigma_scaled, scaled_sd, scaled_sr,
                alpha_sig=None, alpha_noi=None, apply_snc=self.use_snc,
                debug=(debug and gi == 0),
            )
            W_recon = W_q / scales.unsqueeze(0)
            Y_q = torch.matmul(X_search, W_recon.t())
            err = (Y_orig - Y_q).pow(2).mean().item()
            if err < best_error:
                best_error, best_alpha = err, alpha
                best_scales = scales.clone()
            del W_scaled, W_q, W_recon, Y_q, scales, Sigma_scaled

        del X_search, Y_orig, W
        torch.cuda.empty_cache()
        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_lmhead_chunked(self, name, module, debug=False, num_chunks=4):
        print(f"\n  Special handling for {name} (split into {num_chunks} chunks)")
        W = module.weight.data
        original_dtype = W.dtype
        out_f, in_f = W.shape
        print(f"     Shape: {W.shape} ({W.numel() / 1e6:.1f}M params)")

        _, mu, Sigma, sigma_diag, sigma_rowsum = self.get_activation_stats(name)
        if mu is None:
            mu = torch.zeros(in_f, dtype=torch.float32)
            Sigma = torch.zeros(in_f, in_f, dtype=torch.float32)
            sigma_diag = torch.zeros(in_f, dtype=torch.float32)
            sigma_rowsum = torch.zeros(in_f, dtype=torch.float32)

        cs = out_f // num_chunks
        bounds = [(i * cs, out_f if i == num_chunks - 1 else (i + 1) * cs)
                  for i in range(num_chunks)]

        W_chunks = []
        chunk_stats = []
        for ci, (a, b) in enumerate(bounds):
            print(f"     Chunk {ci+1}/{num_chunks}: rows {a}-{b}")
            best_scales, best_alpha, best_error = self.search_best_scale_lmhead_chunk(
                name, module, a, b, debug=(debug and ci == 0)
            )
            W_chunk = W[a:b, :]
            W_scaled = W_chunk * best_scales.unsqueeze(0)
            scaled_mu = (mu.to(self.device).to(original_dtype) / best_scales)
            Sigma_scaled = self._rescale_sigma_for_awq(Sigma, best_scales)
            Sigma_abs = Sigma_scaled.abs()
            scaled_sd = torch.diagonal(Sigma_scaled).clamp(min=0.0).to(self.device).to(original_dtype)
            scaled_sr = (Sigma_abs.sum(dim=1) - torch.diagonal(Sigma_abs)).to(self.device).to(original_dtype)

            W_q, outlier, stats = self.quantize_weight_snc_groupwise(
                W_scaled, scaled_mu, Sigma_scaled, scaled_sd, scaled_sr,
                alpha_sig=None, alpha_noi=None, apply_snc=self.use_snc,
            )
            W_fin_chunk = (W_q / best_scales.unsqueeze(0)).to(original_dtype)
            W_chunks.append(W_fin_chunk)
            chunk_stats.append({
                'alpha': best_alpha, 'error': best_error, 'scales': best_scales,
                'outlier_percent': outlier if outlier is not None else 0.0,
                'flip_stats': stats,
            })
            del W_chunk, W_scaled, W_q, W_fin_chunk, scaled_mu, Sigma_scaled
            torch.cuda.empty_cache()

        W_final = torch.cat(W_chunks, dim=0)
        module.weight.data = W_final

        avg_alpha = float(np.mean([s['alpha'] for s in chunk_stats]))
        avg_error = float(np.mean([s['error'] for s in chunk_stats]))
        avg_outlier = float(np.mean([s['outlier_percent'] for s in chunk_stats]))
        total_flips = sum(s['flip_stats']['total'] for s in chunk_stats)
        keys = ['per_channel_mean', 'per_channel_median', 'per_channel_std',
                'per_channel_p25', 'per_channel_p75', 'per_channel_p90',
                'per_channel_p95', 'per_channel_p99', 'per_channel_zero_pct',
                'sign_disagree_pct']
        agg = {k: float(np.mean([cs['flip_stats'][k] for cs in chunk_stats])) for k in keys}
        agg['total'] = int(total_flips)

        self.layer_scales[name] = {
            'scales': chunk_stats[0]['scales'].cpu(),
            'alpha': avg_alpha, 'error': avg_error,
            'outlier_percent': avg_outlier, 'flip_stats': agg,
        }
        print(f"     Done: alphas={[round(s['alpha'],3) for s in chunk_stats]}, "
              f"flips={total_flips}")
        del W_chunks, chunk_stats, W_final
        torch.cuda.empty_cache()

    # --------------------------------------------- batched sequential pass

    def calibrate_layer_batch(self, batch, calibration_data, n_samples=500):
        print(f"  Calibrating {len(batch)} layers...")
        self.model.eval()
        handles = []
        for name, module in batch:
            h = module.register_forward_hook(self.get_hook(name))
            handles.append((name, h))

        ok = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="  Calibration", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                            truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    ok += 1
                    if (ok + 1) % 32 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for _, h in handles:
            h.remove()
        torch.cuda.empty_cache()
        gc.collect()

    def _run_linear_pass(self, calibration_data, n_samples):
        """Linear-SNC pass over all (non-lm_head if skip_lmhead) layers."""
        print("\n" + "=" * 80)
        print("[Step 1] Linear-SNC (AWQ + SNC standalone, M = I)")
        print("=" * 80)
        if HAS_PSUTIL:
            print(f"  Initial RAM: {psutil.virtual_memory().percent:.1f}%")

        all_layers = [(n, m) for n, m in self.model.named_modules() if isinstance(m, nn.Linear)]
        if self.skip_lmhead:
            skipped = [n for n, _ in all_layers if is_lmhead_layer(n)]
            layer_names = [(n, m) for n, m in all_layers if not is_lmhead_layer(n)]
            if skipped:
                print(f"  Skipping lm_head layers ({len(skipped)}): {skipped}")
        else:
            layer_names = all_layers

        n_layers = len(layer_names)
        n_batches = (n_layers + self.layer_batch_size - 1) // self.layer_batch_size
        print(f"  Total layers to quantize: {n_layers}, batches: {n_batches}")
        print("=" * 80)

        quantized = 0
        for b_idx in range(n_batches):
            bs = b_idx * self.layer_batch_size
            be = min(bs + self.layer_batch_size, n_layers)
            batch = layer_names[bs:be]
            print(f"\n[Batch {b_idx + 1}/{n_batches}] Layers {bs}-{be - 1}")
            self.calibrate_layer_batch(batch, calibration_data, n_samples)

            print(f"  Quantizing {len(batch)} layers...")
            for name, module in tqdm(batch, desc="  Quantization", leave=False):
                try:
                    if is_lmhead_layer(name):
                        # Only reached if skip_lmhead is False.
                        self.quantize_lmhead_chunked(name, module,
                                                     debug=(quantized < 2),
                                                     num_chunks=self.lmhead_chunks)
                    else:
                        self.quantize_layer(name, module)
                    quantized += 1
                except Exception as ex:
                    print(f"\nError on {name}: {ex}")
                    import traceback; traceback.print_exc()
                    continue

            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()
            if HAS_PSUTIL:
                print(f"  Batch {b_idx + 1} done. RAM: {psutil.virtual_memory().percent:.1f}%")

        print("\n[Step 1] Linear-SNC done.")
        return quantized, n_layers

    # ------------------------------------- QK bilinear refinement (optional)

    def infer_head_dim(self):
        """Use model config when available; otherwise probe a K projection."""
        cfg = getattr(self.model, 'config', None)
        if cfg is not None and hasattr(cfg, 'head_dim') and cfg.head_dim is not None:
            return cfg.head_dim
        if cfg is not None and hasattr(cfg, 'hidden_size') and hasattr(cfg, 'num_attention_heads'):
            try:
                return cfg.hidden_size // cfg.num_attention_heads
            except Exception:
                pass
        # Fallback: scan a K projection. Descending order so 128 beats 64 for Llama-3.
        for hd in [256, 128, 96, 80, 64]:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and is_gqa_layer(name) and 'k_proj' in name.lower():
                    if module.weight.shape[0] % hd == 0:
                        return hd
                    break
        return 128

    @torch.no_grad()
    def _apply_qk_snc_to_attn_group(self, attn_group, projs, head_dim):
        """
        Apply bilinear SNC refinement to one attention block.

        For Q-flips, partner = W_K. For each Q output channel j (head h,
        head-dim slot d):
            alpha_sig_j = ((W_K^(g(h)) mu)_d)^2
            alpha_noi_j = (W_K^(g(h)) Sigma W_K^(g(h))^T)_{d,d}
        For K-flips under GQA, aggregate over the gqa_ratio Q heads sharing
        the K projection:
            alpha_sig_j = sum_{h in H_g} ((W_Q^(h) mu)_d)^2
            alpha_noi_j = sum_{h in H_g} (W_Q^(h) Sigma W_Q^(h)^T)_{d,d}
        """
        if 'q_proj' not in projs or 'k_proj' not in projs:
            print(f"      [QK-SNC] {attn_group}: missing Q or K, skipping.")
            return None

        q_name, q_module = projs['q_proj']
        k_name, k_module = projs['k_proj']
        q_key = q_name + '.weight'
        k_key = k_name + '.weight'

        if self.original_state_dict is None:
            print(f"      [QK-SNC] {attn_group}: no FP cache, skipping.")
            return None
        for key in (q_key, k_key):
            if (key not in self.original_state_dict
                or key not in self.qk_awq_scales):
                print(f"      [QK-SNC] {attn_group}: cache missing for {key}, skipping.")
                return None
        if q_name not in self.qk_activation_stats or k_name not in self.qk_activation_stats:
            print(f"      [QK-SNC] {attn_group}: activation stats missing, skipping.")
            return None

        device = q_module.weight.device
        dtype = torch.float32

        # Full-precision Q/K weights (original input space).
        Wq_fp = self.original_state_dict[q_key].to(device=device, dtype=dtype)
        Wk_fp = self.original_state_dict[k_key].to(device=device, dtype=dtype)

        # AWQ per-input-channel scales (applied to the input axis).
        sQ = self.qk_awq_scales[q_key].to(device=device, dtype=dtype)
        sK = self.qk_awq_scales[k_key].to(device=device, dtype=dtype)

        # Working (AWQ-scaled) weights -- the residual e lives in this space.
        Wq_scaled = Wq_fp * sQ.unsqueeze(0)
        Wk_scaled = Wk_fp * sK.unsqueeze(0)

        # Activation statistics (in original input space). Q and K share the
        # same input (layernorm output), so we use the Q-side stats as the
        # canonical reference.
        mu_q, Sigma_q, _, _ = self.qk_activation_stats[q_name]
        mu = mu_q.to(device=device, dtype=dtype)
        Sigma = Sigma_q.to(device=device, dtype=dtype)

        # Transform statistics into AWQ-scaled space for the bilinear flips.
        # x_orig has E[x]=mu, Cov=Sigma; x_scaled = (1/s) * x_orig.
        mu_s_Q = mu / sQ
        mu_s_K = mu / sK
        invQ = 1.0 / sQ
        invK = 1.0 / sK
        Sigma_s_Q = Sigma * invQ.unsqueeze(0) * invQ.unsqueeze(1)
        Sigma_s_K = Sigma * invK.unsqueeze(0) * invK.unsqueeze(1)
        Sigma_s_Q_abs = Sigma_s_Q.abs()
        Sigma_s_K_abs = Sigma_s_K.abs()
        sd_s_Q = torch.diagonal(Sigma_s_Q).clamp(min=0.0)
        sr_s_Q = Sigma_s_Q_abs.sum(dim=1) - torch.diagonal(Sigma_s_Q_abs)
        sd_s_K = torch.diagonal(Sigma_s_K).clamp(min=0.0)
        sr_s_K = Sigma_s_K_abs.sum(dim=1) - torch.diagonal(Sigma_s_K_abs)

        # GQA topology.
        q_out, hidden = Wq_scaled.shape
        k_out = Wk_scaled.shape[0]
        if k_out % head_dim != 0 or q_out % head_dim != 0:
            print(f"      [QK-SNC] {attn_group}: head_dim={head_dim} does not divide "
                  f"q_out={q_out} or k_out={k_out}; skipping.")
            return None
        n_q_heads = q_out // head_dim
        n_k_heads = k_out // head_dim
        if n_k_heads == 0 or n_q_heads % n_k_heads != 0:
            print(f"      [QK-SNC] {attn_group}: invalid head topology "
                  f"({n_q_heads=}, {n_k_heads=}); skipping.")
            return None
        gqa_ratio = n_q_heads // n_k_heads
        print(f"      [QK-SNC] {attn_group}: n_q_heads={n_q_heads}, n_k_heads={n_k_heads}, "
              f"ratio={gqa_ratio}, head_dim={head_dim}")

        # Reshape for per-head computation.
        # Wq_fp_3d : [n_q_heads, head_dim, hidden]
        # Wk_fp_3d : [n_k_heads, head_dim, hidden]
        Wk_fp_3d = Wk_fp.view(n_k_heads, head_dim, hidden)
        Wq_fp_3d = Wq_fp.view(n_q_heads, head_dim, hidden)

        # =============== Q-side alpha vectors ==============================
        # For each K head g and head-dim slot d:
        #   v_k[g, d] = (W_K^(g) mu)_d
        v_k = torch.einsum('ghd,d->gh', Wk_fp_3d, mu)             # [n_k, head_dim]
        alpha_sig_Q_per_k = v_k.pow(2)

        SigWkT = torch.einsum('ij,gdj->gdi', Sigma, Wk_fp_3d)     # [n_k, head_dim, hidden]
        alpha_noi_Q_per_k = (Wk_fp_3d * SigWkT).sum(dim=2).clamp(min=0.0)

        # Broadcast each (n_k, head_dim) row to its gqa_ratio Q heads.
        alpha_sig_Q = alpha_sig_Q_per_k.repeat_interleave(gqa_ratio, dim=0)   # [n_q, head_dim]
        alpha_noi_Q = alpha_noi_Q_per_k.repeat_interleave(gqa_ratio, dim=0)

        alpha_sig_Q_flat = alpha_sig_Q.reshape(q_out).to(q_module.weight.dtype)
        alpha_noi_Q_flat = alpha_noi_Q.reshape(q_out).to(q_module.weight.dtype)

        # =============== Q-side bilinear pass ==============================
        Wq_scaled_dtype = Wq_scaled.to(q_module.weight.dtype)
        mu_s_Q_dev = mu_s_Q.to(q_module.weight.dtype)
        sd_s_Q_dev = sd_s_Q.to(q_module.weight.dtype)
        sr_s_Q_dev = sr_s_Q.to(q_module.weight.dtype)

        Wq_refined, _, q_stats = self.quantize_weight_snc_groupwise(
            Wq_scaled_dtype, mu_s_Q_dev, Sigma_s_Q.to(torch.float32),
            sd_s_Q_dev, sr_s_Q_dev,
            alpha_sig=alpha_sig_Q_flat,
            alpha_noi=alpha_noi_Q_flat,
            apply_snc=True,
            max_flip_percent=self.qk_max_flip_percent,
            snr_budget_p=self.qk_snr_budget_p,
        )

        Wq_back = (Wq_refined / sQ.unsqueeze(0)).to(q_module.weight.dtype)
        q_module.weight.data.copy_(Wq_back)

        # =============== K-side bilinear pass (optional) ===================
        k_stats = None
        if self.qk_refine_k:
            v_q = torch.einsum('hjd,d->hj', Wq_fp_3d, mu)              # [n_q, head_dim]
            alpha_sig_per_qhead = v_q.pow(2)
            SigWqT = torch.einsum('ij,hdj->hdi', Sigma, Wq_fp_3d)      # [n_q, head_dim, hidden]
            alpha_noi_per_qhead = (Wq_fp_3d * SigWqT).sum(dim=2).clamp(min=0.0)

            # Sum over the gqa_ratio Q heads per K group.
            alpha_sig_K = alpha_sig_per_qhead.view(n_k_heads, gqa_ratio, head_dim).sum(dim=1)
            alpha_noi_K = alpha_noi_per_qhead.view(n_k_heads, gqa_ratio, head_dim).sum(dim=1)

            alpha_sig_K_flat = alpha_sig_K.reshape(k_out).to(k_module.weight.dtype)
            alpha_noi_K_flat = alpha_noi_K.reshape(k_out).to(k_module.weight.dtype)

            Wk_scaled_dtype = Wk_scaled.to(k_module.weight.dtype)
            mu_s_K_dev = mu_s_K.to(k_module.weight.dtype)
            sd_s_K_dev = sd_s_K.to(k_module.weight.dtype)
            sr_s_K_dev = sr_s_K.to(k_module.weight.dtype)

            Wk_refined, _, k_stats = self.quantize_weight_snc_groupwise(
                Wk_scaled_dtype, mu_s_K_dev, Sigma_s_K.to(torch.float32),
                sd_s_K_dev, sr_s_K_dev,
                alpha_sig=alpha_sig_K_flat,
                alpha_noi=alpha_noi_K_flat,
                apply_snc=True,
                max_flip_percent=self.qk_max_flip_percent,
                snr_budget_p=self.qk_snr_budget_p,
            )
            Wk_back = (Wk_refined / sK.unsqueeze(0)).to(k_module.weight.dtype)
            k_module.weight.data.copy_(Wk_back)

        return {
            'q_flips': q_stats['total'],
            'k_flips': (k_stats['total'] if k_stats else 0),
            'q_sign_disagree_pct': q_stats['sign_disagree_pct'],
            'k_sign_disagree_pct': (k_stats['sign_disagree_pct'] if k_stats else 0.0),
            'gqa_ratio': gqa_ratio,
        }

    def _run_qk_pass(self):
        print("\n" + "=" * 80)
        print("[Step 2] QK Bilinear SNC")
        print("=" * 80)

        # Group attention projections by attention block.
        attn_groups = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                gi = get_layer_group(name)
                if gi is None:
                    continue
                _, attn = gi
                attn_groups.setdefault(attn, {})
                low = name.lower()
                if 'q_proj' in low or 'query' in low:
                    attn_groups[attn]['q_proj'] = (name, module)
                elif 'k_proj' in low or 'key' in low:
                    attn_groups[attn]['k_proj'] = (name, module)
                elif 'v_proj' in low or 'value' in low:
                    attn_groups[attn]['v_proj'] = (name, module)

        head_dim = self.infer_head_dim()
        print(f"  Found {len(attn_groups)} attention groups, head_dim={head_dim}")

        refined = 0
        qk_summaries = []
        for attn, projs in tqdm(attn_groups.items(), desc="  QK bilinear"):
            try:
                summary = self._apply_qk_snc_to_attn_group(attn, projs, head_dim)
                if summary is not None:
                    refined += 1
                    qk_summaries.append(summary)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"    [QK-SNC] error on {attn}: {e}")
                import traceback; traceback.print_exc()
                continue

        print(f"\n  Refined {refined}/{len(attn_groups)} attention groups")
        if qk_summaries:
            q_tot = sum(s['q_flips'] for s in qk_summaries)
            k_tot = sum(s['k_flips'] for s in qk_summaries)
            mean_ratio = float(np.mean([s['gqa_ratio'] for s in qk_summaries]))
            print(f"  Total Q flips: {q_tot:,}    Total K flips: {k_tot:,}")
            if q_tot > 0 and k_tot > 0:
                print(f"  K-vs-Q flip ratio: {k_tot / max(q_tot, 1):.2f}x  "
                      f"(SNC predicts ~sqrt(|H_g|) = sqrt({mean_ratio:.1f}) = "
                      f"{np.sqrt(mean_ratio):.2f}x more K-side budget under GQA)")
            print(f"  Q sign-disagree vs CLC: "
                  f"{np.mean([s['q_sign_disagree_pct'] for s in qk_summaries]):.2f}%")
            if self.qk_refine_k:
                print(f"  K sign-disagree vs CLC: "
                      f"{np.mean([s['k_sign_disagree_pct'] for s in qk_summaries]):.2f}%")

        # Free caches.
        self.original_state_dict = None
        self.qk_awq_scales.clear()
        self.qk_activation_stats.clear()
        torch.cuda.empty_cache()
        gc.collect()

    # --------------------------------------------------- top-level pipeline

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        try:
            if self.apply_qk_snc:
                self.original_state_dict = {}
            quantized, n_layers = self._run_linear_pass(calibration_data, n_samples)
            if self.apply_qk_snc:
                self._run_qk_pass()
        except torch.cuda.OutOfMemoryError:
            print("\nCUDA OOM. Suggestions: reduce --layer-batch-size, --n-calib, "
                  "--max-tokens-per-sample; or disable --qk-snc.")
            sys.exit(1)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\nCUDA OOM. Suggestions: reduce --layer-batch-size, --n-calib, "
                      "--max-tokens-per-sample; or disable --qk-snc.")
                sys.exit(1)
            raise

        print("\n" + "=" * 80)
        print("SNC Quantization Complete")
        print(f"  Total layers quantized: {quantized}/{n_layers}")
        print("=" * 80)

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            print(f"\nLinear-SNC: optimal alpha mean={np.mean(alphas):.3f}, "
                  f"median={np.median(alphas):.3f}")
            if self.use_snc:
                flip_totals = [info['flip_stats']['total'] for info in self.layer_scales.values()]
                disagree = [info['flip_stats']['sign_disagree_pct'] for info in self.layer_scales.values()]
                print(f"  Total flips: {int(np.sum(flip_totals)):,}, "
                      f"mean/layer: {np.mean(flip_totals):,.1f}, "
                      f"median/layer: {np.median(flip_totals):,.1f}")
                print(f"  SNC vs CLC sign-disagree: {np.mean(disagree):.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AWQ + SNC linear quantization, with optional bilinear refinement on GQA Q/K."
    )
    # Base AWQ args
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--use-snc", action="store_true", default=True,
                        help="Enable SNC (linear, M=I) correction (default: True).")
    parser.add_argument("--no-snc", dest="use_snc", action="store_false")
    parser.add_argument("--use-james-stein", action="store_true", default=True)
    parser.add_argument("--no-james-stein", dest="use_james_stein", action="store_false")

    # Linear-SNC knobs
    parser.add_argument("--snr-lambda", type=float, default=1.0,
                        help="SNR denom weight on alpha_noi (Eq. 9). Default 1.0.")
    parser.add_argument("--snr-beta", type=float, default=1.0,
                        help="SNR denom weight on off-diag Sigma row-sum. Default 1.0.")
    parser.add_argument("--snr-budget-p", type=float, default=0.05,
                        help="Linear-SNC budget fraction. Default 0.05.")
    parser.add_argument("--max-flip-percent", type=float, default=0.05,
                        help="Linear-SNC hard cap on flips per output row. Default 0.05.")
    parser.add_argument("--use-safety-cap", action="store_true", default=False,
                        help="Apply per-channel safety cap from Cor. 1(ii).")

    # QK bilinear SNC
    parser.add_argument("--qk-snc", dest="apply_qk_snc", action="store_true", default=False,
                        help="Enable bilinear SNC refinement on GQA Q/K projections (default: False).")
    parser.add_argument("--no-qk-refine-k", dest="qk_refine_k", action="store_false", default=True,
                        help="If set with --qk-snc, refine ONLY Q (skip K). Default: refine both.")
    parser.add_argument("--qk-max-flip-percent", type=float, default=0.05,
                        help="QK-SNC hard cap on flips per output row. Default 0.05.")
    parser.add_argument("--qk-snr-budget-p", type=float, default=0.05,
                        help="QK-SNC budget fraction. Default 0.05.")

    # lm_head handling
    parser.add_argument("--skip-lmhead", dest="skip_lmhead", action="store_true", default=True,
                        help="Do NOT quantize lm_head (default: True).")
    parser.add_argument("--no-skip-lmhead", dest="skip_lmhead", action="store_false",
                        help="Quantize lm_head as well (chunked).")
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                        help="Chunks for lm_head if it is quantized.")

    # Misc
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_awq_snc")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("AWQ + SNC (Signal-to-Noise Cancellation)")
    print("=" * 80)
    print(f"  Model:  {args.model_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Bits:   {args.bits}, group_size: {args.group_size}, n_calib: {args.n_calib}")
    print(f"  Linear-SNC: enabled={args.use_snc}, "
          f"p={args.snr_budget_p}, max_flip={args.max_flip_percent*100:.2f}%, "
          f"lambda={args.snr_lambda}, beta={args.snr_beta}, "
          f"safety_cap={args.use_safety_cap}")
    print(f"  QK-SNC:     enabled={args.apply_qk_snc}, "
          f"p={args.qk_snr_budget_p}, max_flip={args.qk_max_flip_percent*100:.2f}%, "
          f"refine_K={args.qk_refine_k}")
    print(f"  Skip lm_head: {args.skip_lmhead}")
    print("=" * 80)

    print(f"\nLoading model and tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    quantizer = SNCAWQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_snc=args.use_snc,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        skip_lmhead=args.skip_lmhead,
        max_flip_percent=args.max_flip_percent,
        snr_lambda=args.snr_lambda,
        snr_beta=args.snr_beta,
        snr_budget_p=args.snr_budget_p,
        use_safety_cap=args.use_safety_cap,
        use_james_stein=args.use_james_stein,
        apply_qk_snc=args.apply_qk_snc,
        qk_max_flip_percent=args.qk_max_flip_percent,
        qk_snr_budget_p=args.qk_snr_budget_p,
        qk_refine_k=args.qk_refine_k,
    )

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib,
                                              seqlen=2048, seed=args.seed,
                                              cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset
                       if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib,
                                                    seqlen=2048, seed=args.seed,
                                                    cache_dir=args.cache_dir)

    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()