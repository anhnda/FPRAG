"""
AWQ Quantum-Inspired Weight Correction (awq_quantum_correction.py)

Energy formulation:
    E(s) = sum_alpha lam_alpha (J_V^T s)_alpha^2 + H^T s
    J_V[j] = half_delta[j] * V[j]
    D[i,j] = W_sc[i,j] - midpoint[i,j]   (displacement of original weight from midpoint)
    H[i,j] = -2 * half_delta[i,j] * (G D[i])_j - 2 * lambda_fid * half_delta[i,j] * D[i,j]

Delta-energy for flipping spin (i,j), ds=-2*s[i,j], dv=J_V[i,j]*ds:
    dE = lam^T(2v*dv + dv^2) + H[i,j]*ds

Sign convention: S_nearest[i,j] = +1 means nearest grid point is ceil,
                                   -1 means nearest grid point is floor.
E is minimized at S_nearest when interactions are zero; H pulls toward sign(D).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import argparse
import random
import numpy as np
import gc
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("calibration_utils not found.")
    def get_c4_calibration_data(*args, **kwargs):
        raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs):
        raise NotImplementedError("Please provide calibration_utils.py")


class AWQBaseQuantizer:
    def __init__(self, bits=4, group_size=128, n_grid=20):
        self.bits = bits
        self.group_size = group_size
        self.n_grid = n_grid

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W):
        out_features, in_features = W.shape
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded = n_groups * self.group_size
        if padded > in_features:
            W_pad = torch.zeros(out_features, padded, device=W.device, dtype=W.dtype)
            W_pad[:, :in_features] = W
        else:
            W_pad = W
        W_g = W_pad.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2**self.bits - 1
        scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)
        W_int = torch.round(W_g / scale + zp).clamp(0, max_int)
        W_deq = (W_int - zp) * scale
        W_deq = W_deq.reshape(out_features, padded)
        if padded > in_features:
            W_deq = W_deq[:, :in_features]
        return W_deq

    @torch.no_grad()
    def get_quantization_grid_info(self, W):
        out_features, in_features = W.shape
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded = n_groups * self.group_size
        max_int = 2**self.bits - 1
        if padded > in_features:
            W_pad = torch.zeros(out_features, padded, device=W.device, dtype=W.dtype)
            W_pad[:, :in_features] = W
        else:
            W_pad = W
        W_g = W_pad.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)
        W_div = W_g / scale + zp
        W_int_nearest = torch.round(W_div).clamp(0, max_int)
        W_int_floor   = torch.floor(W_div).clamp(0, max_int)
        W_int_ceil    = torch.ceil(W_div).clamp(0, max_int)
        exact_mask = (W_int_floor == W_int_ceil)
        W_int_ceil[exact_mask  & (W_int_ceil  < max_int)] += 1
        W_int_floor[exact_mask & (W_int_floor > 0)]       -= 1
        floor_val   = (W_int_floor   - zp) * scale
        ceil_val    = (W_int_ceil    - zp) * scale
        nearest_val = (W_int_nearest - zp) * scale
        floor_val   = floor_val.reshape(out_features,   padded)[:, :in_features]
        ceil_val    = ceil_val.reshape(out_features,    padded)[:, :in_features]
        nearest_val = nearest_val.reshape(out_features, padded)[:, :in_features]
        return {
            'floor': floor_val, 'ceil': ceil_val, 'nearest': nearest_val,
            'midpoint': (floor_val + ceil_val) / 2,
            'delta':     ceil_val - floor_val,
        }

    @torch.no_grad()
    def compute_l2_salience(self, activation_data):
        if not activation_data:
            return None
        total  = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in activation_data)
        in_feat = activation_data[0].shape[-1]
        acc = torch.zeros(in_feat, dtype=torch.float32)
        for x in activation_data:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            acc += x_flat.pow(2).sum(dim=0)
        return acc / total

    @torch.no_grad()
    def search_best_scale(self, W, X_calib, activation_salience):
        device = W.device
        dtype  = W.dtype
        activation_salience = activation_salience.to(device).to(dtype).clamp(min=1e-5)
        X      = X_calib.to(device).to(dtype)
        Y_orig = X @ W.t()
        best_error  = float('inf')
        best_alpha  = 0.0
        best_scales = torch.ones(W.shape[1], device=device, dtype=dtype)
        for grid_idx in range(self.n_grid + 1):
            alpha  = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)
            W_scaled = W * scales.unsqueeze(0)
            W_q      = self.quantize_weight_groupwise_asymmetric(W_scaled)
            W_recon  = W_q / scales.unsqueeze(0)
            Y_q      = X @ W_recon.t()
            error    = (Y_orig - Y_q).pow(2).mean().item()
            if error < best_error:
                best_error  = error
                best_alpha  = alpha
                best_scales = scales.clone()
            del W_scaled, W_q, W_recon, Y_q
        del X, Y_orig
        return best_scales, best_alpha, best_error


class QuantumCorrectionEngine:

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 n_grid=20, max_tokens_per_sample=2048, lmhead_chunks=4,
                 lambda_fidelity=0.1,
                 mf_max_iter=30, mf_beta_init=0.1, mf_beta_final=50.0, mf_n_temps=10,
                 sw_soft_threshold=0.1, group_max_size=6, cd_max_sweeps=3,
                 max_calib_samples_correction=512, top_k_eigvecs_G=32):

        self.model               = model
        self.tokenizer           = tokenizer
        self.device              = device
        self.bits                = bits
        self.group_size          = group_size
        self.n_grid              = n_grid
        self.max_tokens_per_sample = max_tokens_per_sample
        self.lmhead_chunks       = lmhead_chunks
        self.lambda_fidelity     = lambda_fidelity
        self.mf_max_iter         = mf_max_iter
        self.mf_beta_init        = mf_beta_init
        self.mf_beta_final       = mf_beta_final
        self.mf_n_temps          = mf_n_temps
        self.sw_soft_threshold   = sw_soft_threshold
        self.group_max_size      = group_max_size
        self.cd_max_sweeps       = cd_max_sweeps
        self.max_calib_samples   = max_calib_samples_correction
        self.top_k_eigvecs_G     = top_k_eigvecs_G
        self.base_quantizer      = AWQBaseQuantizer(bits=bits, group_size=group_size, n_grid=n_grid)
        self.activation_data     = {}
        self.layer_stats         = {}

        print(f"\n{'='*80}")
        print(f"Quantum-Inspired AWQ Correction Engine")
        print(f"{'='*80}")
        print(f"  Bits:{bits} GroupSize:{group_size} lambda_fid:{lambda_fidelity}")
        print(f"  MF: beta={mf_beta_init}->{mf_beta_final} temps={mf_n_temps} iter={mf_max_iter}")
        print(f"  SW threshold:{sw_soft_threshold}  Group max:{group_max_size}(2^g={2**group_max_size})")
        print(f"  CD sweeps:{cd_max_sweeps}  Low-rank k:{top_k_eigvecs_G}")
        print(f"{'='*80}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Activation collection
    # ─────────────────────────────────────────────────────────────────────────

    def _get_hook(self, name):
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                idx = torch.randperm(inp.shape[1])[:self.max_tokens_per_sample].sort()[0]
                inp = inp[:, idx, :]
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def _collect_activations(self, layer_names_modules, calibration_data, n_samples):
        self.activation_data = {}
        handles = []
        for name, module in layer_names_modules:
            handles.append(module.register_forward_hook(self._get_hook(name)))
        successful = 0
        for text in calibration_data[:n_samples]:
            try:
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                self.model(**inputs, use_cache=False, return_dict=True)
                successful += 1
                if successful % 32 == 0:
                    torch.cuda.empty_cache()
            except Exception:
                continue
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
        gc.collect()
        return successful

    @torch.no_grad()
    def _get_calibration_matrix(self, name, max_samples=None):
        if name not in self.activation_data or not self.activation_data[name]:
            return None
        if max_samples is None:
            max_samples = self.max_calib_samples
        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0).float()
        if X.shape[0] > max_samples:
            idx = torch.randperm(X.shape[0])[:max_samples]
            X   = X[idx]
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Group clustering (CPU numpy, no GPU syncs)
    # ─────────────────────────────────────────────────────────────────────────

    def _cluster_soft_spins(self, J_sub_gpu, soft_indices_gpu):
        n = J_sub_gpu.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [[soft_indices_gpu[0].item()]]
        J_np  = J_sub_gpu.abs().cpu().numpy()
        si_np = soft_indices_gpu.cpu().numpy()
        np.fill_diagonal(J_np, 0)
        row_sums = J_np.sum(axis=1)
        used     = np.zeros(n, dtype=bool)
        groups   = []
        while not used.all():
            tmp       = row_sums.copy()
            tmp[used] = -1.0
            seed      = int(tmp.argmax())
            group     = [seed]
            used[seed] = True
            for _ in range(self.group_max_size - 1):
                if used.all():
                    break
                coupling       = J_np[:, group].sum(axis=1)
                coupling[used] = -1.0
                best           = int(coupling.argmax())
                if coupling[best] < 0.1 * row_sums[seed] / max(len(group), 1):
                    break
                group.append(best)
                used[best] = True
            groups.append([int(si_np[i]) for i in group])
        return groups

    # ─────────────────────────────────────────────────────────────────────────
    # Core per-layer correction
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _correct_layer(self, name, module, X_calib, debug=False):
        W = module.weight.data
        original_dtype           = W.dtype
        out_features, in_features = W.shape
        device                   = W.device

        # ── AWQ scaling ───────────────────────────────────────────────────────
        salience = self.base_quantizer.compute_l2_salience(
            self.activation_data.get(name, []))
        if salience is None:
            salience = torch.ones(in_features)
        X_search = X_calib[:min(2048, X_calib.shape[0])].to(device).to(original_dtype)
        best_scales, best_alpha, _ = self.base_quantizer.search_best_scale(
            W, X_search, salience.to(device))
        del X_search
        if debug:
            print(f"    AWQ: alpha={best_alpha:.3f}")

        W_scaled      = W * best_scales.unsqueeze(0)
        grid_info     = self.base_quantizer.get_quantization_grid_info(W_scaled)
        baseline_W_q  = grid_info['nearest'] / best_scales.unsqueeze(0)
        X_corr        = X_calib[:min(self.max_calib_samples,
                                     X_calib.shape[0])].to(device).to(original_dtype)
        Y_orig        = X_corr @ W.t()
        Y_base        = X_corr @ baseline_W_q.t()
        baseline_error = (Y_orig - Y_base).pow(2).mean().item()
        if debug:
            print(f"    Baseline error: {baseline_error:.8f}")

        # ── Low-rank G via truncated SVD ──────────────────────────────────────
        X_for_G = X_corr / best_scales.unsqueeze(0).to(X_corr.dtype)
        n_tok   = X_for_G.shape[0]
        k       = min(self.top_k_eigvecs_G, n_tok, in_features)
        U, S_sv, V = torch.svd_lowrank(X_for_G.float(), q=k, niter=4)
        lam = (S_sv ** 2) / n_tok    # eigenvalues of (1/n) X^T X  [k]
        del X_for_G, U, S_sv
        torch.cuda.empty_cache()
        if debug:
            print(f"    G: top-{k}, lam_1={lam[0]:.4f}, lam_k={lam[-1]:.6f}")

        # ── Ising tensors ─────────────────────────────────────────────────────
        midpoint   = grid_info['midpoint'].float().to(device)
        delta_all  = grid_info['delta'].float().to(device)
        nearest    = grid_info['nearest'].float().to(device)
        W_sc_f32   = W_scaled.float()
        half_delta = delta_all / 2          # always > 0  (ceil > floor)

        # D[i,j] = displacement of original (scaled) weight from midpoint
        # positive  → weight is above midpoint → nearest is ceil  → S_nearest = +1
        # negative  → weight is below midpoint → nearest is floor → S_nearest = -1
        D_all = W_sc_f32 - midpoint         # SIGN FIX: was midpoint - W_sc_f32

        S_nearest = torch.sign(nearest - midpoint)
        S_nearest[S_nearest == 0] = 1.0

        V_dev   = V.to(device)
        lam_dev = lam.to(device)

        # Full energy (per row i):
        #   E_i(s) = (hd*s - D)^T G (hd*s - D) + lambda_fid ||hd*s - D||^2
        # Linear term in s:  H[i,j] * s[i,j]
        #   H[i,j] = -2*hd[i,j]*(GD[i])_j  -  2*lambda_fid*hd[i,j]*D[i,j]
        # (factor of 2 is absorbed here so dE = lam^T(2v*dv+dv^2) + H*ds is exact)
        Vt_D  = D_all @ V_dev               # [out, k]
        G_D   = (Vt_D * lam_dev) @ V_dev.t()  # [out, in]  row-wise G @ D
        H_all = -2.0 * half_delta * (G_D + self.lambda_fidelity * D_all)
        if debug:
            print(f"    V shape: {V_dev.shape}, n_tok={n_tok}, in_features={in_features}")
            print(f"    G_D[0,:3]: {G_D[0,:3].tolist()}, D_all[0,:3]: {D_all[0,:3].tolist()}")
        del G_D, Vt_D

        # ── Sanity check (remove after confirming fix) ────────────────────────
        if debug:
            row = 0
            s   = S_nearest[row]
            hd  = half_delta[row]
            h   = H_all[row]
            v   = (s * hd) @ V_dev          # [k]
            dE_sample = []
            for j in range(min(20, in_features)):
                dv_j = (-2.0 * s[j]) * hd[j] * V_dev[j]
                dE_j = ((2.0 * v + dv_j) * dv_j * lam_dev).sum() \
                       + h[j] * (-2.0 * s[j])
                dE_sample.append(dE_j.item())
            n_pos = sum(x > 0 for x in dE_sample)
            n_neg = sum(x < 0 for x in dE_sample)
            print(f"    Sanity dE@S_nearest (first 20 spins): "
                  f"{n_pos} positive, {n_neg} negative")
            print(f"    D_all mean sign: {D_all[row].sign().mean().item():.3f}, "
                  f"S_nearest mean: {S_nearest[row].mean().item():.3f}")
            print(f"    H_all[0,:5]: {H_all[row,:5].tolist()}")
            print(f"    S_nearest[0,:5]: {S_nearest[row,:5].tolist()}")
            print(f"    D_all[0,:5]: {D_all[row,:5].tolist()}")
        # Sanity: at S_nearest, dE > 0 for all spins (nearest is a local minimum
        # of the uncoupled problem).  The coupling (lam) may cause some flips.

        # ── Phase 1: Mean-Field Annealing ─────────────────────────────────────
        # MF free energy gradient w.r.t. m[i,j]:
        #   eff[i,j] = H[i,j] + 2*hd[i,j] * (V lam V^T (m[i]*hd[i]))_j
        # Fixed-point: m = -tanh(beta * eff)
                # ── Phase 1: Mean-Field Annealing ─────────────────────────────────────
        # MF free energy gradient w.r.t. m[i,j]:
        #   eff[i,j] = H[i,j] + 2*hd[i,j] * (V lam V^T (m[i]*hd[i]))_j
        # Fixed-point: m = -tanh(beta * eff)
        # Init: flip any spin where H*s > 0 (opposes its own field under truncated G)
        S_init = S_nearest.clone()
        wrong_init = (H_all * S_init) > 0
        S_init[wrong_init] *= -1
        if debug:
            print(f"    Init pre-flips: {wrong_init.sum().item()}")
        M = S_init.float()
        del S_init, wrong_init
        betas = torch.logspace(
            np.log10(self.mf_beta_init),
            np.log10(self.mf_beta_final),
            self.mf_n_temps)

        for beta in betas:
            beta_val = beta.item()
            for iteration in range(self.mf_max_iter):
                M_hd   = M * half_delta                                    # [out, in]
                Vt_Mhd = M_hd @ V_dev                                     # [out, k]
                JM     = 2.0 * (Vt_Mhd * lam_dev) @ V_dev.t() * half_delta  # [out, in]
                M_next = -torch.tanh(beta_val * (H_all + JM))
                if iteration % 5 == 4:
                    if (M_next - M).abs().max().item() < 1e-5:
                        M = M_next
                        break
                M = M_next

        S_mf = torch.sign(M)
        S_mf[S_mf == 0] = 1.0
        total_mf_flips = (S_mf != S_nearest).sum().item()
        del M

        # ── Phase 2: Spin-Wave Stability ──────────────────────────────────────
        # stability[i,j] = s[i,j] * (partial E / partial s[i,j])
        #                = s[i,j] * (H[i,j] + 2*hd[i,j]*(V lam V^T (s[i]*hd[i]))_j)
        # positive → spin is stable (moving it increases E)
        # negative → spin is wrong  (moving it decreases E) → flip immediately
        S_hd    = S_mf * half_delta
        Vt_Shd  = S_hd @ V_dev
        JS      = 2.0 * (Vt_Shd * lam_dev) @ V_dev.t() * half_delta
        Stability = S_mf * (H_all + JS)
        del S_hd, Vt_Shd, JS

        wrong_mask = Stability < 0.0
        S_mf[wrong_mask] *= -1
        total_immediate_flips = wrong_mask.sum().item()

        # Recompute stability after immediate flips
        S_hd2   = S_mf * half_delta
        Vt_Shd2 = S_hd2 @ V_dev
        JS2     = 2.0 * (Vt_Shd2 * lam_dev) @ V_dev.t() * half_delta
        Stability2 = S_mf * (H_all + JS2)
        del S_hd2, Vt_Shd2, JS2

        median_stab = Stability2.abs().median(dim=1, keepdim=True).values
        threshold   = self.sw_soft_threshold * median_stab.clamp(min=1e-10)
        # Soft: stable but low-confidence (0 < stability < threshold)
        soft_masks  = (Stability2 >= 0) & (Stability2 < threshold)
        total_sw_soft = soft_masks.sum().item()
        del Stability, Stability2, median_stab, threshold, wrong_mask

        # ── Phase 3: Group Refinement ─────────────────────────────────────────
        S_refined       = S_mf.clone()
        total_group_flips = 0
        rows_with_soft  = (soft_masks.sum(dim=1) >= 3).nonzero(as_tuple=True)[0]

        for row_idx in rows_with_soft.tolist():
            soft_idx = soft_masks[row_idx].nonzero(as_tuple=True)[0]
            n_soft   = soft_idx.shape[0]
            if n_soft == 0:
                continue

            # Cap at 50 worst (lowest stability) soft spins per row
            if n_soft > 50:
                hd_tmp   = half_delta[row_idx]
                s_tmp    = S_mf[row_idx]
                Js_tmp   = 2.0 * hd_tmp * (V_dev @ (lam_dev * (V_dev.t() @ (hd_tmp * s_tmp))))
                stab_tmp = s_tmp * (H_all[row_idx] + Js_tmp)
                _, worst = stab_tmp[soft_idx].sort()
                soft_idx = soft_idx[worst[:50]]
                n_soft   = 50

            hd_row   = half_delta[row_idx]
            J_V_row  = hd_row.unsqueeze(1) * V_dev          # [in, k]
            J_V_soft = J_V_row[soft_idx]                    # [n_soft, k]
            J_sub    = J_V_soft @ (lam_dev.unsqueeze(0) * J_V_soft).t()
            groups   = self._cluster_soft_spins(J_sub, soft_idx)

            s_row = S_refined[row_idx].clone()
            h_row = H_all[row_idx]
            v_row = J_V_row.t() @ s_row                     # [k]

            for group in groups:
                g = len(group)
                if g == 0:
                    continue
                group_idx = torch.tensor(group, device=device, dtype=torch.long)
                s_g = s_row[group_idx]
                J_g = J_V_row[group_idx]
                h_g = h_row[group_idx]

                if g <= self.group_max_size:
                    n_configs = 2 ** g
                    bit_idx  = torch.arange(n_configs, device=device).unsqueeze(1)
                    bit_pos  = torch.arange(g,         device=device).unsqueeze(0)
                    flip_mat = ((bit_idx >> bit_pos) & 1).float()
                    delta_sg = -2.0 * flip_mat * s_g.unsqueeze(0)
                    delta_v  = delta_sg @ J_g
                    dE = (delta_v * (2.0 * v_row.unsqueeze(0) + delta_v)) @ lam_dev \
                         + delta_sg @ h_g
                    best_idx = dE.argmin()
                    if dE[best_idx] < -1e-12:
                        best_flip = flip_mat[best_idx].bool()
                        s_row[group_idx[best_flip]] *= -1
                        total_group_flips += best_flip.sum().item()
                        v_row = v_row + delta_v[best_idx]
                else:
                    for idx in group:
                        j_v = J_V_row[idx]
                        ds  = -2.0 * s_row[idx]
                        dv  = ds * j_v
                        dE  = ((2.0 * v_row + dv) * dv * lam_dev).sum() \
                              + h_row[idx] * ds
                        if dE.item() < -1e-12:
                            s_row[idx] *= -1
                            v_row = v_row + dv
                            total_group_flips += 1

            S_refined[row_idx] = s_row

        del soft_masks, rows_with_soft

        # ── Phase 4: Coordinate Descent (chunked Gauss-Seidel) ────────────────
        # Process chunk_size columns at a time; update V_s incrementally after
        # each chunk so later chunks see the effect of earlier flips.
        diag_J     = (V_dev ** 2) @ lam_dev    # [in]  diagonal of J matrix
        S_final    = S_refined.clone()
        total_cd_flips = 0
        chunk_size = 64

        for sweep in range(self.cd_max_sweeps):
            sweep_flips = 0
            V_s = (S_final * half_delta) @ V_dev    # [out, k]

            for j_start in range(0, in_features, chunk_size):
                j_end      = min(j_start + chunk_size, in_features)
                j_slice    = slice(j_start, j_end)

                DS_chunk   = -2.0 * S_final[:, j_slice]       # [out, chunk]
                hd_chunk   = half_delta[:, j_slice]            # [out, chunk]
                V_chunk    = V_dev[j_start:j_end, :]           # [chunk, k]
                VsVt_chunk = V_s @ V_chunk.t()                 # [out, chunk]
                diag_chunk = diag_J[j_slice]                   # [chunk]
                H_chunk    = H_all[:, j_slice]                 # [out, chunk]

                term1 = 2.0 * DS_chunk * hd_chunk * VsVt_chunk
                term2 = (hd_chunk * DS_chunk) ** 2 * diag_chunk
                dE    = term1 + term2 + H_chunk * DS_chunk     # [out, chunk]
                del term1, term2, VsVt_chunk, DS_chunk

                flip_mask     = dE < -1e-12
                del dE
                n_chunk_flips = flip_mask.sum().item()
                if n_chunk_flips == 0:
                    continue

                flip_float   = flip_mask.float()
                ds_accepted  = -2.0 * S_final[:, j_slice] * flip_float
                dV_s         = (ds_accepted * hd_chunk) @ V_chunk  # [out, k]

                S_final[:, j_slice][flip_mask] *= -1
                V_s         = V_s + dV_s
                sweep_flips += n_chunk_flips

            total_cd_flips += sweep_flips
            if sweep_flips == 0:
                break

        # ── Reconstruct and write back ────────────────────────────────────────
        W_corrected        = (midpoint + half_delta * S_final).to(original_dtype)
        W_final            = W_corrected / best_scales.unsqueeze(0)
        module.weight.data = W_final

        Y_corrected     = X_corr @ W_final.t()
        corrected_error = (Y_orig - Y_corrected).pow(2).mean().item()
        improvement     = (baseline_error - corrected_error) / max(baseline_error, 1e-12) * 100

        if debug:
            print(f"    Corrected: {corrected_error:.8f} ({improvement:+.2f}%)")
            print(f"    Flips MF:{total_mf_flips} imm:{total_immediate_flips} "
                  f"G:{total_group_flips} CD:{total_cd_flips}")
            print(f"    Soft spins: {total_sw_soft}")

        stats = {
            'awq_alpha':        best_alpha,
            'baseline_error':   baseline_error,
            'corrected_error':  corrected_error,
            'improvement_pct':  improvement,
            'mf_flips':         total_mf_flips,
            'immediate_flips':  total_immediate_flips,
            'sw_soft':          total_sw_soft,
            'group_flips':      total_group_flips,
            'cd_flips':         total_cd_flips,
        }

        del (V_dev, lam_dev, H_all, half_delta, diag_J,
             S_nearest, S_mf, S_refined, S_final,
             midpoint, delta_all, nearest, W_sc_f32, D_all, W_corrected,
             X_corr, Y_orig, Y_base, Y_corrected, baseline_W_q)
        torch.cuda.empty_cache()
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Model-level loop
    # ─────────────────────────────────────────────────────────────────────────

    def correct_model(self, calibration_data, n_samples=128, layer_batch_size=16):
        print(f"\n{'='*80}\nQUANTUM-INSPIRED WEIGHT CORRECTION\n{'='*80}")
        layer_list = [(name, module)
                      for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear)]
        n_layers  = len(layer_list)
        n_batches = (n_layers + layer_batch_size - 1) // layer_batch_size
        print(f"  Layers:{n_layers}  Batches:{n_batches}  Calib:{n_samples}")

        total_improvement = []
        t_start = time.time()

        for batch_idx in range(n_batches):
            b_start = batch_idx * layer_batch_size
            b_end   = min(b_start + layer_batch_size, n_layers)
            batch   = layer_list[b_start:b_end]
            print(f"\n[Batch {batch_idx+1}/{n_batches}] Layers {b_start}-{b_end-1}")
            self._collect_activations(batch, calibration_data, n_samples)

            for layer_idx, (name, module) in enumerate(batch):
                global_idx = b_start + layer_idx
                is_lmhead  = 'lm_head' in name.lower()
                X_calib    = self._get_calibration_matrix(name)
                if X_calib is None or X_calib.shape[0] < 10:
                    print(f"  [{global_idx}/{n_layers}] {name}: SKIPPED")
                    continue
                debug = (global_idx < 2)

                if is_lmhead:
                    print(f"  [{global_idx}/{n_layers}] {name}: lm_head - AWQ only")
                    W        = module.weight.data
                    salience = self.base_quantizer.compute_l2_salience(
                        self.activation_data.get(name, []))
                    if salience is not None:
                        X_s    = X_calib[:min(1024, X_calib.shape[0])].to(self.device).to(W.dtype)
                        scales, _, _ = self.base_quantizer.search_best_scale(
                            W, X_s, salience.to(self.device))
                        del X_s
                        W_sc = W * scales.unsqueeze(0)
                        W_q  = self.base_quantizer.quantize_weight_groupwise_asymmetric(W_sc)
                        module.weight.data = (W_q / scales.unsqueeze(0)).to(W.dtype)
                        del W_sc, W_q
                else:
                    print(f"  [{global_idx}/{n_layers}] {name}:", end=" ", flush=True)
                    t0    = time.time()
                    stats = self._correct_layer(name, module, X_calib, debug=debug)
                    dt    = time.time() - t0
                    self.layer_stats[name] = stats
                    total_improvement.append(stats['improvement_pct'])
                    print(f"err {stats['baseline_error']:.6f}->{stats['corrected_error']:.6f} "
                          f"({stats['improvement_pct']:+.2f}%) "
                          f"MF={stats['mf_flips']} imm={stats['immediate_flips']} "
                          f"G={stats['group_flips']} CD={stats['cd_flips']}  [{dt:.1f}s]")

                del X_calib
                torch.cuda.empty_cache()
                gc.collect()

            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()
            if HAS_PSUTIL:
                print(f"  RAM: {psutil.virtual_memory().percent:.1f}%")

        elapsed = time.time() - t_start
        print(f"\n{'='*80}\nCORRECTION COMPLETE ({elapsed:.1f}s)\n{'='*80}")
        if total_improvement:
            imp = np.array(total_improvement)
            print(f"  Layers:{len(imp)}/{n_layers}  "
                  f"mean:{imp.mean():+.2f}%  median:{np.median(imp):+.2f}%  "
                  f"min:{imp.min():+.2f}%  max:{imp.max():+.2f}%")
            print(f"  Improved:{(imp>0).sum()}/{len(imp)}")
        if self.layer_stats:
            mf  = sum(s['mf_flips']       for s in self.layer_stats.values())
            imm = sum(s['immediate_flips'] for s in self.layer_stats.values())
            gf  = sum(s['group_flips']     for s in self.layer_stats.values())
            cd  = sum(s['cd_flips']        for s in self.layer_stats.values())
            print(f"  Flips MF:{mf:,} imm:{imm:,} G:{gf:,} CD:{cd:,} "
                  f"Total:{mf+imm+gf+cd:,}")

def main():
    parser = argparse.ArgumentParser(description="AWQ + Quantum-Inspired Weight Correction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path",           type=str,   default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir",           type=str,   default="./quantized_models/model_awq_quantum")
    parser.add_argument("--bits",                 type=int,   default=4, choices=[3, 4])
    parser.add_argument("--group-size",           type=int,   default=128)
    parser.add_argument("--n-grid",               type=int,   default=20)
    parser.add_argument("--n-calib",              type=int,   default=128)
    parser.add_argument("--calib-dataset",        type=str,   default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--max-tokens-per-sample",type=int,   default=2048)
    parser.add_argument("--cache-dir",            type=str,   default="./calibration_cache")
    parser.add_argument("--lambda-fidelity",      type=float, default=0.1)
    parser.add_argument("--mf-beta-init",         type=float, default=0.1)
    parser.add_argument("--mf-beta-final",        type=float, default=50.0)
    parser.add_argument("--mf-n-temps",           type=int,   default=10)
    parser.add_argument("--mf-max-iter",          type=int,   default=30)
    parser.add_argument("--sw-threshold",         type=float, default=0.1)
    parser.add_argument("--group-max-size",       type=int,   default=6)
    parser.add_argument("--cd-max-sweeps",        type=int,   default=3)
    parser.add_argument("--top-k-eigvecs",        type=int,   default=32)
    parser.add_argument("--max-calib-correction", type=int,   default=512)
    parser.add_argument("--layer-batch-size",     type=int,   default=16)
    parser.add_argument("--lmhead-chunks",        type=int,   default=4)
    parser.add_argument("--seed",                 type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*80)
    print(f"AWQ + Quantum-Inspired Weight Correction  |  Model:{args.model_path}  Device:{device}")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    print(f"\nLoading calibration: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [t['text'] for t in dataset if len(t['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    corrector = QuantumCorrectionEngine(
        model=model, tokenizer=tokenizer, device=device,
        bits=args.bits, group_size=args.group_size, n_grid=args.n_grid,
        max_tokens_per_sample=args.max_tokens_per_sample, lmhead_chunks=args.lmhead_chunks,
        lambda_fidelity=args.lambda_fidelity,
        mf_max_iter=args.mf_max_iter, mf_beta_init=args.mf_beta_init,
        mf_beta_final=args.mf_beta_final, mf_n_temps=args.mf_n_temps,
        sw_soft_threshold=args.sw_threshold, group_max_size=args.group_max_size,
        cd_max_sweeps=args.cd_max_sweeps, max_calib_samples_correction=args.max_calib_correction,
        top_k_eigvecs_G=args.top_k_eigvecs,
    )

    corrector.correct_model(calib_texts, n_samples=args.n_calib, layer_batch_size=args.layer_batch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()