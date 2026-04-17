"""
AWQ + Transverse-Field Ising Model Weight Correction
awq_tfim_correction.py  (v3 — EXACT G, no low-rank approximation)

Fundamental change from v2: abandon the low-rank G approximation.
------------------------------------------------------------------
v2 used G ≈ V·Λ·V^T with rank-k SVD of X_corr. Unit tests confirmed this
made CD non-monotone: flips that reduce rank-k energy can INCREASE true
energy via the off-subspace tail, accumulating to massive errors over
millions of flips.

v3 uses G = X^T X / n exactly (shape [in, in], ~64MB fp32 for in=4096).
All dE formulas use exact G and G[j,j]. CD maintains RG = R @ G
incrementally: after flipping column j, RG changes by dR ⊗ G[j,:].

Cumulative fixes (v1 → v2 → v3):
  - removed exact_mask expansion in get_quantization_grid_info
  - column-sequential CD (no chunked-parallel race)
  - degenerate (hd=0) spins frozen via 'active' mask
  - S_nearest via nearest==ceil comparison (banker's-rounding safe)
  - correct fidelity delta: λ[2·ds·hd·R + (ds·hd)²]
  - EXACT G throughout (this version)

Energy formulation:
    E(s) = ||X_corr @ R(s).T||² / n  +  λ · ||R(s)||²
    R(s)[i,j] = hd[i,j]·s[i,j] - D[i,j]

Single-spin flip delta (ds = -2s):
    dE_quad = 2·ds·hd·(RG)[i,j] + (ds·hd)²·G[j,j]
    dE_fid  = λ·[2·ds·hd·R[i,j] + (ds·hd)²]
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
        W_g   = W_pad.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2 ** self.bits - 1
        scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        zp    = torch.round(-w_min / scale).clamp(0, max_int)
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
        padded   = n_groups * self.group_size
        max_int  = 2 ** self.bits - 1
        if padded > in_features:
            W_pad = torch.zeros(out_features, padded, device=W.device, dtype=W.dtype)
            W_pad[:, :in_features] = W
        else:
            W_pad = W
        W_g   = W_pad.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        zp    = torch.round(-w_min / scale).clamp(0, max_int)
        W_div         = W_g / scale + zp
        W_int_nearest = torch.round(W_div).clamp(0, max_int)
        W_int_floor   = torch.floor(W_div).clamp(0, max_int)
        W_int_ceil    = torch.ceil(W_div).clamp(0, max_int)
        floor_val   = (W_int_floor   - zp) * scale
        ceil_val    = (W_int_ceil    - zp) * scale
        nearest_val = (W_int_nearest - zp) * scale
        floor_val   = floor_val.reshape(out_features, padded)[:, :in_features]
        ceil_val    = ceil_val.reshape(out_features, padded)[:, :in_features]
        nearest_val = nearest_val.reshape(out_features, padded)[:, :in_features]
        return {
            'floor':    floor_val,
            'ceil':     ceil_val,
            'nearest':  nearest_val,
            'midpoint': (floor_val + ceil_val) / 2,
            'delta':    ceil_val - floor_val,
        }

    @torch.no_grad()
    def compute_l2_salience(self, activation_data):
        if not activation_data:
            return None
        total   = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in activation_data)
        in_feat = activation_data[0].shape[-1]
        acc = torch.zeros(in_feat, dtype=torch.float32)
        for x in activation_data:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            acc   += x_flat.pow(2).sum(dim=0)
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
            alpha    = grid_idx / self.n_grid
            scales   = activation_salience.pow(alpha)
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


class TFIsingCorrectionEngine:
    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 n_grid=20, max_tokens_per_sample=2048,
                 lambda_fidelity=0.0, gamma=1.0, gamma_threshold=0.7,
                 group_max_size=6, cd_max_sweeps=3,
                 max_calib_samples=512,
                 layer_batch_size=16, max_rows=512):
        self.model                 = model
        self.tokenizer             = tokenizer
        self.device                = device
        self.bits                  = bits
        self.group_size            = group_size
        self.n_grid                = n_grid
        self.max_tokens_per_sample = max_tokens_per_sample
        self.lambda_fidelity       = lambda_fidelity
        self.gamma                 = gamma
        self.gamma_threshold       = gamma_threshold
        self.group_max_size        = group_max_size
        self.cd_max_sweeps         = cd_max_sweeps
        self.max_calib_samples     = max_calib_samples
        self.layer_batch_size      = layer_batch_size
        self.max_rows              = max_rows
        self.base_quantizer        = AWQBaseQuantizer(
            bits=bits, group_size=group_size, n_grid=n_grid)
        self.activation_data       = {}
        self.layer_stats           = {}

        print(f"\n{'='*80}")
        print(f"TFIM AWQ Correction v3 (exact G)")
        print(f"{'='*80}")
        print(f"  Bits:{bits}  GroupSize:{group_size}  "
              f"lambda_fid:{lambda_fidelity}  gamma:{gamma}  "
              f"gamma_threshold:{gamma_threshold}")
        print(f"  Group max:{group_max_size}(2^g={2**group_max_size})  "
              f"CD sweeps:{cd_max_sweeps}")
        print(f"  Max rows/layer:{max_rows}  Calib samples:{max_calib_samples}")
        print(f"{'='*80}\n")

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
    def _get_calibration_matrix(self, name):
        if name not in self.activation_data or not self.activation_data[name]:
            return None
        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0).float()
        if X.shape[0] > self.max_calib_samples:
            idx = torch.randperm(X.shape[0])[:self.max_calib_samples]
            X   = X[idx]
        return X

    def _cluster_spins(self, J_sub, indices):
        n = J_sub.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [[indices[0].item()]]
        J_np  = J_sub.abs().cpu().numpy()
        si_np = indices.cpu().numpy()
        np.fill_diagonal(J_np, 0)
        row_sums = J_np.sum(axis=1)
        used     = np.zeros(n, dtype=bool)
        groups   = []
        while not used.all():
            tmp        = row_sums.copy()
            tmp[used]  = -1.0
            seed       = int(tmp.argmax())
            group      = [seed]
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

    @torch.no_grad()
    def _correct_layer(self, name, module, X_calib, debug=False):
        W = module.weight.data
        original_dtype            = W.dtype
        out_features, in_features = W.shape
        device                    = W.device

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

        # ── Setup ────────────────────────────────────────────────────────────
        W_scaled        = W * best_scales.unsqueeze(0)
        grid_info       = self.base_quantizer.get_quantization_grid_info(W_scaled)
        X_corr          = X_calib[:min(self.max_calib_samples,
                                       X_calib.shape[0])].to(device).float()
        W_sc_f32        = W_scaled.float()
        baseline_W_q_sc = grid_info['nearest'].float().to(device)
        Y_orig          = X_corr @ W_sc_f32.t()
        Y_base          = X_corr @ baseline_W_q_sc.t()
        baseline_error  = (Y_orig - Y_base).pow(2).mean().item()
        if debug:
            print(f"    Baseline error: {baseline_error:.8f}")

        # ── Exact G ──────────────────────────────────────────────────────────
        n_tok  = X_corr.shape[0]
        G      = (X_corr.t() @ X_corr) / n_tok
        #percdamp = 0.1
        #damp = percdamp * torch.diagonal(G).median()
        #G.diagonal().add_(damp)
        diag_G = torch.diagonal(G).clone()

        # if debug:
        #     diag_raw = diag_G - damp   # recover pre-damp diagonal
        #     print(f"    Ridge: percdamp={percdamp}  damp={damp.item():.4e}")
        #     print(f"    diag_G pre-damp:  min={diag_raw.min().item():.4e}  "
        #         f"median={diag_raw.median().item():.4e}  "
        #         f"max={diag_raw.max().item():.4e}")
        #     print(f"    diag_G post-damp: min={diag_G.min().item():.4e}  "
        #         f"median={diag_G.median().item():.4e}  "
        #         f"max={diag_G.max().item():.4e}")
        #     ratio = damp / diag_raw.clamp(min=1e-20)
        #     n_dominated = (ratio > 1.0).sum().item()
        #     print(f"    Directions where damp > G_jj: {n_dominated}/{in_features}")

        if debug:
            mem_mb = G.element_size() * G.nelement() / 1e6
            print(f"    Exact G: [{in_features},{in_features}]  "
                  f"trace={diag_G.sum().item():.4f}  mem={mem_mb:.1f}MB")

        # ── Grid geometry ─────────────────────────────────────────────────────
        midpoint   = grid_info['midpoint'].float().to(device)
        delta_all  = grid_info['delta'].float().to(device)
        nearest    = grid_info['nearest'].float().to(device)
        ceil_all   = grid_info['ceil'].float().to(device)
        half_delta = delta_all / 2
        active     = delta_all > 1e-12
        D_all      = W_sc_f32 - midpoint

        S_nearest = torch.where(
            nearest >= ceil_all - 1e-12,
            torch.ones_like(D_all),
            -torch.ones_like(D_all),
        )
        del ceil_all

        if debug:
            W_recon  = midpoint + half_delta * S_nearest
            max_diff = (W_recon - nearest).abs().max().item()
            print(f"    Sanity: reconstruction vs nearest max diff = {max_diff:.2e}")
            del W_recon

        ratio = torch.where(
            active,
            D_all.abs() / half_delta.clamp(min=1e-10),
            torch.ones_like(D_all),
        )
        Gamma           = self.gamma * (1.0 - ratio.clamp(0, 1))
        uncertain_mask  = (Gamma > self.gamma_threshold) & active
        total_uncertain = uncertain_mask.sum().item()
        n_degenerate    = (~active).sum().item()

        if debug:
            frac_unc = total_uncertain / (out_features * in_features)
            frac_deg = n_degenerate    / (out_features * in_features)
            print(f"    Uncertain: {total_uncertain} ({frac_unc*100:.1f}%)  "
                  f"degenerate: {n_degenerate} ({frac_deg*100:.2f}%)")

        def _state_mse(S_state):
            W_q_sc = midpoint + half_delta * S_state
            Y_q    = X_corr @ W_q_sc.float().t()
            return (Y_orig - Y_q).pow(2).mean().item()

        if debug:
            mse_init = _state_mse(S_nearest)
            print(f"    MSE @ S_nearest: {mse_init:.8f}  "
                  f"baseline: {baseline_error:.8f}  "
                  f"Δ: {mse_init - baseline_error:+.2e}")

        # ── Phase 1: group exhaustive (exact G) ──────────────────────────────
        S_refined         = S_nearest.clone()
        total_group_flips = 0
        R_state           = half_delta * S_refined - D_all

        rows_with_uncertain = (uncertain_mask.sum(dim=1) >= 2).nonzero(
            as_tuple=True)[0]
        if rows_with_uncertain.shape[0] > self.max_rows:
            row_counts          = uncertain_mask.sum(dim=1)
            _, top_rows         = row_counts.topk(self.max_rows)
            rows_with_uncertain = top_rows

        for row_idx in rows_with_uncertain.tolist():
            unc_idx = uncertain_mask[row_idx].nonzero(as_tuple=True)[0]
            n_unc   = unc_idx.shape[0]
            if n_unc == 0:
                continue
            if n_unc > 50:
                r = D_all[row_idx].abs() / half_delta[row_idx].clamp(min=1e-10)
                _, order = r[unc_idx].sort()
                unc_idx  = unc_idx[order[:50]]
                n_unc    = 50

            hd_row = half_delta[row_idx]
            R_row  = R_state[row_idx]
            s_row  = S_refined[row_idx].clone()

            hd_unc = hd_row[unc_idx]
            G_unc  = G[unc_idx][:, unc_idx]
            J_sub  = G_unc * (hd_unc.unsqueeze(0) * hd_unc.unsqueeze(1))
            groups = self._cluster_spins(J_sub, unc_idx)

            for group in groups:
                g = len(group)
                if g == 0:
                    continue
                group_idx = torch.tensor(group, device=device, dtype=torch.long)
                s_g  = s_row[group_idx]
                hd_g = hd_row[group_idx]
                R_g  = R_row[group_idx]

                if g <= self.group_max_size:
                    n_configs = 2 ** g
                    bit_idx   = torch.arange(n_configs, device=device).unsqueeze(1)
                    bit_pos   = torch.arange(g,         device=device).unsqueeze(0)
                    flip_mat  = ((bit_idx >> bit_pos) & 1).float()
                    delta_sg  = -2.0 * flip_mat * s_g.unsqueeze(0)
                    dshd_g    = delta_sg * hd_g.unsqueeze(0)         # [2^g, g]

                    # Exact quadratic delta:
                    #   ΔE_quad = 2·<dshd_g, (R·G)[group]> + <dshd_g, G_sub·dshd_g>
                    RG_row_group = R_row @ G[:, group_idx]           # [g]
                    G_sub        = G[group_idx][:, group_idx]        # [g, g]
                    lin  = 2.0 * (dshd_g * RG_row_group.unsqueeze(0)).sum(dim=1)
                    quad = (dshd_g @ G_sub * dshd_g).sum(dim=1)
                    dE_quad = lin + quad

                    dE_fid = self.lambda_fidelity * (
                        2.0 * dshd_g * R_g.unsqueeze(0) + dshd_g ** 2
                    ).sum(dim=1)

                    dE       = dE_quad + dE_fid
                    best_idx = dE.argmin()
                    if dE[best_idx] < -1e-10:
                        best_flip = flip_mat[best_idx].bool()
                        dshd_best = dshd_g[best_idx]
                        R_row[group_idx] = R_row[group_idx] + dshd_best
                        s_row[group_idx[best_flip]] *= -1
                        total_group_flips += best_flip.sum().item()
                else:
                    for idx in group:
                        hd_j = hd_row[idx]
                        ds   = -2.0 * s_row[idx]
                        dshd = ds * hd_j
                        RG_idx  = (R_row * G[:, idx]).sum()
                        dE_quad = 2.0 * dshd * RG_idx + dshd ** 2 * diag_G[idx]
                        dE_fid  = self.lambda_fidelity * (
                            2.0 * dshd * R_row[idx] + dshd ** 2)
                        if (dE_quad + dE_fid).item() < -1e-10:
                            R_row[idx] += dshd
                            s_row[idx] *= -1
                            total_group_flips += 1

            S_refined[row_idx] = s_row
            # R_state[row_idx] already updated via R_row view

        del rows_with_uncertain

        if debug:
            mse_p1 = _state_mse(S_refined)
            print(f"    MSE after Phase 1: {mse_p1:.8f}  "
                  f"Δ vs baseline: {mse_p1 - baseline_error:+.2e}")

        # ── Phase 2: column-sequential CD with exact G ────────────────────────
        # Maintain RG[i,:] = R[i,:] @ G
        S_final        = S_refined.clone()
        R_cd           = half_delta * S_final - D_all
        RG             = R_cd @ G                                   # [out, in]
        total_cd_flips = 0

        for sweep in range(self.cd_max_sweeps):
            sweep_flips = 0
            col_order   = torch.randperm(in_features, device=device).tolist()

            for j in col_order:
                unc_col = uncertain_mask[:, j]
                if not unc_col.any():
                    continue

                s_col  = S_final[:, j]
                hd_col = half_delta[:, j]
                R_col  = R_cd[:, j]
                RG_col = RG[:, j]
                Gjj    = diag_G[j]
                G_col  = G[:, j]

                ds_col   = -2.0 * s_col
                dshd_col = ds_col * hd_col
                dE_quad  = 2.0 * dshd_col * RG_col + dshd_col ** 2 * Gjj
                dE_fid   = self.lambda_fidelity * (
                    2.0 * dshd_col * R_col + dshd_col ** 2)
                dE       = dE_quad + dE_fid
                dE_threshold = -3.0 * torch.std(dE[unc_col]).item()  # 3-sigma cutoff
                flip_mask = (dE < dE_threshold) & unc_col
                n_flips   = flip_mask.sum().item()
                if n_flips == 0:
                    continue

                ds_acc = torch.where(flip_mask, ds_col, torch.zeros_like(ds_col))
                dR     = ds_acc * hd_col

                R_cd[:, j]    = R_col + dR
                RG            = RG + dR.unsqueeze(1) * G_col.unsqueeze(0)
                S_final[:, j] = torch.where(flip_mask, -s_col, s_col)
                sweep_flips  += n_flips

            total_cd_flips += sweep_flips
            if debug:
                mse_s = _state_mse(S_final)
                print(f"    CD sweep {sweep}: flips={sweep_flips}  "
                      f"MSE={mse_s:.8f}  Δ vs baseline: {mse_s - baseline_error:+.2e}")
            if sweep_flips == 0:
                break

        # ── Reconstruct ───────────────────────────────────────────────────────
        W_corrected        = (midpoint + half_delta * S_final).to(original_dtype)
        W_final            = W_corrected / best_scales.unsqueeze(0)
        module.weight.data = W_final

        Y_corrected     = X_corr @ W_corrected.float().t()
        corrected_error = (Y_orig - Y_corrected).pow(2).mean().item()
        improvement     = (baseline_error - corrected_error) \
                          / max(baseline_error, 1e-12) * 100

        if debug:
            print(f"    Corrected: {corrected_error:.8f} ({improvement:+.2f}%)")
            print(f"    Group flips:{total_group_flips}  CD flips:{total_cd_flips}")

        stats = {
            'awq_alpha':       best_alpha,
            'baseline_error':  baseline_error,
            'corrected_error': corrected_error,
            'improvement_pct': improvement,
            'uncertain':       total_uncertain,
            'group_flips':     total_group_flips,
            'cd_flips':        total_cd_flips,
        }

        del (G, diag_G, RG, Gamma, half_delta, active,
             uncertain_mask, S_nearest, S_refined, S_final,
             midpoint, delta_all, nearest, W_sc_f32, D_all,
             R_state, R_cd, W_corrected,
             X_corr, Y_orig, Y_base, Y_corrected, baseline_W_q_sc)
        torch.cuda.empty_cache()
        gc.collect()
        return stats

    def correct_model(self, calibration_data, n_samples=128):
        print(f"\n{'='*80}\nTFIM WEIGHT CORRECTION (exact G)\n{'='*80}")
        layer_list = [(name, module)
                      for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear)]
        n_layers  = len(layer_list)
        n_batches = (n_layers + self.layer_batch_size - 1) // self.layer_batch_size
        print(f"  Layers:{n_layers}  Batches:{n_batches}  Calib:{n_samples}")

        total_improvement = []
        t_start = time.time()

        for batch_idx in range(n_batches):
            b_start = batch_idx * self.layer_batch_size
            b_end   = min(b_start + self.layer_batch_size, n_layers)
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
                    print(f"  [{global_idx}/{n_layers}] {name}: lm_head — AWQ only")
                    W        = module.weight.data
                    salience = self.base_quantizer.compute_l2_salience(
                        self.activation_data.get(name, []))
                    if salience is not None:
                        X_s = X_calib[:min(1024, X_calib.shape[0])].to(
                            self.device).to(W.dtype)
                        scales, _, _ = self.base_quantizer.search_best_scale(
                            W, X_s, salience.to(self.device))
                        del X_s
                        W_sc = W * scales.unsqueeze(0)
                        W_q  = self.base_quantizer\
                            .quantize_weight_groupwise_asymmetric(W_sc)
                        module.weight.data = (W_q / scales.unsqueeze(0)).to(W.dtype)
                        del W_sc, W_q
                else:
                    print(f"  [{global_idx}/{n_layers}] {name}:", end=" ", flush=True)
                    t0    = time.time()
                    stats = self._correct_layer(name, module, X_calib, debug=debug)
                    if name in self.activation_data:
                        del self.activation_data[name]
                    gc.collect()
                    dt    = time.time() - t0
                    self.layer_stats[name] = stats
                    total_improvement.append(stats['improvement_pct'])
                    print(f"err {stats['baseline_error']:.6f}->"
                          f"{stats['corrected_error']:.6f} "
                          f"({stats['improvement_pct']:+.2f}%) "
                          f"unc={stats['uncertain']} "
                          f"G={stats['group_flips']} "
                          f"CD={stats['cd_flips']}  [{dt:.1f}s]")

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
                  f"mean:{imp.mean():+.2f}%  "
                  f"median:{np.median(imp):+.2f}%  "
                  f"min:{imp.min():+.2f}%  "
                  f"max:{imp.max():+.2f}%")
            print(f"  Improved:{(imp > 0).sum()}/{len(imp)}")
        if self.layer_stats:
            gf = sum(s['group_flips'] for s in self.layer_stats.values())
            cd = sum(s['cd_flips']    for s in self.layer_stats.values())
            print(f"  Flips  G:{gf:,}  CD:{cd:,}  Total:{gf+cd:,}")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ + TFIM Correction v3 (exact G)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path",            type=str, default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir",            type=str, default="./quantized_models/model_tfim")
    parser.add_argument("--bits",                  type=int, default=4, choices=[3, 4])
    parser.add_argument("--group-size",            type=int, default=128)
    parser.add_argument("--n-grid",                type=int, default=20)
    parser.add_argument("--n-calib",               type=int, default=128)
    parser.add_argument("--calib-dataset",         type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--cache-dir",             type=str, default="./calibration_cache")
    parser.add_argument("--lambda-fidelity",       type=float, default=0.0,
                        help="0 = pure MSE minimization (recommended)")
    parser.add_argument("--gamma",                 type=float, default=1.0)
    parser.add_argument("--gamma-threshold",       type=float, default=0.85)
    parser.add_argument("--group-max-size",        type=int, default=6)
    parser.add_argument("--cd-max-sweeps",         type=int, default=3)
    parser.add_argument("--max-calib-correction",  type=int, default=2048)
    parser.add_argument("--max-rows",              type=int, default=512)
    parser.add_argument("--layer-batch-size",      type=int, default=16)
    parser.add_argument("--seed",                  type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"AWQ + TFIM Correction v3  |  Model:{args.model_path}  Device:{device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    print(f"\nLoading calibration: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset     = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [t['text'] for t in dataset
                       if len(t['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)

    corrector = TFIsingCorrectionEngine(
        model=model, tokenizer=tokenizer, device=device,
        bits=args.bits, group_size=args.group_size, n_grid=args.n_grid,
        max_tokens_per_sample=args.max_tokens_per_sample,
        lambda_fidelity=args.lambda_fidelity,
        gamma=args.gamma, gamma_threshold=args.gamma_threshold,
        group_max_size=args.group_max_size,
        cd_max_sweeps=args.cd_max_sweeps,
        max_calib_samples=args.max_calib_correction,
        layer_batch_size=args.layer_batch_size,
        max_rows=args.max_rows,
    )

    corrector.correct_model(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()