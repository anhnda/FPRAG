"""
AWQ + Transverse-Field Ising Model Weight Correction
awq_tfim_correction.py

Energy formulation:
    E(s) = ||X_corr @ R(s).T||^2 / n  +  λ * ||R(s)||^2

    where R(s)[i,j] = hd[i,j]*s[i,j] - D[i,j]
          D[i,j]    = W_sc[i,j] - midpoint[i,j]   (displacement from midpoint)
          hd[i,j]   = delta[i,j] / 2               (half grid step, > 0)
          s[i,j]    ∈ {-1, +1}                     (-1→floor, +1→ceil)

    Energy change when spin (i,j) flips (ds = -2*s[i,j]):
        dv[i]      = ds * hd[i,j] * V_lam[j]
        dE_quad    = 2*(dv[i] · V_s[i])  +  ||dv[i]||^2
        dE_fid_eff = 2λ * ds * hd[i,j] * R[i,j]   (constant 4λ*hd^2 dropped)
        dE_total   = dE_quad + dE_fid_eff

    where V_s[i] = R[i] @ V_lam  is the lam-weighted residual projection,
    and   V_lam  = V * sqrt(lam)  absorbs eigenvalues once (no separate lam_dev).

Key design:
    1. V_s is centered on the *residual* R(s)=hd*s-D, not on hd*s.
       The quadratic dE_quad is then exact w.r.t. the true objective.
    2. sqrt(lam) absorbed into V_lam once; all inner products auto-weighted.
    3. Γ_i = γ*(1-|D_i/hd_i|) selects uncertain spins; certain spins frozen.
    4. Phase 1: group exhaustive search, Phase 2: CD cleanup (uncertain only).
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


# ─────────────────────────────────────────────────────────────────────────────
# AWQ base quantizer
# ─────────────────────────────────────────────────────────────────────────────

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
        exact_mask = (W_int_floor == W_int_ceil)
        W_int_ceil[exact_mask  & (W_int_ceil  < max_int)] += 1
        W_int_floor[exact_mask & (W_int_floor > 0)]       -= 1
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


# ─────────────────────────────────────────────────────────────────────────────
# TFIM correction engine
# ─────────────────────────────────────────────────────────────────────────────

class TFIsingCorrectionEngine:
    """
    Transverse-Field Ising Model correction engine.

    Γ_i = γ*(1 - |D_i/hd_i|) selects uncertain spins (near midpoint).
    Only uncertain spins are candidates for correction.
    Near-gridpoint spins (Γ ≈ 0) are frozen at nearest rounding.
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 n_grid=20, max_tokens_per_sample=2048,
                 lambda_fidelity=1.0, gamma=1.0, gamma_threshold=0.7,
                 group_max_size=6, cd_max_sweeps=3,
                 max_calib_samples=512, top_k_eigvecs=32,
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
        self.top_k_eigvecs         = top_k_eigvecs
        self.layer_batch_size      = layer_batch_size
        self.max_rows              = max_rows
        self.base_quantizer        = AWQBaseQuantizer(
            bits=bits, group_size=group_size, n_grid=n_grid)
        self.activation_data       = {}
        self.layer_stats           = {}

        print(f"\n{'='*80}")
        print(f"Transverse-Field Ising AWQ Correction Engine")
        print(f"{'='*80}")
        print(f"  Bits:{bits}  GroupSize:{group_size}  "
              f"lambda_fid:{lambda_fidelity}  gamma:{gamma}  "
              f"gamma_threshold:{gamma_threshold}")
        print(f"  Group max:{group_max_size}(2^g={2**group_max_size})  "
              f"CD sweeps:{cd_max_sweeps}  Low-rank k:{top_k_eigvecs}")
        print(f"  Max rows/layer:{max_rows}  Calib samples:{max_calib_samples}")
        print(f"{'='*80}\n")

    # ── Activation collection ─────────────────────────────────────────────────

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

    # ── Group clustering ──────────────────────────────────────────────────────

    def _cluster_spins(self, J_sub, indices):
        """Greedy clustering by coupling strength — CPU numpy, no GPU syncs."""
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

    # ── Core per-layer correction ─────────────────────────────────────────────

# ── Core per-layer correction (FIXED) ────────────────────────────────────

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

        # ── Scaled-space setup ────────────────────────────────────────────────
        W_scaled        = W * best_scales.unsqueeze(0)
        grid_info       = self.base_quantizer.get_quantization_grid_info(W_scaled)
        X_corr          = X_calib[:min(self.max_calib_samples,
                                       X_calib.shape[0])].to(device).float()
        W_sc_f32        = W_scaled.float()
        baseline_W_q_sc = grid_info['nearest']
        Y_orig          = X_corr @ W_sc_f32.t()
        Y_base          = X_corr @ baseline_W_q_sc.float().t()
        baseline_error  = (Y_orig - Y_base).pow(2).mean().item()
        if debug:
            print(f"    Baseline error: {baseline_error:.8f}")

        # ── Low-rank G = X_corr^T X_corr / n, absorb sqrt(lam) into V ────────
        n_tok = X_corr.shape[0]
        k     = min(self.top_k_eigvecs, n_tok, in_features)
        U, S_sv, V = torch.svd_lowrank(X_corr.float(), q=k, niter=4)
        lam    = (S_sv ** 2) / n_tok                                   # [k]
        del U, S_sv
        torch.cuda.empty_cache()
        if debug:
            print(f"    G: top-{k}, lam_1={lam[0]:.4f}, lam_k={lam[-1]:.6f}")

        lam_sqrt = lam.sqrt().to(device)
        V_lam    = V.to(device) * lam_sqrt.unsqueeze(0)                # [in, k]
        del V, lam, lam_sqrt
        torch.cuda.empty_cache()

        # ── Grid geometry ─────────────────────────────────────────────────────
        midpoint   = grid_info['midpoint'].float().to(device)          # [out, in]
        delta_all  = grid_info['delta'].float().to(device)             # [out, in]
        nearest    = grid_info['nearest'].float().to(device)           # [out, in]
        half_delta = delta_all / 2                                     # hd[i,j]

        D_all = W_sc_f32 - midpoint                                    # [out, in]

        S_nearest                 = torch.sign(D_all)
        S_nearest[S_nearest == 0] = 1.0

        Gamma = self.gamma * (
            1.0 - (D_all.abs() / half_delta.clamp(min=1e-10)).clamp(0, 1))
        uncertain_mask  = Gamma > self.gamma_threshold
        total_uncertain = uncertain_mask.sum().item()

        if debug:
            frac_unc   = total_uncertain / (out_features * in_features)
            unc_ratio  = (D_all.abs() / half_delta.clamp(min=1e-10))[uncertain_mask]
            cert_ratio = (D_all.abs() / half_delta.clamp(min=1e-10))[~uncertain_mask]
            print(f"    Uncertain: {total_uncertain} ({frac_unc*100:.1f}%) "
                  f"|D/hd| unc:{unc_ratio.mean():.3f} cert:{cert_ratio.mean():.3f}")
            del unc_ratio, cert_ratio

        diag_J = (V_lam ** 2).sum(dim=1)                               # [in]

        # ── Sanity helper: compute the actual MSE of a state ────────────────
        def _state_mse(S_state):
            W_q_sc = midpoint + half_delta * S_state
            Y_q    = X_corr @ W_q_sc.float().t()
            return (Y_orig - Y_q).pow(2).mean().item()

        if debug:
            mse_init = _state_mse(S_nearest)
            print(f"    MSE check @ nearest: {mse_init:.8f} "
                  f"(should match baseline {baseline_error:.8f})")

        # ── Phase 1: Group exhaustive search on uncertain spins ───────────────
        S_refined         = S_nearest.clone()
        total_group_flips = 0
        R_state           = half_delta * S_refined - D_all             # [out, in]

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
                ratio    = (D_all[row_idx].abs() /
                            half_delta[row_idx].clamp(min=1e-10))
                _, order = ratio[unc_idx].sort()
                unc_idx  = unc_idx[order[:50]]
                n_unc    = 50

            hd_row  = half_delta[row_idx]
            R_row   = R_state[row_idx]
            J_V_row = hd_row.unsqueeze(1) * V_lam
            J_V_unc = J_V_row[unc_idx]
            J_sub   = J_V_unc @ J_V_unc.t()
            groups  = self._cluster_spins(J_sub, unc_idx)

            s_row = S_refined[row_idx].clone()
            v_row = R_row @ V_lam                                      # [k]

            for group in groups:
                g = len(group)
                if g == 0:
                    continue
                group_idx = torch.tensor(group, device=device, dtype=torch.long)
                s_g  = s_row[group_idx]
                J_g  = J_V_row[group_idx]
                hd_g = hd_row[group_idx]
                R_g  = R_row[group_idx]

                if g <= self.group_max_size:
                    n_configs = 2 ** g
                    bit_idx   = torch.arange(
                        n_configs, device=device).unsqueeze(1)
                    bit_pos   = torch.arange(
                        g,         device=device).unsqueeze(0)
                    flip_mat  = ((bit_idx >> bit_pos) & 1).float()
                    delta_sg  = -2.0 * flip_mat * s_g.unsqueeze(0)
                    delta_v   = delta_sg @ J_g

                    dE_quad = (delta_v * (2.0 * v_row.unsqueeze(0) + delta_v)
                               ).sum(dim=1)
                    dE_fid  = (2.0 * self.lambda_fidelity *
                               (delta_sg * hd_g.unsqueeze(0) * R_g.unsqueeze(0))
                               ).sum(dim=1)
                    dE = dE_quad + dE_fid

                    best_idx = dE.argmin()
                    if dE[best_idx] < -1e-10:
                        best_flip = flip_mat[best_idx].bool()
                        ds_best   = delta_sg[best_idx]
                        R_row[group_idx] = R_row[group_idx] + ds_best * hd_g
                        s_row[group_idx[best_flip]] *= -1
                        total_group_flips += best_flip.sum().item()
                        v_row = v_row + delta_v[best_idx]
                else:
                    for idx in group:
                        j_v  = J_V_row[idx]
                        hd_j = hd_row[idx]
                        R_j  = R_row[idx]
                        ds   = -2.0 * s_row[idx]
                        dv   = ds * j_v
                        dE_quad = ((2.0 * v_row + dv) * dv).sum()
                        dE_fid  = 2.0 * self.lambda_fidelity * ds * hd_j * R_j
                        if (dE_quad + dE_fid).item() < -1e-10:
                            R_row[idx] = R_row[idx] + ds * hd_j
                            s_row[idx] *= -1
                            v_row      = v_row + dv
                            total_group_flips += 1

            S_refined[row_idx] = s_row

        del rows_with_uncertain

        if debug:
            mse_after_p1 = _state_mse(S_refined)
            print(f"    MSE after Phase 1: {mse_after_p1:.8f} "
                  f"(Δ vs baseline: {mse_after_p1 - baseline_error:+.2e})")

        # ── Phase 2: CD cleanup — COLUMN-SEQUENTIAL (FIXED) ────────────────────
        # Process one column at a time. All rows in a column are mutually
        # independent (they don't share any R[i,j]), so they flip in parallel
        # without cross-term errors. V_s is updated before the next column,
        # so inter-column cross-terms are captured exactly.

        S_final        = S_refined.clone()
        R_cd           = half_delta * S_final - D_all                  # recompute
        V_s            = R_cd @ V_lam                                  # [out, k]
        total_cd_flips = 0

        for sweep in range(self.cd_max_sweeps):
            sweep_flips = 0

            # Optional: permute column order each sweep to reduce bias
            col_order = torch.randperm(in_features, device=device)

            for j_perm in col_order.tolist():
                j = j_perm
                unc_col = uncertain_mask[:, j]
                if not unc_col.any():
                    continue

                s_col  = S_final[:, j]                                 # [out]
                hd_col = half_delta[:, j]                              # [out]
                R_col  = R_cd[:, j]                                    # [out]
                v_j    = V_lam[j, :]                                   # [k]
                diag_j = diag_J[j]                                     # scalar

                ds_col     = -2.0 * s_col
                VsVj       = V_s @ v_j                                 # [out]
                term_quad1 = 2.0 * ds_col * hd_col * VsVj
                term_quad2 = (ds_col * hd_col) ** 2 * diag_j
                term_fid   = 2.0 * self.lambda_fidelity * ds_col * hd_col * R_col
                dE         = term_quad1 + term_quad2 + term_fid

                flip_mask = (dE < -1e-10) & unc_col
                n_flips   = flip_mask.sum().item()
                if n_flips == 0:
                    continue

                ds_accepted = torch.where(
                    flip_mask, ds_col, torch.zeros_like(ds_col))
                dR_col      = ds_accepted * hd_col

                R_cd[:, j] = R_col + dR_col
                V_s        = V_s + dR_col.unsqueeze(1) * v_j.unsqueeze(0)
                S_final[:, j] = torch.where(flip_mask, -s_col, s_col)

                sweep_flips += n_flips

            total_cd_flips += sweep_flips
            if debug:
                mse_sweep = _state_mse(S_final)
                print(f"    CD sweep {sweep}: flips={sweep_flips}  "
                      f"MSE={mse_sweep:.8f}")
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

        del (V_lam, diag_J, Gamma, half_delta,
             uncertain_mask, S_nearest, S_refined, S_final,
             midpoint, delta_all, nearest, W_sc_f32, D_all,
             R_state, R_cd, V_s, W_corrected,
             X_corr, Y_orig, Y_base, Y_corrected, baseline_W_q_sc)
        torch.cuda.empty_cache()
        return stats
    # ── Model-level loop ──────────────────────────────────────────────────────

    def correct_model(self, calibration_data, n_samples=128):
        print(f"\n{'='*80}\nTFIM WEIGHT CORRECTION\n{'='*80}")
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
            print(f"\n[Batch {batch_idx+1}/{n_batches}] "
                  f"Layers {b_start}-{b_end-1}")
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
                        module.weight.data = (
                            W_q / scales.unsqueeze(0)).to(W.dtype)
                        del W_sc, W_q
                else:
                    print(f"  [{global_idx}/{n_layers}] {name}:",
                          end=" ", flush=True)
                    t0    = time.time()
                    stats = self._correct_layer(
                        name, module, X_calib, debug=debug)
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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AWQ + Transverse-Field Ising Correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path",            type=str,
                        default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir",            type=str,
                        default="./quantized_models/model_tfim")
    parser.add_argument("--bits",                  type=int,
                        default=4, choices=[3, 4])
    parser.add_argument("--group-size",            type=int,   default=128)
    parser.add_argument("--n-grid",                type=int,   default=20)
    parser.add_argument("--n-calib",               type=int,   default=128)
    parser.add_argument("--calib-dataset",         type=str,   default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--max-tokens-per-sample", type=int,   default=2048)
    parser.add_argument("--cache-dir",             type=str,
                        default="./calibration_cache")
    parser.add_argument("--lambda-fidelity",       type=float, default=1.0)
    parser.add_argument("--gamma",                 type=float, default=1.0,
                        help="Transverse field scale")
    parser.add_argument("--gamma-threshold",       type=float, default=0.7,
                        help="Gamma threshold for uncertain spins (0-1)")
    parser.add_argument("--group-max-size",        type=int,   default=6)
    parser.add_argument("--cd-max-sweeps",         type=int,   default=3)
    parser.add_argument("--top-k-eigvecs",         type=int,   default=32)
    parser.add_argument("--max-calib-correction",  type=int,   default=512)
    parser.add_argument("--max-rows",              type=int,   default=512,
                        help="Max rows per layer for group refinement")
    parser.add_argument("--layer-batch-size",      type=int,   default=16)
    parser.add_argument("--seed",                  type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"AWQ + TFIM Correction  |  Model:{args.model_path}  Device:{device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
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
        top_k_eigvecs=args.top_k_eigvecs,
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