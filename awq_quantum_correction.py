"""
AWQ + Transverse-Field Ising Model Weight Correction
awq_tfim_correction.py

Energy formulation:
    E(s) = ||X_corr @ (hd*s).T||^2_G / n  +  λ||hd*s - D||^2

    where G = X_corr^T X_corr / n ≈ V diag(lam) V^T  (low-rank SVD)

Spin variables: s_i ∈ {-1, +1}
    s=+1 → ceil,  s=-1 → floor
    D_i = W_sc_i - midpoint_i  (displacement from midpoint)
    hd_i = delta_i / 2         (half grid step, always > 0)

Transverse field (quantum insight):
    Γ_i = γ * (1 - |D_i / hd_i|)
    Γ→1: near midpoint  → uncertain, worth correcting
    Γ→0: near grid point → certain,  leave at nearest

Local field (fidelity only — G_D unreliable at low rank):
    H_i = -2 * hd_i * λ * D_i

Key fix: lam is absorbed into V_lam = V * sqrt(lam) once, so that all
inner-product computations V_s @ V_lam.t() are automatically lam-weighted.
This ensures Phase 1 and Phase 2 use a consistent energy metric and the
dE sign/magnitude is correct, preventing bad flips.

Correction: group exhaustive search on uncertain spins,
            followed by CD cleanup restricted to uncertain spins.
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
        max_int = 2 ** self.bits - 1
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
        max_int = 2 ** self.bits - 1
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
    """
    Transverse-Field Ising Model correction engine.

    Key insight: Γ_i = γ*(1 - |D_i/hd_i|) identifies which weights are
    genuinely uncertain (near midpoint between floor and ceil).
    Only these weights are candidates for rounding correction.
    Near-gridpoint weights are left at nearest rounding — they are classical.

    Energy metric fix: lam eigenvalues are absorbed into V_lam = V * sqrt(lam)
    once at the start of _correct_layer. All subsequent inner products
    V_s @ V_lam.t() are automatically lam-weighted, making Phase 1 and
    Phase 2 use a consistent quadratic energy. The explicit @ lam_dev
    multiplications in Phase 1 are dropped to avoid double-counting.
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
    def _get_calibration_matrix(self, name):
        if name not in self.activation_data or not self.activation_data[name]:
            return None
        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0).float()
        if X.shape[0] > self.max_calib_samples:
            idx = torch.randperm(X.shape[0])[:self.max_calib_samples]
            X   = X[idx]
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Group clustering
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Core per-layer correction
    # ─────────────────────────────────────────────────────────────────────────

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

        # ── Low-rank G = X_corr^T X_corr / n ─────────────────────────────────
        n_tok = X_corr.shape[0]
        k     = min(self.top_k_eigvecs, n_tok, in_features)
        U, S_sv, V = torch.svd_lowrank(X_corr.float(), q=k, niter=4)
        lam = (S_sv ** 2) / n_tok
        del U, S_sv
        torch.cuda.empty_cache()
        if debug:
            print(f"    G: top-{k}, lam_1={lam[0]:.4f}, lam_k={lam[-1]:.6f}")

        # ── Absorb sqrt(lam) into V once ──────────────────────────────────────
        # V_lam[j, r] = V[j, r] * sqrt(lam[r])
        # All inner products  a^T G b  =  (a^T V_lam)(V_lam^T b)
        # This makes every Phase-1 and Phase-2 dE formula automatically
        # lam-weighted without any explicit @ lam_dev term.
        lam_sqrt = lam.sqrt().to(device)                               # [k]
        V_lam    = V.to(device) * lam_sqrt.unsqueeze(0)                # [in, k]
        del V, lam, lam_sqrt
        torch.cuda.empty_cache()

        # ── Ising tensors ─────────────────────────────────────────────────────
        midpoint   = grid_info['midpoint'].float().to(device)
        delta_all  = grid_info['delta'].float().to(device)
        nearest    = grid_info['nearest'].float().to(device)
        half_delta = delta_all / 2
        D_all      = W_sc_f32 - midpoint

        S_nearest = torch.sign(nearest - midpoint)
        S_nearest[S_nearest == 0] = 1.0

        # H: fidelity only (G_D unreliable at low rank, sign alignment ~0.5)
        # H_i = -2*hd_i*λ*D_i  → pulls s toward sign(D) = S_nearest
        H_all = -2.0 * half_delta * self.lambda_fidelity * D_all

        # diag of G in input space: diag_J[j] = ||V_lam[j]||^2 = lam-weighted
        diag_J = (V_lam ** 2).sum(dim=1)                               # [in]

        # ── Transverse field: quantum uncertainty from grid geometry ──────────
        # Γ_i = γ*(1 - |D_i/hd_i|)
        # Γ→1: near midpoint  → uncertain, candidate for correction
        # Γ→0: near grid point → certain, leave at nearest rounding
        Gamma = self.gamma * (
            1.0 - (D_all.abs() / half_delta.clamp(min=1e-10)).clamp(0, 1))

        # Uncertain spins: near midpoint (Γ > threshold)
        uncertain_mask  = Gamma > self.gamma_threshold
        total_uncertain = uncertain_mask.sum().item()

        if debug:
            frac_unc   = total_uncertain / (out_features * in_features)
            unc_ratio  = (D_all.abs() / half_delta.clamp(min=1e-10))[uncertain_mask]
            cert_ratio = (D_all.abs() / half_delta.clamp(min=1e-10))[~uncertain_mask]
            print(f"    Uncertain: {total_uncertain} ({frac_unc*100:.1f}%) "
                  f"|D/hd| unc:{unc_ratio.mean():.3f} cert:{cert_ratio.mean():.3f}")
            del unc_ratio, cert_ratio

        # ── Phase 1: Group exhaustive search on uncertain spins ───────────────
        # Energy: dE = ||v_row + delta_v||^2 - ||v_row||^2 + h^T ds
        #            = delta_v · (2*v_row + delta_v)  +  h^T ds
        # where v_row = V_lam^T (hd * s)   [k-vector, lam already absorbed]
        # and   delta_v = J_g^T delta_sg    (J_g uses V_lam rows)
        # No explicit lam_dev needed — lam is inside V_lam.
        S_refined         = S_nearest.clone()
        total_group_flips = 0

        rows_with_uncertain = (uncertain_mask.sum(dim=1) >= 2).nonzero(
            as_tuple=True)[0]

        # Prioritize rows with most uncertain spins, cap at max_rows
        if rows_with_uncertain.shape[0] > self.max_rows:
            row_counts          = uncertain_mask.sum(dim=1)
            _, top_rows         = row_counts.topk(self.max_rows)
            rows_with_uncertain = top_rows

        for row_idx in rows_with_uncertain.tolist():
            unc_idx = uncertain_mask[row_idx].nonzero(as_tuple=True)[0]
            n_unc   = unc_idx.shape[0]
            if n_unc == 0:
                continue

            # Cap at 50 per row — pick most uncertain (smallest |D/hd|)
            if n_unc > 50:
                ratio    = (D_all[row_idx].abs() /
                            half_delta[row_idx].clamp(min=1e-10))
                _, order = ratio[unc_idx].sort()
                unc_idx  = unc_idx[order[:50]]
                n_unc    = 50

            hd_row  = half_delta[row_idx]                              # [in]
            # J_V_row[j] = hd[j] * V_lam[j]  →  used in v = J_V_row^T s
            J_V_row = hd_row.unsqueeze(1) * V_lam                     # [in, k]
            J_V_unc = J_V_row[unc_idx]                                 # [n_unc, k]
            # J_sub[i,j] = J_V_unc[i] · J_V_unc[j]  (lam already in V_lam)
            J_sub   = J_V_unc @ J_V_unc.t()                           # [n_unc, n_unc]
            groups  = self._cluster_spins(J_sub, unc_idx)

            s_row = S_refined[row_idx].clone()
            h_row = H_all[row_idx]
            # v_row = V_lam^T (hd * s)  [k]
            v_row = J_V_row.t() @ s_row                               # [k]

            for group in groups:
                g = len(group)
                if g == 0:
                    continue
                group_idx = torch.tensor(group, device=device, dtype=torch.long)
                s_g = s_row[group_idx]
                J_g = J_V_row[group_idx]                              # [g, k]
                h_g = h_row[group_idx]

                if g <= self.group_max_size:
                    # Exhaustive: try all 2^g configurations
                    n_configs = 2 ** g
                    bit_idx   = torch.arange(
                        n_configs, device=device).unsqueeze(1)
                    bit_pos   = torch.arange(
                        g,         device=device).unsqueeze(0)
                    flip_mat  = ((bit_idx >> bit_pos) & 1).float()    # [2^g, g]
                    delta_sg  = -2.0 * flip_mat * s_g.unsqueeze(0)   # [2^g, g]
                    delta_v   = delta_sg @ J_g                        # [2^g, k]
                    # dE = delta_v·(2*v + delta_v) + h·ds
                    # lam already absorbed: no @ lam_dev needed
                    dE = (delta_v * (2.0 * v_row.unsqueeze(0) + delta_v)
                          ).sum(dim=1) + delta_sg @ h_g               # [2^g]
                    best_idx = dE.argmin()
                    if dE[best_idx] < -1e-12:
                        best_flip = flip_mat[best_idx].bool()
                        s_row[group_idx[best_flip]] *= -1
                        total_group_flips += best_flip.sum().item()
                        v_row = v_row + delta_v[best_idx]
                else:
                    # Sequential for oversized groups
                    for idx in group:
                        j_v = J_V_row[idx]                            # [k]
                        ds  = -2.0 * s_row[idx]
                        dv  = ds * j_v
                        # dE = dv·(2*v + dv) + h*ds
                        dE  = ((2.0 * v_row + dv) * dv).sum() \
                              + h_row[idx] * ds
                        if dE.item() < -1e-12:
                            s_row[idx] *= -1
                            v_row = v_row + dv
                            total_group_flips += 1

            S_refined[row_idx] = s_row

        del rows_with_uncertain

        # ── Phase 2: CD cleanup — uncertain spins only ────────────────────────
        # dE for flipping spin (i,j):
        #   ds    = -2 * s[i,j]
        #   dv[i] = ds * hd[i,j] * V_lam[j]
        #   dE    = dv[i]·(2*V_s[i] + dv[i])  +  H[i,j]*ds
        #         = 2*ds*hd[i,j]*(V_s[i]·V_lam[j])
        #           + (ds*hd[i,j])^2 * ||V_lam[j]||^2
        #           + H[i,j]*ds
        # where V_s[i] = sum_j' s[i,j']*hd[i,j']*V_lam[j']  (lam-weighted)
        # diag_J[j] = ||V_lam[j]||^2  (precomputed above)
        #
        # Only flip uncertain spins (near-midpoint).
        # Near-gridpoint spins stay at nearest rounding — they are classical.
        S_final        = S_refined.clone()
        total_cd_flips = 0
        chunk_size     = 64

        for sweep in range(self.cd_max_sweeps):
            sweep_flips = 0
            # V_s[i] = sum_j s[i,j]*hd[i,j]*V_lam[j]  shape [out, k]
            V_s = (S_final * half_delta) @ V_lam                      # [out, k]

            for j_start in range(0, in_features, chunk_size):
                j_end   = min(j_start + chunk_size, in_features)
                j_slice = slice(j_start, j_end)

                unc_chunk = uncertain_mask[:, j_slice]
                if not unc_chunk.any():
                    continue

                DS_chunk   = -2.0 * S_final[:, j_slice]               # [out, chunk]
                hd_chunk   = half_delta[:, j_slice]                    # [out, chunk]
                V_chunk    = V_lam[j_start:j_end, :]                  # [chunk, k]  lam-weighted
                # V_s @ V_chunk.t() = sum_r V_s[i,r]*V_lam[j,r]  (lam-weighted dot)
                VsVt_chunk = V_s @ V_chunk.t()                        # [out, chunk]
                diag_chunk = diag_J[j_slice]                          # [chunk]
                H_chunk    = H_all[:, j_slice]                        # [out, chunk]

                # term1 = 2 * ds * hd * (V_s · V_lam[j])
                term1 = 2.0 * DS_chunk * hd_chunk * VsVt_chunk
                # term2 = (ds * hd)^2 * ||V_lam[j]||^2
                term2 = (hd_chunk * DS_chunk) ** 2 * diag_chunk
                dE    = term1 + term2 + H_chunk * DS_chunk
                del term1, term2, VsVt_chunk, DS_chunk

                # Only flip uncertain spins with clear energy improvement
                flip_mask     = (dE < -1e-12) & unc_chunk
                del dE
                n_chunk_flips = flip_mask.sum().item()
                if n_chunk_flips == 0:
                    continue

                flip_float  = flip_mask.float()
                ds_accepted = -2.0 * S_final[:, j_slice] * flip_float
                dV_s        = (ds_accepted * hd_chunk) @ V_chunk      # [out, k]
                S_final[:, j_slice][flip_mask] *= -1
                V_s         = V_s + dV_s
                sweep_flips += n_chunk_flips

            total_cd_flips += sweep_flips
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

        del (V_lam, H_all, Gamma, half_delta, diag_J,
             uncertain_mask, S_nearest, S_refined, S_final,
             midpoint, delta_all, nearest, W_sc_f32, D_all, W_corrected,
             X_corr, Y_orig, Y_base, Y_corrected, baseline_W_q_sc)
        torch.cuda.empty_cache()
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Model-level loop
    # ─────────────────────────────────────────────────────────────────────────

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
                    print(f"  [{global_idx}/{n_layers}] {name}: "
                          f"lm_head — AWQ only")
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
    print(f"AWQ + TFIM Correction  |  "
          f"Model:{args.model_path}  Device:{device}")
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
        dataset     = load_dataset('wikitext', 'wikitext-2-raw-v1',
                                   split='train')
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