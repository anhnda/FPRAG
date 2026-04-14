"""
AWQ Quantum-Inspired Weight Correction
Transverse-Field Ising Model for Rounding Correction

Energy formulation:
    H = Σ_ij J_ij s_i s_j + Σ_i h_i s_i + Σ_i Γ_i σ_x_i

where:
    J_ij   = hd_i * G_ij * hd_j      (reconstruction coupling via G = X^T X / n)
    h_i    = -2*hd_i*(GD)_i - 2*λ*hd_i*D_i  (pull toward nearest rounding)
    Γ_i    = γ * |D_i / hd_i|        (tunneling cost from grid geometry)
    D_i    = W_sc_i - midpoint_i     (displacement from midpoint)

Transverse-Field Ising MF equations (Sachdev):
    E_i   = sqrt(eff_i^2 + Γ_i^2)
    m^z_i = -tanh(β * E_i) * eff_i / E_i   (rounding decision)
    m^x_i =  tanh(β * E_i) * Γ_i  / E_i   (quantum uncertainty)

Uncertain spins: |m^z_i| < |m^x_i|  →  |eff_i| < Γ_i
These are near-midpoint weights where coupling drives the decision.
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

    Each weight's rounding decision (floor/ceil) is a spin s_i ∈ {-1, +1}.
    The transverse field Γ_i = γ * |D_i / hd_i| encodes the cost of flipping
    away from nearest rounding — derived purely from grid geometry.

    Near-gridpoint weights (Γ_i → 0): classical, pinned at nearest rounding.
    Near-midpoint weights (Γ_i → 1): quantum, free to be corrected by coupling.
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 n_grid=20, max_tokens_per_sample=2048,
                 lambda_fidelity=1.0, gamma=1.0,
                 mf_max_iter=50, mf_beta_init=0.1, mf_beta_final=50.0, mf_n_temps=20,
                 group_max_size=6, cd_max_sweeps=2,
                 max_calib_samples=512, top_k_eigvecs=32,
                 layer_batch_size=16):

        self.model               = model
        self.tokenizer           = tokenizer
        self.device              = device
        self.bits                = bits
        self.group_size          = group_size
        self.n_grid              = n_grid
        self.max_tokens_per_sample = max_tokens_per_sample
        self.lambda_fidelity     = lambda_fidelity
        self.gamma               = gamma
        self.mf_max_iter         = mf_max_iter
        self.mf_beta_init        = mf_beta_init
        self.mf_beta_final       = mf_beta_final
        self.mf_n_temps          = mf_n_temps
        self.group_max_size      = group_max_size
        self.cd_max_sweeps       = cd_max_sweeps
        self.max_calib_samples   = max_calib_samples
        self.top_k_eigvecs       = top_k_eigvecs
        self.layer_batch_size    = layer_batch_size
        self.base_quantizer      = AWQBaseQuantizer(bits=bits, group_size=group_size, n_grid=n_grid)
        self.activation_data     = {}
        self.layer_stats         = {}

        print(f"\n{'='*80}")
        print(f"Transverse-Field Ising AWQ Correction Engine")
        print(f"{'='*80}")
        print(f"  Bits:{bits}  GroupSize:{group_size}  lambda_fid:{lambda_fidelity}  gamma:{gamma}")
        print(f"  MF: beta={mf_beta_init}->{mf_beta_final}  temps={mf_n_temps}  iter={mf_max_iter}")
        print(f"  Group max:{group_max_size}  CD sweeps:{cd_max_sweeps}  Low-rank k:{top_k_eigvecs}")
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
                inputs = self.tokenizer(text, return_tensors="pt",
                                        truncation=True, max_length=512)
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
    # Group clustering for uncertain spins
    # ─────────────────────────────────────────────────────────────────────────

    def _cluster_uncertain_spins(self, J_sub, soft_indices):
        """Greedy clustering by coupling strength — CPU numpy."""
        n = J_sub.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [[soft_indices[0].item()]]
        J_np  = J_sub.abs().cpu().numpy()
        si_np = soft_indices.cpu().numpy()
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
        lam = (S_sv ** 2) / n_tok      # eigenvalues of G  [k]
        del U, S_sv
        torch.cuda.empty_cache()
        if debug:
            print(f"    G: top-{k}, lam_1={lam[0]:.4f}, lam_k={lam[-1]:.6f}")

        # ── Ising tensors ─────────────────────────────────────────────────────
        midpoint   = grid_info['midpoint'].float().to(device)
        delta_all  = grid_info['delta'].float().to(device)
        nearest    = grid_info['nearest'].float().to(device)
        half_delta = delta_all / 2                           # hd > 0 always
        D_all      = W_sc_f32 - midpoint                    # displacement from midpoint
        V_dev      = V.to(device)
        lam_dev    = lam.to(device)

        S_nearest  = torch.sign(nearest - midpoint)
        S_nearest[S_nearest == 0] = 1.0

        # Local field: h_i = -2*hd*(GD)_i - 2*λ*hd*D_i
        # Pulls s toward sign(D) = S_nearest
        Vt_D  = D_all @ V_dev                                # [out, k]
        G_D   = (Vt_D * lam_dev) @ V_dev.t()                # [out, in]
        H_all = -2.0 * half_delta * (G_D + self.lambda_fidelity * D_all)
        del G_D, Vt_D

        # ── Transverse field: Γ_i = γ * |D_i / hd_i| ─────────────────────────
        # Key innovation: cost of flipping derived from grid geometry.
        # Γ_i → 0: weight near grid point → classical, hard to flip
        # Γ_i → 1: weight near midpoint   → quantum, free to be corrected
        Gamma = self.gamma * (1.0 - (D_all.abs() / half_delta.clamp(min=1e-10)).clamp(0, 1))

        if debug:
            print(f"    Gamma: mean={Gamma.mean():.3f} "
                  f"frac_uncertain={((Gamma > 0.5).float().mean()):.3f}")

        # ── TFIM Mean-Field Annealing ─────────────────────────────────────────
        # MF equations for transverse-field Ising model (Sachdev):
        #   eff_i = h_i + 2*hd_i*(V lam V^T (m^z * hd))_i
        #   E_i   = sqrt(eff_i^2 + Γ_i^2)
        #   m^z_i = -tanh(β*E_i) * eff_i / E_i
        #   m^x_i =  tanh(β*E_i) * Γ_i  / E_i
        #
        # Near-gridpoint (Γ→0): E≈|eff|, m^z→-sign(eff)  classical
        # Near-midpoint  (Γ→1): E≈1,    m^z≈-eff          soft/quantum

        # Initialize from uncoupled solution: m^z_i = -sign(h_i)
        # (ignores J coupling, but correct direction for local field)
        Mz = -torch.sign(H_all)
        Mz[Mz == 0] = 1.0
        Mz = Mz.float()

        betas = torch.logspace(
            np.log10(self.mf_beta_init),
            np.log10(self.mf_beta_final),
            self.mf_n_temps)

        for beta in betas:
            beta_val = beta.item()
            for iteration in range(self.mf_max_iter):
                # Effective field: h + J*m^z  (low-rank J via V, lam)
                Mz_hd   = Mz * half_delta                              # [out, in]
                Vt_Mzhd = Mz_hd @ V_dev                               # [out, k]
                J_Mz    = 2.0 * (Vt_Mzhd * lam_dev) @ V_dev.t() \
                          * half_delta                                  # [out, in]
                eff     = H_all + J_Mz                                 # [out, in]

                # TFIM MF update
                E       = torch.sqrt(eff ** 2 + Gamma ** 2).clamp(min=1e-10)
                tanhbE  = torch.tanh(beta_val * E)
                Mz_next = -tanhbE * eff / E

                if iteration % 5 == 4:
                    if (Mz_next - Mz).abs().max().item() < 1e-5:
                        Mz = Mz_next
                        break
                Mz = Mz_next

        # Final m^x for uncertainty
        Mz_hd   = Mz * half_delta
        Vt_Mzhd = Mz_hd @ V_dev
        J_Mz    = 2.0 * (Vt_Mzhd * lam_dev) @ V_dev.t() * half_delta
        eff     = H_all + J_Mz
        E       = torch.sqrt(eff ** 2 + Gamma ** 2).clamp(min=1e-10)
        tanhbE  = torch.tanh(self.mf_beta_final * E)
        Mx      = tanhbE * Gamma / E                                   # quantum uncertainty

        del Vt_Mzhd, J_Mz, tanhbE

        # Rounding decision from m^z
        S_mf = torch.sign(Mz)
        S_mf[S_mf == 0] = 1.0
        total_mf_flips = (S_mf != S_nearest).sum().item()

        # Uncertain spins: |m^z| < |m^x|  ↔  |eff| < Γ
        uncertain_mask = eff.abs() < Gamma
        total_uncertain = uncertain_mask.sum().item()

        if debug:
            frac_flipped = total_mf_flips / (out_features * in_features)
            print(f"    MF flip fraction: {frac_flipped:.4f} (expect <0.10)")
            unc_D  = D_all[uncertain_mask].abs() / half_delta[uncertain_mask].clamp(min=1e-10)
            cert_D = D_all[~uncertain_mask].abs() / half_delta[~uncertain_mask].clamp(min=1e-10)
            print(f"    |D/hd| uncertain:{unc_D.mean():.3f}  certain:{cert_D.mean():.3f}")
            print(f"    (uncertain should be smaller — near midpoint)")
            S_test = S_mf.clone()
            W_test = (midpoint + half_delta * S_test).to(original_dtype)
            Y_test = X_corr @ W_test.float().t()
            mf_error = (Y_orig - Y_test).pow(2).mean().item()
            print(f"    MF error: {mf_error:.8f} vs baseline: {baseline_error:.8f}")
            del S_test, W_test, Y_test, unc_D, cert_D
            print(f"    MF flips:{total_mf_flips}  Uncertain:{total_uncertain}")

        del Mz, eff, E, Mx, Mz_hd

        if debug:
            print(f"    MF flips:{total_mf_flips}  Uncertain:{total_uncertain}")

        # ── Group refinement for uncertain spins ──────────────────────────────
        # For weights where |eff| < Γ, the quantum fluctuation dominates.
        # These need joint optimization — exhaustive search over groups.
        S_refined       = S_mf.clone()
        total_group_flips = 0
        rows_with_uncertain = (uncertain_mask.sum(dim=1) >= 2).nonzero(
            as_tuple=True)[0]

        for row_idx in rows_with_uncertain.tolist():
            unc_idx = uncertain_mask[row_idx].nonzero(as_tuple=True)[0]
            n_unc   = unc_idx.shape[0]
            if n_unc == 0:
                continue

            # Cap at 50 most uncertain per row
            if n_unc > 50:
                # Most uncertain = smallest |eff|/Γ ratio
                ratio   = (H_all[row_idx] + 0.0).abs()  # reuse H as proxy
                _, most = ratio[unc_idx].sort()
                unc_idx = unc_idx[most[:50]]
                n_unc   = 50

            hd_row   = half_delta[row_idx]                             # [in]
            J_V_row  = hd_row.unsqueeze(1) * V_dev                    # [in, k]
            J_V_unc  = J_V_row[unc_idx]                               # [n_unc, k]
            J_sub    = J_V_unc @ (lam_dev.unsqueeze(0) * J_V_unc).t() # [n_unc, n_unc]
            groups   = self._cluster_uncertain_spins(J_sub, unc_idx)

            s_row = S_refined[row_idx].clone()
            h_row = H_all[row_idx]
            v_row = J_V_row.t() @ s_row                               # [k]

            for group in groups:
                g = len(group)
                if g == 0:
                    continue
                group_idx = torch.tensor(group, device=device, dtype=torch.long)
                s_g = s_row[group_idx]
                J_g = J_V_row[group_idx]                               # [g, k]
                h_g = h_row[group_idx]

                if g <= self.group_max_size:
                    # Exhaustive search over 2^g configurations
                    n_configs = 2 ** g
                    bit_idx  = torch.arange(n_configs, device=device).unsqueeze(1)
                    bit_pos  = torch.arange(g,         device=device).unsqueeze(0)
                    flip_mat = ((bit_idx >> bit_pos) & 1).float()      # [2^g, g]
                    delta_sg = -2.0 * flip_mat * s_g.unsqueeze(0)     # [2^g, g]
                    delta_v  = delta_sg @ J_g                          # [2^g, k]
                    # dE = lam^T(2v*dv + dv^2) + h^T*ds
                    dE = (delta_v * (2.0 * v_row.unsqueeze(0) + delta_v)) @ lam_dev \
                         + delta_sg @ h_g
                    best_idx = dE.argmin()
                    if dE[best_idx] < -1e-12:
                        best_flip = flip_mat[best_idx].bool()
                        s_row[group_idx[best_flip]] *= -1
                        total_group_flips += best_flip.sum().item()
                        v_row = v_row + delta_v[best_idx]
                else:
                    # Sequential for larger groups
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

        del uncertain_mask, rows_with_uncertain

        # ── Coordinate Descent cleanup (chunked Gauss-Seidel) ─────────────────
        # Clean up any remaining single-spin improvements
        diag_J     = (V_dev ** 2) @ lam_dev                           # [in]
        S_final    = S_refined.clone()
        total_cd_flips = 0
        chunk_size = 64

        for sweep in range(self.cd_max_sweeps):
            sweep_flips = 0
            V_s = (S_final * half_delta) @ V_dev                      # [out, k]

            for j_start in range(0, in_features, chunk_size):
                j_end      = min(j_start + chunk_size, in_features)
                j_slice    = slice(j_start, j_end)

                DS_chunk   = -2.0 * S_final[:, j_slice]
                hd_chunk   = half_delta[:, j_slice]
                V_chunk    = V_dev[j_start:j_end, :]
                VsVt_chunk = V_s @ V_chunk.t()
                diag_chunk = diag_J[j_slice]
                H_chunk    = H_all[:, j_slice]

                term1 = 2.0 * DS_chunk * hd_chunk * VsVt_chunk
                term2 = (hd_chunk * DS_chunk) ** 2 * diag_chunk
                dE    = term1 + term2 + H_chunk * DS_chunk
                del term1, term2, VsVt_chunk, DS_chunk

                flip_mask     = dE < -1e-12
                del dE
                n_chunk_flips = flip_mask.sum().item()
                if n_chunk_flips == 0:
                    continue

                flip_float  = flip_mask.float()
                ds_accepted = -2.0 * S_final[:, j_slice] * flip_float
                dV_s        = (ds_accepted * hd_chunk) @ V_chunk
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
            'mf_flips':        total_mf_flips,
            'uncertain':       total_uncertain,
            'group_flips':     total_group_flips,
            'cd_flips':        total_cd_flips,
        }

        del (V_dev, lam_dev, H_all, Gamma, half_delta, diag_J,
             S_nearest, S_mf, S_refined, S_final,
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
            print(f"\n[Batch {batch_idx+1}/{n_batches}] Layers {b_start}-{b_end-1}")
            self._collect_activations(batch, calibration_data, n_samples)

            for layer_idx, (name, module) in enumerate(batch):
                global_idx = b_start + layer_idx
                is_lmhead  = 'lm_head' in name.lower()
                X_calib    = self._get_calibration_matrix(name)
                if X_calib is None or X_calib.shape[0] < 10:
                    print(f"  [{global_idx}/{n_layers}] {name}: SKIPPED")
                    continue
                debug = (global_idx < 3)

                if is_lmhead:
                    # AWQ only for lm_head
                    print(f"  [{global_idx}/{n_layers}] {name}: lm_head AWQ only")
                    W        = module.weight.data
                    salience = self.base_quantizer.compute_l2_salience(
                        self.activation_data.get(name, []))
                    if salience is not None:
                        X_s    = X_calib[:min(1024, X_calib.shape[0])].to(
                            self.device).to(W.dtype)
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
                    print(f"err {stats['baseline_error']:.6f}->"
                          f"{stats['corrected_error']:.6f} "
                          f"({stats['improvement_pct']:+.2f}%) "
                          f"MF={stats['mf_flips']} "
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
                  f"mean:{imp.mean():+.2f}%  median:{np.median(imp):+.2f}%  "
                  f"min:{imp.min():+.2f}%  max:{imp.max():+.2f}%")
            print(f"  Improved:{(imp > 0).sum()}/{len(imp)}")
        if self.layer_stats:
            mf = sum(s['mf_flips']    for s in self.layer_stats.values())
            gf = sum(s['group_flips'] for s in self.layer_stats.values())
            cd = sum(s['cd_flips']    for s in self.layer_stats.values())
            print(f"  Flips  MF:{mf:,}  G:{gf:,}  CD:{cd:,}")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ + Transverse-Field Ising Correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path",           type=str,   default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir",           type=str,   default="./quantized_models/model_tfim")
    parser.add_argument("--bits",                 type=int,   default=4, choices=[3, 4])
    parser.add_argument("--group-size",           type=int,   default=128)
    parser.add_argument("--n-grid",               type=int,   default=20)
    parser.add_argument("--n-calib",              type=int,   default=128)
    parser.add_argument("--calib-dataset",        type=str,   default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--max-tokens-per-sample",type=int,   default=2048)
    parser.add_argument("--cache-dir",            type=str,   default="./calibration_cache")
    parser.add_argument("--lambda-fidelity",      type=float, default=1.0)
    parser.add_argument("--gamma",                type=float, default=1.0,
                        help="Transverse field scale: Gamma_i = gamma*|D_i/hd_i|")
    parser.add_argument("--mf-beta-init",         type=float, default=0.1)
    parser.add_argument("--mf-beta-final",        type=float, default=50.0)
    parser.add_argument("--mf-n-temps",           type=int,   default=20)
    parser.add_argument("--mf-max-iter",          type=int,   default=50)
    parser.add_argument("--group-max-size",       type=int,   default=6)
    parser.add_argument("--cd-max-sweeps",        type=int,   default=2)
    parser.add_argument("--top-k-eigvecs",        type=int,   default=32)
    parser.add_argument("--max-calib-correction", type=int,   default=512)
    parser.add_argument("--layer-batch-size",     type=int,   default=16)
    parser.add_argument("--seed",                 type=int,   default=42)
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
        lambda_fidelity=args.lambda_fidelity, gamma=args.gamma,
        mf_max_iter=args.mf_max_iter, mf_beta_init=args.mf_beta_init,
        mf_beta_final=args.mf_beta_final, mf_n_temps=args.mf_n_temps,
        group_max_size=args.group_max_size, cd_max_sweeps=args.cd_max_sweeps,
        max_calib_samples=args.max_calib_correction,
        top_k_eigvecs=args.top_k_eigvecs,
        layer_batch_size=args.layer_batch_size,
    )

    corrector.correct_model(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()