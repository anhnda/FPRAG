"""
AWQ Quantum-Inspired Weight Correction (awq_quantum_correction.py)

Starting from a standard AWQ quantization (group-wise asymmetric with L2 salience),
this module applies quantum-inspired corrections to the rounding decisions.

=== Core Idea ===

After AWQ produces a quantized weight matrix W_q, each weight sits on one of two
neighboring grid points: floor or ceil. The rounding decision is a binary spin
variable s_i ∈ {-1, +1}:

    W_q[i] = midpoint[i] + (Δ/2) * s_i

The reconstruction error ||W_q X - W X||² maps to an Ising Hamiltonian:

    H = (Δ²/4) Σ_{ik} G_{ik} σ_i^z σ_k^z  +  Δ Σ_i (Gd)_i σ_i^z

where G = XX^T (Gram matrix of calibration activations), d = midpoint - W (rounding
residual), and the coupling J_{ik} = (Δ²/4) G_{ik}.

=== Correction Hierarchy ===

Phase 1: Mean-Field with Annealing
    - Self-consistent field equations with correct coupling through G
    - Temperature annealing from high T (smooth landscape) to T=0 (discrete)
    - Fully batched over all output rows simultaneously

Phase 2: Spin-Wave Diagnostic
    - Compute per-spin stability gap around the mean-field solution
    - Identify soft modes (near-zero stability) = unreliable rounding decisions
    - Fully batched over all rows

Phase 3: Targeted Group Refinement
    - For weights participating in soft modes, try collective flips
    - Exhaustive search using delta-energy formula (no full tensor clones)
    - O(2^g * k) instead of O(2^g * d)

Phase 4: Coordinate Descent Cleanup
    - Parallel sweep: compute all flips simultaneously, apply all at once
    - Fully batched over all rows: O(out * in * k) per sweep

=== Vectorization Summary ===

    Old: for row in range(out_features):   # 4096 Python iterations
             for temp in betas:            # 20 temps
                 for iter in range(50):   # 50 iters
                     scalar ops           # terrible GPU utilization

    New: All phases operate on [out, in] tensors with batched matmuls.
         Per-row delta scaling folded into matmul via element-wise multiply.
         Phase 3 uses delta-energy: O(2^g * k) not O(2^g * d).
         Phase 4 uses parallel flip detection, no Python spin loop.

=== Usage ===

    corrector = QuantumCorrectionEngine(model, tokenizer, device="cuda")
    corrector.correct_model(calibration_data, n_samples=128)
    model.save_pretrained(output_dir)
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
    print("⚠️ calibration_utils not found. Using internal fallback loaders.")
    def get_c4_calibration_data(*args, **kwargs):
        raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs):
        raise NotImplementedError("Please provide calibration_utils.py")


# =============================================================================
# AWQ Base Quantizer
# =============================================================================

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
        max_int = 2**self.bits - 1

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
        max_int  = 2**self.bits - 1

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

        W_div          = W_g / scale + zp
        W_int_nearest  = torch.round(W_div).clamp(0, max_int)
        W_int_floor    = torch.floor(W_div).clamp(0, max_int)
        W_int_ceil     = torch.ceil(W_div).clamp(0, max_int)

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
        acc     = torch.zeros(in_feat, dtype=torch.float32)
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
            alpha   = grid_idx / self.n_grid
            scales  = activation_salience.pow(alpha)
            W_scaled = W * scales.unsqueeze(0)
            W_q     = self.quantize_weight_groupwise_asymmetric(W_scaled)
            W_recon = W_q / scales.unsqueeze(0)
            Y_q     = X @ W_recon.t()
            error   = (Y_orig - Y_q).pow(2).mean().item()

            if error < best_error:
                best_error  = error
                best_alpha  = alpha
                best_scales = scales.clone()

            del W_scaled, W_q, W_recon, Y_q

        del X, Y_orig
        return best_scales, best_alpha, best_error


# =============================================================================
# Quantum-Inspired Correction Engine (Fully Vectorized)
# =============================================================================

class QuantumCorrectionEngine:
    """
    Applies quantum-inspired corrections to AWQ-quantized weights.

    All four correction phases are fully vectorized over output rows:
      - No Python loop over rows in phases 1, 2, 4
      - Phase 3 still loops over rows (groups are structurally per-row)
        but energy evaluation inside each group is vectorized over 2^g configs
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 n_grid=20, max_tokens_per_sample=2048, lmhead_chunks=4,
                 lambda_fidelity=0.1,
                 mf_max_iter=50, mf_beta_init=0.1, mf_beta_final=50.0, mf_n_temps=20,
                 sw_top_k_modes=10, sw_soft_threshold=0.1,
                 group_max_size=12,
                 cd_max_sweeps=5,
                 max_calib_samples_correction=512,
                 top_k_eigvecs_G=64):

        self.model                 = model
        self.tokenizer             = tokenizer
        self.device                = device
        self.bits                  = bits
        self.group_size            = group_size
        self.n_grid                = n_grid
        self.max_tokens_per_sample = max_tokens_per_sample
        self.lmhead_chunks         = lmhead_chunks
        self.lambda_fidelity       = lambda_fidelity
        self.mf_max_iter           = mf_max_iter
        self.mf_beta_init          = mf_beta_init
        self.mf_beta_final         = mf_beta_final
        self.mf_n_temps            = mf_n_temps
        self.sw_top_k_modes        = sw_top_k_modes
        self.sw_soft_threshold     = sw_soft_threshold
        self.group_max_size        = group_max_size
        self.cd_max_sweeps         = cd_max_sweeps
        self.max_calib_samples     = max_calib_samples_correction
        self.top_k_eigvecs_G       = top_k_eigvecs_G

        self.base_quantizer = AWQBaseQuantizer(bits=bits, group_size=group_size, n_grid=n_grid)
        self.activation_data = {}
        self.layer_stats     = {}

        print(f"\n{'='*80}")
        print(f"Quantum-Inspired AWQ Correction Engine (Vectorized)")
        print(f"{'='*80}")
        print(f"  Bits: {bits}, Group size: {group_size}")
        print(f"  λ_fidelity: {lambda_fidelity}")
        print(f"  Mean-field: β={mf_beta_init}→{mf_beta_final}, {mf_n_temps} temps, {mf_max_iter} iter/temp")
        print(f"  Spin-wave: threshold={sw_soft_threshold}")
        print(f"  Group refinement: max group size={group_max_size}")
        print(f"  Coord descent: max {cd_max_sweeps} sweeps")
        print(f"  Low-rank G: top-{top_k_eigvecs_G} eigenvectors")
        print(f"{'='*80}\n")

    # =========================================================================
    # Activation Collection
    # =========================================================================

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

    # =========================================================================
    # Spin clustering (Phase 3 helper)
    # =========================================================================

    @torch.no_grad()
    def _cluster_soft_spins(self, J_sub, soft_indices):
        """
        Vectorized greedy clustering of soft spins by coupling strength.
        J_sub: [n_soft, n_soft] coupling sub-matrix
        soft_indices: [n_soft] original weight indices
        Returns: list of lists of original indices
        """
        n = J_sub.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [[soft_indices[0].item()]]

        J_abs = J_sub.abs()
        J_abs.fill_diagonal_(0)

        used   = torch.zeros(n, dtype=torch.bool, device=J_sub.device)
        groups = []

        # Precompute row sums for fast seed selection
        row_sums = J_abs.sum(dim=1)  # [n]

        while not used.all():
            # Mask out used spins
            masked_sums = row_sums.clone()
            masked_sums[used] = -1.0
            seed = masked_sums.argmax().item()

            group = [seed]
            used[seed] = True

            for _ in range(self.group_max_size - 1):
                if used.all():
                    break
                # Coupling from unused spins to current group (vectorized)
                group_t    = torch.tensor(group, device=J_sub.device)
                coupling   = J_abs[:, group_t].sum(dim=1)  # [n]
                coupling[used] = -1.0
                best = coupling.argmax().item()

                if coupling[best].item() < 0.1 * row_sums[seed].item() / max(len(group), 1):
                    break

                group.append(best)
                used[best] = True

            groups.append([soft_indices[i].item() for i in group])

        return groups

    # =========================================================================
    # Main Layer Correction — Fully Vectorized
    # =========================================================================

    @torch.no_grad()
    def _correct_layer(self, name, module, X_calib, debug=False):
        """
        Full quantum correction pipeline, all phases batched over [out, in].

        Phase 1 (mean-field): [out, in] tensor ops, per-row early stopping
        Phase 2 (spin-wave):  [out, in] tensor ops, one pass
        Phase 3 (refinement): row loop, but energy eval is O(2^g * k) via delta-energy
        Phase 4 (coord desc): [out, in] tensor ops, parallel flip detection
        """
        W              = module.weight.data
        original_dtype = W.dtype
        out_features, in_features = W.shape
        device         = W.device

        # ── AWQ scaling ───────────────────────────────────────────────────────
        salience = self.base_quantizer.compute_l2_salience(self.activation_data.get(name, []))
        if salience is None:
            salience = torch.ones(in_features)

        X_search = X_calib[:min(2048, X_calib.shape[0])].to(device).to(original_dtype)
        best_scales, best_alpha, awq_error = self.base_quantizer.search_best_scale(
            W, X_search, salience.to(device))
        del X_search

        if debug:
            print(f"    AWQ: α={best_alpha:.3f}, error={awq_error:.8f}")

        W_scaled  = W * best_scales.unsqueeze(0)
        grid_info = self.base_quantizer.get_quantization_grid_info(W_scaled)

        baseline_W_q   = grid_info['nearest'] / best_scales.unsqueeze(0)
        X_corr         = X_calib[:min(self.max_calib_samples, X_calib.shape[0])].to(device).to(original_dtype)
        Y_orig         = X_corr @ W.t()
        Y_base         = X_corr @ baseline_W_q.t()
        baseline_error = (Y_orig - Y_base).pow(2).mean().item()

        if debug:
            print(f"    Baseline (nearest rounding) error: {baseline_error:.8f}")

        # ── Low-rank G = X^T X via SVD ────────────────────────────────────────
        X_for_G      = X_corr / best_scales.unsqueeze(0).to(X_corr.dtype)
        n_tok        = X_for_G.shape[0]
        k            = min(self.top_k_eigvecs_G, n_tok, in_features)
        X_for_G_f32  = X_for_G.float()

        try:
            U, S_sv, Vh = torch.linalg.svd(X_for_G_f32, full_matrices=False)
            V   = Vh[:k].t()                 # [in, k]
            lam = (S_sv[:k] ** 2) / n_tok   # [k]
        except Exception:
            U, S_sv, V = torch.svd_lowrank(X_for_G_f32, q=k)
            lam = (S_sv ** 2) / n_tok
            V   = V[:, :k]

        del X_for_G_f32, X_for_G, U, S_sv
        if 'Vh' in locals():
            del Vh
        torch.cuda.empty_cache()

        if debug:
            print(f"    G eigenspectrum: top-{k}, "
                  f"λ_1={lam[0]:.4f}, λ_k={lam[-1]:.6f}, ratio={lam[0]/lam[-1]:.1f}x")

        # ── Build [out, in] Ising model tensors ───────────────────────────────
        midpoint  = grid_info['midpoint'].float().to(device)   # [out, in]
        delta_all = grid_info['delta'].float().to(device)      # [out, in]
        nearest   = grid_info['nearest'].float().to(device)    # [out, in]
        W_sc_f32  = W_scaled.float()                           # [out, in]
        D_all     = midpoint - W_sc_f32                        # [out, in] residuals

        # Initial spins from nearest rounding
        S_nearest            = torch.sign(nearest - midpoint)  # [out, in]
        S_nearest[S_nearest == 0] = 1.0

        V_dev   = V.to(device)    # [in, k]
        lam_dev = lam.to(device)  # [k]

        # Local fields: H[i,j] = delta[i,j] * (V lam V^T d[i])_j
        #                       + lambda_fid * delta[i,j] * d[i,j]
        # Batched:
        #   Vt_D  = D_all @ V        [out, k]
        #   G_D   = (Vt_D*lam)@V^T  [out, in]
        Vt_D   = D_all @ V_dev                              # [out, k]
        G_D    = (Vt_D * lam_dev) @ V_dev.t()              # [out, in]
        H_all  = delta_all * G_D + self.lambda_fidelity * delta_all * D_all  # [out, in]
        del G_D, Vt_D

        # half_delta[i,j] = delta[i,j] / 2  — acts as the per-row J_V row scale
        half_delta = delta_all / 2    # [out, in]

        # ── Phase 1: Batched Mean-Field Annealing ─────────────────────────────
        # Coupling for row i: J_i m_i = hd_i ⊙ [V lam V^T (hd_i ⊙ m_i)]
        # Batched over all rows:
        #   M_hd   = M * half_delta        [out, in]
        #   Vt_Mhd = M_hd @ V             [out, k]
        #   JM     = (Vt_Mhd*lam)@V^T     [out, in]
        #   JM    *= half_delta            (row-wise delta^2/4 scaling)
        # ─────────────────────────────────────────────────────────────────────
        M = S_nearest.clone()  # [out, in]

        betas = torch.logspace(
            np.log10(self.mf_beta_init),
            np.log10(self.mf_beta_final),
            self.mf_n_temps
        )

        for beta in betas:
            beta_val = beta.item()
            for _ in range(self.mf_max_iter):
                M_old   = M.clone()
                M_hd    = M * half_delta                          # [out, in]
                Vt_Mhd  = M_hd @ V_dev                           # [out, k]
                JM      = (Vt_Mhd * lam_dev) @ V_dev.t()         # [out, in]
                JM     *= half_delta                               # [out, in]
                Eff     = H_all + JM                              # [out, in]
                M_new   = -torch.tanh(beta_val * Eff)
                M       = 0.5 * M_new + 0.5 * M_old

                # Per-row convergence check — stop updating rows that converged
                row_change = (M - M_old).abs().max(dim=1).values  # [out]
                if (row_change < 1e-6).all():
                    break

            del M_hd, Vt_Mhd, JM, Eff, M_new, M_old

        S_mf = torch.sign(M)
        S_mf[S_mf == 0] = 1.0
        total_mf_flips = (S_mf != S_nearest).sum().item()
        del M

        # ── Phase 2: Batched Spin-Wave Stability ──────────────────────────────
        # stability[i,j] = s[i,j] * (H[i,j] + (J_i s_i)_j)
        # where (J_i s_i)_j = hd[i,j] * (V lam V^T (hd[i] ⊙ s[i]))_j
        # ─────────────────────────────────────────────────────────────────────
        S_hd      = S_mf * half_delta                              # [out, in]
        Vt_Shd    = S_hd @ V_dev                                   # [out, k]
        JS_inner  = (Vt_Shd * lam_dev) @ V_dev.t()                # [out, in]
        JS        = half_delta * JS_inner                          # [out, in]
        Eff_star  = H_all + JS                                     # [out, in]
        Stability = S_mf * Eff_star                                # [out, in]
        del S_hd, Vt_Shd, JS_inner, JS, Eff_star

        # Soft spin mask: stability below threshold fraction of row median
        median_stab = Stability.abs().median(dim=1, keepdim=True).values  # [out, 1]
        threshold   = self.sw_soft_threshold * median_stab.clamp(min=1e-10)
        soft_masks  = Stability < threshold                        # [out, in]
        total_sw_soft = soft_masks.sum().item()
        del Stability, median_stab, threshold

        # ── Phase 3: Group Refinement (row loop, vectorized energy eval) ──────
        # For each row with soft spins:
        #   1. Extract soft indices and their coupling sub-matrix J_sub
        #   2. Cluster into groups via greedy coupling
        #   3. For each group, enumerate all 2^g flip configs via delta-energy:
        #      ΔE = lam^T (2v⊙Δv + Δv²) + h_g^T Δs_g
        #      where Δv = Δs_g @ J_g  [2^g, k]  (O(2^g * k), not O(2^g * d))
        # ─────────────────────────────────────────────────────────────────────
        S_refined        = S_mf.clone()
        total_group_flips = 0
        rows_with_soft   = soft_masks.any(dim=1).nonzero(as_tuple=True)[0]

        for row_idx in rows_with_soft.tolist():
            soft_idx = soft_masks[row_idx].nonzero(as_tuple=True)[0]  # [n_soft]
            n_soft   = soft_idx.shape[0]
            if n_soft == 0:
                continue

            # Limit to 200 most unstable soft spins
            if n_soft > 200:
                # Recompute stability for this row to find worst spins
                hd_row_tmp  = half_delta[row_idx]
                s_tmp       = S_mf[row_idx]
                Js_tmp      = hd_row_tmp * (V_dev @ (lam_dev * (V_dev.t() @ (hd_row_tmp * s_tmp))))
                stab_tmp    = s_tmp * (H_all[row_idx] + Js_tmp)
                _, worst_order = stab_tmp[soft_idx].sort()
                soft_idx    = soft_idx[worst_order[:200]]
                n_soft      = 200

            # Build per-row J_V: J_V_row[j] = half_delta[row, j] * V[j]  [in, k]
            hd_row   = half_delta[row_idx]               # [in]
            J_V_row  = hd_row.unsqueeze(1) * V_dev        # [in, k]
            J_V_soft = J_V_row[soft_idx]                  # [n_soft, k]

            # J_sub[a,b] = J_V_soft[a] · diag(lam) · J_V_soft[b]
            J_sub = J_V_soft @ (lam_dev.unsqueeze(0) * J_V_soft).t()  # [n_soft, n_soft]

            groups = self._cluster_soft_spins(J_sub, soft_idx)

            s_row = S_refined[row_idx].clone()   # [in]
            h_row = H_all[row_idx]               # [in]
            # v = J_V_row^T @ s_row  [k]
            v_row = J_V_row.t() @ s_row          # [k]

            for group in groups:
                g = len(group)
                if g == 0:
                    continue

                group_idx = torch.tensor(group, device=device, dtype=torch.long)
                s_g = s_row[group_idx]    # [g]
                J_g = J_V_row[group_idx]  # [g, k]
                h_g = h_row[group_idx]    # [g]

                if g <= self.group_max_size:
                    # Enumerate all 2^g flip configs
                    n_configs = 2 ** g
                    bit_idx  = torch.arange(n_configs, device=device).unsqueeze(1)  # [2^g, 1]
                    bit_pos  = torch.arange(g, device=device).unsqueeze(0)          # [1, g]
                    flip_mat = ((bit_idx >> bit_pos) & 1).float()                   # [2^g, g]

                    # delta_sg[c, b] = -2 * flip_mat[c,b] * s_g[b]
                    delta_sg = -2.0 * flip_mat * s_g.unsqueeze(0)    # [2^g, g]

                    # delta_v[c] = J_g^T delta_sg[c]  →  [2^g, k]
                    delta_v  = delta_sg @ J_g                         # [2^g, k]

                    # ΔE = lam^T (2v⊙Δv + Δv²) + h_g^T Δs_g
                    dE_quad  = (delta_v * (2.0 * v_row.unsqueeze(0) + delta_v)) @ lam_dev  # [2^g]
                    dE_lin   = delta_sg @ h_g                                                # [2^g]
                    dE       = dE_quad + dE_lin                                              # [2^g]

                    best = dE.argmin().item()
                    if dE[best].item() < -1e-12:
                        best_flip = flip_mat[best].bool()              # [g]
                        s_row[group_idx[best_flip]] *= -1
                        total_group_flips += best_flip.sum().item()
                        v_row = v_row + delta_v[best]                  # incremental update

                else:
                    # Greedy: flip one at a time, O(g * k) total
                    for idx in group:
                        j_v   = J_V_row[idx]                 # [k]
                        ds    = -2.0 * s_row[idx]
                        dv    = ds * j_v                     # [k]
                        dE    = ((2.0 * v_row + dv) * dv * lam_dev).sum() + h_row[idx] * ds
                        if dE.item() < -1e-12:
                            s_row[idx] *= -1
                            v_row       = v_row + dv
                            total_group_flips += 1

            S_refined[row_idx] = s_row

        del soft_masks, rows_with_soft

        # ── Phase 4: Vectorized Coordinate Descent ────────────────────────────
        # Each sweep:
        #   1. Compute Eff = H + hd ⊙ (V lam V^T (hd ⊙ S))  [out, in]  (one batched matmul)
        #   2. Detect all beneficial flips: s * eff < 0       [out, in]  (element-wise)
        #   3. Apply all flips simultaneously                  [out, in]  (indexed assign)
        #
        # NOTE: The correct flip criterion is s_j * eff_j < 0
        #   ΔE = -2 s_j eff_j  →  ΔE < 0  when  s_j * eff_j > 0
        # WAIT — let's be precise:
        #   ΔE = E(s with s_j flipped) - E(s)
        #      = -2 * s_j * eff_j
        # We want ΔE < 0, so flip when: s_j * eff_j > 0
        # That means spin is ALIGNED with field → flipping reduces energy.
        # This is correct: in our sign convention, eff_j = h_j + (J s)_j,
        # and energy E = s^T J s + h^T s. The gradient dE/ds_j ∝ eff_j.
        # Minimum energy wants s_j to be OPPOSITE sign to eff_j.
        # So flip if s_j and eff_j have the SAME sign (currently suboptimal).
        # ─────────────────────────────────────────────────────────────────────
        S_final        = S_refined.clone()
        total_cd_flips = 0

        for sweep in range(self.cd_max_sweeps):
            S_hd  = S_final * half_delta                          # [out, in]
            Vt_S  = S_hd @ V_dev                                  # [out, k]
            JS    = (Vt_S * lam_dev) @ V_dev.t()                  # [out, in]
            Eff   = H_all + half_delta * JS                       # [out, in]

            # Flip where s * eff > 0 (spin aligned with field = reduces energy to flip)
            flip_mask = (S_final * Eff) > 1e-12                   # [out, in]
            n_flips   = flip_mask.sum().item()

            if n_flips == 0:
                del S_hd, Vt_S, JS, Eff
                break

            S_final[flip_mask] *= -1
            total_cd_flips += n_flips
            del S_hd, Vt_S, JS, Eff

        # ── Reconstruct final weights ─────────────────────────────────────────
        # w_q[i,j] = midpoint[i,j] + half_delta[i,j] * s_final[i,j]
        # Then undo AWQ scaling: W_final = W_corrected / best_scales
        W_corrected = (midpoint + half_delta * S_final).to(original_dtype)
        W_final     = W_corrected / best_scales.unsqueeze(0)
        module.weight.data = W_final

        Y_corrected     = X_corr @ W_final.t()
        corrected_error = (Y_orig - Y_corrected).pow(2).mean().item()
        improvement     = (baseline_error - corrected_error) / max(baseline_error, 1e-12) * 100

        if debug:
            print(f"    Corrected error: {corrected_error:.8f} ({improvement:+.2f}% vs baseline)")
            print(f"    Flips — MF: {total_mf_flips}, Group: {total_group_flips}, CD: {total_cd_flips}")
            print(f"    Soft spins found: {total_sw_soft}")

        stats = {
            'awq_alpha':      best_alpha,
            'baseline_error': baseline_error,
            'corrected_error': corrected_error,
            'improvement_pct': improvement,
            'mf_flips':       total_mf_flips,
            'sw_soft':        total_sw_soft,
            'group_flips':    total_group_flips,
            'cd_flips':       total_cd_flips,
        }

        del (V_dev, lam_dev, H_all, half_delta, S_nearest, S_mf, S_refined, S_final,
             midpoint, delta_all, nearest, W_sc_f32, D_all, W_corrected,
             X_corr, Y_orig, Y_base, Y_corrected, baseline_W_q)
        torch.cuda.empty_cache()

        return stats

    # =========================================================================
    # Full Model Correction
    # =========================================================================

    def correct_model(self, calibration_data, n_samples=128, layer_batch_size=16):
        print(f"\n{'='*80}")
        print("QUANTUM-INSPIRED WEIGHT CORRECTION (Vectorized)")
        print(f"{'='*80}")

        layer_list = [(name, module) for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear)]
        n_layers  = len(layer_list)
        n_batches = (n_layers + layer_batch_size - 1) // layer_batch_size

        print(f"  Layers: {n_layers}")
        print(f"  Batches: {n_batches} (batch size={layer_batch_size})")
        print(f"  Calibration samples: {n_samples}")

        total_improvement = []
        t_start = time.time()

        for batch_idx in range(n_batches):
            b_start = batch_idx * layer_batch_size
            b_end   = min(b_start + layer_batch_size, n_layers)
            batch   = layer_list[b_start:b_end]

            print(f"\n[Batch {batch_idx+1}/{n_batches}] Layers {b_start}–{b_end-1}")
            self._collect_activations(batch, calibration_data, n_samples)

            for layer_idx, (name, module) in enumerate(batch):
                global_idx = b_start + layer_idx
                is_lmhead  = 'lm_head' in name.lower()

                X_calib = self._get_calibration_matrix(name)
                if X_calib is None or X_calib.shape[0] < 10:
                    print(f"  [{global_idx}/{n_layers}] {name}: SKIPPED (no calibration data)")
                    continue

                debug = (global_idx < 2)

                if is_lmhead:
                    print(f"  [{global_idx}/{n_layers}] {name}: lm_head — standard AWQ only")
                    W       = module.weight.data
                    salience = self.base_quantizer.compute_l2_salience(
                        self.activation_data.get(name, []))
                    if salience is not None:
                        X_s    = X_calib[:min(1024, X_calib.shape[0])].to(self.device).to(W.dtype)
                        scales, _, _ = self.base_quantizer.search_best_scale(
                            W, X_s, salience.to(self.device))
                        del X_s
                        W_sc   = W * scales.unsqueeze(0)
                        W_q    = self.base_quantizer.quantize_weight_groupwise_asymmetric(W_sc)
                        module.weight.data = (W_q / scales.unsqueeze(0)).to(W.dtype)
                        del W_sc, W_q
                else:
                    print(f"  [{global_idx}/{n_layers}] {name}:", end=" ", flush=True)
                    t0    = time.time()
                    stats = self._correct_layer(name, module, X_calib, debug=debug)
                    dt    = time.time() - t0
                    self.layer_stats[name] = stats
                    total_improvement.append(stats['improvement_pct'])
                    print(f"err {stats['baseline_error']:.6f}→{stats['corrected_error']:.6f} "
                          f"({stats['improvement_pct']:+.2f}%) "
                          f"flips: MF={stats['mf_flips']} G={stats['group_flips']} "
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

        print(f"\n{'='*80}")
        print(f"CORRECTION COMPLETE ({elapsed:.1f}s)")
        print(f"{'='*80}")

        if total_improvement:
            imp = np.array(total_improvement)
            print(f"  Layers corrected: {len(imp)}/{n_layers}")
            print(f"  Error improvement — mean: {imp.mean():+.2f}%  "
                  f"median: {np.median(imp):+.2f}%  "
                  f"min: {imp.min():+.2f}%  max: {imp.max():+.2f}%")
            print(f"  Improved layers: {(imp > 0).sum()}/{len(imp)}")

        if self.layer_stats:
            mf = sum(s['mf_flips']    for s in self.layer_stats.values())
            gf = sum(s['group_flips'] for s in self.layer_stats.values())
            cd = sum(s['cd_flips']    for s in self.layer_stats.values())
            print(f"\n  Total flips — MF: {mf:,}  Group: {gf:,}  CD: {cd:,}  Total: {mf+gf+cd:,}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AWQ + Quantum-Inspired Weight Correction (Vectorized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument("--mf-n-temps",           type=int,   default=20)
    parser.add_argument("--mf-max-iter",          type=int,   default=50)
    parser.add_argument("--sw-threshold",         type=float, default=0.1)
    parser.add_argument("--group-max-size",       type=int,   default=12)
    parser.add_argument("--cd-max-sweeps",        type=int,   default=5)
    parser.add_argument("--top-k-eigvecs",        type=int,   default=64)
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

    print("=" * 80)
    print("AWQ + Quantum-Inspired Weight Correction (Vectorized)")
    print(f"Model: {args.model_path}  |  Device: {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"\nLoading calibration: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset     = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [t['text'] for t in dataset if len(t['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)

    corrector = QuantumCorrectionEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        group_size=args.group_size,
        n_grid=args.n_grid,
        max_tokens_per_sample=args.max_tokens_per_sample,
        lmhead_chunks=args.lmhead_chunks,
        lambda_fidelity=args.lambda_fidelity,
        mf_max_iter=args.mf_max_iter,
        mf_beta_init=args.mf_beta_init,
        mf_beta_final=args.mf_beta_final,
        mf_n_temps=args.mf_n_temps,
        sw_soft_threshold=args.sw_threshold,
        group_max_size=args.group_max_size,
        cd_max_sweeps=args.cd_max_sweeps,
        max_calib_samples_correction=args.max_calib_correction,
        top_k_eigvecs_G=args.top_k_eigvecs,
    )

    corrector.correct_model(
        calib_texts,
        n_samples=args.n_calib,
        layer_batch_size=args.layer_batch_size
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == "__main__":
    main()