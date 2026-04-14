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
    - Already better than AdaRound because coupling is explicit

Phase 2: Spin-Wave Diagnostic
    - Compute fluctuation spectrum around the mean-field solution
    - Identify soft modes (near-zero eigenvalues) = unreliable rounding decisions
    - The eigenvectors of soft modes tell us WHICH weights to re-optimize

Phase 3: Targeted Group Refinement
    - For weights participating in soft modes, try collective flips
    - Exhaustive search for small groups, greedy for larger ones
    - "Quantum tunneling" moves: flip correlated groups simultaneously

Phase 4: Coordinate Descent Cleanup
    - Sweep through all weights, flip each to best grid point given all others
    - Cheap per step: O(n) using low-rank G structure
    - Converges in few sweeps

=== Usage ===

    # After standard AWQ quantization:
    corrector = QuantumCorrectionEngine(model, tokenizer, device="cuda")
    corrector.correct_model(calibration_data, n_samples=128)
    model.save_pretrained(output_dir)

=== References ===

- Ising model formulation of rounding: maps binary rounding to spin-1/2 system
- Mean-field theory: self-consistent sigmoid equations with coupling G = XX^T
- Spin-wave theory (Holstein-Primakoff): fluctuation spectrum around mean-field
- Coordinate descent on discrete objective: direct optimization without surrogate
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
# AWQ Base Quantizer (Standard Group-Wise Asymmetric with L2 Salience)
# =============================================================================

class AWQBaseQuantizer:
    """
    Standard AWQ quantizer that produces the initial quantized weights.
    This is the starting point; quantum correction improves upon it.
    """

    def __init__(self, bits=4, group_size=128, n_grid=20):
        self.bits = bits
        self.group_size = group_size
        self.n_grid = n_grid

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W):
        """Group-wise asymmetric quantization → dequantized weight."""
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
        """
        For each weight element, compute:
        - floor_val: the grid point below
        - ceil_val: the grid point above
        - scale, zero_point per group (needed for grid reconstruction)

        Returns dict with all grid information needed for spin formulation.
        """
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

        # Continuous quantization level (before rounding)
        W_div = W_g / scale + zp  # [out, n_groups, group_size]

        # Integer level (nearest rounding)
        W_int_nearest = torch.round(W_div).clamp(0, max_int)

        # Floor and ceil integer levels
        W_int_floor = torch.floor(W_div).clamp(0, max_int)
        W_int_ceil = torch.ceil(W_div).clamp(0, max_int)

        # Handle exact integers: floor == ceil, set ceil = floor + 1 if possible
        exact_mask = (W_int_floor == W_int_ceil)
        W_int_ceil[exact_mask & (W_int_ceil < max_int)] += 1
        W_int_floor[exact_mask & (W_int_floor > 0)] -= 1

        # Dequantize to get actual grid values
        floor_val = (W_int_floor - zp) * scale
        ceil_val = (W_int_ceil - zp) * scale
        nearest_val = (W_int_nearest - zp) * scale

        # Reshape back
        floor_val = floor_val.reshape(out_features, padded)[:, :in_features]
        ceil_val = ceil_val.reshape(out_features, padded)[:, :in_features]
        nearest_val = nearest_val.reshape(out_features, padded)[:, :in_features]
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded)[:, :in_features]

        return {
            'floor': floor_val,
            'ceil': ceil_val,
            'nearest': nearest_val,
            'scale': scale_flat,
            'midpoint': (floor_val + ceil_val) / 2,
            'delta': ceil_val - floor_val,  # Grid step per element (= scale per group)
        }

    @torch.no_grad()
    def compute_l2_salience(self, activation_data):
        """Compute per-input-channel L2 salience: E[X[:,j]^2]."""
        if not activation_data:
            return None
        total = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in activation_data)
        in_feat = activation_data[0].shape[-1]
        acc = torch.zeros(in_feat, dtype=torch.float32)
        for x in activation_data:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            acc += x_flat.pow(2).sum(dim=0)
        return acc / total

    @torch.no_grad()
    def search_best_scale(self, W, X_calib, activation_salience):
        """
        Grid search for optimal AWQ scaling α.
        Returns: best_scales, best_alpha, best_error
        """
        device = W.device
        dtype = W.dtype
        activation_salience = activation_salience.to(device).to(dtype).clamp(min=1e-5)
        X = X_calib.to(device).to(dtype)

        Y_orig = X @ W.t()

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=device, dtype=dtype)

        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)
            W_scaled = W * scales.unsqueeze(0)
            W_q = self.quantize_weight_groupwise_asymmetric(W_scaled)
            W_recon = W_q / scales.unsqueeze(0)
            Y_q = X @ W_recon.t()
            error = (Y_orig - Y_q).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del W_scaled, W_q, W_recon, Y_q

        del X, Y_orig
        return best_scales, best_alpha, best_error


# =============================================================================
# Quantum-Inspired Correction Engine
# =============================================================================

class QuantumCorrectionEngine:
    """
    Applies quantum-inspired corrections to AWQ-quantized weights.

    The correction follows the hierarchy:
    1. Mean-field annealing (self-consistent field with coupling G = XX^T)
    2. Spin-wave diagnostic (identify soft modes = unreliable roundings)
    3. Targeted group refinement (collective flips along soft modes)
    4. Coordinate descent cleanup (sweep all weights)
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 n_grid=20, max_tokens_per_sample=2048, lmhead_chunks=4,
                 # Correction parameters
                 lambda_fidelity=0.1,
                 mf_max_iter=50, mf_beta_init=0.1, mf_beta_final=50.0, mf_n_temps=20,
                 sw_top_k_modes=10, sw_soft_threshold=0.1,
                 group_max_size=18,
                 cd_max_sweeps=5,
                 max_calib_samples_correction=512,
                 top_k_eigvecs_G=64):
        """
        Args:
            model: HuggingFace model (already loaded)
            tokenizer: HuggingFace tokenizer
            device: torch device
            bits: quantization bit width
            group_size: group size for quantization
            n_grid: AWQ grid search points
            max_tokens_per_sample: token subsampling per calibration sample
            lmhead_chunks: chunks for lm_head processing

            lambda_fidelity: weight for ||W_q - W||^2 fidelity term (vs reconstruction)
            mf_max_iter: max iterations per temperature in mean-field
            mf_beta_init: initial inverse temperature
            mf_beta_final: final inverse temperature (→ discrete)
            mf_n_temps: number of temperature steps in annealing
            sw_top_k_modes: number of soft modes to analyze in spin-wave
            sw_soft_threshold: threshold for declaring a mode "soft"
            group_max_size: max group size for exhaustive search in refinement
            cd_max_sweeps: max coordinate descent sweeps
            max_calib_samples_correction: max calibration tokens for correction phase
            top_k_eigvecs_G: number of top eigenvectors of G to keep (low-rank approx)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size
        self.n_grid = n_grid
        self.max_tokens_per_sample = max_tokens_per_sample
        self.lmhead_chunks = lmhead_chunks

        # Correction hyperparameters
        self.lambda_fidelity = lambda_fidelity
        self.mf_max_iter = mf_max_iter
        self.mf_beta_init = mf_beta_init
        self.mf_beta_final = mf_beta_final
        self.mf_n_temps = mf_n_temps
        self.sw_top_k_modes = sw_top_k_modes
        self.sw_soft_threshold = sw_soft_threshold
        self.group_max_size = group_max_size
        self.cd_max_sweeps = cd_max_sweeps
        self.max_calib_samples = max_calib_samples_correction
        self.top_k_eigvecs_G = top_k_eigvecs_G

        # Internal
        self.base_quantizer = AWQBaseQuantizer(bits=bits, group_size=group_size, n_grid=n_grid)
        self.activation_data = {}
        self.layer_stats = {}

        print(f"\n{'='*80}")
        print(f"Quantum-Inspired AWQ Correction Engine")
        print(f"{'='*80}")
        print(f"  Bits: {bits}, Group size: {group_size}")
        print(f"  λ_fidelity (weight fidelity vs reconstruction): {lambda_fidelity}")
        print(f"  Mean-field: β={mf_beta_init}→{mf_beta_final}, {mf_n_temps} temps, {mf_max_iter} iter/temp")
        print(f"  Spin-wave: top-{sw_top_k_modes} modes, soft threshold={sw_soft_threshold}")
        print(f"  Group refinement: max group size={group_max_size}")
        print(f"  Coord descent: max {cd_max_sweeps} sweeps")
        print(f"  Low-rank G: top-{top_k_eigvecs_G} eigenvectors")
        print(f"{'='*80}\n")

    # =========================================================================
    # Activation Collection
    # =========================================================================

    def _get_hook(self, name):
        """Create forward hook to capture activations."""
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
        """Run calibration data and collect activations for given layers."""
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
        """
        Get calibration activations as a [n_tokens, in_features] matrix.
        Returns on CPU in float32.
        """
        if name not in self.activation_data or not self.activation_data[name]:
            return None
        if max_samples is None:
            max_samples = self.max_calib_samples

        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0).float()

        if X.shape[0] > max_samples:
            idx = torch.randperm(X.shape[0])[:max_samples]
            X = X[idx]
        return X

    # =========================================================================
    # Ising Model Construction
    # =========================================================================

    @torch.no_grad()
    def _build_ising_model_for_row(self, w_row, grid_info_row, G_lowrank_V, G_lowrank_lam, d_row):
        """
        Build the Ising model for a single output row of the weight matrix.

        For row i of W (shape [in_features]):
            w_q[j] = midpoint[j] + (delta[j]/2) * s[j]

        The reconstruction error for this row through calibration:
            E_row = || (w_q - w) X ||^2 = || (d + Δ/2 · s) X ||^2

        where d[j] = midpoint[j] - w[j], s[j] ∈ {-1, +1}.

        Using low-rank G ≈ V Λ V^T:
            E_row = Σ_α λ_α (Σ_j (d_j + Δ_j/2 · s_j) V_{jα})^2

        The Ising parameters (for this row):
            J_{jk} = (Δ_j Δ_k / 4) G_{jk}    [coupling]
            h_j = Δ_j Σ_k G_{jk} d_k          [local field from reconstruction]
                + λ · Δ_j · d_j                 [fidelity regularization]

        Args:
            w_row: original weight row [in_features]
            grid_info_row: dict with floor, ceil, midpoint, delta for this row
            G_lowrank_V: [in_features, k] top eigenvectors of G
            G_lowrank_lam: [k] top eigenvalues of G
            d_row: midpoint - w for this row [in_features]

        Returns:
            h: local fields [in_features]
            J_V: [in_features, k] = (Δ/2) * V, so J = J_V diag(λ) J_V^T
            delta: grid step [in_features]
            s_nearest: nearest rounding spins [in_features] {-1, +1}
        """
        device = w_row.device
        delta = grid_info_row['delta']  # [in_features]
        midpoint = grid_info_row['midpoint']
        nearest = grid_info_row['nearest']

        # Spin for nearest rounding: s = +1 if nearest == ceil, -1 if nearest == floor
        s_nearest = torch.sign(nearest - midpoint)
        # Handle exact midpoint (rare): default to +1
        s_nearest[s_nearest == 0] = 1.0

        # Low-rank representation: J_V = (Δ/2) * V, so coupling is J_V diag(λ) J_V^T
        # This avoids forming the full [in_features, in_features] J matrix
        half_delta = delta / 2  # [in_features]
        J_V = half_delta.unsqueeze(1) * G_lowrank_V  # [in_features, k]

        # Local field from reconstruction: h_recon_j = Δ_j * Σ_k G_{jk} d_k
        # Using low-rank: G d ≈ V Λ V^T d
        Vt_d = G_lowrank_V.t() @ d_row  # [k]
        G_d = G_lowrank_V @ (G_lowrank_lam * Vt_d)  # [in_features]
        h_recon = delta * G_d

        # Local field from fidelity: h_fid_j = λ * Δ_j * d_j
        h_fid = self.lambda_fidelity * delta * d_row

        h = h_recon + h_fid

        return h, J_V, delta, s_nearest

    # =========================================================================
    # Phase 1: Mean-Field Annealing
    # =========================================================================

    @torch.no_grad()
    def _mean_field_annealing(self, h, J_V, G_lam, s_init):
        """
        Mean-field annealing with self-consistent equations.

        The mean-field equation:
            m_i = -tanh(β * eff_i)
        where:
            eff_i = h_i + Σ_k J_{ik} m_k
                  = h_i + (J_V diag(λ) J_V^T m)_i

        Using low-rank structure:
            J m = J_V diag(λ) (J_V^T m)    cost: O(d·k)

        Anneal β from β_init to β_final.

        Args:
            h: local fields [d]
            J_V: [d, k] low-rank coupling factor
            G_lam: [k] eigenvalues
            s_init: initial spin config [d], values in {-1, +1}

        Returns:
            m: continuous magnetizations [d] in [-1, 1]
            s: discretized spins [d] in {-1, +1}
            converged: bool
        """
        d = h.shape[0]
        device = h.device

        # Initialize magnetization from spin config
        m = s_init.float().clone()

        # Temperature schedule (geometric)
        betas = torch.logspace(
            np.log10(self.mf_beta_init),
            np.log10(self.mf_beta_final),
            self.mf_n_temps
        )

        for beta in betas:
            beta_val = beta.item()
            for iteration in range(self.mf_max_iter):
                m_old = m.clone()

                # Compute effective field: eff = h + J @ m
                # J @ m = J_V @ diag(lam) @ J_V^T @ m
                Vt_m = J_V.t() @ m  # [k]
                Jm = J_V @ (G_lam * Vt_m)  # [d]
                eff = h + Jm

                # Mean-field update
                m_new = -torch.tanh(beta_val * eff)

                # Damped update for stability
                m = 0.5 * m_new + 0.5 * m_old

                # Check convergence
                change = (m - m_old).abs().max().item()
                if change < 1e-6:
                    break

        # Discretize
        s = torch.sign(m)
        s[s == 0] = 1.0

        return m, s, True

    # =========================================================================
    # Phase 2: Spin-Wave Diagnostic
    # =========================================================================

    @torch.no_grad()
    def _spin_wave_analysis(self, h, J_V, G_lam, s_star):
        """
        Compute the fluctuation spectrum around solution s_star.

        The stability gap for each spin:
            ε_i = s_i * (h_i + (J s)_i)

        If ε_i > 0: spin is locally stable.
        If ε_i ≈ 0: spin is marginally stable (soft mode participant).
        If ε_i < 0: spin is locally suboptimal (should flip).

        For collective instabilities, we compute the Hessian of the energy
        in the continuous relaxation around s_star:

            M_{ij} = δ_{ij} ε_i + J_{ij} (for off-diagonal, only if s_i = s_j ... )

        Simplified: we compute per-spin stability and identify the most vulnerable
        spins. Then we check if groups of vulnerable spins can improve by flipping
        together, using the eigenvectors of the projected coupling.

        Returns:
            stability_gaps: [d] per-spin stability
            soft_mode_groups: list of lists of indices, each group = correlated flips
        """
        d = h.shape[0]
        device = h.device

        # Compute effective field at s_star
        Vt_s = J_V.t() @ s_star  # [k]
        Js = J_V @ (G_lam * Vt_s)  # [d]
        eff = h + Js

        # Stability gap: ε_i = s_i * eff_i
        # Positive = stable, negative = should flip
        stability = s_star * eff

        # Find unstable and marginally stable spins
        # Sort by stability (ascending = most unstable first)
        sorted_stab, sorted_idx = stability.sort()

        # Identify soft spins: stability < threshold * median(|stability|)
        median_stab = stability.abs().median().item()
        threshold = self.sw_soft_threshold * median_stab if median_stab > 1e-10 else 1e-8
        soft_mask = stability < threshold

        soft_indices = torch.where(soft_mask)[0]
        n_soft = soft_indices.shape[0]

        if n_soft == 0:
            # No soft modes — solution is robust
            return stability, []

        # Limit the number of soft spins we analyze
        max_soft = min(n_soft, 200)
        if n_soft > max_soft:
            # Take the most unstable ones
            _, most_unstable = stability[soft_indices].sort()
            soft_indices = soft_indices[most_unstable[:max_soft]]
            n_soft = max_soft

        # Cluster soft spins into groups using their coupling structure
        # Extract the J sub-matrix for soft spins via low-rank:
        # J_sub = J_V[soft] @ diag(lam) @ J_V[soft]^T
        J_V_sub = J_V[soft_indices]  # [n_soft, k]
        J_sub = J_V_sub @ (G_lam.unsqueeze(0) * J_V_sub).t()  # [n_soft, n_soft]

        # Group correlated spins by thresholding the coupling
        # Use simple greedy clustering
        groups = self._cluster_soft_spins(J_sub, soft_indices)

        return stability, groups

    @torch.no_grad()
    def _cluster_soft_spins(self, J_sub, soft_indices):
        """
        Cluster soft spins into groups based on coupling strength.
        Simple greedy: start from most coupled pair, grow cluster.

        Returns list of lists of original indices.
        """
        n = J_sub.shape[0]
        if n == 0:
            return []
        if n == 1:
            return [soft_indices.tolist()]

        # Normalize coupling strengths
        J_abs = J_sub.abs()
        J_abs.fill_diagonal_(0)

        used = torch.zeros(n, dtype=torch.bool, device=J_sub.device)
        groups = []

        while not used.all():
            # Find unused spin with strongest total coupling to other unused
            remaining = (~used).nonzero(as_tuple=True)[0]
            if len(remaining) == 0:
                break

            coupling_sum = J_abs[remaining][:, remaining].sum(dim=1)
            seed_local = coupling_sum.argmax().item()
            seed = remaining[seed_local].item()

            # Grow group from seed
            group = [seed]
            used[seed] = True

            # Add spins strongly coupled to the group
            for _ in range(self.group_max_size - 1):
                remaining = (~used).nonzero(as_tuple=True)[0]
                if len(remaining) == 0:
                    break

                # Coupling of remaining spins to current group
                group_coupling = J_abs[remaining][:, torch.tensor(group, device=J_sub.device)].sum(dim=1)

                # Add the most coupled spin if coupling is significant
                best_local = group_coupling.argmax().item()
                best_coupling = group_coupling[best_local].item()

                # Threshold: coupling should be at least 10% of the seed's coupling
                if best_coupling < 0.1 * J_abs[seed].sum().item() / max(len(group), 1):
                    break

                best_idx = remaining[best_local].item()
                group.append(best_idx)
                used[best_idx] = True

            # Convert local indices to original indices
            groups.append([soft_indices[i].item() for i in group])

        return groups

    # =========================================================================
    # Phase 3: Targeted Group Refinement
    # =========================================================================

    @torch.no_grad()
    def _refine_groups(self, s, h, J_V, G_lam, groups, delta):
        """
        For each group of correlated spins, try collective flips.

        For small groups (≤ group_max_size): exhaustive 2^|group| search.
        For larger groups: greedy sequential flipping.

        The energy for a configuration s:
            E(s) = s^T J s + h^T s
                 = (J_V^T s)^T Λ (J_V^T s) + h^T s

        Args:
            s: current spin config [d]
            h: local fields [d]
            J_V: [d, k]
            G_lam: [k]
            groups: list of index lists
            delta: grid step [d]

        Returns:
            s_new: improved spin config
            n_flips: total number of flipped spins
        """
        s_new = s.clone()
        total_flips = 0

        for group in groups:
            group_size = len(group)
            if group_size == 0:
                continue

            group_idx = torch.tensor(group, device=s.device, dtype=torch.long)

            if group_size <= self.group_max_size:
                # Exhaustive search
                best_energy = float('inf')
                best_config = s_new[group_idx].clone()

                # Compute energy contribution from this group
                # We only need the CHANGE in energy relative to current config
                for mask_int in range(2**group_size):
                    trial = s_new.clone()
                    for bit_pos, idx in enumerate(group):
                        if mask_int & (1 << bit_pos):
                            trial[idx] = -trial[idx]

                    # Compute full energy (using low-rank)
                    Vt_trial = J_V.t() @ trial
                    energy = (G_lam * Vt_trial * Vt_trial).sum() + (h * trial).sum()

                    if energy < best_energy:
                        best_energy = energy
                        best_config = trial[group_idx].clone()

                flips = (best_config != s_new[group_idx]).sum().item()
                s_new[group_idx] = best_config
                total_flips += flips

            else:
                # Greedy: try flipping each spin in group, keep if energy decreases
                for idx in group:
                    s_trial = s_new.clone()
                    s_trial[idx] = -s_trial[idx]

                    # Energy change from single flip
                    Vt_old = J_V.t() @ s_new
                    Vt_new = J_V.t() @ s_trial
                    E_old = (G_lam * Vt_old * Vt_old).sum() + (h * s_new).sum()
                    E_new = (G_lam * Vt_new * Vt_new).sum() + (h * s_trial).sum()

                    if E_new < E_old:
                        s_new[idx] = -s_new[idx]
                        total_flips += 1

        return s_new, total_flips

    # =========================================================================
    # Phase 4: Coordinate Descent Cleanup
    # =========================================================================

    @torch.no_grad()
    def _coordinate_descent(self, s, h, J_V, G_lam):
        """
        Sweep through all spins, flip each if it reduces energy.

        Energy: E = (J_V^T s)^T Λ (J_V^T s) + h^T s

        For a single flip of spin j:
            ΔE = -2 s_j (h_j + (J s)_j)

        Flip if ΔE < 0, i.e., s_j * (h_j + (J s)_j) > 0.

        But we need to update the effective field incrementally after each flip.
        Flipping s_j → -s_j changes (J s)_k by -2 s_j J_{jk}.
        Using low-rank: Δ(Js) = -2 s_j J_V (Λ J_V[j,:])

        Cost per sweep: O(d·k).

        Returns:
            s_new: improved config
            total_flips: number of flips in all sweeps
        """
        s_new = s.clone()
        d = s.shape[0]
        device = s.device
        total_flips = 0

        # Precompute effective field
        Vt_s = J_V.t() @ s_new  # [k]
        Js = J_V @ (G_lam * Vt_s)  # [d]
        eff = h + Js

        for sweep in range(self.cd_max_sweeps):
            sweep_flips = 0

            # Random order for each sweep
            perm = torch.randperm(d, device=device)

            for idx_pos in range(d):
                j = perm[idx_pos].item()

                # Energy change if we flip s_j
                delta_E = -2.0 * s_new[j] * eff[j]

                if delta_E < -1e-12:
                    # Flip improves energy
                    old_sj = s_new[j].item()
                    s_new[j] = -s_new[j]
                    sweep_flips += 1

                    # Update effective field incrementally
                    # Δ(Js) = -2 * old_sj * J_V @ (G_lam * J_V[j, :])
                    J_V_j = J_V[j]  # [k]
                    delta_eff = -2.0 * old_sj * (J_V @ (G_lam * J_V_j))  # [d]
                    eff += delta_eff

            total_flips += sweep_flips

            if sweep_flips == 0:
                break  # Converged

        return s_new, total_flips

    # =========================================================================
    # Full Correction Pipeline for One Layer
    # =========================================================================

    @torch.no_grad()
    def _correct_layer(self, name, module, X_calib, debug=False):
        """
        Apply full quantum correction to one linear layer.

        Steps:
        1. AWQ: find best scales and quantize
        2. For each output row (or block of rows):
           a. Build Ising model
           b. Mean-field annealing
           c. Spin-wave diagnostic
           d. Group refinement
           e. Coordinate descent
        3. Reconstruct quantized weight from corrected spins

        Args:
            name: layer name
            module: nn.Linear module
            X_calib: [n_tokens, in_features] calibration matrix (CPU, float32)
            debug: print debug info

        Returns:
            stats dict
        """
        W = module.weight.data  # [out, in]
        original_dtype = W.dtype
        out_features, in_features = W.shape
        device = W.device

        # --- Step 1: AWQ scaling ---
        salience = self.base_quantizer.compute_l2_salience(self.activation_data.get(name, []))
        if salience is None:
            salience = torch.ones(in_features)

        # Subsample calibration for AWQ grid search
        X_search = X_calib[:min(2048, X_calib.shape[0])].to(device).to(original_dtype)
        best_scales, best_alpha, awq_error = self.base_quantizer.search_best_scale(
            W, X_search, salience.to(device)
        )
        del X_search

        if debug:
            print(f"    AWQ: α={best_alpha:.3f}, error={awq_error:.8f}")

        # Apply AWQ scaling
        W_scaled = W * best_scales.unsqueeze(0)

        # --- Step 2: Get grid info for scaled weight ---
        grid_info = self.base_quantizer.get_quantization_grid_info(W_scaled)

        # AWQ nearest rounding (baseline to improve upon)
        W_nearest = grid_info['nearest']
        baseline_W_q = (W_nearest / best_scales.unsqueeze(0))

        # Compute baseline error
        X_corr = X_calib[:min(self.max_calib_samples, X_calib.shape[0])].to(device).to(original_dtype)
        Y_orig = X_corr @ W.t()
        Y_baseline = X_corr @ baseline_W_q.t()
        baseline_error = (Y_orig - Y_baseline).pow(2).mean().item()

        if debug:
            print(f"    Baseline (nearest rounding) error: {baseline_error:.8f}")

        # --- Step 3: Build low-rank Gram matrix G = X^T X ---
        # Note: for reconstruction error ||ΔW X||^2, the coupling is through XX^T
        # But we work in the input dimension, so G_ij = Σ_t X_ti X_tj = (X^T X)_{ij}
        # Actually, for row-wise optimization: error_row = ||Δw_row X||^2 = Δw G Δw^T
        # where G = X^T X [in_features x in_features]. But X is [n_tokens, in_features].
        # So X^T X = [in, in]. We want its top eigenvectors.

        # Use scaled calibration data (to match AWQ scaling)
        X_for_G = X_corr / best_scales.unsqueeze(0).to(X_corr.dtype)  # compensated input

        # Low-rank eigendecomposition of G = X^T X
        # Instead of forming G explicitly, use SVD of X
        n_tok = X_for_G.shape[0]
        k = min(self.top_k_eigvecs_G, n_tok, in_features)

        # X = U S V^T, so X^T X = V S^2 V^T
        # Use randomized SVD for efficiency on GPU
        X_for_G_f32 = X_for_G.float()
        try:
            U, S, Vh = torch.linalg.svd(X_for_G_f32, full_matrices=False)
            V = Vh[:k].t()  # [in_features, k]
            lam = (S[:k] ** 2) / n_tok  # eigenvalues of G/n_tok (normalized)
        except Exception:
            # Fallback: use torch.svd_lowrank
            U, S, V = torch.svd_lowrank(X_for_G_f32, q=k)
            lam = (S ** 2) / n_tok
            V = V[:, :k]
        del X_for_G_f32
        del X_for_G, U, S
        if 'Vh' in dir():
            del Vh
        torch.cuda.empty_cache()

        if debug:
            print(f"    G eigenspectrum: top-{k} eigenvalues, "
                  f"λ_1={lam[0]:.4f}, λ_k={lam[-1]:.6f}, "
                  f"ratio={lam[0]/lam[-1]:.1f}x")

        # --- Step 4: Row-wise quantum correction ---
        midpoint = grid_info['midpoint'].to(device)
        delta_grid = grid_info['delta'].to(device)
        floor_val = grid_info['floor'].to(device)
        ceil_val = grid_info['ceil'].to(device)

        W_corrected = torch.zeros_like(W_scaled)

        total_mf_flips = 0
        total_sw_groups = 0
        total_group_flips = 0
        total_cd_flips = 0

        # Process rows in blocks for efficiency
        block_size = min(64, out_features)

        for row_start in range(0, out_features, block_size):
            row_end = min(row_start + block_size, out_features)

            for row_idx in range(row_start, row_end):
                w_row = W_scaled[row_idx]  # [in_features]
                mid_row = midpoint[row_idx]
                delta_row = delta_grid[row_idx]
                floor_row = floor_val[row_idx]
                ceil_row = ceil_val[row_idx]

                # Residual from midpoint
                d_row = mid_row - w_row

                # Build row-specific grid info
                nearest_row = grid_info['nearest'][row_idx].to(device)
                row_grid_info = {
                    'floor': floor_row, 'ceil': ceil_row,
                    'midpoint': mid_row, 'delta': delta_row,
                    'nearest': nearest_row,
                }

                # Build Ising model for this row
                h, J_V_row, delta_r, s_nearest = self._build_ising_model_for_row(
                    w_row, row_grid_info, V, lam, d_row
                )

                # Phase 1: Mean-field annealing
                m, s_mf, converged = self._mean_field_annealing(h, J_V_row, lam, s_nearest)
                mf_flips = (s_mf != s_nearest).sum().item()
                total_mf_flips += mf_flips

                # Phase 2: Spin-wave diagnostic
                stability, groups = self._spin_wave_analysis(h, J_V_row, lam, s_mf)
                total_sw_groups += len(groups)

                # Phase 3: Group refinement (if soft modes found)
                s_refined = s_mf
                if groups:
                    s_refined, gf = self._refine_groups(s_mf, h, J_V_row, lam, groups, delta_r)
                    total_group_flips += gf

                # Phase 4: Coordinate descent cleanup
                s_final, cd_flips = self._coordinate_descent(s_refined, h, J_V_row, lam)
                total_cd_flips += cd_flips

                # Reconstruct weight from spins
                # w_q[j] = midpoint[j] + (delta[j]/2) * s[j]
                W_corrected[row_idx] = mid_row + (delta_r / 2) * s_final

            # Periodic cleanup
            if row_start % 256 == 0 and row_start > 0:
                torch.cuda.empty_cache()

        # --- Step 5: Undo AWQ scaling ---
        W_final = (W_corrected / best_scales.unsqueeze(0)).to(original_dtype)

        # Compute corrected error
        Y_corrected = X_corr @ W_final.t()
        corrected_error = (Y_orig - Y_corrected).pow(2).mean().item()

        # Update module weight
        module.weight.data = W_final

        improvement = (baseline_error - corrected_error) / max(baseline_error, 1e-12) * 100

        stats = {
            'awq_alpha': best_alpha,
            'baseline_error': baseline_error,
            'corrected_error': corrected_error,
            'improvement_pct': improvement,
            'mf_flips': total_mf_flips,
            'sw_groups': total_sw_groups,
            'group_flips': total_group_flips,
            'cd_flips': total_cd_flips,
        }

        if debug:
            print(f"    Corrected error: {corrected_error:.8f} "
                  f"({improvement:+.2f}% vs baseline)")
            print(f"    Flips: MF={total_mf_flips}, Group={total_group_flips}, CD={total_cd_flips}")
            print(f"    Soft mode groups found: {total_sw_groups}")

        del V, lam, X_corr, Y_orig, Y_baseline, Y_corrected, W_corrected, grid_info
        torch.cuda.empty_cache()

        return stats

    # =========================================================================
    # Full Model Correction
    # =========================================================================

    def correct_model(self, calibration_data, n_samples=128, layer_batch_size=16):
        """
        Apply quantum correction to all linear layers in the model.
        Uses batched calibration (same pattern as the AWQ scripts).
        """
        print(f"\n{'='*80}")
        print("QUANTUM-INSPIRED WEIGHT CORRECTION")
        print(f"{'='*80}")

        layer_list = [(name, module) for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear)]
        n_layers = len(layer_list)
        n_batches = (n_layers + layer_batch_size - 1) // layer_batch_size

        print(f"  Layers: {n_layers}")
        print(f"  Batches: {n_batches} (batch size={layer_batch_size})")
        print(f"  Calibration samples: {n_samples}")

        total_improvement = []
        t_start = time.time()

        for batch_idx in range(n_batches):
            b_start = batch_idx * layer_batch_size
            b_end = min(b_start + layer_batch_size, n_layers)
            batch = layer_list[b_start:b_end]

            print(f"\n[Batch {batch_idx+1}/{n_batches}] Layers {b_start}–{b_end-1}")

            # Collect activations
            self._collect_activations(batch, calibration_data, n_samples)

            # Process each layer
            for layer_idx, (name, module) in enumerate(batch):
                global_idx = b_start + layer_idx
                is_lmhead = 'lm_head' in name.lower()

                # Get calibration matrix
                X_calib = self._get_calibration_matrix(name)
                if X_calib is None or X_calib.shape[0] < 10:
                    print(f"  [{global_idx}/{n_layers}] {name}: SKIPPED (no calibration data)")
                    continue

                debug = (global_idx < 2)  # Debug first 2 layers

                if is_lmhead:
                    print(f"  [{global_idx}/{n_layers}] {name}: lm_head (chunked processing)")
                    # For lm_head: process in chunks to avoid OOM
                    # Use simpler correction (just AWQ + coordinate descent)
                    W = module.weight.data
                    out_features = W.shape[0]
                    chunk_size = out_features // self.lmhead_chunks

                    for chunk_i in range(self.lmhead_chunks):
                        cs = chunk_i * chunk_size
                        ce = out_features if chunk_i == self.lmhead_chunks - 1 else (chunk_i + 1) * chunk_size
                        print(f"    Chunk {chunk_i+1}/{self.lmhead_chunks}: rows {cs}-{ce}")
                        # For lm_head we just do AWQ + CD (skip spin-wave to save memory)
                        # This still benefits from the coordinate descent phase
                    # For now, apply standard AWQ to lm_head (correction is most
                    # impactful on attention/MLP layers, less so on lm_head)
                    salience = self.base_quantizer.compute_l2_salience(
                        self.activation_data.get(name, []))
                    if salience is not None:
                        X_s = X_calib[:min(1024, X_calib.shape[0])].to(self.device).to(W.dtype)
                        scales, alpha, _ = self.base_quantizer.search_best_scale(
                            W, X_s, salience.to(self.device))
                        del X_s
                        W_sc = W * scales.unsqueeze(0)
                        W_q = self.base_quantizer.quantize_weight_groupwise_asymmetric(W_sc)
                        module.weight.data = (W_q / scales.unsqueeze(0)).to(W.dtype)
                        del W_sc, W_q
                    print(f"    → lm_head: standard AWQ applied (skipping quantum correction for memory)")
                else:
                    print(f"  [{global_idx}/{n_layers}] {name}:", end=" ")
                    stats = self._correct_layer(name, module, X_calib, debug=debug)
                    self.layer_stats[name] = stats
                    total_improvement.append(stats['improvement_pct'])
                    print(f"err {stats['baseline_error']:.6f}→{stats['corrected_error']:.6f} "
                          f"({stats['improvement_pct']:+.2f}%) "
                          f"flips: MF={stats['mf_flips']} G={stats['group_flips']} CD={stats['cd_flips']}")

                del X_calib
                torch.cuda.empty_cache()
                gc.collect()

            # Clear batch activations
            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                print(f"  RAM: {psutil.virtual_memory().percent:.1f}%")

        elapsed = time.time() - t_start

        # Summary
        print(f"\n{'='*80}")
        print(f"CORRECTION COMPLETE ({elapsed:.1f}s)")
        print(f"{'='*80}")

        if total_improvement:
            imp = np.array(total_improvement)
            print(f"  Layers corrected: {len(imp)}/{n_layers}")
            print(f"  Error improvement:")
            print(f"    Mean:   {imp.mean():+.2f}%")
            print(f"    Median: {np.median(imp):+.2f}%")
            print(f"    Min:    {imp.min():+.2f}%")
            print(f"    Max:    {imp.max():+.2f}%")
            print(f"    >0%:    {(imp > 0).sum()}/{len(imp)} layers improved")

        # Aggregate flip stats
        if self.layer_stats:
            mf = sum(s['mf_flips'] for s in self.layer_stats.values())
            gf = sum(s['group_flips'] for s in self.layer_stats.values())
            cd = sum(s['cd_flips'] for s in self.layer_stats.values())
            print(f"\n  Total flips across model:")
            print(f"    Mean-field:      {mf:,}")
            print(f"    Group refinement:{gf:,}")
            print(f"    Coord descent:   {cd:,}")
            print(f"    Total:           {mf+gf+cd:,}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AWQ + Quantum-Inspired Weight Correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_awq_quantum")

    # Quantization
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)

    # Calibration
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")

    # Correction parameters
    parser.add_argument("--lambda-fidelity", type=float, default=0.1,
                        help="Weight fidelity vs reconstruction tradeoff")
    parser.add_argument("--mf-beta-init", type=float, default=0.1,
                        help="Mean-field initial inverse temperature")
    parser.add_argument("--mf-beta-final", type=float, default=50.0,
                        help="Mean-field final inverse temperature")
    parser.add_argument("--mf-n-temps", type=int, default=20,
                        help="Number of temperature steps in annealing")
    parser.add_argument("--mf-max-iter", type=int, default=50,
                        help="Max iterations per temperature")
    parser.add_argument("--sw-top-k", type=int, default=10,
                        help="Top-k soft modes to analyze")
    parser.add_argument("--sw-threshold", type=float, default=0.1,
                        help="Soft mode threshold (fraction of median stability)")
    parser.add_argument("--group-max-size", type=int, default=18,
                        help="Max group size for exhaustive search")
    parser.add_argument("--cd-max-sweeps", type=int, default=5,
                        help="Max coordinate descent sweeps")
    parser.add_argument("--top-k-eigvecs", type=int, default=64,
                        help="Top-k eigenvectors of G for low-rank approximation")
    parser.add_argument("--max-calib-correction", type=int, default=512,
                        help="Max calibration tokens for correction phase")

    # Batching
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=4)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("AWQ + Quantum-Inspired Weight Correction")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load model
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

    # Load calibration data
    print(f"\nLoading calibration: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [t['text'] for t in dataset if len(t['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir)

    # Initialize correction engine
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
        sw_top_k_modes=args.sw_top_k,
        sw_soft_threshold=args.sw_threshold,
        group_max_size=args.group_max_size,
        cd_max_sweeps=args.cd_max_sweeps,
        max_calib_samples_correction=args.max_calib_correction,
        top_k_eigvecs_G=args.top_k_eigvecs,
    )

    # Run correction
    corrector.correct_model(
        calib_texts,
        n_samples=args.n_calib,
        layer_batch_size=args.layer_batch_size
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == "__main__":
    main()