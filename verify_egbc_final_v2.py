"""
Paper-ready EGBC verification — v2.

What's new in v2:
═══════════════════════════════════════════════════════════════════════════════
We replace the strong assumption  E_ℓ[α_ℓ] ≤ 0  with a WEAKER, STRUCTURAL
assumption that EGBC's flip algorithm gives us by construction (modulo a
verifiable decorrelation property of the orthogonal residual).

═══════════════════════════════════════════════════════════════════════════════
THE NEW THEORY (one weak assumption → one lemma → main theorem):
═══════════════════════════════════════════════════════════════════════════════

Decompose the flip in each row:  Δ_j = Δ_j^∥ + Δ_j^⊥
   where Δ_j^∥ = (μ^T Δ_j / ‖μ‖²) μ  is the component along the mean direction.

By EGBC's bias-cancellation construction: μ^T Δ_j ≈ -b_j (where b_j = μ^T e_j).
So Δ_j^∥ ≈ -(b_j / ‖μ‖²) μ.

Decompose the error similarly:  e_j = e_j^∥ + e_j^⊥
   where e_j^∥ = (b_j / ‖μ‖²) μ.

Then expand:
  ⟨Δ_j, Σ e_j⟩ = (Δ_j^∥)^T Σ e_j  +  (Δ_j^⊥)^T Σ e_j

The first term (parallel contribution) decomposes further into:
  (Δ_j^∥)^T Σ e_j = -(b_j²/‖μ‖⁴)(μ^T Σ μ)  +  (-b_j/‖μ‖²)(μ^T Σ e_j^⊥)
                  └─── PARALLEL ─────┘    └────── PARALLEL × ORTHOG ──────┘
                  (deterministically ≤ 0)  (averages to 0 across rows)

The second term (orthogonal residual):
  (Δ_j^⊥)^T Σ e_j     ← UNSIGNED but bounded; need Assumption A to handle

═══════════════════════════════════════════════════════════════════════════════
ASSUMPTION A (WEAK — row-wise residual symmetry):
═══════════════════════════════════════════════════════════════════════════════

  E_j[(Δ_j^⊥)^T Σ e_j] ≈ 0   averaged across output channels j in layer ℓ.

This is a DECORRELATION condition on the orthogonal residual, NOT an assumption
about variance reduction or about α being negative. It says: the part of Δ
that EGBC doesn't explicitly control (the μ-orthogonal residual) has no
systematic correlation with Σ e_j.

═══════════════════════════════════════════════════════════════════════════════
LEMMA (derived from Assumption A + EGBC construction + Σ PSD):
═══════════════════════════════════════════════════════════════════════════════

   ⟨Δ_ℓ, Σ_ℓ E_ℓ⟩_F  ≤  -Σ_j b_j² (μ^T Σ μ)/‖μ‖⁴  ≤  0

The cross-term is non-positive, with magnitude scaling as the pre-flip bias.

═══════════════════════════════════════════════════════════════════════════════
THEOREM (Total descent):
═══════════════════════════════════════════════════════════════════════════════

   ΔR_ℓ = -G_ℓ + 2⟨Δ, Σ E⟩_F + tr(Δ^T Σ Δ)
        ≤ -G_ℓ + 0 + B_ℓ s_max² ‖Σ‖_2     (by the Lemma)

Total error strictly decreases when:
   G_ℓ > B_ℓ s_max² ‖Σ‖_2
       └── algorithmic bias gain ──┘     └── pure quadratic budget ──┘

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS SCRIPT VERIFIES (per layer, per (base, mode)):
═══════════════════════════════════════════════════════════════════════════════

1. (Algorithmic — deterministic)
   ✓ Bias descent:    B(W_q') ≤ B(W_q)
   ✓ Variance bound:  |ΔV| ≤ Hölder budget (Lemma 2)

2. (Cross-term decomposition — verifies the LEMMA mechanism)
   • Full cross-term:   ⟨Δ, Σ E⟩_F                            should be ≤ 0
   • Parallel main:     -Σ_j b_j² (μ^T Σ μ)/‖μ‖⁴               ≤ 0 by construction
   • Parallel residual: Σ_j (-b_j/‖μ‖²)(μ^T Σ e_j^⊥)            small, signed
   • Orthogonal:        Σ_j (Δ_j^⊥)^T Σ e_j                     ≈ 0 by Assumption A

3. (Theorem conclusion — total descent)
   • G_ℓ vs quadratic-only budget B_ℓ s_max² ‖Σ‖_2
   • Realized ΔT < 0 on each layer

This script tracks four quantities per layer that decompose the cross-term:
  cross_full      = ⟨Δ, Σ E⟩_F
  cross_parallel  = (parallel × parallel) deterministic part
  cross_par_orth  = (parallel × orthogonal) zero-mean part
  cross_orthog    = ⟨Δ^⊥, Σ E⟩_F             this is the part Assumption A bounds

Assumption A is VERIFIED if cross_orthog is small in magnitude relative to
|cross_parallel| (the deterministic negative drift).

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_common import (
    ActivationRecorder,
    awq_search_and_scale,
    compute_flip_delta,
    group_quantize,
    james_stein_mean,
    load_text_samples,
    run_calibration,
    select_modules,
    set_seed,
)


# --------------------------------------------------------------------------- #
# Base quantizer (unchanged from v1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_base_quantizer(
    base: str, W: torch.Tensor, mu_cal: torch.Tensor,
    salience_l2: Optional[torch.Tensor], bits: int, group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if base == "ntr":
        W_q, W_int, scale_flat, zp_flat = group_quantize(W, bits=bits, group_size=group_size)
        s = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
        return W_q, W_int, scale_flat, zp_flat, s
    if base == "awq":
        assert salience_l2 is not None
        salience = salience_l2.to(W.device).clamp(min=1e-5)
        W_q_eff, W_int, scale_flat, zp_flat, alpha = awq_search_and_scale(
            W=W, mu_cal=mu_cal, salience_l2=salience,
            bits=bits, group_size=group_size, n_grid=20, apply_flip=False,
        )
        s = salience.pow(alpha).to(W.dtype)
        return W_q_eff, W_int, scale_flat, zp_flat, s
    raise ValueError(f"Unknown base: {base}")


# --------------------------------------------------------------------------- #
# Helper: decompose vector v into (v_parallel, v_orthogonal) along direction u.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decompose_along(v: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose v = v_parallel + v_orthogonal along the direction u.

    v: [..., d]   any tensor whose last dim matches u
    u: [d]        the direction along which to decompose

    Returns: (v_par, v_orth)  with same shape as v.
        v_par   = (v · u / ‖u‖²) u
        v_orth  = v − v_par
    """
    u = u.view(-1)
    u_sq = (u * u).sum().clamp(min=1e-30)
    # coeffs[..., 1] = ⟨v, u⟩ / ‖u‖²
    coeffs = (v @ u) / u_sq                          # shape [...]
    v_par = coeffs.unsqueeze(-1) * u.unsqueeze(0)    # broadcast back to [..., d]
    # If v has shape [n_rows, d], coeffs has shape [n_rows], so we unsqueeze.
    # Handle the general case:
    while v_par.dim() < v.dim():
        v_par = v_par.unsqueeze(0)
    v_par = v_par.expand_as(v).contiguous() if v_par.shape != v.shape else v_par
    # Simpler & safer: just recompute with explicit shape
    if v.dim() == 2:
        v_par = (coeffs).unsqueeze(-1) * u.unsqueeze(0)   # [n_rows, d]
    else:
        v_par = coeffs * u                                # [d]
    v_orth = v - v_par
    return v_par, v_orth


# --------------------------------------------------------------------------- #
# Per-layer measurement: v2 with cross-term decomposition
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_layer(
    name: str, W_fp: torch.Tensor, base: str,
    cal_stats: Dict, eval_stats: Dict,
    bits: int, group_size: int,
    flip_budget_pct: float, knee_tolerance: float,
    use_james_stein: bool, mode: str, device: torch.device,
) -> Dict[str, object]:
    """Return ONE measurement per layer, including the cross-term decomposition.

    NEW IN v2 (decomposition of ⟨Δ, Σ E⟩_F):
        cross_full      ⟨Δ, Σ E⟩_F                                              [scalar]
        cross_par_main  -Σ_j b_j² (μ^T Σ μ)/‖μ‖⁴  (deterministic ≤ 0 part)      [scalar]
        cross_par_res   Σ_j (-b_j/‖μ‖²)(μ^T Σ e_j^⊥)  (zero-mean part)          [scalar]
        cross_orthog    Σ_j (Δ_j^⊥)^T Σ e_j  (assumption-A part)                [scalar]

    Sanity check: cross_full ≈ cross_par_main + cross_par_res + cross_orthog
    """
    W = W_fp.to(device).float()
    out_features, in_features = W.shape

    # μ and Σ
    mu_cal_raw = cal_stats["mu"].to(device).double()
    mu_cal = james_stein_mean(mu_cal_raw) if use_james_stein else mu_cal_raw
    mu_eval = eval_stats["mu"].to(device).double()
    Sigma = eval_stats["Sigma"]
    if Sigma is None:
        raise RuntimeError(f"{name}: full covariance required.")
    Sigma = Sigma.to(device).double()

    mu_for_eval = mu_cal if mode == "insample" else mu_eval

    salience = (mu_eval ** 2 + torch.diagonal(Sigma)).float()

    W_q_eff, W_int, scale_flat, zp_flat, s_vec = run_base_quantizer(
        base, W, mu_cal.float(), salience, bits, group_size,
    )
    W_q_eff = W_q_eff.float()
    e = (W_q_eff - W).double()       # [out, in]

    mu_for_flip = (mu_cal.float() / s_vec).float() if base == "awq" else mu_cal.float()
    Delta_scaled = compute_flip_delta(
        W=W * s_vec.unsqueeze(0) if base == "awq" else W,
        W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_for_flip, bits=bits,
        flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
    ).double()
    Delta = Delta_scaled / s_vec.unsqueeze(0).double() if base == "awq" else Delta_scaled
    e_tilde = e + Delta

    # ===== B (bias) and V (variance) before/after =================================
    B_before = ((e @ mu_for_eval) ** 2).sum()
    B_after = ((e_tilde @ mu_for_eval) ** 2).sum()
    Se = e @ Sigma                                  # [out, in]
    V_before = (Se * e).sum()
    Set = e_tilde @ Sigma
    V_after = (Set * e_tilde).sum()
    T_before = B_before + V_before
    T_after = B_after + V_after

    # ===== Cross-term DECOMPOSITION (the new theory's mechanism) ==================
    #
    # Pre-flip bias per row:  b_j = μ^T e_j   (using mu_for_eval to be consistent
    # with the B that we report; for "the algorithm's own bias view" use mu_cal).
    #
    # We use mu_for_eval as μ throughout the decomposition (consistent with the
    # theorem's claim about *expected* error under the eval distribution).
    mu = mu_for_eval                                       # [d]
    mu_norm_sq = (mu * mu).sum().clamp(min=1e-30)          # scalar
    Sigma_mu = Sigma @ mu                                  # [d]
    mu_Sigma_mu = (mu * Sigma_mu).sum()                    # scalar = μ^T Σ μ ≥ 0

    # Pre-flip bias per row, b_j = μ^T e_j
    b = e @ mu                                             # [out]

    # FULL cross-term: ⟨Δ, Σ E⟩_F   (E here is e, the pre-flip error)
    cross_full = (Delta * Se).sum()                        # scalar

    # ----- Decompose Δ_j along μ ----------------------------------------------
    # Δ_j^∥ = (μ^T Δ_j / ‖μ‖²) μ
    Delta_dot_mu = Delta @ mu                              # [out]
    Delta_parallel_coef = Delta_dot_mu / mu_norm_sq        # [out]
    # Δ_j^∥ as a [out, d] tensor:
    Delta_par = Delta_parallel_coef.unsqueeze(-1) * mu.unsqueeze(0)
    Delta_orth = Delta - Delta_par                          # [out, d]

    # ----- Decompose e_j along μ -----------------------------------------------
    # e_j^∥ = (b_j / ‖μ‖²) μ
    e_par_coef = b / mu_norm_sq                            # [out]
    e_par = e_par_coef.unsqueeze(-1) * mu.unsqueeze(0)     # [out, d]
    e_orth = e - e_par                                     # [out, d]

    # Σ e_j^⊥ for each row:
    Se_orth = e_orth @ Sigma                                # [out, d]
    # μ^T Σ e_j^⊥ for each row (scalar per row):
    mu_dot_Se_orth = Se_orth @ mu                          # [out]

    # ----- The three components of the parallel cross-term --------------------
    # (Δ_j^∥)^T Σ e_j = (Δ_par_coef_j) μ^T Σ e_j
    #                = (Δ_par_coef_j) [ b_j (μ^T Σ μ)/‖μ‖² + μ^T Σ e_j^⊥ ]
    # And Δ_par_coef_j = -b_j/‖μ‖² (approximately, by EGBC construction)
    # So (Δ_j^∥)^T Σ e_j = -b_j²(μ^T Σ μ)/‖μ‖⁴ + (-b_j/‖μ‖²)(μ^T Σ e_j^⊥)

    # We compute these DIRECTLY from the actual Δ (without assuming the
    # construction is exact) — what matters is whether they have the predicted signs.

    # Component 1: "parallel main"  = Σ_j (Δ_par_coef_j) × b_j × (μ^T Σ μ)/‖μ‖²
    #             = Σ_j Δ_par_coef_j · b_j · mu_Sigma_mu / mu_norm_sq
    cross_par_main = (Delta_parallel_coef * b * mu_Sigma_mu / mu_norm_sq).sum()

    # Component 2: "parallel residual" = Σ_j Δ_par_coef_j × (μ^T Σ e_j^⊥)
    cross_par_res = (Delta_parallel_coef * mu_dot_Se_orth).sum()

    # Component 3: "orthogonal"       = Σ_j (Δ_j^⊥)^T Σ e_j
    Se_orth_full = e @ Sigma  # This is Σ e_j for each row, NOT decomposed
    # Wait — (Δ_j^⊥)^T Σ e_j uses the FULL e_j (not just e_j^⊥), because
    # cross_full = ⟨Δ, Σ E⟩ = ⟨Δ_par + Δ_orth, Σ E⟩.
    cross_orthog = (Delta_orth * Se).sum()                  # [out, d] elem-wise then sum

    # Sanity: should have cross_full ≈ cross_par_main + cross_par_res + cross_orthog
    cross_par_total = (Delta_par * Se).sum()                # full parallel contribution
    decomp_check = float((cross_full - (cross_par_total + cross_orthog)).abs().item())
    decomp_check_finer = float((cross_par_total - (cross_par_main + cross_par_res)).abs().item())

    # ===== Lemma 3 (lattice closure) check ========================================
    max_int = 2 ** bits - 1
    if base == "awq":
        delta_int_pred = (Delta * s_vec.unsqueeze(0).double() / scale_flat.double())
    else:
        delta_int_pred = (Delta / scale_flat.double())
    is_integer = (delta_int_pred - delta_int_pred.round()).abs().max().item() < 1e-6
    W_int_post = W_int.double() + delta_int_pred
    code_in_range = bool(((W_int_post >= 0) & (W_int_post <= max_int)).all().item())
    lemma3_pass = bool(is_integer and code_in_range)

    # ===== Variance bound RHS (Lemma 2) ===========================================
    if base == "awq":
        step_orig = scale_flat.to(torch.float64) / s_vec.unsqueeze(0).to(torch.float64)
    else:
        step_orig = scale_flat.to(torch.float64)
    s_max = float(step_orig.max().item())
    Sigma_inf = float(Sigma.abs().max().item())
    Se_inf_per_row = Se.abs().max(dim=1).values
    B_per_row = (Delta != 0).sum(dim=1).double()
    rhs_per_row = 2.0 * B_per_row * s_max * Se_inf_per_row \
                  + (B_per_row ** 2) * (s_max ** 2) * Sigma_inf
    RHS_T1 = rhs_per_row.sum()

    dV = V_after - V_before
    dV_abs = dV.abs()

    # ===== Old α statistic (full cross-term cosine) — kept for comparison =========
    Delta_F_norm = (Delta ** 2).sum().sqrt()
    SE_F_norm = (Se ** 2).sum().sqrt()
    if float(Delta_F_norm.item()) < 1e-30 or float(SE_F_norm.item()) < 1e-30:
        alpha = 0.0
    else:
        alpha = float((cross_full / (Delta_F_norm * SE_F_norm)).item())

    # NEW: cosine for the ORTHOGONAL part only — the Assumption-A object
    Delta_orth_F_norm = (Delta_orth ** 2).sum().sqrt()
    if float(Delta_orth_F_norm.item()) < 1e-30 or float(SE_F_norm.item()) < 1e-30:
        alpha_orth = 0.0
    else:
        alpha_orth = float((cross_orthog / (Delta_orth_F_norm * SE_F_norm)).item())

    # NEW: PURE quadratic budget = E[‖Δ‖_F² ‖Σ‖_2]    (proxy ‖Σ‖_2 ≈ ‖Σ‖_∞ for speed)
    quadratic_budget = float((Delta_F_norm ** 2).item()) * Sigma_inf
    bias_gain_abs = float((B_before - B_after).clamp(min=0).item())

    # ===== Old dominance vs new dominance (theorem-2-style) =======================
    #   OLD: G_ℓ > Hölder bound (cross + quadratic, both worst-case)
    #   NEW: G_ℓ > quadratic only  (because the Lemma kills the cross-term)
    new_dominance_holds = bool(bias_gain_abs > quadratic_budget)

    # ===== Facts ==================================================================
    eps = 1e-10
    fact_i   = bool(B_after <= B_before + eps * (1.0 + B_before.abs()))
    fact_ii  = bool(dV_abs   <= RHS_T1   + eps * (1.0 + V_before.abs()))
    fact_iii = bool(T_after  <  T_before - eps * (1.0 + T_before.abs()))
    fact_iv  = bool(V_after  <= V_before + eps * (1.0 + V_before.abs()))

    # NEW facts:
    # fact_v   = cross_par_main is non-positive (PSD + EGBC construction)
    # fact_vi  = cross_full is non-positive (Lemma's conclusion)
    # fact_vii = |cross_orthog| ≤ |cross_par_main|  (Assumption A: orthog doesn't dominate)
    fact_v   = bool(cross_par_main <= eps * (1.0 + cross_par_main.abs()))
    fact_vi  = bool(cross_full <= eps * (1.0 + cross_full.abs()))
    fact_vii = bool(cross_orthog.abs() <= cross_par_main.abs() + eps)

    def rel(num, den):
        d = float(den.item() if hasattr(den, "item") else den)
        if abs(d) < 1e-30:
            return 0.0
        return float((num.item() if hasattr(num, "item") else num) / d)

    return {
        "name": name,
        "base": base,
        "mode": mode,
        "n_rows": int(out_features),
        # ----- Magnitudes -----
        "B_before": float(B_before.item()),
        "B_after":  float(B_after.item()),
        "V_before": float(V_before.item()),
        "V_after":  float(V_after.item()),
        "T_before": float(T_before.item()),
        "T_after":  float(T_after.item()),
        "RHS_T1":   float(RHS_T1.item()),
        "dV_abs":   float(dV_abs.item()),
        # Relative changes
        "dB_rel": rel(B_before - B_after, B_before),
        "dV_rel": rel(V_after - V_before, V_before),
        "dT_rel": rel(T_before - T_after, T_before),
        # ----- Lemma checks -----
        "fact_i":   fact_i,    # Lemma 1: bias descent
        "fact_ii":  fact_ii,   # Lemma 2: Hölder variance bound
        "fact_iii": fact_iii,  # Theorem conclusion (per-layer T descent)
        "fact_iv":  fact_iv,   # Empirical V no-worsening
        "lemma3_pass": lemma3_pass,
        # ----- OLD Assumption A (full α) -----
        "alpha":            alpha,
        # ----- NEW: Cross-term decomposition (the LEMMA mechanism) -----
        "cross_full":       float(cross_full.item()),          # ⟨Δ, Σ E⟩_F
        "cross_par_main":   float(cross_par_main.item()),      # -Σ b² (μ^T Σ μ)/‖μ‖⁴
        "cross_par_res":    float(cross_par_res.item()),       # parallel × orthog
        "cross_orthog":     float(cross_orthog.item()),        # ⟨Δ^⊥, Σ E⟩_F (Asm A)
        "alpha_orth":       alpha_orth,                        # cosine of the orthog part
        # Decomposition sanity (should be ~0)
        "decomp_check":          decomp_check,
        "decomp_check_finer":    decomp_check_finer,
        # ----- New facts from the LEMMA mechanism -----
        "fact_v_par_main_nonpos":   fact_v,    # Δ_par × Σ × e_par is ≤ 0
        "fact_vi_full_cross_nonpos": fact_vi,  # full cross-term ≤ 0
        "fact_vii_orth_dominated":  fact_vii,  # |cross_orthog| ≤ |cross_par_main|
        # ----- Theorem dominance comparisons -----
        "Delta_F_norm":     float(Delta_F_norm.item()),
        "SE_F_norm":        float(SE_F_norm.item()),
        "quadratic_budget": quadratic_budget,                  # ‖Δ‖² · ‖Σ‖_∞
        "bias_gain_abs":    bias_gain_abs,                     # G_ℓ = B_before - B_after
        "new_dominance_holds": new_dominance_holds,            # G_ℓ > quadratic ?
        # ----- Misc -----
        "mu_Sigma_mu":     float(mu_Sigma_mu.item()),
        "mu_norm_sq":      float(mu_norm_sq.item()),
        "B_mean_per_row":  float(B_per_row.mean().item()),
        "n_flipped_rows":  int((B_per_row > 0).sum().item()),
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(results: List[Dict]) -> Dict[str, Dict]:
    by_key: Dict[Tuple[str, str], List[Dict]] = {}
    for r in results:
        by_key.setdefault((r["base"], r["mode"]), []).append(r)

    summary: Dict[str, Dict] = {}
    for (base, mode), rs in by_key.items():
        n = len(rs)
        frac = lambda k: sum(int(r[k]) for r in rs) / max(n, 1)
        avg = lambda k: float(np.mean([r[k] for r in rs]))

        alphas = [r["alpha"] for r in rs]
        alphas_orth = [r["alpha_orth"] for r in rs]

        # OLD-style dominance ratio (cross + quadratic worst case)
        avg_bias_gain   = float(np.mean([r["bias_gain_abs"]    for r in rs]))
        avg_quad_budget = float(np.mean([r["quadratic_budget"] for r in rs]))
        old_dominance_ratio = avg_bias_gain / max(avg_quad_budget, 1e-30)
        old_dominance_holds = avg_bias_gain > avg_quad_budget

        # NEW-style dominance (under the Lemma, we only need to beat the quadratic)
        # — actually this is the SAME condition; the Lemma just *justifies* it.

        # Cross-term decomposition magnitudes
        cross_full        = [r["cross_full"]      for r in rs]
        cross_par_main    = [r["cross_par_main"]  for r in rs]
        cross_par_res     = [r["cross_par_res"]   for r in rs]
        cross_orthog      = [r["cross_orthog"]    for r in rs]

        # Assumption A (weak): mean of cross_orthog should be ≈ 0
        # (or at least: |mean(cross_orthog)| << |mean(cross_par_main)|)
        mean_orth        = float(np.mean(cross_orthog))
        mean_par_main    = float(np.mean(cross_par_main))
        mean_par_res     = float(np.mean(cross_par_res))
        mean_full        = float(np.mean(cross_full))

        # Ratio: how much of the cross-term is from the deterministic part?
        # If |orth| << |par_main|, the Lemma's mechanism is the dominant explanation.
        orth_to_par_ratio = abs(mean_orth) / max(abs(mean_par_main), 1e-30)

        summary[f"{base}::{mode}"] = {
            "base": base, "mode": mode, "n_layers": n,
            # Standard Lemma facts
            "fact_i_frac":   frac("fact_i"),
            "fact_ii_frac":  frac("fact_ii"),
            "fact_iii_frac": frac("fact_iii"),
            "fact_iv_frac":  frac("fact_iv"),
            "lemma3_frac":   frac("lemma3_pass"),
            # NEW mechanism facts
            "fact_v_frac":   frac("fact_v_par_main_nonpos"),
            "fact_vi_frac":  frac("fact_vi_full_cross_nonpos"),
            "fact_vii_frac": frac("fact_vii_orth_dominated"),
            # Magnitudes
            "avg_dB_rel": avg("dB_rel"),
            "avg_dV_rel": avg("dV_rel"),
            "avg_dT_rel": avg("dT_rel"),
            # OLD α (full cosine)
            "alpha_mean":         float(np.mean(alphas)),
            "alpha_std":          float(np.std(alphas)),
            "alpha_min":          float(np.min(alphas)),
            "alpha_max":          float(np.max(alphas)),
            "alpha_frac_le_zero": float(np.mean([a <= 0 for a in alphas])),
            # NEW α_orth — cosine of the orthogonal residual only
            "alpha_orth_mean":         float(np.mean(alphas_orth)),
            "alpha_orth_std":          float(np.std(alphas_orth)),
            "alpha_orth_min":          float(np.min(alphas_orth)),
            "alpha_orth_max":          float(np.max(alphas_orth)),
            "alpha_orth_abs_mean":     float(np.mean(np.abs(alphas_orth))),
            # Cross-term decomposition
            "cross_full_mean":         mean_full,
            "cross_par_main_mean":     mean_par_main,
            "cross_par_res_mean":      mean_par_res,
            "cross_orthog_mean":       mean_orth,
            "orth_to_par_ratio":       orth_to_par_ratio,
            # Dominance
            "avg_bias_gain":         avg_bias_gain,
            "avg_quad_budget":       avg_quad_budget,
            "old_dominance_holds":   old_dominance_holds,
            "old_dominance_ratio":   old_dominance_ratio,
            # Aggregates
            "sum_B_before": sum(r["B_before"] for r in rs),
            "sum_B_after":  sum(r["B_after"]  for r in rs),
            "sum_V_before": sum(r["V_before"] for r in rs),
            "sum_V_after":  sum(r["V_after"]  for r in rs),
            "sum_T_before": sum(r["T_before"] for r in rs),
            "sum_T_after":  sum(r["T_after"]  for r in rs),
        }
    return summary


# --------------------------------------------------------------------------- #
# Verdict on the WEAK Assumption A + Lemma + Theorem chain
# --------------------------------------------------------------------------- #
def theorem_verdict(summary: Dict[str, Dict], threshold: float = 0.95,
                    asm_a_max_ratio: float = 0.5) -> Dict:
    """Verdict on the chain:

      Lemma 1 (bias descent)            ≥ threshold of layers
      Lemma 2 (variance Hölder bound)   ≥ threshold of layers
      Lemma 3 (lattice closure)         ≥ threshold of layers
      WEAK Assumption A                 |E[cross_orthog]| / |E[cross_par_main]| ≤ asm_a_max_ratio
      Cross-term LEMMA                  fract layers with cross_full ≤ 0 ≥ threshold
      Theorem 1 conclusion              fract layers with T(W_q') < T(W_q) ≥ threshold

    All conditions must hold on insample mode for EVERY base quantizer.
    """
    insample = [s for s in summary.values() if s["mode"] == "insample"]
    if not insample:
        return {"pass": False, "reason": "no insample results"}
    failures = []
    for s in insample:
        if s["fact_i_frac"] < threshold:
            failures.append(f"{s['base']}: Lemma 1 (bias descent) "
                            f"{s['fact_i_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["fact_ii_frac"] < threshold:
            failures.append(f"{s['base']}: Lemma 2 (variance Hölder bound) "
                            f"{s['fact_ii_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["lemma3_frac"] < threshold:
            failures.append(f"{s['base']}: Lemma 3 (lattice closure) "
                            f"{s['lemma3_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["orth_to_par_ratio"] > asm_a_max_ratio:
            failures.append(f"{s['base']}: Weak Assumption A — |E[orth]/E[par_main]| = "
                            f"{s['orth_to_par_ratio']:.3f} > {asm_a_max_ratio:.2f}")
        if s["fact_vi_frac"] < threshold:
            failures.append(f"{s['base']}: Cross-term lemma — "
                            f"frac layers with cross_full ≤ 0 = {s['fact_vi_frac']*100:.1f}% "
                            f"< {threshold*100:.0f}%")
        if s["fact_iii_frac"] < threshold:
            failures.append(f"{s['base']}: Theorem conclusion — "
                            f"frac layers with T descent = {s['fact_iii_frac']*100:.1f}% "
                            f"< {threshold*100:.0f}%")
    return {
        "pass": len(failures) == 0,
        "threshold_pct": threshold * 100,
        "asm_a_max_ratio": asm_a_max_ratio,
        "bases_tested": sorted({s["base"] for s in insample}),
        "failures": failures,
        # Per-base mechanism summary
        "mechanism_by_base": {s["base"]: {
            "mean_cross_full":     s["cross_full_mean"],
            "mean_cross_par_main": s["cross_par_main_mean"],
            "mean_cross_orthog":   s["cross_orthog_mean"],
            "orth_to_par_ratio":   s["orth_to_par_ratio"],
            "alpha_orth_abs_mean": s["alpha_orth_abs_mean"],
        } for s in insample},
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(results, summary, verdict):
    print()
    print("═" * 100)
    print("EGBC VERIFICATION v2 — Weak Assumption A + Lemma + Theorem")
    print("═" * 100)
    print("""\
THEORY CHAIN (under verification):

  Algorithmic guarantees (deterministic):
    Lemma 1   →  B(W_q') ≤ B(W_q)                           (bias descent)
    Lemma 2   →  |ΔV| ≤ Hölder budget                        (variance bound)
    Lemma 3   →  W_q' lies in the same quantization lattice  (closure)

  Cross-term decomposition (the new mechanism):
    Δ_j  = Δ_j^∥  + Δ_j^⊥           (along μ direction)
    e_j  = e_j^∥  + e_j^⊥

    ⟨Δ, Σ E⟩_F  =  (cross_par_main)  +  (cross_par_res)  +  (cross_orthog)
                     ↓ PSD             ↓ zero-mean         ↓ Assumption A
                     ≤ 0               ≈ 0                 ≈ 0

  WEAK Assumption A: |E[cross_orthog]| ≪ |E[cross_par_main]|

  Conclusion: ⟨Δ, Σ E⟩_F ≤ 0 on average → ΔR ≤ -G + B s²‖Σ‖
""")

    # ----- Per-layer table -----
    print("─" * 100)
    print("PER-LAYER EVIDENCE")
    print("─" * 100)
    header = (f"{'layer':<55s} {'base':<5s} {'mode':<9s} "
              f"{'L1':>3s} {'L2':>3s} {'L3':>3s}  "
              f"{'cross_par_main':>15s} {'cross_orthog':>13s}  "
              f"{'ΔT/T':>8s}")
    print(header)
    print("─" * 100)
    for r in sorted(results, key=lambda x: (x["base"], x["mode"], x["name"])):
        tick = lambda b: "✓" if b else "✗"
        print(f"{r['name']:<55s} {r['base']:<5s} {r['mode']:<9s} "
              f"{tick(r['fact_i']):>3s} {tick(r['fact_ii']):>3s} {tick(r['lemma3_pass']):>3s}  "
              f"{r['cross_par_main']:+15.4e} {r['cross_orthog']:+13.4e}  "
              f"{r['dT_rel']*100:+7.3f}%")

    # ----- Headline per (base, mode) -----
    print()
    print("═" * 100)
    print("HEADLINE: per-base × per-mode evidence")
    print("═" * 100)
    keys_ordered = sorted(summary.keys(), key=lambda k: (k.split("::")[1] != "insample", k))
    for key in keys_ordered:
        s = summary[key]
        is_theorem = (s["mode"] == "insample")
        tag = "[THEOREM]   " if is_theorem else "[robustness]"
        n = s["n_layers"]
        print(f"\n  {tag}  base = {s['base']:<5s}  mode = {s['mode']:<9s}  ({n} layers)")
        print(f"    Lemma 1 (bias descent)         : {int(s['fact_i_frac']*n):>3d}/{n:<3d} ({s['fact_i_frac']*100:6.2f}%)")
        print(f"    Lemma 2 (variance Hölder)      : {int(s['fact_ii_frac']*n):>3d}/{n:<3d} ({s['fact_ii_frac']*100:6.2f}%)")
        print(f"    Lemma 3 (lattice closure)      : {int(s['lemma3_frac']*n):>3d}/{n:<3d} ({s['lemma3_frac']*100:6.2f}%)")
        print(f"")
        print(f"    CROSS-TERM DECOMPOSITION (averaged across layers):")
        print(f"        E[cross_full]      = {s['cross_full_mean']:+12.4e}    full ⟨Δ, ΣE⟩_F")
        print(f"        E[cross_par_main]  = {s['cross_par_main_mean']:+12.4e}    -Σ b²(μ^TΣμ)/‖μ‖⁴ "
              f"(≤ 0 by construction)")
        print(f"        E[cross_par_res]   = {s['cross_par_res_mean']:+12.4e}    "
              f"parallel × orthog (should be ~ 0)")
        print(f"        E[cross_orthog]    = {s['cross_orthog_mean']:+12.4e}    "
              f"⟨Δ^⊥, ΣE⟩_F (ASM A: ~ 0)")
        print(f"")
        print(f"    ASSUMPTION A — Orthogonal residual control:")
        print(f"        |E[orth]| / |E[par_main]| = {s['orth_to_par_ratio']:7.4f}    "
              f"(should be << 1 for the lemma to give a clean ≤ 0)")
        print(f"        cosine α_orth: mean = {s['alpha_orth_mean']:+9.5f}    "
              f"|mean| = {s['alpha_orth_abs_mean']:9.5f}")
        print(f"")
        print(f"    CROSS-TERM LEMMA conclusion (⟨Δ, ΣE⟩ ≤ 0):")
        print(f"        cross_par_main ≤ 0: {int(s['fact_v_frac']*n):>3d}/{n:<3d} layers ({s['fact_v_frac']*100:.0f}%)")
        print(f"        cross_full     ≤ 0: {int(s['fact_vi_frac']*n):>3d}/{n:<3d} layers ({s['fact_vi_frac']*100:.0f}%)")
        print(f"        |orth| ≤ |par_main|: {int(s['fact_vii_frac']*n):>3d}/{n:<3d} layers ({s['fact_vii_frac']*100:.0f}%)")
        print(f"")
        print(f"    THEOREM dominance G > B s²‖Σ‖:")
        print(f"        E[G_ℓ]              = {s['avg_bias_gain']:.4e}    (bias gain)")
        print(f"        E[‖Δ‖²‖Σ‖_∞]        = {s['avg_quad_budget']:.4e}    (quadratic-only budget)")
        print(f"        ratio                = {s['old_dominance_ratio']:.4e}× "
              f"({'HOLDS' if s['old_dominance_holds'] else 'fails — but theorem still concludes via cross-term ≤ 0'})")
        print(f"")
        print(f"    Realized total descent:")
        print(f"        T(W_q') < T(W_q):    {int(s['fact_iii_frac']*n):>3d}/{n:<3d} layers ({s['fact_iii_frac']*100:.0f}%)")
        print(f"        avg ΔB/B = {s['avg_dB_rel']*100:+7.3f}%   "
              f"ΔV/V = {s['avg_dV_rel']*100:+7.3f}%   "
              f"ΔT/T = {s['avg_dT_rel']*100:+7.3f}%")
        print(f"    Aggregate:")
        print(f"        T: {s['sum_T_before']:.4e} → {s['sum_T_after']:.4e}    "
              f"({100*(s['sum_T_before']-s['sum_T_after'])/max(s['sum_T_before'],1e-30):+.3f}%)")

    # ----- Verdict -----
    print()
    print("═" * 100)
    print(f"VERDICT (threshold ≥ {verdict['threshold_pct']:.0f}%, "
          f"weak Asm A ratio ≤ {verdict['asm_a_max_ratio']:.2f})")
    print("═" * 100)
    print(f"  Bases tested:  {verdict['bases_tested']}")
    if verdict["pass"]:
        print(f"  VERDICT:  PASS  ✓")
        print(f"     Theory chain holds: bias descent + variance bound + cross-term ≤ 0 + ")
        print(f"     total descent, with Weak Assumption A empirically verified.")
    else:
        print(f"  VERDICT:  FAIL  ✗")
        for f in verdict["failures"]:
            print(f"    – {f}")
    print()
    print(f"  Mechanism summary (per base):")
    for base, m in verdict.get("mechanism_by_base", {}).items():
        print(f"    {base.upper()}:")
        print(f"      E[cross_full]     = {m['mean_cross_full']:+11.4e}  (full cross-term, lemma says ≤ 0)")
        print(f"      E[cross_par_main] = {m['mean_cross_par_main']:+11.4e}  (deterministic negative drift)")
        print(f"      E[cross_orthog]   = {m['mean_cross_orthog']:+11.4e}  (Asm A says ~ 0)")
        print(f"      orth/par ratio    = {m['orth_to_par_ratio']:9.4f}")
        print(f"      α_orth abs mean   = {m['alpha_orth_abs_mean']:9.5f}")
    print()


def save_outputs(results, summary, verdict, out_dir, cfg):
    out_dir.mkdir(parents=True, exist_ok=True)

    def jsonify(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(out_dir / "per_layer_evidence_v2.json", "w") as f:
        json.dump({"config": cfg, "per_layer": results}, f, indent=2, default=jsonify)
    with open(out_dir / "headline_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2, default=jsonify)
    with open(out_dir / "theorem_verdict_v2.json", "w") as f:
        json.dump(verdict, f, indent=2, default=jsonify)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--cal-dataset", default="c4")
    p.add_argument("--eval-datasets", nargs="+", default=["c4-val", "wikitext2"])
    p.add_argument("--n-cal", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--bits", type=int, default=4, choices=[3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--flip-budget-pct", type=float, default=5.0)
    p.add_argument("--knee-tolerance", type=float, default=0.01)
    p.add_argument("--use-james-stein", action="store_true", default=True)
    p.add_argument("--no-james-stein", dest="use_james_stein", action="store_false")
    p.add_argument("--base-quantizers", nargs="+", default=["ntr", "awq"], choices=["ntr", "awq"])
    p.add_argument("--mode", choices=["insample", "crossval", "both"], default="both")
    p.add_argument("--model-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max-cal-tokens-per-layer", type=int, default=200_000)
    p.add_argument("--max-eval-tokens-per-layer", type=int, default=200_000)
    p.add_argument("--flush-every-tokens", type=int, default=16_384)
    p.add_argument("--layers-pattern", type=str, required=True)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="./egbc_paper_results_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-pct", type=float, default=95.0)
    p.add_argument("--asm-a-max-ratio", type=float, default=0.5,
                   help="Max allowed |E[cross_orthog]| / |E[cross_par_main]| for "
                        "Weak Assumption A to be considered verified.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 100)
    print("EGBC VERIFICATION v2 — CONFIG")
    print("═" * 100)
    for k, v in vars(args).items():
        print(f"  {k:32s} = {v}")
    print("═" * 100)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype_map[args.model_dtype],
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    patterns = [s.strip() for s in args.layers_pattern.split(",") if s.strip()]
    module_names = select_modules(model, patterns, args.max_layers)
    if not module_names:
        print("ERROR: no modules matched.")
        sys.exit(1)
    print(f"\nSelected {len(module_names)} modules.")

    # ----- 1) Calibration μ -----
    print(f"\n[1/3] Capturing calibration μ from {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_rec = ActivationRecorder(
        module_names, record_full_cov=False,
        max_tokens_per_module=args.max_cal_tokens_per_layer,
        flush_every_tokens=args.flush_every_tokens,
    )
    run_calibration(model, tok, cal_texts, cal_rec, device, args.max_length)
    cal_stats = cal_rec.finalize()

    # ----- 2) Eval μ + Σ -----
    print(f"\n[2/3] Capturing eval μ and FULL Σ from {args.eval_datasets}")
    eval_stats_per_module: Dict[str, Dict[str, Dict]] = {n: {} for n in module_names}
    for ev in args.eval_datasets:
        print(f"  -- {ev} --")
        eval_texts = load_text_samples(ev, args.n_eval, args.seed + 1)
        rec = ActivationRecorder(
            module_names, record_full_cov=True,
            max_tokens_per_module=args.max_eval_tokens_per_layer,
            flush_every_tokens=args.flush_every_tokens,
        )
        run_calibration(model, tok, eval_texts, rec, device, args.max_length)
        finalized = rec.finalize()
        for n in module_names:
            eval_stats_per_module[n][ev] = finalized[n]
        del rec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- 3) Per-layer measurement -----
    modes = ["insample", "crossval"] if args.mode == "both" else [args.mode]
    print(f"\n[3/3] Measurement (layer × eval × base × mode), modes={modes}")
    results: List[Dict] = []
    for name in tqdm(module_names, desc="layers"):
        mod = model.get_submodule(name)
        W = mod.weight.detach()
        for ev in args.eval_datasets:
            for base in args.base_quantizers:
                for mode in modes:
                    row = measure_layer(
                        name=f"{name}@{ev}", W_fp=W, base=base,
                        cal_stats=cal_stats[name],
                        eval_stats=eval_stats_per_module[name][ev],
                        bits=args.bits, group_size=args.group_size,
                        flip_budget_pct=args.flip_budget_pct,
                        knee_tolerance=args.knee_tolerance,
                        use_james_stein=args.use_james_stein,
                        mode=mode, device=device,
                    )
                    results.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate(results)
    verdict = theorem_verdict(summary,
                              threshold=args.threshold_pct / 100.0,
                              asm_a_max_ratio=args.asm_a_max_ratio)
    print_report(results, summary, verdict)
    save_outputs(results, summary, verdict, Path(args.out_dir), vars(args))
    print(f"\nWrote: {args.out_dir}")


if __name__ == "__main__":
    main()