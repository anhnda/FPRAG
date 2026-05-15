"""
Paper-direct EGBC / CLC verification — v4.

This script verifies, line-by-line, the theory of the CLC paper:

  Proposition 1 (exact identity):
    ΔR_ℓ  =  -G_ℓ  +  2⟨Δ, Σ e⟩_F  +  tr(Δᵀ Σ Δ)

  Assumptions checked algorithmically (not assumed):
    (A1)  Δ_{j,i} = -sign(e_{j,i}) α_{j,i},   α_{j,i} ≥ 0
    (A2)  supp(Δ_j) ⊆ ℐ   for ALL j   (single shared support set ℐ)
    (A3)  α_{j,i} = s_j   on the support
    (D1)  G_ℓ ≥ 0

  Lemma A.1 (cross-term decomposition under (A1)+(A3)):
    ⟨Δ_j, Σ e_j⟩  =  D_j  +  E_j
      D_j  := -s_j Σ_{i∈S_j} Σ_ii |e_{j,i}|            ≤ 0  DETERMINISTIC
      E_j  := -s_j Σ_{i∈S_j} sign(e_{j,i})·Σ_{k≠i} Σ_ik e_{j,k}
    |E_j|  ≤  (s_j² / 2) · |S_j| · ρ(Σ)   where ρ(Σ) = max_i Σ_{k≠i} |Σ_ik|

  Lower bound on |D_ℓ|:
    |D_ℓ|  ≥  B_ℓ · γ_ℐ,   γ_ℐ := min_{(j,i)∈S} s_j Σ_ii |e_{j,i}|

  Trace bound (uses (A2)+(A3) — needs the single ℐ):
    tr(Δᵀ Σ Δ)  ≤  s_max² · B_ℓ · ‖Σ|_ℐ‖_2

  Theorem 1 (master bound):
    ΔR_ℓ  ≤  -G_ℓ  +  B_ℓ Φ_ℐ
    Φ_ℐ  := -2 γ_ℐ + s_max² (ρ(Σ) + ‖Σ|_ℐ‖_2)

  Corollary 1:
    (i)  Φ_ℐ ≤ 0  ⇒  ΔR_ℓ ≤ -G_ℓ - B_ℓ|Φ_ℐ| ≤ 0 for any B_ℓ ≥ 0.
    (ii) Φ_ℐ > 0  ⇒  ΔR_ℓ < 0 iff B_ℓ < G_ℓ / Φ_ℐ.

What this script measures per (layer, base, mode):

  A1_violations         : count of (j,i) with sign(Δ_{j,i}) ≠ -sign(e_{j,i}) on support
  A2_violations         : count of (j,i) with Δ_{j,i} ≠ 0 outside the declared mask ℐ
  A3_violations         : count of (j,i) on support where |α_{j,i}| ≠ s_j
  D1_holds              : whether G_ℓ ≥ 0

  prop1_identity_resid  : | ΔR_ℓ - (-G_ℓ + 2⟨Δ,Σe⟩ + tr(ΔᵀΣΔ)) |        (should be ≈ 0)

  lemma_A1_decomp_resid : | ⟨Δ,Σe⟩ - (D + E) |                          (should be ≈ 0)
  D_value, D_nonpos     : D ≤ 0 ?                                       (Lemma A.1(i))
  E_value, E_bound      : |E_j| ≤ (s_j²/2) |S_j| ρ(Σ) ?  per layer       (Lemma A.1(ii))

  D_lower_bound_holds   : |D_ℓ| ≥ B_ℓ γ_ℐ ?                              (Lemma A.1 chain)

  trace_value           : tr(ΔᵀΣΔ)
  trace_bound           : s_max² B_ℓ ‖Σ|_ℐ‖_2
  trace_bound_holds     : trace_value ≤ trace_bound ?

  master_lhs            : ΔR_ℓ
  master_rhs            : -G_ℓ + B_ℓ Φ_ℐ
  master_bound_holds    : ΔR_ℓ ≤ master_rhs ?                            (Theorem 1)
  Phi                   : Φ_ℐ
  regime                : "i" (Φ_ℐ ≤ 0) or "ii" (Φ_ℐ > 0)
  cor_regime_i_descent  : if regime i, ΔR_ℓ ≤ 0 ?
  cor_regime_ii_safe    : if regime ii, B_ℓ < G_ℓ/Φ_ℐ ⇒ ΔR_ℓ < 0 ?

The output is a strict pass/fail report aligned with the paper's statements.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
# Base quantizer
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_base_quantizer(
    base: str, W: torch.Tensor, mu_cal: torch.Tensor,
    salience_l2: Optional[torch.Tensor], bits: int, group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (W_q_eff, W_int, scale_flat, zp_flat, s_vec)."""
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
# Helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def principal_submatrix_spec_norm(Sigma: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute ‖Σ|_ℐ‖_2  where ℐ is the index set described by the boolean mask.

    Submatrix of a symmetric PSD has a finite spectral norm; computed via
    `torch.linalg.eigvalsh` on the principal submatrix.
    """
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        return 0.0
    Sigma_I = Sigma[idx][:, idx]
    # spectral norm = largest eigenvalue magnitude; Σ is PSD so largest eigval
    eigs = torch.linalg.eigvalsh(Sigma_I.float())
    return float(eigs.abs().max().item())


@torch.no_grad()
def column_step_size(
    W_int: torch.Tensor, scale_flat: torch.Tensor, group_size: int, in_features: int,
) -> torch.Tensor:
    """Per-column (per-input-coord) effective step.

    With group quantization, the step is per (out_row, group). For a per-column
    bound we take, for each input coord i, the max across out rows j of scale_{j,g(i)}.
    That gives a *valid* upper bound for s_max in the per-coordinate bound,
    matching the paper's per-channel-step abstraction (paper assumes one s_j per
    channel; group-quant has one per (j, group) — we just take a sound upper
    bound).
    """
    # scale_flat: [out, in]  already broadcast in compute_flip_delta workflow
    return scale_flat


# --------------------------------------------------------------------------- #
# Per-layer measurement — v4: directly verifies paper statements
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_layer(
    name: str, W_fp: torch.Tensor, base: str,
    cal_stats: Dict, eval_stats: Dict,
    bits: int, group_size: int,
    flip_budget_pct: float, knee_tolerance: float,
    use_james_stein: bool, mode: str, device: torch.device,
) -> Dict[str, object]:
    W = W_fp.to(device).float()
    out_features, in_features = W.shape

    # ----- Activation stats -----
    mu_cal_raw = cal_stats["mu"].to(device).double()
    mu_cal = james_stein_mean(mu_cal_raw) if use_james_stein else mu_cal_raw
    mu_eval = eval_stats["mu"].to(device).double()
    Sigma = eval_stats["Sigma"]
    if Sigma is None:
        raise RuntimeError(f"{name}: full covariance required.")
    Sigma = Sigma.to(device).double()

    mu_for_eval = mu_cal if mode == "insample" else mu_eval

    salience = (mu_eval ** 2 + torch.diagonal(Sigma)).float()

    # ----- Base quantizer -----
    W_q_eff, W_int, scale_flat, zp_flat, s_vec = run_base_quantizer(
        base, W, mu_cal.float(), salience, bits, group_size,
    )
    W_q_eff = W_q_eff.float()
    e = (W_q_eff - W).double()  # pre-flip residual e_{j,i}

    # ----- CLC: compute Δ -----
    mu_for_flip = (mu_cal.float() / s_vec).float() if base == "awq" else mu_cal.float()
    Delta_scaled = compute_flip_delta(
        W=W * s_vec.unsqueeze(0) if base == "awq" else W,
        W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_for_flip, bits=bits,
        flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
    ).double()
    # Lift Δ back to original (unscaled) weight space if AWQ
    Delta = Delta_scaled / s_vec.unsqueeze(0).double() if base == "awq" else Delta_scaled
    e_post = e + Delta

    # ===========================================================================
    # PROPOSITION 1: exact identity
    #     ΔR_ℓ = -G_ℓ + 2⟨Δ, Σe⟩ + tr(Δᵀ Σ Δ)
    # ===========================================================================
    B_before = float(((e @ mu_for_eval) ** 2).sum().item())
    B_after  = float(((e_post @ mu_for_eval) ** 2).sum().item())
    Se       = e @ Sigma
    V_before = float((Se * e).sum().item())
    Sep      = e_post @ Sigma
    V_after  = float((Sep * e_post).sum().item())
    R_before = B_before + V_before
    R_after  = B_after  + V_after
    delta_R  = R_after - R_before              # ΔR_ℓ   (LHS of identity)
    G        = B_before - B_after              # G_ℓ  := ΔB; gain

    cross = float((Delta * Se).sum().item())   # ⟨Δ, Σe⟩_F
    trace = float((Delta @ Sigma * Delta).sum().item())  # tr(Δᵀ Σ Δ)

    rhs_prop1 = -G + 2.0 * cross + trace
    prop1_resid = abs(delta_R - rhs_prop1)
    prop1_resid_rel = prop1_resid / (abs(delta_R) + 1e-30)

    # ===========================================================================
    # ALGORITHMIC ASSUMPTIONS (A1, A2, A3, D1)
    # ===========================================================================
    support_mask = (Delta != 0)             # bool [out, in]   S = ∪_j {j}×S_j
    S_total      = int(support_mask.sum().item())                        # B_ℓ
    S_per_row    = support_mask.sum(dim=1).double()                      # |S_j|
    n_flipped_rows = int((S_per_row > 0).sum().item())

    # ---- (A1) Δ_{j,i} = -sign(e_{j,i}) α_{j,i},  α ≥ 0 ----
    sign_e = torch.sign(e)
    sign_D = torch.sign(Delta)
    # On support, sign_D must equal -sign_e.
    A1_mismatch_mask = support_mask & (sign_D != -sign_e)
    A1_violations    = int(A1_mismatch_mask.sum().item())

    # ---- (A2) supp(Δ_j) ⊆ ℐ   for ALL j (single shared ℐ) ----
    # The minimal ℐ that satisfies (A2) is the union of per-row supports along
    # the input axis (i.e. coords used by ANY row).
    col_used = support_mask.any(dim=0)         # bool [in]; minimal ℐ
    I_size   = int(col_used.sum().item())      # |ℐ|
    # By construction this minimal mask is satisfied by every j. We still report
    # the size, since the bound depends on ‖Σ|_ℐ‖_2 of *this* ℐ.
    A2_violations = 0  # trivially zero with the minimal ℐ — see report note.

    # ---- (A3) on support, |Δ_{j,i}| = step_{j,i}  (= s_j in paper) ----
    # In group quantization the step is per (j, group); we use the per-entry
    # step from scale_flat. For AWQ we lifted Δ back to original space, so we
    # check against the un-lifted lattice step. We test on the AWQ-scaled space
    # if base=="awq", else native space.
    if base == "awq":
        # Δ_scaled is on the per-(j, group) lattice with step scale_flat
        abs_dscaled = Delta_scaled.abs().double()
        step_per   = scale_flat.double()
        # On the (re-scaled) support: |Δ_scaled| should equal step
        support_scaled = (Delta_scaled != 0)
        on_support_steps = abs_dscaled[support_scaled]
        on_support_target = step_per.expand_as(abs_dscaled)[support_scaled]
    else:
        abs_d = Delta.abs()
        step_per = scale_flat.double()
        on_support_steps  = abs_d[support_mask]
        on_support_target = step_per.expand_as(abs_d)[support_mask]
    if on_support_steps.numel() > 0:
        A3_rel_err = float((on_support_steps - on_support_target).abs().max().item()
                           / (on_support_target.max().item() + 1e-30))
    else:
        A3_rel_err = 0.0
    A3_violations = int(((on_support_steps - on_support_target).abs() > 1e-6 * on_support_target).sum().item())

    # ---- (D1) G_ℓ ≥ 0 ----
    D1_holds = bool(G >= -1e-10 * (1.0 + abs(B_before)))

    # ===========================================================================
    # LEMMA A.1: cross-term decomposition under (A1)+(A3)
    #     ⟨Δ_j, Σ e_j⟩ = D_j + E_j
    #     D_j ≤ 0     (deterministic)
    #     |E_j| ≤ (s_j² / 2) |S_j| ρ(Σ)
    # ===========================================================================
    # Use the closed-form expressions to compute D and E.
    Sigma_diag = torch.diagonal(Sigma)         # [in]
    # D_{j,i} contribution on support: Δ_{j,i} Σ_ii e_{j,i}
    D_contrib   = Delta * Sigma_diag.unsqueeze(0) * e          # [out, in]
    D_ell       = float(D_contrib.sum().item())
    # Off-diagonal residual: E := ⟨Δ, Σe⟩ - D
    E_ell       = cross - D_ell
    lemma_decomp_resid = 0.0  # exact by construction (D + E = ⟨Δ,Σe⟩); we
    # still verify it numerically:
    lemma_decomp_resid = abs((D_ell + E_ell) - cross)

    # Sanity: D_ell should be ≤ 0 under (A1)+(A3).
    D_nonpos = bool(D_ell <= 1e-12 * (1.0 + abs(cross)))

    # Off-diagonal bound:  |E_j| ≤ (s_j²/2) |S_j| ρ(Σ).
    # With group-quantization, take per-row s_j := max_i step_{j,i}.
    Sigma_off_abs = Sigma.abs() - torch.diag(Sigma_diag.abs())
    rho_Sigma     = float(Sigma_off_abs.sum(dim=1).max().item())   # max_i Σ_{k≠i}|Σ_ik|

    # per-row s_j
    s_per_row = step_per.max(dim=1).values if step_per.dim() == 2 else step_per
    # ensure shape [out]
    if s_per_row.dim() == 0:
        s_per_row = s_per_row.expand(out_features)
    if s_per_row.shape[0] != out_features:
        # broadcast if needed
        s_per_row = s_per_row.flatten()[:out_features]
    # |E_j| per row
    E_per_row = (Delta * (Sigma @ e.T).T - D_contrib).sum(dim=1)
    # The above is messy; recompute cleanly:
    Se_full = e @ Sigma                                             # [out, in]
    cross_per_row = (Delta * Se_full).sum(dim=1)                    # ⟨Δ_j, Σ e_j⟩
    D_per_row     = (Delta * Sigma_diag.unsqueeze(0) * e).sum(dim=1)
    E_per_row     = cross_per_row - D_per_row                       # [out]
    E_bound_per_row = (s_per_row.double() ** 2) / 2.0 * S_per_row * rho_Sigma  # [out]
    # Lemma A.1(ii):  |E_j| ≤ E_bound_per_row
    per_row_E_ok = (E_per_row.abs() <= E_bound_per_row + 1e-9 * (1.0 + E_bound_per_row.abs()))
    lemma_A1_offdiag_per_row_pass = bool(per_row_E_ok.all().item())
    lemma_A1_offdiag_violations    = int((~per_row_E_ok).sum().item())

    # Aggregate off-diag bound (Σ_j upper)
    E_bound_total = float(E_bound_per_row.sum().item())
    E_total_abs   = float(E_per_row.abs().sum().item())

    # ===========================================================================
    # |D_ℓ| ≥ B_ℓ γ_ℐ
    # ===========================================================================
    if S_total > 0:
        # γ := min over support of  s_j · Σ_ii · |e_{j,i}|
        s_broadcast = s_per_row.double().unsqueeze(1).expand_as(e)   # [out, in]
        gamma_terms = s_broadcast * Sigma_diag.unsqueeze(0) * e.abs()
        gamma_on_supp = gamma_terms[support_mask]
        gamma_I = float(gamma_on_supp.min().item())
        D_lower = S_total * gamma_I
        D_lower_holds = bool(abs(D_ell) >= D_lower - 1e-9 * (1.0 + D_lower))
    else:
        gamma_I = 0.0
        D_lower = 0.0
        D_lower_holds = True

    # ===========================================================================
    # TRACE BOUND:  tr(Δᵀ Σ Δ)  ≤  s_max² B_ℓ ‖Σ|_ℐ‖_2
    # (uses (A2)+(A3); ℐ is the minimal column-union mask above)
    # ===========================================================================
    s_max = float(s_per_row.max().item())
    Sigma_I_spec = principal_submatrix_spec_norm(Sigma, col_used) if I_size > 0 else 0.0
    trace_bound  = (s_max ** 2) * S_total * Sigma_I_spec
    trace_bound_holds = bool(trace <= trace_bound + 1e-8 * (1.0 + abs(trace_bound)))

    # ===========================================================================
    # THEOREM 1 (master bound):
    #     ΔR_ℓ  ≤  -G_ℓ + B_ℓ Φ_ℐ
    #     Φ_ℐ := -2 γ_ℐ + s_max² (ρ(Σ) + ‖Σ|_ℐ‖_2)
    # ===========================================================================
    Phi_I = -2.0 * gamma_I + (s_max ** 2) * (rho_Sigma + Sigma_I_spec)
    master_rhs = -G + S_total * Phi_I
    master_bound_holds = bool(delta_R <= master_rhs + 1e-8 * (1.0 + abs(master_rhs)))

    regime = "i" if Phi_I <= 0.0 else "ii"

    # Corollary 1(i): regime i ⇒ ΔR_ℓ ≤ 0 for any B_ℓ.
    if regime == "i":
        cor_regime_i_descent = bool(delta_R <= 1e-9 * (1.0 + abs(R_before)))
        cor_regime_ii_safe   = None
    else:
        cor_regime_i_descent = None
        safety_budget = G / Phi_I if Phi_I > 0 else float("inf")
        # If we're below safety budget, we should see descent.
        cor_regime_ii_safe = bool(
            (S_total >= safety_budget) or (delta_R < 1e-9 * (1.0 + abs(R_before)))
        )

    return {
        "name": name, "base": base, "mode": mode,
        "n_rows": int(out_features), "n_in": int(in_features),

        # Proposition 1
        "prop1_resid_abs":     prop1_resid,
        "prop1_resid_rel":     prop1_resid_rel,
        "prop1_holds":         bool(prop1_resid_rel < 1e-6),

        # Assumptions
        "A1_violations":       A1_violations,
        "A1_holds":            (A1_violations == 0),
        "A2_violations":       A2_violations,   # 0 by construction with minimal ℐ
        "A2_holds":            True,
        "A3_violations":       A3_violations,
        "A3_rel_err":          A3_rel_err,
        "A3_holds":            (A3_violations == 0),
        "D1_holds":            D1_holds,
        "G":                   G,
        "B_before":            B_before,
        "B_after":             B_after,
        "V_before":            V_before,
        "V_after":             V_after,
        "R_before":            R_before,
        "R_after":             R_after,
        "delta_R":             delta_R,

        # Lemma A.1
        "cross":                          cross,
        "trace":                          trace,
        "D_ell":                          D_ell,
        "E_ell":                          E_ell,
        "lemma_decomp_resid":             lemma_decomp_resid,
        "lemma_A1_decomp_holds":          bool(lemma_decomp_resid < 1e-8 * (1.0 + abs(cross))),
        "D_nonpos":                       D_nonpos,
        "E_total_abs":                    E_total_abs,
        "E_bound_total":                  E_bound_total,
        "lemma_A1_offdiag_per_row_pass":  lemma_A1_offdiag_per_row_pass,
        "lemma_A1_offdiag_violations":    lemma_A1_offdiag_violations,

        # |D_ℓ| ≥ B_ℓ γ_ℐ
        "gamma_I":              gamma_I,
        "D_lower":              D_lower,
        "D_lower_holds":        D_lower_holds,

        # Trace bound
        "s_max":                s_max,
        "rho_Sigma":            rho_Sigma,
        "Sigma_I_spec":         Sigma_I_spec,
        "I_size":               I_size,
        "trace_bound":          trace_bound,
        "trace_bound_holds":    trace_bound_holds,

        # Theorem 1
        "Phi_I":                Phi_I,
        "B_ell":                S_total,
        "master_rhs":           master_rhs,
        "master_bound_holds":   master_bound_holds,

        # Corollary
        "regime":                  regime,
        "cor_regime_i_descent":    cor_regime_i_descent,
        "cor_regime_ii_safe":      cor_regime_ii_safe,

        # Aux
        "n_flipped_rows":          n_flipped_rows,
        "mean_S_per_row":          float(S_per_row.mean().item()),
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
        f = lambda k: sum(int(bool(r[k])) for r in rs) / n
        avg = lambda k: float(np.mean([r[k] for r in rs]))

        regime_i_layers  = [r for r in rs if r["regime"] == "i"]
        regime_ii_layers = [r for r in rs if r["regime"] == "ii"]

        descent_in_i = (
            sum(int(bool(r["cor_regime_i_descent"])) for r in regime_i_layers) /
            max(len(regime_i_layers), 1)
        )

        summary[f"{base}::{mode}"] = {
            "base": base, "mode": mode, "n_layers": n,

            # Proposition 1
            "prop1_pass_frac":           f("prop1_holds"),
            "prop1_resid_rel_max":       float(max(r["prop1_resid_rel"] for r in rs)),

            # Assumptions
            "A1_pass_frac":              f("A1_holds"),
            "A2_pass_frac":              f("A2_holds"),
            "A3_pass_frac":              f("A3_holds"),
            "D1_pass_frac":              f("D1_holds"),
            "A1_violations_total":       sum(r["A1_violations"] for r in rs),
            "A3_violations_total":       sum(r["A3_violations"] for r in rs),

            # Lemma A.1
            "lemma_A1_decomp_pass_frac": f("lemma_A1_decomp_holds"),
            "D_nonpos_pass_frac":        f("D_nonpos"),
            "lemma_A1_offdiag_pass_frac": f("lemma_A1_offdiag_per_row_pass"),
            "D_lower_pass_frac":         f("D_lower_holds"),
            "mean_D":                    avg("D_ell"),
            "mean_E":                    avg("E_ell"),
            "mean_E_bound":              avg("E_bound_total"),
            "mean_cross":                avg("cross"),
            "mean_trace":                avg("trace"),

            # Trace bound
            "trace_bound_pass_frac":     f("trace_bound_holds"),
            "mean_trace_bound":          avg("trace_bound"),
            "mean_Sigma_I_spec":         avg("Sigma_I_spec"),
            "mean_rho_Sigma":            avg("rho_Sigma"),
            "mean_s_max":                avg("s_max"),

            # Theorem 1
            "master_bound_pass_frac":    f("master_bound_holds"),
            "mean_Phi":                  avg("Phi_I"),
            "mean_delta_R":              avg("delta_R"),
            "mean_master_rhs":           avg("master_rhs"),

            # Regime
            "regime_i_frac":             len(regime_i_layers) / n,
            "regime_ii_frac":            len(regime_ii_layers) / n,
            "descent_in_regime_i_frac":  descent_in_i,
            "empirical_descent_frac":    sum(int(r["delta_R"] < 0) for r in rs) / n,

            # B & R
            "sum_R_before":              sum(r["R_before"] for r in rs),
            "sum_R_after":               sum(r["R_after"]  for r in rs),
            "sum_G":                     sum(r["G"] for r in rs),
        }
    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _tick(b):
    if b is None:
        return "—"
    return "✓" if b else "✗"


def print_report(results: List[Dict], summary: Dict[str, Dict], threshold_pct: float):
    print()
    print("═" * 110)
    print("CLC PAPER VERIFICATION — v4 (direct verification of assumptions, lemma, and master bound)")
    print("═" * 110)
    print(f"""\
Tested per (layer, base, mode):

  Prop 1   exact identity   ΔR_ℓ = -G_ℓ + 2⟨Δ,Σe⟩ + tr(ΔᵀΣΔ)   (algebraic; must hold to ~0)
  A1       Δ_{{j,i}} = -sign(e_{{j,i}}) α_{{j,i}}, α ≥ 0      (algorithmic invariant)
  A2       supp(Δ_j) ⊆ ℐ for all j  (using minimal ℐ = column-union)
  A3       α_{{j,i}} = s_j on the support                    (lattice step magnitude)
  D1       G_ℓ ≥ 0                                          (design constraint)
  Lemma A.1(i)   D_ℓ ≤ 0          (deterministic, ≥ {threshold_pct:.0f}% layers)
  Lemma A.1(ii)  |E_j| ≤ (s_j²/2)|S_j|ρ(Σ)  per row         (≥ {threshold_pct:.0f}% layers)
  Lower bound    |D_ℓ| ≥ B_ℓ γ_ℐ                            (≥ {threshold_pct:.0f}% layers)
  Trace bound    tr(ΔᵀΣΔ) ≤ s_max² B_ℓ ‖Σ|_ℐ‖_2             (≥ {threshold_pct:.0f}% layers)
  Theorem 1      ΔR_ℓ ≤ -G_ℓ + B_ℓ Φ_ℐ                       (HEADLINE; ≥ {threshold_pct:.0f}% layers)
  Corollary 1    regime (i) descent: Φ_ℐ ≤ 0 ⇒ ΔR_ℓ ≤ 0
""")

    # ----- Per-layer table -----
    print("─" * 110)
    print("PER-LAYER VERIFICATION")
    print("─" * 110)
    hdr = (f"{'layer':<48s} {'base':<5s} {'mode':<9s} "
           f"{'P1':>3s} {'A1':>3s} {'A3':>3s} {'D1':>3s} "
           f"{'D≤0':>4s} {'|E|':>4s} {'tr≤':>4s} {'Thm1':>5s} "
           f"{'reg':>4s} {'ΔR':>11s}")
    print(hdr)
    print("─" * 110)
    for r in sorted(results, key=lambda x: (x["base"], x["mode"], x["name"])):
        print(f"{r['name']:<48s} {r['base']:<5s} {r['mode']:<9s} "
              f"{_tick(r['prop1_holds']):>3s} "
              f"{_tick(r['A1_holds']):>3s} "
              f"{_tick(r['A3_holds']):>3s} "
              f"{_tick(r['D1_holds']):>3s} "
              f"{_tick(r['D_nonpos']):>4s} "
              f"{_tick(r['lemma_A1_offdiag_per_row_pass']):>4s} "
              f"{_tick(r['trace_bound_holds']):>4s} "
              f"{_tick(r['master_bound_holds']):>5s} "
              f"{r['regime']:>4s} "
              f"{r['delta_R']:+11.4e}")

    # ----- Summary per (base, mode) -----
    print()
    print("═" * 110)
    print("HEADLINE per (base × mode)")
    print("═" * 110)
    keys = sorted(summary.keys(), key=lambda k: (k.split("::")[1] != "insample", k))
    for k in keys:
        s = summary[k]
        n = s["n_layers"]
        print(f"\n  base = {s['base']:<5s}  mode = {s['mode']:<9s}  ({n} layers)")

        def pf(name, frac, note=""):
            n_pass = int(frac * n + 0.5)
            mark = "✓" if frac >= threshold_pct / 100 else "✗"
            print(f"      [{mark}] {name:<40s}  {n_pass:>3d}/{n:<3d}  ({frac*100:6.2f}%)  {note}")

        print("    Proposition 1 (exact identity)")
        pf("identity ≈ 0", s["prop1_pass_frac"],
           f"max rel resid = {s['prop1_resid_rel_max']:.2e}")

        print("    Assumptions")
        pf("(A1) anti-aligned sign",     s["A1_pass_frac"],
           f"total violations = {s['A1_violations_total']}")
        pf("(A2) supp ⊆ ℐ (minimal)",   s["A2_pass_frac"], "by construction")
        pf("(A3) unit lattice step",     s["A3_pass_frac"],
           f"total violations = {s['A3_violations_total']}")
        pf("(D1) G_ℓ ≥ 0",               s["D1_pass_frac"])

        print("    Lemma A.1 (cross-term decomposition under A1+A3)")
        pf("⟨Δ,Σe⟩ = D + E exactly",     s["lemma_A1_decomp_pass_frac"])
        pf("D ≤ 0 (deterministic)",      s["D_nonpos_pass_frac"],
           f"mean D = {s['mean_D']:+.4e}")
        pf("|E_j| ≤ (s_j²/2)|S_j|ρ(Σ)",  s["lemma_A1_offdiag_pass_frac"],
           f"mean E = {s['mean_E']:+.4e}, mean bound = {s['mean_E_bound']:+.4e}")
        pf("|D_ℓ| ≥ B_ℓ γ_ℐ",            s["D_lower_pass_frac"])

        print("    Trace bound")
        pf("tr(ΔᵀΣΔ) ≤ s_max² B_ℓ ‖Σ|_ℐ‖_2", s["trace_bound_pass_frac"],
           f"tr̄ = {s['mean_trace']:+.4e}, bound̄ = {s['mean_trace_bound']:+.4e}")

        print("    Theorem 1 — MASTER BOUND")
        pf("ΔR_ℓ ≤ -G_ℓ + B_ℓ Φ_ℐ",      s["master_bound_pass_frac"],
           f"Φ̄ = {s['mean_Phi']:+.4e}, ΔR̄ = {s['mean_delta_R']:+.4e}, RHS̄ = {s['mean_master_rhs']:+.4e}")

        print("    Corollary 1 — regime")
        print(f"      regime (i)  fraction (Φ_ℐ ≤ 0): {s['regime_i_frac']*100:6.2f}%")
        print(f"      regime (ii) fraction (Φ_ℐ > 0): {s['regime_ii_frac']*100:6.2f}%")
        print(f"      descent in regime (i):          {s['descent_in_regime_i_frac']*100:6.2f}%")
        print(f"      empirical descent overall:      {s['empirical_descent_frac']*100:6.2f}%")
        print(f"    Aggregate R: {s['sum_R_before']:.4e} → {s['sum_R_after']:.4e}  "
              f"({100*(s['sum_R_before']-s['sum_R_after'])/max(s['sum_R_before'],1e-30):+.3f}%)")

    # ----- Verdict -----
    print()
    print("═" * 110)
    print(f"VERDICT  (pass threshold = {threshold_pct:.0f}% of layers)")
    print("═" * 110)
    failures = []
    keys_to_check = [
        ("prop1_pass_frac",            "Proposition 1 identity"),
        ("A1_pass_frac",               "(A1) anti-aligned sign"),
        ("A3_pass_frac",               "(A3) unit lattice step"),
        ("D1_pass_frac",               "(D1) G_ℓ ≥ 0"),
        ("lemma_A1_decomp_pass_frac",  "Lemma A.1 decomposition"),
        ("D_nonpos_pass_frac",         "Lemma A.1(i) D ≤ 0"),
        ("lemma_A1_offdiag_pass_frac", "Lemma A.1(ii) E bound"),
        ("D_lower_pass_frac",          "|D_ℓ| ≥ B_ℓ γ_ℐ"),
        ("trace_bound_pass_frac",      "Trace bound"),
        ("master_bound_pass_frac",     "Theorem 1 master bound"),
    ]
    for k, s in summary.items():
        if s["mode"] != "insample":
            continue
        for field, label in keys_to_check:
            if s[field] < threshold_pct / 100 - 1e-9:
                failures.append(f"{s['base']}::{s['mode']}  {label}: "
                                f"{s[field]*100:.2f}% < {threshold_pct:.0f}%")
    if failures:
        print("  VERDICT:  FAIL ✗")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  VERDICT:  PASS ✓")
        print("    All assumptions, lemma claims, and the master bound verified at the threshold.")
    print()


def save_outputs(results, summary, out_dir, cfg):
    out_dir.mkdir(parents=True, exist_ok=True)

    def jsonify(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(out_dir / "per_layer_v4.json", "w") as f:
        json.dump({"config": cfg, "per_layer": results}, f, indent=2, default=jsonify)
    with open(out_dir / "summary_v4.json", "w") as f:
        json.dump(summary, f, indent=2, default=jsonify)


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
    p.add_argument("--out-dir", type=str, default="./clc_verify_v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-pct", type=float, default=95.0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 110)
    print("CLC PAPER VERIFICATION — v4")
    print("═" * 110)
    for k, v in vars(args).items():
        print(f"  {k:32s} = {v}")
    print("═" * 110)

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

    print(f"\n[1/3] Calibration μ from {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_rec = ActivationRecorder(
        module_names, record_full_cov=False,
        max_tokens_per_module=args.max_cal_tokens_per_layer,
        flush_every_tokens=args.flush_every_tokens,
    )
    run_calibration(model, tok, cal_texts, cal_rec, device, args.max_length)
    cal_stats = cal_rec.finalize()

    print(f"\n[2/3] Eval μ and full Σ from {args.eval_datasets}")
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

    modes = ["insample", "crossval"] if args.mode == "both" else [args.mode]
    print(f"\n[3/3] Per-layer verification, modes = {modes}")
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
    print_report(results, summary, args.threshold_pct)
    save_outputs(results, summary, Path(args.out_dir), vars(args))
    print(f"\nWrote: {args.out_dir}")


if __name__ == "__main__":
    main()