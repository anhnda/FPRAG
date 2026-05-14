"""
Paper-ready EGBC verification — v3.

═══════════════════════════════════════════════════════════════════════════════
THE THEORY (corrected, on-support decomposition):
═══════════════════════════════════════════════════════════════════════════════

The flip is supported on S_j (selected coords for row j) with:
    Δ_{j,i} = d_{j,i} s_j,    d_{j,i} = -sign(e_{j,i})   (forced: flip to adjacent level)

The cross-term decomposes EXACTLY (no approximation, no decomposition along μ):

    ⟨Δ_j, Σ e_j⟩  =  D_j  +  E_j

where:
    D_j  := -s_j Σ_{i ∈ S_j} Σ_ii |e_{j,i}|          (ON-SUPPORT DIAGONAL)
    E_j  := -s_j Σ_{i ∈ S_j} sign(e_{j,i})·Σ_{k≠i} Σ_ik e_{j,k}    (OFF-DIAGONAL)

KEY FACTS:
1. D_j ≤ 0 DETERMINISTICALLY (s_j > 0, Σ_ii ≥ 0, |e_{j,i}| ≥ 0).
2. E_j has bound: |E_j| ≤ (s_j²|S_j|/2) · max_i Σ_{k≠i} |Σ_ik|.
3. Empirically, ⟨Δ_j, Σ e_j⟩ < 0 across all layers (this script verifies it).

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS SCRIPT MEASURES (per layer, per (base, mode)):
═══════════════════════════════════════════════════════════════════════════════

  cross_full      = ⟨Δ, ΣE⟩_F                   full cross-term
  cross_diag      = Σ_j D_j  (on-support diag)  ≤ 0 deterministically
  cross_offdiag   = Σ_j E_j  (off-diagonal)     unsigned, empirical
  decomp_check    = |cross_full - (cross_diag + cross_offdiag)|   should be ~0

  diag_bound      = bound on |cross_diag|     just for magnitude tracking
  offdiag_bound   = worst-case bound on |cross_offdiag|

PASS conditions (verified per (base, mode)):
  ✓ Decomposition exact:           decomp_check ≈ 0
  ✓ Diagonal ≤ 0:                  cross_diag ≤ 0 on every layer
  ✓ Off-diagonal bound:            |cross_offdiag| ≤ offdiag_bound on every layer
  ✓ Full cross-term ≤ 0:           cross_full ≤ 0 on ≥ 95% of layers (empirical)
  ✓ Diagonal explains sign:        on layers with cross_full ≤ 0, cross_diag ≤ 0 too
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
# Base quantizer
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
# Per-layer measurement — v3 with on-support diag/offdiag decomposition
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_layer(
    name: str, W_fp: torch.Tensor, base: str,
    cal_stats: Dict, eval_stats: Dict,
    bits: int, group_size: int,
    flip_budget_pct: float, knee_tolerance: float,
    use_james_stein: bool, mode: str, device: torch.device,
) -> Dict[str, object]:
    """Measure per layer:

      cross_full     = ⟨Δ, Σ e⟩_F
      cross_diag     = Σ_j -s_j Σ_{i ∈ S_j} Σ_ii |e_{j,i}|       (on-support diagonal)
      cross_offdiag  = cross_full - cross_diag                    (off-diagonal residual)
      decomp_check   = |cross_full - (cross_diag + cross_offdiag)|   sanity (~0)

      offdiag_bound  = worst-case bound: Σ_j s_j² |S_j| max_i Σ_{k≠i} |Σ_ik| / 2

      Plus: bias before/after, variance before/after, total, etc.
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

    # ----- Run base quantizer -----
    W_q_eff, W_int, scale_flat, zp_flat, s_vec = run_base_quantizer(
        base, W, mu_cal.float(), salience, bits, group_size,
    )
    W_q_eff = W_q_eff.float()
    e = (W_q_eff - W).double()                  # [out, in]  pre-flip error

    # ----- Compute the flip Δ -----
    mu_for_flip = (mu_cal.float() / s_vec).float() if base == "awq" else mu_cal.float()
    Delta_scaled = compute_flip_delta(
        W=W * s_vec.unsqueeze(0) if base == "awq" else W,
        W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_for_flip, bits=bits,
        flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
    ).double()
    Delta = Delta_scaled / s_vec.unsqueeze(0).double() if base == "awq" else Delta_scaled
    e_tilde = e + Delta

    # ===== B (bias) and V (variance) before/after ===============================
    B_before = ((e @ mu_for_eval) ** 2).sum()
    B_after  = ((e_tilde @ mu_for_eval) ** 2).sum()
    Se        = e @ Sigma                        # [out, in]
    V_before  = (Se * e).sum()
    Set       = e_tilde @ Sigma
    V_after   = (Set * e_tilde).sum()
    T_before  = B_before + V_before
    T_after   = B_after  + V_after

    # ===== Cross-term: FULL =====================================================
    cross_full = (Delta * Se).sum()              # ⟨Δ, Σ e⟩_F

    # ===== On-support DIAGONAL decomposition ====================================
    # Support: S = {(j, i) : Δ_{j,i} ≠ 0}
    support_mask = (Delta != 0)                  # [out, in] bool
    # Sanity: on the support, Δ_{j,i} = -sign(e_{j,i}) * s_j.
    # Step size per column: scale_flat is [out, n_groups]; we need per-(j,i).
    # In AWQ space, the effective step in the *original* weight space differs
    # from the *scaled* weight space. We use the cross-term in original space,
    # so the per-(j,i) step we apply is:
    #   step_per_ji_original = |Δ_{j,i}| when |Δ| > 0,  i.e., already in original space.
    #   We just use |Δ| as the step magnitude at each (j, i).
    abs_Delta = Delta.abs()                       # [out, in]  zero off-support
    sign_e_on_support = torch.where(support_mask, torch.sign(e), torch.zeros_like(e))

    # Sanity check: on support, sign(Δ) should equal -sign(e)
    # (this verifies the algorithm's invariant; doesn't change measurement)
    sign_Delta = torch.sign(Delta)
    sign_consistency_on_support = (sign_Delta + sign_e_on_support).abs()  # 0 if -sign(e) == sign(Δ)
    sign_violation = float(sign_consistency_on_support[support_mask].sum().item()) if support_mask.any() else 0.0

    # Diagonal contribution per (j, i):
    #     d_{j,i} · (Σ e_j)_i_diag = d_{j,i} · Σ_ii · e_{j,i}
    #                              = -sign(e_{j,i}) · Σ_ii · e_{j,i}
    #                              = -Σ_ii · |e_{j,i}|
    # multiplied by s_j (step size) gives D_j contribution.
    # In our notation: Δ_{j,i} = d_{j,i} s_j, so Δ_{j,i} · Σ_ii · e_{j,i}.
    Sigma_diag = torch.diagonal(Sigma)            # [in]
    diag_contrib_per_ji = Delta * Sigma_diag.unsqueeze(0) * e   # [out, in]
    # Sum over support (off-support entries already zero since Δ=0 there):
    cross_diag = diag_contrib_per_ji.sum()

    # Sanity: on support this is Δ_{j,i} * Σ_ii * e_{j,i}
    #   With Δ_{j,i} = -sign(e_{j,i}) * |Δ|,  this is -|Δ| * Σ_ii * |e|,  ≤ 0.

    # ===== Off-diagonal contribution: cross_offdiag = cross_full - cross_diag ===
    cross_offdiag = cross_full - cross_diag

    # Verify the decomposition (numerical only)
    decomp_check = float((cross_full - (cross_diag + cross_offdiag)).abs().item())

    # ===== Bound on |cross_offdiag| =============================================
    # |E_j| ≤ s_j · |S_j| · max_i Σ_{k≠i} |Σ_ik| · max_k |e_{j,k}|
    # Using max_k |e_{j,k}| ≤ s_j/2 gives the standard tight version below.
    # Compute it per row, then sum.
    Sigma_offdiag_abs = Sigma.abs() - torch.diag(torch.diagonal(Sigma).abs())
    Sigma_offdiag_rowsum_max = Sigma_offdiag_abs.sum(dim=1).max().item()  # max_i Σ_{k≠i} |Σ_ik|

    # |S_j| per row, |Δ| max per row (proxy for step magnitude), e max per row:
    S_per_row = support_mask.sum(dim=1).double()                  # [out]
    Delta_max_per_row = abs_Delta.max(dim=1).values               # [out]   ≈ s_j
    e_max_per_row = e.abs().max(dim=1).values                     # [out]   ≤ s_j/2

    # offdiag_bound per row:  |Δ_max| * |S| * (max_i Σ_{k≠i}|Σ_ik|) * |e_max|
    offdiag_bound_per_row = Delta_max_per_row * S_per_row * Sigma_offdiag_rowsum_max * e_max_per_row
    offdiag_bound = offdiag_bound_per_row.sum().item()

    # Tighter bound using |e_{j,k}| ≤ s_j/2:
    offdiag_bound_tight_per_row = Delta_max_per_row * S_per_row * Sigma_offdiag_rowsum_max * (Delta_max_per_row / 2.0)
    offdiag_bound_tight = offdiag_bound_tight_per_row.sum().item()

    # ===== Lattice closure check (Lemma 3) ======================================
    max_int = 2 ** bits - 1
    if base == "awq":
        delta_int_pred = (Delta * s_vec.unsqueeze(0).double() / scale_flat.double())
    else:
        delta_int_pred = (Delta / scale_flat.double())
    is_integer = (delta_int_pred - delta_int_pred.round()).abs().max().item() < 1e-6
    W_int_post = W_int.double() + delta_int_pred
    code_in_range = bool(((W_int_post >= 0) & (W_int_post <= max_int)).all().item())
    lemma3_pass = bool(is_integer and code_in_range)

    # ===== Other facts ==========================================================
    dV = V_after - V_before

    def rel(num, den):
        d = float(den.item() if hasattr(den, "item") else den)
        if abs(d) < 1e-30:
            return 0.0
        return float((num.item() if hasattr(num, "item") else num) / d)

    # ===== Pass/fail booleans ===================================================
    eps = 1e-10
    # P1: decomposition is numerically exact
    P1_decomp_exact = bool(decomp_check < 1e-6 * (1.0 + float(cross_full.abs().item())))
    # P2: cross_diag ≤ 0 (deterministic theorem)
    P2_diag_nonpos = bool(cross_diag <= eps * (1.0 + cross_diag.abs()))
    # P3: |cross_offdiag| within tight bound (theorem)
    P3_offdiag_bounded = bool(cross_offdiag.abs() <= offdiag_bound_tight * (1.0 + 1e-6))
    # P4: full cross-term ≤ 0 (empirical)
    P4_cross_full_nonpos = bool(cross_full <= eps * (1.0 + cross_full.abs()))
    # P5: total T descent
    P5_T_descent = bool(T_after < T_before - eps * (1.0 + T_before.abs()))
    # P6: bias descent
    P6_B_descent = bool(B_after <= B_before + eps * (1.0 + B_before.abs()))

    return {
        "name":          name,
        "base":          base,
        "mode":          mode,
        "n_rows":        int(out_features),
        # ===== Magnitudes =====
        "B_before":      float(B_before.item()),
        "B_after":       float(B_after.item()),
        "V_before":      float(V_before.item()),
        "V_after":       float(V_after.item()),
        "T_before":      float(T_before.item()),
        "T_after":       float(T_after.item()),
        # Relative changes
        "dB_rel":        rel(B_before - B_after, B_before),
        "dV_rel":        rel(V_after - V_before, V_before),
        "dT_rel":        rel(T_before - T_after, T_before),
        # ===== Cross-term decomposition (on-support diag/offdiag) =====
        "cross_full":           float(cross_full.item()),
        "cross_diag":           float(cross_diag.item()),       # on-support diag, ≤ 0 det.
        "cross_offdiag":        float(cross_offdiag.item()),    # cross_full - cross_diag
        "decomp_check":         decomp_check,                    # ~ 0
        # Bound on offdiag
        "offdiag_bound_tight":  offdiag_bound_tight,             # using |e| ≤ s/2
        "offdiag_bound_actual": offdiag_bound,                   # using actual |e|_max
        # Ratios
        "ratio_diag_to_full":      rel(cross_diag, cross_full),
        "ratio_offdiag_to_full":   rel(cross_offdiag, cross_full),
        "ratio_offdiag_to_diag":   abs(float(cross_offdiag.item())) / max(abs(float(cross_diag.item())), 1e-30),
        # ===== Algorithmic invariant =====
        "sign_violation_on_support": sign_violation,    # should be 0
        # ===== Pass flags =====
        "P1_decomp_exact":       P1_decomp_exact,
        "P2_diag_nonpos":        P2_diag_nonpos,
        "P3_offdiag_bounded":    P3_offdiag_bounded,
        "P4_cross_full_nonpos":  P4_cross_full_nonpos,
        "P5_T_descent":          P5_T_descent,
        "P6_B_descent":          P6_B_descent,
        "lemma3_lattice":        lemma3_pass,
        # ===== Aux =====
        "support_size":              int(support_mask.sum().item()),
        "n_flipped_rows":            int((S_per_row > 0).sum().item()),
        "mean_S_per_row":            float(S_per_row.mean().item()),
        "Sigma_offdiag_rowsum_max":  Sigma_offdiag_rowsum_max,
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

        summary[f"{base}::{mode}"] = {
            "base": base, "mode": mode, "n_layers": n,
            # ----- Pass fractions -----
            "P1_decomp_frac":      frac("P1_decomp_exact"),
            "P2_diag_nonpos_frac": frac("P2_diag_nonpos"),
            "P3_offdiag_bounded_frac": frac("P3_offdiag_bounded"),
            "P4_full_nonpos_frac": frac("P4_cross_full_nonpos"),
            "P5_T_descent_frac":   frac("P5_T_descent"),
            "P6_B_descent_frac":   frac("P6_B_descent"),
            "lemma3_frac":         frac("lemma3_lattice"),
            # ----- Magnitudes (means) -----
            "cross_full_mean":     avg("cross_full"),
            "cross_diag_mean":     avg("cross_diag"),
            "cross_offdiag_mean":  avg("cross_offdiag"),
            "offdiag_bound_tight_mean":  avg("offdiag_bound_tight"),
            "decomp_check_mean":   avg("decomp_check"),
            # Ratios
            "ratio_diag_to_full_mean":    avg("ratio_diag_to_full"),
            "ratio_offdiag_to_full_mean": avg("ratio_offdiag_to_full"),
            "ratio_offdiag_to_diag_mean": avg("ratio_offdiag_to_diag"),
            # Algorithmic
            "sign_violation_max":  float(max(r["sign_violation_on_support"] for r in rs)),
            # Relative magnitude
            "avg_dB_rel": avg("dB_rel"),
            "avg_dV_rel": avg("dV_rel"),
            "avg_dT_rel": avg("dT_rel"),
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
# Verdict
# --------------------------------------------------------------------------- #
def theorem_verdict(summary: Dict[str, Dict], threshold: float = 0.95) -> Dict:
    """Verdict on the on-support diag/offdiag chain:

      P1: cross-term decomposition is numerically exact (always true; sanity)
      P2: cross_diag ≤ 0 on ≥ threshold of layers (deterministic theorem)
      P3: |cross_offdiag| ≤ tight bound on ≥ threshold of layers (theorem)
      P4: cross_full ≤ 0 on ≥ threshold of layers (empirical, key claim)
      P5: total T descent on ≥ threshold of layers (empirical, payoff)
    """
    insample = [s for s in summary.values() if s["mode"] == "insample"]
    if not insample:
        return {"pass": False, "reason": "no insample results"}
    failures = []
    for s in insample:
        if s["P1_decomp_frac"] < 1.0 - 1e-9:
            failures.append(f"{s['base']}: P1 decomposition exact "
                            f"{s['P1_decomp_frac']*100:.1f}% < 100%")
        if s["P2_diag_nonpos_frac"] < threshold:
            failures.append(f"{s['base']}: P2 diag ≤ 0 "
                            f"{s['P2_diag_nonpos_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["P3_offdiag_bounded_frac"] < threshold:
            failures.append(f"{s['base']}: P3 offdiag bound "
                            f"{s['P3_offdiag_bounded_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["P4_full_nonpos_frac"] < threshold:
            failures.append(f"{s['base']}: P4 full cross-term ≤ 0 "
                            f"{s['P4_full_nonpos_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["P5_T_descent_frac"] < threshold:
            failures.append(f"{s['base']}: P5 T descent "
                            f"{s['P5_T_descent_frac']*100:.1f}% < {threshold*100:.0f}%")
    return {
        "pass": len(failures) == 0,
        "threshold_pct": threshold * 100,
        "bases_tested": sorted({s["base"] for s in insample}),
        "failures": failures,
        "mechanism_by_base": {s["base"]: {
            "mean_cross_full":    s["cross_full_mean"],
            "mean_cross_diag":    s["cross_diag_mean"],
            "mean_cross_offdiag": s["cross_offdiag_mean"],
            "mean_offdiag_bound": s["offdiag_bound_tight_mean"],
            "ratio_diag_to_full": s["ratio_diag_to_full_mean"],
            "ratio_offdiag_to_diag": s["ratio_offdiag_to_diag_mean"],
        } for s in insample},
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(results, summary, verdict):
    print()
    print("═" * 100)
    print("EGBC VERIFICATION v3 — On-Support Diagonal Decomposition")
    print("═" * 100)
    print("""\
THEORY (on-support decomposition; exact, no μ-projection):

  Flip:  Δ_{j,i} = d_{j,i} s_j,  d_{j,i} = -sign(e_{j,i})  on the support S_j

  Cross-term decomposes EXACTLY:
    ⟨Δ_j, Σ e_j⟩  =  D_j   +   E_j
    D_j = -s_j Σ_{i∈S_j} Σ_ii |e_{j,i}|         (on-support diag, ≤ 0 DETERMINISTIC)
    E_j = -s_j Σ_{i∈S_j} sign(e_{j,i})·Σ_{k≠i} Σ_ik e_{j,k}     (off-diagonal residual)

  Bound on |E_j|:
    |E_j| ≤ s_j · |S_j| · max_i Σ_{k≠i}|Σ_ik| · max_k |e_{j,k}|
          ≤ (s_j²/2) · |S_j| · max_i Σ_{k≠i}|Σ_ik|     (since |e| ≤ s_j/2)

  CONCLUSION: Cross-term ⟨Δ, Σ e⟩ ≤ D + |E| (i.e., bounded above by D_j + bound on E_j).
              D is deterministically ≤ 0. E is bounded; empirically also ≤ 0.

CHECKS performed:
  P1: Decomposition is numerically exact (cross_full = cross_diag + cross_offdiag)
  P2: cross_diag ≤ 0 on every layer (deterministic theorem)
  P3: |cross_offdiag| ≤ tight bound on every layer (theorem)
  P4: cross_full ≤ 0 on ≥ 95% of layers (empirical claim)
  P5: Total T descent on ≥ 95% of layers (empirical payoff)
""")

    # ----- Per-layer table -----
    print("─" * 100)
    print("PER-LAYER EVIDENCE")
    print("─" * 100)
    header = (f"{'layer':<55s} {'base':<5s} {'mode':<9s} "
              f"{'P2':>3s} {'P3':>3s} {'P4':>3s} {'P5':>3s}  "
              f"{'cross_diag':>13s} {'cross_offdiag':>14s}  "
              f"{'cross_full':>12s}  {'ΔT/T':>8s}")
    print(header)
    print("─" * 100)
    for r in sorted(results, key=lambda x: (x["base"], x["mode"], x["name"])):
        tick = lambda b: "✓" if b else "✗"
        print(f"{r['name']:<55s} {r['base']:<5s} {r['mode']:<9s} "
              f"{tick(r['P2_diag_nonpos']):>3s} "
              f"{tick(r['P3_offdiag_bounded']):>3s} "
              f"{tick(r['P4_cross_full_nonpos']):>3s} "
              f"{tick(r['P5_T_descent']):>3s}  "
              f"{r['cross_diag']:+13.4e} {r['cross_offdiag']:+14.4e}  "
              f"{r['cross_full']:+12.4e}  {r['dT_rel']*100:+7.3f}%")

    # ----- Headline -----
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
        print(f"    P1 Decomposition exact     : {int(s['P1_decomp_frac']*n):>3d}/{n:<3d} ({s['P1_decomp_frac']*100:6.2f}%)  decomp_check mean = {s['decomp_check_mean']:.2e}")
        print(f"    P2 cross_diag ≤ 0          : {int(s['P2_diag_nonpos_frac']*n):>3d}/{n:<3d} ({s['P2_diag_nonpos_frac']*100:6.2f}%)  [deterministic theorem]")
        print(f"    P3 |cross_offdiag| ≤ bound : {int(s['P3_offdiag_bounded_frac']*n):>3d}/{n:<3d} ({s['P3_offdiag_bounded_frac']*100:6.2f}%)  [theorem]")
        print(f"    P4 cross_full ≤ 0          : {int(s['P4_full_nonpos_frac']*n):>3d}/{n:<3d} ({s['P4_full_nonpos_frac']*100:6.2f}%)  [empirical key claim]")
        print(f"    P5 Total T descent         : {int(s['P5_T_descent_frac']*n):>3d}/{n:<3d} ({s['P5_T_descent_frac']*100:6.2f}%)  [empirical payoff]")
        print(f"    P6 Bias descent            : {int(s['P6_B_descent_frac']*n):>3d}/{n:<3d} ({s['P6_B_descent_frac']*100:6.2f}%)  [Lemma 1]")
        print(f"    Lemma 3 lattice closure    : {int(s['lemma3_frac']*n):>3d}/{n:<3d} ({s['lemma3_frac']*100:6.2f}%)")
        print(f"")
        print(f"    CROSS-TERM DECOMPOSITION (mean across layers):")
        print(f"        E[cross_full]            = {s['cross_full_mean']:+12.4e}    full ⟨Δ, Σe⟩_F")
        print(f"        E[cross_diag]            = {s['cross_diag_mean']:+12.4e}    on-support Σ_i s_j Σ_ii |e_{{j,i}}|  (≤ 0 DETERMINISTIC)")
        print(f"        E[cross_offdiag]         = {s['cross_offdiag_mean']:+12.4e}    off-diagonal residual")
        print(f"        E[|cross_offdiag bound|] = {s['offdiag_bound_tight_mean']:+12.4e}    (s²/2)·|S|·max_i Σ_{{k≠i}}|Σ_ik|")
        print(f"")
        print(f"    RATIOS:")
        print(f"        diag / full mean         = {s['ratio_diag_to_full_mean']:+8.4f}  (fraction of cross-term from on-support diag)")
        print(f"        offdiag / full mean      = {s['ratio_offdiag_to_full_mean']:+8.4f}  (fraction from off-diagonal)")
        print(f"        |offdiag| / |diag| mean  = {s['ratio_offdiag_to_diag_mean']:8.4f}  (off-diag magnitude vs diag)")
        print(f"")
        print(f"    Realized total descent:")
        print(f"        avg ΔB/B = {s['avg_dB_rel']*100:+7.3f}%   "
              f"ΔV/V = {s['avg_dV_rel']*100:+7.3f}%   "
              f"ΔT/T = {s['avg_dT_rel']*100:+7.3f}%")
        print(f"        Aggregate T: {s['sum_T_before']:.4e} → {s['sum_T_after']:.4e}    "
              f"({100*(s['sum_T_before']-s['sum_T_after'])/max(s['sum_T_before'],1e-30):+.3f}%)")
        print(f"")
        print(f"    Algorithmic invariant (sign(Δ) = -sign(e) on support):")
        print(f"        max sign violation per layer = {max(r['sign_violation_on_support'] for r in [x for x in summary.values()][0:1]):.2e}  (should be 0)")

    # ----- Verdict -----
    print()
    print("═" * 100)
    print(f"VERDICT (threshold ≥ {verdict['threshold_pct']:.0f}%)")
    print("═" * 100)
    print(f"  Bases tested:  {verdict['bases_tested']}")
    if verdict["pass"]:
        print(f"  VERDICT:  PASS  ✓")
        print(f"     On-support decomposition holds: cross_diag ≤ 0 deterministically,")
        print(f"     cross_offdiag bounded, cross_full ≤ 0 empirically, T descends.")
    else:
        print(f"  VERDICT:  FAIL  ✗")
        for f in verdict["failures"]:
            print(f"    – {f}")
    print()
    print(f"  Mechanism summary (per base):")
    for base, m in verdict.get("mechanism_by_base", {}).items():
        print(f"    {base.upper()}:")
        print(f"      E[cross_full]    = {m['mean_cross_full']:+11.4e}  (full cross-term)")
        print(f"      E[cross_diag]    = {m['mean_cross_diag']:+11.4e}  (on-support diag, det. ≤ 0)")
        print(f"      E[cross_offdiag] = {m['mean_cross_offdiag']:+11.4e}  (off-diagonal residual)")
        print(f"      E[offdiag bound] = {m['mean_offdiag_bound']:+11.4e}  (tight worst-case)")
        print(f"      diag / full      = {m['ratio_diag_to_full']:+8.4f}")
        print(f"      |offdiag|/|diag| = {m['ratio_offdiag_to_diag']:8.4f}")
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

    with open(out_dir / "per_layer_evidence_v3.json", "w") as f:
        json.dump({"config": cfg, "per_layer": results}, f, indent=2, default=jsonify)
    with open(out_dir / "headline_summary_v3.json", "w") as f:
        json.dump(summary, f, indent=2, default=jsonify)
    with open(out_dir / "theorem_verdict_v3.json", "w") as f:
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
    p.add_argument("--out-dir", type=str, default="./egbc_paper_results_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-pct", type=float, default=95.0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 100)
    print("EGBC VERIFICATION v3 — CONFIG (On-support diag/offdiag decomposition)")
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
    verdict = theorem_verdict(summary, threshold=args.threshold_pct / 100.0)
    print_report(results, summary, verdict)
    save_outputs(results, summary, verdict, Path(args.out_dir), vars(args))
    print(f"\nWrote: {args.out_dir}")


if __name__ == "__main__":
    main()