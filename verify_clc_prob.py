"""
Verification of the probabilistic MSE descent theorem for CLC.

═══════════════════════════════════════════════════════════════════════════════
THE THEOREM (probabilistic):
═══════════════════════════════════════════════════════════════════════════════

  Assumption (A1): Residuals e_{j,i} are sub-Gaussian with parameter σ ≤ s_j/2.
  Assumption (A2): The selection support S_j is independent of e_{j,i}.

  Then with probability ≥ 1 − δ, where δ := exp(-gap²/(2·λ²·s⁴·B)):

      ΔR_ℓ ≤ −G_ℓ + s²·B·λ + 2τ_δ

  with τ_δ = s²·√(B·log(1/δ))·λ, and ΔR_ℓ < 0 (MSE descent) whenever:

      gap := G_ℓ − s²·B·λ > 0    (deterministic descent condition)

  Pr[ ΔR_ℓ ≥ 0 ] ≤ exp( −gap² / (2·λ²·s⁴·B) )

═══════════════════════════════════════════════════════════════════════════════
PERFORMANCE NOTES
═══════════════════════════════════════════════════════════════════════════════
This version is fully vectorized:
  - per-row G_j, ΔR_j, s_j, B_j computed in bulk over [out, in] tensors
  - per-row λ_j upper-bounded by Gershgorin on Σ_{S_j S_j}, computed as one
    [out, d] @ [d, d] GEMM (no Python loop, no per-row eigvalsh, no per-row
    index_select)
  - exact eigvalsh kept only for the LAYER-aggregate λ (one call per layer)
  - everything reuses Se = e @ Σ and Set = e_tilde @ Σ that were already
    computed for the layer-level numbers
"""

import argparse
import gc
import json
import math
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
# Per-layer measurement  (FULLY VECTORIZED)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_layer(
    name: str, W_fp: torch.Tensor, base: str,
    cal_stats: Dict, eval_stats: Dict,
    bits: int, group_size: int,
    flip_budget_pct: float, knee_tolerance: float,
    use_james_stein: bool, mode: str, device: torch.device,
    delta_target: float = 0.05,
) -> Dict[str, object]:
    """Vectorized per-layer measurement.

    Layer aggregate (single instance):
        G_ℓ        := B_before − B_after
        B          := total support size across all rows
        s_max      := max step magnitude (layer-wide)
        λ          := ‖Σ‖_op   (eigvalsh once)
        gap_ℓ      := G_ℓ − s_max²·B·λ
        fail_prob_bound_ℓ := exp(−gap_ℓ² / (2·λ²·s_max⁴·B))   if gap_ℓ > 0

    Per-row refinement (vectorized over rows):
        For each row j:
            G_j   = (μ·e_j)² − (μ·e_tilde_j)²
            ΔR_j  = T_after_j − T_before_j   (uses reused Se, Set)
            s_j   = max |Δ_{j,·}| on support
            B_j   = |S_j|
            λ_j   ≤ Gershgorin upper bound on ‖Σ_{S_j S_j}‖_op
                   = max_{i ∈ S_j}  Σ_{k ∈ S_j} |Σ_{ik}|
                   computed in bulk as M @ |Σ| with M = support_mask.float()
        gap_j  := G_j − s_j²·B_j·λ_j
        pred_j := gap_j > 0
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

    # ----- Base quantizer -----
    W_q_eff, W_int, scale_flat, zp_flat, s_vec = run_base_quantizer(
        base, W, mu_cal.float(), salience, bits, group_size,
    )
    W_q_eff = W_q_eff.float()
    e = (W_q_eff - W).double()                  # [out, in] pre-flip error

    # ----- Flip Δ -----
    mu_for_flip = (mu_cal.float() / s_vec).float() if base == "awq" else mu_cal.float()
    Delta_scaled = compute_flip_delta(
        W=W * s_vec.unsqueeze(0) if base == "awq" else W,
        W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_for_flip, bits=bits,
        flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
    ).double()
    Delta = Delta_scaled / s_vec.unsqueeze(0).double() if base == "awq" else Delta_scaled
    e_tilde = e + Delta

    # ===== Layer-aggregate B (bias), V (variance), Total =====
    # Reuse these matmuls for the per-row pass.
    Se     = e @ Sigma                                  # [out, in]
    Set    = e_tilde @ Sigma                            # [out, in]
    mu_e   = e @ mu_for_eval                            # [out]
    mu_et  = e_tilde @ mu_for_eval                      # [out]

    B_before = (mu_e ** 2).sum()
    B_after  = (mu_et ** 2).sum()
    V_before = (Se * e).sum()
    V_after  = (Set * e_tilde).sum()
    T_before = B_before + V_before
    T_after  = B_after  + V_after
    G_layer  = float((B_before - B_after).item())
    dR_layer = float((T_after - T_before).item())

    # ===== Layer-level theorem inputs =====
    support_mask = (Delta != 0)                         # [out, in]  bool
    abs_Delta    = Delta.abs()
    B_support_layer = int(support_mask.sum().item())

    if B_support_layer > 0:
        s_max_layer = float(abs_Delta[support_mask].max().item())
    else:
        s_max_layer = 0.0

    # λ (layer) = ‖Σ‖_op via eigvalsh (one call, not per row)
    try:
        Sigma_eigs = torch.linalg.eigvalsh(Sigma)
        lam_layer = float(Sigma_eigs.max().item())
    except Exception:
        lam_layer = float(torch.linalg.matrix_norm(Sigma, ord=2).item())

    sBL = (s_max_layer ** 2) * B_support_layer * lam_layer
    gap = G_layer - sBL
    pred_descent = bool(gap > 0)

    if gap > 0 and lam_layer > 0 and s_max_layer > 0 and B_support_layer > 0:
        denom = 2.0 * (lam_layer ** 2) * (s_max_layer ** 4) * B_support_layer
        exponent = -(gap ** 2) / max(denom, 1e-300)
        fail_prob_bound = math.exp(max(exponent, -700.0))
    else:
        fail_prob_bound = 1.0

    actual_descent = bool(dR_layer < 0)
    slack_ratio = gap / sBL if sBL > 1e-30 else float("inf")

    # ========================================================================
    # PER-ROW REFINED — fully vectorized
    # ========================================================================
    # All quantities below are [out]-shape tensors. No Python row loop.

    # Per-row B (count) and s (max step on support)
    B_r = support_mask.sum(dim=1)                                       # [out] int64
    # Mask abs_Delta to support and reduce; rows with no support get 0
    abs_D_masked = torch.where(support_mask, abs_Delta,
                               torch.zeros((), dtype=abs_Delta.dtype, device=device))
    s_r = abs_D_masked.amax(dim=1)                                      # [out] double

    # Per-row G_j and ΔR_j   (reuse mu_e, mu_et, Se, Set)
    eSe_row    = (Se * e).sum(dim=1)                                    # [out]
    etSet_row  = (Set * e_tilde).sum(dim=1)                             # [out]
    G_r        = mu_e ** 2 - mu_et ** 2                                 # [out]
    T_before_r = mu_e ** 2 + eSe_row
    T_after_r  = mu_et ** 2 + etSet_row
    dR_r       = T_after_r - T_before_r                                 # [out]

    # Per-row λ_j upper bound via Gershgorin restricted to S_j:
    #   ‖Σ_{S_j S_j}‖_op  ≤  max_{i ∈ S_j}  Σ_{k ∈ S_j} |Σ_{ik}|
    # Implementation:
    #   M = support_mask.double()                          [out, d]
    #   row_sums_S[j, i] = Σ_k M[j,k] · |Σ_{ik}|
    #                   = sum_{k ∈ S_j} |Σ_{ik}|
    #   λ_j ≤ max_{i ∈ S_j} row_sums_S[j, i]
    # This is one [out, d] @ [d, d] GEMM, then a masked row-max.
    abs_Sigma = Sigma.abs()                                             # [d, d] double
    M = support_mask.to(Sigma.dtype)                                    # [out, d] double
    row_sums_S = M @ abs_Sigma                                          # [out, d]
    # Mask rows-not-in-S_j to -inf so amax picks i ∈ S_j only
    NEG_INF = torch.full((), float("-inf"), dtype=row_sums_S.dtype, device=device)
    row_sums_S = torch.where(support_mask, row_sums_S, NEG_INF)
    lam_r = row_sums_S.amax(dim=1)                                      # [out]
    # rows with B_r == 0 get -inf; replace with 0
    lam_r = torch.where(B_r > 0, lam_r, torch.zeros_like(lam_r))
    # numerical safety
    lam_r = torch.nan_to_num(lam_r, nan=0.0, posinf=0.0, neginf=0.0)

    # gap_j, predictions, failure prob
    B_r_d  = B_r.to(s_r.dtype)
    sBL_r  = (s_r ** 2) * B_r_d * lam_r                                 # [out]
    gap_r  = G_r - sBL_r                                                # [out]
    pred_r   = gap_r > 0
    actual_r = dR_r < 0
    flipped_r = B_r > 0

    # Failure prob bound per row (only meaningful where pred_r holds)
    denom_r = 2.0 * (lam_r ** 2) * (s_r ** 4) * B_r_d
    # Avoid 0/0; clamp tiny denom
    denom_safe = torch.where(denom_r > 1e-300, denom_r,
                             torch.full_like(denom_r, 1e-300))
    exponent_r = -(gap_r ** 2) / denom_safe
    exponent_r = torch.clamp(exponent_r, min=-700.0, max=0.0)
    fp_r_raw = torch.exp(exponent_r)
    # Where pred_r is False or λ/s/B == 0, bound is trivially 1.0
    valid_pred = pred_r & (lam_r > 0) & (s_r > 0) & (B_r > 0)
    fp_r = torch.where(valid_pred, fp_r_raw, torch.ones_like(fp_r_raw))

    # Slack ratio per row
    slack_r = torch.where(sBL_r > 1e-30, gap_r / sBL_r,
                          torch.full_like(gap_r, float("inf")))

    # Aggregate over rows that had any flip
    n_rows_flipped       = int(flipped_r.sum().item())
    rows_pred_descent    = int((pred_r & flipped_r).sum().item())
    rows_actual_descent  = int((actual_r & flipped_r).sum().item())
    rows_pred_and_actual = int((pred_r & actual_r & flipped_r).sum().item())
    rows_pred_not_actual = int((pred_r & ~actual_r & flipped_r).sum().item())
    rows_notpred_actual  = int((~pred_r & actual_r & flipped_r).sum().item())

    if n_rows_flipped > 0:
        mean_fp_bound = float(fp_r[flipped_r].mean().item())
        slack_flipped = slack_r[flipped_r]
        finite = torch.isfinite(slack_flipped)
        mean_slack = float(slack_flipped[finite].mean().item()) if finite.any() else 0.0
    else:
        mean_fp_bound = 1.0
        mean_slack = 0.0

    # Agreement rate over flipped rows:
    #   correct = (pred ∧ actual) ∨ (¬pred ∧ ¬actual)
    correct_r = ((pred_r & actual_r) | (~pred_r & ~actual_r)) & flipped_r
    agreement_rate = float(correct_r.sum().item()) / max(n_rows_flipped, 1)

    return {
        "name":            name,
        "base":            base,
        "mode":            mode,
        "n_rows":          int(out_features),
        # ===== Layer-aggregate theorem inputs =====
        "G_layer":         G_layer,
        "B_support":       B_support_layer,
        "s_max":           s_max_layer,
        "lambda_layer":    lam_layer,
        "sBL":             sBL,
        "gap":             gap,
        "slack_ratio":     slack_ratio,
        # ===== Layer-aggregate theorem outputs =====
        "fail_prob_bound": fail_prob_bound,
        "pred_descent":    pred_descent,
        "actual_descent":  actual_descent,
        "agree_layer":     bool(pred_descent == actual_descent or (pred_descent and actual_descent)),
        # ===== Per-row refined =====
        "n_rows_flipped":          n_rows_flipped,
        "rows_pred_descent":       rows_pred_descent,
        "rows_actual_descent":     rows_actual_descent,
        "rows_pred_and_actual":    rows_pred_and_actual,
        "rows_pred_not_actual":    rows_pred_not_actual,
        "rows_notpred_actual":     rows_notpred_actual,
        "agreement_rate_per_row":  agreement_rate,
        "mean_fail_prob_per_row":  mean_fp_bound,
        "mean_slack_per_row":      mean_slack,
        # ===== Magnitudes =====
        "B_before":      float(B_before.item()),
        "B_after":       float(B_after.item()),
        "V_before":      float(V_before.item()),
        "V_after":       float(V_after.item()),
        "T_before":      float(T_before.item()),
        "T_after":       float(T_after.item()),
        "dT_rel":        dR_layer / max(float(T_before.item()), 1e-30),
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
        n_pred = sum(int(r["pred_descent"]) for r in rs)
        n_actual = sum(int(r["actual_descent"]) for r in rs)
        n_pred_and_actual = sum(int(r["pred_descent"] and r["actual_descent"]) for r in rs)
        n_pred_not_actual = sum(int(r["pred_descent"] and not r["actual_descent"]) for r in rs)
        n_notpred_actual = sum(int((not r["pred_descent"]) and r["actual_descent"]) for r in rs)
        n_agree = sum(int(r["pred_descent"] == r["actual_descent"]) for r in rs)

        total_rows_flipped = sum(r["n_rows_flipped"] for r in rs)
        total_rows_pred = sum(r["rows_pred_descent"] for r in rs)
        total_rows_actual = sum(r["rows_actual_descent"] for r in rs)
        total_rows_pred_and_actual = sum(r["rows_pred_and_actual"] for r in rs)
        total_rows_pred_not_actual = sum(r["rows_pred_not_actual"] for r in rs)
        total_rows_notpred_actual = sum(r["rows_notpred_actual"] for r in rs)

        avg = lambda k: float(np.mean([r[k] for r in rs]))
        median = lambda k: float(np.median([r[k] for r in rs]))

        summary[f"{base}::{mode}"] = {
            "base": base, "mode": mode, "n_layers": n,
            "layer_pred_descent_frac":   n_pred / max(n, 1),
            "layer_actual_descent_frac": n_actual / max(n, 1),
            "layer_agreement_frac":      n_agree / max(n, 1),
            "layer_pred_implies_actual": (n_pred_and_actual / n_pred) if n_pred > 0 else None,
            "layer_n_pred":              n_pred,
            "layer_n_actual":            n_actual,
            "layer_n_pred_and_actual":   n_pred_and_actual,
            "layer_n_pred_not_actual":   n_pred_not_actual,
            "layer_n_notpred_actual":    n_notpred_actual,
            "row_total_flipped":         total_rows_flipped,
            "row_pred_descent_frac":     total_rows_pred / max(total_rows_flipped, 1),
            "row_actual_descent_frac":   total_rows_actual / max(total_rows_flipped, 1),
            "row_pred_implies_actual":   (total_rows_pred_and_actual / total_rows_pred) if total_rows_pred > 0 else None,
            "row_pred_not_actual":       total_rows_pred_not_actual,
            "row_notpred_actual":        total_rows_notpred_actual,
            "mean_G":                avg("G_layer"),
            "mean_sBL":              avg("sBL"),
            "mean_gap":              avg("gap"),
            "median_slack":          median("slack_ratio"),
            "mean_fail_prob_bound":  avg("fail_prob_bound"),
            "mean_lambda":           avg("lambda_layer"),
            "mean_s_max":            avg("s_max"),
            "mean_B_support":        avg("B_support"),
            "mean_dT_rel":           avg("dT_rel"),
        }
    return summary


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def verdict(summary: Dict[str, Dict], agreement_threshold: float = 0.95) -> Dict:
    insample = [s for s in summary.values() if s["mode"] == "insample"]
    if not insample:
        return {"pass": False, "reason": "no insample results"}

    failures = []
    details = {}
    for s in insample:
        key = s["base"]
        details[key] = {
            "row_pred_implies_actual": s["row_pred_implies_actual"],
            "row_pred_descent_frac":   s["row_pred_descent_frac"],
            "row_actual_descent_frac": s["row_actual_descent_frac"],
            "empirical_disagreement":  (s["row_pred_not_actual"] / max(s["row_total_flipped"], 1)),
            "predicted_fail_bound":    s["mean_fail_prob_bound"],
        }
        if s["row_pred_implies_actual"] is not None and s["row_pred_implies_actual"] < agreement_threshold:
            failures.append(
                f"{key}: row_pred_implies_actual = {s['row_pred_implies_actual']*100:.1f}% "
                f"< {agreement_threshold*100:.0f}%"
            )

    return {
        "pass": len(failures) == 0,
        "agreement_threshold_pct": agreement_threshold * 100,
        "bases_tested": sorted({s["base"] for s in insample}),
        "failures": failures,
        "per_base": details,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(results, summary, verd):
    print()
    print("═" * 100)
    print("CLC PROBABILISTIC MSE-DESCENT THEOREM — VERIFICATION")
    print("═" * 100)
    print("""\
THEOREM. Under (A1) sub-Gaussian residuals (σ ≤ s/2) and (A2) selection
independent of e, the post-CLC layer MSE satisfies

    Pr[ ΔR_ℓ ≥ 0 ] ≤ exp( −(G_ℓ − s²·B·λ)² / (2·λ²·s⁴·B) )

whenever G_ℓ > s²·B·λ.

VERIFICATION (per layer and per row):
  1. Compute G, B, λ, s; form gap = G − s²·B·λ.
     (Per-row λ_j is the Gershgorin upper bound on ‖Σ_{S_j S_j}‖_op,
      which keeps the sufficient condition rigorous.)
  2. PRED_descent   := gap > 0.
  3. ACTUAL_descent := ΔR < 0.
  4. Theorem prediction: PRED ⇒ ACTUAL on ≥ 95% of instances.
""")

    print("─" * 100)
    print("PER-LAYER EVIDENCE")
    print("─" * 100)
    header = (f"{'layer':<55s} {'base':<5s} {'mode':<9s} "
              f"{'gap':>11s} {'s²Bλ':>11s} "
              f"{'pred':>5s} {'real':>5s}  "
              f"{'fp_bnd':>9s} {'slack':>8s}  {'ΔT/T':>8s}")
    print(header)
    print("─" * 100)
    for r in sorted(results, key=lambda x: (x["base"], x["mode"], x["name"])):
        tick = lambda b: "✓" if b else "✗"
        slack = r["slack_ratio"]
        slack_s = f"{slack:8.2e}" if math.isfinite(slack) else "    inf"
        print(f"{r['name']:<55s} {r['base']:<5s} {r['mode']:<9s} "
              f"{r['gap']:+11.4e} {r['sBL']:+11.4e} "
              f"{tick(r['pred_descent']):>5s} {tick(r['actual_descent']):>5s}  "
              f"{r['fail_prob_bound']:9.2e} {slack_s:>8s}  "
              f"{r['dT_rel']*100:+7.3f}%")

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
        print(f"    THEOREM INPUTS (layer averages):")
        print(f"        E[G_ℓ]            = {s['mean_G']:+12.4e}     bias descent")
        print(f"        E[s²·B·λ]         = {s['mean_sBL']:+12.4e}     descent threshold")
        print(f"        E[gap = G−s²Bλ]   = {s['mean_gap']:+12.4e}     surplus")
        print(f"        median slack      = {s['median_slack']:+12.4e}     (gap)/(s²Bλ)")
        print(f"        E[λ = ‖Σ‖_op]     = {s['mean_lambda']:+12.4e}")
        print(f"        E[s_max]          = {s['mean_s_max']:+12.4e}")
        print(f"        E[|S|]            = {s['mean_B_support']:+12.4e}")
        print(f"        E[fail prob bnd]  = {s['mean_fail_prob_bound']:.4e}")
        print(f"")
        print(f"    LAYER-LEVEL CHECK ({n} layers):")
        print(f"        PRED descent (gap > 0):     {s['layer_n_pred']}/{n} ({s['layer_pred_descent_frac']*100:6.2f}%)")
        print(f"        ACTUAL descent (ΔR < 0):    {s['layer_n_actual']}/{n} ({s['layer_actual_descent_frac']*100:6.2f}%)")
        print(f"        PRED ∧ ACTUAL:              {s['layer_n_pred_and_actual']}/{n}")
        if s['layer_pred_implies_actual'] is not None:
            print(f"        PRED ⇒ ACTUAL rate:         {s['layer_pred_implies_actual']*100:6.2f}%  ← theorem")
        print(f"        PRED ∧ ¬ACTUAL (violations): {s['layer_n_pred_not_actual']}")
        print(f"        ¬PRED ∧ ACTUAL (slack):     {s['layer_n_notpred_actual']}")
        print(f"")
        print(f"    ROW-LEVEL CHECK ({s['row_total_flipped']} flipped rows):")
        print(f"        PRED descent:               {s['row_pred_descent_frac']*100:6.2f}%")
        print(f"        ACTUAL descent:             {s['row_actual_descent_frac']*100:6.2f}%")
        if s['row_pred_implies_actual'] is not None:
            print(f"        PRED ⇒ ACTUAL rate:         {s['row_pred_implies_actual']*100:6.2f}%  ← theorem")
        print(f"        PRED ∧ ¬ACTUAL (violations): {s['row_pred_not_actual']}")
        print(f"        ¬PRED ∧ ACTUAL (slack):     {s['row_notpred_actual']}")
        print(f"")
        print(f"    Empirical ΔT/T (mean):           {s['mean_dT_rel']*100:+7.3f}%")

    print()
    print("═" * 100)
    print(f"VERDICT (PRED ⇒ ACTUAL rate ≥ {verd['agreement_threshold_pct']:.0f}%)")
    print("═" * 100)
    print(f"  Bases tested:  {verd['bases_tested']}")
    if verd["pass"]:
        print(f"  VERDICT:  PASS  ✓")
        print(f"     Sufficient condition gap = G − s²Bλ > 0 implies MSE descent")
        print(f"     in practice, on the regime where the theorem applies.")
    else:
        print(f"  VERDICT:  FAIL  ✗")
        for f in verd["failures"]:
            print(f"    – {f}")
    print()
    print(f"  Per-base details (insample):")
    for base, d in verd.get("per_base", {}).items():
        print(f"    {base.upper()}:")
        if d["row_pred_implies_actual"] is not None:
            print(f"      Row PRED ⇒ ACTUAL rate   = {d['row_pred_implies_actual']*100:6.2f}%")
        print(f"      Row PRED descent fraction = {d['row_pred_descent_frac']*100:6.2f}%")
        print(f"      Row ACTUAL descent frac.  = {d['row_actual_descent_frac']*100:6.2f}%")
        print(f"      Empirical disagreement    = {d['empirical_disagreement']*100:6.4f}%")
        print(f"      Predicted fail prob bound = {d['predicted_fail_bound']:.4e}")
    print()


def save_outputs(results, summary, verd, out_dir, cfg):
    out_dir.mkdir(parents=True, exist_ok=True)

    def jsonify(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not math.isfinite(o):
            return str(o)
        return o

    with open(out_dir / "per_layer_evidence_prob.json", "w") as f:
        json.dump({"config": cfg, "per_layer": results}, f, indent=2, default=jsonify)
    with open(out_dir / "headline_summary_prob.json", "w") as f:
        json.dump(summary, f, indent=2, default=jsonify)
    with open(out_dir / "theorem_verdict_prob.json", "w") as f:
        json.dump(verd, f, indent=2, default=jsonify)


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
    p.add_argument("--out-dir", type=str, default="./clc_prob_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-pct", type=float, default=95.0)
    p.add_argument("--delta-target", type=float, default=0.05)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 100)
    print("CLC PROBABILISTIC THEOREM — CONFIG")
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

    print(f"\n[1/3] Capturing calibration μ from {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_rec = ActivationRecorder(
        module_names, record_full_cov=False,
        max_tokens_per_module=args.max_cal_tokens_per_layer,
        flush_every_tokens=args.flush_every_tokens,
    )
    run_calibration(model, tok, cal_texts, cal_rec, device, args.max_length)
    cal_stats = cal_rec.finalize()

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

    modes = ["insample", "crossval"] if args.mode == "both" else [args.mode]
    print(f"\n[3/3] Probabilistic-bound measurement, modes={modes}")
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
                        delta_target=args.delta_target,
                    )
                    results.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate(results)
    verd = verdict(summary, agreement_threshold=args.threshold_pct / 100.0)
    print_report(results, summary, verd)
    save_outputs(results, summary, verd, Path(args.out_dir), vars(args))
    print(f"\nWrote: {args.out_dir}")


if __name__ == "__main__":
    main()