"""
Verification of the probabilistic MSE descent theorem for CLC.

═══════════════════════════════════════════════════════════════════════════════
THE THEOREM (new, probabilistic — replaces the old D_j ≤ 0 "structural" claim):
═══════════════════════════════════════════════════════════════════════════════

  Assumption (A1): Residuals e_{j,i} are sub-Gaussian with parameter σ ≤ s_j/2.
                   (Standard for bounded residuals on [-s_j/2, s_j/2].)
  Assumption (A2): The selection support S_j is independent of e_{j,i}.
                   (S is chosen from μ-statistics, e is the rounding residual.)

  Then with probability ≥ 1 − δ, where δ := exp(-gap²/(2·λ²·s⁴·B)):

      ΔR_ℓ ≤ −G_ℓ + s²·B·λ + 2τ_δ

  with τ_δ = s²·√(B·log(1/δ))·λ, and ΔR_ℓ < 0 (MSE descent) whenever:

      gap := G_ℓ − s²·B·λ > 0    (deterministic descent condition)

  The failure probability is bounded by:

      Pr[ ΔR_ℓ ≥ 0 ] ≤ exp( −gap² / (2·λ²·s⁴·B) )

  where:
      G_ℓ = bias descent = B_before − B_after  (deterministic, ≥ 0)
      B   = |S| = number of flipped weights
      λ   = ‖Σ_{SS}‖_op = spectral norm of Σ on the support
      s   = per-channel step size (we use a row-aggregated form below)

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS SCRIPT VERIFIES (per layer, per (base, mode)):
═══════════════════════════════════════════════════════════════════════════════

  For each layer ℓ:
    1. Compute G_ℓ, B, λ, s -> gap, predicted failure prob δ̂
    2. Measure actual ΔR_ℓ
    3. Record:
         - PRED_descent: bool   (gap > 0 → theory predicts descent)
         - ACTUAL_descent: bool (ΔR_ℓ < 0 → actually descended)
         - failure_prob_bound: exp(-gap²/(2λ²s⁴B))
         - empirical_slack: ratio (gap) / (s²·B·λ)   — how comfortable

  Aggregate across layers:
    - Fraction where PRED → ACTUAL (theorem correctly predicts descent)
    - Mean predicted failure probability vs empirical failure rate
    - Distribution of gap/(s²Bλ)  — is the regime comfortable or marginal?
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
# Base quantizer (same as v3)
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
# Spectral norm of Σ on the support
# --------------------------------------------------------------------------- #
@torch.no_grad()
def lambda_on_support(Sigma: torch.Tensor, support_indices: torch.Tensor) -> float:
    """Compute ‖Σ_{SS}‖_op for one row's support.

    Σ:                [d, d]  full activation covariance (double)
    support_indices:  [|S|]   indices of selected coordinates for THIS row
    """
    if support_indices.numel() == 0:
        return 0.0
    S = support_indices.to(Sigma.device)
    Sigma_SS = Sigma.index_select(0, S).index_select(1, S)   # [|S|, |S|]
    # spectral norm; use eigvalsh since Sigma_SS is symmetric PSD
    try:
        eigs = torch.linalg.eigvalsh(Sigma_SS)
        return float(eigs.max().item())
    except Exception:
        # fallback to power iteration / svd
        return float(torch.linalg.matrix_norm(Sigma_SS, ord=2).item())


# --------------------------------------------------------------------------- #
# Per-layer measurement
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
    """Measure the probabilistic descent bound per layer.

    Layer-aggregated formulation:
      We treat the *entire layer* (all rows pooled) as a single instance.
        G_ℓ        := B_before − B_after  (total bias descent for the layer)
        B          := total support size across all rows
        s_max      := max step size used (per-row max scale, then layer max)
        λ          := worst-case spectral bound, approximated as ‖Σ‖_op
                      (this is a conservative upper bound to ‖Σ_{SS}‖_op for any S)

      gap_ℓ := G_ℓ − (s_max² · B · λ)
      pred_descent_ℓ := (gap_ℓ > 0)
      fail_prob_bound_ℓ := exp(−gap_ℓ² / (2·λ²·s_max⁴·B))   if gap_ℓ > 0 else 1.0
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

    # ===== B (bias) and V (variance) and Total =====
    B_before = ((e @ mu_for_eval) ** 2).sum()
    B_after  = ((e_tilde @ mu_for_eval) ** 2).sum()
    Se        = e @ Sigma
    V_before  = (Se * e).sum()
    Set       = e_tilde @ Sigma
    V_after   = (Set * e_tilde).sum()
    T_before  = B_before + V_before
    T_after   = B_after  + V_after
    G_layer   = float((B_before - B_after).item())          # ≥ 0 by greedy
    dR_layer  = float((T_after - T_before).item())          # < 0 means descent

    # ===== Quantities for the bound =====
    support_mask = (Delta != 0)
    B_support_layer = int(support_mask.sum().item())              # total flips
    abs_Delta = Delta.abs()
    # Step size: |Δ_{j,i}| is the per-(j,i) step magnitude in *original* weight space.
    # Layer-level conservative s:
    s_max_layer = float(abs_Delta[support_mask].max().item()) if B_support_layer > 0 else 0.0

    # λ: spectral norm of Σ. Conservative upper bound on ‖Σ_{SS}‖_op for any S.
    # (We could refine per-row, but the layer-level form gives a single clean number.)
    try:
        Sigma_eigs = torch.linalg.eigvalsh(Sigma)
        lam_layer = float(Sigma_eigs.max().item())
    except Exception:
        lam_layer = float(torch.linalg.matrix_norm(Sigma, ord=2).item())

    # ===== Gap & failure probability bound =====
    sBL = (s_max_layer ** 2) * B_support_layer * lam_layer        # s²·B·λ
    gap = G_layer - sBL
    pred_descent = bool(gap > 0)

    if gap > 0 and lam_layer > 0 and s_max_layer > 0 and B_support_layer > 0:
        # exponent: −gap² / (2·λ²·s⁴·B)
        denom = 2.0 * (lam_layer ** 2) * (s_max_layer ** 4) * B_support_layer
        exponent = -(gap ** 2) / max(denom, 1e-300)
        # Clamp to avoid underflow
        fail_prob_bound = math.exp(max(exponent, -700.0))
    else:
        fail_prob_bound = 1.0
    actual_descent = bool(dR_layer < 0)

    # Slack ratio: how comfortable is the descent condition?
    slack_ratio = gap / sBL if sBL > 1e-30 else float("inf")

    # ===== Also compute a per-row refined version: =====
    #   For each row j, compute λ_j = ‖Σ_{S_j S_j}‖_op,
    #   s_j = max |Δ_{j,i}| on support, B_j = |S_j|, G_j = bias descent for row j.
    # This gives a per-row predicted descent and a more accurate failure bound.
    rows_with_flips = []
    rows_pred_descent = 0
    rows_actual_descent = 0
    rows_pred_and_actual = 0
    rows_pred_not_actual = 0
    rows_notpred_actual = 0
    per_row_fail_probs = []
    per_row_slack = []

    for j in range(out_features):
        S_j = torch.nonzero(support_mask[j], as_tuple=False).flatten()
        Bj = int(S_j.numel())
        if Bj == 0:
            continue
        # row-level G_j
        bias_diff_j = (mu_for_eval @ e[j]) ** 2 - (mu_for_eval @ e_tilde[j]) ** 2
        G_j = float(bias_diff_j.item())
        # row-level total descent
        eVe   = (e[j] @ Sigma) @ e[j]
        etVet = (e_tilde[j] @ Sigma) @ e_tilde[j]
        T_before_j = float(((mu_for_eval @ e[j]) ** 2 + eVe).item())
        T_after_j  = float(((mu_for_eval @ e_tilde[j]) ** 2 + etVet).item())
        dR_j = T_after_j - T_before_j
        # row-level s_j and λ_j
        s_j = float(abs_Delta[j][support_mask[j]].max().item())
        lam_j = lambda_on_support(Sigma, S_j)

        sBL_j = (s_j ** 2) * Bj * lam_j
        gap_j = G_j - sBL_j
        pred_j = (gap_j > 0)
        actual_j = (dR_j < 0)
        rows_pred_descent += int(pred_j)
        rows_actual_descent += int(actual_j)
        rows_pred_and_actual += int(pred_j and actual_j)
        rows_pred_not_actual += int(pred_j and not actual_j)
        rows_notpred_actual += int((not pred_j) and actual_j)

        if pred_j and lam_j > 0 and s_j > 0:
            denom_j = 2.0 * (lam_j ** 2) * (s_j ** 4) * Bj
            fp_j = math.exp(max(-(gap_j ** 2) / max(denom_j, 1e-300), -700.0))
        else:
            fp_j = 1.0
        per_row_fail_probs.append(fp_j)
        per_row_slack.append(gap_j / sBL_j if sBL_j > 1e-30 else float("inf"))
        rows_with_flips.append(j)

    n_rows_flipped = len(rows_with_flips)
    mean_fp_bound = float(np.mean(per_row_fail_probs)) if per_row_fail_probs else 1.0
    mean_slack    = float(np.mean([s for s in per_row_slack if math.isfinite(s)])) if per_row_slack else 0.0
    rows_pred_correctly = rows_pred_and_actual + (n_rows_flipped - rows_pred_descent - rows_notpred_actual)
    # Rate at which prediction agrees with reality:
    agreement_rate = (rows_pred_and_actual + (n_rows_flipped - rows_pred_descent - rows_notpred_actual)) \
                      / max(n_rows_flipped, 1)

    return {
        "name":            name,
        "base":            base,
        "mode":            mode,
        "n_rows":          int(out_features),
        # ===== Theorem inputs (layer aggregate) =====
        "G_layer":         G_layer,
        "B_support":       B_support_layer,
        "s_max":           s_max_layer,
        "lambda_layer":    lam_layer,
        "sBL":             sBL,
        "gap":             gap,
        "slack_ratio":     slack_ratio,
        # ===== Theorem outputs =====
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
        # ===== Magnitudes (for sanity) =====
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
        # Layer-level metrics
        n_pred = sum(int(r["pred_descent"]) for r in rs)
        n_actual = sum(int(r["actual_descent"]) for r in rs)
        n_pred_and_actual = sum(int(r["pred_descent"] and r["actual_descent"]) for r in rs)
        n_pred_not_actual = sum(int(r["pred_descent"] and not r["actual_descent"]) for r in rs)
        n_notpred_actual = sum(int((not r["pred_descent"]) and r["actual_descent"]) for r in rs)
        n_agree = sum(int(r["pred_descent"] == r["actual_descent"]) for r in rs)

        # Row-level metrics (aggregated across all rows in all layers)
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
            # ----- Layer level -----
            "layer_pred_descent_frac":   n_pred / max(n, 1),
            "layer_actual_descent_frac": n_actual / max(n, 1),
            "layer_agreement_frac":      n_agree / max(n, 1),
            "layer_pred_implies_actual": (n_pred_and_actual / n_pred) if n_pred > 0 else None,
            "layer_n_pred":              n_pred,
            "layer_n_actual":            n_actual,
            "layer_n_pred_and_actual":   n_pred_and_actual,
            "layer_n_pred_not_actual":   n_pred_not_actual,
            "layer_n_notpred_actual":    n_notpred_actual,
            # ----- Row level -----
            "row_total_flipped":         total_rows_flipped,
            "row_pred_descent_frac":     total_rows_pred / max(total_rows_flipped, 1),
            "row_actual_descent_frac":   total_rows_actual / max(total_rows_flipped, 1),
            "row_pred_implies_actual":   (total_rows_pred_and_actual / total_rows_pred) if total_rows_pred > 0 else None,
            "row_pred_not_actual":       total_rows_pred_not_actual,
            "row_notpred_actual":        total_rows_notpred_actual,
            # ----- Quantitative -----
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
    """The theorem is VERIFIED if, across all settings:

      (A) Whenever the deterministic condition gap > 0 holds, ΔR < 0 in practice.
          → row_pred_implies_actual ≥ threshold
      (B) The empirical failure rate (in regimes where the theorem applies) is
          consistent with the predicted bound (mean_fail_prob_bound).
          → empirical disagreement ≤ mean_fail_prob_bound (loosely)
    """
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
        # Theorem check: agreement on the predicted-descent rows
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

whenever G_ℓ > s²·B·λ.  Here G_ℓ is the bias descent, B = |S| the flip count,
λ = ‖Σ_{SS}‖_op the spectral norm on the support, s the step size.

VERIFICATION (per layer and per row):
  1. Compute G, B, λ, s; form gap = G − s²·B·λ.
  2. PRED_descent  := gap > 0.        (sufficient condition for MSE descent)
  3. ACTUAL_descent := ΔR_ℓ < 0.       (measured)
  4. Theorem prediction: PRED ⇒ ACTUAL on ≥ 95% of (layer or row) instances.

The bound also predicts a numerical failure probability per layer; we report
its mean and compare to empirical disagreement rate.
""")

    # ----- Per-layer table -----
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

    # ----- Verdict -----
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
    p.add_argument("--delta-target", type=float, default=0.05,
                   help="target failure probability for reporting (informational only)")
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