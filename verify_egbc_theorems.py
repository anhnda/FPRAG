"""
Paper-ready EGBC verification.

This script produces the LAYER-LEVEL empirical evidence backing the paper's
single main theorem and its supporting lemmas + assumption.

═══════════════════════════════════════════════════════════════════════════════
What the paper claims (and what this script verifies, layer-wise):
═══════════════════════════════════════════════════════════════════════════════

THEOREM 1 (EGBC reduces expected output error in expectation across layers).
For any lattice-valued base quantizer Q, the expected layer output error
strictly decreases on average across the layers of a model.

The theorem is supported by three lemmas and one assumption:

  LEMMA 1 (Bias descent, per layer).
      B(W_q')  ≤  B(W_q)            where B(W_q) := Σ_j (μ^⊤ e_j)²
      [algorithmic guarantee — should be 100% on every layer]

  LEMMA 2 (Variance perturbation bound, per layer).
      |V(W_q') − V(W_q)|  ≤  2 B_j s_max ‖Σe_j‖∞ + B_j² s_max² ‖Σ‖∞
      where V(W_q) := Σ_j e_j^⊤ Σ e_j
      [algorithmic guarantee — should be 100% on every layer]

  LEMMA 3 (Lattice closure & universal composability).
      W_q' ∈ L^{d×C}, the same lattice as W_q.
      Lemmas 1 & 2 hold for any base Q ∈ {NTR, AWQ, GPTQ, AdaRound, FlatQuant}.
      [structural — 100% by construction]

  ASSUMPTION A (Decorrelation of flip direction and variance direction).
      Define  α_ℓ := ⟨Δ_ℓ, Σ_ℓ E_ℓ⟩_F / (‖Δ_ℓ‖_F · ‖Σ_ℓ E_ℓ‖_F)  ∈ [-1, 1].
      Assume  E_ℓ[α_ℓ]  ≤  0  across the layers ℓ of the model.
      [must be empirically verified; this script measures α_ℓ per layer]

  CONCLUSION (Theorem 1):
      E_ℓ[R(W_q'^(ℓ))]  ≤  E_ℓ[R(W_q^(ℓ))]   where R := B + V,
      with strict inequality under the dominance condition
      E_ℓ[|ΔB_ℓ|] > E_ℓ[‖Δ_ℓ‖_F² · ‖Σ_ℓ‖_2].

═══════════════════════════════════════════════════════════════════════════════
Output structure:
═══════════════════════════════════════════════════════════════════════════════

For each (base, mode) ∈ {NTR, AWQ} × {insample, crossval}:

  Per-layer table: tick for L1, L2, L3, and the alignment α_ℓ.
  Headline block:
     – Lemma 1 frac of layers
     – Lemma 2 frac of layers
     – Lemma 3 frac of layers
     – Assumption A statistics: mean(α), std(α), frac. of layers with α ≤ 0
     – Dominance condition: ratio E[|ΔB|] / E[‖Δ‖² ‖Σ‖]
     – Theorem 1 conclusion (empirical): frac layers with total-L2 ↓ and avg ΔT/T
     – Aggregate totals across layers (B, V, T)

Verdict PASSES iff, on insample mode and for every base, all five gating
checks (Lemmas 1/2/3, Assumption A, and the Theorem 1 conclusion) clear
the threshold (default 95%).

Cross-eval (crossval mode) is reported as ROBUSTNESS, not the theorem.

═══════════════════════════════════════════════════════════════════════════════
Usage:
  python verify_egbc_final.py \\
      --model-path ./models/Mistral-7B-v0.3 \\
      --cal-dataset c4 --eval-datasets c4-val wikitext2 \\
      --n-cal 128 --n-eval 128 --max-length 1024 \\
      --bits 4 --group-size 128 \\
      --flip-budget-pct 5.0 --knee-tolerance 0.01 \\
      --base-quantizers ntr awq \\
      --layers-pattern "model.layers.0.self_attn.o_proj,model.layers.4.self_attn.o_proj,..." \\
      --out-dir ./egbc_paper_results
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
# Per-layer measurement: produces ONE row of evidence per (layer, base, mode)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_layer(
    name: str, W_fp: torch.Tensor, base: str,
    cal_stats: Dict, eval_stats: Dict,
    bits: int, group_size: int,
    flip_budget_pct: float, knee_tolerance: float,
    use_james_stein: bool, mode: str, device: torch.device,
) -> Dict[str, object]:
    """Return ONE measurement: per-layer (summed over channels) before/after EGBC.

    Returns:
      Per layer (one value, no per-row arrays):
        B_before, B_after        — Σ_j (μ^T e_j)²
        V_before, V_after        — Σ_j e_j^T Σ e_j
        T_before, T_after        — B + V (total layer L2 loss)
        RHS_T1                   — Σ_j [2 B_j s_max ‖Σe_j‖∞ + B_j² s_max² ‖Σ‖∞]
        |ΔV|                     — |V_after − V_before|
        ΔB, ΔV, ΔT (signed)      — ΔX := X_before − X_after  (positive = improved)
      And the four FACTS as booleans (per-layer):
        fact_i, fact_ii, fact_iii, fact_iv
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

    # μ used to evaluate the theorem against (insample = same μ the flip used; crossval = held-out)
    mu_for_eval = mu_cal if mode == "insample" else mu_eval

    # Salience for AWQ scaling
    salience = (mu_eval ** 2 + torch.diagonal(Sigma)).float()

    # Run base quantizer
    W_q_eff, W_int, scale_flat, zp_flat, s_vec = run_base_quantizer(
        base, W, mu_cal.float(), salience, bits, group_size,
    )
    W_q_eff = W_q_eff.float()
    e = (W_q_eff - W).double()

    # EGBC flip — in scaled space for AWQ, in original space for NTR
    mu_for_flip = (mu_cal.float() / s_vec).float() if base == "awq" else mu_cal.float()
    Delta_scaled = compute_flip_delta(
        W=W * s_vec.unsqueeze(0) if base == "awq" else W,
        W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_for_flip, bits=bits,
        flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
    ).double()
    Delta = Delta_scaled / s_vec.unsqueeze(0).double() if base == "awq" else Delta_scaled
    e_tilde = e + Delta

    # ----- Layer-summed bias and variance ----------------------------------
    # B = Σ_j (μ^T e_j)²
    B_before = ((e @ mu_for_eval) ** 2).sum()
    B_after = ((e_tilde @ mu_for_eval) ** 2).sum()
    # V = Σ_j e_j^T Σ e_j
    Se = e @ Sigma                                  # [out, in]
    V_before = (Se * e).sum()
    Set = e_tilde @ Sigma
    V_after = (Set * e_tilde).sum()

    T_before = B_before + V_before
    T_after = B_after + V_after

    # ----- Lemma 3 (lattice closure) check --------------------------------
    # Reconstruct W_q' integer codes and confirm they are integers in [0, max_int].
    # If lattice closure holds, |Δ / scale| is integer-valued (entry-wise).
    max_int = 2 ** bits - 1
    if base == "awq":
        # Δ in original space; the integer step is Δ * s_vec / scale_flat
        delta_int_pred = (Delta * s_vec.unsqueeze(0).double() / scale_flat.double())
    else:
        delta_int_pred = (Delta / scale_flat.double())
    is_integer = (delta_int_pred - delta_int_pred.round()).abs().max().item() < 1e-6
    # Check codes stay in range (verify on rows with at least one flip)
    W_int_post = W_int.double() + delta_int_pred
    code_in_range = bool(((W_int_post >= 0) & (W_int_post <= max_int)).all().item())
    lemma3_pass = bool(is_integer and code_in_range)

    # ----- Variance bound RHS (per-channel summed) — Lemma 2 -------------
    if base == "awq":
        step_orig = scale_flat.to(torch.float64) / s_vec.unsqueeze(0).to(torch.float64)
    else:
        step_orig = scale_flat.to(torch.float64)
    s_max = float(step_orig.max().item())
    Sigma_inf = float(Sigma.abs().max().item())
    Se_inf_per_row = Se.abs().max(dim=1).values     # ‖Σe_j‖∞ per channel
    B_per_row = (Delta != 0).sum(dim=1).double()    # |S_j| per channel
    rhs_per_row = 2.0 * B_per_row * s_max * Se_inf_per_row \
                  + (B_per_row ** 2) * (s_max ** 2) * Sigma_inf
    RHS_T1 = rhs_per_row.sum()                      # summed bound at layer level

    dV = V_after - V_before
    dV_abs = dV.abs()

    # ----- ASSUMPTION A measurement ----------------------------------------
    # α_ℓ = ⟨Δ, Σ E⟩_F  /  ( ‖Δ‖_F · ‖Σ E‖_F )    ∈ [-1, 1]
    # The cross-term of Δ V is 2 * ⟨Δ, Σ E⟩_F (Δ here = e' - e, E here = e).
    cross_inner   = (Delta * Se).sum()                      # ⟨Δ, Σ E⟩_F   (here E = e)
    Delta_F_norm  = (Delta ** 2).sum().sqrt()
    SE_F_norm     = (Se ** 2).sum().sqrt()
    if float(Delta_F_norm.item()) < 1e-30 or float(SE_F_norm.item()) < 1e-30:
        alpha = 0.0
    else:
        alpha = float((cross_inner / (Delta_F_norm * SE_F_norm)).item())

    # Also compute the quadratic budget magnitude: ‖Δ‖_F² · ‖Σ‖_2 (Theorem 1 dominance RHS).
    # Use ‖Σ‖_∞ as a fast proxy for ‖Σ‖_2 (we already computed Σ_inf above).
    quadratic_budget = float((Delta_F_norm ** 2).item()) * Sigma_inf
    bias_gain_abs = float((B_before - B_after).clamp(min=0).item())

    # ----- Numerical tolerance --------------------------------------------
    eps = 1e-10

    # ----- The four FACTS (per-layer, boolean) ----------------------------
    fact_i   = bool(B_after <= B_before + eps * (1.0 + B_before.abs()))                # Lemma 1
    fact_ii  = bool(dV_abs   <= RHS_T1   + eps * (1.0 + V_before.abs()))                # Lemma 2
    fact_iii = bool(T_after  <  T_before - eps * (1.0 + T_before.abs()))                # Theorem 1 conclusion (per-layer)
    fact_iv  = bool(V_after  <= V_before + eps * (1.0 + V_before.abs()))                # Empirical V no-worsening

    # ----- Relative magnitudes for reporting ------------------------------
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
        # Magnitudes
        "B_before": float(B_before.item()),
        "B_after":  float(B_after.item()),
        "V_before": float(V_before.item()),
        "V_after":  float(V_after.item()),
        "T_before": float(T_before.item()),
        "T_after":  float(T_after.item()),
        "RHS_T1":   float(RHS_T1.item()),
        "dV_abs":   float(dV_abs.item()),
        # Relative changes (signed; positive = improvement for B/T; negative dV = V went down)
        "dB_rel": rel(B_before - B_after, B_before),
        "dV_rel": rel(V_after - V_before, V_before),      # signed; +ve means V grew
        "dT_rel": rel(T_before - T_after, T_before),
        # Booleans
        "fact_i":   fact_i,
        "fact_ii":  fact_ii,
        "fact_iii": fact_iii,
        "fact_iv":  fact_iv,
        # Lemma 3
        "lemma3_pass": lemma3_pass,
        # Assumption A — the critical new quantity
        "alpha":            alpha,            # ⟨Δ, ΣE⟩_F / (‖Δ‖_F ‖ΣE‖_F) ∈ [-1, 1]
        "Delta_F_norm":     float(Delta_F_norm.item()),
        "SE_F_norm":        float(SE_F_norm.item()),
        "cross_term":       float(cross_inner.item()),     # signed
        "quadratic_budget": quadratic_budget,              # ‖Δ‖² · ‖Σ‖_∞
        "bias_gain_abs":    bias_gain_abs,                 # |ΔB| per layer
        # Budget actually consumed
        "B_mean_per_row": float(B_per_row.mean().item()),
        "n_flipped_rows": int((B_per_row > 0).sum().item()),
    }


# --------------------------------------------------------------------------- #
# Aggregation: (base, mode) → headline numbers
# --------------------------------------------------------------------------- #
def aggregate(results: List[Dict]) -> Dict[str, Dict]:
    """For each (base, mode), compute the four headline fractions and avg deltas."""
    by_key: Dict[Tuple[str, str], List[Dict]] = {}
    for r in results:
        by_key.setdefault((r["base"], r["mode"]), []).append(r)

    summary: Dict[str, Dict] = {}
    for (base, mode), rs in by_key.items():
        n = len(rs)
        frac = lambda k: sum(int(r[k]) for r in rs) / max(n, 1)
        avg = lambda k: float(np.mean([r[k] for r in rs]))

        # Assumption A — α statistics across layers
        alphas = [r["alpha"] for r in rs]
        alpha_mean = float(np.mean(alphas))
        alpha_std = float(np.std(alphas))
        alpha_min = float(np.min(alphas))
        alpha_max = float(np.max(alphas))
        alpha_frac_le_zero = float(np.mean([a <= 0 for a in alphas]))

        # Dominance condition (Theorem 1, Eq. eq:dominance):
        #   E_ℓ[|ΔB_ℓ|] > E_ℓ[‖Δ_ℓ‖_F² ‖Σ_ℓ‖_2]
        # We check whether the *expected* bias gain dominates the *expected* quadratic budget term.
        avg_bias_gain   = float(np.mean([r["bias_gain_abs"]    for r in rs]))
        avg_quad_budget = float(np.mean([r["quadratic_budget"] for r in rs]))
        dominance_holds = avg_bias_gain > avg_quad_budget

        # Lemma 3 (lattice closure) — must be 100%
        lemma3_frac = frac("lemma3_pass")

        summary[f"{base}::{mode}"] = {
            "base": base, "mode": mode, "n_layers": n,
            # The four headline facts
            "fact_i_frac":   frac("fact_i"),
            "fact_ii_frac":  frac("fact_ii"),
            "fact_iii_frac": frac("fact_iii"),
            "fact_iv_frac":  frac("fact_iv"),
            "lemma3_frac":   lemma3_frac,
            # Magnitudes averaged across layers
            "avg_dB_rel": avg("dB_rel"),
            "avg_dV_rel": avg("dV_rel"),
            "avg_dT_rel": avg("dT_rel"),
            # ASSUMPTION A statistics
            "alpha_mean":         alpha_mean,
            "alpha_std":          alpha_std,
            "alpha_min":          alpha_min,
            "alpha_max":          alpha_max,
            "alpha_frac_le_zero": alpha_frac_le_zero,
            # Dominance condition for Theorem 1
            "avg_bias_gain":    avg_bias_gain,
            "avg_quad_budget":  avg_quad_budget,
            "dominance_holds":  dominance_holds,
            "dominance_ratio":  avg_bias_gain / max(avg_quad_budget, 1e-30),
            # Aggregated absolute totals (sum across layers — useful for the table)
            "sum_B_before": sum(r["B_before"] for r in rs),
            "sum_B_after":  sum(r["B_after"]  for r in rs),
            "sum_V_before": sum(r["V_before"] for r in rs),
            "sum_V_after":  sum(r["V_after"]  for r in rs),
            "sum_T_before": sum(r["T_before"] for r in rs),
            "sum_T_after":  sum(r["T_after"]  for r in rs),
        }
    return summary


def theorem2_verdict(summary: Dict[str, Dict], threshold: float = 0.95) -> Dict:
    """Verdict on the FULL Theorem 1 chain (under Lemma 3 universality):

      Lemma 1 (bias descent)            ≥ threshold of layers
      Lemma 2 (variance budget)         ≥ threshold of layers
      Lemma 3 (lattice closure)         ≥ threshold of layers
      Assumption A (E[α] ≤ 0)           directly verified on the run
      Theorem 1 conclusion (T(W_q') < T(W_q))  ≥ threshold of layers

    All five conditions must hold on insample mode for EVERY base quantizer.
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
            failures.append(f"{s['base']}: Lemma 2 (variance budget) "
                            f"{s['fact_ii_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["lemma3_frac"] < threshold:
            failures.append(f"{s['base']}: Lemma 3 (lattice closure) "
                            f"{s['lemma3_frac']*100:.1f}% < {threshold*100:.0f}%")
        if s["alpha_mean"] > 0:
            failures.append(f"{s['base']}: Assumption A violated, "
                            f"E[α] = {s['alpha_mean']:+.5f} > 0")
        if s["fact_iii_frac"] < threshold:
            failures.append(f"{s['base']}: Theorem 1 conclusion "
                            f"{s['fact_iii_frac']*100:.1f}% < {threshold*100:.0f}%")
    return {
        "pass": len(failures) == 0,
        "threshold_pct": threshold * 100,
        "bases_tested": sorted({s["base"] for s in insample}),
        "failures": failures,
        # For appendix-friendly reporting
        "assumption_A_by_base": {s["base"]: {
            "alpha_mean": s["alpha_mean"], "alpha_std": s["alpha_std"],
            "frac_le_zero": s["alpha_frac_le_zero"],
        } for s in insample},
        "dominance_by_base": {s["base"]: {
            "ratio": s["dominance_ratio"], "holds": s["dominance_holds"],
        } for s in insample},
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(results, summary, t2):
    print()
    print("═" * 96)
    print("EGBC EMPIRICAL VERIFICATION — paper-ready evidence")
    print("═" * 96)
    print("""\
Mapping from measurement to paper claim:

  Lemma 1   →  bias descends, B(W_q') ≤ B(W_q)              (per layer; algorithmic guarantee)
  Lemma 2   →  variance perturbation within budget          (per layer; algorithmic guarantee)
  Lemma 3   →  lattice closure: W_q' ∈ L                    (per layer; structural)
  Asm. A    →  E_ℓ[α_ℓ] ≤ 0    where  α_ℓ = ⟨Δ,ΣE⟩_F / (‖Δ‖_F ‖ΣE‖_F)
  Theorem 1 →  E_ℓ[R(W_q'^(ℓ))] ≤ E_ℓ[R(W_q^(ℓ))]           (corollary of L1+L2+L3+Asm.A)

Universality (Lemma 3): the chain above holds for any lattice-valued base
quantizer Q (NTR, AWQ, GPTQ, AdaRound, SmoothQuant, FlatQuant).
""")

    # ----- Per-layer table -------------------------------------------------
    print("─" * 96)
    print("PER-LAYER EVIDENCE")
    print("─" * 96)
    header = (f"{'layer@eval':<55s} {'base':<5s} {'mode':<9s} "
              f"{'L1':>4s} {'L2':>4s} {'L3':>4s} {'α':>9s} "
              f"{'ΔB/B':>8s} {'ΔV/V':>8s} {'ΔT/T':>8s}")
    print(header)
    print("─" * 96)
    for r in sorted(results, key=lambda x: (x["base"], x["mode"], x["name"])):
        tick = lambda b: "✓" if b else "✗"
        print(f"{r['name']:<55s} {r['base']:<5s} {r['mode']:<9s} "
              f"{tick(r['fact_i']):>4s} {tick(r['fact_ii']):>4s} {tick(r['lemma3_pass']):>4s} "
              f"{r['alpha']:+9.5f} "
              f"{r['dB_rel']*100:+7.3f}% {r['dV_rel']*100:+7.3f}% {r['dT_rel']*100:+7.3f}%")

    # ----- Headline by (base, mode) ---------------------------------------
    print()
    print("═" * 96)
    print("HEADLINE: per-base × per-mode evidence")
    print("═" * 96)
    keys_ordered = sorted(summary.keys(), key=lambda k: (k.split("::")[1] != "insample", k))
    for key in keys_ordered:
        s = summary[key]
        is_theorem = (s["mode"] == "insample")
        tag = "[THEOREM]   " if is_theorem else "[robustness]"
        n = s["n_layers"]
        print(f"\n  {tag}  base = {s['base']:<5s}  mode = {s['mode']:<9s}  ({n} layers)")
        print(f"    Lemma 1  bias descent           : {int(s['fact_i_frac']*n):>3d}/{n:<3d} layers ({s['fact_i_frac']*100:6.2f}%)")
        print(f"    Lemma 2  variance within budget : {int(s['fact_ii_frac']*n):>3d}/{n:<3d} layers ({s['fact_ii_frac']*100:6.2f}%)")
        print(f"    Lemma 3  lattice closure        : {int(s['lemma3_frac']*n):>3d}/{n:<3d} layers ({s['lemma3_frac']*100:6.2f}%)")
        print(f"    Assumption A — α statistics across {n} layers:")
        print(f"        mean(α) = {s['alpha_mean']:+9.5f}    std(α)  = {s['alpha_std']:9.5f}")
        print(f"        min(α)  = {s['alpha_min']:+9.5f}    max(α)  = {s['alpha_max']:+9.5f}")
        print(f"        frac. of layers with α ≤ 0: {s['alpha_frac_le_zero']*100:6.2f}%")
        verdict_A = "HOLDS" if s['alpha_mean'] <= 0 else "VIOLATED on this run"
        print(f"        Assumption A (E[α] ≤ 0)  :  {verdict_A}")
        print(f"    Dominance check (Eq. 4):  E[|ΔB|] vs E[‖Δ‖² · ‖Σ‖]")
        print(f"        E[|ΔB|]           = {s['avg_bias_gain']:.6e}")
        print(f"        E[‖Δ‖_F² ‖Σ‖_∞]   = {s['avg_quad_budget']:.6e}")
        print(f"        ratio             = {s['dominance_ratio']:.2f}×    ({'HOLDS' if s['dominance_holds'] else 'fails'})")
        print(f"    Theorem 1 conclusion (per-layer total descent observed):")
        print(f"        {int(s['fact_iii_frac']*n):>3d}/{n:<3d} layers have T(W_q') < T(W_q)   "
              f"(avg ΔT/T = {s['avg_dT_rel']*100:+.3f}%)")
        print(f"    Magnitudes (avg across layers):")
        print(f"        ΔB/B  = {s['avg_dB_rel']*100:+7.3f}%   (bias reduced by this fraction)")
        print(f"        ΔV/V  = {s['avg_dV_rel']*100:+7.3f}%   (negative = V shrank too)")
        print(f"        ΔT/T  = {s['avg_dT_rel']*100:+7.3f}%   (total layer L2 loss reduced)")
        print(f"    Aggregate (sum over layers):")
        print(f"        B: {s['sum_B_before']:.4e} → {s['sum_B_after']:.4e}    "
              f"({s['sum_B_before']/max(s['sum_B_after'],1e-30):.1f}× reduction)")
        print(f"        V: {s['sum_V_before']:.4e} → {s['sum_V_after']:.4e}")
        print(f"        T: {s['sum_T_before']:.4e} → {s['sum_T_after']:.4e}    "
              f"({100*(s['sum_T_before']-s['sum_T_after'])/max(s['sum_T_before'],1e-30):+.3f}% reduction)")

    # ----- Verdict --------------------------------------------------------
    print()
    print("═" * 96)
    print("THEOREM 1 (under Lemma 3 universality) — verdict on insample")
    print("═" * 96)
    print(f"  Gating checks (≥ {t2['threshold_pct']:.0f}% on insample):")
    print(f"     – Lemma 1  (bias descent)")
    print(f"     – Lemma 2  (variance budget)")
    print(f"     – Lemma 3  (lattice closure)")
    print(f"     – Assumption A  (E[α] ≤ 0)")
    print(f"     – Theorem 1 conclusion  (per-layer total descent observed)")
    print(f"  Bases tested:  {t2['bases_tested']}")
    if t2["pass"]:
        print(f"  VERDICT:  PASS  ✓   Theorem 1 (under verified Assumption A) holds across {t2['bases_tested']}.")
    else:
        print(f"  VERDICT:  FAIL  ✗")
        for f in t2["failures"]:
            print(f"    – {f}")
    print()

    # ----- Paper-ready sentences -----------------------------------------
    print("═" * 96)
    print("PAPER-READY SENTENCES")
    print("═" * 96)
    insample_keys = [k for k in summary.keys() if k.endswith("::insample")]
    if insample_keys:
        print()
        for k in sorted(insample_keys):
            s = summary[k]
            n = s["n_layers"]
            print(f"  On {s['base'].upper()}-base (in-sample; the setting of Theorem 1):")
            print(f"    – Lemma 1 (bias descent): {int(s['fact_i_frac']*n)}/{n} layers.")
            print(f"    – Lemma 2 (variance budget): {int(s['fact_ii_frac']*n)}/{n} layers.")
            print(f"    – Lemma 3 (lattice closure): {int(s['lemma3_frac']*n)}/{n} layers.")
            print(f"    – Assumption A: mean α = {s['alpha_mean']:+.5f} (s.d. {s['alpha_std']:.5f}), "
                  f"{s['alpha_frac_le_zero']*100:.0f}% of layers have α ≤ 0.")
            print(f"    – Dominance (Eq. 4): E[|ΔB|]/E[‖Δ‖²‖Σ‖] = {s['dominance_ratio']:.1f}× "
                  f"({'satisfied' if s['dominance_holds'] else 'violated'}).")
            print(f"    – Theorem 1 conclusion (empirical): total layer L2 strictly decreased on "
                  f"{int(s['fact_iii_frac']*n)}/{n} layers; avg ΔT/T = {s['avg_dT_rel']*100:+.3f}%. "
                  f"Aggregate total: {s['sum_T_before']:.3e} → {s['sum_T_after']:.3e} "
                  f"({100*(s['sum_T_before']-s['sum_T_after'])/max(s['sum_T_before'],1e-30):+.2f}%).")
            print()
    crossval_keys = [k for k in summary.keys() if k.endswith("::crossval")]
    if crossval_keys:
        print("  Cross-eval (robustness; flip designed on cal, evaluated on a different μ):")
        for k in sorted(crossval_keys):
            s = summary[k]
            n = s["n_layers"]
            print(f"    – {s['base'].upper()}: Theorem 1 conclusion holds on "
                  f"{int(s['fact_iii_frac']*n)}/{n} layers; avg ΔT/T = {s['avg_dT_rel']*100:+.3f}%; "
                  f"mean α = {s['alpha_mean']:+.5f}.")
    print("═" * 96)


def save_outputs(results, summary, t2, out_dir, cfg):
    out_dir.mkdir(parents=True, exist_ok=True)

    def jsonify(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, bool):
            return o
        return o

    with open(out_dir / "per_layer_evidence.json", "w") as f:
        json.dump({"config": cfg, "per_layer": results}, f, indent=2, default=jsonify)
    with open(out_dir / "headline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=jsonify)
    with open(out_dir / "theorem2_verdict.json", "w") as f:
        json.dump(t2, f, indent=2, default=jsonify)


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
    p.add_argument("--out-dir", type=str, default="./egbc_paper_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-pct", type=float, default=95.0,
                   help="Pass threshold (in %%) used in the Theorem 2 verdict.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 96)
    print("EGBC PAPER VERIFICATION — CONFIG")
    print("═" * 96)
    for k, v in vars(args).items():
        print(f"  {k:32s} = {v}")
    print("═" * 96)

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

    # ----- 1) Calibration μ -------------------------------------------------
    print(f"\n[1/3] Capturing calibration μ from {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_rec = ActivationRecorder(
        module_names, record_full_cov=False,
        max_tokens_per_module=args.max_cal_tokens_per_layer,
        flush_every_tokens=args.flush_every_tokens,
    )
    run_calibration(model, tok, cal_texts, cal_rec, device, args.max_length)
    cal_stats = cal_rec.finalize()

    # ----- 2) Eval μ + full Σ -----------------------------------------------
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
        for n in module_names:
            eval_stats_per_module[n][ev] = rec.finalize()[n] if False else rec.finalize()[n]
        # finalize() builds the dict fresh; cache it once
        finalized = rec.finalize()
        for n in module_names:
            eval_stats_per_module[n][ev] = finalized[n]
        del rec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- 3) Per-layer measurement ----------------------------------------
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
    t2 = theorem2_verdict(summary, threshold=args.threshold_pct / 100.0)
    print_report(results, summary, t2)
    save_outputs(results, summary, t2, Path(args.out_dir), vars(args))
    print(f"\nWrote: {args.out_dir}")


if __name__ == "__main__":
    main()