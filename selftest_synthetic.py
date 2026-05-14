"""
Synthetic self-test of the EGBC theorem verification math.

Goal: prove that on randomly generated (W, μ, Σ) the verification code reports
Theorem 1 (i) and (ii) at 100% pass rate, and (iii) at 100% whenever the
dominance condition is satisfied for some rows.

This is NOT a verification of the theorem itself (which is a math claim, not an
empirical claim).  It's a smoke test that the *measurement code* in
verify_egbc_theorems.py implements the right formulas.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_common import compute_flip_delta, group_quantize


def run_one(out_features=256, in_features=512, bits=4, group_size=128,
            flip_budget_pct=5.0, knee_tolerance=0.01, seed=0,
            mu_eval_noise_scale=0.0):
    """One trial of the synthetic check.

    Theorem 1 assumes a SINGLE μ.  The flip operator uses μ_cal to decide flips,
    and we evaluate bias against μ_eval.  When μ_eval = μ_cal (the in-sample
    setting that the theorem actually states), all three claims should hold.
    `mu_eval_noise_scale > 0` introduces a calibration-vs-eval mismatch and
    tests robustness, NOT the theorem itself.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random FP weight matrix
    W = torch.randn(out_features, in_features, dtype=torch.float32)

    # Calibration μ used by the flip operator
    mu_cal = 0.1 * torch.randn(in_features, dtype=torch.float32)
    k = 5
    idx = torch.randperm(in_features)[:k]
    mu_cal[idx] *= 20.0

    # Evaluation μ:  by default equal to mu_cal (in-sample theorem check).
    # Set mu_eval_noise_scale > 0 to stress-test robustness (NOT the theorem).
    if mu_eval_noise_scale > 0:
        mu_eval = mu_cal + mu_eval_noise_scale * torch.randn(in_features, dtype=torch.float32)
    else:
        mu_eval = mu_cal.clone()

    # PSD Σ
    A = torch.randn(in_features, in_features, dtype=torch.float32) * 0.05
    Sigma = (A @ A.t()) + 0.1 * torch.eye(in_features, dtype=torch.float32)

    # Run base quantizer (NTR here)
    W_q, W_int, scale_flat, zp_flat = group_quantize(W, bits=bits, group_size=group_size)
    e = (W_q - W).double()

    # Apply flip
    Delta = compute_flip_delta(
        W=W, W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_cal, bits=bits,
        flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
    ).double()
    e_tilde = e + Delta

    # Theorem 1 quantities — exactly the same arithmetic as in measure_layer().
    mu_e = mu_eval.double()
    Sig = Sigma.double()

    bias_before = (e @ mu_e) ** 2
    bias_after  = (e_tilde @ mu_e) ** 2

    Se = e @ Sig
    var_before = (Se * e).sum(dim=1)
    Set = e_tilde @ Sig
    var_after = (Set * e_tilde).sum(dim=1)

    s_max = float(scale_flat.max().item())
    Sigma_inf = float(Sig.abs().max().item())
    Se_inf = Se.abs().max(dim=1).values
    B = (Delta != 0).sum(dim=1).double()
    rhs = B * s_max * (2.0 * Se_inf + s_max * Sigma_inf)

    eps = 1e-10  # realistic tolerance for float32 weights / float64 accumulators

    # (i) bias descent
    pass_i = bias_after <= bias_before + eps * (1.0 + bias_before)

    # (ii) variance budget
    var_change = (var_after - var_before).abs()
    pass_ii = var_change <= rhs + eps * (1.0 + var_before)

    # (iii) net descent under dominance
    dom = bias_before > rhs
    total_before = bias_before + var_before
    total_after  = bias_after  + var_after
    pass_iii_under_dom = (total_after < total_before)[dom] if dom.any() else torch.tensor([], dtype=torch.bool)

    # Strict bias decrease for flipped rows
    flipped_rows = (B > 0)
    bias_strict_dec = (bias_after < bias_before - eps * (1.0 + bias_before))

    return {
        "out": out_features, "in": in_features,
        "rows_flipped":   int(flipped_rows.sum().item()),
        "T1(i)_pass":     float(pass_i.float().mean().item()),
        "T1(ii)_pass":    float(pass_ii.float().mean().item()),
        "T1(iii)_pass":   float(pass_iii_under_dom.float().mean().item())
                          if pass_iii_under_dom.numel() > 0 else float("nan"),
        "T1(iii)_n_dom":  int(dom.sum().item()),
        "strict_dec_on_flipped":
            float(bias_strict_dec[flipped_rows].float().mean().item())
            if flipped_rows.any() else float("nan"),
        "bias_before_mean": float(bias_before.mean().item()),
        "bias_after_mean":  float(bias_after.mean().item()),
        "var_change_mean":  float((var_after - var_before).abs().mean().item()),
        "rhs_mean":         float(rhs.mean().item()),
        "rhs_slack_mean":   float((rhs - var_change).mean().item()),
    }


def main():
    print("Synthetic self-test of EGBC theorem verification code")
    print("=" * 72)
    print("\nMODE A: in-sample (μ_eval = μ_cal) — this is what the theorem states.")
    print("-" * 72)
    for seed in range(3):
        r = run_one(seed=seed, mu_eval_noise_scale=0.0)
        print(f"\nseed={seed}")
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k:32s} = {v:.6f}")
            else:
                print(f"  {k:32s} = {v}")

    print("\n\nMODE B: noisy calibration (μ_eval ≠ μ_cal) — robustness, NOT the theorem.")
    print("-" * 72)
    for seed in range(3):
        r = run_one(seed=seed, mu_eval_noise_scale=0.02)
        print(f"\nseed={seed}")
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k:32s} = {v:.6f}")
            else:
                print(f"  {k:32s} = {v}")

    print("\nExpected behaviour in MODE A:")
    print("  T1(i)_pass     -> 1.000000   (algorithmic guarantee)")
    print("  T1(ii)_pass    -> 1.000000   (algorithmic guarantee)")
    print("  T1(iii)_pass under dom -> 1.000000")
    print("  rhs_slack_mean -> large positive (bound is loose, which is fine)")
    print("\nMode B may show T1(i) below 1.0 if the calibration-eval mismatch")
    print("is large — that's about generalization of the flip, not the theorem.")


if __name__ == "__main__":
    main()