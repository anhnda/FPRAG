"""
Smoke test: verify the four-fact logic on a small synthetic problem.

Expected behaviour (in-sample, μ_eval = μ_cal):
  Fact (i)   : 100%       (algorithmic guarantee from EGBC)
  Fact (ii)  : 100%       (algorithmic guarantee from the budget bound)
  Fact (iii) : depends    (the substantive empirical claim)
  Fact (iv)  : depends    (the substantive empirical claim)

If (i) or (ii) is below 100% on synthetic data, there's a measurement bug.
(iii) and (iv) are NOT guaranteed by the algorithm; they're the empirical
claim the paper makes, and they hold on real LLM data per the spreadsheet.
On random synthetic data they may not hold — that's expected.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_egbc_paper import measure_layer
from verify_common import group_quantize  # only used to confirm we can import the chain


def run(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_features, in_features = 256, 512
    W = torch.randn(out_features, in_features, dtype=torch.float32)

    # Calibration μ
    mu_cal = 0.1 * torch.randn(in_features, dtype=torch.float32)
    idx = torch.randperm(in_features)[:5]
    mu_cal[idx] *= 20.0

    mu_eval = mu_cal.clone()  # in-sample
    A = torch.randn(in_features, in_features, dtype=torch.float32) * 0.05
    Sigma = (A @ A.t()) + 0.1 * torch.eye(in_features, dtype=torch.float32)

    cal_stats  = {"mu": mu_cal.float(),  "Sigma": None}
    eval_stats = {"mu": mu_eval.float(), "Sigma": Sigma.float()}

    res = measure_layer(
        name=f"synth/seed={seed}",
        W_fp=W,
        base="ntr",
        cal_stats=cal_stats,
        eval_stats=eval_stats,
        bits=4, group_size=128,
        flip_budget_pct=5.0, knee_tolerance=0.01,
        use_james_stein=False,
        mode="insample",
        device=torch.device("cpu"),
    )

    print(f"seed={seed}  base=ntr  mode=insample")
    for k in ("fact_i", "fact_ii", "fact_iii", "fact_iv"):
        print(f"  {k:10s} = {res[k]}")
    print(f"  dB/B (bias reduced by)       = {res['dB_rel']*100:+7.3f}%")
    print(f"  dV/V (variance change)       = {res['dV_rel']*100:+7.3f}%   (>0 means V grew)")
    print(f"  dT/T (total reduced by)      = {res['dT_rel']*100:+7.3f}%")
    print(f"  B: {res['B_before']:.4e} -> {res['B_after']:.4e}")
    print(f"  V: {res['V_before']:.4e} -> {res['V_after']:.4e}")
    print(f"  T: {res['T_before']:.4e} -> {res['T_after']:.4e}")
    print()
    return res


def main():
    print("=" * 64)
    print("Four-fact smoke test (no LLM)")
    print("=" * 64)
    print()
    for seed in (0, 1, 2):
        r = run(seed)
        assert r["fact_i"], f"BUG: Fact (i) failed on seed {seed} — bias should never grow."
        assert r["fact_ii"], f"BUG: Fact (ii) failed on seed {seed} — variance bound was breached."
    print("Smoke test OK.  Facts (i) and (ii) at 100% across seeds.")
    print("Facts (iii)/(iv) may vary on random data — that's expected.")


if __name__ == "__main__":
    main()