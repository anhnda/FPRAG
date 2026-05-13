"""
Verify the EGBC theory.

The theory has two pieces:

(P1) Gap identity (exact, fp64):
    G_j = R_BC,j - R_Flip,j
        = -2(δ^T e_j)(δ^T Δ_j)        -- T1, the cross-term
          - (δ^T Δ_j)^2                -- T2 ≤ 0
          - ε_j^2                       -- T3 ≤ 0,  ε_j = μ_cal^T (e_j + Δ_j)
          - 2 ε_j (δ^T (e_j+Δ_j))      -- T4
          - V_j                         -- T5,  V_j = 2 e^T Σ Δ + Δ^T Σ Δ

(P2) Mechanistic prediction. Define the per-row sign-correlation between
the rounding error's cal-projection and its eval-shift-projection:

    ρ_j  =  sign(μ_cal^T e_j) * sign(δ^T e_j)        ∈ {-1, +1}    (defined when both nonzero)

Then on the layers that dominate end-to-end loss (i.e., layers where R_BC,j is
large in absolute terms), we predict:

    E_j[ρ_j] > 0     => sign(T1) = +1 => G_layer > 0 (Flip wins per-layer MSE).

This replaces the failed "α = cos(δ, μ_cal) > 0" claim. The new claim is at
the row level using rounding-error projections, not at the vector level
using activation cosines.

Falsifiable predictions:
  N1. On layers with G_layer > 0 measured via full-Σ, we have E_j[ρ_j] > 0.
  N2. On layers with G_layer < 0, we have E_j[ρ_j] < 0.
  N3. The Spearman correlation between E_j[ρ_j] across layers and G_layer is
      strong (|ρ| > 0.5).
  N4. The Pearson correlation between T1 (signed) and a row-sum proxy
      Σ_j sign(μ_cal^T e_j) * sign(δ^T e_j) * |μ_cal^T e_j| * |δ^T e_j|
      across layers is very strong (> 0.9).

Pipeline:
  Read full-covariance verification output (from verify_flip_vs_bc_theory.py),
  recompute per-row ρ_j and the row-sum proxy, output:
    - E_j[ρ_j] per layer per eval
    - sign agreement with G_layer
    - correlation N3
    - correlation N4

Then: run a one-shot Spearman test across all (layer, eval) pairs.

Usage:
  python verify_egbc_theory.py \
      --model-path ./models/Mistral-7B-v0.3 \
      --cal-dataset c4 --eval-datasets c4-val wikitext2 \
      --n-cal 128 --n-eval 128 --max-length 1024 \
      --bits 4 --group-size 128 --flip-budget-pct 5.0 --knee-tolerance 0.01 \
      --layers-pattern "model.layers.0.self_attn.o_proj,model.layers.4.self_attn.o_proj,model.layers.8.mlp.down_proj,model.layers.12.mlp.down_proj,model.layers.17.self_attn.o_proj,model.layers.23.self_attn.o_proj,model.layers.27.mlp.down_proj,model.layers.31.mlp.down_proj" \
      --max-layers 0 \
      --out-dir ./egbc_theory_results

CRITICAL: Run this with FULL covariance (do NOT pass --no-full-cov), otherwise
the diagonal approximation distorts both G and the rho test.
"""

import argparse
import fnmatch
import gc
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Reuse the verification's plumbing — never reimplement the algorithm here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_flip_vs_bc_theory import (
    set_seed,
    load_text_samples,
    select_modules,
    group_quantize,
    compute_flip_delta,
    ActivationRecorder,
    run_calibration,
)


# ============================================================================
# Per-layer measurement: G_j and rho_j, both computed on full Σ.
# ============================================================================
@torch.no_grad()
def measure_layer(
    name: str,
    W_fp: torch.Tensor,
    cal_stats: Dict,
    eval_stats: Dict,
    bits: int,
    group_size: int,
    flip_budget_pct: float,
    knee_tolerance: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    For one (layer, eval) pair, compute everything we need to test the new theory:
        - G_j per row
        - rho_j = sign(μ_cal^T e_j) * sign(δ^T e_j) per row
        - T1, T2, T3, T4, T5 per row
        - R_BC, R_Flip per row, mean over rows

    Returns a dict of summary statistics PLUS the per-row arrays so the caller
    can aggregate across (layer, eval).
    """
    W = W_fp.to(device).float()
    out_features, in_features = W.shape

    mu_cal = cal_stats["mu"].to(device).double()
    mu_eval = eval_stats["mu"].to(device).double()
    Sigma_eval = eval_stats["Sigma"]
    if Sigma_eval is None:
        raise ValueError(
            f"Layer {name}: full covariance is required for this theory test. "
            "Re-run the upstream verification without --no-full-cov."
        )
    Sigma_eval = Sigma_eval.to(device).double()
    delta = mu_eval - mu_cal

    # 1. Quantize and compute Flip delta (paper-aligned algorithm)
    W_q, W_int, scale_flat, zp_flat = group_quantize(W, bits=bits, group_size=group_size)
    e = (W_q - W).double()
    Delta = compute_flip_delta(
        W=W, W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_cal.float(), bits=bits,
        flip_budget_pct=flip_budget_pct,
        knee_tolerance=knee_tolerance,
    ).double()

    # 2. Per-row scalar projections
    mu_cal_dot_e = e @ mu_cal                           # [out]
    delta_dot_e = e @ delta                             # [out]
    delta_dot_Delta = Delta @ delta                     # [out]
    eps = (e + Delta) @ mu_cal                          # [out]
    delta_dot_etilde = delta_dot_e + delta_dot_Delta    # [out]
    mu_eval_dot_e = e @ mu_eval                         # [out]
    mu_eval_dot_etilde = (e + Delta) @ mu_eval          # [out]

    # 3. Quadratic forms with full Σ_eval (these are where diag broke the analysis)
    eS = e @ Sigma_eval                                  # [out, in]
    q_ee = (eS * e).sum(dim=1)                           # [out]
    q_eD = (eS * Delta).sum(dim=1)
    q_DD = (Delta @ Sigma_eval * Delta).sum(dim=1)
    et = e + Delta
    q_ee_tilde = ((et @ Sigma_eval) * et).sum(dim=1)

    # 4. Per-row eval risks
    R_BC = delta_dot_e ** 2 + q_ee
    R_Flip = mu_eval_dot_etilde ** 2 + q_ee_tilde

    G = R_BC - R_Flip                                    # [out]

    # 5. T1..T5 decomposition (Prop 1, exact)
    T1 = -2 * delta_dot_e * delta_dot_Delta
    T2 = -(delta_dot_Delta ** 2)
    T3 = -(eps ** 2)
    T4 = -2 * eps * delta_dot_etilde
    V  = 2 * q_eD + q_DD
    T5 = -V

    # 6. Sanity check on the gap identity
    G_check = T1 + T2 + T3 + T4 + T5
    gap_rel_err = ((G - G_check).abs() / (G.abs() + 1e-30)).max().item()

    # 7. The new theory's row-level quantity
    # rho_j = sign(μ_cal^T e_j) * sign(δ^T e_j)  ∈ {-1, +1} (or 0 if either is exactly zero)
    sign_mu_e = torch.sign(mu_cal_dot_e)
    sign_d_e = torch.sign(delta_dot_e)
    rho = sign_mu_e * sign_d_e                           # [out]
    nontrivial = (sign_mu_e != 0) & (sign_d_e != 0)
    n_nontrivial = int(nontrivial.sum().item())
    rho_mean = float(rho[nontrivial].mean().item()) if n_nontrivial > 0 else float("nan")
    frac_rho_positive = float((rho == 1).float().mean().item())

    # 8. The "row-sum proxy" predicted by the theory to track T1 (Prediction N4)
    # T1 = -2 (δ^T e)(δ^T Δ).  By construction, μ_cal^T Δ ≈ -μ_cal^T e (greedy cancels bias),
    # so δ^T Δ has the sign of -δ projection onto μ_cal scaled by μ_cal^T e.
    # The cleanest scalar proxy that tracks sign(T1) per row is:
    #     proxy_j = (μ_cal^T e_j) * (δ^T e_j)
    # because sign(T1_j) ≈ sign((δ^T e_j) * -(δ^T Δ_j)) and δ^T Δ_j inherits a flipped
    # sign from μ_cal^T e_j via the greedy. So large positive proxy_j ↔ large positive T1_j.
    proxy = mu_cal_dot_e * delta_dot_e                   # [out]

    return {
        "name": name,
        "d": in_features,
        "out_features": out_features,
        # Per-row arrays (kept on CPU as fp32 lists for aggregation)
        "rho": rho.cpu().tolist(),
        "G": G.cpu().tolist(),
        "T1": T1.cpu().tolist(),
        "R_BC": R_BC.cpu().tolist(),
        "R_Flip": R_Flip.cpu().tolist(),
        "proxy": proxy.cpu().tolist(),
        "mu_cal_dot_e": mu_cal_dot_e.cpu().tolist(),
        "delta_dot_e": delta_dot_e.cpu().tolist(),
        # Per-layer summaries
        "R_BC_mean": float(R_BC.mean().item()),
        "R_Flip_mean": float(R_Flip.mean().item()),
        "G_layer": float(G.mean().item()),
        "T1_layer": float(T1.mean().item()),
        "T2_layer": float(T2.mean().item()),
        "T3_layer": float(T3.mean().item()),
        "T4_layer": float(T4.mean().item()),
        "T5_layer": float(T5.mean().item()),
        "rho_mean": rho_mean,
        "frac_rho_positive": frac_rho_positive,
        "n_nontrivial_rows": n_nontrivial,
        "gap_identity_max_rel_err": gap_rel_err,
    }


# ============================================================================
# Theory verdict
# ============================================================================
def evaluate_theory(per_layer_results: List[Dict]) -> Dict:
    """
    Tests the three predictions of the new theory:

    N1+N2 (sign agreement): does sign(E_j[ρ_j]) match sign(G_layer) across layers?
    N3 (Spearman rank): how well does ρ_mean predict G_layer across layers?
    N4 (T1 mechanism): does the row-sum proxy track T1_layer with corr > 0.9?

    All correlations are computed across (layer, eval) pairs.
    """
    import scipy.stats as st  # spearmanr; if unavailable, fall back to numpy

    # Aggregate per (layer, eval) pair
    rho_means = np.array([r["rho_mean"] for r in per_layer_results])
    G_layers = np.array([r["G_layer"] for r in per_layer_results])
    T1_layers = np.array([r["T1_layer"] for r in per_layer_results])
    R_BC_means = np.array([r["R_BC_mean"] for r in per_layer_results])

    # N1+N2: per-layer sign agreement (rho vs G)
    valid = ~np.isnan(rho_means)
    sign_rho = np.sign(rho_means[valid])
    sign_G   = np.sign(G_layers[valid])
    sign_agreement = float((sign_rho == sign_G).mean())

    # Restricted to "layers that matter" (top-k by R_BC_mean)
    if len(R_BC_means) >= 4:
        idx_topk = np.argsort(R_BC_means)[-len(R_BC_means)//2:]   # top half by R
        sign_agreement_topR = float((np.sign(rho_means[idx_topk]) == np.sign(G_layers[idx_topk])).mean())
    else:
        sign_agreement_topR = float("nan")

    # N3: Spearman correlation rho_mean vs G_layer
    if valid.sum() >= 3:
        try:
            sp = st.spearmanr(rho_means[valid], G_layers[valid])
            spearman_rho_G = float(sp.correlation)
            spearman_rho_G_p = float(sp.pvalue)
        except Exception:
            spearman_rho_G = float(np.corrcoef(rho_means[valid].argsort().argsort(),
                                                G_layers[valid].argsort().argsort())[0, 1])
            spearman_rho_G_p = float("nan")
    else:
        spearman_rho_G = float("nan")
        spearman_rho_G_p = float("nan")

    # N4: signed-row-sum proxy vs T1, computed within each layer (we use per-row data)
    # The proxy is row-level: proxy_j = (μ_cal^T e_j)(δ^T e_j).
    # We expect sign(proxy_j) to correlate with sign(T1_j) row-by-row,
    # and the magnitude of layer-mean(proxy * 2) should track |T1_layer|.
    n4_per_layer = []
    for r in per_layer_results:
        T1_arr = np.array(r["T1"])
        proxy_arr = np.array(r["proxy"])
        # Pearson within a layer
        if np.std(T1_arr) > 0 and np.std(proxy_arr) > 0:
            pearson = float(np.corrcoef(T1_arr, proxy_arr)[0, 1])
        else:
            pearson = float("nan")
        n4_per_layer.append({
            "name": r["name"],
            "T1_layer": float(np.mean(T1_arr)),
            "proxy_layer": float(np.mean(proxy_arr)),
            "pearson_T1_proxy_within_layer": pearson,
        })

    return {
        "N1_N2_sign_agreement_all": sign_agreement,
        "N1_N2_sign_agreement_topR_half": sign_agreement_topR,
        "N3_spearman_rho_vs_G": spearman_rho_G,
        "N3_spearman_p": spearman_rho_G_p,
        "N4_per_layer": n4_per_layer,
        "n_layers": int(valid.sum()),
        "n_layers_with_rho_positive": int((rho_means[valid] > 0).sum()),
        "n_layers_with_G_positive": int((G_layers[valid] > 0).sum()),
    }


# ============================================================================
# Reporting
# ============================================================================
def print_report(per_layer_results: List[Dict], verdict: Dict):
    print("\n" + "=" * 88)
    print("EGBC THEORY — VERIFICATION REPORT")
    print("=" * 88)
    print("""
The new theory rests on a single mechanistic claim:

    G_layer > 0  ⇔  E_j[ρ_j] > 0    where    ρ_j = sign(μ_cal^T e_j) · sign(δ^T e_j)

Predictions tested:
  N1+N2 — sign agreement between E_j[ρ_j] and G_layer, on all layers AND on
          the top-half-by-R_BC layers (where the action actually is).
  N3   — Spearman rank correlation of E_j[ρ_j] vs G_layer across (layer, eval).
  N4   — Per-layer Pearson correlation of T1_j vs proxy_j = (μ_cal^T e_j)(δ^T e_j),
         which should be ~1 if the mechanism is right.
""")

    # Per-layer table
    print("-" * 88)
    print(f"{'layer':<55s} {'G_layer':>12s} {'E[ρ_j]':>9s} {'%ρ+':>7s} {'R_BC':>10s}")
    print("-" * 88)
    for r in sorted(per_layer_results, key=lambda x: -x["R_BC_mean"]):
        print(f"{r['name']:<55s} {r['G_layer']:+12.3e} {r['rho_mean']:+9.4f}"
              f" {r['frac_rho_positive']*100:6.2f}% {r['R_BC_mean']:10.3e}")

    print()
    print("-" * 88)
    print("PREDICTIONS")
    print("-" * 88)
    print(f"  N1+N2 sign agreement (all layers):       "
          f"{verdict['N1_N2_sign_agreement_all']*100:.1f}%   "
          f"(predicted > 50%; the theory predicts > 80%)")
    print(f"  N1+N2 sign agreement (top half by R_BC): "
          f"{verdict['N1_N2_sign_agreement_topR_half']*100:.1f}%   "
          f"(predicted > 80% — these are the layers that matter)")
    print(f"  N3 Spearman corr(E[ρ_j], G_layer):       "
          f"{verdict['N3_spearman_rho_vs_G']:+.4f}   "
          f"(predicted > +0.5; theory predicts > +0.7)")
    print(f"     p-value: {verdict['N3_spearman_p']:.4f}")
    print()
    print("  N4 (per-layer Pearson corr(T1_j, proxy_j) within each layer):")
    for n4 in verdict["N4_per_layer"]:
        print(f"    {n4['name']:<55s} pearson = {n4['pearson_T1_proxy_within_layer']:+.4f}")

    print()
    n4_corrs = [n["pearson_T1_proxy_within_layer"]
                for n in verdict["N4_per_layer"]
                if not np.isnan(n["pearson_T1_proxy_within_layer"])]
    if n4_corrs:
        print(f"  N4 mean per-layer Pearson: {np.mean(n4_corrs):+.4f}   "
              f"(predicted > +0.9)")

    print()
    print("-" * 88)
    print("OVERALL VERDICT")
    print("-" * 88)
    # Pass / fail logic
    pass_N1N2 = verdict["N1_N2_sign_agreement_topR_half"] >= 0.8
    pass_N3 = (verdict["N3_spearman_rho_vs_G"] > 0.5
               and not np.isnan(verdict["N3_spearman_rho_vs_G"]))
    pass_N4 = (n4_corrs and np.mean(n4_corrs) > 0.9)
    n_pass = sum([pass_N1N2, pass_N3, pass_N4])
    print(f"  N1+N2 (top-R sign agreement ≥ 80%):  {'PASS' if pass_N1N2 else 'FAIL'}")
    print(f"  N3 (Spearman ≥ +0.5):                {'PASS' if pass_N3 else 'FAIL'}")
    print(f"  N4 (mean within-layer Pearson > 0.9): "
          f"{'PASS' if pass_N4 else 'FAIL'}")
    print()
    if n_pass == 3:
        print("  → Theory survives all three predictions on this run.")
    elif n_pass == 2:
        print("  → Theory partially supported. Re-examine the failing prediction.")
    else:
        print("  → Theory fails. The mechanism is not what was claimed.")
    print("=" * 88)


def save_outputs(per_layer_results: List[Dict], verdict: Dict, out_dir: Path, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-layer JSON (omits the per-row arrays for size; keeps summaries)
    summary_only = []
    for r in per_layer_results:
        s = {k: v for k, v in r.items()
             if k not in ("rho", "G", "T1", "R_BC", "R_Flip", "proxy",
                          "mu_cal_dot_e", "delta_dot_e")}
        summary_only.append(s)
    with open(out_dir / "per_layer_summary.json", "w") as f:
        json.dump({"config": cfg, "per_layer": summary_only}, f, indent=2)
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    # Per-row arrays as numpy archive for later plotting
    np.savez(
        out_dir / "per_row_data.npz",
        **{f"{r['name']}__rho": np.array(r["rho"]) for r in per_layer_results},
        **{f"{r['name']}__G": np.array(r["G"]) for r in per_layer_results},
        **{f"{r['name']}__T1": np.array(r["T1"]) for r in per_layer_results},
        **{f"{r['name']}__proxy": np.array(r["proxy"]) for r in per_layer_results},
    )


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cal-dataset", default="c4")
    parser.add_argument("--eval-datasets", nargs="+", default=["c4-val", "wikitext2"])
    parser.add_argument("--n-cal", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--flip-budget-pct", type=float, default=5.0)
    parser.add_argument("--knee-tolerance", type=float, default=0.01)
    parser.add_argument("--model-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-cal-tokens-per-layer", type=int, default=200_000)
    parser.add_argument("--max-eval-tokens-per-layer", type=int, default=200_000)
    parser.add_argument("--flush-every-tokens", type=int, default=16_384)
    parser.add_argument("--layers-pattern", type=str, required=True,
                        help="Comma-separated fnmatch patterns. Recommend ~8 layers "
                             "spread across depth, mixing o_proj and down_proj.")
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="./egbc_theory_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("EGBC THEORY — VERIFICATION CONFIG")
    print("=" * 88)
    print("NOTE: this script REQUIRES full covariance. Do not run with --no-full-cov.")
    for k, v in vars(args).items():
        print(f"  {k:30s} = {v}")
    print("=" * 88)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.model_dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    patterns = [p.strip() for p in args.layers_pattern.split(",") if p.strip()]
    module_names = select_modules(model, patterns, args.max_layers)
    if not module_names:
        print("ERROR: No modules matched.")
        sys.exit(1)
    print(f"\nSelected {len(module_names)} modules.")

    # ---- Capture cal stats (μ only, no full Σ needed for cal) ----
    print(f"\n[1/3] Capturing calibration μ from {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_rec = ActivationRecorder(
        module_names, record_full_cov=False,
        max_tokens_per_module=args.max_cal_tokens_per_layer,
        flush_every_tokens=args.flush_every_tokens,
    )
    run_calibration(model, tokenizer, cal_texts, cal_rec, device, args.max_length)
    cal_stats = cal_rec.finalize()

    # ---- Capture eval stats per dataset (μ AND full Σ) ----
    print(f"\n[2/3] Capturing eval μ and FULL Σ from {args.eval_datasets}")
    print("      (this is the expensive step — full d×d covariance per layer)")
    eval_stats_per_module: Dict[str, Dict[str, Dict]] = {n: {} for n in module_names}
    for eval_name in args.eval_datasets:
        print(f"\n  -- {eval_name} --")
        eval_texts = load_text_samples(eval_name, args.n_eval, args.seed + 1)
        rec = ActivationRecorder(
            module_names, record_full_cov=True,
            max_tokens_per_module=args.max_eval_tokens_per_layer,
            flush_every_tokens=args.flush_every_tokens,
        )
        run_calibration(model, tokenizer, eval_texts, rec, device, args.max_length)
        finalized = rec.finalize()
        for n in module_names:
            eval_stats_per_module[n][eval_name] = finalized[n]
        del rec
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Per-layer measurement ----
    print(f"\n[3/3] Per-layer measurement: G, ρ, T1..T5, proxy")
    per_layer_results: List[Dict] = []
    for name in tqdm(module_names, desc="layers"):
        mod = model.get_submodule(name)
        W = mod.weight.detach()
        for eval_name in args.eval_datasets:
            row = measure_layer(
                name=f"{name}@{eval_name}",
                W_fp=W,
                cal_stats=cal_stats[name],
                eval_stats=eval_stats_per_module[name][eval_name],
                bits=args.bits,
                group_size=args.group_size,
                flip_budget_pct=args.flip_budget_pct,
                knee_tolerance=args.knee_tolerance,
                device=device,
            )
            per_layer_results.append(row)
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Verdict ----
    verdict = evaluate_theory(per_layer_results)
    print_report(per_layer_results, verdict)
    save_outputs(per_layer_results, verdict, Path(args.out_dir), vars(args))
    print(f"\n✅ Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()