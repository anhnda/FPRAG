"""
Verify the two EGBC theorems empirically.

Theorem 1 (Net output-error descent of EGBC).  For each output channel j:
    (i)  Bias descent:        B_j(e'_j)        ≤  B_j(e_j)
    (ii) Variance budget:     |V_j(e'_j) − V_j(e_j)|  ≤  B_j · s_max · (2 ‖Σ e_j‖_∞ + s_max ‖Σ‖_∞)
    (iii) Net descent under dominance:
          if  B_j(e_j) > RHS_of(ii)  then  B_j(e'_j) + V_j(e'_j)  <  B_j(e_j) + V_j(e_j)

Theorem 2 (Universal post-correction).  The same three claims hold when the base
quantizer Q is replaced by AWQ (with L2 salience scaling) instead of plain NTR.

Notation in code (one layer, one eval dataset):
    W       FP weight                                [out, in]
    W_q     dequantized weight from base Q           [out, in]
    e       = W_q − W                                [out, in]
    Δ       EGBC flip in dequantized space           [out, in]
    e_tilde = e + Δ                                  [out, in]
    μ       calibration-set activation mean          [in]
    Σ       eval-set activation covariance           [in, in]
    s_max   = max coord-wise scale step              scalar
    B_j     per-row flip budget (= count of flips actually applied)

The code reports, per row, whether (i), (ii), (iii) hold, and aggregates pass-rates
per (layer, eval, base-quantizer) and globally.  It also reports the slack — i.e.
how strongly the inequalities hold — so you see whether the theorem is tight or loose.

Usage:
    python verify_egbc_theorems.py \
        --model-path ./models/Mistral-7B-v0.3 \
        --cal-dataset c4 --eval-datasets c4-val wikitext2 \
        --n-cal 128 --n-eval 128 --max-length 1024 \
        --bits 4 --group-size 128 --flip-budget-pct 5.0 --knee-tolerance 0.01 \
        --base-quantizers ntr awq \
        --layers-pattern "model.layers.0.self_attn.o_proj,model.layers.4.self_attn.o_proj,model.layers.8.mlp.down_proj,model.layers.12.mlp.down_proj,model.layers.17.self_attn.o_proj,model.layers.23.self_attn.o_proj,model.layers.27.mlp.down_proj,model.layers.31.mlp.down_proj" \
        --out-dir ./egbc_theorem_results

REQUIRES full covariance on the eval side.  Theorem 1(ii)/(iii) cannot be checked
with a diagonal approximation: Σ_ij off-diagonal terms enter both ‖Σ e‖_∞ and the
variance gap.
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
# Base quantizer interface
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_base_quantizer(
    base: str,
    W: torch.Tensor,
    mu_cal: torch.Tensor,
    salience_l2: Optional[torch.Tensor],
    bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return W_q (dequantized), and the integer-space artefacts the flip operator needs.

    For 'awq', we return W_q, W_int, scale_flat, zp_flat already expressed in the
    *scaled* space, plus the per-input-channel scale vector so the caller can map
    everything back.  For 'ntr', the scaled space IS the input space.
    """
    if base == "ntr":
        W_q, W_int, scale_flat, zp_flat = group_quantize(W, bits=bits, group_size=group_size)
        s = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
        return W_q, W_int, scale_flat, zp_flat, s

    if base == "awq":
        assert salience_l2 is not None, "AWQ base needs L2 salience (E[X^2])."
        # Run the L2-AWQ search; this returns the BEST α and the corresponding effective W_q in the ORIGINAL space.
        # We need to RE-RUN the quantization in the scaled space to keep W_int/scale/zp aligned for flipping.
        # awq_search_and_scale returns: W_q_eff (orig space), W_int (scaled), scale_flat (scaled), zp_flat, alpha.
        # And we recompute the chosen scale s = salience^alpha here for clarity.
        salience = salience_l2.to(W.device).clamp(min=1e-5)
        W_q_eff, W_int, scale_flat, zp_flat, best_alpha = awq_search_and_scale(
            W=W, mu_cal=mu_cal, salience_l2=salience,
            bits=bits, group_size=group_size, n_grid=20,
            apply_flip=False,
        )
        s = salience.pow(best_alpha).to(W.dtype)
        # Sanity: W_q_eff ≈ ((W_int - zp) * scale) / s
        return W_q_eff, W_int, scale_flat, zp_flat, s

    raise ValueError(f"Unknown base quantizer: {base}")


# --------------------------------------------------------------------------- #
# Per-layer measurement
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_layer(
    name: str,
    W_fp: torch.Tensor,
    base: str,
    cal_stats: Dict,
    eval_stats: Dict,
    bits: int,
    group_size: int,
    flip_budget_pct: float,
    knee_tolerance: float,
    use_james_stein: bool,
    device: torch.device,
    mode: str = "insample",
) -> Dict[str, object]:
    """Run the base quantizer, apply EGBC, and check Theorem 1 (i)(ii)(iii) per row.

    mode='insample' :  μ in the theorem is the calibration μ (the one EGBC used).
                       This is what Theorem 1 literally states.
                       Σ still comes from the eval set, since the eval set defines
                       the deployment distribution we care about for V_j.
    mode='crossval' :  μ in the theorem is the eval-set μ (different from cal).
                       This is a robustness check: does the flip designed against
                       μ_cal still reduce bias when measured against μ_eval?
                       NOT what Theorem 1 states; reported separately.
    """
    W = W_fp.to(device).float()
    out_features, in_features = W.shape

    # ---- means and covariance ----------------------------------------------
    mu_cal_raw = cal_stats["mu"].to(device).double()
    mu_cal = james_stein_mean(mu_cal_raw) if use_james_stein else mu_cal_raw
    mu_eval = eval_stats["mu"].to(device).double()
    Sigma = eval_stats["Sigma"]
    if Sigma is None:
        raise RuntimeError(f"Layer {name}: full covariance required for theorem check.")
    Sigma = Sigma.to(device).double()

    # The μ we EVALUATE Theorem 1 against:
    if mode == "insample":
        mu_for_eval = mu_cal
    elif mode == "crossval":
        mu_for_eval = mu_eval
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- salience (E[X^2]) ≈ μ^2 + diag(Σ) ---------------------------------
    diag_Sigma_cal = cal_stats.get("diag_second", None)
    if diag_Sigma_cal is not None:
        salience = diag_Sigma_cal.to(device).float()
    else:
        # Use eval-side as a reasonable proxy.  Honest-cross would be cal-side; both are tested in practice.
        salience = (mu_eval ** 2 + torch.diagonal(Sigma)).float()

    # ---- run base Q (in float for clean arithmetic) ------------------------
    W_q_eff, W_int, scale_flat, zp_flat, s_vec = run_base_quantizer(
        base=base, W=W, mu_cal=mu_cal.float(), salience_l2=salience,
        bits=bits, group_size=group_size,
    )
    W_q_eff = W_q_eff.float()
    e = (W_q_eff - W).double()

    # ---- EGBC flip in dequantized ORIGINAL space ---------------------------
    # In AWQ, the flip operates in the scaled space; map μ to the scaled space before flipping,
    # then map the resulting Δ back via /s.
    mu_for_flip = (mu_cal.float() / s_vec).float() if base == "awq" else mu_cal.float()
    Delta_scaled = compute_flip_delta(
        W=W * s_vec.unsqueeze(0) if base == "awq" else W,
        W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_for_flip, bits=bits,
        flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
    ).double()
    if base == "awq":
        Delta = (Delta_scaled / s_vec.unsqueeze(0).double())
    else:
        Delta = Delta_scaled
    e_tilde = e + Delta

    # ---- per-row quantities for Theorem 1 ----------------------------------
    #   bias before / after — evaluated against mu_for_eval per the mode setting
    bias_before = (e @ mu_for_eval) ** 2              # [out]
    bias_after = (e_tilde @ mu_for_eval) ** 2         # [out]

    #   variance before / after (these are the EXPENSIVE quadratic forms)
    Se = e @ Sigma                                    # [out, in]
    var_before = (Se * e).sum(dim=1)                  # [out]
    Set = e_tilde @ Sigma
    var_after = (Set * e_tilde).sum(dim=1)            # [out]

    #   variance budget RHS:  2 B_j s_max ‖Σ e_j‖∞  +  B_j² s_max² ‖Σ‖∞
    # Derivation:   |ΔV_j|  =  |2 Δ_jᵀ Σ e_j  +  Δ_jᵀ Σ Δ_j|
    #              ≤  2 ‖Δ_j‖_1 ‖Σ e_j‖_∞  +  ‖Δ_j‖_1² ‖Σ‖_∞
    #              ≤  2 B_j s_max ‖Σ e_j‖_∞  +  B_j² s_max² ‖Σ‖_∞
    # using ‖Δ_j‖_1 ≤ |S_j|·‖Δ_j‖_∞ ≤ B_j · s_max.
    # NOTE on AWQ space: the flip operates in the scaled space; Δ in original
    # space is Δ_scaled / s_vec, so the per-coord step in the space where Σ
    # acts is scale_flat / s_vec.  s_max must be taken in that same space.
    if base == "awq":
        step_orig = (scale_flat.to(torch.float64) / s_vec.unsqueeze(0).to(torch.float64))
    else:
        step_orig = scale_flat.to(torch.float64)
    s_max = float(step_orig.max().item())
    Sigma_inf = float(Sigma.abs().max().item())                     # entry-wise ∞-norm
    Se_inf_per_row = Se.abs().max(dim=1).values                     # [out]

    #   actual budget used per row (flips actually applied)
    B_per_row = (Delta != 0).sum(dim=1).double()                    # [out]
    var_rhs = 2.0 * B_per_row * s_max * Se_inf_per_row \
              + (B_per_row ** 2) * (s_max ** 2) * Sigma_inf

    #   total channel error before/after
    total_before = bias_before + var_before
    total_after = bias_after + var_after

    # ---- Theorem 1 checks --------------------------------------------------
    eps_num = 1e-10  # realistic tolerance for float32 weights / float64 accumulators

    # (i) Bias descent
    pass_i_mask = bias_after <= bias_before + eps_num * (1 + bias_before)
    bias_strict_decrease = bias_after < bias_before - eps_num * (1 + bias_before)
    bias_no_change = (B_per_row == 0)
    pass_i = float(pass_i_mask.float().mean().item())

    # (ii) Variance budget
    var_change = (var_after - var_before).abs()
    pass_ii_mask = var_change <= var_rhs + eps_num * (1 + var_before)
    pass_ii = float(pass_ii_mask.float().mean().item())
    # slack (positive = the bound is not tight; large positive = very loose bound)
    var_slack = (var_rhs - var_change)

    # (iii) Net descent under dominance condition  bias_before > RHS
    dom_mask = bias_before > var_rhs
    n_dom = int(dom_mask.sum().item())
    if n_dom > 0:
        pass_iii_mask = (total_after < total_before)[dom_mask]
        pass_iii = float(pass_iii_mask.float().mean().item())
    else:
        pass_iii = float("nan")

    # ---- gap statistics ----------------------------------------------------
    bias_gap = (bias_before - bias_after).clamp(min=0)
    bias_gap_mean = float(bias_gap.mean().item())
    bias_gap_max = float(bias_gap.max().item())

    var_gap = var_after - var_before                                # signed
    var_gap_mean = float(var_gap.mean().item())
    var_gap_max_abs = float(var_gap.abs().max().item())

    total_gap = (total_before - total_after)
    total_gap_mean = float(total_gap.mean().item())

    return {
        "name": name,
        "base": base,
        "mode": mode,
        "n_rows": int(out_features),
        "n_flipped_rows": int((B_per_row > 0).sum().item()),
        "B_mean": float(B_per_row.mean().item()),
        "B_max": int(B_per_row.max().item()),
        "s_max": s_max,
        "Sigma_inf": Sigma_inf,
        # Theorem 1 outcomes
        "T1_i_pass_rate": pass_i,
        "T1_i_strict_decrease_rate": float(bias_strict_decrease.float().mean().item()),
        "T1_i_no_change_rate": float(bias_no_change.float().mean().item()),
        "T1_ii_pass_rate": pass_ii,
        "T1_ii_slack_mean": float(var_slack.mean().item()),
        "T1_ii_slack_min": float(var_slack.min().item()),
        "T1_iii_dominance_rows": n_dom,
        "T1_iii_pass_rate": pass_iii,
        # Gap magnitudes
        "bias_before_mean": float(bias_before.mean().item()),
        "bias_after_mean": float(bias_after.mean().item()),
        "var_before_mean": float(var_before.mean().item()),
        "var_after_mean": float(var_after.mean().item()),
        "total_before_mean": float(total_before.mean().item()),
        "total_after_mean": float(total_after.mean().item()),
        "bias_gap_mean": bias_gap_mean,
        "bias_gap_max": bias_gap_max,
        "var_gap_mean": var_gap_mean,
        "var_gap_max_abs": var_gap_max_abs,
        "total_gap_mean": total_gap_mean,
        # Per-row arrays for later inspection
        "_per_row": {
            "bias_before": bias_before.cpu().numpy(),
            "bias_after": bias_after.cpu().numpy(),
            "var_before": var_before.cpu().numpy(),
            "var_after": var_after.cpu().numpy(),
            "var_rhs": var_rhs.cpu().numpy(),
            "B": B_per_row.cpu().numpy(),
        },
    }


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def evaluate_theorems(results: List[Dict]) -> Dict:
    """Aggregate per-(base, mode) results into a verdict.

    Theorem 1's literal claim is the 'insample' mode (μ in the theorem is the
    same μ the flip operator used).  'crossval' mode reports robustness to
    cal-eval distribution shift; it is informative but NOT what the theorem states.
    """
    by_key: Dict[Tuple[str, str], List[Dict]] = {}
    for r in results:
        by_key.setdefault((r["base"], r.get("mode", "insample")), []).append(r)

    summary: Dict[str, object] = {}
    for (base, mode), rs in by_key.items():
        T1_i = float(np.mean([r["T1_i_pass_rate"] for r in rs]))
        T1_ii = float(np.mean([r["T1_ii_pass_rate"] for r in rs]))
        iii_rates = [r["T1_iii_pass_rate"] for r in rs if r["T1_iii_dominance_rows"] > 0]
        T1_iii = float(np.mean(iii_rates)) if iii_rates else float("nan")
        dom_layer_frac = float(np.mean([1.0 if r["T1_iii_dominance_rows"] > 0 else 0.0 for r in rs]))
        n_rows_total = sum(r["n_rows"] for r in rs)
        n_pass_i = sum(int(r["T1_i_pass_rate"] * r["n_rows"]) for r in rs)
        n_pass_ii = sum(int(r["T1_ii_pass_rate"] * r["n_rows"]) for r in rs)
        strict_dec_rate = float(np.mean([r["T1_i_strict_decrease_rate"] for r in rs]))
        no_change_rate = float(np.mean([r["T1_i_no_change_rate"] for r in rs]))

        summary[f"{base}::{mode}"] = {
            "base": base,
            "mode": mode,
            "T1_i_layer_avg":   T1_i,
            "T1_ii_layer_avg":  T1_ii,
            "T1_iii_layer_avg": T1_iii,
            "T1_iii_layer_frac_with_dominance": dom_layer_frac,
            "T1_i_row_total_pass_rate":  n_pass_i / max(n_rows_total, 1),
            "T1_ii_row_total_pass_rate": n_pass_ii / max(n_rows_total, 1),
            "T1_i_strict_decrease_rate": strict_dec_rate,
            "T1_i_no_change_rate":       no_change_rate,
            "n_layers": len(rs),
            "n_rows_total": n_rows_total,
        }

    # Theorem 2: literal claim is the 'insample' case across all bases.
    insample_summaries = [s for s in summary.values() if s["mode"] == "insample"]
    if insample_summaries:
        T2_pass = all(
            s["T1_i_layer_avg"] >= 0.999 and s["T1_ii_layer_avg"] >= 0.999 and
            (np.isnan(s["T1_iii_layer_avg"]) or s["T1_iii_layer_avg"] >= 0.95)
            for s in insample_summaries
        )
    else:
        T2_pass = False

    return {
        "by_key": summary,
        "T2_universal_pass": T2_pass,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(results: List[Dict], verdict: Dict):
    print("\n" + "=" * 96)
    print("EGBC THEOREM VERIFICATION REPORT")
    print("=" * 96)
    print("""\
Theorem 1 (per-row, per-layer):
  (i)   B_j(e') ≤ B_j(e)
  (ii)  |V_j(e') − V_j(e)| ≤ 2 B_j s_max ‖Σ e_j‖∞ + B_j² s_max² ‖Σ‖∞
  (iii) if B_j(e) > RHS_ii  then  B_j(e') + V_j(e') < B_j(e) + V_j(e)

Theorem 2: claims (i)(ii)(iii) hold for any lattice-valued base quantizer Q.
We verify across multiple Q ∈ {NTR, AWQ}.

Modes:
  insample — μ in the theorem = μ EGBC used (THE THEOREM)
  crossval — μ in the theorem = eval-set μ (robustness, NOT the theorem)
""")

    # Per-row table (base, mode, layer)
    print("-" * 96)
    print(f"{'layer@eval':<55s} {'base':<5s} {'mode':<9s} "
          f"{'T1(i)%':>7s} {'T1(ii)%':>8s} {'T1(iii)%':>9s} {'flip_rows':>10s}")
    print("-" * 96)
    for r in sorted(results, key=lambda x: (x["base"], x.get("mode","-"), x["name"])):
        iii = r["T1_iii_pass_rate"]
        iii_str = "  n/a   " if np.isnan(iii) else f"{iii*100:8.2f}"
        print(f"{r['name']:<55s} {r['base']:<5s} {r.get('mode','-'):<9s} "
              f"{r['T1_i_pass_rate']*100:6.2f} {r['T1_ii_pass_rate']*100:7.2f} "
              f"{iii_str:>9s} {r['n_flipped_rows']:>10d}")

    print()
    print("-" * 96)
    print("THEOREM 1 — aggregate (per base × mode):")
    print("-" * 96)
    # Insample first, crossval second
    keys_ordered = sorted(verdict["by_key"].keys(), key=lambda k: (k.split("::")[1] != "insample", k))
    for key in keys_ordered:
        s = verdict["by_key"][key]
        is_theorem = (s["mode"] == "insample")
        tag = "[THEOREM]   " if is_theorem else "[robustness]"
        print(f"  {tag}  base = {s['base']:<5s}  mode = {s['mode']}")
        print(f"    T1(i)  layer-avg pass rate:                  {s['T1_i_layer_avg']*100:7.3f}%   "
              + ("(predicted: 100.000%)" if is_theorem else "(generalisation; theorem mute)"))
        print(f"    T1(ii) layer-avg pass rate:                  {s['T1_ii_layer_avg']*100:7.3f}%   "
              + ("(predicted: 100.000%)" if is_theorem else "(generalisation; theorem mute)"))
        if np.isnan(s["T1_iii_layer_avg"]):
            print(f"    T1(iii) no layer met the dominance condition")
        else:
            print(f"    T1(iii) layer-avg pass rate (where dom.):    {s['T1_iii_layer_avg']*100:7.3f}%   "
                  + ("(predicted: 100.000%)" if is_theorem else "(generalisation)"))
            print(f"    T1(iii) fraction of layers with dominance:   {s['T1_iii_layer_frac_with_dominance']*100:7.2f}%")
        print(f"    Rows with STRICT bias decrease:              {s['T1_i_strict_decrease_rate']*100:7.3f}%")
        print(f"    Rows with NO flips (bias unchanged):         {s['T1_i_no_change_rate']*100:7.3f}%")
        print()

    print("-" * 96)
    print("THEOREM 2 — universality across base quantizers (literal claim, insample only):")
    print("-" * 96)
    insample_keys = [k for k in verdict["by_key"] if k.endswith("::insample")]
    insample_bases = [k.split("::")[0] for k in insample_keys]
    print(f"    Verified for base quantizers: {insample_bases}")
    print(f"    Universal pass: {'PASS' if verdict['T2_universal_pass'] else 'FAIL'}")
    print("=" * 96)

    print("""\
Notes on interpretation:
  - INSAMPLE rows test the literal theorem.  Expected: T1(i)/T1(ii) at 100%.
    Any failure here is a bug.
  - CROSSVAL rows test robustness: does a flip designed against μ_cal still
    reduce bias when measured against the deployment μ_eval?  T1(i) below
    100% in this row is generalisation gap, NOT a theorem violation.
  - T1(iii) is conditional.  Reported only over rows where the dominance
    condition holds.  Low dominance fraction => the theorem trivially says
    "no improvement available" for those rows.
""")


def save_outputs(results: List[Dict], verdict: Dict, out_dir: Path, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Strip the per-row arrays from JSON and dump them separately as .npz
    summary = []
    for r in results:
        s = {k: v for k, v in r.items() if k != "_per_row"}
        summary.append(s)
    with open(out_dir / "per_layer_summary.json", "w") as f:
        json.dump({"config": cfg, "per_layer": summary}, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o))
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o))
    # Per-row data
    archive: Dict[str, np.ndarray] = {}
    for r in results:
        key = f"{r['base']}__{r['name'].replace('@','_AT_').replace('.', '_')}"
        for sub_k, sub_v in r["_per_row"].items():
            archive[f"{key}__{sub_k}"] = sub_v
    np.savez(out_dir / "per_row_data.npz", **archive)


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
    p.add_argument("--base-quantizers", nargs="+", default=["ntr", "awq"],
                   choices=["ntr", "awq"])
    p.add_argument("--mode", choices=["insample", "crossval", "both"], default="both",
                   help="insample: evaluate Theorem 1 against the same μ EGBC used "
                        "(this is what the theorem literally states). "
                        "crossval: use eval-set μ; tests robustness to cal-eval shift. "
                        "both: report both side-by-side (default).")
    p.add_argument("--model-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max-cal-tokens-per-layer", type=int, default=200_000)
    p.add_argument("--max-eval-tokens-per-layer", type=int, default=200_000)
    p.add_argument("--flush-every-tokens", type=int, default=16_384)
    p.add_argument("--layers-pattern", type=str, required=True,
                   help="Comma-separated fnmatch patterns.")
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="./egbc_theorem_results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 96)
    print("EGBC THEOREM VERIFICATION — CONFIG")
    print("=" * 96)
    print("FULL covariance IS REQUIRED on eval side.")
    for k, v in vars(args).items():
        print(f"  {k:32s} = {v}")
    print("=" * 96)

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
        print("ERROR: no modules matched the pattern.")
        sys.exit(1)
    print(f"\nSelected {len(module_names)} modules:")
    for n in module_names:
        print(f"  - {n}")

    # ---- 1) Calibration μ (no full Σ) ----------------------------------------
    print(f"\n[1/3] Capturing calibration μ from {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_rec = ActivationRecorder(
        module_names, record_full_cov=False,
        max_tokens_per_module=args.max_cal_tokens_per_layer,
        flush_every_tokens=args.flush_every_tokens,
    )
    run_calibration(model, tok, cal_texts, cal_rec, device, args.max_length)
    cal_stats = cal_rec.finalize()
    # Diagonal of second moment as a quick salience estimate (used for AWQ scaling).
    # If you want a true E[X^2], the recorder can be extended; this approximation is
    # standard in 7B-class AWQ tooling.

    # ---- 2) Eval μ and FULL Σ -----------------------------------------------
    print(f"\n[2/3] Capturing eval μ and FULL Σ from {args.eval_datasets}")
    print("      (expensive: per-layer d×d covariance)")
    eval_stats_per_module: Dict[str, Dict[str, Dict]] = {n: {} for n in module_names}
    for ev in args.eval_datasets:
        print(f"\n  -- {ev} --")
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

    # ---- 3) Per-(layer, eval, base, mode) measurement -----------------------
    if args.mode == "both":
        modes = ["insample", "crossval"]
    else:
        modes = [args.mode]

    print(f"\n[3/3] Theorem 1 checks per (layer × eval × base × mode), modes={modes}")
    results: List[Dict] = []
    for name in tqdm(module_names, desc="layers"):
        mod = model.get_submodule(name)
        W = mod.weight.detach()
        for ev in args.eval_datasets:
            for base in args.base_quantizers:
                for mode in modes:
                    row = measure_layer(
                        name=f"{name}@{ev}",
                        W_fp=W, base=base,
                        cal_stats=cal_stats[name],
                        eval_stats=eval_stats_per_module[name][ev],
                        bits=args.bits, group_size=args.group_size,
                        flip_budget_pct=args.flip_budget_pct,
                        knee_tolerance=args.knee_tolerance,
                        use_james_stein=args.use_james_stein,
                        device=device,
                        mode=mode,
                    )
                    row["mode"] = mode
                    results.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    verdict = evaluate_theorems(results)
    print_report(results, verdict)
    save_outputs(results, verdict, Path(args.out_dir), vars(args))
    print(f"\n✅ Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()