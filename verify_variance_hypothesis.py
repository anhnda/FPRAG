"""
Test the variance hypothesis from the EGBC theory discussion.

Hypothesis (from Insight 2 in the previous analysis):
    BC's correction is a per-row constant -> low MEAN error, HIGH per-token VARIANCE.
    Flip's correction is input-dependent -> nonzero MEAN error, LOW per-token VARIANCE.
    PPL is more sensitive to per-token variance than to mean squared error,
    so Flip wins on PPL while losing on mean MSE.

For each layer and each eval distribution, we measure:
    Y_BC,j(x)    = x^T e_j        - mu_cal^T e_j        (input-dependent error after BC)
    Y_Flip,j(x)  = x^T (e_j + Delta_j)                  (input-dependent error after Flip)

Then per output row j, on the eval batch:
    R_BC,j     = E_x[Y_BC,j(x)^2]      (already in main verification, recomputed here as sanity)
    Var_BC,j   = Var_x[Y_BC,j(x)^2]    = E[Y^4] - (E[Y^2])^2
    same for Flip

Hypothesis predicts Var_BC,j > Var_Flip,j on rows where Flip wins end-to-end.

Pipeline:
    1. Load model, quantize each selected layer (matching verify_flip_vs_bc_theory).
    2. For each layer, capture cal stats once (forward over cal_texts).
    3. Compute Delta_j from mu_cal (paper-aligned greedy flip).
    4. For each eval distribution, run a fresh forward and stream-accumulate
       per-row 2nd and 4th moments of (x . v_j). NO raw token storage.
    5. Output: per-layer-per-eval table with R, Var of squared error, and
       the variance gap V_BC - V_Flip.

Speed: same wall-clock order as the main verify script, since we reuse the
same forward pass + hook structure. The added cost per hooked module is
4 reductions over a [batch, out_features] tensor, all on GPU in fp32.

Usage:
    python verify_variance_hypothesis.py \
        --model-path ./models/Mistral-7B-v0.3 \
        --cal-dataset c4 --eval-datasets c4-val wikitext2 \
        --n-cal 128 --n-eval 128 \
        --max-length 1024 \
        --bits 4 --group-size 128 \
        --flip-budget-pct 5.0 --knee-tolerance 0.01 \
        --layers-pattern "model.layers.*.mlp.down_proj,model.layers.*.self_attn.o_proj" \
        --max-layers 16 \
        --out-dir ./variance_results/mistral
"""

import argparse
import fnmatch
import gc
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Reuse plumbing from the main verification script: it's the SOLE source of
# truth for the flip algorithm, dataset loading, and module selection.
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
# Per-row per-token error moments — captured during a forward pass
# ============================================================================
class ErrorMomentRecorder:
    """
    For each selected layer, hooks the layer's input x and computes the per-token
    BC-corrected and Flip-corrected output errors, then accumulates the 2nd and
    4th central moments per output row.

    Important: we need W, W_q, Delta, and mu_cal to be pre-computed and stored
    BEFORE this recorder runs. The hook does only forward GEMMs and reductions.
    No raw tokens are stored.

    Per output row j, we track on GPU (fp32):
        n         : token count
        sum_Y2_BC : sum_t Y_BC(x_t,j)^2
        sum_Y4_BC : sum_t Y_BC(x_t,j)^4
        sum_Y2_FL : same for Flip
        sum_Y4_FL : same for Flip

    After the eval forward pass, finalize() returns per-row means E[Y^2] and
    variances Var[Y^2] = E[Y^4] - E[Y^2]^2 for each row, each method.
    """

    def __init__(
        self,
        module_payloads: Dict[str, Dict],
        max_tokens_per_module: int = 200_000,
        flush_every_tokens: int = 16_384,
    ):
        """
        module_payloads[name] is a dict with keys:
            "e"       : [out, in] fp32 on GPU. W_q - W.
            "Delta"   : [out, in] fp32 on GPU. Flip correction.
            "mu_cal_dot_e" : [out] fp32 on GPU. -BC bias per row.
                            (BC's correction is c_BC,j = -mu_cal^T e_j;
                             we precompute this so the hook does not.)
        """
        self.payloads = module_payloads
        self.max_tokens_per_module = max_tokens_per_module
        self.flush_every_tokens = flush_every_tokens
        self.stats: Dict[str, Dict] = {}
        self._handles = []

    def _ensure_init(self, name: str, out_features: int, device):
        if name in self.stats:
            return
        self.stats[name] = {
            "n": 0,
            "buf_n": 0,
            # CPU fp64 totals
            "sum_Y2_BC_cpu": torch.zeros(out_features, dtype=torch.float64),
            "sum_Y4_BC_cpu": torch.zeros(out_features, dtype=torch.float64),
            "sum_Y2_FL_cpu": torch.zeros(out_features, dtype=torch.float64),
            "sum_Y4_FL_cpu": torch.zeros(out_features, dtype=torch.float64),
            # GPU fp32 working buffers
            "buf_sum_Y2_BC": torch.zeros(out_features, dtype=torch.float32, device=device),
            "buf_sum_Y4_BC": torch.zeros(out_features, dtype=torch.float32, device=device),
            "buf_sum_Y2_FL": torch.zeros(out_features, dtype=torch.float32, device=device),
            "buf_sum_Y4_FL": torch.zeros(out_features, dtype=torch.float32, device=device),
            "out_features": out_features,
            "device": device,
        }

    def _flush(self, entry: Dict):
        if entry["buf_n"] == 0:
            return
        for k in ("sum_Y2_BC", "sum_Y4_BC", "sum_Y2_FL", "sum_Y4_FL"):
            entry[f"{k}_cpu"] += entry[f"buf_{k}"].double().cpu()
            entry[f"buf_{k}"].zero_()
        entry["n"] += entry["buf_n"]
        entry["buf_n"] = 0

    def _hook(self, name: str):
        payload = self.payloads.get(name)
        if payload is None:
            return None
        e = payload["e"]                         # [out, in]
        Delta = payload["Delta"]                 # [out, in]
        mu_cal_dot_e = payload["mu_cal_dot_e"]   # [out]

        def hook(_module, inputs, _output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.detach().float()

            entry = self.stats.get(name)
            if entry is None:
                self._ensure_init(name, e.shape[0], x.device)
                entry = self.stats[name]

            already = entry["n"] + entry["buf_n"]
            remaining = self.max_tokens_per_module - already
            if remaining <= 0:
                return
            if x.shape[0] > remaining:
                idx = torch.randperm(x.shape[0], device=x.device)[:remaining]
                x = x[idx]

            # Y_BC(x, j)   = x^T e_j   - mu_cal^T e_j
            # Y_Flip(x, j) = x^T (e_j + Delta_j)
            # Compute both batched: [B, out]
            # Note: e and Delta are [out, in], so x @ e.T = [B, in] @ [in, out] = [B, out]
            Y_BC = x @ e.t() - mu_cal_dot_e.unsqueeze(0)            # [B, out]
            Y_FL = x @ (e + Delta).t()                              # [B, out]

            Y_BC_sq = Y_BC * Y_BC
            Y_FL_sq = Y_FL * Y_FL

            entry["buf_n"] += x.shape[0]
            entry["buf_sum_Y2_BC"] += Y_BC_sq.sum(dim=0)
            entry["buf_sum_Y4_BC"] += (Y_BC_sq * Y_BC_sq).sum(dim=0)
            entry["buf_sum_Y2_FL"] += Y_FL_sq.sum(dim=0)
            entry["buf_sum_Y4_FL"] += (Y_FL_sq * Y_FL_sq).sum(dim=0)

            if entry["buf_n"] >= self.flush_every_tokens:
                self._flush(entry)
        return hook

    def attach(self, model: nn.Module):
        self._handles = []
        for name, _ in self.payloads.items():
            try:
                mod = model.get_submodule(name)
            except AttributeError:
                continue
            if not isinstance(mod, nn.Linear):
                continue
            hook = self._hook(name)
            if hook is None:
                continue
            self._handles.append(mod.register_forward_hook(hook))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def finalize(self) -> Dict[str, Dict]:
        """Returns per-module: mean Y^2 (= R), var of Y^2, for each method."""
        out = {}
        for name, e in self.stats.items():
            self._flush(e)
            n = max(e["n"], 1)
            EY2_BC = e["sum_Y2_BC_cpu"] / n
            EY4_BC = e["sum_Y4_BC_cpu"] / n
            EY2_FL = e["sum_Y2_FL_cpu"] / n
            EY4_FL = e["sum_Y4_FL_cpu"] / n
            # Var(Y^2) = E[Y^4] - E[Y^2]^2.   Clamp to zero in case of fp drift.
            Var_BC = (EY4_BC - EY2_BC * EY2_BC).clamp(min=0.0)
            Var_FL = (EY4_FL - EY2_FL * EY2_FL).clamp(min=0.0)
            out[name] = {
                "n_tokens": n,
                "EY2_BC": EY2_BC, "EY4_BC": EY4_BC, "VarY2_BC": Var_BC,
                "EY2_FL": EY2_FL, "EY4_FL": EY4_FL, "VarY2_FL": Var_FL,
                "out_features": e["out_features"],
            }
            # Free GPU buffers
            for k in ("buf_sum_Y2_BC", "buf_sum_Y4_BC",
                      "buf_sum_Y2_FL", "buf_sum_Y4_FL"):
                if k in e:
                    del e[k]
        torch.cuda.empty_cache()
        return out


# ============================================================================
# Per-layer setup: quantize, compute Delta, build payload for the moment recorder
# ============================================================================
@torch.no_grad()
def build_layer_payloads(
    model: nn.Module,
    module_names: List[str],
    cal_stats: Dict[str, Dict],
    bits: int,
    group_size: int,
    flip_budget_pct: float,
    knee_tolerance: float,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    For each module, compute e = W_q - W, Delta from greedy flip, and
    mu_cal^T e per row. Keep them on GPU in fp32 for use during the eval forward.
    """
    payloads = {}
    for name in tqdm(module_names, desc="quantize + flip"):
        mod = model.get_submodule(name)
        W_orig = mod.weight.detach().to(device).float()
        mu_cal = cal_stats[name]["mu"].to(device).float()

        W_q, W_int, scale_flat, zp_flat = group_quantize(
            W_orig, bits=bits, group_size=group_size
        )
        e = (W_q - W_orig)

        Delta = compute_flip_delta(
            W=W_orig, W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
            mu_cal=mu_cal, bits=bits,
            flip_budget_pct=flip_budget_pct,
            knee_tolerance=knee_tolerance,
        )

        mu_cal_dot_e = e @ mu_cal   # [out]
        payloads[name] = {
            "e": e, "Delta": Delta, "mu_cal_dot_e": mu_cal_dot_e,
        }
        # We do NOT modify the model's weights; eval forward uses the original
        # FP weights and we hook the *inputs* to compute corrected errors.
        del W_q, W_int, scale_flat, zp_flat, W_orig
        torch.cuda.empty_cache()
    return payloads


# ============================================================================
# Reporting
# ============================================================================
@dataclass
class VarRow:
    layer_name: str
    eval_name: str
    n_eval_tokens: int
    out_features: int

    R_BC: float                # mean E_x[Y_BC^2] averaged over rows j
    R_Flip: float
    G_layer: float             # R_BC - R_Flip  (positive = Flip wins on MSE)

    VarY2_BC: float            # mean Var_x[Y_BC^2] averaged over rows j
    VarY2_Flip: float
    VarGap_layer: float        # VarY2_BC - VarY2_Flip  (positive = Flip wins on variance)

    # Per-row signs: fraction of rows where Flip beats BC on each metric
    frac_rows_G_positive: float        # rows where R_BC > R_Flip
    frac_rows_VarGap_positive: float   # rows where VarY2_BC > VarY2_Flip
    frac_rows_BothFlip_wins: float     # both metrics favor Flip


def summarize(per_layer_per_eval: Dict[str, Dict[str, Dict]]) -> Dict[str, dict]:
    """
    per_layer_per_eval[layer_name][eval_name] = dict from ErrorMomentRecorder.finalize()
    Returns a list of VarRow dicts.
    """
    rows: List[VarRow] = []
    for layer_name, evals in per_layer_per_eval.items():
        for eval_name, m in evals.items():
            EY2_BC = m["EY2_BC"]
            EY2_FL = m["EY2_FL"]
            VarBC = m["VarY2_BC"]
            VarFL = m["VarY2_FL"]
            row_G = EY2_BC - EY2_FL
            row_VG = VarBC - VarFL
            rows.append(VarRow(
                layer_name=layer_name,
                eval_name=eval_name,
                n_eval_tokens=int(m["n_tokens"]),
                out_features=int(m["out_features"]),
                R_BC=float(EY2_BC.mean()),
                R_Flip=float(EY2_FL.mean()),
                G_layer=float(row_G.mean()),
                VarY2_BC=float(VarBC.mean()),
                VarY2_Flip=float(VarFL.mean()),
                VarGap_layer=float(row_VG.mean()),
                frac_rows_G_positive=float((row_G > 0).float().mean()),
                frac_rows_VarGap_positive=float((row_VG > 0).float().mean()),
                frac_rows_BothFlip_wins=float(((row_G > 0) & (row_VG > 0)).float().mean()),
            ))

    # Aggregate by eval_name
    summary = {}
    by_eval = {}
    for r in rows:
        by_eval.setdefault(r.eval_name, []).append(r)
    for eval_name, rs in by_eval.items():
        # Cross-layer correlation between G_layer and VarGap_layer:
        # if positive, the layers where Flip wins MSE are also the ones
        # where it wins variance.
        Gs = np.array([r.G_layer for r in rs])
        VGs = np.array([r.VarGap_layer for r in rs])
        if len(rs) >= 3 and np.std(Gs) > 0 and np.std(VGs) > 0:
            corr_G_VG = float(np.corrcoef(Gs, VGs)[0, 1])
        else:
            corr_G_VG = float("nan")
        # Hypothesis check: is mean(VarGap) > 0 even when mean(G) < 0?
        mean_G = float(np.mean(Gs))
        mean_VG = float(np.mean(VGs))
        # By module type
        types = {}
        for r in rs:
            t = ("o_proj" if "o_proj" in r.layer_name
                 else ("down_proj" if "down_proj" in r.layer_name
                       else "other"))
            types.setdefault(t, []).append(r)
        per_type = {}
        for t, rrows in types.items():
            per_type[t] = {
                "n": len(rrows),
                "mean_G": float(np.mean([r.G_layer for r in rrows])),
                "mean_VarGap": float(np.mean([r.VarGap_layer for r in rrows])),
                "frac_layers_Flip_wins_MSE": float(np.mean([r.G_layer > 0 for r in rrows])),
                "frac_layers_Flip_wins_Var": float(np.mean([r.VarGap_layer > 0 for r in rrows])),
            }

        summary[eval_name] = {
            "n_layers": len(rs),
            "mean_G_layer": mean_G,
            "mean_VarGap_layer": mean_VG,
            "corr_G_VarGap": corr_G_VG,
            "per_module_type": per_type,
        }
    return summary, rows


def print_summary(summary: Dict, rows: List[VarRow]):
    print("\n" + "=" * 80)
    print("VARIANCE HYPOTHESIS — SUMMARY")
    print("=" * 80)
    print("""
Hypothesis recap:
  - BC error per token:   Y_BC(x,j)   = x^T e_j   - mu_cal^T e_j
  - Flip error per token: Y_Flip(x,j) = x^T (e_j + Delta_j)
  - Verification measures R = mean of Y^2 (already in main script).
  - Hypothesis: BC has higher VARIANCE of Y^2 across tokens than Flip,
    even when BC has lower MEAN of Y^2. If true, PPL prefers Flip.

Signs:
  G_layer       = mean_j (R_BC,j - R_Flip,j)        > 0 means Flip wins MSE
  VarGap_layer  = mean_j (Var_BC,j - Var_Flip,j)    > 0 means Flip wins variance
""")
    for eval_name, s in summary.items():
        print(f"\n[eval = {eval_name}]   ({s['n_layers']} layers)")
        print(f"  mean G_layer       = {s['mean_G_layer']:+.4e}   "
              f"({'Flip wins MSE' if s['mean_G_layer']>0 else 'BC wins MSE'})")
        print(f"  mean VarGap_layer  = {s['mean_VarGap_layer']:+.4e}   "
              f"({'Flip wins variance' if s['mean_VarGap_layer']>0 else 'BC wins variance'})")
        print(f"  corr(G, VarGap)    = {s['corr_G_VarGap']:+.4f}")
        print(f"  by module type:")
        for t, p in s["per_module_type"].items():
            print(f"    {t:10s}  n={p['n']:2d}  "
                  f"mean G = {p['mean_G']:+.3e}  "
                  f"mean VarGap = {p['mean_VarGap']:+.3e}  "
                  f"Flip wins MSE on {p['frac_layers_Flip_wins_MSE']*100:.0f}% layers, "
                  f"Flip wins Var on {p['frac_layers_Flip_wins_Var']*100:.0f}% layers")

    print("\n" + "=" * 80)
    print("Per-layer detail (sorted by VarGap, descending = Flip variance-wins first):")
    print("=" * 80)
    by_eval = {}
    for r in rows:
        by_eval.setdefault(r.eval_name, []).append(r)
    for eval_name, rs in by_eval.items():
        rs_sorted = sorted(rs, key=lambda r: r.VarGap_layer, reverse=True)
        print(f"\n[eval = {eval_name}]")
        print(f"  {'layer':<55s}  {'G':>12s}  {'VarGap':>12s}  {'%rows G+':>9s}  {'%rows V+':>9s}")
        for r in rs_sorted:
            print(f"  {r.layer_name:<55s}  {r.G_layer:+12.3e}  {r.VarGap_layer:+12.3e}  "
                  f"{r.frac_rows_G_positive*100:>8.1f}%  "
                  f"{r.frac_rows_VarGap_positive*100:>8.1f}%")


def write_outputs(summary: Dict, rows: List[VarRow], out_dir: Path, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"config": cfg, "summary": summary}, f, indent=2)
    import csv
    with open(out_dir / "per_layer.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))


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
    parser.add_argument("--layers-pattern", type=str,
                        default="model.layers.*.mlp.down_proj,"
                                "model.layers.*.self_attn.o_proj")
    parser.add_argument("--max-layers", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default="./variance_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("VARIANCE-HYPOTHESIS VERIFICATION — CONFIG")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"  {k:30s} = {v}")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[1/5] Loading model: {args.model_path}")
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

    print(f"\n[2/5] Selecting modules")
    patterns = [p.strip() for p in args.layers_pattern.split(",") if p.strip()]
    module_names = select_modules(model, patterns, args.max_layers)
    if not module_names:
        print("ERROR: No modules matched.")
        sys.exit(1)
    print(f"  {len(module_names)} modules selected.")

    # ---- Cal stats (we only need mu_cal) ----
    print(f"\n[3/5] Capturing calibration μ from: {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_recorder = ActivationRecorder(
        module_names, record_full_cov=False,
        max_tokens_per_module=args.max_cal_tokens_per_layer,
        flush_every_tokens=args.flush_every_tokens,
    )
    run_calibration(model, tokenizer, cal_texts, cal_recorder,
                    device, args.max_length)
    cal_stats = cal_recorder.finalize()

    # ---- Quantize + compute Delta once per layer ----
    print(f"\n[4/5] Per-layer quantization and Delta computation")
    payloads = build_layer_payloads(
        model=model,
        module_names=module_names,
        cal_stats=cal_stats,
        bits=args.bits,
        group_size=args.group_size,
        flip_budget_pct=args.flip_budget_pct,
        knee_tolerance=args.knee_tolerance,
        device=device,
    )

    # ---- Eval-side: per-token Y^2 and Y^4 moments ----
    print(f"\n[5/5] Capturing eval moments on: {args.eval_datasets}")
    per_layer_per_eval: Dict[str, Dict[str, Dict]] = {n: {} for n in module_names}
    for eval_name in args.eval_datasets:
        print(f"\n  -- {eval_name} --")
        eval_texts = load_text_samples(eval_name, args.n_eval, args.seed + 1)
        rec = ErrorMomentRecorder(
            module_payloads=payloads,
            max_tokens_per_module=args.max_eval_tokens_per_layer,
            flush_every_tokens=args.flush_every_tokens,
        )
        rec.attach(model)
        model.eval()
        try:
            for t in tqdm(eval_texts, desc="    forward"):
                try:
                    enc = tokenizer(t, return_tensors="pt",
                                    truncation=True, max_length=args.max_length)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.no_grad():
                        model(**enc, use_cache=False)
                except Exception as e:
                    print(f"    [warn] skipped: {e}")
                    continue
        finally:
            rec.detach()
        finalized = rec.finalize()
        for n in module_names:
            if n in finalized:
                per_layer_per_eval[n][eval_name] = finalized[n]
        del rec
        gc.collect()
        torch.cuda.empty_cache()

    # Free GPU payloads
    for v in payloads.values():
        for k in ("e", "Delta", "mu_cal_dot_e"):
            if k in v:
                del v[k]
    payloads.clear()
    torch.cuda.empty_cache()

    summary, rows = summarize(per_layer_per_eval)
    print_summary(summary, rows)
    write_outputs(summary, rows, Path(args.out_dir), vars(args))
    print(f"\n✅ Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()