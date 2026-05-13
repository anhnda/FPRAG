"""
Diagnose the α vs G relationship per-layer, across eval distributions.

Reads per_layer_results.json from a verify_flip_vs_bc_theory.py run and
prints:
  - Per-layer α and G for each eval distribution, side-by-side.
  - Cross-eval consistency of α (correlation of α_c4val vs α_wiki).
  - Decomposition into T1..T5 — which term dominates G?
  - Whether the "missing term" hypothesis (δ⊥·Δ correlated with structure)
    is empirically supported.

Usage:
  python diagnose_alpha_g.py flip_vs_bc_results/llama3_matched/per_layer_results.json
"""

import json
import sys
from collections import defaultdict
import numpy as np


def main(path):
    with open(path) as f:
        data = json.load(f)
    results = data["results"]

    by_eval = defaultdict(list)
    for r in results:
        by_eval[r["eval_name"]].append(r)

    eval_names = list(by_eval.keys())
    print(f"Eval distributions found: {eval_names}")
    print(f"Layers per eval: {[len(by_eval[e]) for e in eval_names]}")

    # Align layers across eval distributions by layer_name.
    layer_order = [r["layer_name"] for r in by_eval[eval_names[0]]]
    aligned = {}  # layer_name -> {eval_name -> row}
    for r in results:
        aligned.setdefault(r["layer_name"], {})[r["eval_name"]] = r

    # 1. Per-layer side-by-side table
    print()
    print("=" * 110)
    print(f"{'layer':<55s}", end="")
    for e in eval_names:
        print(f" | {e:>10s} α  {e:>10s} G ", end="")
    print()
    print("-" * 110)
    for ln in layer_order:
        print(f"{ln:<55s}", end="")
        for e in eval_names:
            r = aligned[ln].get(e)
            if r:
                print(f" | {r['alpha']:+10.4f}  {r['G_layer']:+10.3e}", end="")
            else:
                print(f" | {'--':>10s}  {'--':>10s}", end="")
        print()

    # 2. Cross-eval α consistency
    if len(eval_names) >= 2:
        print()
        print("=" * 80)
        print("α consistency across eval distributions")
        print("=" * 80)
        for i, e1 in enumerate(eval_names):
            for e2 in eval_names[i+1:]:
                a1 = np.array([aligned[ln][e1]["alpha"] for ln in layer_order
                              if e1 in aligned[ln] and e2 in aligned[ln]])
                a2 = np.array([aligned[ln][e2]["alpha"] for ln in layer_order
                              if e1 in aligned[ln] and e2 in aligned[ln]])
                if len(a1) >= 3:
                    corr = np.corrcoef(a1, a2)[0, 1]
                    print(f"  corr(α_{e1}, α_{e2}) = {corr:+.4f}  "
                          f"(over {len(a1)} layers)")
                    # Sign agreement
                    same_sign = np.sum(np.sign(a1) == np.sign(a2))
                    print(f"  sign agreement: {same_sign}/{len(a1)} layers")

    # 3. T1..T5 dominance per layer
    print()
    print("=" * 110)
    print("T1..T5 contribution to G per layer (mean over j; T1+T2+T3+T4+T5 = G)")
    print("=" * 110)
    for e in eval_names:
        print(f"\n[eval = {e}]")
        print(f"  {'layer':<55s}  {'T1':>12s} {'T2':>12s} {'T3':>12s} {'T4':>12s} {'T5':>12s} | {'G':>12s}")
        for ln in layer_order:
            r = aligned[ln].get(e)
            if not r:
                continue
            print(f"  {ln:<55s}  {r['T1']:+12.3e} {r['T2']:+12.3e} {r['T3']:+12.3e} "
                  f"{r['T4']:+12.3e} {r['T5']:+12.3e} | {r['G_layer']:+12.3e}")

    # 4. Which term has the same sign as G?
    print()
    print("=" * 80)
    print("Which term drives the sign of G?")
    print("=" * 80)
    for e in eval_names:
        rows = by_eval[e]
        n = len(rows)
        if n == 0:
            continue
        counts = {f"T{i}": 0 for i in range(1, 6)}
        for r in rows:
            terms = {f"T{i}": r[f"T{i}"] for i in range(1, 6)}
            # Which term has the largest |T_i| and matches sign(G)?
            ranked = sorted(terms.items(), key=lambda kv: abs(kv[1]), reverse=True)
            dominant = ranked[0][0]
            counts[dominant] += 1
        print(f"\n[eval = {e}]  largest-|T| per layer:")
        for k, v in counts.items():
            print(f"    {k}: {v}/{n} ({v/n*100:.0f}%)")

    # 5. Test the "missing term" hypothesis: T1 = -2 (δ·e)(δ·Δ)
    # If the proof sketch were tight, T1 would dominate and have sign +(when α>0) / -(when α<0).
    # The mismatch is concentrated in (δ^⊥)·Δ — but we can probe it by checking whether
    # sign(T1) matches sign(α) across layers.
    print()
    print("=" * 80)
    print("Does sign(T1) match sign(α)? (Prop 2 claims yes)")
    print("=" * 80)
    for e in eval_names:
        rows = by_eval[e]
        agree = sum(1 for r in rows if np.sign(r["T1"]) == np.sign(r["alpha"]) and r["alpha"] != 0)
        nontrivial = sum(1 for r in rows if r["alpha"] != 0)
        if nontrivial > 0:
            print(f"  [eval = {e}]  sign(T1) == sign(α) on {agree}/{nontrivial} layers "
                  f"({agree/nontrivial*100:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python diagnose_alpha_g.py path/to/per_layer_results.json")
        sys.exit(1)
    main(sys.argv[1])