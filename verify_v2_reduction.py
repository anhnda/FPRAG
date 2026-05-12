"""
V2: Reduction Comparison & Per-Channel Identity Check (Proposition 1)
======================================================================

This script answers two questions per layer:

  Q1 (identity / unit-test for Prop. 1):
       Is the EXACT formula
            Delta L_j = 2 b_j beta_j + beta_j^2 + 2 gamma_j + nu_j^2
       satisfied for SFA's actual updates?

  Q2 (substantive comparison):
       How much L_rec does each method actually remove?
         - BC removes exactly L_fmr = sum_j b_j^2.
         - SFA removes ?  (measured: L_rec(before) - L_rec(after))
       SFA should remove MORE than BC, with the surplus living in the
       centered subspace.

Definitions:
  e_j           = (W_q - W)_{:, j}                : [d]
  Delta_j       = (W_q' - W_q)_{:, j}             : [d]   (SFA update)
  b_j           = mu^T e_j                        : scalar
  beta_j        = mu^T Delta_j                    : scalar
  gamma_j       = (1/N) tilde_r_j^T tilde_X Delta_j
  nu_j^2        = (1/N) || tilde_X Delta_j ||^2
  tilde_r_j     = tilde_X^T e_j        (shape [N])
  tilde_X       = X - mu^T 1           (shape [N, d])

Inputs needed:
  --fp-model       : full-precision reference
  --base-q-model   : the base quantized model BEFORE SFA refinement
                     (e.g., AWQ output without flipping)
  --sfa-q-model    : the SFA-refined model (with flipping applied)

To produce both quantized models, run your awq_dh_xl.py twice:
  - once with --no-heuristic (gives base-q-model)
  - once with default flags  (gives sfa-q-model)

Output: CSV with one row per layer + summary.

USAGE
-----
python verify_v2_reduction.py \\
    --fp-model      ./models/Mistral-7B-v0.3 \\
    --base-q-model  ./quantized_models/awq_base \\
    --sfa-q-model   ./quantized_models/awq_sfa \\
    --output        v2_reduction.csv \\
    --n-calib       64
"""

import os
import gc
import argparse
import random
import csv
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ----------------------------------------------------------------------------
# Shared utilities (parallel to V1)
# ----------------------------------------------------------------------------
def load_calibration_texts(tokenizer, n_samples=64, seed=42):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    random.seed(seed)
    random.shuffle(texts)
    return texts[:n_samples]


class ActivationCollector:
    def __init__(self, model, max_tokens_per_sample=256):
        self.model = model
        self.max_tokens_per_sample = max_tokens_per_sample
        self.activations = {}
        self.handles = []

    def _make_hook(self, name):
        def hook(_m, input, _out):
            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                idx = torch.randperm(inp.shape[1])[:self.max_tokens_per_sample].sort()[0]
                inp = inp[:, idx, :]
            self.activations.setdefault(name, []).append(
                inp.detach().reshape(-1, inp.shape[-1]).cpu().float()
            )
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get(self, name):
        if name not in self.activations or len(self.activations[name]) == 0:
            return None
        return torch.cat(self.activations[name], dim=0)

    def clear(self):
        self.activations = {}


def run_calibration(model, tokenizer, collector, texts, device, max_length=512):
    model.eval()
    collector.register()
    try:
        with torch.no_grad():
            for text in tqdm(texts, desc="  Calibrating", leave=False):
                try:
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=max_length)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    model(**inputs, use_cache=False, return_dict=True)
                except Exception:
                    continue
    finally:
        collector.remove()


def get_weights(model):
    """Return dict: name -> W [out, d] float32 on CPU."""
    out = OrderedDict()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out[name] = m.weight.data.detach().cpu().float()
    return out


# ----------------------------------------------------------------------------
# Core per-layer measurement
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_layer(X, W_fp, W_base, W_sfa):
    """
    Compute everything needed for one layer.

    Args:
        X      : [N, d] float32  activations
        W_fp   : [out, d] float32  FP weights
        W_base : [out, d] float32  base-quantized (pre-SFA) weights
        W_sfa  : [out, d] float32  SFA-refined weights

    Returns a dict of scalars (sums across channels).
    """
    N, d = X.shape
    out_features = W_fp.shape[0]

    # Move to GPU if it fits
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        bytes_needed = (N * d + 3 * out_features * d + 4 * N * out_features) * 4
        if bytes_needed < 6 * (1024**3):
            X = X.cuda(); W_fp = W_fp.cuda(); W_base = W_base.cuda(); W_sfa = W_sfa.cuda()
        else:
            use_gpu = False  # keep on CPU

    # Errors
    e_base = W_base - W_fp                        # [out, d]
    e_sfa  = W_sfa  - W_fp                        # [out, d]
    Delta  = W_sfa - W_base                       # [out, d]  (SFA update)

    # Statistics
    mu = X.mean(dim=0, keepdim=True)              # [1, d]
    Xc = X - mu                                   # [N, d]

    # --- Pre-SFA quantities ---
    r_base = X @ e_base.t()                       # [N, out]
    b_base = (mu @ e_base.t()).squeeze(0)         # [out]
    tilde_r_base = r_base - b_base.unsqueeze(0)   # [N, out]
    L_rec_base = (r_base.pow(2).sum() / N).item()
    L_fmr_base = b_base.pow(2).sum().item()
    L_cent_base = (tilde_r_base.pow(2).sum() / N).item()

    # --- BC counterfactual: shift output by -b_base ---
    # New residual is tilde_r_base; L_rec drops by exactly L_fmr_base.
    L_rec_bc = (tilde_r_base.pow(2).sum() / N).item()      # = L_cent_base
    bc_reduction = L_rec_base - L_rec_bc                    # should equal L_fmr_base

    # --- SFA quantities ---
    r_sfa = X @ e_sfa.t()                          # [N, out]
    b_sfa = (mu @ e_sfa.t()).squeeze(0)            # [out]
    tilde_r_sfa = r_sfa - b_sfa.unsqueeze(0)
    L_rec_sfa = (r_sfa.pow(2).sum() / N).item()
    L_fmr_sfa = b_sfa.pow(2).sum().item()
    L_cent_sfa = (tilde_r_sfa.pow(2).sum() / N).item()
    sfa_reduction = L_rec_base - L_rec_sfa

    # --- Decomposition of SFA's reduction (Proposition 1, exact identity) ---
    # Delta L_j = 2 b_j beta_j + beta_j^2 + 2 gamma_j + nu_j^2
    beta  = (mu @ Delta.t()).squeeze(0)            # [out]
    XcD   = Xc @ Delta.t()                          # [N, out]  = tilde_X @ Delta_j per channel
    gamma = (tilde_r_base * XcD).sum(dim=0) / N    # [out]
    nu_sq = XcD.pow(2).sum(dim=0) / N              # [out]

    # Predicted vs measured (per channel, then summed)
    predicted_delta_L_per_ch = 2 * b_base * beta + beta.pow(2) + 2 * gamma + nu_sq   # [out]
    # Measured per-channel: (||r_sfa_j||^2 - ||r_base_j||^2) / N
    measured_delta_L_per_ch = (r_sfa.pow(2).sum(dim=0) - r_base.pow(2).sum(dim=0)) / N  # [out]

    # Identity gap (should be ~0)
    abs_err = (predicted_delta_L_per_ch - measured_delta_L_per_ch).abs()
    base_scale = measured_delta_L_per_ch.abs().clamp(min=1e-10)
    rel_err_per_ch = (abs_err / base_scale)
    identity_max_abs = abs_err.max().item()
    identity_mean_rel = rel_err_per_ch.mean().item()

    # Aggregate split of SFA's reduction
    sfa_mean_part   = -(2 * b_base * beta + beta.pow(2)).sum().item()    # how much L_fmr decreased
    sfa_cent_part   = -(2 * gamma + nu_sq).sum().item()                  # how much L_cent decreased

    # Cleanup
    del r_base, tilde_r_base, r_sfa, tilde_r_sfa, XcD, mu, Xc
    del e_base, e_sfa, Delta, beta, gamma, nu_sq, b_base, b_sfa
    del predicted_delta_L_per_ch, measured_delta_L_per_ch, abs_err, rel_err_per_ch
    if use_gpu:
        torch.cuda.empty_cache()

    return {
        # Loss values
        'L_rec_base':   L_rec_base,
        'L_fmr_base':   L_fmr_base,
        'L_cent_base':  L_cent_base,
        'L_rec_bc':     L_rec_bc,
        'L_rec_sfa':    L_rec_sfa,
        'L_fmr_sfa':    L_fmr_sfa,
        'L_cent_sfa':   L_cent_sfa,
        # Reductions
        'bc_reduction':  bc_reduction,
        'sfa_reduction': sfa_reduction,
        # SFA reduction split
        'sfa_reduction_mean_part': sfa_mean_part,
        'sfa_reduction_cent_part': sfa_cent_part,
        # Identity quality (Prop. 1 check)
        'identity_max_abs':   identity_max_abs,
        'identity_mean_rel':  identity_mean_rel,
    }


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model",      required=True)
    ap.add_argument("--base-q-model",  required=True,
                    help="Base quantized model (pre-SFA, e.g. AWQ with --no-heuristic)")
    ap.add_argument("--sfa-q-model",   required=True,
                    help="SFA-refined quantized model")
    ap.add_argument("--output",        default="v2_reduction.csv")
    ap.add_argument("--n-calib",       type=int, default=64)
    ap.add_argument("--max-tokens",    type=int, default=256)
    ap.add_argument("--max-length",    type=int, default=512)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("=" * 80)
    print("V2: BC vs SFA Reduction + Per-Channel Identity Check")
    print("=" * 80)

    # --- Load FP model & collect activations ---
    print("\n[1/5] Loading FP model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    print("\n[2/5] Collecting activations on FP model...")
    texts = load_calibration_texts(tokenizer, n_samples=args.n_calib, seed=args.seed)
    collector = ActivationCollector(fp_model, max_tokens_per_sample=args.max_tokens)
    run_calibration(fp_model, tokenizer, collector, texts, args.device, args.max_length)
    activations = {name: collector.get(name) for name in collector.activations}
    collector.clear()
    print(f"  Collected activations for {len(activations)} linear layers")

    # --- Extract FP weights then free GPU memory ---
    print("\n[3/5] Extracting FP weights...")
    W_fp_dict = get_weights(fp_model)
    fp_model = fp_model.to("cpu")
    del fp_model; gc.collect(); torch.cuda.empty_cache()

    # --- Load base quantized model & extract weights ---
    print("\n[4/5] Loading base-Q & SFA-Q models, extracting weights...")
    base_q = AutoModelForCausalLM.from_pretrained(
        args.base_q_model, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    ).eval()
    W_base_dict = get_weights(base_q)
    del base_q; gc.collect()

    sfa_q = AutoModelForCausalLM.from_pretrained(
        args.sfa_q_model, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    ).eval()
    W_sfa_dict = get_weights(sfa_q)
    del sfa_q; gc.collect(); torch.cuda.empty_cache()

    # --- Per-layer measurement ---
    print("\n[5/5] Per-layer measurement...")
    common = [n for n in W_fp_dict
              if n in W_base_dict and n in W_sfa_dict and activations.get(n) is not None]
    print(f"  Layers in common: {len(common)}")

    rows = []
    for name in tqdm(common, desc="  Layers"):
        X = activations[name]
        W_fp = W_fp_dict[name]; W_base = W_base_dict[name]; W_sfa = W_sfa_dict[name]
        if W_fp.shape != W_base.shape or W_fp.shape != W_sfa.shape:
            continue
        # Skip if SFA made no change (identity check still valid but degenerate)
        if (W_sfa - W_base).abs().sum().item() < 1e-10:
            continue

        try:
            stats = measure_layer(X, W_fp, W_base, W_sfa)
        except Exception as exc:
            print(f"\n  ⚠️  Failed on {name}: {exc}")
            continue

        rows.append({
            'layer': name,
            'd': W_fp.shape[1],
            'out': W_fp.shape[0],
            'N_samples': X.shape[0],
            **stats,
        })
        activations[name] = None
        gc.collect()

    if not rows:
        print("\n❌ No layers measured.")
        return

    fieldnames = ['layer', 'd', 'out', 'N_samples',
                  'L_rec_base', 'L_fmr_base', 'L_cent_base',
                  'L_rec_bc', 'L_rec_sfa', 'L_fmr_sfa', 'L_cent_sfa',
                  'bc_reduction', 'sfa_reduction',
                  'sfa_reduction_mean_part', 'sfa_reduction_cent_part',
                  'identity_max_abs', 'identity_mean_rel']
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Wrote per-layer results to {args.output}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    bc_red   = np.array([r['bc_reduction']  for r in rows])
    sfa_red  = np.array([r['sfa_reduction'] for r in rows])
    sfa_mean = np.array([r['sfa_reduction_mean_part'] for r in rows])
    sfa_cent = np.array([r['sfa_reduction_cent_part'] for r in rows])
    id_max   = np.array([r['identity_max_abs']  for r in rows])
    id_rel   = np.array([r['identity_mean_rel'] for r in rows])

    print(f"\n[Identity check]  Δ L_j  predicted vs measured per channel:")
    print(f"  Max absolute error (across all layers, all channels): {id_max.max():.4e}")
    print(f"  Mean relative error (averaged across channels per layer, then across layers):")
    print(f"    median = {np.median(id_rel):.2e}, max = {id_rel.max():.2e}")
    if id_max.max() > 1e-2:
        print(f"  ⚠️  Identity may be violated — investigate!")
    else:
        print(f"  ✓ Identity holds (Proposition 1 verified for SFA's actual updates).")

    print(f"\n[Reduction comparison]  (sums across all layers):")
    print(f"  Total BC  reduction = {bc_red.sum():.4e}")
    print(f"  Total SFA reduction = {sfa_red.sum():.4e}")
    if bc_red.sum() > 0:
        print(f"  Ratio SFA/BC = {sfa_red.sum() / bc_red.sum():.2f}x")
    print(f"\n  SFA reduction decomposition:")
    print(f"    Mean-part contribution    = {sfa_mean.sum():.4e}   ({sfa_mean.sum()/sfa_red.sum()*100:.1f}%)")
    print(f"    Centered-part contribution= {sfa_cent.sum():.4e}   ({sfa_cent.sum()/sfa_red.sum()*100:.1f}%)")
    print(f"\n  → The centered-part contribution is SFA's structural advantage over BC.")

    # Per-layer wins
    sfa_wins = int(np.sum(sfa_red > bc_red))
    print(f"\n  Layers where SFA strictly beats BC: {sfa_wins} / {len(rows)} ({sfa_wins/len(rows)*100:.1f}%)")
    if sfa_wins < len(rows):
        losses = [(r['layer'], r['sfa_reduction'], r['bc_reduction']) for r in rows
                  if r['sfa_reduction'] <= r['bc_reduction']]
        print(f"  Layers where SFA does NOT beat BC:")
        for n, s, b in losses[:5]:
            print(f"    {n}: SFA={s:.4e}, BC={b:.4e}")
        if len(losses) > 5:
            print(f"    ... and {len(losses) - 5} more")


if __name__ == "__main__":
    main()