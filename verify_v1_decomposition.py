"""
V1: Verification of the Orthogonal Decomposition (Proposition 1)
=================================================================

Tests the central claim: the layer-wise reconstruction loss decomposes as
    L_rec = L_fmr + L_cent
where
    L_fmr  = sum_j (mu^T e_j)^2                          (first-moment, rank-1)
    L_cent = sum_j (1/m) || tilde_X^T e_j ||^2           (centered, rank-(m-1))

and crucially, that L_cent >> L_fmr on real layers, so BC (which can only
remove L_fmr) leaves most of the reconstruction error untouched.

What this script measures (per layer):
  1. L_rec       : 1/m || X^T e ||_F^2     (true reconstruction loss)
  2. L_fmr       : sum_j b_j^2              (BC's ceiling)
  3. L_cent      : 1/m || tilde_X^T e ||_F^2
  4. Identity    : |L_rec - (L_fmr + L_cent)| / L_rec   (must be ~0)
  5. Ratio R     : L_cent / L_fmr           (BC's "wasted space" indicator)

Output: a CSV with one row per quantized linear layer, plus an aggregated summary.

USAGE
-----
python verify_v1_decomposition.py \\
    --fp-model ./models/Mistral-7B-v0.3 \\
    --q-model  ./quantized_models/model_awq_js_xl \\
    --output   v1_decomposition.csv \\
    --n-calib  64
"""

import os
import gc
import argparse
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ----------------------------------------------------------------------------
# Calibration data loading (matches the existing pipeline style)
# ----------------------------------------------------------------------------
def load_calibration_texts(tokenizer, n_samples=64, seqlen=2048, seed=42):
    """Load a small calibration corpus. Uses WikiText-2 raw for simplicity."""
    try:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
        random.seed(seed)
        random.shuffle(texts)
        return texts[:n_samples]
    except Exception as e:
        raise RuntimeError(f"Failed to load WikiText-2: {e}")


# ----------------------------------------------------------------------------
# Activation hooking: collect inputs to every nn.Linear layer
# ----------------------------------------------------------------------------
class ActivationCollector:
    """Hooks every nn.Linear layer and stores its inputs (flattened over tokens)."""

    def __init__(self, model, max_tokens_per_sample=256):
        self.model = model
        self.max_tokens_per_sample = max_tokens_per_sample
        self.activations = {}   # name -> list[Tensor on CPU, float32]
        self.handles = []

    def _make_hook(self, name):
        def hook(_module, input, _output):
            inp = input[0] if isinstance(input, tuple) else input
            # Subsample tokens if sequence is too long
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                idx = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                idx = idx.sort()[0]
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
        """Return stacked activations [N, d] (float32) on CPU, or None."""
        if name not in self.activations or len(self.activations[name]) == 0:
            return None
        return torch.cat(self.activations[name], dim=0)

    def clear(self):
        self.activations = {}


def run_calibration(model, tokenizer, collector, texts, device, max_length=512):
    """Run forward passes to fill the activation collector."""
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


# ----------------------------------------------------------------------------
# Quantization-error extraction
# ----------------------------------------------------------------------------
def get_quantization_errors(fp_model, q_model):
    """
    Return a dict: layer_name -> e = (W_q - W) [out, d], float32, on CPU.

    Assumes both models have identical architecture and matching parameter names.
    """
    fp_state = dict(fp_model.named_modules())
    q_state = dict(q_model.named_modules())

    errors = OrderedDict()
    for name, fp_mod in fp_state.items():
        if not isinstance(fp_mod, nn.Linear):
            continue
        if name not in q_state or not isinstance(q_state[name], nn.Linear):
            continue
        W_fp = fp_mod.weight.data.detach().cpu().float()       # [out, d]
        W_q = q_state[name].weight.data.detach().cpu().float() # [out, d]
        if W_fp.shape != W_q.shape:
            print(f"  ⚠️  Shape mismatch for {name}: {W_fp.shape} vs {W_q.shape}, skipping")
            continue
        errors[name] = W_q - W_fp
    return errors


# ----------------------------------------------------------------------------
# Core measurement: compute L_rec, L_fmr, L_cent for one layer
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_layer_decomposition(X, e):
    """
    Given activations X [N, d] and quantization error e = W_q - W [out, d] (note: e
    has shape [out, d] in PyTorch convention; we treat each row as e_j for one output
    channel j), compute the orthogonal decomposition.

    Args:
        X: [N, d] float32 calibration activations (rows = samples)
        e: [out, d] float32 quantization errors (rows = output channels)

    Returns dict with:
        L_rec  : 1/N * ||X @ e^T||_F^2
        L_fmr  : sum_j (mu^T e_j)^2          where mu = X.mean(0)
        L_cent : 1/N * ||tilde_X @ e^T||_F^2 where tilde_X = X - mu
        identity_gap : |L_rec - (L_fmr + L_cent)| / max(L_rec, eps)
        L_cent_over_L_fmr : ratio
    """
    N, d = X.shape
    out_features, d2 = e.shape
    assert d == d2, f"Dimension mismatch: X is [N={N}, d={d}], e is [out={out_features}, d={d2}]"

    # Move to GPU if possible for speed
    if torch.cuda.is_available():
        # Memory-safe: process in chunks if either is huge
        bytes_needed = (N * out_features + N * d + out_features * d) * 4
        if bytes_needed < 4 * (1024**3):  # < 4 GB
            X_g = X.cuda()
            e_g = e.cuda()
        else:
            X_g, e_g = X, e
    else:
        X_g, e_g = X, e

    # --- L_rec: full reconstruction loss ---
    # r = X @ e^T : [N, out]    each column = r_j
    r = X_g @ e_g.t()                                          # [N, out]
    L_rec = (r.pow(2).sum() / N).item()

    # --- L_fmr: first-moment part ---
    # b_j = mu^T e_j = (1/N) sum_t r_{t,j}
    mu = X_g.mean(dim=0, keepdim=True)                         # [1, d]
    b = (mu @ e_g.t()).squeeze(0)                              # [out]
    L_fmr = b.pow(2).sum().item()

    # --- L_cent: centered part ---
    # tilde_r = r - b   (broadcast along rows)
    tilde_r = r - b.unsqueeze(0)                               # [N, out]
    L_cent = (tilde_r.pow(2).sum() / N).item()

    # --- Identity check ---
    total_reconstructed = L_fmr + L_cent
    if L_rec > 0:
        identity_gap = abs(L_rec - total_reconstructed) / L_rec
    else:
        identity_gap = 0.0

    # --- Ratio ---
    if L_fmr > 1e-12:
        ratio = L_cent / L_fmr
    else:
        ratio = float('inf')

    # Cleanup
    del r, tilde_r, mu, b, X_g, e_g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'L_rec': L_rec,
        'L_fmr': L_fmr,
        'L_cent': L_cent,
        'identity_gap': identity_gap,
        'L_cent_over_L_fmr': ratio,
    }


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model", required=True, help="Full-precision model path")
    ap.add_argument("--q-model", required=True, help="Quantized model path")
    ap.add_argument("--output", default="v1_decomposition.csv")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("V1: Orthogonal Decomposition Verification")
    print("=" * 80)
    print(f"  FP model:  {args.fp_model}")
    print(f"  Q  model:  {args.q_model}")
    print(f"  Calibration samples: {args.n_calib}")
    print("=" * 80)

    # --- Load FP model and tokenizer ---
    print("\n[1/4] Loading FP model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    fp_model.eval()

    # --- Collect activations on FP model ---
    print("\n[2/4] Collecting calibration activations (on FP model)...")
    texts = load_calibration_texts(tokenizer, n_samples=args.n_calib, seed=args.seed)
    collector = ActivationCollector(fp_model, max_tokens_per_sample=args.max_tokens)
    run_calibration(fp_model, tokenizer, collector, texts, args.device, args.max_length)

    # We keep activations in CPU memory; free GPU model state next
    activations = {name: collector.get(name) for name in collector.activations}
    collector.clear()
    print(f"  Collected activations for {len(activations)} linear layers")

    # Move FP model to CPU to free GPU memory for Q model
    fp_model = fp_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    # --- Load quantized model and extract per-layer errors ---
    print("\n[3/4] Loading quantized model & extracting errors...")
    q_model = AutoModelForCausalLM.from_pretrained(
        args.q_model, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )
    q_model.eval()

    errors = get_quantization_errors(fp_model, q_model)
    print(f"  Extracted errors for {len(errors)} layers")

    # Free FP model entirely (we have e = W_q - W stored on CPU)
    del fp_model, q_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Measure decomposition per layer ---
    print("\n[4/4] Measuring decomposition per layer...")
    rows = []
    common_layers = [n for n in errors if n in activations and activations[n] is not None]
    print(f"  Layers with both errors and activations: {len(common_layers)}")

    for name in tqdm(common_layers, desc="  Layers"):
        X = activations[name]              # [N, d]
        e = errors[name]                   # [out, d]

        # Quick sanity skip for degenerate cases
        if X.shape[0] < 8 or e.abs().sum().item() < 1e-10:
            continue

        try:
            stats = measure_layer_decomposition(X, e)
        except Exception as exc:
            print(f"\n  ⚠️  Failed on {name}: {exc}")
            continue

        rows.append({
            'layer': name,
            'd': e.shape[1],
            'out': e.shape[0],
            'N_samples': X.shape[0],
            **stats,
        })

        # Free activation memory aggressively
        activations[name] = None
        gc.collect()

    # --- Write CSV ---
    if not rows:
        print("\n❌ No layers measured. Aborting.")
        return

    import csv
    fieldnames = ['layer', 'd', 'out', 'N_samples',
                  'L_rec', 'L_fmr', 'L_cent',
                  'identity_gap', 'L_cent_over_L_fmr']
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
    L_rec_vals  = np.array([r['L_rec']  for r in rows])
    L_fmr_vals  = np.array([r['L_fmr']  for r in rows])
    L_cent_vals = np.array([r['L_cent'] for r in rows])
    gaps        = np.array([r['identity_gap'] for r in rows])
    ratios      = np.array([r['L_cent_over_L_fmr'] for r in rows
                            if np.isfinite(r['L_cent_over_L_fmr'])])

    print(f"\nIdentity check  L_rec = L_fmr + L_cent  (must be ~ 0):")
    print(f"  max rel-gap : {gaps.max():.2e}")
    print(f"  mean rel-gap: {gaps.mean():.2e}")
    print(f"  median rel-gap: {np.median(gaps):.2e}")
    if gaps.max() > 1e-3:
        print(f"  ⚠️  Identity gap exceeds 1e-3 — possible numerical issue.")
    else:
        print(f"  ✓ Identity holds to numerical precision.")

    print(f"\nGlobal totals (summed across layers):")
    print(f"  Total L_rec  = {L_rec_vals.sum():.4e}")
    print(f"  Total L_fmr  = {L_fmr_vals.sum():.4e}   ({L_fmr_vals.sum()/L_rec_vals.sum()*100:.2f}% of L_rec)")
    print(f"  Total L_cent = {L_cent_vals.sum():.4e}  ({L_cent_vals.sum()/L_rec_vals.sum()*100:.2f}% of L_rec)")
    print(f"\n  → BC ceiling is {L_fmr_vals.sum()/L_rec_vals.sum()*100:.2f}% of total reconstruction loss.")
    print(f"  → SFA-accessible additional space is {L_cent_vals.sum()/L_rec_vals.sum()*100:.2f}%.")

    print(f"\nRatio  L_cent / L_fmr  (across {len(ratios)} layers):")
    print(f"  min:    {ratios.min():.2f}")
    print(f"  25th:   {np.percentile(ratios, 25):.2f}")
    print(f"  median: {np.median(ratios):.2f}")
    print(f"  75th:   {np.percentile(ratios, 75):.2f}")
    print(f"  max:    {ratios.max():.2f}")
    print(f"  mean:   {ratios.mean():.2f}")
    print()
    print(f"  → If median ratio >> 1, the theory's central claim is supported:")
    print(f"    most reconstruction error lives outside BC's reach.")


if __name__ == "__main__":
    main()