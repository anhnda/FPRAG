"""
V3: LayerNorm Survival Verification (Proposition 2)
====================================================

Tests the headline claim: BC's correction is largely erased by the next
LayerNorm, while SFA's centered correction survives.

For each linear layer that feeds into a LayerNorm-style normalization, we
compute:

  1. BC wasted fraction (closed-form):
        waste_BC = C * bar_b^2 / sum_j b_j^2
     where bar_b = (1/C) sum_j b_j.
     → Large waste_BC means much of BC's output-space gain is killed by LN.

  2. Output-space vs LN-space reduction:
        R_BC  = ΔL^LN(BC)  / ΔL_rec(BC)
        R_SFA = ΔL^LN(SFA) / ΔL_rec(SFA)
     Computed by linearized LN sensitivity:
        residual after LN ≈ (1/s_t) (r_t - mean(r_t) * 1)
                              − (1/s_t) <r_t, u_t> u_t
     where u_t = (y_t - mean(y_t) 1) / ||y_t - mean(y_t) 1||.
     We approximate using the projector
          P_t = I − (1/C) 11^T − u_t u_t^T
     applied row-wise to the residual.

  3. R_SFA / R_BC : the LN-amplified advantage factor.

To approximate the FP output y_t at the layer's output, we run a forward
pass on the FP model and store both the layer INPUT (for computing X)
and the layer OUTPUT (= X @ W^T, used to construct u_t).

We do NOT require running BC or SFA on the actual model — both
counterfactuals are computed analytically from the FP activations and the
quantization-error matrix e = W_q - W.

USAGE
-----
python verify_v3_ln_survival.py \\
    --fp-model      ./models/Mistral-7B-v0.3 \\
    --base-q-model  ./quantized_models/awq_base \\
    --sfa-q-model   ./quantized_models/awq_sfa \\
    --output        v3_ln_survival.csv \\
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
# Calibration data
# ----------------------------------------------------------------------------
def load_calibration_texts(tokenizer, n_samples=64, seed=42):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    random.seed(seed)
    random.shuffle(texts)
    return texts[:n_samples]


# ----------------------------------------------------------------------------
# Identifying layers followed by normalization
# ----------------------------------------------------------------------------
def is_norm_module(m):
    """Heuristic: does this module behave like LayerNorm / RMSNorm?"""
    cls_name = m.__class__.__name__.lower()
    return ('layernorm' in cls_name) or ('rmsnorm' in cls_name)


def find_linear_to_norm_pairs(model):
    """
    Identify (linear_name, norm_name) pairs where the Linear's output feeds
    (directly or with simple residual) into a Normalization module.

    For a clean V3, we focus on linear layers whose output is the input to a
    LayerNorm. In Transformer blocks, this is most cleanly observed when:
      - The linear is a "down_proj" / "o_proj" feeding a post-attention norm
      - The linear is the FFN output feeding the next block's norm.

    Rather than parsing the graph, we use a simpler heuristic: identify all
    norm modules, then mark linear layers whose name shares a prefix with a
    norm module that comes "after" it in named_modules order.

    Returns:
        candidates: set of linear-layer names that we'll treat as "LN-fed".

    Note: this is a heuristic. For Mistral / Llama, ALL block-internal linears
    feed (eventually, through residual) into a downstream RMSNorm, so
    treating every linear as LN-fed is acceptable as a first pass. The most
    informative comparison is between "LN-fed" and "lm_head" (no LN after).
    """
    names_in_order = [name for name, _ in model.named_modules()]
    type_by_name = {name: m for name, m in model.named_modules()}

    norm_indices = [i for i, n in enumerate(names_in_order)
                    if is_norm_module(type_by_name[n])]

    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    # Mark a linear as "LN-fed" if there is at least one norm module that
    # appears AFTER it in the module ordering.
    candidates = set()
    last_norm_pos = max(norm_indices) if norm_indices else -1
    for lname in linear_names:
        lpos = names_in_order.index(lname)
        if lpos < last_norm_pos:
            candidates.add(lname)
    # lm_head and similar trailing layers are typically NOT LN-fed
    return candidates


# ----------------------------------------------------------------------------
# Activation collector storing BOTH input and output of each linear
# ----------------------------------------------------------------------------
class IOActivationCollector:
    def __init__(self, model, max_tokens_per_sample=256):
        self.model = model
        self.max_tokens_per_sample = max_tokens_per_sample
        self.inputs = {}   # name -> list[Tensor [Ni, d] fp32 cpu]
        self.outputs = {}  # name -> list[Tensor [Ni, out] fp32 cpu]
        self.handles = []

    def _make_hook(self, name):
        def hook(_m, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                idx = torch.randperm(inp.shape[1])[:self.max_tokens_per_sample].sort()[0]
                inp = inp[:, idx, :]
                out = out[:, idx, :]
            self.inputs.setdefault(name, []).append(
                inp.detach().reshape(-1, inp.shape[-1]).cpu().float())
            self.outputs.setdefault(name, []).append(
                out.detach().reshape(-1, out.shape[-1]).cpu().float())
        return hook

    def register(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._make_hook(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_input(self, name):
        if name not in self.inputs or len(self.inputs[name]) == 0:
            return None
        return torch.cat(self.inputs[name], dim=0)

    def get_output(self, name):
        if name not in self.outputs or len(self.outputs[name]) == 0:
            return None
        return torch.cat(self.outputs[name], dim=0)

    def clear(self):
        self.inputs = {}; self.outputs = {}


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
    out = OrderedDict()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out[name] = m.weight.data.detach().cpu().float()
    return out


# ----------------------------------------------------------------------------
# Linearized LN loss
# ----------------------------------------------------------------------------
@torch.no_grad()
def linearized_ln_loss(y_fp, r, eps=1e-5):
    """
    Approximate the squared LN-space reconstruction error to first order in r.

    LN(y + r) - LN(y) ≈ (1/s_t) * P_t * r_t
        where  s_t = sqrt(Var_C(y_t) + eps)
        and    P_t = I - (1/C) 11^T - u_t u_t^T
        and    u_t = (y_t - mean(y_t)*1) / sqrt(C * Var_C(y_t))

    L^LN = (1/N) sum_t || (1/s_t) P_t r_t ||^2
         = (1/N) sum_t (1/s_t^2) [ ||r_t - mean(r_t) 1||^2 - <r_t - mean(r_t) 1, u_t>^2 ]

    Args:
        y_fp : [N, C] float32 FP output (= X W^T for the FP linear)
        r    : [N, C] float32 residual to assess (e.g. r = X @ e^T for BC=0 case,
               or r = r - b after BC, etc.)
    Returns:
        scalar L^LN
    """
    N, C = y_fp.shape
    # Center y
    y_mean = y_fp.mean(dim=1, keepdim=True)                # [N, 1]
    y_centered = y_fp - y_mean                              # [N, C]
    var_y = y_centered.pow(2).mean(dim=1, keepdim=True)    # [N, 1]
    s = (var_y + eps).sqrt()                               # [N, 1]

    # Center the residual across channels (LN removes channel-mean)
    r_mean = r.mean(dim=1, keepdim=True)
    r_centered = r - r_mean

    # Component along u_t: <r - r_mean*1, y - y_mean*1> / ||y - y_mean*1||
    inner = (r_centered * y_centered).sum(dim=1, keepdim=True)       # [N, 1]
    norm_y_c_sq = (y_centered.pow(2).sum(dim=1, keepdim=True))       # [N, 1]
    proj_along_u_sq = inner.pow(2) / norm_y_c_sq.clamp(min=1e-12)    # [N, 1]

    # ||P r||^2 = ||r_centered||^2 - <r_centered, u>^2
    per_sample_sq = (r_centered.pow(2).sum(dim=1, keepdim=True) - proj_along_u_sq)
    # Scale by 1/s^2 and average over N
    loss = (per_sample_sq / s.pow(2)).sum() / N
    return loss.item()


# ----------------------------------------------------------------------------
# Per-layer LN-survival measurement
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_ln_survival(X, y_fp, W_fp, W_base, W_sfa):
    """
    For one layer, compute all four loss values:
      L_rec(base), L_rec(BC), L_rec(SFA), and the same in LN-space.

    Returns dict.
    """
    N, d = X.shape
    out = W_fp.shape[0]

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        bytes_needed = (N * (d + out) + 3 * out * d + 4 * N * out) * 4
        if bytes_needed < 6 * (1024**3):
            X = X.cuda(); y_fp = y_fp.cuda()
            W_fp = W_fp.cuda(); W_base = W_base.cuda(); W_sfa = W_sfa.cuda()
        else:
            use_gpu = False

    e_base = W_base - W_fp
    e_sfa = W_sfa - W_fp

    # Output-space residuals
    r_base = X @ e_base.t()              # [N, out]
    r_sfa  = X @ e_sfa.t()               # [N, out]

    mu = X.mean(dim=0, keepdim=True)
    b_base = (mu @ e_base.t()).squeeze(0)     # [out]
    r_bc = r_base - b_base.unsqueeze(0)       # BC counterfactual residual

    # L_rec values
    L_rec_base = (r_base.pow(2).sum() / N).item()
    L_rec_bc   = (r_bc.pow(2).sum() / N).item()
    L_rec_sfa  = (r_sfa.pow(2).sum() / N).item()

    # L_LN values (linearized)
    L_ln_base = linearized_ln_loss(y_fp, r_base)
    L_ln_bc   = linearized_ln_loss(y_fp, r_bc)
    L_ln_sfa  = linearized_ln_loss(y_fp, r_sfa)

    # Reductions
    bc_red_rec  = L_rec_base - L_rec_bc
    sfa_red_rec = L_rec_base - L_rec_sfa
    bc_red_ln   = L_ln_base  - L_ln_bc
    sfa_red_ln  = L_ln_base  - L_ln_sfa

    # Wasted fraction (closed form)
    bar_b = b_base.mean().item()
    sum_b_sq = b_base.pow(2).sum().item()
    if sum_b_sq > 1e-12:
        waste_bc = (out * bar_b**2) / sum_b_sq
    else:
        waste_bc = 0.0

    # Survival ratios
    R_BC  = (bc_red_ln  / bc_red_rec)  if bc_red_rec  > 1e-12 else float('nan')
    R_SFA = (sfa_red_ln / sfa_red_rec) if sfa_red_rec > 1e-12 else float('nan')

    if use_gpu:
        del X, y_fp, W_fp, W_base, W_sfa, e_base, e_sfa, r_base, r_sfa, r_bc, mu, b_base
        torch.cuda.empty_cache()

    return {
        'L_rec_base': L_rec_base,
        'L_rec_bc':   L_rec_bc,
        'L_rec_sfa':  L_rec_sfa,
        'L_ln_base':  L_ln_base,
        'L_ln_bc':    L_ln_bc,
        'L_ln_sfa':   L_ln_sfa,
        'bc_red_rec':  bc_red_rec,
        'sfa_red_rec': sfa_red_rec,
        'bc_red_ln':   bc_red_ln,
        'sfa_red_ln':  sfa_red_ln,
        'waste_bc_closedform': waste_bc,
        'R_BC':  R_BC,
        'R_SFA': R_SFA,
    }


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model",      required=True)
    ap.add_argument("--base-q-model",  required=True)
    ap.add_argument("--sfa-q-model",   required=True)
    ap.add_argument("--output",        default="v3_ln_survival.csv")
    ap.add_argument("--n-calib",       type=int, default=64)
    ap.add_argument("--max-tokens",    type=int, default=256)
    ap.add_argument("--max-length",    type=int, default=512)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("=" * 80)
    print("V3: LayerNorm Survival Verification")
    print("=" * 80)

    # --- Load FP model & tokenizer ---
    print("\n[1/4] Loading FP model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    ln_fed_layers = find_linear_to_norm_pairs(fp_model)
    print(f"  Identified {len(ln_fed_layers)} linear layers as LN-fed")

    # --- Collect both inputs and outputs on FP model ---
    print("\n[2/4] Collecting (input, output) pairs on FP model...")
    texts = load_calibration_texts(tokenizer, n_samples=args.n_calib, seed=args.seed)
    collector = IOActivationCollector(fp_model, max_tokens_per_sample=args.max_tokens)
    run_calibration(fp_model, tokenizer, collector, texts, args.device, args.max_length)

    inputs_dict  = {n: collector.get_input(n)  for n in collector.inputs}
    outputs_dict = {n: collector.get_output(n) for n in collector.outputs}
    collector.clear()
    print(f"  Collected for {len(inputs_dict)} linear layers")

    # --- Extract FP weights then free GPU memory ---
    print("\n[3/4] Extracting weights from FP / base-Q / SFA-Q ...")
    W_fp_dict = get_weights(fp_model)
    fp_model = fp_model.to("cpu")
    del fp_model; gc.collect(); torch.cuda.empty_cache()

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
    print("\n[4/4] Per-layer LN-survival measurement...")
    common = [n for n in W_fp_dict
              if n in W_base_dict and n in W_sfa_dict
              and inputs_dict.get(n) is not None
              and outputs_dict.get(n) is not None]
    print(f"  Layers in common: {len(common)}")

    rows = []
    for name in tqdm(common, desc="  Layers"):
        X = inputs_dict[name]
        y_fp = outputs_dict[name]
        W_fp = W_fp_dict[name]; W_base = W_base_dict[name]; W_sfa = W_sfa_dict[name]
        if W_fp.shape != W_base.shape or W_fp.shape != W_sfa.shape:
            continue
        if (W_sfa - W_base).abs().sum().item() < 1e-10:
            continue

        try:
            stats = measure_ln_survival(X, y_fp, W_fp, W_base, W_sfa)
        except Exception as exc:
            print(f"\n  ⚠️  Failed on {name}: {exc}")
            continue

        rows.append({
            'layer': name,
            'is_ln_fed': int(name in ln_fed_layers),
            'd': W_fp.shape[1], 'out': W_fp.shape[0],
            'N_samples': X.shape[0],
            **stats,
        })
        inputs_dict[name]  = None
        outputs_dict[name] = None
        gc.collect()

    if not rows:
        print("\n❌ No layers measured."); return

    fieldnames = ['layer', 'is_ln_fed', 'd', 'out', 'N_samples',
                  'L_rec_base', 'L_rec_bc', 'L_rec_sfa',
                  'L_ln_base', 'L_ln_bc', 'L_ln_sfa',
                  'bc_red_rec', 'sfa_red_rec',
                  'bc_red_ln',  'sfa_red_ln',
                  'waste_bc_closedform', 'R_BC', 'R_SFA']
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

    waste = np.array([r['waste_bc_closedform'] for r in rows])
    R_BC  = np.array([r['R_BC']  for r in rows if np.isfinite(r['R_BC'])])
    R_SFA = np.array([r['R_SFA'] for r in rows if np.isfinite(r['R_SFA'])])

    print(f"\n[BC wasted fraction] waste_BC = C*bar_b^2 / sum_j b_j^2:")
    print(f"  min:    {waste.min():.4f}")
    print(f"  25th:   {np.percentile(waste, 25):.4f}")
    print(f"  median: {np.median(waste):.4f}")
    print(f"  75th:   {np.percentile(waste, 75):.4f}")
    print(f"  max:    {waste.max():.4f}")
    print(f"  mean:   {waste.mean():.4f}")
    print(f"  → A median > 0.5 means: in a typical layer, more than HALF of")
    print(f"     BC's output-space gain is structurally erased by LayerNorm.")

    print(f"\n[Survival ratios R = ΔL^LN / ΔL_rec]")
    print(f"  R_BC  (median = {np.median(R_BC):.3f},   mean = {R_BC.mean():.3f})")
    print(f"  R_SFA (median = {np.median(R_SFA):.3f},  mean = {R_SFA.mean():.3f})")

    # Pair up for ratio analysis
    pair_rows = [r for r in rows
                 if np.isfinite(r['R_BC']) and np.isfinite(r['R_SFA']) and r['R_BC'] > 1e-6]
    if pair_rows:
        ratio = np.array([r['R_SFA'] / r['R_BC'] for r in pair_rows])
        print(f"\n[R_SFA / R_BC] (only layers with R_BC > 1e-6):")
        print(f"  median: {np.median(ratio):.2f}x")
        print(f"  mean:   {ratio.mean():.2f}x")
        print(f"  → Median > 1 means SFA's reduction translates to LN space")
        print(f"     more efficiently than BC's.")

    # Separate LN-fed vs not (informational)
    ln_rows  = [r for r in rows if r['is_ln_fed']]
    nln_rows = [r for r in rows if not r['is_ln_fed']]
    if ln_rows and nln_rows:
        print(f"\n[Breakdown by LN-fed]:")
        print(f"  LN-fed     ({len(ln_rows)} layers): median waste_BC = "
              f"{np.median([r['waste_bc_closedform'] for r in ln_rows]):.3f}")
        print(f"  not LN-fed ({len(nln_rows)} layers): median waste_BC = "
              f"{np.median([r['waste_bc_closedform'] for r in nln_rows]):.3f}")


if __name__ == "__main__":
    main()