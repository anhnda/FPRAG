"""
verify_bc_overfit.py  (v2 — memory-bounded)
============================================

Tests two predictions of the "BC overfits, Flip generalizes" theory:

  E1. BC's expected val-bias equals  L_cent · (1/N + 1/N_val).
      The 1/N comes from train-side noise in mu_hat; the 1/N_val comes from
      val-side noise in mu_val (since we can't access the population mean).

  E3. There exists N* below which Flip's val-bias < BC's val-bias.

MEMORY STRATEGY (v2)
--------------------
v1 stored every layer's activations for the whole calibration pool in CPU
RAM. With 2048-token sequences and Mistral-7B (~225 linear layers, d up to
14336), that is hundreds of GB. v2 fixes this:

  * Process layers in BATCHES (default 8). Each batch does its OWN forward
    pass collecting activations only for the layers in that batch.
  * Cap stored tokens per layer to `pool_tokens` (default 8192). We only need
    n_val + max(n_train_list) ≈ 4-5k anyway; anything more is waste.
  * Store activations in fp16 on CPU (half the memory of fp32; we re-cast to
    fp32 for the measurement math).

USAGE
-----
python verify_bc_overfit.py \\
    --fp-model ./models/Mistral-7B-v0.3 \\
    --q-model  ./quantized_models/awq_no_flip \\
    --output-dir ./verify_bc_results \\
    --n-train-list 32 64 128 256 \\
    --n-val 4096 --pool-tokens 8192 \\
    --repetitions 5 \\
    --layer-batch-size 8 \\
    --n-calib-seqs 64

NOTES
-----
- --q-model should be an AWQ checkpoint with no flips applied. Generate it by
  running awq_js_xl.py with --no-heuristic.
"""

import os
import gc
import json
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ----------------------------------------------------------------------------
# Calibration data
# ----------------------------------------------------------------------------
def load_calibration_texts(tokenizer, n_total, seed=42):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    random.seed(seed)
    random.shuffle(texts)
    if len(texts) < n_total:
        print(f"  ⚠️  only {len(texts)} texts available, using all of them")
        n_total = len(texts)
    return texts[:n_total]


# ----------------------------------------------------------------------------
# Batched activation collector with per-layer token cap
# ----------------------------------------------------------------------------
class BatchedActCollector:
    """
    Collects activations only for a specified set of layer names, capping the
    total number of stored tokens per layer to `pool_tokens`.

    Stores in CPU fp16 to halve memory.
    """
    def __init__(self, model, target_names, pool_tokens=8192):
        self.model = model
        self.target_names = set(target_names)
        self.pool_tokens = pool_tokens
        self.acts = {n: [] for n in target_names}
        self.counts = {n: 0 for n in target_names}
        self.handles = []

    def _hook(self, name):
        def fn(_m, inp, _out):
            if self.counts[name] >= self.pool_tokens:
                return
            x = inp[0] if isinstance(inp, tuple) else inp
            x_flat = x.detach().reshape(-1, x.shape[-1])
            remaining = self.pool_tokens - self.counts[name]
            if x_flat.shape[0] > remaining:
                idx = torch.randperm(x_flat.shape[0])[:remaining]
                x_flat = x_flat[idx]
            self.acts[name].append(x_flat.to(dtype=torch.float16, device='cpu'))
            self.counts[name] += x_flat.shape[0]
        return fn

    def register(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) and name in self.target_names:
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self, name):
        if not self.acts.get(name):
            return None
        return torch.cat(self.acts[name], dim=0)

    def free(self, name):
        if name in self.acts:
            self.acts[name] = []
            self.counts[name] = 0

    def all_full(self):
        return all(self.counts[n] >= self.pool_tokens for n in self.target_names)


def run_forward_for_batch(model, tokenizer, texts, collector, device, max_length=2048):
    model.eval()
    collector.register()
    try:
        with torch.no_grad():
            for t in tqdm(texts, desc="    forward", leave=False):
                if collector.all_full():
                    break
                try:
                    enc = tokenizer(t, return_tensors="pt",
                                    truncation=True, max_length=max_length)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    model(**enc, use_cache=False, return_dict=True)
                except Exception:
                    continue
    finally:
        collector.remove()


# ----------------------------------------------------------------------------
# Module lookup helpers
# ----------------------------------------------------------------------------
def list_common_linear_names(fp_model, q_model, exclude=('lm_head', 'embed_tokens')):
    fp_names = {n for n, m in fp_model.named_modules() if isinstance(m, nn.Linear)}
    q_names = {n for n, m in q_model.named_modules() if isinstance(m, nn.Linear)}
    common = fp_names & q_names
    common = {n for n in common if not any(tok in n for tok in exclude)}
    return sorted(common)

# ----------------------------------------------------------------------------
# Flip on already-quantized weights
# ----------------------------------------------------------------------------
@torch.no_grad()
def flip_on_quantized(W_fp, W_q, mu_hat, group_size=128, bits=4,
                       max_flip_percent=0.01, knee_tolerance=0.0):
    device = W_q.device
    out_f, in_f = W_q.shape
    n_groups = (in_f + group_size - 1) // group_size
    padded = n_groups * group_size
    max_int = (1 << bits) - 1

    if padded > in_f:
        W_fp_p = torch.zeros(out_f, padded, device=device, dtype=W_q.dtype)
        W_fp_p[:, :in_f] = W_fp
        W_q_p = torch.zeros(out_f, padded, device=device, dtype=W_q.dtype)
        W_q_p[:, :in_f] = W_q
        mu_p = torch.zeros(padded, device=device, dtype=W_q.dtype)
        mu_p[:in_f] = mu_hat
    else:
        W_fp_p, W_q_p, mu_p = W_fp, W_q, mu_hat

    Wg = W_fp_p.reshape(out_f, n_groups, group_size)
    w_min = Wg.min(dim=2, keepdim=True)[0]
    w_max = Wg.max(dim=2, keepdim=True)[0]
    scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)

    scale_flat = scale.repeat(1, 1, group_size).reshape(out_f, padded)
    zp_flat = zp.repeat(1, 1, group_size).reshape(out_f, padded)

    W_int = torch.round(W_q_p / scale_flat + zp_flat).clamp(0, max_int)

    e_p = W_q_p - W_fp_p
    current_b = (e_p * mu_p.unsqueeze(0)).sum(dim=1)

    W_div = W_fp_p / scale_flat
    flip_dir = torch.sign(W_div + zp_flat - W_int)
    flip_dir[flip_dir == 0] = 1.0
    flip_impacts = mu_p.unsqueeze(0) * flip_dir * scale_flat

    target_sign = torch.sign(current_b).unsqueeze(1)
    valid = (torch.sign(flip_impacts) == target_sign)

    w_int_after = W_int + flip_dir
    in_range = (w_int_after >= 0) & (w_int_after <= max_int)
    valid = valid & in_range

    sorted_mu, _ = torch.sort(mu_p.abs(), descending=True)
    n = sorted_mu.numel()
    first_half = sorted_mu[:n // 2].cpu().float().numpy()
    if first_half.size >= 3:
        y = first_half
        y_min, y_max = y.min(), y.max()
        if y_max - y_min > 1e-10:
            y_n = (y - y_min) / (y_max - y_min)
            x_n = np.linspace(0, 1, y.size)
            line = y_n[0] + (y_n[-1] - y_n[0]) * x_n
            knee = int(np.argmax(np.abs(y_n - line)))
            knee = min(knee + int(knee_tolerance * n), n - 1)
            threshold = float(sorted_mu[knee].item())
        else:
            threshold = float(sorted_mu[int(0.05 * n)].item())
    else:
        threshold = float(sorted_mu[0].item())
    is_outlier = mu_p.abs() > threshold
    valid = valid & (~is_outlier).unsqueeze(0)

    rd = (W_div + zp_flat - W_int).abs()
    rd_masked = rd.clone()
    rd_masked[~valid] = -1.0
    sorted_idx = torch.argsort(rd_masked, dim=1, descending=True)

    sorted_impact = torch.gather(flip_impacts, 1, sorted_idx)
    sorted_valid = torch.gather(valid.long(), 1, sorted_idx)
    sorted_impact = sorted_impact * sorted_valid

    cumsum = torch.cumsum(sorted_impact, dim=1)
    resid = torch.abs(current_b.unsqueeze(1) - cumsum)
    init = torch.abs(current_b).unsqueeze(1)
    all_resid = torch.cat([init, resid], dim=1)
    best_k = torch.argmin(all_resid, dim=1)

    rng = torch.arange(padded, device=device).unsqueeze(0)
    flip_mask_sorted = rng < best_k.unsqueeze(1)
    flip_mask_sorted = flip_mask_sorted & sorted_valid.bool()

    sorted_dir = torch.gather(flip_dir, 1, sorted_idx)
    sorted_dir[~flip_mask_sorted] = 0.0

    max_flips_per_row = int(max_flip_percent * in_f)
    cumflips = flip_mask_sorted.long().cumsum(dim=1)
    within = cumflips <= max_flips_per_row
    sorted_dir[~within] = 0.0

    W_int.scatter_add_(1, sorted_idx, sorted_dir)
    W_int.clamp_(0, max_int)

    W_new = (W_int - zp_flat) * scale_flat
    if padded > in_f:
        W_new = W_new[:, :in_f]
    return W_new.to(W_q.dtype)


# ----------------------------------------------------------------------------
# Per-layer measurement
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_layer(W_fp, W_q, X_pool, n_train_list, n_val, repetitions,
                  group_size, bits, max_flip_percent, device='cuda'):
    N_pool, d = X_pool.shape
    out_f = W_q.shape[0]

    max_N = max(n_train_list)
    if N_pool < max_N + n_val:
        raise RuntimeError(f"pool {N_pool} < max_N {max_N} + n_val {n_val}")

    perm = torch.randperm(N_pool)
    val_idx = perm[:n_val]
    train_pool_idx = perm[n_val:]

    X_val = X_pool[val_idx].to(device).float()
    mu_val = X_val.mean(dim=0)
    tildeX_val = X_val - mu_val.unsqueeze(0)

    W_fp_d = W_fp.to(device)
    W_q_d = W_q.to(device)
    e = W_q_d - W_fp_d

    r_cent = tildeX_val @ e.t()
    L_cent = (r_cent.pow(2).sum() / n_val).item()
    r_full = X_val @ e.t()
    L_rec_val = (r_full.pow(2).sum() / n_val).item()
    L_fmr_val = (mu_val @ e.t()).pow(2).sum().item()

    results = []
    for N in n_train_list:
        if train_pool_idx.numel() < N:
            continue
        for rep in range(repetitions):
            rep_perm = torch.randperm(train_pool_idx.numel())[:N]
            train_idx = train_pool_idx[rep_perm]
            X_train = X_pool[train_idx].to(device).float()
            mu_hat = X_train.mean(dim=0)

            # BC
            b_BC = (mu_hat @ e.t())
            BC_train_bias = 0.0
            BC_val_bias = ((mu_val @ e.t()) - b_BC).pow(2).sum().item()

            # Flip
            W_q_flipped = flip_on_quantized(
                W_fp_d, W_q_d, mu_hat,
                group_size=group_size, bits=bits,
                max_flip_percent=max_flip_percent
            )
            e_flip = W_q_flipped - W_fp_d
            Flip_train_bias = (mu_hat @ e_flip.t()).pow(2).sum().item()
            Flip_val_bias = (mu_val @ e_flip.t()).pow(2).sum().item()

            r_BC = (X_val @ e.t()) - b_BC.unsqueeze(0)
            L_rec_BC_val = (r_BC.pow(2).sum() / n_val).item()
            r_Flip = X_val @ e_flip.t()
            L_rec_Flip_val = (r_Flip.pow(2).sum() / n_val).item()

            results.append({
                'N': N, 'rep': rep,
                'BC_train_bias': BC_train_bias,
                'BC_val_bias': BC_val_bias,
                'Flip_train_bias': Flip_train_bias,
                'Flip_val_bias': Flip_val_bias,
                'L_rec_BC_val': L_rec_BC_val,
                'L_rec_Flip_val': L_rec_Flip_val,
            })

            del X_train, W_q_flipped, e_flip, b_BC
            if device == 'cuda':
                torch.cuda.empty_cache()

    del X_val, mu_val, tildeX_val, W_fp_d, W_q_d, e, r_cent, r_full
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'L_cent': L_cent,
        'L_rec_val_uncorrected': L_rec_val,
        'L_fmr_val_uncorrected': L_fmr_val,
        'n_val': n_val,
        'd': d,
        'out_features': out_f,
        'measurements': results,
    }


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_plots(records, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not available; skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # E1
    fig, ax = plt.subplots(figsize=(7, 7))
    predicted = np.array([r['L_cent'] * (1.0 / r['N'] + 1.0 / r['n_val']) for r in records])
    measured = np.array([r['BC_val_bias'] for r in records])
    ax.loglog(predicted, measured, 'o', alpha=0.3, markersize=4)
    lo = min(predicted.min(), measured.min())
    hi = max(predicted.max(), measured.max())
    ax.loglog([lo, hi], [lo, hi], 'k--', label='y = x (theory)')
    ax.set_xlabel('predicted BC val-bias  =  L_cent · (1/N + 1/N_val)')
    ax.set_ylabel('measured BC val-bias')
    ax.set_title('E1: BC val-bias vs theoretical prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'E1_bc_prediction.png'), dpi=140)
    plt.close(fig)

    slope = (measured * predicted).sum() / (predicted ** 2).sum()
    ss_res = ((measured - slope * predicted) ** 2).sum()
    ss_tot = ((measured - measured.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"  E1: BC val-bias ≈ {slope:.4f} × predicted   R² = {r2:.4f}")

    # E3
    by_N = defaultdict(lambda: {'BC': [], 'Flip': []})
    for r in records:
        by_N[r['N']]['BC'].append(r['BC_val_bias'])
        by_N[r['N']]['Flip'].append(r['Flip_val_bias'])

    Ns = sorted(by_N.keys())
    BC_med = [np.median(by_N[n]['BC']) for n in Ns]
    BC_p25 = [np.percentile(by_N[n]['BC'], 25) for n in Ns]
    BC_p75 = [np.percentile(by_N[n]['BC'], 75) for n in Ns]
    Flip_med = [np.median(by_N[n]['Flip']) for n in Ns]
    Flip_p25 = [np.percentile(by_N[n]['Flip'], 25) for n in Ns]
    Flip_p75 = [np.percentile(by_N[n]['Flip'], 75) for n in Ns]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(Ns, BC_med, yerr=[np.array(BC_med) - np.array(BC_p25),
                                   np.array(BC_p75) - np.array(BC_med)],
                fmt='o-', label='BC val-bias (median ± IQR)', capsize=4)
    ax.errorbar(Ns, Flip_med, yerr=[np.array(Flip_med) - np.array(Flip_p25),
                                     np.array(Flip_p75) - np.array(Flip_med)],
                fmt='s-', label='Flip val-bias (median ± IQR)', capsize=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N (training pool size for correction)')
    ax.set_ylabel('val-bias  =  sum_j (μ_val^T ẽ_j)^2')
    ax.set_title('E3: BC vs Flip val-bias across calibration sizes')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'E3_crossover.png'), dpi=140)
    plt.close(fig)

    print(f"  Plots: {output_dir}/E1_bc_prediction.png, E3_crossover.png")


# ----------------------------------------------------------------------------
# Main (v2: batched layer processing)
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model", required=True)
    ap.add_argument("--q-model", required=True,
                    help="AWQ checkpoint with no flip (run awq_js_xl.py --no-heuristic)")
    ap.add_argument("--output-dir", default="./verify_bc_results")
    ap.add_argument("--n-train-list", type=int, nargs='+', default=[32, 64, 128, 256])
    ap.add_argument("--n-val", type=int, default=4096,
                    help="Held-out val pool size in tokens.")
    ap.add_argument("--pool-tokens", type=int, default=8192,
                    help="Hard cap on tokens stored per layer. Must be >= n_val + max(n_train_list).")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--n-calib-seqs", type=int, default=64,
                    help="Number of calibration SEQUENCES to load. With seqs of length 2048, "
                         "a few dozen will saturate pool_tokens for most layers.")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--max-flip-percent", type=float, default=0.01)
    ap.add_argument("--layer-batch-size", type=int, default=8,
                    help="Number of linear layers to collect activations for per forward pass.")
    ap.add_argument("--layer-filter", type=str, default=None)
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("verify_bc_overfit.py  (v2 — memory-bounded)")
    print("=" * 80)
    print(f"  FP model:        {args.fp_model}")
    print(f"  Q  model:        {args.q_model}")
    print(f"  N train list:    {args.n_train_list}")
    print(f"  N val:           {args.n_val}")
    print(f"  Pool tokens:     {args.pool_tokens}")
    print(f"  Reps:            {args.repetitions}")
    print(f"  Calib seqs:      {args.n_calib_seqs} (max_length {args.max_length})")
    print(f"  Layer batch:     {args.layer_batch_size}")
    print(f"  Flip budget:     {args.max_flip_percent*100:.2f}%")
    print("=" * 80)

    required_pool = args.n_val + max(args.n_train_list)
    if args.pool_tokens < required_pool:
        print(f"❌ pool_tokens={args.pool_tokens} < required {required_pool}")
        return

    # --- 1. Load tokenizer & FP model (kept on GPU for forwards) ---
    print("\n[1/3] Loading FP model")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    fp_model.eval()

    # --- 2. Load Q model to CPU (we only need its weights) ---
    print("\n[2/3] Loading quantized model (CPU)")
    q_model = AutoModelForCausalLM.from_pretrained(
        args.q_model, dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    q_model.eval()

    # --- 3. List target layers ---
    layer_names = list_common_linear_names(fp_model, q_model)
    if args.layer_filter:
        layer_names = [n for n in layer_names if args.layer_filter in n]
    if args.max_layers:
        layer_names = layer_names[:args.max_layers]
    print(f"  Target layers: {len(layer_names)}")

    texts = load_calibration_texts(tokenizer, n_total=args.n_calib_seqs, seed=args.seed)

    fp_mod_dict = dict(fp_model.named_modules())
    q_mod_dict = dict(q_model.named_modules())

    # --- 4. Per-batch loop ---
    print("\n[3/3] Streaming forward passes by layer-batch")
    flat_records = []
    per_layer_summary = []

    n_batches = (len(layer_names) + args.layer_batch_size - 1) // args.layer_batch_size
    for batch_idx in range(n_batches):
        b_start = batch_idx * args.layer_batch_size
        b_end = min(b_start + args.layer_batch_size, len(layer_names))
        batch_layers = layer_names[b_start:b_end]
        print(f"\n[Batch {batch_idx+1}/{n_batches}]  layers {b_start}-{b_end-1}")

        collector = BatchedActCollector(fp_model, batch_layers, pool_tokens=args.pool_tokens)
        run_forward_for_batch(fp_model, tokenizer, texts, collector, args.device,
                              max_length=args.max_length)

        for name in tqdm(batch_layers, desc="  measuring", leave=False):
            X = collector.stacked(name)
            if X is None or X.shape[0] < required_pool:
                avail = 0 if X is None else X.shape[0]
                print(f"    ⚠️  {name}: only {avail} tokens, need {required_pool}; skipping")
                collector.free(name)
                continue

            fp_m = fp_mod_dict.get(name)
            q_m = q_mod_dict.get(name)
            if fp_m is None or q_m is None:
                collector.free(name)
                continue
            W_fp = fp_m.weight.data.detach().cpu().float()
            W_q = q_m.weight.data.detach().cpu().float()
            if W_fp.shape != W_q.shape:
                print(f"    ⚠️  {name}: shape mismatch; skipping")
                collector.free(name)
                continue

            try:
                res = measure_layer(
                    W_fp, W_q, X,
                    n_train_list=args.n_train_list,
                    n_val=args.n_val,
                    repetitions=args.repetitions,
                    group_size=args.group_size,
                    bits=args.bits,
                    max_flip_percent=args.max_flip_percent,
                    device=args.device,
                )
            except Exception as ex:
                print(f"    ⚠️  {name}: {ex}")
                collector.free(name)
                continue

            per_layer_summary.append({
                'layer': name,
                'd': res['d'],
                'out': res['out_features'],
                'L_cent': res['L_cent'],
                'L_fmr_val_uncorrected': res['L_fmr_val_uncorrected'],
                'L_rec_val_uncorrected': res['L_rec_val_uncorrected'],
            })
            for m in res['measurements']:
                flat_records.append({
                    'layer': name,
                    'd': res['d'],
                    'L_cent': res['L_cent'],
                    'n_val': res['n_val'],
                    **m,
                })

            collector.free(name)
            del W_fp, W_q, X
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()

        del collector
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    if not flat_records:
        print("\n❌ No records collected.")
        return

    # --- Write outputs ---
    import csv
    flat_csv = os.path.join(args.output_dir, 'measurements.csv')
    with open(flat_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat_records[0].keys()))
        w.writeheader()
        for r in flat_records:
            w.writerow(r)

    summary_csv = os.path.join(args.output_dir, 'per_layer_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer_summary[0].keys()))
        w.writeheader()
        for r in per_layer_summary:
            w.writerow(r)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n  Wrote: {flat_csv}")
    print(f"  Wrote: {summary_csv}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    make_plots(flat_records, args.output_dir)

    by_N = defaultdict(lambda: {'BC': [], 'Flip': [], 'pred': []})
    for r in flat_records:
        by_N[r['N']]['BC'].append(r['BC_val_bias'])
        by_N[r['N']]['Flip'].append(r['Flip_val_bias'])
        by_N[r['N']]['pred'].append(r['L_cent'] * (1.0 / r['N'] + 1.0 / r['n_val']))

    print(f"\n{'N':>6} {'BC_med':>14} {'Flip_med':>14} {'pred':>14}  "
          f"{'BC/pred':>10}  {'Flip/BC':>10}")
    for N in sorted(by_N.keys()):
        bc_m = np.median(by_N[N]['BC'])
        flip_m = np.median(by_N[N]['Flip'])
        pred_m = np.median(by_N[N]['pred'])
        ratio_bc = bc_m / pred_m if pred_m > 0 else float('nan')
        ratio_fb = flip_m / bc_m if bc_m > 0 else float('nan')
        print(f"{N:>6} {bc_m:>14.4e} {flip_m:>14.4e} {pred_m:>14.4e}  "
              f"{ratio_bc:>10.3f}  {ratio_fb:>10.3f}")

    print(f"\n  Interpretation:")
    print(f"   - BC/pred ≈ 1.0  → E1 confirmed (BC overfit matches theory)")
    print(f"   - Flip/BC < 1.0  → Flip beats BC at this N")
    print(f"   - Flip/BC > 1.0  → BC beats Flip at this N")


if __name__ == "__main__":
    main()