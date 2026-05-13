"""
verify_bc_overfit.py
====================

Tests two predictions of the "BC overfits, Flip generalizes" theory:

  E1. BC's expected val-bias equals L_cent / N exactly.
      Specifically: E[ sum_j ((mu* - mu_hat)^T e_j)^2 ] = (1/N) * sum_j e_j^T Sigma e_j
                                                         = L_cent / N.
      We measure BC val-bias across many train/val splits and compare to L_cent/N.

  E3. As N decreases, BC's val-bias grows as 1/N; Flip's val-bias stays roughly
      flat (bounded by its budget-imposed irreducible residual). There exists a
      crossover N* below which Flip beats BC.

SETUP
-----
We freeze an already-quantized checkpoint (so e_j = W_q - W is fixed).
We isolate the *correction step* by varying only the calibration pool used to
compute mu_hat for BC/Flip. AWQ scales were chosen on a SEPARATE pool (the
quantized checkpoint is the result), so the AWQ choice does NOT co-vary with
our N-sweep — this is what isolates the correction-step generalization.

For each N in {32, 64, 128, 256}:
  for r in repetitions:
    sample N training points -> mu_hat
    BC: b_j = mu_hat^T e_j     (absorbs into a virtual bias)
    Flip: greedy budgeted flips on W_q using mu_hat
    measure on a large held-out val pool:
      BC_val_bias  = sum_j (mu_val^T e_j - b_j)^2
      Flip_val_bias = sum_j (mu_val^T tilde_e_j)^2

For each layer we ALSO compute L_cent = (1/|V|) ||tilde_X_val @ e^T||_F^2,
the theoretical BC val-bias prediction at sample size N being L_cent / N.

USAGE
-----
python verify_bc_overfit.py \\
    --fp-model ./models/Mistral-7B-v0.3 \\
    --q-model  ./quantized_models/model_awq_no_flip \\
    --output-dir ./verify_bc_results \\
    --n-train-list 32 64 128 256 \\
    --n-val 2048 \\
    --repetitions 5

NOTES
-----
- q-model should be an AWQ checkpoint with NO flipping (run awq_js_xl.py
  with --no-heuristic so we have a clean "post-projection" W_q to operate on).
- The Flip step here reimplements the same greedy logic from awq_js_xl.py
  but operates on already-quantized W_q (it re-derives the per-group scale/zp
  from W_q so it can take integer-level flips).
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
    """Load a pool of calibration texts. We need enough to split N_train + N_val."""
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    random.seed(seed)
    random.shuffle(texts)
    if len(texts) < n_total:
        raise RuntimeError(f"Not enough calibration texts: have {len(texts)}, need {n_total}")
    return texts[:n_total]


# ----------------------------------------------------------------------------
# Activation collection (collects per-token vectors per layer)
# ----------------------------------------------------------------------------
class ActCollector:
    def __init__(self, model, max_tokens=256):
        self.model = model
        self.max_tokens = max_tokens
        self.acts = {}                                  # name -> list of [N_tok, d] cpu fp32
        self.handles = []

    def hook(self, name):
        def _hook(_m, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3 and x.shape[1] > self.max_tokens:
                idx = torch.randperm(x.shape[1])[:self.max_tokens].sort()[0]
                x = x[:, idx, :]
            self.acts.setdefault(name, []).append(
                x.detach().reshape(-1, x.shape[-1]).cpu().float()
            )
        return _hook

    def register(self):
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self.hook(n)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_stacked(self, name):
        if name not in self.acts or not self.acts[name]:
            return None
        return torch.cat(self.acts[name], dim=0)

    def clear(self):
        self.acts = {}


def run_forward(model, tokenizer, texts, collector, device, max_length=512):
    model.eval()
    collector.register()
    try:
        with torch.no_grad():
            for t in tqdm(texts, desc="    forward", leave=False):
                try:
                    enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    model(**enc, use_cache=False, return_dict=True)
                except Exception:
                    continue
    finally:
        collector.remove()


# ----------------------------------------------------------------------------
# Quantization error extraction
# ----------------------------------------------------------------------------
def collect_q_errors(fp_model, q_model):
    """Return name -> (W_fp, W_q, e=W_q-W_fp) all CPU float32. Same shape, same name."""
    fp_mods = dict(fp_model.named_modules())
    q_mods = dict(q_model.named_modules())
    out = {}
    for name, fp_m in fp_mods.items():
        if not isinstance(fp_m, nn.Linear):
            continue
        if name not in q_mods or not isinstance(q_mods[name], nn.Linear):
            continue
        W_fp = fp_m.weight.data.detach().cpu().float()
        W_q = q_mods[name].weight.data.detach().cpu().float()
        if W_fp.shape != W_q.shape:
            continue
        out[name] = (W_fp, W_q, W_q - W_fp)
    return out


# ----------------------------------------------------------------------------
# Flip on already-quantized weights
# ----------------------------------------------------------------------------
@torch.no_grad()
def flip_on_quantized(W_fp, W_q, mu_hat, group_size=128, bits=4,
                       max_flip_percent=0.01, knee_tolerance=0.0):
    """
    Apply the same greedy Flip algorithm from awq_js_xl.py to an already-quantized
    W_q. We re-derive the per-group scale/zp from (W_fp, W_q) so we know what an
    "integer flip" looks like.

    Returns: W_q_flipped [out, d] on same device as input.
    """
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

    # Re-derive per-group scale/zp by matching W_q = (W_int - zp) * scale
    # We know W_q values; recover scale as (max - min) / max_int over each group
    Wg = W_fp_p.reshape(out_f, n_groups, group_size)
    w_min = Wg.min(dim=2, keepdim=True)[0]
    w_max = Wg.max(dim=2, keepdim=True)[0]
    scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)

    scale_flat = scale.repeat(1, 1, group_size).reshape(out_f, padded)
    zp_flat = zp.repeat(1, 1, group_size).reshape(out_f, padded)

    # Recover integer codes from W_q
    W_int = torch.round(W_q_p / scale_flat + zp_flat).clamp(0, max_int)

    # Current bias per output channel: b_j = mu^T e_j
    e_p = W_q_p - W_fp_p
    current_b = (e_p * mu_p.unsqueeze(0)).sum(dim=1)        # [out]

    # Flip direction in fp space is sign(W_fp / scale + zp - W_int) — i.e.
    # which side of the rounding boundary the true fp value lies on.
    W_div = W_fp_p / scale_flat
    flip_dir = torch.sign(W_div + zp_flat - W_int)
    flip_dir[flip_dir == 0] = 1.0
    flip_impacts = mu_p.unsqueeze(0) * flip_dir * scale_flat  # [out, padded]

    # Sign filter: only consider flips that reduce |b_j|
    target_sign = torch.sign(current_b).unsqueeze(1)
    valid = (torch.sign(flip_impacts) == target_sign)

    # In-range check after flipping
    w_int_after = W_int + flip_dir
    in_range = (w_int_after >= 0) & (w_int_after <= max_int)
    valid = valid & in_range

    # Knee-based outlier mask on |mu|
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

    # Sort by rounding deviation (descending: most "boundary-adjacent" first)
    rd = (W_div + zp_flat - W_int).abs()
    rd_masked = rd.clone()
    rd_masked[~valid] = -1.0
    sorted_idx = torch.argsort(rd_masked, dim=1, descending=True)

    sorted_impact = torch.gather(flip_impacts, 1, sorted_idx)
    sorted_valid = torch.gather(valid.long(), 1, sorted_idx)
    sorted_impact = sorted_impact * sorted_valid

    # Cumulative residual after k flips: |current_b - cumsum_k sorted_impact|
    cumsum = torch.cumsum(sorted_impact, dim=1)
    resid = torch.abs(current_b.unsqueeze(1) - cumsum)
    init = torch.abs(current_b).unsqueeze(1)
    all_resid = torch.cat([init, resid], dim=1)
    best_k = torch.argmin(all_resid, dim=1)

    # Build flip mask under per-row budget
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
# Core measurement per layer
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_layer(W_fp, W_q, X_pool, n_train_list, n_val, repetitions,
                  group_size, bits, max_flip_percent, device='cuda'):
    """
    For one layer:
      - X_pool: [N_pool, d] cpu fp32, calibration pool (will be split per-rep)
      - returns dict: per (N, rep) measurements for both BC and Flip
                      plus L_cent computed on full val
    """
    N_pool, d = X_pool.shape
    out_f = W_q.shape[0]

    # Validate we have enough data
    max_N = max(n_train_list)
    if N_pool < max_N + n_val:
        raise RuntimeError(f"Pool {N_pool} < max_N {max_N} + n_val {n_val}")

    # Hold out val pool — fixed across all reps and N
    perm = torch.randperm(N_pool)
    val_idx = perm[:n_val]
    train_pool_idx = perm[n_val:]   # remaining used for training, sampled per-rep

    X_val = X_pool[val_idx].to(device)            # [n_val, d]
    mu_val = X_val.mean(dim=0)                    # [d]
    tildeX_val = X_val - mu_val.unsqueeze(0)      # centered

    W_fp_d = W_fp.to(device)
    W_q_d = W_q.to(device)
    e = W_q_d - W_fp_d                            # [out, d]

    # L_cent on val: (1/n_val) * ||tildeX_val @ e^T||_F^2
    r_cent = tildeX_val @ e.t()                   # [n_val, out]
    L_cent = (r_cent.pow(2).sum() / n_val).item()

    # Full reconstruction loss on val too
    r_full = X_val @ e.t()
    L_rec_val = (r_full.pow(2).sum() / n_val).item()

    # First-moment loss using val mean directly
    L_fmr_val = (mu_val @ e.t()).pow(2).sum().item()

    results = []
    for N in n_train_list:
        if train_pool_idx.numel() < N:
            continue
        for rep in range(repetitions):
            # Resample train of size N from train_pool_idx (without replacement within rep)
            rep_perm = torch.randperm(train_pool_idx.numel())[:N]
            train_idx = train_pool_idx[rep_perm]
            X_train = X_pool[train_idx].to(device)
            mu_hat = X_train.mean(dim=0)

            # === BC: bias correction b_j = mu_hat^T e_j ===
            b_BC = (mu_hat @ e.t())                              # [out]
            BC_train_bias = 0.0                                  # exact zero by construction
            # On val: residual is (mu_val^T e_j - b_BC[j])
            BC_val_bias = ((mu_val @ e.t()) - b_BC).pow(2).sum().item()

            # === Flip: greedy budgeted on W_q using mu_hat ===
            W_q_flipped = flip_on_quantized(
                W_fp_d, W_q_d, mu_hat,
                group_size=group_size, bits=bits,
                max_flip_percent=max_flip_percent
            )
            e_flip = W_q_flipped - W_fp_d
            Flip_train_bias = (mu_hat @ e_flip.t()).pow(2).sum().item()
            Flip_val_bias = (mu_val @ e_flip.t()).pow(2).sum().item()

            # Also: full L_rec on val for both, since that's what matters end-to-end
            r_BC = (X_val @ e.t()) - b_BC.unsqueeze(0)
            L_rec_BC_val = (r_BC.pow(2).sum() / n_val).item()
            r_Flip = X_val @ e_flip.t()
            L_rec_Flip_val = (r_Flip.pow(2).sum() / n_val).item()

            results.append({
                'N': N,
                'rep': rep,
                'BC_train_bias': BC_train_bias,
                'BC_val_bias': BC_val_bias,
                'Flip_train_bias': Flip_train_bias,
                'Flip_val_bias': Flip_val_bias,
                'L_rec_BC_val': L_rec_BC_val,
                'L_rec_Flip_val': L_rec_Flip_val,
            })

            del X_train, W_q_flipped, e_flip
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
    """records: list of dicts, one per (layer, N, rep) with all relevant fields."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not available; skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # E1: BC val-bias vs predicted L_cent * (1/N + 1/N_val)
    # The val mean is itself an estimate, so the residual variance has TWO sources:
    # train-side noise (1/N) and val-side noise (1/N_val). With N_val >> N the
    # second term is small but not zero — important for the cleanest fit.
    fig, ax = plt.subplots(figsize=(7, 7))
    n_val_used = records[0].get('n_val', None)
    if n_val_used is None:
        # fall back; will be set below by caller
        n_val_used = 2048
    predicted = np.array([r['L_cent'] * (1.0 / r['N'] + 1.0 / n_val_used) for r in records])
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

    # E1 (linear) — slope and R^2
    slope = (measured * predicted).sum() / (predicted ** 2).sum()
    ss_res = ((measured - slope * predicted) ** 2).sum()
    ss_tot = ((measured - measured.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"  E1: BC val-bias ≈ {slope:.4f} × (L_cent / N)   R² = {r2:.4f}")

    # E3: BC vs Flip val-bias as a function of N (aggregated across layers)
    # Per-N: median across (layer, rep)
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

    print(f"  Plots written to {output_dir}/")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model", required=True)
    ap.add_argument("--q-model", required=True, help="AWQ checkpoint, no flip (run awq_js_xl.py --no-heuristic)")
    ap.add_argument("--output-dir", default="./verify_bc_results")
    ap.add_argument("--n-train-list", type=int, nargs='+', default=[32, 64, 128, 256])
    ap.add_argument("--n-val", type=int, default=2048,
                    help="Held-out val pool size (in tokens, after flattening)")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--n-pool-samples", type=int, default=512,
                    help="Number of calibration sequences to load (will yield ~n_pool_samples * max_tokens tokens)")
    ap.add_argument("--max-tokens-per-sample", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--max-flip-percent", type=float, default=0.01)
    ap.add_argument("--layer-filter", type=str, default=None,
                    help="If set, only process layers whose name contains this substring")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="If set, limit to first N layers (for debugging)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("verify_bc_overfit.py — E1 + E3")
    print("=" * 80)
    print(f"  FP model:    {args.fp_model}")
    print(f"  Q  model:    {args.q_model}")
    print(f"  N values:    {args.n_train_list}")
    print(f"  N val:       {args.n_val}")
    print(f"  Reps:        {args.repetitions}")
    print(f"  Pool seqs:   {args.n_pool_samples}")
    print(f"  Flip budget: {args.max_flip_percent*100:.2f}%")
    print("=" * 80)

    # --- 1. Load models, tokenizer, calibration ---
    print("\n[1/4] Loading FP model + tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    fp_model.eval()

    print("\n[2/4] Forward pass to collect activations")
    texts = load_calibration_texts(tokenizer, n_total=args.n_pool_samples, seed=args.seed)
    collector = ActCollector(fp_model, max_tokens=args.max_tokens_per_sample)
    run_forward(fp_model, tokenizer, texts, collector, args.device, max_length=args.max_length)

    pool_acts = {n: collector.get_stacked(n) for n in list(collector.acts.keys())}
    collector.clear()
    pool_acts = {n: a for n, a in pool_acts.items() if a is not None}
    print(f"  Activations for {len(pool_acts)} layers (one tensor each, [N_pool_tokens, d])")

    # Move FP model to CPU, load Q model
    fp_model = fp_model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()

    print("\n[3/4] Loading quantized model and extracting weight pairs")
    q_model = AutoModelForCausalLM.from_pretrained(
        args.q_model, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True
    )
    q_model.eval()

    weight_pairs = collect_q_errors(fp_model, q_model)
    print(f"  Weight pairs for {len(weight_pairs)} layers")

    del fp_model, q_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- 4. Per-layer measurement ---
    print("\n[4/4] Per-layer measurement")
    layer_names = [n for n in weight_pairs if n in pool_acts]
    if args.layer_filter:
        layer_names = [n for n in layer_names if args.layer_filter in n]
    if args.max_layers:
        layer_names = layer_names[:args.max_layers]
    print(f"  Will process {len(layer_names)} layers")

    flat_records = []
    per_layer_summary = []

    for name in tqdm(layer_names, desc="layers"):
        W_fp, W_q, _ = weight_pairs[name]
        X_pool = pool_acts[name]
        N_pool = X_pool.shape[0]
        if N_pool < args.n_val + max(args.n_train_list):
            print(f"  ⚠️  Skip {name}: pool {N_pool} too small")
            continue
        try:
            res = measure_layer(
                W_fp, W_q, X_pool,
                n_train_list=args.n_train_list,
                n_val=args.n_val,
                repetitions=args.repetitions,
                group_size=args.group_size,
                bits=args.bits,
                max_flip_percent=args.max_flip_percent,
                device=args.device,
            )
        except Exception as ex:
            print(f"\n  ⚠️  {name}: {ex}")
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

        # Free this layer's activations
        pool_acts[name] = None
        gc.collect()

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

    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n  Wrote: {flat_csv}")
    print(f"  Wrote: {summary_csv}")

    # --- Plots + summary stats ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    make_plots(flat_records, args.output_dir)

    # Per-N aggregate table
    by_N = defaultdict(lambda: {'BC': [], 'Flip': [], 'pred': []})
    for r in flat_records:
        by_N[r['N']]['BC'].append(r['BC_val_bias'])
        by_N[r['N']]['Flip'].append(r['Flip_val_bias'])
        # Theory: BC_val_bias ≈ L_cent · (1/N + 1/N_val)
        by_N[r['N']]['pred'].append(r['L_cent'] * (1.0 / r['N'] + 1.0 / r['n_val']))

    print(f"\n{'N':>6} {'BC_med':>14} {'Flip_med':>14} {'predicted':>14}  "
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
    print(f"   - Flip/BC < 1.0  → Flip beats BC at this N (a crossover exists below it)")
    print(f"   - Flip/BC > 1.0  → BC beats Flip at this N (we're above the crossover)")


if __name__ == "__main__":
    main()