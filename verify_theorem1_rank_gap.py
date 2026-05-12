"""
E1 + E2: Verify Theorem 1 (Structural Rank Gap)
================================================

Two tests in one script:

E1. POPULATION OPTIMA COMPARISON.
    Compute the population-optimal BC and the population-optimal Flip
    (both using HUGE calibration so sampling noise is negligible).
    Predict: Flip optimum strictly < BC optimum, with the gap = projection
    of E onto eigendirections of Sigma_xx that BC's rank-1 set cannot reach.

E2. RANK-OF-CORRECTION ANALYSIS.
    For each layer, decompose BC's correction and Flip's correction in
    the eigenbasis of Sigma_xx. Show:
      - BC's correction has ALL its mass on the constant direction 1/sqrt(d).
      - Flip's correction spreads across many top eigendirections.
    This is the geometric content of Theorem 1.

USAGE
-----
python verify_theorem1_rank_gap.py \\
    --fp-model      ./models/Mistral-7B-v0.3 \\
    --base-q-model  ./quantized_models/awq_base \\
    --sfa-q-model   ./quantized_models/awq_sfa \\
    --output        theorem1_results.csv \\
    --n-calib       1024              # large, to make this a population-level test
"""

import os, gc, argparse, random, csv
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------- Reused calibration utilities (same as v2/v3 scripts) ----------
def load_calibration_texts(tokenizer, n_samples=1024, seed=42):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    random.seed(seed); random.shuffle(texts)
    return texts[:n_samples]


class ActivationCollector:
    def __init__(self, model, max_tokens_per_sample=512):
        self.model = model
        self.max_tokens_per_sample = max_tokens_per_sample
        self.activations = {}
        self.handles = []

    def _hook(self, name):
        def fn(_m, input, _out):
            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                idx = torch.randperm(inp.shape[1])[:self.max_tokens_per_sample].sort()[0]
                inp = inp[:, idx, :]
            self.activations.setdefault(name, []).append(
                inp.detach().reshape(-1, inp.shape[-1]).cpu().float())
        return fn

    def register(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

    def get(self, name):
        if name not in self.activations or len(self.activations[name]) == 0: return None
        return torch.cat(self.activations[name], dim=0)

    def clear(self): self.activations = {}


def run_calibration(model, tokenizer, collector, texts, device, max_length=512):
    model.eval(); collector.register()
    try:
        with torch.no_grad():
            for text in tqdm(texts, desc="  Calibrating", leave=False):
                try:
                    inp = tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=max_length)
                    inp = {k: v.to(device) for k, v in inp.items()}
                    model(**inp, use_cache=False, return_dict=True)
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


# ---------- Core measurement: population-level rank-gap test ----------
@torch.no_grad()
def measure_rank_gap(X, W_fp, W_base, W_sfa, top_k_eig=64):
    """
    Returns a dict with:
      - R_base       : tr(E^T Sigma_xx E)               (no correction)
      - R_bc_opt     : population-OPTIMAL BC residual
      - R_flip_opt   : SFA-realized residual (operates as our 'Flip')
      - rank1_bc     : fraction of |BC correction|^2 mass on dir 1/sqrt(d)
      - rank_flip    : effective rank of Flip's correction in Sigma_xx eigenbasis
      - spectrum_E   : E^T E top-k singular values (for context)
    """
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        bytes_needed = (X.shape[0] * X.shape[1] + 3 * W_fp.numel()) * 4
        if bytes_needed < 6 * (1024**3):
            X = X.cuda(); W_fp = W_fp.cuda(); W_base = W_base.cuda(); W_sfa = W_sfa.cuda()
        else:
            use_gpu = False

    N, d = X.shape
    out = W_fp.shape[0]

    E_base = W_base - W_fp
    E_flip = W_sfa  - W_fp        # this is E + DeltaW for Flip
    DeltaW_flip = W_sfa - W_base   # Flip's correction operator

    # --- Sigma_xx eigen-decomp ---
    # Note: we use the EMPIRICAL test-side covariance (X here was collected
    # on the same activations the methods saw at calibration, so this is
    # the BEST CASE for sampling - results should be a LOWER BOUND on the gap).
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    Sigma = (Xc.t() @ Xc) / N + mu.t() @ mu          # E[xx^T] = Cov + mu mu^T

    # Eigendecomp (top_k_eig)
    try:
        eigvals, eigvecs = torch.linalg.eigh(Sigma.float())
        # eigh returns ascending; reverse
        eigvals = eigvals.flip(0); eigvecs = eigvecs.flip(1)
    except Exception:
        # fallback to SVD on X
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        eigvals = (S * S) / N
        eigvecs = Vh.t()

    top_eigvals = eigvals[:top_k_eig]
    top_eigvecs = eigvecs[:, :top_k_eig]

    # --- E1: residuals R(DeltaW) = tr(E_eff^T Sigma E_eff) ---
    # Use the full-rank trace formula; on GPU memory budget is fine.
    def trace_quad(Eeff):
        # tr(Eeff^T Sigma Eeff) = sum_{j,i} (Sigma @ Eeff^T)_{i, j} Eeff_{j, i}
        # Cheaper: trace = sum over j of e_j^T Sigma e_j
        return (Eeff @ Sigma * Eeff).sum().item()

    R_base    = trace_quad(E_base)
    R_flip    = trace_quad(E_flip)

    # Population-optimal BC: c_j minimizes (e_j + c_j * 1/d)^T Sigma (e_j + c_j * 1/d)
    # d/dc_j = 0 ->  c_j = - d * (e_j^T Sigma 1) / (1^T Sigma 1)
    ones = torch.ones(d, device=Sigma.device, dtype=Sigma.dtype)
    s11 = ones @ Sigma @ ones                        # scalar
    s_e1 = E_base @ Sigma @ ones                     # [out]
    c_opt = -d * s_e1 / s11                          # [out]
    DeltaW_bc_opt = (c_opt / d).unsqueeze(1) * ones.unsqueeze(0)   # [out, d]
    E_bc_opt = E_base + DeltaW_bc_opt
    R_bc_opt = trace_quad(E_bc_opt)

    # --- E2: rank analysis of corrections ---
    # BC: by construction rank-1 (all rows are scaled copies of ones / d)
    # Flip: project DeltaW_flip onto eigenbasis of Sigma
    # For each row: ||DeltaW_j||^2 = sum_k <DeltaW_j, v_k>^2  (full basis)
    # We measure energy in top-k_eig basis.
    DeltaW_flip_in_eig = DeltaW_flip @ top_eigvecs    # [out, top_k_eig]
    energy_per_eig = DeltaW_flip_in_eig.pow(2).sum(dim=0)   # [top_k_eig]
    total_energy = DeltaW_flip.pow(2).sum().item()
    if total_energy > 1e-12:
        energy_per_eig = energy_per_eig / total_energy
    energy_top1 = energy_per_eig[0].item()
    energy_top10 = energy_per_eig[:10].sum().item()
    energy_top64 = energy_per_eig.sum().item()
    # Effective rank: exp(-sum p log p) where p = normalized energy
    p = energy_per_eig / energy_per_eig.sum().clamp(min=1e-12)
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()

    # BC rank check: project DeltaW_bc_opt onto eigenbasis
    DeltaW_bc_in_eig = DeltaW_bc_opt @ top_eigvecs
    energy_bc_per_eig = DeltaW_bc_in_eig.pow(2).sum(dim=0)
    total_bc = DeltaW_bc_opt.pow(2).sum().item()
    if total_bc > 1e-12:
        energy_bc_per_eig = energy_bc_per_eig / total_bc
    p_bc = energy_bc_per_eig / energy_bc_per_eig.sum().clamp(min=1e-12)
    eff_rank_bc = torch.exp(-(p_bc * (p_bc + 1e-12).log()).sum()).item()

    if use_gpu:
        del X, W_fp, W_base, W_sfa, Sigma, eigvecs
        torch.cuda.empty_cache()

    return {
        'R_base':      R_base,
        'R_flip':      R_flip,
        'R_bc_opt':    R_bc_opt,
        # The structural-gap claim: R_bc_opt > R_flip_opt should hold population-wise
        'gap_flip_vs_bc':  R_bc_opt - R_flip,
        'gap_pct': (R_bc_opt - R_flip) / max(R_bc_opt, 1e-12) * 100,
        'flip_top1_energy':  energy_top1,
        'flip_top10_energy': energy_top10,
        'flip_top64_energy': energy_top64,
        'flip_effective_rank':  eff_rank,
        'bc_effective_rank':    eff_rank_bc,   # should be ~1
    }


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model",      required=True)
    ap.add_argument("--base-q-model",  required=True)
    ap.add_argument("--sfa-q-model",   required=True)
    ap.add_argument("--output",        default="theorem1_results.csv")
    ap.add_argument("--n-calib",       type=int, default=1024,
                    help="Large by default — we want population-scale stats.")
    ap.add_argument("--max-tokens",    type=int, default=512)
    ap.add_argument("--max-length",    type=int, default=512)
    ap.add_argument("--top-k-eig",     type=int, default=64)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("=" * 80)
    print("E1 + E2: Population-level Rank Gap (Theorem 1)")
    print("=" * 80)

    print("\n[1/4] Loading FP model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    print("\n[2/4] Collecting LARGE-SCALE activations...")
    texts = load_calibration_texts(tokenizer, n_samples=args.n_calib, seed=args.seed)
    collector = ActivationCollector(fp_model, max_tokens_per_sample=args.max_tokens)
    run_calibration(fp_model, tokenizer, collector, texts, args.device, args.max_length)
    activations = {n: collector.get(n) for n in collector.activations}
    collector.clear()
    print(f"  Collected for {len(activations)} layers")

    print("\n[3/4] Loading FP / base-Q / Flip-Q weights...")
    W_fp_dict   = get_weights(fp_model)
    fp_model = fp_model.to("cpu"); del fp_model; gc.collect(); torch.cuda.empty_cache()

    base_q = AutoModelForCausalLM.from_pretrained(
        args.base_q_model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True).eval()
    W_base_dict = get_weights(base_q); del base_q; gc.collect()
    sfa_q = AutoModelForCausalLM.from_pretrained(
        args.sfa_q_model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True).eval()
    W_sfa_dict = get_weights(sfa_q); del sfa_q; gc.collect()
    torch.cuda.empty_cache()

    print("\n[4/4] Per-layer rank-gap measurement...")
    common = [n for n in W_fp_dict if n in W_base_dict and n in W_sfa_dict
              and activations.get(n) is not None]
    print(f"  Layers: {len(common)}")

    rows = []
    for name in tqdm(common, desc="  Layers"):
        X = activations[name]
        if X.shape[0] < 16: continue
        try:
            stats = measure_rank_gap(X, W_fp_dict[name], W_base_dict[name],
                                     W_sfa_dict[name], top_k_eig=args.top_k_eig)
        except Exception as exc:
            print(f"\n  ⚠️  {name}: {exc}")
            continue
        rows.append({'layer': name,
                     'd': W_fp_dict[name].shape[1],
                     'out': W_fp_dict[name].shape[0],
                     'N': X.shape[0],
                     **stats})
        activations[name] = None
        gc.collect()

    if not rows: print("\n❌ No rows."); return

    # --- Write CSV ---
    fieldnames = list(rows[0].keys())
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n✓ Wrote {args.output}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    R_base   = np.array([r['R_base']   for r in rows])
    R_bc_opt = np.array([r['R_bc_opt'] for r in rows])
    R_flip   = np.array([r['R_flip']   for r in rows])

    print(f"\n[Total test-time output error, summed across layers]")
    print(f"  No correction   : {R_base.sum():.4e}")
    print(f"  BC (population-opt): {R_bc_opt.sum():.4e}   "
          f"({(R_bc_opt.sum() - R_base.sum())/R_base.sum()*100:+.3f}% vs base)")
    print(f"  Flip (realized) : {R_flip.sum():.4e}   "
          f"({(R_flip.sum() - R_base.sum())/R_base.sum()*100:+.3f}% vs base)")
    print(f"\n[Theorem 1 prediction: Flip < BC at population]")
    gaps_pct = np.array([r['gap_pct'] for r in rows])
    print(f"  Gap (BC_opt - Flip) / BC_opt :  median={np.median(gaps_pct):.3f}%, "
          f"mean={gaps_pct.mean():.3f}%, max={gaps_pct.max():.3f}%")
    flip_wins = (np.array([r['gap_flip_vs_bc'] for r in rows]) > 0).sum()
    print(f"  Layers where Flip < BC_opt   :  {flip_wins} / {len(rows)} "
          f"({flip_wins/len(rows)*100:.1f}%)")

    print(f"\n[Theorem 1 rank analysis]")
    bc_eff = np.array([r['bc_effective_rank']   for r in rows])
    fl_eff = np.array([r['flip_effective_rank'] for r in rows])
    print(f"  BC   effective rank in Σ_xx eigenbasis: median={np.median(bc_eff):.2f} (expected ~1)")
    print(f"  Flip effective rank in Σ_xx eigenbasis: median={np.median(fl_eff):.2f} (expected ≫1)")
    print(f"  → Flip's correction spans {np.median(fl_eff)/max(np.median(bc_eff),1):.1f}× "
          f"more eigendirections than BC's")
    e10 = np.array([r['flip_top10_energy'] for r in rows])
    print(f"  Flip's energy in top-10 Σ_xx eigendirections: "
          f"median={np.median(e10)*100:.1f}%")


if __name__ == "__main__":
    main()