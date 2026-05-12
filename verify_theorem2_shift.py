"""
E3: Verify Theorem 2 (Sign-Stability under Distribution Shift)
===============================================================

Tests the claim that Flip's decisions are sign-stable across distributions
while BC's correction magnitudes shift linearly with the distribution gap.

What this script does:
  1. Collect activations on TWO different distributions (calibration = WikiText
     vs. test = C4 by default).
  2. For each layer, compute:
       mu_cal, mu_test                : per-coord population means
       b_j_cal = mu_cal^T e_j         : BC's correction signal on calibration
       b_j_test = mu_test^T e_j       : BC's correction signal on test
       Δb_j = b_j_cal - b_j_test      : BC's shift-induced error per channel
  3. Compute Flip's decision pattern on cal vs. test:
       For each (j, i) with i in the knee region of mu_cal:
            decision_cal  = sign(mu_cal_i) * sign(b_j_cal),   fires if 2|b_j_cal| > |mu_cal_i| s_j
            decision_test = sign(mu_test_i) * sign(b_j_test), fires if 2|b_j_test| > |mu_test_i| s_j
       Count: agreements vs disagreements.
  4. Report:
       - ||Δb||  =  BC's shift-induced error (norm across all channels)
       - Sign-agreement rate on Flip decisions
       - Effective # of "robust" coords (sign agrees AND fires on both)

USAGE
-----
python verify_theorem2_shift.py \\
    --fp-model      ./models/Mistral-7B-v0.3 \\
    --sfa-q-model   ./quantized_models/awq_sfa \\
    --output        theorem2_shift.csv \\
    --cal-dataset   wikitext \\
    --test-dataset  c4 \\
    --n-calib       512 --n-test 512
"""

import os, gc, argparse, random, csv
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------- Calibration data from two different sources ----------
def load_texts(dataset_name, tokenizer, n_samples=512, seed=42):
    if dataset_name == 'wikitext':
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    elif dataset_name == 'c4':
        ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
        texts = []
        for item in ds:
            if len(texts) >= n_samples * 4: break
            if len(item['text'].strip()) > 200:
                texts.append(item['text'])
    elif dataset_name == 'ag_news':
        ds = load_dataset('ag_news', split='test')
        texts = [item['text'] for item in ds if len(item['text'].strip()) > 200]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
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

    def get_mean(self, name):
        if name not in self.activations or len(self.activations[name]) == 0:
            return None
        X = torch.cat(self.activations[name], dim=0)
        return X.mean(dim=0)

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


def get_weights_with_scale(model):
    """Return weight + estimate group-wise step size s_j per channel.
    For real EGBC verification, you'd pass in the actual scales used during
    quantization; here we approximate s_j as (max-min)/(2^bits-1) per output channel."""
    out = OrderedDict()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data.detach().cpu().float()
            # rough per-output-channel step (approximation)
            w_min = W.min(dim=1, keepdim=True)[0]
            w_max = W.max(dim=1, keepdim=True)[0]
            s_j = (w_max - w_min) / 15.0   # assume 4-bit
            out[name] = (W, s_j)
    return out


# ---------- Core measurement ----------
@torch.no_grad()
def measure_shift_stability(mu_cal, mu_test, W_fp, W_sfa, s_j_approx,
                             knee_quantile=0.5, top_p=0.05):
    """
    Args:
      mu_cal, mu_test : [d] population means
      W_fp, W_sfa     : [out, d] FP and Flip-quantized weights
      s_j_approx      : [out, 1] per-channel quantization step
      knee_quantile   : fraction of |mu| considered 'knee-eligible'
      top_p           : flipping budget fraction per output channel
    """
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        bytes_needed = (mu_cal.numel() + mu_test.numel() + 2 * W_fp.numel()) * 4
        if bytes_needed < 4 * (1024**3):
            mu_cal = mu_cal.cuda(); mu_test = mu_test.cuda()
            W_fp = W_fp.cuda(); W_sfa = W_sfa.cuda()
            s_j_approx = s_j_approx.cuda()
        else:
            use_gpu = False

    d = mu_cal.shape[0]
    out, _ = W_fp.shape

    # E = W_sfa - W_fp (the *applied* Flip plus original quantization error)
    # For decision-mimicry, we use the BASE-Q E so we replay the same flip
    # decisions Flip would make.  Here we substitute W_sfa - W_fp as a proxy
    # (this slightly overstates the gap but is structurally correct).
    E = W_sfa - W_fp

    # Per-channel bias signals on each distribution
    b_cal  = mu_cal  @ E.t()     # [out]
    b_test = mu_test @ E.t()     # [out]

    # BC's shift-induced error: |b_cal - b_test|^2 sum
    # BC applies correction c_j = -b_cal,j ; on test the residual is (b_cal - b_test).
    delta_b = b_cal - b_test
    bc_shift_error_l2 = delta_b.pow(2).sum().item()
    bc_test_residual  = delta_b.pow(2).sum().item()       # same as above
    bc_no_correction_residual = b_test.pow(2).sum().item()

    # ---- Mimic Flip's decision rule on each distribution ----
    # Knee mask: top quantile of |mu| but excluding extreme outliers
    abs_mu_cal  = mu_cal.abs()
    abs_mu_test = mu_test.abs()

    # Define knee region as |mu| in [q_low, q_high] quantile
    # (top extreme excluded, near-zero excluded)
    q_low_cal  = torch.quantile(abs_mu_cal,  1 - knee_quantile)
    q_high_cal = torch.quantile(abs_mu_cal,  0.99)
    knee_cal  = (abs_mu_cal  >= q_low_cal)  & (abs_mu_cal  <= q_high_cal)

    q_low_test  = torch.quantile(abs_mu_test,  1 - knee_quantile)
    q_high_test = torch.quantile(abs_mu_test,  0.99)
    knee_test = (abs_mu_test >= q_low_test) & (abs_mu_test <= q_high_test)

    # For each (j, i): decision is 'flip' iff
    #    2|b_j| > |mu_i| s_j   AND  knee_mask[i] is true
    # The sign of the flip is chosen to reduce |b_j|.
    # We measure agreement of the FIRE decision and SIGN decision across cal/test.

    # Fire indicator: shape [out, d]
    s = s_j_approx.squeeze(-1).unsqueeze(1)            # [out, 1]
    fire_cal  = (2 * b_cal.abs().unsqueeze(1)  > abs_mu_cal.unsqueeze(0)  * s) \
                & knee_cal.unsqueeze(0).expand(out, -1)
    fire_test = (2 * b_test.abs().unsqueeze(1) > abs_mu_test.unsqueeze(0) * s) \
                & knee_test.unsqueeze(0).expand(out, -1)

    # Sign of the proposed flip:
    # we want to reduce |b_j|; per (j,i) the sign of the weight change is
    # sign(-mu_i * b_j) — but only the SIGN matters for stability.
    sign_cal  = torch.sign(-mu_cal.unsqueeze(0)  * b_cal.unsqueeze(1))
    sign_test = torch.sign(-mu_test.unsqueeze(0) * b_test.unsqueeze(1))

    # Coordinates that would fire on calibration AND test, with same sign
    fire_both       = fire_cal & fire_test
    sign_agrees     = (sign_cal == sign_test) & (sign_cal != 0)
    robust_decisions = (fire_both & sign_agrees).sum().item()

    # Disagreements: would fire on cal but with opposite sign on test
    sign_disagree_in_fire_both = (fire_both & ~sign_agrees).sum().item()

    # Cal-only or test-only firings
    fire_cal_only  = (fire_cal  & ~fire_test).sum().item()
    fire_test_only = (~fire_cal & fire_test).sum().item()
    total_fire_cal = fire_cal.sum().item()

    # Sign-stability rate on cal firings: of cal-firing decisions,
    # what fraction would also fire on test with the same sign?
    if total_fire_cal > 0:
        stability_rate = robust_decisions / total_fire_cal
    else:
        stability_rate = float('nan')

    # ---- Distribution-shift magnitude ----
    shift_l2 = (mu_cal - mu_test).pow(2).sum().sqrt().item()
    shift_rel = shift_l2 / mu_cal.pow(2).sum().sqrt().clamp(min=1e-12).item()

    if use_gpu:
        del mu_cal, mu_test, W_fp, W_sfa, E, fire_cal, fire_test, sign_cal, sign_test
        torch.cuda.empty_cache()

    return {
        'shift_l2':    shift_l2,
        'shift_rel':   shift_rel,
        'bc_no_correction_residual':   bc_no_correction_residual,
        'bc_test_residual':           bc_test_residual,
        'bc_reduction':  bc_no_correction_residual - bc_test_residual,  # may be negative!
        # Flip-decision counts
        'fire_cal':         total_fire_cal,
        'fire_both':        fire_both.sum().item(),
        'sign_disagree':    sign_disagree_in_fire_both,
        'flip_stability':   stability_rate,
        # Magnitude check
        'bc_shift_error_l2_ratio': bc_shift_error_l2 /
                                   max(bc_no_correction_residual, 1e-12),
    }


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model",      required=True)
    ap.add_argument("--sfa-q-model",   required=True)
    ap.add_argument("--cal-dataset",   default='wikitext')
    ap.add_argument("--test-dataset",  default='c4')
    ap.add_argument("--n-calib",       type=int, default=512)
    ap.add_argument("--n-test",        type=int, default=512)
    ap.add_argument("--max-tokens",    type=int, default=512)
    ap.add_argument("--max-length",    type=int, default=512)
    ap.add_argument("--output",        default="theorem2_shift.csv")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("=" * 80)
    print("E3: Sign-Stability under Distribution Shift (Theorem 2)")
    print("=" * 80)
    print(f"  Cal dataset:  {args.cal_dataset}")
    print(f"  Test dataset: {args.test_dataset}")

    print("\n[1/4] Loading FP model...")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    # Collect means on cal distribution
    print(f"\n[2/4] Collecting μ_cal on {args.cal_dataset}...")
    cal_texts = load_texts(args.cal_dataset, tokenizer, args.n_calib, args.seed)
    collector = ActivationCollector(fp_model, max_tokens_per_sample=args.max_tokens)
    run_calibration(fp_model, tokenizer, collector, cal_texts, args.device, args.max_length)
    mu_cal = {n: collector.get_mean(n) for n in collector.activations}
    collector.clear()
    print(f"  Got μ_cal for {len(mu_cal)} layers")

    # Collect means on test distribution
    print(f"\n[3/4] Collecting μ_test on {args.test_dataset}...")
    test_texts = load_texts(args.test_dataset, tokenizer, args.n_test, args.seed + 1)
    collector = ActivationCollector(fp_model, max_tokens_per_sample=args.max_tokens)
    run_calibration(fp_model, tokenizer, collector, test_texts, args.device, args.max_length)
    mu_test = {n: collector.get_mean(n) for n in collector.activations}
    collector.clear()
    print(f"  Got μ_test for {len(mu_test)} layers")

    # Weights
    W_fp = get_weights_with_scale(fp_model)
    fp_model = fp_model.to('cpu'); del fp_model; gc.collect(); torch.cuda.empty_cache()

    sfa_q = AutoModelForCausalLM.from_pretrained(
        args.sfa_q_model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True).eval()
    W_sfa = get_weights_with_scale(sfa_q)
    del sfa_q; gc.collect()

    print("\n[4/4] Per-layer shift-stability measurement...")
    common = [n for n in W_fp if n in W_sfa and n in mu_cal and n in mu_test
              and mu_cal[n] is not None and mu_test[n] is not None]
    print(f"  Layers: {len(common)}")

    rows = []
    for name in tqdm(common, desc="  Layers"):
        W_fp_w, s_j = W_fp[name]
        W_sfa_w, _  = W_sfa[name]
        if W_fp_w.shape != W_sfa_w.shape: continue
        try:
            stats = measure_shift_stability(
                mu_cal[name], mu_test[name], W_fp_w, W_sfa_w, s_j)
        except Exception as exc:
            print(f"\n  ⚠️  {name}: {exc}"); continue
        rows.append({'layer': name, **stats})

    if not rows: print("\n❌ No rows."); return

    fieldnames = list(rows[0].keys())
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n✓ Wrote {args.output}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    shifts = np.array([r['shift_rel'] for r in rows])
    stab   = np.array([r['flip_stability'] for r in rows
                       if not np.isnan(r['flip_stability'])])
    bc_ratio = np.array([r['bc_shift_error_l2_ratio'] for r in rows])
    bc_red = np.array([r['bc_reduction'] for r in rows])

    print(f"\n[Distribution shift μ_cal vs μ_test]")
    print(f"  Relative shift ||μ_cal - μ_test||/||μ_cal||:")
    print(f"    median = {np.median(shifts)*100:.2f}%, mean = {shifts.mean()*100:.2f}%")
    print(f"    → If median > a few %, the cal and test distributions are meaningfully different.")

    print(f"\n[Theorem 2 prediction: Flip decisions sign-stable]")
    print(f"  Flip-stability rate (cal-firings that also fire on test with same sign):")
    print(f"    median = {np.median(stab)*100:.2f}%, mean = {stab.mean()*100:.2f}%")
    print(f"  → High stability means Flip's calibration decisions transfer to test data.")

    print(f"\n[BC's shift-induced cost as fraction of original bias error]")
    print(f"  ||BC_correction - optimal||^2 / ||original bias||^2:")
    print(f"    median = {np.median(bc_ratio)*100:.2f}%, mean = {bc_ratio.mean()*100:.2f}%")
    bc_pos = (bc_red > 0).sum()
    print(f"  Layers where BC reduces bias on test:    {bc_pos}/{len(rows)}")
    print(f"  Layers where BC INCREASES bias on test:  {len(rows)-bc_pos}/{len(rows)}")
    print(f"  → If many layers see BC increase test bias, this is direct evidence")
    print(f"    that BC's correction is fragile under distribution shift.")


if __name__ == "__main__":
    main()