"""
Verify the Flip-vs-BC theory from the EGBC paper (Section 4).

Tests three predictions:
    P1. cos(delta, mu_cal) > 0 on every layer, every eval distribution.
    P2. Per-layer gain G^layer > 0 whenever alpha > 0, and its magnitude
        correlates with alpha * ||delta|| * E_j[(mu_cal^T e_j)^2].
    P3. Under controlled cal-eval mismatch, Flip's advantage grows with ||delta||
        as long as alpha stays positive.

Also verifies the exact gap identity (Proposition 1):
        G_j = T1 + T2 + T3 + T4 + T5
with T1 the cross-term, T2..T3 non-positive, T4 small, T5 the
variance-perturbation term.

Pipeline per layer:
    1. Hook the FP16 layer to capture activations on cal set (Pile / C4-train)
       and on each eval set (WikiText2-test, C4-validation, etc.).
    2. Quantize once with the SAME base scheme (group-wise asym, AWQ-style
       optional). Record e_j = W_q[j] - W[j] per output row.
    3. Build two corrections:
         BC:    c_BC,j(x) = - mu_cal^T e_j              (scalar per row)
         Flip:  Delta_j   = greedy flip on mu_cal       (sparse weight delta)
    4. For each eval distribution:
         a. Compute mu_eval, delta = mu_eval - mu_cal, Sigma_eval.
         b. Compute alpha = cos(delta, mu_cal).
         c. Compute per-row eval risk R_eval for BC and Flip.
         d. Decompose G_j = R_eval(BC) - R_eval(Flip) into T1..T5.
         e. Aggregate G^layer = mean_j G_j.

Outputs:
    - JSON with per-layer measurements.
    - CSV summary (one row per layer x eval_distribution).
    - Console summary table with P1/P2/P3 verdicts.

Usage:
    python verify_flip_vs_bc_theory.py \
        --model-path ./models/Mistral-7B-v0.3 \
        --cal-dataset c4 \
        --eval-datasets wikitext2 c4 \
        --n-cal 128 --n-eval 64 \
        --bits 4 --group-size 128 \
        --flip-budget-pct 1.0 \
        --layers-pattern "model.layers.*.mlp.down_proj,model.layers.*.self_attn.o_proj" \
        --max-layers 8 \
        --out-dir ./flip_vs_bc_results
"""

import argparse
import fnmatch
import gc
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ----------------------------------------------------------------------------- 
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- 
# Data loading
# -----------------------------------------------------------------------------
def load_text_samples(name: str, n_samples: int, seed: int) -> List[str]:
    """
    Returns a list of raw text samples for calibration/eval activation capture.
    Uses streaming where possible to avoid downloading huge dumps.
    """
    from datasets import load_dataset

    name = name.lower()
    random.seed(seed)

    if name in ("c4", "c4-train"):
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = []
        for item in ds:
            t = item["text"]
            if len(t.strip()) > 500:
                texts.append(t)
            if len(texts) >= n_samples:
                break
        return texts

    if name in ("c4-val", "c4-validation"):
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in ds:
            t = item["text"]
            if len(t.strip()) > 500:
                texts.append(t)
            if len(texts) >= n_samples:
                break
        return texts

    if name in ("wikitext2", "wiki", "wikitext"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x["text"] for x in ds if len(x["text"].strip()) > 200]
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)
        return texts

    if name in ("pile", "the_pile"):
        # Many Pile mirrors are gated; fall back to C4 if unavailable.
        try:
            ds = load_dataset("monology/pile-uncopyrighted",
                              split="train", streaming=True)
            texts = []
            for item in ds:
                t = item["text"]
                if len(t.strip()) > 500:
                    texts.append(t)
                if len(texts) >= n_samples:
                    break
            if texts:
                return texts
        except Exception as e:
            print(f"  [pile] unavailable ({e}); falling back to c4 train.")
        return load_text_samples("c4", n_samples, seed)

    raise ValueError(f"Unknown dataset: {name}")


# ----------------------------------------------------------------------------- 
# Activation capture
# -----------------------------------------------------------------------------
class ActivationRecorder:
    """
    Captures input activations for selected Linear modules.

    Stores running first- and second-moment statistics so we never need to keep
    every token in memory.

    For each module name we keep:
        n          : total token count
        sum_x      : sum over tokens of x          (in float64)
        sum_xx_diag: sum over tokens of x*x        (in float64; only diag of XtX)
        sum_xxT    : sum over tokens of x x^T      (in float64; full d x d)
                     -- only kept if record_full_cov=True (expensive)

    The reason for sum_xxT is computing Sigma_eval for the T5 term. If d is too
    large to fit d x d in memory we set record_full_cov=False and approximate
    T5 with a diagonal Sigma instead.
    """

    def __init__(
        self,
        module_names: List[str],
        record_full_cov: bool = True,
        max_tokens_per_module: int = 100_000,
        dtype: torch.dtype = torch.float32,
        flush_every_tokens: int = 16_384,
        cov_device: Optional[torch.device] = None,
    ):
        """
        record_full_cov: if True, accumulate full d×d second moment for T5.
        flush_every_tokens: how often to flush the GPU accumulator to CPU fp64.
            Smaller values are safer (lower peak GPU memory) but slower.
        cov_device: where to keep the running second-moment matrix during a
            forward pass. Default: same GPU as the activations. The final
            fp64 accumulator always lives on CPU.
        """
        self.module_names = module_names
        self.record_full_cov = record_full_cov
        self.max_tokens_per_module = max_tokens_per_module
        self.dtype = dtype
        self.flush_every_tokens = flush_every_tokens
        self.cov_device = cov_device
        self.stats: Dict[str, Dict] = {}
        self._handles = []

    def _ensure_init(self, name: str, d: int, device: torch.device):
        if name in self.stats:
            return
        # Final accumulators on CPU in fp64 (numerically stable, no GPU pressure).
        entry = {
            "n": 0,
            "sum_x_cpu": torch.zeros(d, dtype=torch.float64, device="cpu"),
            "sum_xx_diag_cpu": torch.zeros(d, dtype=torch.float64, device="cpu"),
            "d": d,
        }
        cov_dev = self.cov_device if self.cov_device is not None else device
        # Working buffers on GPU in fp32 (fast).
        entry["buf_sum_x"] = torch.zeros(d, dtype=torch.float32, device=cov_dev)
        entry["buf_sum_xx_diag"] = torch.zeros(d, dtype=torch.float32, device=cov_dev)
        entry["buf_n"] = 0
        entry["cov_device"] = cov_dev
        if self.record_full_cov:
            entry["sum_xxT_cpu"] = torch.zeros(d, d, dtype=torch.float64, device="cpu")
            entry["buf_sum_xxT"] = torch.zeros(d, d, dtype=torch.float32, device=cov_dev)
        self.stats[name] = entry

    def _flush(self, entry: Dict):
        """Move GPU fp32 buffers into CPU fp64 totals and zero the buffers."""
        if entry["buf_n"] == 0:
            return
        entry["sum_x_cpu"] += entry["buf_sum_x"].double().cpu()
        entry["sum_xx_diag_cpu"] += entry["buf_sum_xx_diag"].double().cpu()
        entry["buf_sum_x"].zero_()
        entry["buf_sum_xx_diag"].zero_()
        if self.record_full_cov:
            entry["sum_xxT_cpu"] += entry["buf_sum_xxT"].double().cpu()
            entry["buf_sum_xxT"].zero_()
        entry["n"] += entry["buf_n"]
        entry["buf_n"] = 0

    def _hook(self, name: str):
        def hook(_module, inputs, _output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            # Keep on GPU; only convert dtype.
            x = x.detach().to(self.dtype)

            entry = self.stats.get(name)
            if entry is None:
                self._ensure_init(name, x.shape[-1], x.device)
                entry = self.stats[name]

            # Stop accumulating once we've already captured enough tokens.
            already = entry["n"] + entry["buf_n"]
            remaining = self.max_tokens_per_module - already
            if remaining <= 0:
                return
            if x.shape[0] > remaining:
                idx = torch.randperm(x.shape[0], device=x.device)[:remaining]
                x = x[idx]

            # Move x to the covariance device if needed (usually same GPU).
            if x.device != entry["cov_device"]:
                x = x.to(entry["cov_device"])

            entry["buf_n"] += x.shape[0]
            entry["buf_sum_x"] += x.sum(dim=0)
            entry["buf_sum_xx_diag"] += (x * x).sum(dim=0)
            if self.record_full_cov:
                # GPU fp32 GEMM — this is the expensive op, but it's fast on GPU.
                entry["buf_sum_xxT"].addmm_(x.t(), x)

            # Periodic flush to keep GPU buffers bounded.
            if entry["buf_n"] >= self.flush_every_tokens:
                self._flush(entry)
        return hook

    def attach(self, model: nn.Module):
        self._handles = []
        for name, mod in model.named_modules():
            if name in self.module_names and isinstance(mod, nn.Linear):
                h = mod.register_forward_hook(self._hook(name))
                self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def finalize(self) -> Dict[str, Dict]:
        """Returns per-module dict with keys: mu (d,), Exx_diag (d,), Sigma (d,d) or None, n."""
        out = {}
        for name, e in self.stats.items():
            self._flush(e)  # drain any GPU residue
            n = max(e["n"], 1)
            mu = (e["sum_x_cpu"] / n).float()
            Exx_diag = (e["sum_xx_diag_cpu"] / n).float()
            sigma = None
            if self.record_full_cov and "sum_xxT_cpu" in e:
                ExxT = (e["sum_xxT_cpu"] / n)
                sigma = (ExxT - torch.outer(mu.double(), mu.double())).float()
            out[name] = {
                "mu": mu, "Exx_diag": Exx_diag, "Sigma": sigma, "n": n, "d": e["d"]
            }
            # Free GPU working buffers — they're large.
            for k in ("buf_sum_x", "buf_sum_xx_diag", "buf_sum_xxT"):
                if k in e:
                    del e[k]
        torch.cuda.empty_cache()
        return out


# ----------------------------------------------------------------------------- 
# Module selection
# -----------------------------------------------------------------------------
# Module names that should never be analyzed by this script, regardless of
# the user-supplied pattern:
#   - lm_head (and equivalents): huge [vocab, hidden] Linear. Memory blowup
#     in analyze_layer's intermediates; needs the chunked treatment from
#     awq_*_xl.quantize_lmhead_half_by_half, which we don't reimplement here.
#   - embed_tokens: nn.Embedding, not nn.Linear in HF, but some custom models
#     wrap it as Linear. The theory's notion of "output channel j" is ill-
#     defined for a vocabulary table.
DEFAULT_EXCLUDE_PATTERNS = (
    "lm_head",
    "*.lm_head",
    "embed_tokens",
    "*.embed_tokens",
    "embed_out",
    "*.embed_out",
)


def select_modules(model: nn.Module, patterns: List[str], max_layers: int,
                    extra_exclude: List[str] = None) -> List[str]:
    """
    Pick nn.Linear modules whose names match any pattern in `patterns`, then
    drop anything matched by DEFAULT_EXCLUDE_PATTERNS or `extra_exclude`.

    The exclusion is unconditional: even if the user passes a pattern that
    matches lm_head, we refuse to analyze it (see DEFAULT_EXCLUDE_PATTERNS
    comment for why). A warning is printed in that case.
    """
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = list(DEFAULT_EXCLUDE_PATTERNS) + (extra_exclude or [])

    selected = []
    rejected_by_user_pattern = []
    for n in all_linear:
        if not any(fnmatch.fnmatch(n, p) for p in patterns):
            continue
        if any(fnmatch.fnmatch(n, p) for p in exclude):
            rejected_by_user_pattern.append(n)
            continue
        selected.append(n)

    if rejected_by_user_pattern:
        print(f"  [warn] {len(rejected_by_user_pattern)} module(s) matched the "
              f"include pattern but were force-excluded "
              f"(lm_head/embed_tokens etc.):")
        for n in rejected_by_user_pattern:
            print(f"          {n}")
        print(f"          (analyze_layer's intermediates would OOM on these. "
              f"To override, edit DEFAULT_EXCLUDE_PATTERNS in the source.)")

    if max_layers > 0 and len(selected) > max_layers:
        # Spread the picks across depth instead of taking only the first N.
        idx = np.linspace(0, len(selected) - 1, max_layers).round().astype(int)
        selected = [selected[i] for i in sorted(set(idx))]
    return selected


# ----------------------------------------------------------------------------- 
# Calibration pass
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_calibration(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    recorder: ActivationRecorder,
    device: torch.device,
    max_length: int = 512,
):
    recorder.attach(model)
    model.eval()
    try:
        for t in tqdm(texts, desc="forward"):
            try:
                enc = tokenizer(t, return_tensors="pt",
                                truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc, use_cache=False)
            except Exception as e:
                # Don't let one bad sample kill the run.
                print(f"  [warn] skipping a sample due to {e}")
                continue
    finally:
        recorder.detach()


# ----------------------------------------------------------------------------- 
# Quantization (group-wise asymmetric)
# -----------------------------------------------------------------------------
@torch.no_grad()
def group_quantize(
    W: torch.Tensor, bits: int, group_size: int
) -> torch.Tensor:
    """
    Group-wise asymmetric quantize-dequantize. Returns W_q with the same shape and dtype as W.
    """
    out_features, in_features = W.shape
    device = W.device
    dtype = W.dtype

    n_groups = (in_features + group_size - 1) // group_size
    padded = n_groups * group_size
    if padded > in_features:
        pad = torch.zeros(out_features, padded - in_features, device=device, dtype=dtype)
        Wp = torch.cat([W, pad], dim=1)
    else:
        Wp = W

    Wg = Wp.reshape(out_features, n_groups, group_size)
    w_min = Wg.min(dim=2, keepdim=True)[0]
    w_max = Wg.max(dim=2, keepdim=True)[0]
    max_int = 2 ** bits - 1

    scale = (w_max - w_min) / max_int
    scale = scale.clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)

    scale_flat = scale.repeat(1, 1, group_size).reshape(out_features, padded)
    zp_flat = zp.repeat(1, 1, group_size).reshape(out_features, padded)

    W_int = torch.round(Wp / scale_flat + zp_flat).clamp(0, max_int)
    W_dq = (W_int - zp_flat) * scale_flat
    return W_dq[:, :in_features].to(dtype), W_int[:, :in_features], scale_flat[:, :in_features], zp_flat[:, :in_features]


# ----------------------------------------------------------------------------- 
# Flip correction (paper-aligned greedy)
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_flip_delta(
    W: torch.Tensor,
    W_int: torch.Tensor,
    scale_flat: torch.Tensor,
    zp_flat: torch.Tensor,
    mu_cal: torch.Tensor,
    bits: int,
    flip_budget_pct: float,
    knee_tolerance: float = 0.0,
) -> torch.Tensor:
    """
    Implements the EGBC-style greedy flip on a single linear layer, mirroring
    awq_js_xl.quantize_weight_heuristic_groupwise but stripped to its essentials.

    Returns Delta = W_q_after_flip - W_q  (the additive correction to W_q).
    """
    out_features, in_features = W.shape
    device = W.device

    # Dequantized W_q before flipping
    W_q = (W_int - zp_flat) * scale_flat
    max_int = 2 ** bits - 1

    # Per-row bias b_j = mu^T (W_q - W)
    diff = W_q - W                            # [out, in]
    b = diff @ mu_cal                         # [out]

    # Direction in INT space that would change W_q by +scale (i.e. floor->ceil or vice versa)
    # We use sign(W/scale + zp - W_int) like awq_js_xl does.
    Wdiv = W / scale_flat
    flip_dir = torch.sign(Wdiv + zp_flat - W_int)
    flip_dir[flip_dir == 0] = 1.0             # tie-break

    # Per-element impact on b_j if we flip: delta_b = mu_i * flip_dir * scale
    flip_impacts = mu_cal.unsqueeze(0) * flip_dir * scale_flat   # [out, in]

    # We only flip if it reduces |b_j| -> sign(flip_impact) must equal sign(-b)
    # i.e. impact must be opposite to b
    target_sign = -torch.sign(b).unsqueeze(1)
    valid = (torch.sign(flip_impacts) == target_sign)

    # Cannot leave [0, max_int]
    proposed_int = W_int + flip_dir
    in_range = (proposed_int >= 0) & (proposed_int <= max_int)
    valid = valid & in_range

    # Outlier masking (Kneedle-style on |mu|): exclude top-|mu| dimensions
    abs_mu = mu_cal.abs()
    sorted_mu, _ = torch.sort(abs_mu, descending=True)
    n = sorted_mu.numel()
    first_half = sorted_mu[: max(n // 2, 3)]
    # Kneedle: max distance from chord
    if first_half.numel() >= 3:
        y = first_half.float()
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)
        x = torch.linspace(0, 1, y.numel(), device=y.device)
        line = y[0] + (y[-1] - y[0]) * x
        dist = (y - line).abs()
        knee_idx = int(dist.argmax().item())
        knee_idx = min(max(knee_idx + int(knee_tolerance * n), 0), n - 1)
        tau = sorted_mu[knee_idx].item()
    else:
        tau = sorted_mu[0].item()

    is_outlier = (abs_mu > tau)
    valid = valid & (~is_outlier).unsqueeze(0)

    # Per-row greedy: sort eligible flips by rounding cost (distance to midpoint), descending.
    rounding_cost = (Wdiv + zp_flat - W_int).abs()      # [out, in], in [0, 1]
    rc_masked = rounding_cost.clone()
    rc_masked[~valid] = -1.0

    sort_idx = torch.argsort(rc_masked, dim=1, descending=True)
    sorted_impacts = torch.gather(flip_impacts, 1, sort_idx)
    sorted_valid = torch.gather(valid.long(), 1, sort_idx)
    sorted_impacts = sorted_impacts * sorted_valid

    cumsum = torch.cumsum(sorted_impacts, dim=1)
    residual = (b.unsqueeze(1) + cumsum).abs()
    # residual[:, k-1] is |b_j + sum_{t<=k} v|, i.e. after k flips
    full_res = torch.cat([b.abs().unsqueeze(1), residual], dim=1)
    best_k = torch.argmin(full_res, dim=1)              # [out]

    # Build flip mask in sorted order
    idx_range = torch.arange(in_features, device=device).unsqueeze(0)
    flip_mask_sorted = idx_range < best_k.unsqueeze(1)
    flip_mask_sorted = flip_mask_sorted & sorted_valid.bool()

    # Cap by budget %
    max_flips = max(int(flip_budget_pct / 100.0 * in_features), 1)
    cumcount = flip_mask_sorted.long().cumsum(dim=1)
    flip_mask_sorted = flip_mask_sorted & (cumcount <= max_flips)

    # Project back to original index order
    sorted_flip_dir = torch.gather(flip_dir, 1, sort_idx)
    sorted_flip_dir = sorted_flip_dir * flip_mask_sorted.float()

    # Scatter back: int delta per element
    int_delta = torch.zeros_like(W_int)
    int_delta.scatter_add_(1, sort_idx, sorted_flip_dir)

    # Convert int delta to weight delta
    Delta = int_delta * scale_flat

    return Delta  # [out, in]


# ----------------------------------------------------------------------------- 
# Gap decomposition
# -----------------------------------------------------------------------------
@dataclass
class LayerResult:
    layer_name: str
    eval_name: str
    d: int
    out_features: int
    n_cal_tokens: int
    n_eval_tokens: int

    alpha: float                  # cos(delta, mu_cal)
    norm_delta: float
    norm_mu_cal: float
    norm_mu_eval: float

    eps_rms: float                # sqrt(mean_j eps_j^2)  where eps_j = mu_cal^T (e_j + Delta_j)
    mean_mu_cal_dot_e_sq: float   # E_j[(mu_cal^T e_j)^2]
    flip_budget_used_mean: float  # mean nnz(Delta_j) / d
    flip_budget_used_max: float

    R_eval_BC: float
    R_eval_Flip: float
    G_layer: float                # R_BC - R_Flip
    T1: float
    T2: float
    T3: float
    T4: float
    T5: float
    G_layer_check: float          # T1+T2+T3+T4+T5 (should match G_layer)

    # Convenience flags for the predictions
    P1_alpha_positive: bool
    P2_flip_wins: bool


@torch.no_grad()
def decompose_gap(
    W: torch.Tensor,            # [out, in], fp32
    e: torch.Tensor,             # [out, in], W_q - W
    Delta: torch.Tensor,         # [out, in], flip correction
    mu_cal: torch.Tensor,        # [in]
    mu_eval: torch.Tensor,       # [in]
    Sigma_eval: Optional[torch.Tensor],  # [in,in] or None
    Exx_diag_eval: torch.Tensor, # [in], used if Sigma_eval is None
) -> Dict[str, torch.Tensor]:
    """
    Computes per-row gap quantities R_BC, R_Flip and decomposition T1..T5.

    For each row j of an output, define
        e_j   = W_q[j] - W[j]                 (rounding error)
        Δ_j   = flip correction at row j      (sparse)
        ẽ_j   = e_j + Δ_j                     (after flip)

    BC adds a per-row scalar correction so that x^T (W_q[j]+e_j) + c_BC,j cancels
    the calibration mean. Equivalently the post-correction "weight delta" for BC is
        Δ_j^BC = - (μ_cal^T e_j / ||μ_cal||^2) · μ_cal  ?  No — BC is a *bias*, not weight.
    Actually, the BC bias is b_j = -μ_cal^T e_j, and the BC effective output error is
        x^T e_j - μ_cal^T e_j
    i.e. the constant -μ_cal^T e_j is added to the output regardless of x.

    Flip's correction is a weight delta, so the Flip effective output error is
        x^T (e_j + Δ_j) = x^T ẽ_j.

    Per-row eval risk on an eval batch with mean μ_eval and centered second-moment Σ_eval:
        R_BC,j  = E_eval[ (x^T e_j - μ_cal^T e_j)^2 ]
        R_Flip,j = E_eval[ (x^T ẽ_j)^2 ]

    Using E[x x^T] = Σ_eval + μ_eval μ_eval^T:
        E[(x^T v)^2] = v^T Σ_eval v + (μ_eval^T v)^2
    """
    device = W.device
    out_features, in_features = W.shape

    eps = (mu_cal.unsqueeze(0) * (e + Delta)).sum(dim=1)   # [out] : μ_cal^T ẽ_j
    mu_cal_dot_e = (mu_cal.unsqueeze(0) * e).sum(dim=1)    # [out]
    mu_eval_dot_e = (mu_eval.unsqueeze(0) * e).sum(dim=1)
    mu_eval_dot_etilde = (mu_eval.unsqueeze(0) * (e + Delta)).sum(dim=1)
    delta_vec = mu_eval - mu_cal
    delta_dot_e = (delta_vec.unsqueeze(0) * e).sum(dim=1)
    delta_dot_Delta = (delta_vec.unsqueeze(0) * Delta).sum(dim=1)
    delta_dot_etilde = delta_dot_e + delta_dot_Delta

    # --- Quadratic forms with Sigma_eval ---
    # We need:
    #   q_ee = e_j^T Σ_eval e_j
    #   q_ee_tilde = ẽ_j^T Σ_eval ẽ_j  (NOTE: paper's V_j uses these)
    #   q_eD = e_j^T Σ_eval Δ_j
    #   q_DD = Δ_j^T Σ_eval Δ_j
    if Sigma_eval is not None:
        # Compute row-wise quadratic forms efficiently
        # eSe = sum_j e_j @ Sigma @ e_j
        eS = e @ Sigma_eval                            # [out, in]
        q_ee = (eS * e).sum(dim=1)                     # [out]
        q_ee_tilde = ((e + Delta) @ Sigma_eval * (e + Delta)).sum(dim=1)
        q_eD = (eS * Delta).sum(dim=1)
        q_DD = (Delta @ Sigma_eval * Delta).sum(dim=1)
    else:
        # Diagonal approximation: Sigma_eval ≈ diag(Exx_diag_eval - mu_eval^2)
        var_diag = (Exx_diag_eval - mu_eval ** 2).clamp(min=0.0)
        q_ee = ((e ** 2) * var_diag.unsqueeze(0)).sum(dim=1)
        et = e + Delta
        q_ee_tilde = ((et ** 2) * var_diag.unsqueeze(0)).sum(dim=1)
        q_eD = (e * Delta * var_diag.unsqueeze(0)).sum(dim=1)
        q_DD = ((Delta ** 2) * var_diag.unsqueeze(0)).sum(dim=1)

    # --- Eval risks (per row) ---
    # R_BC,j = E[(x^T e_j - μ_cal^T e_j)^2]
    #        = (μ_eval^T e_j - μ_cal^T e_j)^2 + Σ_eval-quadratic
    #        = (δ^T e_j)^2 + e_j^T Σ_eval e_j
    R_BC = delta_dot_e ** 2 + q_ee

    # R_Flip,j = E[(x^T ẽ_j)^2] = (μ_eval^T ẽ_j)^2 + ẽ_j^T Σ_eval ẽ_j
    R_Flip = mu_eval_dot_etilde ** 2 + q_ee_tilde

    G = R_BC - R_Flip                                 # [out]

    # --- Decomposition T1..T5 (from Proposition 1) ---
    # G_j = -2 (δ^T e_j)(δ^T Δ_j)    -- T1 cross-term
    #       - (δ^T Δ_j)^2            -- T2 ≤ 0
    #       - eps_j^2                -- T3 ≤ 0
    #       - 2 eps_j (δ^T ẽ_j)      -- T4 small
    #       - V_j                    -- T5
    # where V_j = 2 e_j^T Σ_eval Δ_j + Δ_j^T Σ_eval Δ_j
    T1 = -2 * delta_dot_e * delta_dot_Delta
    T2 = -(delta_dot_Delta ** 2)
    T3 = -(eps ** 2)
    T4 = -2 * eps * delta_dot_etilde
    V  = 2 * q_eD + q_DD
    T5 = -V

    return {
        "R_BC": R_BC, "R_Flip": R_Flip, "G": G,
        "T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5,
        "eps": eps,
        "mu_cal_dot_e": mu_cal_dot_e,
    }


# ----------------------------------------------------------------------------- 
# Per-layer driver
# -----------------------------------------------------------------------------
@torch.no_grad()
def analyze_layer(
    name: str,
    W_fp: torch.Tensor,
    cal_stats: Dict,
    eval_stats_by_name: Dict[str, Dict],
    bits: int,
    group_size: int,
    flip_budget_pct: float,
    knee_tolerance: float,
    device: torch.device,
    fp64_for_decomp: bool = True,
) -> List[LayerResult]:
    """
    Returns one LayerResult per eval distribution.
    """
    W = W_fp.to(device).float()
    out_features, in_features = W.shape
    mu_cal = cal_stats["mu"].to(device).float()

    # Quantize once on this device
    W_q, W_int, scale_flat, zp_flat = group_quantize(W, bits=bits, group_size=group_size)
    e = (W_q - W)

    # Flip delta (computed using mu_cal only — exactly what the paper does)
    Delta = compute_flip_delta(
        W=W, W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
        mu_cal=mu_cal, bits=bits,
        flip_budget_pct=flip_budget_pct,
        knee_tolerance=knee_tolerance,
    )

    # Use float64 for the actual gap arithmetic to keep T1..T5 free of fp drift
    if fp64_for_decomp:
        W64 = W.double()
        e64 = e.double()
        Delta64 = Delta.double()
        mu_cal64 = mu_cal.double()
    else:
        W64, e64, Delta64, mu_cal64 = W, e, Delta, mu_cal

    # Flip-budget usage stats
    nnz_per_row = (Delta != 0).sum(dim=1).float()
    flip_used_mean = (nnz_per_row.mean() / in_features).item()
    flip_used_max = (nnz_per_row.max() / in_features).item()

    results: List[LayerResult] = []
    for eval_name, est in eval_stats_by_name.items():
        mu_eval = est["mu"].to(device).float()
        Exx_diag = est["Exx_diag"].to(device).float()
        Sigma = est["Sigma"]
        if Sigma is not None:
            Sigma = Sigma.to(device).float()

        if fp64_for_decomp:
            mu_eval64 = mu_eval.double()
            Exx_diag64 = Exx_diag.double()
            Sigma64 = Sigma.double() if Sigma is not None else None
        else:
            mu_eval64, Exx_diag64, Sigma64 = mu_eval, Exx_diag, Sigma

        dec = decompose_gap(
            W=W64, e=e64, Delta=Delta64,
            mu_cal=mu_cal64, mu_eval=mu_eval64,
            Sigma_eval=Sigma64, Exx_diag_eval=Exx_diag64,
        )

        delta_vec = (mu_eval - mu_cal).float()
        norm_delta = delta_vec.norm().item()
        norm_mu_cal = mu_cal.norm().item()
        norm_mu_eval = mu_eval.norm().item()
        cos_dm = (delta_vec @ mu_cal).item() / (norm_delta * norm_mu_cal + 1e-12)

        R_BC = dec["R_BC"].mean().item()
        R_Flip = dec["R_Flip"].mean().item()
        G = dec["G"].mean().item()
        T1 = dec["T1"].mean().item()
        T2 = dec["T2"].mean().item()
        T3 = dec["T3"].mean().item()
        T4 = dec["T4"].mean().item()
        T5 = dec["T5"].mean().item()
        eps_rms = dec["eps"].pow(2).mean().sqrt().item()
        mu_cal_dot_e_sq = dec["mu_cal_dot_e"].pow(2).mean().item()

        results.append(LayerResult(
            layer_name=name,
            eval_name=eval_name,
            d=in_features,
            out_features=out_features,
            n_cal_tokens=cal_stats["n"],
            n_eval_tokens=est["n"],
            alpha=cos_dm,
            norm_delta=norm_delta,
            norm_mu_cal=norm_mu_cal,
            norm_mu_eval=norm_mu_eval,
            eps_rms=eps_rms,
            mean_mu_cal_dot_e_sq=mu_cal_dot_e_sq,
            flip_budget_used_mean=flip_used_mean,
            flip_budget_used_max=flip_used_max,
            R_eval_BC=R_BC, R_eval_Flip=R_Flip,
            G_layer=G,
            T1=T1, T2=T2, T3=T3, T4=T4, T5=T5,
            G_layer_check=T1 + T2 + T3 + T4 + T5,
            P1_alpha_positive=(cos_dm > 0.0),
            P2_flip_wins=(G > 0.0),
        ))

    # Cleanup
    del W, W_q, W_int, scale_flat, zp_flat, e, Delta
    if fp64_for_decomp:
        del W64, e64, Delta64
    torch.cuda.empty_cache()
    return results


# ----------------------------------------------------------------------------- 
# Reporting
# -----------------------------------------------------------------------------
def predictions_summary(results: List[LayerResult]) -> Dict[str, float]:
    """Aggregate P1/P2/P3 verdicts and the gap-identity sanity check."""
    by_eval: Dict[str, List[LayerResult]] = {}
    for r in results:
        by_eval.setdefault(r.eval_name, []).append(r)

    summary = {}
    for eval_name, rs in by_eval.items():
        n = len(rs)
        p1 = sum(r.P1_alpha_positive for r in rs) / n
        p2 = sum(r.P2_flip_wins for r in rs) / n

        # P2 stronger form: layers with alpha>0 should also have G>0
        both = sum(1 for r in rs if r.P1_alpha_positive and r.P2_flip_wins)
        cond = sum(1 for r in rs if r.P1_alpha_positive)
        p2_cond = both / cond if cond > 0 else float("nan")

        # Correlation: |G| vs alpha * ||delta|| * E[(mu_cal^T e)^2]
        x = np.array([r.alpha * r.norm_delta * r.mean_mu_cal_dot_e_sq for r in rs])
        y = np.array([r.G_layer for r in rs])
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
        else:
            corr = float("nan")

        # Gap-identity sanity: max relative residual
        rel_errs = []
        for r in rs:
            denom = abs(r.G_layer) + 1e-12
            rel_errs.append(abs(r.G_layer - r.G_layer_check) / denom)
        max_rel_err = float(max(rel_errs)) if rel_errs else float("nan")

        summary[eval_name] = {
            "n_layers": n,
            "P1_frac_alpha_positive": p1,
            "P2_frac_flip_wins": p2,
            "P2_cond_on_P1": p2_cond,
            "corr_G_with_alpha_delta_e2": corr,
            "max_gap_identity_rel_err": max_rel_err,
            "mean_alpha": float(np.mean([r.alpha for r in rs])),
            "mean_G_layer": float(np.mean([r.G_layer for r in rs])),
        }
    return summary


def write_outputs(results: List[LayerResult], out_dir: Path, args_dict: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "per_layer_results.json", "w") as f:
        json.dump({
            "config": args_dict,
            "results": [asdict(r) for r in results],
        }, f, indent=2)

    # CSV
    import csv
    with open(out_dir / "per_layer_results.csv", "w", newline="") as f:
        if not results:
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Summary
    summ = predictions_summary(results)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)

    # Console
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    for eval_name, s in summ.items():
        print(f"\n[eval = {eval_name}]")
        print(f"  layers analyzed             : {s['n_layers']}")
        print(f"  P1 (alpha > 0)              : {s['P1_frac_alpha_positive']*100:6.2f}% of layers"
              f"   (mean alpha = {s['mean_alpha']:.4f})")
        print(f"  P2 (G_layer > 0)            : {s['P2_frac_flip_wins']*100:6.2f}% of layers"
              f"   (mean G = {s['mean_G_layer']:.4e})")
        print(f"  P2 | P1 (Flip wins when α>0): {s['P2_cond_on_P1']*100:6.2f}%")
        print(f"  corr(G, α·||δ||·E[(μ^T e)²]): {s['corr_G_with_alpha_delta_e2']: .4f}")
        print(f"  gap-identity max rel error  : {s['max_gap_identity_rel_err']:.2e}")


# ----------------------------------------------------------------------------- 
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cal-dataset", default="c4")
    parser.add_argument("--eval-datasets", nargs="+", default=["wikitext2", "c4-val"])
    parser.add_argument("--n-cal", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--flip-budget-pct", type=float, default=1.0,
                        help="Per-row flip budget as percent of in_features "
                             "(paper notation: B_j = floor(p/100 * |I|)).")
    parser.add_argument("--knee-tolerance", type=float, default=0.0,
                        help="Kneedle offset on |mu_cal| outlier mask. "
                             "Positive = mask MORE dimensions as outliers (more "
                             "conservative); negative = mask fewer. Same semantics "
                             "as --knee-tolerance in awq_js_xl.py.")
    parser.add_argument("--model-dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Dtype for loading the model weights.")
    parser.add_argument("--max-cal-tokens-per-layer", type=int, default=200_000,
                        help="Cap on tokens accumulated per layer for calibration stats.")
    parser.add_argument("--max-eval-tokens-per-layer", type=int, default=100_000,
                        help="Cap on tokens accumulated per layer for eval stats.")
    parser.add_argument("--layers-pattern", type=str,
                        default="model.layers.*.mlp.down_proj,model.layers.*.self_attn.o_proj",
                        help="Comma-separated fnmatch patterns for module names.")
    parser.add_argument("--max-layers", type=int, default=8,
                        help="Cap the number of layers we analyze (spread across depth). 0 = no cap.")
    parser.add_argument("--no-full-cov", action="store_true",
                        help="Skip d×d covariance storage; use diagonal approximation for T5. "
                             "Recommended on Llama-3-8B / d>=4096 if GPU is <= 40GB.")
    parser.add_argument("--flush-every-tokens", type=int, default=16_384,
                        help="How often to flush GPU fp32 covariance buffers to CPU fp64. "
                             "Smaller = less GPU peak, more CPU<->GPU copies.")
    parser.add_argument("--out-dir", type=str, default="./flip_vs_bc_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("FLIP vs BC THEORY VERIFICATION — CONFIG")
    print("=" * 80)
    print(f"  model              : {args.model_path}  (dtype={args.model_dtype})")
    print(f"  cal dataset        : {args.cal_dataset}  (n_samples={args.n_cal})")
    print(f"  eval datasets      : {args.eval_datasets}  (n_samples={args.n_eval})")
    print(f"  max seq length     : {args.max_length}")
    print(f"  max tokens/layer   : cal={args.max_cal_tokens_per_layer:,}  "
          f"eval={args.max_eval_tokens_per_layer:,}")
    print(f"  quantization       : {args.bits}-bit, group_size={args.group_size}")
    print(f"  flip budget        : {args.flip_budget_pct}% of in_features per row")
    print(f"  knee tolerance     : {args.knee_tolerance}  "
          f"(>0 = mask MORE outliers, <0 = mask fewer)")
    print(f"  full covariance    : {'OFF (diag fallback)' if args.no_full_cov else 'AUTO'}")
    print(f"  layers pattern     : {args.layers_pattern}")
    print(f"  max layers         : {args.max_layers if args.max_layers > 0 else 'no cap'}")
    print(f"  seed               : {args.seed}")
    print(f"  device             : {device}")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/5] Loading model: {args.model_path}")
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

    print(f"[2/5] Selecting modules with patterns: {args.layers_pattern}")
    patterns = [p.strip() for p in args.layers_pattern.split(",") if p.strip()]
    module_names = select_modules(model, patterns, args.max_layers)
    if not module_names:
        print("ERROR: No modules matched the given pattern.")
        sys.exit(1)
    print(f"  Selected {len(module_names)} modules:")
    for n in module_names:
        print(f"    {n}")

    record_full_cov = not args.no_full_cov
    # Auto-disable full cov for large d if user did not opt out
    d_max = max(model.get_submodule(n).in_features for n in module_names)
    cov_bytes = d_max * d_max * 8  # float64
    if cov_bytes > 6 * 1024**3 and record_full_cov:
        print(f"  d_max={d_max} -> full covariance would need ~{cov_bytes/1e9:.1f} GB. Disabling.")
        record_full_cov = False

    # ---- Capture cal stats ----
    print(f"\n[3/5] Capturing CALIBRATION activations from: {args.cal_dataset}")
    cal_texts = load_text_samples(args.cal_dataset, args.n_cal, args.seed)
    cal_recorder = ActivationRecorder(module_names, record_full_cov=False,
                                      max_tokens_per_module=args.max_cal_tokens_per_layer,
                                      flush_every_tokens=args.flush_every_tokens)
    run_calibration(model, tokenizer, cal_texts, cal_recorder, device, args.max_length)
    cal_stats = cal_recorder.finalize()

    # ---- Capture eval stats per dataset ----
    print(f"\n[4/5] Capturing EVAL activations from: {args.eval_datasets}")
    eval_stats_per_module: Dict[str, Dict[str, Dict]] = {n: {} for n in module_names}
    for eval_name in args.eval_datasets:
        print(f"  -- {eval_name} --")
        eval_texts = load_text_samples(eval_name, args.n_eval, args.seed + 1)
        rec = ActivationRecorder(module_names, record_full_cov=record_full_cov,
                                 max_tokens_per_module=args.max_eval_tokens_per_layer,
                                 flush_every_tokens=args.flush_every_tokens)
        run_calibration(model, tokenizer, eval_texts, rec, device, args.max_length)
        finalized = rec.finalize()
        for n in module_names:
            eval_stats_per_module[n][eval_name] = finalized[n]

    # ---- Per-layer analysis ----
    print(f"\n[5/5] Quantizing and decomposing the gap per layer")
    all_results: List[LayerResult] = []
    for name in tqdm(module_names, desc="layers"):
        mod = model.get_submodule(name)
        W = mod.weight.detach()
        per_layer = analyze_layer(
            name=name,
            W_fp=W,
            cal_stats=cal_stats[name],
            eval_stats_by_name=eval_stats_per_module[name],
            bits=args.bits,
            group_size=args.group_size,
            flip_budget_pct=args.flip_budget_pct,
            knee_tolerance=args.knee_tolerance,
            device=device,
        )
        all_results.extend(per_layer)
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    write_outputs(all_results, out_dir, vars(args))
    print(f"\n✅ Results written to {out_dir}")


if __name__ == "__main__":
    main()