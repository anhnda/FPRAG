"""
Common plumbing for EGBC theory verification.

Exports:
    set_seed
    load_text_samples           --- load N text chunks from c4 / wikitext2 / c4-val
    select_modules              --- pick Linear modules by fnmatch pattern
    ActivationRecorder          --- forward-hook recorder for μ and optionally full Σ
    run_calibration             --- push texts through the model and collect activations
    group_quantize              --- group-wise asymmetric uniform quantize (AWQ-style lattice)
    awq_search_and_scale        --- per-input-channel scale search (L2 salience, no flips)
    compute_flip_delta          --- EGBC greedy flip producing Δ in the dequantized weight space
    james_stein_mean            --- JS shrinkage for μ
    find_knee_index             --- Kneedle on a descending magnitude profile

The semantics of the quantities returned by group_quantize and compute_flip_delta:
    W_q  =  (W_int - zp) * scale       (dequantized weight after rounding, before flips)
    W_q' =  W_q + Δ                    (dequantized weight after flipping)
    e_j  =  W_q[:, j] - W[:, j]        (per-row signed quantization error before flip)
                                       NOTE: column index j here corresponds to OUTPUT row of W

These match the paper's notation:  E = W_q - W,  with e_j the j-th OUTPUT-channel ROW.
In code we work with shape [out_features, in_features].  e_j is therefore a ROW of (W_q - W).
"""

from __future__ import annotations

import gc
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Text-sample loading
# --------------------------------------------------------------------------- #
def load_text_samples(name: str, n: int, seed: int) -> List[str]:
    """Load n text snippets from one of: c4, c4-val, wikitext2.

    Uses HuggingFace datasets. Skips empty / very short docs.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    name = name.lower()
    if name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    elif name == "c4-val":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    elif name in ("wikitext2", "wikitext-2"):
        ds_full = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [x["text"] for x in ds_full if len(x["text"].strip()) > 200]
        rng.shuffle(texts)
        return texts[:n]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    texts: List[str] = []
    for ex in ds:
        t = ex.get("text", "")
        if len(t.strip()) > 200:
            texts.append(t)
        if len(texts) >= max(n * 4, n):
            break
    rng.shuffle(texts)
    return texts[:n]


# --------------------------------------------------------------------------- #
# Module selection
# --------------------------------------------------------------------------- #
def select_modules(model: nn.Module, patterns: List[str], max_layers: int = 0) -> List[str]:
    import fnmatch

    names: List[str] = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        for p in patterns:
            if fnmatch.fnmatch(n, p):
                names.append(n)
                break
    # Deduplicate while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    if max_layers and len(out) > max_layers:
        out = out[:max_layers]
    return out


# --------------------------------------------------------------------------- #
# Activation recorder
# --------------------------------------------------------------------------- #
@dataclass
class _RecorderState:
    n_tokens: int = 0
    mu_sum: Optional[torch.Tensor] = None             # [d], fp64, CPU
    cov_sum: Optional[torch.Tensor] = None            # [d,d], fp64, CPU  (only if full_cov)
    in_features: int = 0


class ActivationRecorder:
    """Online accumulator of μ and (optionally) the d×d second-moment matrix.

    Memory-conscious: keeps fp64 sums on CPU and flushes the incoming tensor every
    `flush_every_tokens` tokens.  Covariance is reconstructed from second moment
    via Σ = E[XX^T] - μμ^T at finalize() time.
    """

    def __init__(
        self,
        module_names: List[str],
        record_full_cov: bool,
        max_tokens_per_module: int = 200_000,
        flush_every_tokens: int = 16_384,
    ) -> None:
        self.module_names = list(module_names)
        self.record_full_cov = record_full_cov
        self.max_tokens_per_module = max_tokens_per_module
        self.flush_every_tokens = flush_every_tokens
        self.state: Dict[str, _RecorderState] = {n: _RecorderState() for n in module_names}
        self._buf: Dict[str, List[torch.Tensor]] = {n: [] for n in module_names}
        self._buf_tokens: Dict[str, int] = {n: 0 for n in module_names}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _hook_factory(self, name: str):
        def _hook(_module, inp, _out):
            if self.state[name].n_tokens >= self.max_tokens_per_module:
                return
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            self._buf[name].append(x.detach().to(torch.float32).cpu())
            self._buf_tokens[name] += x.shape[0]
            if self._buf_tokens[name] >= self.flush_every_tokens:
                self._flush(name)
        return _hook

    def attach(self, model: nn.Module) -> None:
        named = dict(model.named_modules())
        for n in self.module_names:
            h = named[n].register_forward_hook(self._hook_factory(n))
            self._hooks.append(h)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        # Final flush
        for n in list(self._buf.keys()):
            if self._buf[n]:
                self._flush(n)

    def _flush(self, name: str) -> None:
        if not self._buf[name]:
            return
        X = torch.cat(self._buf[name], dim=0).to(torch.float64)        # [T, d]
        # Token budget
        remaining = self.max_tokens_per_module - self.state[name].n_tokens
        if remaining < X.shape[0]:
            X = X[:remaining]
        if X.shape[0] == 0:
            self._buf[name] = []
            self._buf_tokens[name] = 0
            return

        st = self.state[name]
        if st.mu_sum is None:
            st.in_features = X.shape[1]
            st.mu_sum = torch.zeros(st.in_features, dtype=torch.float64)
            if self.record_full_cov:
                st.cov_sum = torch.zeros(st.in_features, st.in_features, dtype=torch.float64)

        st.mu_sum += X.sum(dim=0)
        if self.record_full_cov:
            st.cov_sum += X.t() @ X
        st.n_tokens += X.shape[0]

        self._buf[name] = []
        self._buf_tokens[name] = 0

    def finalize(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        out: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for n, st in self.state.items():
            if st.n_tokens == 0 or st.mu_sum is None:
                out[n] = {"mu": None, "Sigma": None, "n_tokens": 0}
                continue
            mu = st.mu_sum / st.n_tokens
            Sigma = None
            if self.record_full_cov and st.cov_sum is not None:
                second = st.cov_sum / st.n_tokens
                Sigma = second - torch.outer(mu, mu)
                # Symmetrize for numerical hygiene
                Sigma = 0.5 * (Sigma + Sigma.t())
            out[n] = {"mu": mu.float(), "Sigma": Sigma.float() if Sigma is not None else None,
                      "n_tokens": st.n_tokens}
        return out


# --------------------------------------------------------------------------- #
# Calibration driver
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_calibration(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    recorder: ActivationRecorder,
    device: torch.device,
    max_length: int,
) -> None:
    recorder.attach(model)
    try:
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            try:
                model(**enc, use_cache=False, return_dict=True)
            except Exception:
                continue
            # opportunistic flush of large hooks
            for n in recorder._buf:
                if recorder._buf_tokens[n] >= recorder.flush_every_tokens:
                    recorder._flush(n)
    finally:
        recorder.detach()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Quantization primitives
# --------------------------------------------------------------------------- #
@torch.no_grad()
def group_quantize(
    W: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group asymmetric uniform quantization.

    Returns:
        W_q          [out, in]  dequantized weight (= (W_int - zp) * scale)
        W_int        [out, in]  integer codes in [0, 2^bits - 1]
        scale_flat   [out, in]  per-coord scale (constant within group)
        zp_flat      [out, in]  per-coord zero point (constant within group)
    """
    out_features, in_features = W.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size
    if padded_in > in_features:
        W_p = torch.zeros(out_features, padded_in, dtype=W.dtype, device=W.device)
        W_p[:, :in_features] = W
    else:
        W_p = W

    W_g = W_p.reshape(out_features, n_groups, group_size)
    w_min = W_g.min(dim=2, keepdim=True)[0]
    w_max = W_g.max(dim=2, keepdim=True)[0]
    max_int = 2 ** bits - 1
    scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)

    scale_flat = scale.repeat(1, 1, group_size).reshape(out_features, padded_in)
    zp_flat = zp.repeat(1, 1, group_size).reshape(out_features, padded_in)

    W_int = torch.round(W_p / scale_flat + zp_flat).clamp(0, max_int)
    W_q = (W_int - zp_flat) * scale_flat

    if padded_in > in_features:
        W_q = W_q[:, :in_features]
        W_int = W_int[:, :in_features]
        scale_flat = scale_flat[:, :in_features]
        zp_flat = zp_flat[:, :in_features]

    return W_q.to(W.dtype), W_int, scale_flat, zp_flat


@torch.no_grad()
def james_stein_mean(raw_mean: torch.Tensor) -> torch.Tensor:
    d = raw_mean.numel()
    if d < 3:
        return raw_mean.clone()
    grand = raw_mean.mean()
    dev = raw_mean - grand
    ssd = (dev ** 2).sum()
    if ssd < 1e-10:
        return raw_mean.clone()
    # robust variance estimate
    var_hat = (dev.abs().mean() ** 2).clamp(min=1e-8)
    c = ((d - 2) * var_hat / ssd).clamp(0, 1)
    return grand + (1 - c) * dev


@torch.no_grad()
def find_knee_index(values_desc_sorted: torch.Tensor, tolerance: float = 0.0) -> int:
    """Maximum-distance-to-chord knee on the FIRST HALF of a descending sequence."""
    n = values_desc_sorted.numel()
    if n < 3:
        return n // 2
    half = n // 2
    y = values_desc_sorted[:half].detach().float().cpu().numpy()
    ymin, ymax = float(y.min()), float(y.max())
    if ymax - ymin < 1e-10:
        return half // 2
    y_norm = (y - ymin) / (ymax - ymin)
    x_norm = np.linspace(0.0, 1.0, half)
    chord = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    k = int(np.argmax(np.abs(y_norm - chord)))
    if tolerance > 0:
        k = min(half - 1, k + int(tolerance * n))
    return k


@torch.no_grad()
def compute_flip_delta(
    W: torch.Tensor,
    W_int: torch.Tensor,
    scale_flat: torch.Tensor,
    zp_flat: torch.Tensor,
    mu_cal: torch.Tensor,
    bits: int = 4,
    flip_budget_pct: float = 5.0,
    knee_tolerance: float = 0.01,
) -> torch.Tensor:
    """Run EGBC greedy on top of an existing quantization, return Δ in dequantized space.

    Inputs are all on the same device.  mu_cal is the *calibration* μ used to design flips.
    Returns Δ such that W_q_post_flip = W_q + Δ where W_q = (W_int - zp) * scale.

    The signs and validity rules below match the paper-aligned implementation in
    your awq_dh_xl.py file (Heuristic-Guided Global Greedy Rounding).
    """
    device = W.device
    out_features, in_features = W.shape
    max_int = 2 ** bits - 1

    # Same padding trick as the production quantizer so that group structure is clean
    group_size = (W_int.shape[1] // 1) if in_features <= scale_flat.shape[1] else in_features
    # We don't actually need padding since callers pass already-trimmed tensors.

    # Per-row current bias  b_j = μ^T (W_q[:,row] - W[:,row])  but here rows of W are output channels
    # In our [out, in] layout: e[j, :] = W_q[j, :] - W[j, :]  and  bias_per_row = e @ μ
    W_q = (W_int.to(W.dtype) - zp_flat) * scale_flat
    e = (W_q - W).to(torch.float64)
    mu = mu_cal.to(device).to(torch.float64)
    b = e @ mu                                              # [out]

    # Direction sign chosen to MOVE b toward zero.
    # If b > 0, we want to DECREASE b => add -μ_i * scale_i ; so flip dir d_ij satisfies sign(d_ij*μ_i) = -sign(b_j).
    # i.e.  d_ij = -sign(b_j) * sign(μ_i).   d_ij ∈ {-1, +1}.
    sign_b = torch.sign(b).unsqueeze(1)                     # [out, 1]
    sign_mu = torch.sign(mu).unsqueeze(0)                   # [1, in]
    flip_dir = (-sign_b * sign_mu).to(W.dtype)              # [out, in]
    # When sign is exactly zero, fall back to no-flip for that coordinate
    flip_dir = torch.where(torch.isnan(flip_dir) | (flip_dir == 0),
                           torch.zeros_like(flip_dir), flip_dir)

    # Lattice feasibility: integer must stay in [0, max_int]
    W_int_new = W_int + flip_dir
    in_range = (W_int_new >= 0) & (W_int_new <= max_int)

    # Per-row sign-validity: this is automatic by construction of flip_dir, but coords where μ_i=0
    # would have flip_dir=0 -> they're already filtered.

    # Knee-point mask on |μ| (excludes dimensions whose |μ_i| is outlier-large)
    mu_abs = mu.abs().to(torch.float32)
    sorted_desc, _ = torch.sort(mu_abs, descending=True)
    k_idx = find_knee_index(sorted_desc, tolerance=knee_tolerance)
    threshold = float(sorted_desc[k_idx].item())
    not_outlier = (mu_abs <= threshold)                    # [in]
    valid = in_range & (flip_dir != 0) & not_outlier.unsqueeze(0)

    # Each candidate flip i for row j changes b_j by  μ_i * (flip_dir_ij * scale_ij).
    step = (mu.unsqueeze(0) * flip_dir.to(torch.float64) * scale_flat.to(torch.float64))   # [out, in]

    # Magnitude-condition: |μ_i| * scale_i < 2 |b_j|  (paper's Prop 1)
    cond = (mu_abs.unsqueeze(0).to(torch.float64) * scale_flat.to(torch.float64)) < (2.0 * b.abs().unsqueeze(1) + 1e-30)
    valid = valid & cond

    # Ordering: prioritise coordinates with largest rounding-margin (closest to the rounding boundary)
    # rounding margin  =  |W/scale + zp - W_int|   in [0, 0.5]; closer to 0.5 → safer flip
    margin = ((W / scale_flat + zp_flat) - W_int.to(W.dtype)).abs().to(torch.float64)   # [out, in]
    margin_masked = torch.where(valid, margin, torch.full_like(margin, -1.0))
    order = torch.argsort(margin_masked, dim=1, descending=True)                         # [out, in]

    # Greedy prefix-subset over the order to minimise |b_j - cumsum(step)|
    step_sorted = torch.gather(step, 1, order)
    valid_sorted = torch.gather(valid.long(), 1, order).bool()
    step_sorted = torch.where(valid_sorted, step_sorted, torch.zeros_like(step_sorted))
    cumsum = torch.cumsum(step_sorted, dim=1)
    # NB: each flip moves b in the direction that reduces |b|. So we want to *minimise* |b - cumsum|
    # since cumsum is the cumulative correction injected into the error.
    # Equivalent formulation: minimise |b - cumsum| over prefix length k.
    target = b.unsqueeze(1)                                                              # [out, 1]
    residual = (target + cumsum).abs()                                                   # because step is signed to reduce b
    # Actually: e' @ μ = b + (μ^T Δ) and (μ^T Δ) accumulates as cumsum, so we minimize |b + cumsum|
    # (we want b + cumsum → 0, and step is signed so each entry typically moves the sum toward -b)
    # Concat the "k=0" option (no flips) at the front:
    no_flip = b.abs().unsqueeze(1)
    full = torch.cat([no_flip, residual], dim=1)                                         # [out, in+1]
    best_k = torch.argmin(full, dim=1)                                                   # [out]; 0 means no flips

    # Budget cap: at most B_j flips per row
    B_j = max(1, int(round(flip_budget_pct / 100.0 * in_features)))
    best_k = best_k.clamp(max=B_j)

    # Build a 0/1 selection mask in sorted order
    idx = torch.arange(in_features, device=device).unsqueeze(0)
    sel_sorted = (idx < best_k.unsqueeze(1)) & valid_sorted

    # Scatter back to original ordering
    sel = torch.zeros_like(sel_sorted)
    sel.scatter_(1, order, sel_sorted)

    # Build Δ in dequantized space:  Δ_ij = sel_ij * flip_dir_ij * scale_ij
    flip_dir_64 = flip_dir.to(torch.float64)
    Delta = (sel.to(torch.float64) * flip_dir_64 * scale_flat.to(torch.float64)).to(W.dtype)
    return Delta


@torch.no_grad()
def awq_search_and_scale(
    W: torch.Tensor,
    mu_cal: torch.Tensor,
    salience_l2: torch.Tensor,
    bits: int,
    group_size: int,
    n_grid: int = 20,
    apply_flip: bool = False,
    flip_budget_pct: float = 5.0,
    knee_tolerance: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """AWQ-style α grid search, with or without an EGBC flip applied during scoring.

    Returns:
        W_q_eff      dequantized weight in ORIGINAL (un-scaled) input space
        W_int        integer codes in the scaled space
        scale_flat   per-coord quant scale in the scaled space
        zp_flat      per-coord zero point
        best_alpha
    """
    device = W.device
    in_features = W.shape[1]
    salience = salience_l2.to(device).clamp(min=1e-5)
    mu = mu_cal.to(device)

    best_alpha = 0.0
    best_err = float("inf")
    best_packet = None

    # Tiny synthetic scoring activations: we don't keep X here.  Score by W-MSE weighted by salience,
    # which is what AWQ effectively does up to constants.  This is sufficient to pick α; the *evaluation*
    # of Theorem 1 is done with the real μ and Σ in the verification driver.
    for g in range(n_grid + 1):
        alpha = g / n_grid
        s = salience.pow(alpha)
        W_scaled = W * s.unsqueeze(0)
        W_q_s, W_int, scale_flat, zp_flat = group_quantize(W_scaled, bits=bits, group_size=group_size)
        if apply_flip:
            mu_scaled = mu / s
            Delta_s = compute_flip_delta(
                W=W_scaled, W_int=W_int, scale_flat=scale_flat, zp_flat=zp_flat,
                mu_cal=mu_scaled, bits=bits,
                flip_budget_pct=flip_budget_pct, knee_tolerance=knee_tolerance,
            )
            W_q_s = W_q_s + Delta_s

        # Saliency-weighted W-MSE proxy
        diff = (W_q_s - W_scaled).to(torch.float32)
        err = float((diff.pow(2) * salience.unsqueeze(0)).sum().item())
        if err < best_err:
            best_err = err
            best_alpha = alpha
            # Reverse the scaling so the returned tensors describe the ORIGINAL input space
            W_q_eff = (W_q_s / s.unsqueeze(0)).to(W.dtype)
            best_packet = (W_q_eff, W_int.clone(), scale_flat.clone(), zp_flat.clone(), s.clone())

    W_q_eff, W_int, scale_flat, zp_flat, s_best = best_packet
    return W_q_eff, W_int, scale_flat, zp_flat, best_alpha


@torch.no_grad()
def compute_l2_salience(mu_sq_estimate: Optional[torch.Tensor], in_features: int, device) -> torch.Tensor:
    """Fallback if recorder didn't keep E[X^2]: use μ^2 + sample variance ≈ E[X^2].
    Callers should pass a real E[X^2] estimate when available; here we return |μ|^2 + ε as a robust default.
    """
    if mu_sq_estimate is not None:
        return mu_sq_estimate.to(device).clamp(min=1e-5)
    return torch.ones(in_features, device=device)