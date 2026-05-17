"""
RTN + James-Stein Bias Correction (XL Version)

Extends raw RTN with a single, theoretically-motivated correction step:
after RTN quantization, the per-output bias of each Linear layer is updated by

        b_new = b + (W - W_q) · E[X]_JS

where E[X]_JS is the James-Stein shrinkage estimate of the per-input-channel
activation mean, computed from a small calibration set.

Why this works
--------------
For Y = X W^T + b, the expected output error introduced by replacing W with W_q is

        E[ΔY] = E[X] · (W - W_q)^T

This is a constant vector per output channel and can be absorbed exactly into
the bias term — making the *mean* of the quantization error zero per output
channel, without touching the weights themselves. The variance of ΔY is
unchanged; only the bias is corrected. This is "RTN, but with the first-order
output statistics fixed for free."

Using James-Stein for E[X]
--------------------------
When p (= in_features) ≥ 3, the James-Stein estimator dominates the MLE
(sample mean) in total MSE over the channel-wise mean vector:

        μ̂_JS[j] = μ̄ + (1 - c) · (X̄[j] - μ̄),   c = (p - 2) σ² / Σ(X̄[j] - μ̄)²

It shrinks noisy per-channel means toward the grand mean μ̄, which is the
right thing to do under a small-calibration regime (few hundred samples,
thousands of channels). Plain RTN ignores activation statistics entirely;
JS gives us a *low-variance* estimate of the only first-order activation
quantity that matters for bias correction.

XL handling
-----------
- Batched sequential calibration to bound peak RAM/VRAM
- lm_head split into chunks for the (W - W_q) · E[X] matmul
- --skip-lm-head defaults to True (common practice — quantizing lm_head on
  large-vocab models tends to hurt PPL more than it saves memory)

Notes
-----
- This is a *bias-only* correction. The weights are exactly the RTN weights.
  If a layer has no bias, one is created (shape [out_features], dtype = W.dtype).
- Calibration is much lighter than AWQ: no grid search, no scaling search.
  We only need E[X] per layer.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not installed. RAM monitoring disabled.")

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️  calibration_utils not found. Use --calib-dataset wikitext2-simple as fallback.")
    def get_c4_calibration_data(*a, **k): raise NotImplementedError("calibration_utils.py missing")
    def get_wikitext2_calibration_data(*a, **k): raise NotImplementedError("calibration_utils.py missing")


# ---------------------------------------------------------------------------
# James-Stein estimator for the per-channel activation mean
# ---------------------------------------------------------------------------

def compute_james_stein_mean(raw_means, variance_estimate=None):
    """
    Apply James-Stein shrinkage to a vector of per-channel sample means.

    Args:
        raw_means: [p] tensor of sample means (one per input channel)
        variance_estimate: optional scalar σ² estimate; if None, uses a
            robust (mean-absolute-deviation)² estimate.

    Returns:
        [p] tensor of shrunk means. Falls back to raw_means when p < 3
        or when the sum of squared deviations is degenerate.
    """
    p = raw_means.numel()
    if p < 3:
        return raw_means

    grand_mean = raw_means.mean()
    deviations = raw_means - grand_mean
    sum_sq_dev = (deviations ** 2).sum()

    if sum_sq_dev < 1e-10:
        return raw_means

    if variance_estimate is None:
        # Robust scalar variance proxy: (mean |X - μ̄|)²
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp(min=1e-8)

    c = ((p - 2) * variance_estimate) / sum_sq_dev
    c = c.clamp(0.0, 1.0)  # keep in the safe shrinkage regime

    return grand_mean + (1.0 - c) * deviations


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class RTN_JS_XL_Quantizer:
    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 use_james_stein=True, skip_lm_head=True,
                 max_tokens_per_sample=2048, layer_batch_size=16,
                 lmhead_chunks=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size
        self.use_james_stein = use_james_stein
        self.skip_lm_head = skip_lm_head
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.lmhead_chunks = lmhead_chunks

        self.activation_data = {}      # name -> list of CPU float32 tensors
        self.layer_stats = {}          # name -> {'error': float, 'bias_norm': float, ...}

        print(f"\n[RTN + James-Stein Bias Correction (XL) Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Skip lm_head: {skip_lm_head}")
        print(f"  Use James-Stein for E[X]: {use_james_stein}")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  lm_head chunks: {lmhead_chunks}")
        print(f"  Tokens per sample: {max_tokens_per_sample}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")

    # ----- activation hook ----------------------------------------------------

    def get_hook(self, name):
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            inp = input[0] if isinstance(input, tuple) else input

            # Token subsampling along seq dim
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                idx = torch.randperm(seq_len, device=inp.device)[:self.max_tokens_per_sample]
                idx = idx.sort()[0]
                inp = inp[:, idx, :]

            # Store on CPU in float32 for numerically clean accumulation
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_activation_mean(self, name, in_features):
        """
        Compute per-input-channel mean E[X[:, j]] (JS-shrunk if enabled).

        Returns:
            [in_features] tensor on CPU float32, or None if no calibration data.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        mean_sum = torch.zeros(in_features, dtype=torch.float32)
        total = 0
        for x in self.activation_data[name]:
            x_flat = x.reshape(-1, x.shape[-1])
            mean_sum += x_flat.sum(dim=0)
            total += x_flat.shape[0]
        if total == 0:
            return None
        raw_mean = mean_sum / total

        if self.use_james_stein:
            return compute_james_stein_mean(raw_mean)
        return raw_mean

    # ----- core RTN quantization ---------------------------------------------

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W):
        """Standard group-wise asymmetric RTN. Returns dequantized W (same dtype)."""
        out_features, in_features = W.shape
        device = W.device
        dtype = W.dtype

        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in = n_groups * self.group_size

        if padded_in > in_features:
            W_padded = torch.zeros(out_features, padded_in, device=device, dtype=dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        W_g = W_padded.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]

        max_int = 2 ** self.bits - 1
        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        W_int = torch.round(W_g / scale + zp).clamp(0, max_int)
        W_dq_g = (W_int - zp) * scale

        W_dq = W_dq_g.reshape(out_features, padded_in)
        if padded_in > in_features:
            W_dq = W_dq[:, :in_features]
        return W_dq.to(dtype)

    # ----- bias correction ----------------------------------------------------

    @torch.no_grad()
    def apply_bias_correction(self, module, W_orig, W_q, ex_mean):
        """
        b_new = b + (W_orig - W_q) · E[X]_JS

        Args:
            module: nn.Linear
            W_orig: original full-precision weight [out, in] (on device)
            W_q:    quantized (dequantized) weight  [out, in] (on device)
            ex_mean: [in] activation-mean estimate (on device, weight dtype)
        Returns:
            float: L2 norm of the correction vector, for logging.
        """
        # delta_b[i] = sum_j (W_orig[i, j] - W_q[i, j]) * E[X][j]
        delta = W_orig - W_q                          # [out, in]
        delta_b = delta.matmul(ex_mean)               # [out]

        out_features = module.weight.shape[0]
        if module.bias is None:
            module.bias = nn.Parameter(
                torch.zeros(out_features, device=module.weight.device, dtype=module.weight.dtype),
                requires_grad=False,
            )
        module.bias.data = (module.bias.data + delta_b.to(module.bias.dtype))
        return delta_b.float().norm().item()

    @torch.no_grad()
    def apply_bias_correction_chunked(self, module, W_orig, W_q, ex_mean, n_chunks):
        """
        Same as apply_bias_correction but materializes (W - W_q) · E[X] in chunks
        along the output dimension. Used for lm_head on big-vocab models.
        """
        out_features = module.weight.shape[0]
        chunk_size = (out_features + n_chunks - 1) // n_chunks

        if module.bias is None:
            module.bias = nn.Parameter(
                torch.zeros(out_features, device=module.weight.device, dtype=module.weight.dtype),
                requires_grad=False,
            )

        total_norm_sq = 0.0
        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            delta_chunk = W_orig[start:end] - W_q[start:end]
            delta_b_chunk = delta_chunk.matmul(ex_mean)
            module.bias.data[start:end] = (
                module.bias.data[start:end] + delta_b_chunk.to(module.bias.dtype)
            )
            total_norm_sq += delta_b_chunk.float().pow(2).sum().item()
            del delta_chunk, delta_b_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return float(np.sqrt(total_norm_sq))

    # ----- per-layer driver ---------------------------------------------------

    @torch.no_grad()
    def quantize_layer(self, name, module, is_lmhead=False):
        """
        Apply RTN then (if calibration available) JS bias correction.
        For lm_head, chunk the bias-correction matmul.
        """
        W = module.weight.data
        in_features = W.shape[1]
        original_dtype = W.dtype

        # Keep original weights for the correction term BEFORE we overwrite them.
        # For lm_head we keep on the same device but the per-chunk matmul keeps
        # peak memory bounded.
        W_orig = W.clone()

        # 1. RTN
        W_q = self.quantize_weight_groupwise_asymmetric(W)
        module.weight.data = W_q.to(original_dtype)

        # 2. JS bias correction (only if we have calibration data)
        ex_mean = self.get_activation_mean(name, in_features)
        if ex_mean is not None:
            ex_mean = ex_mean.to(self.device).to(original_dtype)
            if is_lmhead:
                bias_norm = self.apply_bias_correction_chunked(
                    module, W_orig, module.weight.data, ex_mean, n_chunks=self.lmhead_chunks
                )
            else:
                bias_norm = self.apply_bias_correction(
                    module, W_orig, module.weight.data, ex_mean
                )
        else:
            bias_norm = 0.0

        # 3. Cheap layer-wise diagnostic: weight quantization MSE
        with torch.no_grad():
            mse = (W_orig - module.weight.data).float().pow(2).mean().item()

        self.layer_stats[name] = {
            'weight_mse': mse,
            'bias_correction_norm': bias_norm,
            'had_calib': ex_mean is not None,
        }

        # Cleanup
        del W_orig, W_q
        if ex_mean is not None:
            del ex_mean
        if name in self.activation_data:
            del self.activation_data[name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ----- calibration --------------------------------------------------------

    def calibrate_layer_batch(self, layer_batch, calibration_texts, n_samples):
        """Register hooks for a batch of layers, run a forward pass, collect E[X]."""
        self.activation_data = {}
        handles = []
        for name, module in layer_batch:
            h = module.register_forward_hook(self.get_hook(name))
            handles.append(h)

        successful = 0
        with torch.no_grad():
            for i, text in enumerate(tqdm(calibration_texts[:n_samples],
                                          desc="  Calibration", leave=False)):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                            truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1
                    if (i + 1) % 32 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for h in handles:
            h.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if successful == 0:
            print("⚠️  No successful calibration passes in this batch.")

    # ----- top-level driver ---------------------------------------------------

    def quantize_model(self, calibration_texts, n_samples=128):
        print("\n" + "=" * 80)
        print("Batched Sequential RTN + JS Bias Correction")
        print("=" * 80)

        # Collect all Linear layers, optionally skipping lm_head
        all_layers = [(name, module) for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear)]

        layers_to_calibrate = []
        layers_to_rtn_only = []
        for name, module in all_layers:
            is_lmhead = ('lm_head' in name.lower()) or name.endswith('lm_head')
            if is_lmhead and self.skip_lm_head:
                # leave entirely in full precision
                continue
            layers_to_calibrate.append((name, module, is_lmhead))

        n_total = len(layers_to_calibrate)
        n_skipped = len(all_layers) - n_total
        print(f"  Total Linear layers: {len(all_layers)}")
        print(f"  To quantize:         {n_total}")
        print(f"  Skipped (lm_head):   {n_skipped}")

        n_batches = (n_total + self.layer_batch_size - 1) // self.layer_batch_size
        print(f"  Batches:             {n_batches} "
              f"(batch size = {self.layer_batch_size})")

        quantized = 0
        for b in range(n_batches):
            start = b * self.layer_batch_size
            end = min(start + self.layer_batch_size, n_total)
            batch_triples = layers_to_calibrate[start:end]
            batch_for_hooks = [(n, m) for (n, m, _) in batch_triples]

            print(f"\n[Batch {b+1}/{n_batches}] Layers {start}-{end-1}")

            # 1. Calibrate (collect activation means)
            self.calibrate_layer_batch(batch_for_hooks, calibration_texts, n_samples)

            # 2. Quantize each layer + apply bias correction
            for name, module, is_lmhead in tqdm(batch_triples,
                                                desc="  Quantize+Correct",
                                                leave=False):
                try:
                    self.quantize_layer(name, module, is_lmhead=is_lmhead)
                    quantized += 1
                except Exception as e:
                    print(f"\n⚠️  Error on {name}: {e}")
                    continue

            # 3. Wipe batch activations
            self.activation_data = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                print(f"  RAM after batch {b+1}: {psutil.virtual_memory().percent:.1f}%")

        # ----- summary --------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"✓ RTN + JS complete: {quantized}/{n_total} layers")
        print("=" * 80)

        if self.layer_stats:
            mses = [s['weight_mse'] for s in self.layer_stats.values()]
            biases = [s['bias_correction_norm'] for s in self.layer_stats.values()
                      if s['had_calib']]
            n_with_calib = sum(1 for s in self.layer_stats.values() if s['had_calib'])

            print(f"\nWeight quantization MSE:")
            print(f"  mean   = {np.mean(mses):.4e}")
            print(f"  median = {np.median(mses):.4e}")
            print(f"  max    = {np.max(mses):.4e}")

            print(f"\nBias-correction stats ({n_with_calib} layers got E[X]):")
            if biases:
                print(f"  ||Δb||₂  mean   = {np.mean(biases):.4e}")
                print(f"            median = {np.median(biases):.4e}")
                print(f"            max    = {np.max(biases):.4e}")
            else:
                print("  (no layers received calibration — bias correction was a no-op)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_wikitext2_simple(n_samples=128):
    print("Loading WikiText-2 (simple)...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    return texts[:n_samples]


def main():
    parser = argparse.ArgumentParser(
        description="RTN + James-Stein bias correction (XL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-calib", type=int, default=128,
                        help="Calibration samples")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                        help="Chunks for lm_head bias-correction matmul "
                             "(only used if --quantize-lm-head is set)")

    parser.add_argument("--skip-lm-head", action="store_true", default=True,
                        help="Leave lm_head in full precision (default: True)")
    parser.add_argument("--quantize-lm-head", dest="skip_lm_head",
                        action="store_false",
                        help="Quantize lm_head as well (with chunked bias correction)")

    parser.add_argument("--use-james-stein", action="store_true", default=True,
                        help="Use James-Stein shrinkage for E[X] (default: True)")
    parser.add_argument("--no-james-stein", dest="use_james_stein",
                        action="store_false",
                        help="Use the raw sample mean E[X] instead of JS")

    parser.add_argument("--model-path", type=str,
                        default="./models/Mistral-7B-v0.3")
    parser.add_argument("--output-dir", type=str,
                        default="./quantized_models/model_rtn_js_xl")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("RTN + James-Stein Bias Correction (XL)")
    print(f"Target Model: {args.model_path}")
    print("=" * 80)
    print(f"Device:               {device}")
    print(f"Bits:                 {args.bits}")
    print(f"Group size:           {args.group_size}")
    print(f"Skip lm_head:         {args.skip_lm_head}")
    print(f"Use James-Stein:      {args.use_james_stein}")
    print(f"Layer batch size:     {args.layer_batch_size}")
    print(f"lm_head chunks:       {args.lmhead_chunks}")
    print(f"Max tokens / sample:  {args.max_tokens_per_sample}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir
        )
    elif args.calib_dataset == "wikitext2-simple":
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir
        )

    quantizer = RTN_JS_XL_Quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        group_size=args.group_size,
        use_james_stein=args.use_james_stein,
        skip_lm_head=args.skip_lm_head,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
    )
    quantizer.quantize_model(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved RTN+JS-quantized model to {args.output_dir}")


if __name__ == "__main__":
    main()