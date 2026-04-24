"""
GPTQ Implementation for Extra Large Models (XL) - FIXED
Based on "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

Key Features:
- Layer-wise quantization with Hessian-based weight updates
- Cholesky decomposition for numerical stability
- Batched block processing (B=128)
- TRUE sequential quantization: each layer calibrated on outputs of all
  previously-quantized layers, matching the official llama.py flow exactly
- Special handling for large lm_head layers

Algorithm (from GPTQ paper):
1. Accumulate Hessian: H = 2XX^T + lambdaI (dampening)
2. Compute Cholesky decomposition: H^-1 = Cholesky(H)^T
3. For each block of B=128 columns:
   - Quantize column j: Q[:,j] = quant(W[:,j])
   - Compute error: err = (W[:,j] - Q[:,j]) / H^-1[j,j]
   - Update remaining columns in block: W[:,j+1:i+B] -= err * H^-1[j,j+1:i+B]
4. After block, update all remaining weights: W[:,(i+B):] -= E * H^-1[i:(i+B),(i+B):]

Calibration loop (matches official llama_sequential exactly):
  For each transformer layer i:
    1. Run UNQUANTIZED layer i with inps (= quantized outputs of layers 0..i-1)
       -> hooks fire, activations collected -> Hessian built
    2. Quantize layer i's sub-layers via fasterquant()
    3. Run QUANTIZED layer i with inps -> outs
    4. inps, outs = outs, inps   (propagate quantization error forward)
"""

import gc
import math
import os
import random
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory monitoring disabled.")
    print("   Install with: pip install psutil")

from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data


# ---------------------------------------------------------------------------
# Low-level quantization helpers  (unchanged, match official quant.py exactly)
# ---------------------------------------------------------------------------

def quantize(x, scale, zero, maxq):
    """
    Quantize and dequantize tensor x using scale and zero point.
    Exactly matches official GPTQ quant.py::quantize()
    """
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    """Quantizer class — exactly matches official GPTQ quant.py::Quantizer."""

    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape

        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3]).flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            tmp = shape[0] if weight else (shape[1] if len(shape) != 3 else shape[2])
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return

        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero  = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero  = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero  = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# ---------------------------------------------------------------------------
# Per-layer GPTQ  (unchanged, matches official gptq.py exactly)
# ---------------------------------------------------------------------------

class GPTQQuantizer:
    """GPTQ class — exactly matches official GPTQ gptq.py::GPTQ."""

    def __init__(self, layer, device='cuda'):
        self.layer = layer
        self.dev = device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows    = W.shape[0]
        self.columns = W.shape[1]
        self.H        = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()

    def add_batch(self, inp):
        """Exactly matches official GPTQ gptq.py::GPTQ.add_batch()."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1, bits=4, sym=True):
        """Exactly matches official GPTQ gptq.py::GPTQ.fasterquant()
        (simplified: no actorder / static_groups)."""
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        self.quantizer.configure(bits, perchannel=True, sym=sym)
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q      = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H    = torch.linalg.cholesky(H)
        H    = torch.cholesky_inverse(H)
        H    = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2    = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1     = W[:, i1:i2].clone()
            Q1     = torch.zeros_like(W1)
            Err1   = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1  = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(
                        W[:, (i1 + i):(i1 + i + groupsize)], weight=True
                    )

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i]     = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i]  = err1

            Q[:, i1:i2]  = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        error = torch.sum(Losses).item()

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        return error, (time.time() - tick)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helper: find all nn.Linear sub-layers within a module
# ---------------------------------------------------------------------------

def find_layers(module, layer_types=(nn.Linear,), prefix=''):
    """Recursively collect all layers of given types. Returns {name: module}."""
    if isinstance(module, tuple(layer_types)):
        return {prefix: module}
    result = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        result.update(find_layers(child, layer_types, full))
    return result


# ---------------------------------------------------------------------------
# FIXED: True sequential model-level quantization
# ---------------------------------------------------------------------------

class GPTQStandXLQuantizer:
    """
    GPTQ Quantizer — FIXED to match the official llama_sequential() flow exactly.

    THE BUG IN THE OLD VERSION
    --------------------------
    Old approach (wrong):
        Run the full FP16 model once with hooks on all layers simultaneously.
        Every layer's Hessian H = 2XX^T was built from FP16 activations.
        But at inference time layer N receives activations that have already
        been corrupted by quantization in layers 0..N-1.  The Hessian is
        optimised for the wrong input distribution, so the weight updates
        compensate for the wrong errors -> perplexity explodes.

    THE FIX
    -------
    For each transformer block i:
      1. Forward-pass UNQUANTIZED block i with inps
         (hooks fire -> Hessian accumulated for every sub-layer in block i)
      2. fasterquant() each sub-layer -> weights are now int4 in place
      3. Forward-pass QUANTIZED block i with the same inps -> outs
      4. inps, outs = outs, inps
         Block i+1 now sees activations already corrupted by quantization
         in blocks 0..i, exactly as at inference time.

    This matches official llama_sequential() line-for-line.
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 blocksize=128, percdamp=0.01, max_tokens_per_sample=2048,
                 skip_lmhead=True, sym=True):
        self.model                 = model
        self.tokenizer             = tokenizer
        self.device                = device
        self.bits                  = bits
        self.group_size            = group_size
        self.blocksize             = blocksize
        self.percdamp              = percdamp
        self.max_tokens_per_sample = max_tokens_per_sample
        self.skip_lmhead           = skip_lmhead
        self.sym                   = sym
        self.layer_stats           = {}

        quant_type  = "SYMMETRIC" if sym else "ASYMMETRIC"
        quant_range = (f"[-{2**(bits-1)}, {2**(bits-1)-1}]" if sym
                       else f"[0, {2**bits - 1}]")

        print(f"\n[GPTQ Quantizer Initialized - XL Version (FIXED)]")
        print(f"  Target bits       : {bits}")
        print(f"  Group size        : {group_size}")
        print(f"  Block size        : {blocksize}")
        print(f"  Dampening         : {percdamp}")
        print(f"  Token subsampling : {max_tokens_per_sample} tokens/sample")
        print(f"  Quantization      : {'Group-wise' if group_size > 0 else 'Per-channel'} "
              f"{quant_type} {quant_range}")
        print(f"  Skip lm_head      : {skip_lmhead}")
        print(f"  Calibration       : TRUE SEQUENTIAL (quantized-error propagation)")

    # ------------------------------------------------------------------
    # Step 1: capture inputs to the very first transformer layer
    # ------------------------------------------------------------------

    def _capture_layer0_inputs(self, transformer_layers, calibration_data,
                               n_samples, hidden_size, dtype):
        """
        Run the model just far enough to collect inputs to transformer_layers[0].

        Uses the same Catcher trick as official llama_sequential():
          - Replace layer 0 with a Catcher that stores inp and raises ValueError
            to abort the forward pass early (avoids running the whole model).
          - Restore layer 0 afterwards.

        Returns
        -------
        inps         : Tensor (n_actual, max_tokens, hidden_size) on CPU.
                       Shorter sequences are left-packed; padding slots are zero.
        sample_kwargs: list[dict] — per-sample attention_mask, position_ids, etc.
                       Each dict's tensors are shaped for THAT sample's actual
                       sequence length, so RoPE never sees a length mismatch.
        seq_lens     : list[int] — actual (unpadded) sequence length per sample,
                       used by _run_layer to trim inps before passing to the layer.
        """
        inps = torch.zeros(
            (n_samples, self.max_tokens_per_sample, hidden_size),
            dtype=dtype, device='cpu'
        )
        # Per-sample storage — NOT a single shared dict
        per_sample_kwargs = []
        per_sample_seqlen = []
        cache = {'i': 0}

        class Catcher(nn.Module):
            def __init__(self_, module):       # noqa: N805
                super().__init__()
                self_.module = module

            def forward(self_, inp, **kwargs): # noqa: N805
                idx       = cache['i']
                actual_len = inp.shape[1]
                store_len  = min(actual_len, inps.shape[1])
                inps[idx, :store_len] = inp[0, :store_len].detach().cpu()
                cache['i'] += 1

                # Store THIS sample's kwargs (attention_mask / position_ids
                # are shaped for its own sequence length — do NOT share across
                # samples, or RoPE will crash on length mismatch).
                per_sample_kwargs.append({
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                })
                per_sample_seqlen.append(store_len)

                raise ValueError   # abort forward pass early — inputs captured

        original_layer0       = transformer_layers[0]
        transformer_layers[0] = Catcher(original_layer0)

        successful = 0
        with torch.no_grad():
            for text in calibration_data[:n_samples]:
                try:
                    toks = self.tokenizer(
                        text, return_tensors="pt",
                        truncation=True,
                        max_length=self.max_tokens_per_sample
                    )
                    toks = {k: v.to(self.device) for k, v in toks.items()}
                    self.model(**toks, use_cache=False, return_dict=True)
                except ValueError:
                    # Expected: Catcher raised it to abort early
                    successful += 1
                except Exception as e:
                    if successful == 0:
                        print(f"  Warning: forward pass error during capture: {e}")
                finally:
                    del toks
                    if successful % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        transformer_layers[0] = original_layer0

        if successful == 0:
            raise RuntimeError("No calibration samples captured for layer 0.")

        print(f"  Captured inputs for {successful}/{n_samples} samples.")
        return inps[:successful], per_sample_kwargs, per_sample_seqlen

    # ------------------------------------------------------------------
    # Step 2: forward pass helper (used for both Hessian collection
    #         and quantized-output generation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_layer(self, layer, inps, sample_kwargs, seq_lens, n_samples):
        """
        Run `layer` on each of the n_samples inputs in `inps`.

        Parameters
        ----------
        layer        : transformer block (on GPU)
        inps         : Tensor (n_samples, max_tokens, hidden) on CPU
        sample_kwargs: list[dict] — per-sample kwargs (attention_mask,
                       position_ids, …).  Each dict's tensors are already
                       trimmed to that sample's actual sequence length, so
                       RoPE never sees a shape mismatch.
        seq_lens     : list[int] — actual sequence length for each sample
        n_samples    : int

        Returns
        -------
        outs : Tensor (n_samples, max_tokens, hidden) on CPU.
               Positions beyond seq_len are zero-padded (they are never
               used as inputs to the next layer — only [:seq_len] matters).
        """
        outs = torch.zeros_like(inps)
        for j in range(n_samples):
            slen  = seq_lens[j]
            # Trim to actual sequence length — this is what makes RoPE happy.
            inp_j = inps[j, :slen].unsqueeze(0).to(self.device)
            kw_j  = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in sample_kwargs[j].items()
            }
            out = layer(inp_j, **kw_j)
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            outs[j, :slen] = out_tensor[0].detach().cpu()
            del inp_j, kw_j, out, out_tensor
        return outs

    # ------------------------------------------------------------------
    # Step 3: quantize one transformer block
    # ------------------------------------------------------------------

    def _quantize_block(self, block_idx, layer, inps, sample_kwargs, seq_lens, n_samples):
        """
        Quantize all Linear sub-layers in one transformer block.

        Exactly matches the inner loop of official llama_sequential():

          PASS 1 — build Hessians
            Hooks attached to every sub-layer.
            Run UNQUANTIZED layer on inps -> hooks fire -> H accumulated.

          fasterquant()
            Quantize weights in-place using H.

          PASS 2 — propagate quantized outputs
            Run QUANTIZED layer on the same inps.
            Return these outputs so the caller can feed them as inputs
            to the next block.

        Parameters
        ----------
        block_idx   : int, for progress logging
        layer       : transformer block nn.Module (must already be on GPU)
        inps        : Tensor (n_samples, max_tokens, hidden) on CPU
        sample_kwargs: list[dict] — per-sample kwargs (see _run_layer)
        seq_lens    : list[int]   — actual sequence length per sample
        n_samples   : int

        Returns
        -------
        outs_quant : Tensor (n_samples, max_tokens, hidden) on CPU
                     Outputs of the NOW-QUANTIZED layer.
        stats      : dict {sublayer_name: {'error', 'time', 'nsamples'}}
        """
        sub_layers = find_layers(layer)
        if self.skip_lmhead:
            sub_layers = {n: m for n, m in sub_layers.items()
                          if 'lm_head' not in n.lower()}

        # ---- PASS 1: accumulate Hessians --------------------------------
        gptq = {name: GPTQQuantizer(module, device=self.device)
                for name, module in sub_layers.items()}

        def make_hook(name):
            def hook(_, inp, __):
                act = inp[0].data
                # Token subsampling if sequence is very long
                if act.dim() == 3 and act.shape[1] > self.max_tokens_per_sample:
                    idx = torch.randperm(act.shape[1],
                                         device=act.device)[:self.max_tokens_per_sample]
                    act = act[:, idx.sort()[0], :]
                gptq[name].add_batch(act)
            return hook

        handles = [m.register_forward_hook(make_hook(n))
                   for n, m in sub_layers.items()]

        # Run UNQUANTIZED layer — hooks collect activations
        _ = self._run_layer(layer, inps, sample_kwargs, seq_lens, n_samples)

        for h in handles:
            h.remove()

        # ---- Quantize each sub-layer in place ---------------------------
        stats = {}
        for name in tqdm(sub_layers, desc=f"  Quantizing block {block_idx}", leave=False):
            error, elapsed = gptq[name].fasterquant(
                blocksize=self.blocksize,
                percdamp=self.percdamp,
                groupsize=self.group_size,
                bits=self.bits,
                sym=self.sym
            )
            stats[name] = {
                'error'   : error,
                'time'    : elapsed,
                'nsamples': gptq[name].nsamples,
            }
            gptq[name].free()

        del gptq
        torch.cuda.empty_cache()

        # ---- PASS 2: re-run QUANTIZED layer -> outputs for next block ---
        # Weights are now int4; outputs carry the quantization error that
        # subsequent blocks must learn to handle.
        outs_quant = self._run_layer(layer, inps, sample_kwargs, seq_lens, n_samples)

        torch.cuda.empty_cache()
        return outs_quant, stats

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def quantize_model_sequential(self, calibration_data, n_samples=128):
        """
        TRUE SEQUENTIAL quantization — matches official llama_sequential().

        High-level flow
        ---------------
        1. Move embeddings + layer 0 to GPU.
        2. Run model with a Catcher on layer 0 to collect layer-0 inputs
           (inps).  Move embeddings back to CPU.
        3. For each transformer block i:
             a. Move block to GPU.
             b. _quantize_block(): Hessian pass -> quantize -> quantized pass
             c. inps <- quantized outputs  (KEY: error propagation)
             d. Move block back to CPU, free VRAM.
        4. Save stats, restore model config.
        """
        print("\n" + "=" * 80)
        print("GPTQ TRUE SEQUENTIAL QUANTIZATION (FIXED)")
        print("=" * 80)

        if HAS_PSUTIL:
            print(f"Initial System RAM: {psutil.virtual_memory().percent:.1f}%")

        # ---- Locate transformer layer list ----------------------------------
        if not (hasattr(self.model, 'model') and hasattr(self.model.model, 'layers')):
            raise NotImplementedError(
                "Cannot find model.model.layers. "
                "Subclass and override _get_transformer_layers() for your architecture."
            )

        inner = self.model.model
        transformer_layers = inner.layers

        # Collect every model-level module that runs BEFORE the layer loop.
        # These must be on GPU during the Catcher capture pass so that the
        # forward pass reaches layer 0 successfully.
        #
        # Mistral / older LLaMA:  embed_tokens, norm
        # LLaMA-3 / newer:        embed_tokens, norm, rotary_emb
        #   rotary_emb is called as:
        #     position_embeddings = self.rotary_emb(hidden_states, position_ids)
        #   BEFORE the decoder-layer loop, so it must be on GPU.
        #   Its output (cos, sin tensors) is then passed into every decoder
        #   layer as the `position_embeddings` kwarg — already stored
        #   per-sample in sample_kwargs by the Catcher, so RoPE length
        #   mismatches are handled correctly by the existing fix.
        PRE_MODULE_ATTRS = ['embed_tokens', 'rotary_emb', 'norm']
        pre_modules = [
            getattr(inner, attr)
            for attr in PRE_MODULE_ATTRS
            if hasattr(inner, attr)
        ]

        detected = [attr for attr in PRE_MODULE_ATTRS if hasattr(inner, attr)]
        print(f"  Detected pre-modules: {detected}")

        dtype       = next(iter(self.model.parameters())).dtype
        hidden_size = self.model.config.hidden_size
        use_cache   = self.model.config.use_cache
        self.model.config.use_cache = False

        # ---- Move pre-modules + layer 0 to GPU for input capture ------------
        print("\nMoving embeddings + layer 0 to GPU for input capture...")
        for m in pre_modules:
            m.to(self.device)
        transformer_layers[0].to(self.device)

        # ---- Capture inps for layer 0 ---------------------------------------
        print(f"Capturing layer-0 inputs from {n_samples} calibration samples...")
        inps, sample_kwargs, seq_lens = self._capture_layer0_inputs(
            transformer_layers, calibration_data, n_samples, hidden_size, dtype
        )
        n_actual = inps.shape[0]

        # Move pre-modules + layer 0 back to CPU; done with them
        for m in pre_modules:
            m.cpu()
        transformer_layers[0].cpu()
        torch.cuda.empty_cache()

        # ---- Sequential per-block quantization ------------------------------
        total_error    = 0.0
        total_time     = 0.0
        total_sublayers = 0

        for i, layer in enumerate(transformer_layers):
            print(f"\n{'─' * 60}")
            print(f"Block {i + 1}/{len(transformer_layers)}")

            layer.to(self.device)

            # _quantize_block handles both the Hessian pass and the
            # quantized-output pass internally.
            outs_quant, block_stats = self._quantize_block(
                i, layer, inps, sample_kwargs, seq_lens, n_actual
            )

            # Aggregate stats
            for sub_name, s in block_stats.items():
                full_name = f"model.layers.{i}.{sub_name}"
                self.layer_stats[full_name] = s
                total_error    += s['error']
                total_time     += s['time']
                total_sublayers += 1

            block_error = sum(s['error'] for s in block_stats.values())
            block_time  = sum(s['time']  for s in block_stats.values())
            print(f"  Sub-layers quantized : {len(block_stats)}")
            print(f"  Block error          : {block_error:.6f}  |  "
                  f"time: {block_time:.2f}s")

            if HAS_PSUTIL:
                print(f"  RAM: {psutil.virtual_memory().percent:.1f}%")

            # KEY LINE: propagate quantization error to the next block.
            # inps now contains outputs of the *quantized* layer i, so
            # block i+1's Hessian will be built from the same distribution
            # it will see at inference time.
            inps = outs_quant

            layer.cpu()
            torch.cuda.empty_cache()
            gc.collect()

        self.model.config.use_cache = use_cache

        print(f"\n{'=' * 80}")
        print("GPTQ True Sequential Quantization Complete!")
        print(f"   Blocks processed     : {len(transformer_layers)}")
        print(f"   Sub-layers quantized : {total_sublayers}")
        print(f"   Total error          : {total_error:.6f}")
        print(f"   Total time           : {total_time:.2f}s")
        print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Calibration data loader helper
# ---------------------------------------------------------------------------

def load_wikitext2_simple(n_samples=128):
    from datasets import load_dataset
    print("Loading WikiText-2 (simple/fast approach)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts   = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    return texts[:n_samples]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ Post-Training Quantization for XL Models (FIXED)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128,
                        help="Calibration samples")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for quantization (-1 for per-channel)")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Quantization bit width")
    parser.add_argument("--blocksize", type=int, default=128,
                        help="Block size for GPTQ algorithm")
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="Percentage dampening")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                        help="Max tokens to store per sample")
    parser.add_argument("--output-dir", type=str,
                        default="./quantized_models/model_gptq_xl",
                        help="Output directory")
    parser.add_argument("--model-path", type=str,
                        default="./models/Mistral-7B-v0.3",
                        help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"],
                        help="Calibration dataset")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                        help="Directory to cache calibration data")
    parser.add_argument("--skip-lmhead", action="store_true", default=True,
                        help="Skip lm_head quantization (default: True)")
    parser.add_argument("--quantize-lmhead", dest="skip_lmhead",
                        action="store_false",
                        help="Enable lm_head quantization")
    parser.add_argument("--sym", action="store_true", default=True,
                        help="Symmetric quantization (default: True)")
    parser.add_argument("--asym", dest="sym", action="store_false",
                        help="Asymmetric quantization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("GPTQ: Accurate Post-Training Quantization (XL Version — FIXED)")
    print(f"Target Model: {args.model_path}")
    print("=" * 80)
    print(f"Device: {device}  |  Bits: {args.bits}  |  Group size: {args.group_size}")
    print(f"Block size: {args.blocksize}  |  Dampening: {args.percdamp}")
    print("=" * 80)

    # Load model — start on CPU; we page blocks to GPU one at a time
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cpu",          # blocks moved to GPU one at a time
        trust_remote_code=True
    )
    model.eval()

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib,
            seqlen=args.max_tokens_per_sample,
            seed=args.seed, cache_dir=args.cache_dir
        )
    elif args.calib_dataset == "wikitext2-simple":
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib,
            seqlen=args.max_tokens_per_sample,
            seed=args.seed, cache_dir=args.cache_dir
        )

    # Initialise and run
    quantizer = GPTQStandXLQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        group_size=args.group_size,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        max_tokens_per_sample=args.max_tokens_per_sample,
        skip_lmhead=args.skip_lmhead,
        sym=args.sym
    )

    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    # Save
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("GPTQ QUANTIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()