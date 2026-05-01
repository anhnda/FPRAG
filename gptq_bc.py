"""
GPTQ + Bias Correction — XL version

Faithful port of the official GPTQ llama.py extended with naive bias correction.

Bias Correction Algorithm:
1. During the calibration hook phase, accumulate the running mean of input
   activations \\bar{X} per sublayer alongside the Hessian.
2. After GPTQ quantization writes W_q into module.weight.data, compute:
       bias_correction = (W_orig - W_q) @ x_mean         [out_features]
   This equals E[X (W_orig - W_q)^T] in expectation, i.e. the mean output
   error introduced by quantization.
3. Add the correction to the bias term (creating one if needed):
       bias_new = bias_old + bias_correction
   so that the quantized layer reproduces the original layer's mean output.

Sign convention:
    Y_orig  - Y_quant  = X (W_orig - W_q)^T
    mean over samples  = (W_orig - W_q) @ x_mean
    bias <- bias + correction   (NOT minus — that inverts the fix)

Key implementation notes:
  * x_mean is captured by the SAME hook used by GPTQ.add_batch, so it
    automatically reflects the inputs actually seen by this sublayer
    (including any upstream true-sequential quantization drift).
  * actorder permutes columns inside fasterquant but unpermutes Q before
    writing back, so module.weight.data is always in the original column
    order — x_mean (also in original order) lines up directly.
  * Computed in float32 for numerical stability, cast back to layer dtype.

Everything else mirrors the faithful gptq_xl.py port line-for-line.
"""

import argparse
import copy
import gc
import math
import os
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

# ---------------------------------------------------------------------------
# quant.py — EXACT copy of official
# ---------------------------------------------------------------------------

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False,
                  norm=2.4, grid=100, maxshrink=.8, trits=False):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

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

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# ---------------------------------------------------------------------------
# gptq.py — extended with mean-activation tracking for bias correction.
# Everything else is the faithful port.
# ---------------------------------------------------------------------------

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        # --- Bias correction additions -------------------------------------
        # Running mean of input activations per input channel, in fp32.
        # Shape: [columns]. Updated in add_batch with the same running-mean
        # convention as the Hessian so the two stay in lockstep.
        self.x_mean = torch.zeros(self.columns, dtype=torch.float32, device=self.dev)
        # Snapshot of the original (pre-quantization) weights, kept on the
        # layer's device, in fp32. Used after fasterquant to compute the
        # correction (W_orig - W_q) @ x_mean. We keep our own copy because
        # fasterquant overwrites layer.weight.data.
        self.W_orig = layer.weight.data.detach().to(torch.float32).clone()
        # -------------------------------------------------------------------

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        # --- Bias correction: running mean of activations ------------------
        # `inp` here is shape [columns, n_tokens] (after the .t() above for
        # Linear). Each column of inp is one token's activation vector.
        # We want the mean across all tokens ever seen, per input channel.
        # Use the same running-update convention as the Hessian so weighting
        # is identical:
        #     new_mean = old_mean * n/(n+t) + sum(new) / (n+t)
        n_tokens = inp.shape[1]
        inp_f32  = inp.float()
        new_n    = self.nsamples + tmp
        # nsamples increments by `tmp` (number of sequences/batches), but the
        # Hessian's normalization treats each call as `tmp` units regardless
        # of token count. We mirror that exactly: weight by tmp / new_n on
        # the per-token mean of this batch.
        batch_mean = inp_f32.mean(dim=1)  # [columns]
        self.x_mean.mul_(self.nsamples / new_n)
        self.x_mean.add_(batch_mean * (tmp / new_n))
        # -------------------------------------------------------------------

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1,
                    actorder=False, static_groups=False):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            dead = dead[perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q      = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1      = W[:, i1:i2].clone()
            Q1      = torch.zeros_like(W1)
            Err1    = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1   = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i):(i1 + i + groupsize)], weight=True
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()

                Q1[:, i]      = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2]      = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        error   = torch.sum(Losses).item()
        elapsed = time.time() - tick

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        return error, elapsed

    @torch.no_grad()
    def apply_bias_correction(self):
        """
        Compute and apply naive bias correction.

        Must be called AFTER fasterquant(). Uses:
          * self.W_orig  — original weights snapshot (fp32)
          * self.layer.weight.data — quantized weights (in original column order)
          * self.x_mean  — mean input activation per input channel (fp32)

        Correction per output channel:
            bc[o] = sum_c (W_orig[o,c] - W_q[o,c]) * x_mean[c]
        Equivalently:
            bc = (W_orig - W_q) @ x_mean

        bias_new = bias_old + bc

        Returns the L2 norm of the correction (for logging).
        """
        layer = self.layer

        # Bring the quantized weights into a [out, in] view in fp32.
        W_q = layer.weight.data
        if isinstance(layer, nn.Conv2d):
            W_q_view = W_q.flatten(1).float()
        elif isinstance(layer, transformers.Conv1D):
            W_q_view = W_q.t().float()
        else:
            W_q_view = W_q.float()

        # W_orig was captured in the same view convention (we copied
        # layer.weight.data before fasterquant ran). For Conv2d/Conv1D the
        # snapshot is in the *layer's native* shape; reshape consistently.
        if isinstance(layer, nn.Conv2d):
            W_orig_view = self.W_orig.flatten(1)
        elif isinstance(layer, transformers.Conv1D):
            W_orig_view = self.W_orig.t()
        else:
            W_orig_view = self.W_orig

        # Both [out, columns]. x_mean is [columns]. Correction is [out].
        delta_W = W_orig_view - W_q_view
        bc = delta_W.matmul(self.x_mean)        # [out]

        # Cast to layer dtype on the layer's device.
        target_dtype = layer.weight.data.dtype
        bc = bc.to(dtype=target_dtype, device=layer.weight.data.device)

        # Apply: bias_new = bias_old + bc. Create bias if absent.
        if layer.bias is None:
            out_features = layer.weight.shape[0]
            new_bias = nn.Parameter(
                torch.zeros(out_features, dtype=target_dtype,
                            device=layer.weight.data.device)
            )
            layer.bias = new_bias
            layer.bias.data.copy_(bc)
        else:
            layer.bias.data.add_(bc.to(layer.bias.data.dtype))

        norm = bc.float().norm().item()
        return norm

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        # Bias-correction buffers
        self.x_mean = None
        self.W_orig = None
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# modelutils.py — find_layers, EXACT copy
# ---------------------------------------------------------------------------

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers,
            name=name + '.' + name1 if name != '' else name1
        ))
    return res


# ---------------------------------------------------------------------------
# llama_sequential — faithful port + bias correction hook after fasterquant
# ---------------------------------------------------------------------------

@torch.no_grad()
def llama_sequential(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'norm') and model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )

    cache = {'i': 0, 'kwargs': {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.detach()
            cache['i'] += 1
            cache['kwargs'] = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'norm') and model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    layer_kwargs = {
        k: v for k, v in cache['kwargs'].items()
        if k != 'use_cache'
    }
    cache.clear()
    gc.collect()

    print('Ready.')

    quantizers: Dict[str, Quantizer] = {}
    total_error = 0.0
    total_time  = 0.0
    total_bc_norm = 0.0
    n_bc_applied = 0

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full  = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}

            gptq: Dict[str, GPTQ] = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f'  Block {i} sublayer {name} — quantizing...')
                error, elapsed = gptq[name].fasterquant(
                    blocksize=args.blocksize,
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                )

                # ------------------------------------------------------------
                # Bias correction: apply immediately after fasterquant, while
                # W_orig and x_mean are still alive in the GPTQ object and
                # the layer is still on `dev`.
                # ------------------------------------------------------------
                if args.bias_correction:
                    bc_norm = gptq[name].apply_bias_correction()
                    total_bc_norm += bc_norm
                    n_bc_applied  += 1
                    print(f'    error: {error:.4f}  time: {elapsed:.2f}s  '
                          f'bc_norm: {bc_norm:.4f}')
                else:
                    print(f'    error: {error:.4f}  time: {elapsed:.2f}s')

                total_error += error
                total_time  += elapsed

                layer_key = 'model.layers.%d.%s' % (i, name)
                quantizers[layer_key] = gptq[name].quantizer
                gptq[name].free()

        # Re-run the layer with quantized weights (and corrected biases)
        # to produce the inputs for the next block. This is the same step
        # the faithful port does — no behavioural change here, but note
        # that downstream sublayers in later blocks will now see inputs
        # produced with bias correction applied, which is exactly what
        # we want for the running-mean correction to compose correctly
        # across blocks.
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        gc.collect()

        inps, outs = outs, inps

        if HAS_PSUTIL:
            print(f'  Block {i+1}/{len(layers)} done.  '
                  f'RAM: {psutil.virtual_memory().percent:.1f}%')

    model.config.use_cache = use_cache

    print(f'\nTotal error: {total_error:.4f}  Total time: {total_time:.2f}s')
    if args.bias_correction and n_bc_applied > 0:
        print(f'Bias correction: applied to {n_bc_applied} sublayers, '
              f'mean ||bc||_2 = {total_bc_norm / n_bc_applied:.4f}')
    return quantizers


# ---------------------------------------------------------------------------
# Calibration dataloader (unchanged)
# ---------------------------------------------------------------------------

def build_dataloader(calib_texts, tokenizer, seqlen, nsamples, seed):
    rng = random.Random(seed)

    full_ids = []
    for text in calib_texts:
        ids = tokenizer(text, return_tensors='pt').input_ids[0].tolist()
        full_ids.extend(ids)
        if len(full_ids) > seqlen * nsamples * 4:
            break

    if len(full_ids) < seqlen * nsamples:
        while len(full_ids) < seqlen * nsamples:
            full_ids = full_ids + full_ids

    dataloader = []
    max_start = len(full_ids) - seqlen - 1
    for _ in range(nsamples):
        start = rng.randint(0, max_start)
        chunk = full_ids[start:start + seqlen]
        input_ids = torch.tensor([chunk], dtype=torch.long)
        dataloader.append((input_ids,))

    return dataloader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GPTQ + Bias Correction — faithful port for XL models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-path', type=str, default='./models/Mistral-7B-v0.3')
    parser.add_argument('--output-dir', type=str, default='./quantized_models/model_gptq_bc_xl')
    parser.add_argument('--calib-dataset', type=str, default='c4', choices=['c4', 'wikitext2'])
    parser.add_argument('--cache-dir', type=str, default='./calibration_cache')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--wbits', type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument('--groupsize', type=int, default=128)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=0.01)
    parser.add_argument('--sym', action='store_true', default=True)
    parser.add_argument('--asym', dest='sym', action='store_false')
    parser.add_argument('--true-sequential', action='store_true', default=True)
    parser.add_argument('--no-true-sequential', dest='true_sequential', action='store_false')
    parser.add_argument('--act-order', action='store_true', default=False)
    parser.add_argument('--static-groups', action='store_true', default=False)
    parser.add_argument('--bias-correction', action='store_true', default=True,
                        help='Apply naive bias correction after each sublayer (default: on)')
    parser.add_argument('--no-bias-correction', dest='bias_correction',
                        action='store_false')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 80)
    print('GPTQ + Bias Correction — faithful port of official llama.py')
    print('=' * 80)
    print(f'Model        : {args.model_path}')
    print(f'Bits         : {args.wbits}')
    print(f'Groupsize    : {args.groupsize}')
    print(f'Seqlen       : {args.seqlen}')
    print(f'Nsamples     : {args.nsamples}')
    print(f'Sym          : {args.sym}')
    print(f'TrueSeq      : {args.true_sequential}')
    print(f'ActOrder     : {args.act_order}')
    print(f'StaticGroups : {args.static_groups}')
    print(f'BiasCorrect  : {args.bias_correction}')
    print('=' * 80)

    print('\nLoading model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
    )
    model.eval()
    model.seqlen = args.seqlen

    print(f'\nLoading calibration: {args.calib_dataset}')
    if args.calib_dataset == 'c4':
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.nsamples, seqlen=args.seqlen,
            seed=args.seed, cache_dir=args.cache_dir
        )
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.nsamples, seqlen=args.seqlen,
            seed=args.seed, cache_dir=args.cache_dir
        )

    print('Building dataloader...')
    dataloader = build_dataloader(
        calib_texts, tokenizer, args.seqlen, args.nsamples, args.seed
    )
    print(f'  {len(dataloader)} calibration batches, each {args.seqlen} tokens')

    tick = time.time()
    quantizers = llama_sequential(model, dataloader, dev, args)
    print(f'\nTotal wall time: {time.time() - tick:.2f}s')

    print(f'\nSaving to {args.output_dir}...')
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print('Done.')


if __name__ == '__main__':
    main()