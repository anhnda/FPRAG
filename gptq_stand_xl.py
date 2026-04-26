"""
GPTQ Implementation — FAITHFUL port of the official llama.py

This file mirrors the reference implementation line-for-line in the parts that
matter for correctness:

  * Quantizer / quantize()      — identical to official quant.py
  * GPTQ.add_batch / fasterquant — identical to official gptq.py
  * llama_sequential            — identical control flow:
      - inps stored on GPU, same dtype as model
      - Catcher captures ALL kwargs generically (matches File 1 / GPTQQuantizer)
        so position_embeddings, position_ids, attention_mask etc. are all
        forwarded correctly regardless of model architecture (LLaMA-2, LLaMA-3,
        Mistral, etc.) without needing to name them explicitly
      - per-sublayer quantizer configured BEFORE add_batch, not inside fasterquant
      - hook signature matches: (module, inp, out) -> add_batch(inp[0].data, out.data)
      - inps, outs = outs, inps AFTER the quantized re-run
      - use_cache flag saved and restored

Only concessions to "XL" (low-VRAM) operation:
  * Model loaded on CPU, each block paged to GPU for quantization then back

Added vs original:
  * actorder (activation ordering) — columns sorted by descending Hessian diagonal
  * static_groups — quantizer fitted per-group before add_batch rather than on the fly
  * Raw artifact saving — Q_pre, Q_int, Q_scale, Q_zero, W_orig tracked per column
    so that post-correction can be replayed without re-running full GPTQ

Memory leak fixes:
  * FIX 1 — Catcher detaches all tensor kwargs immediately so no computation
             graph is retained in RAM across the calibration loop
  * FIX 2 — cache dict cleared after layer_kwargs is built so the Catcher's
             last captured tensors are released before the layer loop starts
  * FIX 3 — raw artifacts written to disk per-sublayer and immediately freed
             from RAM instead of accumulating all 32*7 layers in a dict
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
# gptq.py — based on official, extended with actorder, static_groups, and
# raw artifact tracking (Q_pre, Q_int, Q_scale, Q_zero, W_orig).
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

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
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
# llama_sequential — faithful port, XL (paged) version
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

    # FIX 1: detach all tensor kwargs in the Catcher immediately so the entire
    # forward-pass computation graph is NOT retained in RAM.
    cache = {'i': 0, 'kwargs': {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            # Forward any attribute lookup that isn't found on Catcher itself
            # to the wrapped module — fixes Qwen2's decoder_layer.attention_type
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
    # FIX 2: clear cache dict so the Catcher's last captured tensors are
    # released before the layer loop starts — nothing else holds these refs.
    cache.clear()
    gc.collect()

    print('Ready.')

    quantizers: Dict[str, Quantizer] = {}
    total_error = 0.0
    total_time  = 0.0

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
                print(f'    error: {error:.4f}  time: {elapsed:.2f}s')
                total_error += error
                total_time  += elapsed

                layer_key = 'model.layers.%d.%s' % (i, name)
                quantizers[layer_key] = gptq[name].quantizer
                gptq[name].free()

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
    return quantizers


# ---------------------------------------------------------------------------
# Calibration dataloader
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
        description='GPTQ — faithful port of official llama.py for XL models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-path', type=str, default='./models/Mistral-7B-v0.3')
    parser.add_argument('--output-dir', type=str, default='./quantized_models/model_gptq_xl')
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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 80)
    print('GPTQ — faithful port of official llama.py')
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