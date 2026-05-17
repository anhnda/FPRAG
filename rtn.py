"""
Raw Round-To-Nearest (RTN) Quantization - Baseline

The simplest possible PTQ baseline:
- No calibration data required
- No activation statistics
- No grid search, no scaling, no heuristics
- Group-wise asymmetric quantization [0, 2^bits - 1]
- Per-group min/max → scale, zero_point → round → clamp → dequantize

This is the standard reference point against which AWQ, GPTQ, etc. are compared.

Algorithm (per Linear layer W of shape [out, in]):
    1. Pad in_features to multiple of group_size
    2. Reshape W into [out, n_groups, group_size]
    3. Per group:
         scale = (max - min) / (2^bits - 1)
         zp    = round(-min / scale)  clamped to [0, 2^bits - 1]
         w_q   = round(W / scale + zp).clamp(0, 2^bits - 1)
         W_dq  = (w_q - zp) * scale
    4. Reshape back, strip padding, write back to module.

Options:
    --skip-lm-head   (default: True)  Leave lm_head in full precision.
                                       Common practice — lm_head is small relative to
                                       backbone and quantizing it tends to hurt PPL
                                       disproportionately on big-vocab models.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import gc


class RTNQuantizer:
    def __init__(self, model, device="cuda", bits=4, group_size=128, skip_lm_head=True):
        self.model = model
        self.device = device
        self.bits = bits
        self.group_size = group_size
        self.skip_lm_head = skip_lm_head

        print(f"\n[RTN Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Skip lm_head: {skip_lm_head}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric(self, W):
        """
        Standard group-wise asymmetric RTN.

        Args:
            W: [out_features, in_features] weight tensor
        Returns:
            W_dequant: same shape, dtype as W, after fake-quantization
        """
        out_features, in_features = W.shape
        device = W.device
        dtype = W.dtype

        # Pad in_features to a multiple of group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in = n_groups * self.group_size

        if padded_in > in_features:
            W_padded = torch.zeros(out_features, padded_in, device=device, dtype=dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        # [out, n_groups, group_size]
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

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Replace module.weight with its fake-quantized version."""
        W = module.weight.data
        W_dq = self.quantize_weight_groupwise_asymmetric(W)
        module.weight.data = W_dq
        del W_dq

    def quantize_model(self):
        """Iterate over all Linear modules and apply RTN."""
        print("\n" + "=" * 80)
        print("Raw RTN Quantization")
        print("=" * 80)

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        n_total = len(layer_names)
        n_quantized = 0
        n_skipped = 0

        print(f"  Found {n_total} Linear layers")

        for name, module in tqdm(layer_names, desc="RTN"):
            is_lmhead = ('lm_head' in name.lower()) or name.endswith('lm_head')
            if is_lmhead and self.skip_lm_head:
                n_skipped += 1
                continue
            try:
                self.quantize_layer(name, module)
                n_quantized += 1
            except Exception as e:
                print(f"\n⚠️  Error quantizing {name}: {e}")
                n_skipped += 1
                continue

            # Light cleanup; RTN is cheap but big lm_heads can spike
            if (n_quantized % 32) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("\n" + "=" * 80)
        print(f"✓ RTN Complete: quantized {n_quantized}/{n_total} layers "
              f"(skipped {n_skipped})")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Raw RTN (Round-To-Nearest) quantization baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Quantization bit width")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for group-wise quantization")
    parser.add_argument("--skip-lm-head", action="store_true", default=True,
                        help="Leave lm_head in full precision (default: True)")
    parser.add_argument("--quantize-lm-head", dest="skip_lm_head", action="store_false",
                        help="Quantize lm_head as well")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                        help="Model name or local path")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_rtn",
                        help="Where to save the quantized model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Raw RTN Quantization")
    print(f"Target Model: {args.model_path}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Bits: {args.bits}")
    print(f"Group size: {args.group_size}")
    print(f"Skip lm_head: {args.skip_lm_head}")
    print("=" * 80)

    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    quantizer = RTNQuantizer(
        model=model,
        device=device,
        bits=args.bits,
        group_size=args.group_size,
        skip_lm_head=args.skip_lm_head
    )
    quantizer.quantize_model()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved RTN-quantized model to {args.output_dir}")


if __name__ == "__main__":
    main()