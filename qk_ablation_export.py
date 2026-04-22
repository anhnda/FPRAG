"""
ablation_awq_attention.py

Produce four model snapshots that selectively un-quantize attention layers
of an already-AWQ-quantized model, by overwriting chosen Linear weights
with their FP16 counterparts from the original checkpoint.

Outputs four directories, each a full HuggingFace model ready to be loaded
by any external evaluator:

    <output-root>/awq_baseline/       (pristine AWQ, copy of input)
    <output-root>/awq_except_q/       (Q restored to FP)
    <output-root>/awq_except_qk/      (Q and K restored to FP)
    <output-root>/fp_ceiling/         (all target Linears restored to FP)

Usage:
    python ablation_awq_attention.py \
        --awq-path    ./quantized_models/Llama-3-8B_awq_raw \
        --fp-path     /models/Llama-3-8B \
        --output-root ./ablation_models/Llama-3-8B
"""

import argparse
import gc
import os
import shutil
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------- #
# Layer matching.                                                       #
# --------------------------------------------------------------------- #

def match_layer(name, kinds):
    """True if the Linear's leaf name is in `kinds`."""
    return name.split('.')[-1] in kinds


# --------------------------------------------------------------------- #
# FP state dict.                                                        #
# --------------------------------------------------------------------- #

@torch.no_grad()
def load_fp_state_dict(fp_path, dtype):
    """Load the original FP model's weights into a CPU state dict."""
    print(f"Loading FP weights from {fp_path} ...")
    fp_model = AutoModelForCausalLM.from_pretrained(
        fp_path, torch_dtype=dtype, device_map='cpu', trust_remote_code=True,
    )
    sd = {k: v.detach().clone() for k, v in fp_model.state_dict().items()}
    del fp_model
    gc.collect()
    return sd


# --------------------------------------------------------------------- #
# Weight overwrite.                                                     #
# --------------------------------------------------------------------- #

@torch.no_grad()
def overwrite_from_fp(model, fp_sd, kinds):
    """Overwrite all Linears whose leaf name is in `kinds` with FP weights.
    Returns (n_done, n_missing)."""
    if not kinds:
        return 0, 0
    n_done, n_missing = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not match_layer(name, kinds):
            continue
        key = name + '.weight'
        if key not in fp_sd:
            n_missing += 1
            continue
        module.weight.data.copy_(
            fp_sd[key].to(module.weight.device).to(module.weight.dtype)
        )
        n_done += 1
    return n_done, n_missing


# --------------------------------------------------------------------- #
# Save helper.                                                          #
# --------------------------------------------------------------------- #

def save_snapshot(model, tokenizer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print(f"  → saved to {out_dir}")


# --------------------------------------------------------------------- #
# Main.                                                                 #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--awq-path', type=str, required=True,
                        help='Path to the AWQ-quantized model directory')
    parser.add_argument('--fp-path', type=str, required=True,
                        help='Path to the original FP model directory')
    parser.add_argument('--output-root', type=str, required=True,
                        help='Root directory for the four output snapshots')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--skip-baseline-copy', action='store_true',
                        help='If set, do not re-save the pristine AWQ model; '
                             'your evaluator can point at --awq-path directly.')
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    # The four configurations.
    # Each maps a short name → set of Linear leaf names to restore to FP.
    configs = [
        ('awq_baseline',   set()),
        ('awq_except_q',   {'q_proj'}),
        ('awq_except_qk',  {'q_proj', 'k_proj'}),
        ('fp_ceiling',     {'q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj', 'lm_head'}),
    ]

    # --- Load FP weights once (CPU) ---
    fp_sd = load_fp_state_dict(args.fp_path, dtype=dtype)

    # --- Iterate over configurations ---
    os.makedirs(args.output_root, exist_ok=True)
    for cfg_name, kinds in configs:
        out_dir = os.path.join(args.output_root, cfg_name)
        print(f"\n{'=' * 80}\n[{cfg_name}]  restore → {sorted(kinds) if kinds else '∅'}\n{'=' * 80}")

        # Fast path for the pristine-AWQ config: just copy the directory.
        if not kinds:
            if args.skip_baseline_copy:
                print("  (skipped per --skip-baseline-copy)")
                continue
            if os.path.abspath(out_dir) == os.path.abspath(args.awq_path):
                print("  (output == input; nothing to copy)")
                continue
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            shutil.copytree(args.awq_path, out_dir)
            print(f"  copied pristine AWQ → {out_dir}")
            continue

        # Load the AWQ-quantized model afresh for each configuration.
        # This avoids any risk of state leaking between configurations
        # and keeps peak memory at one model's worth.
        print(f"  loading AWQ model from {args.awq_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.awq_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.awq_path, torch_dtype=dtype,
            device_map='cpu', trust_remote_code=True,
        )
        model.eval()

        n_done, n_missing = overwrite_from_fp(model, fp_sd, kinds)
        print(f"  overwrote {n_done} Linears, {n_missing} missing")
        if n_done == 0:
            print("  ! nothing to overwrite; skipping save")
            del model; gc.collect()
            continue

        save_snapshot(model, tokenizer, out_dir)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'=' * 80}\nDone. Output snapshots:")
    for cfg_name, _ in configs:
        out_dir = os.path.join(args.output_root, cfg_name)
        marker = '✓' if os.path.exists(out_dir) else '✗'
        print(f"  {marker}  {out_dir}")
    print(f"\nRun your evaluator on each directory in turn.")


if __name__ == '__main__':
    main()