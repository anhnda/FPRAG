"""
Diagnose Qwen2.5 PPL=22 issue.

Run this BEFORE the full eval to identify which of the three failure modes hit.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

MODEL_PATH = "/models/Qwen2.5-7B"  # change as needed

print("=" * 60)
print("DIAGNOSTIC 1: Is this an AWQ checkpoint or fp16 base?")
print("=" * 60)

config_path = os.path.join(MODEL_PATH, "config.json")
with open(config_path) as f:
    config = json.load(f)

print(f"  model_type:       {config.get('model_type')}")
print(f"  torch_dtype:      {config.get('torch_dtype')}")
print(f"  quantization_cfg: {config.get('quantization_config', 'NONE — this is fp16/bf16 base')}")

is_quantized = "quantization_config" in config
print(f"  -> is_quantized:  {is_quantized}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Load in correct dtype")
print("=" * 60)

# Use bfloat16 for Qwen2.5 (its native training dtype)
# Use `dtype=` not `torch_dtype=` on recent transformers
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,           # NEW arg name
        device_map="cuda",
        trust_remote_code=True,
    )
except TypeError:
    # Fallback for older transformers
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

print(f"  model.dtype:                    {model.dtype}")
print(f"  first param dtype:              {next(model.parameters()).dtype}")
print(f"  embed_tokens dtype:             {model.model.embed_tokens.weight.dtype}")
print(f"  layer 0 q_proj weight dtype:    {model.model.layers[0].self_attn.q_proj.weight.dtype}")
print(f"  layer 0 q_proj weight shape:    {model.model.layers[0].self_attn.q_proj.weight.shape}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Tiny PPL sanity check (50 tokens)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
text = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models are evaluated using perplexity, "
    "which measures how well a probability distribution predicts a sample."
)
ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

with torch.no_grad():
    out = model(ids, labels=ids)

ppl = torch.exp(out.loss).item()
print(f"  Loss: {out.loss.item():.4f}")
print(f"  PPL on tiny sample: {ppl:.2f}")
print(f"  Expected for Qwen2.5-7B base: 5-15 (varies by text)")
print(f"  If >50: model is broken (wrong dtype, corrupt weights, wrong kernel)")