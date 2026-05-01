"""
Final differential diagnostic for Qwen2.5 PPL bug.

Tests the SAME 4096-token chunk under 4 conditions to isolate the cause.
All four numbers should be ~6-8 for a healthy Qwen2.5-7B.
Whichever condition gives PPL ~22 identifies the bug.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "/models/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
full_text = "\n".join([x for x in dataset["text"] if x])
ids_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids


def run(label, ids, attn_impl="sdpa", clear_gen_config=False):
    print(f"\n--- {label} ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    if clear_gen_config:
        # Nuke generation_config to prevent any leakage into forward
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig()
    model.eval()

    print(f"  attn={attn_impl}  cleared_gen_config={clear_gen_config}")
    print(f"  ids shape: {tuple(ids.shape)}")

    with torch.no_grad():
        out = model(ids.cuda(), labels=ids.cuda())
    ppl = torch.exp(out.loss).item()
    print(f"  PPL: {ppl:.4f}")

    del model
    torch.cuda.empty_cache()
    return ppl


# Test 1: First 2048 tokens (single window, no sliding logic)
test_2k = ids_full[:, :2048]
ppl_2k_sdpa = run("T1: 2048 tokens, sdpa, default gen_config", test_2k, "sdpa", False)

# Test 2: First 2048 tokens, eager attention
ppl_2k_eager = run("T2: 2048 tokens, eager, default gen_config", test_2k, "eager", False)

# Test 3: First 2048 tokens, sdpa, gen_config wiped
ppl_2k_clean = run("T3: 2048 tokens, sdpa, cleared gen_config", test_2k, "sdpa", True)

# Test 4: First 4096 tokens (longer context)
test_4k = ids_full[:, :4096]
ppl_4k = run("T4: 4096 tokens, sdpa, default gen_config", test_4k, "sdpa", False)


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  T1 (2k, sdpa, default):    PPL = {ppl_2k_sdpa:.4f}")
print(f"  T2 (2k, eager, default):   PPL = {ppl_2k_eager:.4f}")
print(f"  T3 (2k, sdpa, no-gencfg):  PPL = {ppl_2k_clean:.4f}")
print(f"  T4 (4k, sdpa, default):    PPL = {ppl_4k:.4f}")

print("\nInterpretation:")
print("  - All ~6-8:                Healthy. Bug is in sliding window logic.")
print("  - T2 << T1:                Attention backend bug — use eager.")
print("  - T3 << T1:                Generation config pollution.")
print("  - T4 >> T1:                Long-context bug (RoPE/sliding window).")
print("  - All ~22:                 Something fundamental to forward pass.")