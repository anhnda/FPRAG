"""
Look at Qwen2.5-7B's top-5 predictions at the catastrophically-failing positions.

If the model is putting high mass on weird/wrong tokens at these positions,
something is genuinely wrong with the model's behavior — not a numerical edge case.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

QWEN_PATH = "/models/Qwen2.5-7B"

tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
full_text = "\n".join([x for x in ds["text"] if x])
ids = tok(full_text, return_tensors="pt", add_special_tokens=False).input_ids[:, :2048].cuda()

model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()

with torch.no_grad():
    out = model(ids)
logits = out.logits[0]  # [T, V]

# Catastrophic positions from your output (these are pos in shifted labels;
# logit position = pos, predicting ids[pos+1])
worst_positions = [1941, 1725, 288, 993, 1950, 1959, 1455, 371, 27, 1984]

for pos in worst_positions:
    actual_token_id = ids[0, pos + 1].item()
    actual_token_str = tok.decode([actual_token_id])

    # Context: 8 tokens before and including current position
    ctx_start = max(0, pos - 7)
    ctx_ids = ids[0, ctx_start:pos + 1].tolist()
    ctx_str = tok.decode(ctx_ids)

    # Top-5 predictions
    probs = F.softmax(logits[pos].float(), dim=-1)
    top5 = torch.topk(probs, 5)

    # Probability of actual token
    actual_prob = probs[actual_token_id].item()
    actual_rank = (probs > probs[actual_token_id]).sum().item() + 1

    print(f"\n--- Position {pos} ---")
    print(f"  Context:   {repr(ctx_str)}")
    print(f"  Actual:    {repr(actual_token_str)} (id={actual_token_id})")
    print(f"  Actual prob: {actual_prob:.2e}    rank: {actual_rank}")
    print(f"  Top-5 predictions:")
    for prob, idx in zip(top5.values, top5.indices):
        print(f"    {prob.item():>8.4f}  id={idx.item():>6}  {repr(tok.decode([idx.item()]))}")

    # Logit stats
    logit_max = logits[pos].max().item()
    logit_min = logits[pos].min().item()
    logit_mean = logits[pos].mean().item()
    logit_std = logits[pos].std().item()
    print(f"  Logit stats: max={logit_max:.2f} min={logit_min:.2f} mean={logit_mean:.2f} std={logit_std:.2f}")