"""
Three more diagnostic checks for the Qwen2.5 PPL bug.
Run all three and paste output.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, json

MODEL_PATH = "/models/Qwen2.5-7B"

print("=" * 60)
print("CHECK A: Files in model directory")
print("=" * 60)
for f in sorted(os.listdir(MODEL_PATH)):
    path = os.path.join(MODEL_PATH, f)
    size = os.path.getsize(path) / 1e6
    print(f"  {f:50s} {size:>10.2f} MB")

print("\n" + "=" * 60)
print("CHECK B: Tokenizer-model vocab alignment")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  tokenizer.vocab_size:       {tokenizer.vocab_size}")
print(f"  len(tokenizer):             {len(tokenizer)}")
print(f"  tokenizer special tokens:   {len(tokenizer.all_special_tokens)}")

with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)
print(f"  config.vocab_size:          {config['vocab_size']}")
print(f"  config.tie_word_embeddings: {config.get('tie_word_embeddings', 'N/A')}")
print(f"  config.rope_theta:          {config.get('rope_theta')}")
print(f"  config.max_pos_embeddings:  {config.get('max_position_embeddings')}")
print(f"  config.sliding_window:      {config.get('sliding_window')}")
print(f"  config.use_sliding_window:  {config.get('use_sliding_window')}")

print("\nExpected for Qwen2.5-7B base:")
print("  vocab_size: 152064")
print("  tie_word_embeddings: false  (7B does NOT tie; only 0.5B/1.5B tie)")
print("  rope_theta: 1000000.0")
print("  max_position_embeddings: 131072")
print("  sliding_window: 131072")
print("  use_sliding_window: false")

print("\n" + "=" * 60)
print("CHECK C: generation_config.json contents")
print("=" * 60)
gen_path = os.path.join(MODEL_PATH, "generation_config.json")
if os.path.exists(gen_path):
    with open(gen_path) as f:
        gen_config = json.load(f)
    for k, v in gen_config.items():
        print(f"  {k}: {v}")
else:
    print("  No generation_config.json (fine, defaults will be used)")

print("\n" + "=" * 60)
print("CHECK D: Single forward pass — verify logits are sane")
print("=" * 60)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)

# Feed "The capital of France is" — model should put high prob on " Paris"
text = "The capital of France is"
ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
print(f"  Input tokens: {ids[0].tolist()}")
print(f"  Decoded:      {[tokenizer.decode([t]) for t in ids[0].tolist()]}")

with torch.no_grad():
    out = model(ids)
logits = out.logits[0, -1]   # last position predicts next token
probs = torch.softmax(logits.float(), dim=-1)
top5 = torch.topk(probs, 5)
print(f"\n  Top 5 predictions for next token after '{text}':")
for prob, idx in zip(top5.values, top5.indices):
    tok_str = tokenizer.decode([idx.item()])
    print(f"    {prob.item():.4f}  id={idx.item():>6}  {repr(tok_str)}")
print("\n  Expected: ' Paris' should be #1 with prob > 0.5")
print("  If top-1 is unrelated, model weights are scrambled despite loading OK")