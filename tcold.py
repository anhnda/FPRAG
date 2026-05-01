"""
Final test: does Qwen2.5-7B PPL converge to a sensible value as we evaluate
longer chunks? If yes, the "PPL 20" was a cold-start artifact.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

QWEN_PATH = "/models/Qwen2.5-7B"

tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
full_text = "\n".join([x for x in ds["text"] if x])
ids_full = tok(full_text, return_tensors="pt", add_special_tokens=False).input_ids

model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()

# Test 1: PPL of single chunks of varying lengths
print("Test 1: PPL of single forward pass at varying chunk sizes")
print("(measures how cold-start penalty amortizes)")
print(f"{'n_tokens':>10}  {'PPL':>10}  {'NLL/token':>12}")
for n in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    if n > ids_full.size(1):
        break
    ids = ids_full[:, :n].cuda()
    with torch.no_grad():
        out = model(ids, labels=ids)
    ppl = torch.exp(out.loss).item()
    print(f"{n:>10}  {ppl:>10.4f}  {out.loss.item():>12.4f}")

print()
print("Test 2: PPL of 2048-token chunks at different positions in the stream")
print("(measures whether position-0 cold-start is the culprit)")
print(f"{'start':>10}  {'PPL':>10}")
for start in [0, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 200000]:
    if start + 2048 > ids_full.size(1):
        break
    ids = ids_full[:, start:start+2048].cuda()
    with torch.no_grad():
        out = model(ids, labels=ids)
    ppl = torch.exp(out.loss).item()
    print(f"{start:>10}  {ppl:>10.4f}")

print()
print("Test 3: Chunk starting at position 0, but with first 512 tokens masked from loss")
print("(measures pure 'warm' loss ignoring the cold-start tokens)")
ids = ids_full[:, :2048].cuda()
labels = ids.clone()
labels[:, :512] = -100  # ignore first 512 from loss
with torch.no_grad():
    out = model(ids, labels=labels)
ppl_warm = torch.exp(out.loss).item()
print(f"  PPL (first 512 masked): {ppl_warm:.4f}")
print(f"  vs PPL (no masking):    19.6377")
print(f"  If warm PPL << 19, cold-start is confirmed as the cause.")