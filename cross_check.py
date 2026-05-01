"""
Cross-model sanity: identical 2048-token WikiText chunk, identical code,
on Qwen vs Llama. If Qwen gives 20 and Llama gives 6, the bug is Qwen-specific.
If both inflate, your pipeline is the problem.

Edit LLAMA_PATH to point to your working Llama-3-8B / Llama-2-7B / Mistral-7B.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

QWEN_PATH = "/models/Qwen2.5-7B"
LLAMA_PATH = "/models/Llama-3-8B"   # ← EDIT to your known-good model path


def build_chunk(model_path, n_tokens=2048):
    """Tokenize WikiText-2 with this model's tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join([x for x in ds["text"] if x])
    ids = tok(full_text, return_tensors="pt", add_special_tokens=False).input_ids[:, :n_tokens]
    return tok, ids


def measure(model_path, label):
    print(f"\n=== {label} ===")
    tok, ids = build_chunk(model_path)
    print(f"  Tokenizer: {type(tok).__name__}")
    print(f"  BOS={tok.bos_token_id}  EOS={tok.eos_token_id}")
    print(f"  First 5 IDs: {ids[0, :5].tolist()}")
    print(f"  Decoded:     {[tok.decode([t]) for t in ids[0, :5].tolist()]}")
    print(f"  Total tokens: {ids.size(1)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Test A: no BOS prepended
    with torch.no_grad():
        out_a = model(ids.cuda(), labels=ids.cuda())
    ppl_a = torch.exp(out_a.loss).item()
    print(f"  PPL (no BOS):   {ppl_a:.4f}")

    # Test B: BOS prepended (if model has one)
    if tok.bos_token_id is not None and tok.bos_token_id != tok.eos_token_id:
        bos = torch.tensor([[tok.bos_token_id]])
        ids_with_bos = torch.cat([bos, ids], dim=1).cuda()
        # Mask BOS from loss so we compare same set of predicted tokens
        labels = ids_with_bos.clone()
        labels[:, 0] = -100
        with torch.no_grad():
            out_b = model(ids_with_bos, labels=labels)
        ppl_b = torch.exp(out_b.loss).item()
        print(f"  PPL (with BOS): {ppl_b:.4f}")

    del model
    torch.cuda.empty_cache()
    return ppl_a


qwen_ppl = measure(QWEN_PATH, "Qwen2.5-7B")
llama_ppl = measure(LLAMA_PATH, "Llama-3-8B (reference)")

print("\n" + "=" * 50)
print(f"Qwen2.5-7B PPL:  {qwen_ppl:.4f}")
print(f"Llama-3-8B PPL:  {llama_ppl:.4f}")
print(f"Ratio:           {qwen_ppl / llama_ppl:.2f}x")
print("If Qwen/Llama ratio >2, bug is Qwen-specific.")
print("If both >15, the WikiText preprocessing or pipeline is the issue.")