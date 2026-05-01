"""
Compute perplexity manually, bypassing HF's loss_function entirely.

This sidesteps any possible bug in transformers >=4.46 ForCausalLMLoss that
might miscompute loss for Qwen specifically.

Also tests the obvious "smoking gun" hypothesis: HF maybe shifts labels wrong
internally for Qwen.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

QWEN_PATH = "/models/Qwen2.5-7B"
LLAMA_PATH = "/models/Llama-3-8B"  # known-good reference


def manual_ppl(model_path, label, n_tokens=2048):
    print(f"\n=== {label} ===")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join([x for x in ds["text"] if x])
    ids = tok(full_text, return_tensors="pt", add_special_tokens=False).input_ids[:, :n_tokens].cuda()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Method 1: Use HF's built-in loss
    with torch.no_grad():
        out_hf = model(ids, labels=ids)
    ppl_hf = torch.exp(out_hf.loss).item()

    # Method 2: Compute logits, manually shift, manually compute CE
    with torch.no_grad():
        out_logits = model(ids)
    logits = out_logits.logits  # [1, T, V]

    # Shift: predict token t+1 from logits at position t
    shift_logits = logits[:, :-1, :].contiguous().float()  # [1, T-1, V]
    shift_labels = ids[:, 1:].contiguous()                 # [1, T-1]

    loss_manual = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    ppl_manual = torch.exp(loss_manual).item()

    # Method 3: Per-token NLL to look at distribution
    nll_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(1, -1)

    # Statistics
    nll_mean = nll_per_token.mean().item()
    nll_median = nll_per_token.median().item()
    nll_max = nll_per_token.max().item()
    nll_p99 = torch.quantile(nll_per_token.float(), 0.99).item()
    nll_p90 = torch.quantile(nll_per_token.float(), 0.90).item()

    print(f"  HF loss():             PPL = {ppl_hf:.4f}")
    print(f"  Manual CE:             PPL = {ppl_manual:.4f}")
    print(f"  Per-token NLL stats:")
    print(f"    mean   = {nll_mean:.4f}    -> exp = {torch.exp(torch.tensor(nll_mean)).item():.4f}")
    print(f"    median = {nll_median:.4f}  -> exp = {torch.exp(torch.tensor(nll_median)).item():.4f}")
    print(f"    p90    = {nll_p90:.4f}     -> exp = {torch.exp(torch.tensor(nll_p90)).item():.4f}")
    print(f"    p99    = {nll_p99:.4f}     -> exp = {torch.exp(torch.tensor(nll_p99)).item():.4f}")
    print(f"    max    = {nll_max:.4f}")

    # If median exp << mean exp, it's a heavy-tail issue (a few catastrophic tokens)
    print(f"  Mean/median PPL ratio: {torch.exp(torch.tensor(nll_mean)).item() / torch.exp(torch.tensor(nll_median)).item():.2f}x")
    print(f"  (high ratio = a few tokens dominate loss)")

    # Look at the worst 10 tokens
    top_k = 10
    worst_idx = torch.topk(nll_per_token[0], top_k).indices
    print(f"\n  Top {top_k} worst-predicted tokens:")
    for rank, pos in enumerate(worst_idx.tolist()):
        nll = nll_per_token[0, pos].item()
        token_id = ids[0, pos + 1].item()  # +1 because we shifted
        token_str = tok.decode([token_id])
        # Show prev 5 tokens for context
        ctx_start = max(0, pos - 4)
        ctx_ids = ids[0, ctx_start:pos + 2].tolist()
        ctx_str = tok.decode(ctx_ids)
        print(f"    [{rank+1}] pos={pos:>4} NLL={nll:>6.2f} tok={token_id} {repr(token_str):>20s} ctx: {repr(ctx_str)}")

    del model
    torch.cuda.empty_cache()
    return ppl_hf, ppl_manual


qwen_results = manual_ppl(QWEN_PATH, "Qwen2.5-7B")
llama_results = manual_ppl(LLAMA_PATH, "Llama-3-8B")

print("\n" + "=" * 60)
print("FINAL")
print("=" * 60)
print(f"Qwen2.5-7B:  HF={qwen_results[0]:.4f}  Manual={qwen_results[1]:.4f}")
print(f"Llama-3-8B:  HF={llama_results[0]:.4f}  Manual={llama_results[1]:.4f}")
print()
print("If HF and Manual agree for both: HF loss is fine, PPL just is what it is.")
print("If they DISAGREE for Qwen but AGREE for Llama: HF ForCausalLMLoss bug for Qwen.")