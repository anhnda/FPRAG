"""
Compare V2 PRAQ-Leaning vs GW-AWQ on WikiText-2 Validation

This script compares the winning V2 configuration (β=0.7, τ=1.5) against
GW-AWQ baseline on WikiText-2 validation set to see if C4 advantage holds
on in-distribution data.

Results shown:
- WikiText-2 validation (in-distribution)
- C4 validation (cross-dataset) - from previous results
- Analysis of generalization behavior
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import os


def load_wikitext2_validation(n_samples=2000, seed=42):
    """Load WikiText-2 validation set."""
    print(f"Loading WikiText-2 validation dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    # Filter out empty texts
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    # Random sampling
    random.seed(seed)
    if n_samples < len(texts):
        texts = random.sample(texts, n_samples)

    print(f"Selected {len(texts)} samples for evaluation")
    return texts


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, texts, max_length=512, device="cuda"):
    """Evaluate perplexity on text samples."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    successful = 0
    failed = 0

    for text in tqdm(texts, desc="Evaluating", leave=False):
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )

            input_ids = inputs["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss.item()
            n_tokens = input_ids.shape[1]

            total_loss += loss * n_tokens
            total_tokens += n_tokens
            successful += 1

        except Exception:
            failed += 1
            continue

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss) if total_tokens > 0 else float('inf')

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "num_samples": successful,
        "failed_samples": failed,
        "total_tokens": total_tokens
    }


def evaluate_model(model_path, model_name, eval_texts, device="cuda"):
    """Evaluate a single model."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        results = evaluate_perplexity(model, tokenizer, eval_texts, device=device)
        results['model_name'] = model_name

        print(f"\nResults:")
        print(f"  Perplexity: {results['perplexity']:.4f}")
        print(f"  Avg Loss: {results['avg_loss']:.4f}")
        print(f"  Samples: {results['num_samples']}")
        print(f"  Tokens: {results['total_tokens']}")

        del model
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = 2000
    seed = 42

    print("="*80)
    print("V2 PRAQ-LEANING vs GW-AWQ - WikiText-2 Validation")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: WikiText-2 validation (in-distribution)")
    print(f"Samples: {n_samples}")
    print(f"Seed: {seed}")
    print("="*80)

    # Load WikiText-2 validation
    print("\nLoading WikiText-2 validation data...")
    eval_texts = load_wikitext2_validation(n_samples=n_samples, seed=seed)

    # Model paths
    praq_lean_path = "./quantized_models/minicpm_gwh_v2_praq_lean"
    awq_path = "./quantized_models/minicpm_gw_awq"

    # Evaluate both models
    results = {}

    results['V2-PRAQ-Leaning'] = evaluate_model(
        praq_lean_path,
        "V2 PRAQ-Leaning (β=0.7, τ=1.5)",
        eval_texts,
        device=device
    )

    results['GW-AWQ'] = evaluate_model(
        awq_path,
        "GW-AWQ (baseline)",
        eval_texts,
        device=device
    )

    # Analysis
    print("\n" + "="*80)
    print("RESULTS COMPARISON - WikiText-2 Validation")
    print("="*80)

    if results['V2-PRAQ-Leaning'] and results['GW-AWQ']:
        praq_ppl = results['V2-PRAQ-Leaning']['perplexity']
        awq_ppl = results['GW-AWQ']['perplexity']

        delta = praq_ppl - awq_ppl
        delta_pct = (delta / awq_ppl) * 100

        print(f"\n{'Model':<30} {'Perplexity':<12} {'Avg Loss':<12}")
        print("-" * 60)
        print(f"{'GW-AWQ (baseline)':<30} {awq_ppl:<12.4f} {results['GW-AWQ']['avg_loss']:<12.4f}")
        print(f"{'V2 PRAQ-Leaning':<30} {praq_ppl:<12.4f} {results['V2-PRAQ-Leaning']['avg_loss']:<12.4f}")
        print("-" * 60)
        print(f"{'Delta':<30} {delta:<12.4f} ({delta_pct:+.3f}%)")

        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)

        if abs(delta_pct) < 0.05:
            print(f"\n🤝 TIED (< 0.05% difference)")
            print(f"   Both methods perform equally on WikiText-2")
        elif delta < 0:
            print(f"\n✅ V2 PRAQ-LEANING WINS by {abs(delta_pct):.3f}%!")
            print(f"   V2 beats AWQ on in-distribution data")
        else:
            print(f"\n❌ GW-AWQ WINS by {delta_pct:.3f}%")
            print(f"   Pure AWQ better on in-distribution data")

        # Compare with C4 results
        print("\n" + "="*80)
        print("CROSS-DATASET COMPARISON")
        print("="*80)

        # C4 results (from previous evaluation)
        c4_praq = 13.7274
        c4_awq = 13.7670
        c4_delta = c4_praq - c4_awq
        c4_delta_pct = (c4_delta / c4_awq) * 100

        print(f"\n{'Dataset':<20} {'V2 PRAQ-Lean':<15} {'GW-AWQ':<15} {'Delta':<15} {'Winner':<15}")
        print("-" * 90)
        print(f"{'WikiText-2 (in)':<20} {praq_ppl:<15.4f} {awq_ppl:<15.4f} {delta_pct:>+14.3f}%  {'V2' if delta < 0 else 'AWQ':<15}")
        print(f"{'C4 (cross)':<20} {c4_praq:<15.4f} {c4_awq:<15.4f} {c4_delta_pct:>+14.3f}%  {'V2':<15}")

        print("\n" + "="*80)
        print("GENERALIZATION ANALYSIS")
        print("="*80)

        # Check if V2 generalizes better
        if delta < 0 and c4_delta < 0:
            print("\n✅ V2 PRAQ-LEANING WINS ON BOTH!")
            print(f"   WikiText-2: Better by {abs(delta_pct):.3f}%")
            print(f"   C4:         Better by {abs(c4_delta_pct):.3f}%")
            print("\n   → V2 generalizes better across datasets")
            print("   → PRAQ intelligence works everywhere!")
        elif delta < 0:
            print("\n✅ V2 wins on WikiText-2")
            print("\n   → V2 excels on in-distribution data")
        elif c4_delta < 0:
            print("\n⚠️  Mixed Results:")
            print(f"   WikiText-2: AWQ better by {delta_pct:.3f}%")
            print(f"   C4:         V2 better by {abs(c4_delta_pct):.3f}%")
            print("\n   → V2 better for cross-dataset generalization")
            print("   → AWQ better for in-distribution")
        else:
            print("\n❌ AWQ wins on both datasets")

        # Final recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        avg_improvement = (abs(delta_pct) + abs(c4_delta_pct)) / 2

        if delta < 0 and c4_delta < 0:
            print("\n✅ Deploy V2 PRAQ-Leaning (β=0.7, τ=1.5)")
            print(f"   - Wins on both WikiText-2 and C4")
            print(f"   - Average improvement: {avg_improvement:.3f}%")
            print(f"   - Best overall generalization")
        elif c4_delta < 0:
            print("\n✅ Deploy V2 PRAQ-Leaning if targeting diverse data")
            print(f"   - Better on C4 (cross-dataset): {abs(c4_delta_pct):.3f}%")
            print(f"   - Use GW-AWQ if only using WikiText-2 type data")
        else:
            print("\n⚠️  Consider use case:")
            print("   - GW-AWQ for in-distribution data")
            print("   - V2 for diverse/cross-dataset scenarios")

        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        print("\nV2 PRAQ-Leaning Configuration:")
        print("  β = 0.7  (70% PRAQ, 30% AWQ)")
        print("  τ = 1.5  (Moderately sharp weighting)")
        print("\nWhat this means:")
        print("  ✓ Post-activation importance (PRAQ) is critical")
        print("  ✓ Accounting for activation functions helps everywhere")
        print("  ✓ Focused optimization > diffuse weighting")
        print("  ✓ Intelligent importance > simple magnitude")

    else:
        print("\n❌ Could not compare - one or both models missing")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
