"""
Cross-Dataset Test: V1 GWH-PRAQ vs GW-AWQ on The Pile

Tests the winning methods on The Pile validation set to validate
generalization across a third, diverse dataset.

The Pile contains:
- Books, academic papers, code, web text, conversations, etc.
- 22 diverse data sources
- Different from WikiText-2 (Wikipedia) and C4 (web crawl)
- Gold standard for diverse evaluation

Results Context:
- WikiText-2: V1 wins (16.5159 vs 16.5513)
- C4: V1 nearly ties (13.7577 vs 13.7670)
- The Pile: TBD
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import os


def load_pile_validation(n_samples=2000, seed=42):
    """
    Load The Pile validation set.

    The Pile is a diverse dataset containing:
    - Books3, PubMed, ArXiv, GitHub, StackExchange
    - OpenWebText, Wikipedia, YouTube, etc.
    - 22 different high-quality data sources
    """
    print(f"Loading The Pile validation dataset...")
    random.seed(seed)
    np.random.seed(seed)

    try:
        # The Pile validation set
        dataset = load_dataset("monology/pile-uncopyrighted", split="validation", streaming=True)

        texts = []
        print(f"Collecting {n_samples} samples from The Pile validation (seed={seed})...")

        for i, item in enumerate(tqdm(dataset, desc="Loading Pile", total=n_samples)):
            if len(texts) >= n_samples:
                break

            text = item['text']
            # Filter out very short texts
            if len(text.strip()) > 100:
                texts.append(text)

        print(f"Loaded {len(texts)} samples from The Pile")

        # Shuffle with fixed seed
        random.seed(seed)
        random.shuffle(texts)

        return texts[:n_samples]

    except Exception as e:
        print(f"Error loading The Pile: {e}")
        print("\nFalling back to alternative dataset: BookCorpus")
        return load_bookcorpus_validation(n_samples, seed)


def load_bookcorpus_validation(n_samples=2000, seed=42):
    """Fallback: Load BookCorpus if The Pile is unavailable."""
    print(f"Loading BookCorpus validation dataset...")
    random.seed(seed)
    np.random.seed(seed)

    try:
        # BookCorpus
        dataset = load_dataset("bookcorpus", split="train", streaming=True)

        texts = []
        print(f"Collecting {n_samples} samples from BookCorpus (seed={seed})...")

        # Skip first 100k samples (use later ones for validation)
        skip_count = 100000
        sample_count = 0

        for i, item in enumerate(tqdm(dataset, desc="Loading BookCorpus")):
            if i < skip_count:
                continue

            if sample_count >= n_samples:
                break

            text = item['text']
            if len(text.strip()) > 100:
                texts.append(text)
                sample_count += 1

        print(f"Loaded {len(texts)} samples from BookCorpus")

        random.seed(seed)
        random.shuffle(texts)

        return texts[:n_samples]

    except Exception as e:
        print(f"Error loading BookCorpus: {e}")
        print("\nCannot load alternative dataset. Please check your internet connection.")
        return None


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
            if failed <= 3:
                continue
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
        import traceback
        traceback.print_exc()
        return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = 2000
    seed = 42

    print("="*80)
    print("CROSS-DATASET TEST: V1 GWH-PRAQ vs GW-AWQ on The Pile")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: The Pile validation (diverse, 22 sources)")
    print(f"Samples: {n_samples}")
    print(f"Seed: {seed}")
    print("="*80)

    # Load The Pile validation
    print("\nLoading evaluation data...")
    eval_texts = load_pile_validation(n_samples=n_samples, seed=seed)

    if eval_texts is None:
        print("❌ Failed to load dataset. Exiting.")
        return

    # Model paths
    v1_path = "./quantized_models/minicpm_gwh_praq"
    awq_path = "./quantized_models/minicpm_gw_awq"

    # Evaluate both models
    results = {}

    results['V1-GWH-PRAQ'] = evaluate_model(
        v1_path,
        "V1 GWH-PRAQ (original)",
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
    print("RESULTS - The Pile Validation")
    print("="*80)

    if results['V1-GWH-PRAQ'] and results['GW-AWQ']:
        v1_ppl = results['V1-GWH-PRAQ']['perplexity']
        awq_ppl = results['GW-AWQ']['perplexity']

        delta = v1_ppl - awq_ppl
        delta_pct = (delta / awq_ppl) * 100

        print(f"\n{'Model':<30} {'Perplexity':<12} {'Avg Loss':<12}")
        print("-" * 60)
        print(f"{'GW-AWQ (baseline)':<30} {awq_ppl:<12.4f} {results['GW-AWQ']['avg_loss']:<12.4f}")
        print(f"{'V1 GWH-PRAQ':<30} {v1_ppl:<12.4f} {results['V1-GWH-PRAQ']['avg_loss']:<12.4f}")
        print("-" * 60)
        print(f"{'Delta (V1 - AWQ)':<30} {delta:+.4f} ({delta_pct:+.3f}%)")

        print("\n" + "="*80)
        print("ANALYSIS - The Pile")
        print("="*80)

        if abs(delta_pct) < 0.05:
            print(f"\n🤝 TIED (< 0.05% difference)")
            print(f"   Both methods perform equally on The Pile")
            winner_pile = "Tie"
        elif delta < 0:
            print(f"\n✅ V1 GWH-PRAQ WINS by {abs(delta_pct):.3f}%!")
            print(f"   V1 beats AWQ on The Pile (diverse data)")
            winner_pile = "V1"
        else:
            print(f"\n❌ GW-AWQ WINS by {delta_pct:.3f}%")
            print(f"   Pure AWQ better on The Pile")
            winner_pile = "AWQ"

        # Compare with previous results
        print("\n" + "="*80)
        print("THREE-DATASET COMPARISON")
        print("="*80)

        # Previous results
        wt2_v1 = 16.5159
        wt2_awq = 16.5513
        wt2_delta_pct = ((wt2_v1 - wt2_awq) / wt2_awq) * 100

        c4_v1 = 13.7577
        c4_awq = 13.7670
        c4_delta_pct = ((c4_v1 - c4_awq) / c4_awq) * 100

        print(f"\n{'Dataset':<20} {'V1 GWH-PRAQ':<15} {'GW-AWQ':<15} {'Delta':<15} {'Winner':<15}")
        print("-" * 90)
        print(f"{'WikiText-2':<20} {wt2_v1:<15.4f} {wt2_awq:<15.4f} {wt2_delta_pct:>+14.3f}%  {'V1':<15}")
        print(f"{'C4':<20} {c4_v1:<15.4f} {c4_awq:<15.4f} {c4_delta_pct:>+14.3f}%  {'V1':<15}")
        print(f"{'The Pile':<20} {v1_ppl:<15.4f} {awq_ppl:<15.4f} {delta_pct:>+14.3f}%  {winner_pile:<15}")

        print("\n" + "="*80)
        print("CONSISTENCY ANALYSIS")
        print("="*80)

        # Count wins
        v1_wins = 0
        awq_wins = 0
        ties = 0

        if wt2_delta_pct < -0.05:
            v1_wins += 1
        elif wt2_delta_pct > 0.05:
            awq_wins += 1
        else:
            ties += 1

        if c4_delta_pct < -0.05:
            v1_wins += 1
        elif c4_delta_pct > 0.05:
            awq_wins += 1
        else:
            ties += 1

        if delta_pct < -0.05:
            v1_wins += 1
        elif delta_pct > 0.05:
            awq_wins += 1
        else:
            ties += 1

        print(f"\nWins across 3 datasets:")
        print(f"  V1 GWH-PRAQ: {v1_wins}/3")
        print(f"  GW-AWQ:      {awq_wins}/3")
        print(f"  Ties:        {ties}/3")

        # Overall winner
        print("\n" + "="*80)
        print("OVERALL VERDICT")
        print("="*80)

        if v1_wins > awq_wins:
            print(f"\n🏆 V1 GWH-PRAQ is the OVERALL WINNER!")
            print(f"   Wins: {v1_wins}/3 datasets")
            print(f"   Consistent performance across diverse data")
            print(f"\n   ✅ Deploy: V1 GWH-PRAQ (gwh_praq.py)")
        elif awq_wins > v1_wins:
            print(f"\n🏆 GW-AWQ is the OVERALL WINNER!")
            print(f"   Wins: {awq_wins}/3 datasets")
            print(f"   Pure AWQ more robust across diverse data")
            print(f"\n   ✅ Deploy: GW-AWQ")
        else:
            print(f"\n🤝 TIED OVERALL")
            print(f"   Both methods equally strong")
            print(f"   Choose based on preference")

        # Generalization insight
        print("\n" + "="*80)
        print("GENERALIZATION INSIGHTS")
        print("="*80)

        avg_v1 = (wt2_v1 + c4_v1 + v1_ppl) / 3
        avg_awq = (wt2_awq + c4_awq + awq_ppl) / 3
        avg_delta = avg_v1 - avg_awq
        avg_delta_pct = (avg_delta / avg_awq) * 100

        print(f"\nAverage Perplexity across 3 datasets:")
        print(f"  V1 GWH-PRAQ: {avg_v1:.4f}")
        print(f"  GW-AWQ:      {avg_awq:.4f}")
        print(f"  Difference:  {avg_delta:+.4f} ({avg_delta_pct:+.3f}%)")

        if abs(avg_delta_pct) < 0.1:
            print(f"\n  → Methods are essentially equivalent")
            print(f"  → Choose V1 for slight edge + hybrid benefits")
        elif avg_delta < 0:
            print(f"\n  → V1 better on average by {abs(avg_delta_pct):.3f}%")
            print(f"  → V1 generalizes better across datasets")
        else:
            print(f"\n  → AWQ better on average by {avg_delta_pct:.3f}%")
            print(f"  → AWQ generalizes better across datasets")

        # Dataset characteristics
        print("\n" + "="*80)
        print("DATASET CHARACTERISTICS")
        print("="*80)
        print("\nWikiText-2:")
        print("  - Source: Wikipedia articles")
        print("  - Style: Formal, encyclopedic")
        print("  - Domain: General knowledge")

        print("\nC4:")
        print("  - Source: Web crawl")
        print("  - Style: Diverse, noisy")
        print("  - Domain: Mixed (blogs, forums, news, etc.)")

        print("\nThe Pile:")
        print("  - Source: 22 high-quality datasets")
        print("  - Style: Academic, technical, creative")
        print("  - Domain: Books, papers, code, conversations")

        print("\nConclusion:")
        print("  Testing on 3 diverse datasets provides robust validation")
        print("  of generalization capability across domains and styles.")

    else:
        print("\n❌ Could not compare - one or both models missing")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
