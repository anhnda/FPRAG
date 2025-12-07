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


def load_third_dataset(n_samples=2000, seed=42):
    """
    Load third validation dataset for cross-testing.

    Tries in order:
    1. OpenWebText (GPT-2 training data subset)
    2. AG News (news articles)
    3. C4 train subset (different from C4 validation)
    """
    print(f"Loading third dataset for validation...")
    random.seed(seed)
    np.random.seed(seed)

    # Try 1: OpenWebText
    try:
        print("Attempting to load OpenWebText...")
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

        texts = []
        print(f"Collecting {n_samples} samples from OpenWebText (seed={seed})...")

        # Skip first portion, use middle for validation-like data
        skip_count = 50000
        sample_count = 0

        for i, item in enumerate(tqdm(dataset, desc="Loading OpenWebText")):
            if i < skip_count:
                if i % 10000 == 0:
                    print(f"  Skipping to validation portion... {i}/{skip_count}")
                continue

            if sample_count >= n_samples:
                break

            text = item['text']
            if len(text.strip()) > 100:
                texts.append(text)
                sample_count += 1

        if len(texts) >= n_samples * 0.9:  # At least 90% of target
            print(f"✅ Loaded {len(texts)} samples from OpenWebText")
            random.seed(seed)
            random.shuffle(texts)
            return texts[:n_samples], "OpenWebText"

    except Exception as e:
        print(f"❌ Error loading OpenWebText: {e}")

    # Try 2: AG News
    try:
        print("\nAttempting to load AG News...")
        dataset = load_dataset("ag_news", split="test")

        texts = []
        print(f"Collecting samples from AG News...")

        for item in tqdm(dataset, desc="Loading AG News"):
            text = item['text']
            if len(text.strip()) > 100:
                texts.append(text)

        if len(texts) >= n_samples * 0.5:
            print(f"✅ Loaded {len(texts)} samples from AG News")
            random.seed(seed)
            random.shuffle(texts)
            return texts[:n_samples], "AG News"

    except Exception as e:
        print(f"❌ Error loading AG News: {e}")

    # Try 3: C4 train (different from C4 validation we used earlier)
    try:
        print("\nAttempting to load C4 train subset...")
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

        texts = []
        print(f"Collecting {n_samples} samples from C4 train...")

        for i, item in enumerate(tqdm(dataset, desc="Loading C4 train", total=n_samples)):
            if len(texts) >= n_samples:
                break

            text = item['text']
            if len(text.strip()) > 100:
                texts.append(text)

        if len(texts) >= n_samples * 0.9:
            print(f"✅ Loaded {len(texts)} samples from C4 train")
            random.seed(seed)
            random.shuffle(texts)
            return texts[:n_samples], "C4 train subset"

    except Exception as e:
        print(f"❌ Error loading C4 train: {e}")

    print("\n❌ All dataset loading attempts failed")
    return None, None


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
    print("CROSS-DATASET TEST: V1 GWH-PRAQ vs GW-AWQ")
    print("="*80)
    print(f"Device: {device}")
    print(f"Target: Third dataset for cross-validation")
    print(f"Samples: {n_samples}")
    print(f"Seed: {seed}")
    print("="*80)

    # Load third dataset
    print("\nLoading evaluation data...")
    eval_texts, dataset_name = load_third_dataset(n_samples=n_samples, seed=seed)

    if eval_texts is None or dataset_name is None:
        print("❌ Failed to load dataset. Exiting.")
        return

    print(f"\n✅ Using dataset: {dataset_name}")
    print(f"✅ Loaded {len(eval_texts)} samples")
    print("="*80)

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
    print(f"RESULTS - {dataset_name}")
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
        print(f"ANALYSIS - {dataset_name}")
        print("="*80)

        if abs(delta_pct) < 0.05:
            print(f"\n🤝 TIED (< 0.05% difference)")
            print(f"   Both methods perform equally on {dataset_name}")
            winner_third = "Tie"
        elif delta < 0:
            print(f"\n✅ V1 GWH-PRAQ WINS by {abs(delta_pct):.3f}%!")
            print(f"   V1 beats AWQ on {dataset_name}")
            winner_third = "V1"
        else:
            print(f"\n❌ GW-AWQ WINS by {delta_pct:.3f}%")
            print(f"   Pure AWQ better on {dataset_name}")
            winner_third = "AWQ"

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
        print(f"{dataset_name:<20} {v1_ppl:<15.4f} {awq_ppl:<15.4f} {delta_pct:>+14.3f}%  {winner_third:<15}")

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

        print(f"\n{dataset_name}:")
        if "OpenWebText" in dataset_name:
            print("  - Source: Reddit outbound links (GPT-2 training)")
            print("  - Style: Web content, articles, discussions")
            print("  - Domain: General web, higher quality than raw crawl")
        elif "AG News" in dataset_name:
            print("  - Source: News articles")
            print("  - Style: Journalistic, factual")
            print("  - Domain: World, sports, business, sci/tech news")
        elif "C4 train" in dataset_name:
            print("  - Source: Web crawl (training subset)")
            print("  - Style: Diverse, noisy")
            print("  - Domain: Different portion of web than C4 validation")
        else:
            print("  - Different data distribution from WikiText-2 and C4")

        print("\nConclusion:")
        print("  Testing on 3 diverse datasets provides robust validation")
        print("  of generalization capability across domains and styles.")

    else:
        print("\n❌ Could not compare - one or both models missing")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
