"""
Compare GW-AWQ vs GWH-PRAQ with L2 Salience (Asymmetric Quantization)

Tests on 3 datasets: WikiText-2, C4, AG News
Evaluates perplexity to determine which method performs better with L2 norm.

Expected Hypothesis:
- L2 salience (E[X²]) should outperform L1 (E[|X|])
- Both methods benefit from L2's better MSE alignment
- Question: Does PRAQ's hybrid approach still help with L2, or does pure AWQ win?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random


class DatasetEvaluator:
    """Unified dataset evaluator for perplexity measurement."""

    def __init__(self, model_path, model_name, device="cuda", seed=42):
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.seed = seed

        print(f"\n{'='*80}")
        print(f"Loading: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*80}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()

    def load_dataset_texts(self, dataset_name, split, n_samples=500):
        """Load dataset with consistent preprocessing."""
        print(f"\nLoading {dataset_name} ({split})...")

        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
        elif dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
            texts = []
            for item in dataset:
                if len(item['text'].strip()) > 100:
                    texts.append(item['text'])
                if len(texts) >= n_samples * 2:
                    break
        elif dataset_name == "ag_news":
            dataset = load_dataset("ag_news", split=split)
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Random sampling with seed
        random.seed(self.seed)
        if len(texts) > n_samples:
            texts = random.sample(texts, n_samples)

        print(f"  Loaded {len(texts)} texts (filtered: >100 chars, random sampled)")
        return texts

    @torch.no_grad()
    def evaluate_perplexity(self, texts, max_samples=500):
        """Evaluate perplexity on texts."""
        total_loss = 0.0
        total_tokens = 0
        successful = 0

        for text in tqdm(texts[:max_samples], desc=f"Evaluating {self.model_name}"):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs, labels=inputs['input_ids'], use_cache=False)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()
                successful += 1

            except Exception as e:
                continue

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        print(f"  ✓ {self.model_name}: PPL = {perplexity:.4f} ({successful}/{max_samples} samples)")
        return perplexity

    def cleanup(self):
        """Clean up model from memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def run_cross_validation(model_configs, datasets_config, device="cuda"):
    """
    Run cross-validation across multiple models and datasets.

    Args:
        model_configs: List of (model_path, model_name) tuples
        datasets_config: List of (dataset_name, split, n_samples) tuples
    """
    results = {name: {} for _, name in model_configs}

    for dataset_name, split, n_samples in datasets_config:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()} ({split})")
        print(f"{'='*80}")

        # Load dataset once
        evaluator_temp = DatasetEvaluator(
            model_configs[0][0], "temp", device=device
        )
        texts = evaluator_temp.load_dataset_texts(dataset_name, split, n_samples)
        evaluator_temp.cleanup()

        # Evaluate each model
        for model_path, model_name in model_configs:
            evaluator = DatasetEvaluator(model_path, model_name, device=device)
            perplexity = evaluator.evaluate_perplexity(texts, max_samples=n_samples)
            results[model_name][dataset_name] = perplexity
            evaluator.cleanup()

    return results


def print_results_table(results, datasets):
    """Print results in a clean table format."""
    print("\n" + "="*100)
    print("FINAL RESULTS - L2 Salience Comparison")
    print("="*100)

    # Header
    dataset_names = [d[0] for d in datasets]
    header = f"{'Method':<35}"
    for name in dataset_names:
        header += f"{name:>15}"
    header += f"{'Average':>15}"
    print(header)
    print("-"*100)

    # Results
    for method_name, dataset_results in results.items():
        row = f"{method_name:<35}"
        perplexities = []
        for dataset_name in dataset_names:
            ppl = dataset_results.get(dataset_name, float('inf'))
            perplexities.append(ppl)
            row += f"{ppl:>15.4f}"

        avg_ppl = np.mean(perplexities)
        row += f"{avg_ppl:>15.4f}"

        # Mark winner with ✅
        if avg_ppl == min(np.mean([results[m].get(d, float('inf')) for d in dataset_names]) for m in results.keys()):
            row += "  ✅ WINNER"

        print(row)

    print("="*100)

    # Analysis
    print("\nANALYSIS:")
    all_methods = list(results.keys())
    if len(all_methods) == 2:
        method1, method2 = all_methods
        avg1 = np.mean([results[method1][d] for d in dataset_names])
        avg2 = np.mean([results[method2][d] for d in dataset_names])

        if avg1 < avg2:
            improvement = ((avg2 - avg1) / avg2) * 100
            print(f"  → {method1} WINS by {improvement:.2f}% lower perplexity")
        else:
            improvement = ((avg1 - avg2) / avg1) * 100
            print(f"  → {method2} WINS by {improvement:.2f}% lower perplexity")

    print("="*100)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*100)
    print("L2 SALIENCE COMPARISON: GW-AWQ vs GWH-PRAQ (Asymmetric Quantization)")
    print("="*100)
    print(f"Device: {device}")
    print("Hypothesis: L2 salience (E[X²]) should outperform L1 (E[|X|])")
    print("Question: Does PRAQ's hybrid approach still help with L2?")
    print("="*100)

    # Model configurations
    model_configs = [
        ("./quantized_models/minicpm_gw_awq_asym_l2", "GW-AWQ (Asym+L2)"),
        ("./quantized_models/minicpm_gwh_praq_asym_l2", "GWH-PRAQ (Asym+L2)"),
    ]

    # Dataset configurations: (name, split, n_samples)
    datasets_config = [
        ("wikitext", "validation", 500),
        ("c4", "validation", 500),
        ("ag_news", "test", 500),
    ]

    # Run evaluation
    results = run_cross_validation(model_configs, datasets_config, device=device)

    # Print results
    print_results_table(results, datasets_config)

    print("\nKEY FINDINGS:")
    print("1. L2 salience emphasizes high-energy channels (spikes)")
    print("2. Better MSE alignment: E[(δW×X)²] ∝ E[X²]")
    print("3. Compare with L1 results to see improvement")
    print("\nNext: Compare L2 vs L1 versions to measure improvement!")


if __name__ == "__main__":
    main()
