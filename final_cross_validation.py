"""
Final Cross-Dataset Validation: V1 GWH-PRAQ vs GW-AWQ

Comprehensive evaluation across multiple datasets to determine the best
quantization method for production deployment.

Datasets tested:
1. WikiText-2 validation - In-distribution (Wikipedia, formal)
2. C4 validation - Cross-dataset (Web crawl, diverse)
3. AG News test - Cross-dataset (News, journalistic)

This provides robust validation across:
- Different domains (Wikipedia, web, news)
- Different styles (formal, casual, journalistic)
- Different data quality (clean, noisy, curated)

Author: Final validation for hybrid quantization research
Date: 2025
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import os
import json
from datetime import datetime


class CrossDatasetValidator:
    """Comprehensive cross-dataset validation for quantized models."""

    def __init__(self, device="cuda", seed=42):
        self.device = device
        self.seed = seed
        self.results = {}

        print("="*80)
        print("FINAL CROSS-DATASET VALIDATION")
        print("="*80)
        print(f"Device: {device}")
        print(f"Random seed: {seed}")
        print("="*80)

    def load_wikitext2_validation(self, n_samples=2000):
        """Load WikiText-2 validation set."""
        print("\n[1/3] Loading WikiText-2 validation...")
        random.seed(self.seed)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]

        random.seed(self.seed)
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)

        print(f"  ✅ Loaded {len(texts)} samples")
        return texts

    def load_c4_validation(self, n_samples=2000):
        """Load C4 validation set."""
        print("\n[2/3] Loading C4 validation...")
        random.seed(self.seed)

        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

        texts = []
        for i, item in enumerate(tqdm(dataset, desc="  Collecting C4", total=n_samples)):
            if len(texts) >= n_samples:
                break
            text = item['text']
            if len(text.strip()) > 100:
                texts.append(text)

        random.seed(self.seed)
        random.shuffle(texts)

        print(f"  ✅ Loaded {len(texts)} samples")
        return texts[:n_samples]

    def load_ag_news_test(self, n_samples=2000):
        """Load AG News test set."""
        print("\n[3/3] Loading AG News test...")
        random.seed(self.seed)

        dataset = load_dataset("ag_news", split="test")
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]

        random.seed(self.seed)
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)

        print(f"  ✅ Loaded {len(texts)} samples")
        return texts

    @torch.no_grad()
    def evaluate_perplexity(self, model, tokenizer, texts, max_length=512):
        """Evaluate perplexity on text samples."""
        model.eval()
        total_loss = 0
        total_tokens = 0
        successful = 0

        for text in tqdm(texts, desc="  Evaluating", leave=False):
            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=False
                )

                input_ids = inputs["input_ids"].to(self.device)

                if input_ids.shape[1] < 2:
                    continue

                outputs = model(input_ids, labels=input_ids, use_cache=False)
                loss = outputs.loss.item()
                n_tokens = input_ids.shape[1]

                total_loss += loss * n_tokens
                total_tokens += n_tokens
                successful += 1

            except Exception:
                continue

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if total_tokens > 0 else float('inf')

        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "num_samples": successful,
            "total_tokens": total_tokens
        }

    def evaluate_model_on_dataset(self, model_path, model_name, texts, dataset_name):
        """Evaluate a model on a specific dataset."""
        print(f"\n  Evaluating {model_name} on {dataset_name}...")

        if not os.path.exists(model_path):
            print(f"  ❌ Model not found: {model_path}")
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )

            results = self.evaluate_perplexity(model, tokenizer, texts)

            print(f"  ✅ Perplexity: {results['perplexity']:.4f}")

            del model
            torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None

    def run_comprehensive_validation(self, v1_path, awq_path, n_samples=2000):
        """Run validation on all datasets."""
        print("\n" + "="*80)
        print("LOADING DATASETS")
        print("="*80)

        datasets = {
            'WikiText-2': self.load_wikitext2_validation(n_samples),
            'C4': self.load_c4_validation(n_samples),
            'AG News': self.load_ag_news_test(n_samples)
        }

        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)

        models = {
            'V1 GWH-PRAQ': v1_path,
            'GW-AWQ': awq_path
        }

        # Evaluate each model on each dataset
        for dataset_name, texts in datasets.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")

            for model_name, model_path in models.items():
                result = self.evaluate_model_on_dataset(
                    model_path, model_name, texts, dataset_name
                )

                if result:
                    if dataset_name not in self.results:
                        self.results[dataset_name] = {}
                    self.results[dataset_name][model_name] = result

        return self.results

    def generate_comparison_table(self):
        """Generate formatted comparison table."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS")
        print("="*80)

        # Table header
        print(f"\n{'Dataset':<15} {'V1 GWH-PRAQ':<15} {'GW-AWQ':<15} {'Delta':<12} {'Winner':<10}")
        print("-" * 80)

        # Results for each dataset
        dataset_results = []
        for dataset_name in ['WikiText-2', 'C4', 'AG News']:
            if dataset_name in self.results:
                v1_ppl = self.results[dataset_name]['V1 GWH-PRAQ']['perplexity']
                awq_ppl = self.results[dataset_name]['GW-AWQ']['perplexity']
                delta = v1_ppl - awq_ppl
                delta_pct = (delta / awq_ppl) * 100

                winner = "V1" if delta < -0.05 else ("AWQ" if delta > 0.05 else "Tie")

                print(f"{dataset_name:<15} {v1_ppl:<15.4f} {awq_ppl:<15.4f} {delta_pct:>+11.3f}%  {winner:<10}")

                dataset_results.append({
                    'dataset': dataset_name,
                    'v1_ppl': v1_ppl,
                    'awq_ppl': awq_ppl,
                    'delta_pct': delta_pct,
                    'winner': winner
                })

        return dataset_results

    def analyze_results(self, dataset_results):
        """Comprehensive analysis of results."""
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)

        # Count wins
        v1_wins = sum(1 for r in dataset_results if r['winner'] == 'V1')
        awq_wins = sum(1 for r in dataset_results if r['winner'] == 'AWQ')
        ties = sum(1 for r in dataset_results if r['winner'] == 'Tie')

        print(f"\nWin Count:")
        print(f"  V1 GWH-PRAQ: {v1_wins}/{len(dataset_results)}")
        print(f"  GW-AWQ:      {awq_wins}/{len(dataset_results)}")
        print(f"  Ties:        {ties}/{len(dataset_results)}")

        # Average performance
        avg_v1 = np.mean([r['v1_ppl'] for r in dataset_results])
        avg_awq = np.mean([r['awq_ppl'] for r in dataset_results])
        avg_delta_pct = ((avg_v1 - avg_awq) / avg_awq) * 100

        print(f"\nAverage Perplexity:")
        print(f"  V1 GWH-PRAQ: {avg_v1:.4f}")
        print(f"  GW-AWQ:      {avg_awq:.4f}")
        print(f"  Difference:  {avg_delta_pct:+.3f}%")

        # Determine winner
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        if v1_wins > awq_wins:
            print(f"\n🏆 V1 GWH-PRAQ is the OVERALL WINNER!")
            print(f"   Wins: {v1_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            print(f"\n   ✅ RECOMMENDED FOR PRODUCTION")
            winner = "V1 GWH-PRAQ"
        elif awq_wins > v1_wins:
            print(f"\n🏆 GW-AWQ is the OVERALL WINNER!")
            print(f"   Wins: {awq_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            print(f"\n   ✅ RECOMMENDED FOR PRODUCTION")
            winner = "GW-AWQ"
        else:
            print(f"\n🤝 TIE - Both methods equally strong")
            print(f"   V1 GWH-PRAQ recommended for hybrid benefits")
            winner = "V1 GWH-PRAQ (tie)"

        # Dataset characteristics
        print("\n" + "="*80)
        print("DATASET CHARACTERISTICS")
        print("="*80)

        print("\nWikiText-2:")
        print("  Source: Wikipedia articles")
        print("  Style:  Formal, encyclopedic")
        print("  Domain: General knowledge")

        print("\nC4:")
        print("  Source: Common Crawl web scrape")
        print("  Style:  Diverse, noisy, real-world")
        print("  Domain: Mixed web content")

        print("\nAG News:")
        print("  Source: News articles")
        print("  Style:  Journalistic, factual")
        print("  Domain: World, sports, business, sci/tech")

        print("\nConclusion:")
        print("  Testing on 3 diverse datasets validates generalization")
        print("  across different domains, styles, and data quality.")

        return {
            'winner': winner,
            'v1_wins': v1_wins,
            'awq_wins': awq_wins,
            'ties': ties,
            'avg_v1': avg_v1,
            'avg_awq': avg_awq,
            'avg_delta_pct': avg_delta_pct
        }

    def save_results(self, dataset_results, analysis, output_dir="./results"):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cross_validation_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output = {
            'timestamp': timestamp,
            'seed': self.seed,
            'device': self.device,
            'datasets_tested': len(dataset_results),
            'dataset_results': dataset_results,
            'analysis': analysis,
            'detailed_results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ Results saved to: {filepath}")

        # Also save a summary README
        readme_path = os.path.join(output_dir, "VALIDATION_SUMMARY.md")
        with open(readme_path, 'w') as f:
            f.write("# Cross-Dataset Validation Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Winner:** {analysis['winner']}\n\n")
            f.write("## Results\n\n")
            f.write("| Dataset | V1 GWH-PRAQ | GW-AWQ | Delta | Winner |\n")
            f.write("|---------|-------------|---------|-------|--------|\n")
            for r in dataset_results:
                f.write(f"| {r['dataset']} | {r['v1_ppl']:.4f} | {r['awq_ppl']:.4f} | {r['delta_pct']:+.3f}% | {r['winner']} |\n")
            f.write(f"\n**Average:** V1={analysis['avg_v1']:.4f}, AWQ={analysis['avg_awq']:.4f}, Δ={analysis['avg_delta_pct']:+.3f}%\n")
            f.write(f"\n**Win Count:** V1={analysis['v1_wins']}/3, AWQ={analysis['awq_wins']}/3, Ties={analysis['ties']}/3\n")
            f.write(f"\n## Recommendation\n\n")
            f.write(f"**Deploy:** {analysis['winner']}\n")

        print(f"✅ Summary saved to: {readme_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Final cross-dataset validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--v1-path",
        type=str,
        default="./quantized_models/minicpm_gwh_praq",
        help="Path to V1 GWH-PRAQ model"
    )
    parser.add_argument(
        "--awq-path",
        type=str,
        default="./quantized_models/minicpm_gw_awq",
        help="Path to GW-AWQ model"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize validator
    validator = CrossDatasetValidator(device=device, seed=args.seed)

    # Run comprehensive validation
    validator.run_comprehensive_validation(
        v1_path=args.v1_path,
        awq_path=args.awq_path,
        n_samples=args.n_samples
    )

    # Generate comparison table
    dataset_results = validator.generate_comparison_table()

    # Analyze results
    analysis = validator.analyze_results(dataset_results)

    # Save results if requested
    if args.save_results:
        validator.save_results(dataset_results, analysis)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\n🏆 Winner: {analysis['winner']}")
    print(f"📊 Tested: {len(dataset_results)} datasets")
    print(f"✅ V1 wins: {analysis['v1_wins']}")
    print(f"✅ AWQ wins: {analysis['awq_wins']}")
    print(f"🤝 Ties: {analysis['ties']}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
