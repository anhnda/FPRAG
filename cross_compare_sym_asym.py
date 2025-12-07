"""
Cross-Dataset Validation: Symmetric vs Asymmetric Quantization

Comprehensive evaluation across multiple datasets to determine whether
asymmetric quantization provides benefits over symmetric quantization.

Tests both methods:
1. GW-AWQ: Symmetric vs Asymmetric
2. GWH-PRAQ: Symmetric vs Asymmetric

Datasets tested:
1. WikiText-2 validation - In-distribution (Wikipedia, formal)
2. C4 validation - Cross-dataset (Web crawl, diverse)
3. AG News test - Cross-dataset (News, journalistic)

This provides robust validation across:
- Different domains (Wikipedia, web, news)
- Different styles (formal, casual, journalistic)
- Different data quality (clean, noisy, curated)

Author: Cross-validation for symmetric vs asymmetric quantization
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
import matplotlib.pyplot as plt
import seaborn as sns


class SymmetricAsymmetricValidator:
    """Comprehensive cross-dataset validation for symmetric vs asymmetric quantization."""

    def __init__(self, device="cuda", seed=42):
        self.device = device
        self.seed = seed
        self.results = {}

        print("="*80)
        print("CROSS-DATASET VALIDATION: Symmetric vs Asymmetric Quantization")
        print("="*80)
        print(f"Device: {device}")
        print(f"Random seed: {seed}")
        print("="*80)

    def load_wikitext2_validation(self, n_samples=2000):
        """Load WikiText-2 validation set."""
        print("\n[1/3] Loading WikiText-2 validation...")
        random.seed(self.seed)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        # Match final_cross_validation.py: keep longer texts (> 100 chars)
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]

        # Use random sampling with seed for reproducibility
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
            # Match final_cross_validation.py: keep longer texts (> 100 chars)
            if len(text.strip()) > 100:
                texts.append(text)

        # Shuffle with seed for reproducibility
        random.seed(self.seed)
        random.shuffle(texts)

        print(f"  ✅ Loaded {len(texts)} samples")
        return texts[:n_samples]

    def load_ag_news_test(self, n_samples=2000):
        """Load AG News test set."""
        print("\n[3/3] Loading AG News test...")
        random.seed(self.seed)

        dataset = load_dataset("ag_news", split="test")
        # Match final_cross_validation.py: keep longer texts (> 100 chars)
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]

        # Use random sampling with seed for reproducibility
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

                if not np.isnan(loss) and not np.isinf(loss):
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

            del model, tokenizer
            torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None

    def run_comprehensive_validation(self, models_dict, n_samples=2000):
        """
        Run validation on all datasets for all models.

        Args:
            models_dict: Dict of {model_name: model_path}
            n_samples: Number of samples per dataset
        """
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

        # Evaluate each model on each dataset
        for dataset_name, texts in datasets.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")

            for model_name, model_path in models_dict.items():
                result = self.evaluate_model_on_dataset(
                    model_path, model_name, texts, dataset_name
                )

                if result:
                    if dataset_name not in self.results:
                        self.results[dataset_name] = {}
                    self.results[dataset_name][model_name] = result

        return self.results

    def generate_comparison_table(self):
        """Generate unified comparison table showing all methods on all datasets."""
        print("\n" + "="*100)
        print("COMPREHENSIVE RESULTS: Perplexity Across All Datasets")
        print("="*100)

        # Prepare data structure
        methods = ['GW-AWQ (Sym)', 'GW-AWQ (Asym)', 'GWH-PRAQ (Sym)', 'GWH-PRAQ (Asym)']
        datasets = ['WikiText-2', 'C4', 'AG News']

        # Print unified table header
        print(f"\n{'Method':<20}", end='')
        for dataset in datasets:
            print(f"{dataset:>15}", end='')
        print(f"{'Average':>15}")
        print("-" * 100)

        # Collect results for analysis
        awq_results = []
        praq_results = []

        # Print each method's results
        for method in methods:
            print(f"{method:<20}", end='')
            perplexities = []

            for dataset in datasets:
                if dataset in self.results and method in self.results[dataset]:
                    ppl = self.results[dataset][method]['perplexity']
                    perplexities.append(ppl)
                    print(f"{ppl:>15.4f}", end='')
                else:
                    print(f"{'N/A':>15}", end='')

            # Calculate and print average
            if perplexities:
                avg_ppl = np.mean(perplexities)
                print(f"{avg_ppl:>15.4f}")
            else:
                print(f"{'N/A':>15}")

        print("="*100)

        # Prepare detailed comparison data for analysis
        for dataset_name in datasets:
            if dataset_name in self.results:
                # GW-AWQ comparison
                if 'GW-AWQ (Sym)' in self.results[dataset_name] and 'GW-AWQ (Asym)' in self.results[dataset_name]:
                    sym_ppl = self.results[dataset_name]['GW-AWQ (Sym)']['perplexity']
                    asym_ppl = self.results[dataset_name]['GW-AWQ (Asym)']['perplexity']
                    delta = asym_ppl - sym_ppl
                    delta_pct = (delta / sym_ppl) * 100
                    winner = "Sym" if delta > 0.05 else ("Asym" if delta < -0.05 else "Tie")

                    awq_results.append({
                        'dataset': dataset_name,
                        'method': 'GW-AWQ',
                        'sym_ppl': sym_ppl,
                        'asym_ppl': asym_ppl,
                        'delta': delta,
                        'delta_pct': delta_pct,
                        'winner': winner
                    })

                # GWH-PRAQ comparison
                if 'GWH-PRAQ (Sym)' in self.results[dataset_name] and 'GWH-PRAQ (Asym)' in self.results[dataset_name]:
                    sym_ppl = self.results[dataset_name]['GWH-PRAQ (Sym)']['perplexity']
                    asym_ppl = self.results[dataset_name]['GWH-PRAQ (Asym)']['perplexity']
                    delta = asym_ppl - sym_ppl
                    delta_pct = (delta / sym_ppl) * 100
                    winner = "Sym" if delta > 0.05 else ("Asym" if delta < -0.05 else "Tie")

                    praq_results.append({
                        'dataset': dataset_name,
                        'method': 'GWH-PRAQ',
                        'sym_ppl': sym_ppl,
                        'asym_ppl': asym_ppl,
                        'delta': delta,
                        'delta_pct': delta_pct,
                        'winner': winner
                    })

        # Print delta summary
        print("\nDelta Analysis (Asymmetric - Symmetric):")
        print("-" * 100)
        print(f"{'Method':<20}", end='')
        for dataset in datasets:
            print(f"{dataset:>15}", end='')
        print(f"{'Avg Delta':>15}")
        print("-" * 100)

        # GW-AWQ deltas
        if awq_results:
            print(f"{'GW-AWQ':<20}", end='')
            for dataset in datasets:
                result = next((r for r in awq_results if r['dataset'] == dataset), None)
                if result:
                    print(f"{result['delta_pct']:>+14.3f}%", end='')
                else:
                    print(f"{'N/A':>15}", end='')
            avg_delta = np.mean([r['delta_pct'] for r in awq_results])
            print(f"{avg_delta:>+14.3f}%")

        # GWH-PRAQ deltas
        if praq_results:
            print(f"{'GWH-PRAQ':<20}", end='')
            for dataset in datasets:
                result = next((r for r in praq_results if r['dataset'] == dataset), None)
                if result:
                    print(f"{result['delta_pct']:>+14.3f}%", end='')
                else:
                    print(f"{'N/A':>15}", end='')
            avg_delta = np.mean([r['delta_pct'] for r in praq_results])
            print(f"{avg_delta:>+14.3f}%")

        print("="*100)

        all_results = {
            'awq': awq_results,
            'praq': praq_results
        }

        return all_results

    def analyze_results(self, all_results):
        """Comprehensive analysis of results."""
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)

        analysis = {}

        # Analyze GW-AWQ
        if all_results['awq']:
            awq_sym_wins = sum(1 for r in all_results['awq'] if r['winner'] == 'Sym')
            awq_asym_wins = sum(1 for r in all_results['awq'] if r['winner'] == 'Asym')
            awq_ties = sum(1 for r in all_results['awq'] if r['winner'] == 'Tie')

            avg_awq_delta_pct = np.mean([r['delta_pct'] for r in all_results['awq']])

            print(f"\nGW-AWQ Win Count:")
            print(f"  Symmetric:  {awq_sym_wins}/{len(all_results['awq'])}")
            print(f"  Asymmetric: {awq_asym_wins}/{len(all_results['awq'])}")
            print(f"  Ties:       {awq_ties}/{len(all_results['awq'])}")
            print(f"  Avg Δ:      {avg_awq_delta_pct:+.3f}% (Asym vs Sym)")

            analysis['awq'] = {
                'sym_wins': awq_sym_wins,
                'asym_wins': awq_asym_wins,
                'ties': awq_ties,
                'avg_delta_pct': avg_awq_delta_pct,
                'winner': 'Symmetric' if awq_sym_wins > awq_asym_wins else ('Asymmetric' if awq_asym_wins > awq_sym_wins else 'Tie')
            }

        # Analyze GWH-PRAQ
        if all_results['praq']:
            praq_sym_wins = sum(1 for r in all_results['praq'] if r['winner'] == 'Sym')
            praq_asym_wins = sum(1 for r in all_results['praq'] if r['winner'] == 'Asym')
            praq_ties = sum(1 for r in all_results['praq'] if r['winner'] == 'Tie')

            avg_praq_delta_pct = np.mean([r['delta_pct'] for r in all_results['praq']])

            print(f"\nGWH-PRAQ Win Count:")
            print(f"  Symmetric:  {praq_sym_wins}/{len(all_results['praq'])}")
            print(f"  Asymmetric: {praq_asym_wins}/{len(all_results['praq'])}")
            print(f"  Ties:       {praq_ties}/{len(all_results['praq'])}")
            print(f"  Avg Δ:      {avg_praq_delta_pct:+.3f}% (Asym vs Sym)")

            analysis['praq'] = {
                'sym_wins': praq_sym_wins,
                'asym_wins': praq_asym_wins,
                'ties': praq_ties,
                'avg_delta_pct': avg_praq_delta_pct,
                'winner': 'Symmetric' if praq_sym_wins > praq_asym_wins else ('Asymmetric' if praq_asym_wins > praq_sym_wins else 'Tie')
            }

        # Overall verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        total_sym_wins = analysis.get('awq', {}).get('sym_wins', 0) + analysis.get('praq', {}).get('sym_wins', 0)
        total_asym_wins = analysis.get('awq', {}).get('asym_wins', 0) + analysis.get('praq', {}).get('asym_wins', 0)
        total_tests = len(all_results.get('awq', [])) + len(all_results.get('praq', []))

        if total_sym_wins > total_asym_wins:
            print(f"\n🏆 SYMMETRIC QUANTIZATION WINS!")
            print(f"   Symmetric wins: {total_sym_wins}/{total_tests} tests")
            print(f"   Asymmetric wins: {total_asym_wins}/{total_tests} tests")
            print(f"\n   ✅ RECOMMENDATION: Use Symmetric Quantization")
            print(f"      - Better or equal quality across datasets")
            print(f"      - Faster inference (no zero_point overhead)")
            print(f"      - Simpler implementation")
            overall_winner = "Symmetric"
        elif total_asym_wins > total_sym_wins:
            print(f"\n🏆 ASYMMETRIC QUANTIZATION WINS!")
            print(f"   Asymmetric wins: {total_asym_wins}/{total_tests} tests")
            print(f"   Symmetric wins: {total_sym_wins}/{total_tests} tests")
            print(f"\n   ✅ RECOMMENDATION: Use Asymmetric Quantization")
            print(f"      - Better quality across datasets")
            print(f"      - Worth the computational overhead")
            overall_winner = "Asymmetric"
        else:
            print(f"\n🤝 TIE - Both methods equally strong")
            print(f"   Symmetric wins: {total_sym_wins}/{total_tests}")
            print(f"   Asymmetric wins: {total_asym_wins}/{total_tests}")
            print(f"\n   ✅ RECOMMENDATION: Use Symmetric Quantization")
            print(f"      - Equal quality but faster inference")
            overall_winner = "Symmetric (tie)"

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

        analysis['overall'] = {
            'winner': overall_winner,
            'total_sym_wins': total_sym_wins,
            'total_asym_wins': total_asym_wins,
            'total_tests': total_tests
        }

        return analysis

    def save_results(self, all_results, analysis, output_dir="./cross_comparison_results"):
        """Save results to JSON and markdown files."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        filename = f"sym_vs_asym_validation_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output = {
            'timestamp': timestamp,
            'seed': self.seed,
            'device': self.device,
            'results': all_results,
            'analysis': analysis,
            'detailed_results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✅ Results saved to: {filepath}")

        # Save summary markdown
        readme_path = os.path.join(output_dir, "SYM_VS_ASYM_SUMMARY.md")
        with open(readme_path, 'w') as f:
            f.write("# Symmetric vs Asymmetric Quantization Validation\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Winner:** {analysis['overall']['winner']}\n\n")

            # GW-AWQ results
            if all_results['awq']:
                f.write("## GW-AWQ Results\n\n")
                f.write("| Dataset | Symmetric | Asymmetric | Delta | Winner |\n")
                f.write("|---------|-----------|------------|-------|--------|\n")
                for r in all_results['awq']:
                    f.write(f"| {r['dataset']} | {r['sym_ppl']:.4f} | {r['asym_ppl']:.4f} | {r['delta_pct']:+.3f}% | {r['winner']} |\n")
                f.write(f"\n**Winner:** {analysis['awq']['winner']}\n")
                f.write(f"**Win Count:** Sym={analysis['awq']['sym_wins']}/3, Asym={analysis['awq']['asym_wins']}/3\n\n")

            # GWH-PRAQ results
            if all_results['praq']:
                f.write("## GWH-PRAQ Results\n\n")
                f.write("| Dataset | Symmetric | Asymmetric | Delta | Winner |\n")
                f.write("|---------|-----------|------------|-------|--------|\n")
                for r in all_results['praq']:
                    f.write(f"| {r['dataset']} | {r['sym_ppl']:.4f} | {r['asym_ppl']:.4f} | {r['delta_pct']:+.3f}% | {r['winner']} |\n")
                f.write(f"\n**Winner:** {analysis['praq']['winner']}\n")
                f.write(f"**Win Count:** Sym={analysis['praq']['sym_wins']}/3, Asym={analysis['praq']['asym_wins']}/3\n\n")

            # Overall recommendation
            f.write("## Overall Recommendation\n\n")
            f.write(f"**Use:** {analysis['overall']['winner']}\n\n")
            f.write(f"**Total Wins:** Symmetric={analysis['overall']['total_sym_wins']}/{analysis['overall']['total_tests']}, ")
            f.write(f"Asymmetric={analysis['overall']['total_asym_wins']}/{analysis['overall']['total_tests']}\n")

        print(f"✅ Summary saved to: {readme_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-dataset validation: Symmetric vs Asymmetric",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=[
            "GW-AWQ (Sym):./quantized_models/minicpm_gw_awq",
            "GW-AWQ (Asym):./quantized_models/minicpm_gw_awq_asym",
            "GWH-PRAQ (Sym):./quantized_models/minicpm_gwh_praq",
            "GWH-PRAQ (Asym):./quantized_models/minicpm_gwh_praq_asym"
        ],
        help="Models to compare in format 'Name:Path'"
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

    # Parse models
    models_dict = {}
    for model_spec in args.models:
        name, path = model_spec.split(':', 1)
        models_dict[name] = path

    # Initialize validator
    validator = SymmetricAsymmetricValidator(device=device, seed=args.seed)

    # Run comprehensive validation
    validator.run_comprehensive_validation(
        models_dict=models_dict,
        n_samples=args.n_samples
    )

    # Generate comparison table
    all_results = validator.generate_comparison_table()

    # Analyze results
    analysis = validator.analyze_results(all_results)

    # Save results if requested
    if args.save_results:
        validator.save_results(all_results, analysis)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\n🏆 Winner: {analysis['overall']['winner']}")
    print(f"📊 Total tests: {analysis['overall']['total_tests']}")
    print(f"✅ Symmetric wins: {analysis['overall']['total_sym_wins']}")
    print(f"✅ Asymmetric wins: {analysis['overall']['total_asym_wins']}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
