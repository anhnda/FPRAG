"""
Head-to-Head Comparison: GWH-PRAQ-V2 vs GW-AWQ (Asymmetric)

This script compares the two best asymmetric quantization methods:
1. GWH-PRAQ-V2 (Asym) - Enhanced adaptive hybrid approach
2. GW-AWQ (Asym) - Current best baseline

Datasets tested:
1. WikiText-2 validation - In-distribution (Wikipedia)
2. C4 validation - Cross-dataset (Web)
3. AG News test - Cross-dataset (News)

Goal: Determine if V2 enhancements beat the current champion!
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


class HeadToHeadValidator:
    """Compare GWH-PRAQ-V2 vs GW-AWQ head-to-head."""

    def __init__(self, device="cuda", seed=42):
        self.device = device
        self.seed = seed
        self.results = {}

        print("="*80)
        print("HEAD-TO-HEAD: GWH-PRAQ-V2 (Asym) vs GW-AWQ (Asym)")
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

    def run_comparison(self, v2_path, awq_path, n_samples=2000):
        """Run head-to-head comparison."""
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
            'GWH-PRAQ-V2 (Asym)': v2_path,
            'GW-AWQ (Asym)': awq_path
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
        """Generate comparison table."""
        print("\n" + "="*100)
        print("HEAD-TO-HEAD RESULTS: Perplexity Across All Datasets")
        print("="*100)

        methods = ['GWH-PRAQ-V2 (Asym)', 'GW-AWQ (Asym)']
        datasets = ['WikiText-2', 'C4', 'AG News']

        # Print table header
        print(f"\n{'Method':<25}", end='')
        for dataset in datasets:
            print(f"{dataset:>18}", end='')
        print(f"{'Average':>18}")
        print("-" * 100)

        # Collect results
        comparison_data = {}

        # Print each method's results
        for method in methods:
            print(f"{method:<25}", end='')
            perplexities = []

            for dataset in datasets:
                if dataset in self.results and method in self.results[dataset]:
                    ppl = self.results[dataset][method]['perplexity']
                    perplexities.append(ppl)
                    print(f"{ppl:>18.4f}", end='')
                else:
                    print(f"{'N/A':>18}", end='')

            # Calculate and print average
            if perplexities:
                avg_ppl = np.mean(perplexities)
                print(f"{avg_ppl:>18.4f}")
                comparison_data[method] = {
                    'perplexities': perplexities,
                    'average': avg_ppl
                }
            else:
                print(f"{'N/A':>18}")

        print("="*100)

        # Print delta analysis
        if len(comparison_data) == 2:
            v2_data = comparison_data.get('GWH-PRAQ-V2 (Asym)')
            awq_data = comparison_data.get('GW-AWQ (Asym)')

            if v2_data and awq_data:
                print("\nDelta Analysis (V2 - AWQ):")
                print("-" * 100)
                print(f"{'Metric':<25}", end='')
                for dataset in datasets:
                    print(f"{dataset:>18}", end='')
                print(f"{'Average':>18}")
                print("-" * 100)

                print(f"{'Absolute Difference':<25}", end='')
                deltas = []
                for i, dataset in enumerate(datasets):
                    if i < len(v2_data['perplexities']) and i < len(awq_data['perplexities']):
                        delta = v2_data['perplexities'][i] - awq_data['perplexities'][i]
                        deltas.append(delta)
                        print(f"{delta:>+18.4f}", end='')
                    else:
                        print(f"{'N/A':>18}", end='')

                avg_delta = np.mean(deltas) if deltas else 0
                print(f"{avg_delta:>+18.4f}")

                print(f"{'Percentage Change':<25}", end='')
                pct_changes = []
                for i, dataset in enumerate(datasets):
                    if i < len(v2_data['perplexities']) and i < len(awq_data['perplexities']):
                        pct = ((v2_data['perplexities'][i] - awq_data['perplexities'][i]) /
                               awq_data['perplexities'][i]) * 100
                        pct_changes.append(pct)
                        print(f"{pct:>+17.3f}%", end='')
                    else:
                        print(f"{'N/A':>18}", end='')

                avg_pct = np.mean(pct_changes) if pct_changes else 0
                print(f"{avg_pct:>+17.3f}%")

                print("="*100)

                # Win/loss record
                v2_wins = sum(1 for d in deltas if d < -0.01)  # V2 wins if 0.01+ lower
                awq_wins = sum(1 for d in deltas if d > 0.01)  # AWQ wins if 0.01+ lower
                ties = len(deltas) - v2_wins - awq_wins

                return {
                    'v2_data': v2_data,
                    'awq_data': awq_data,
                    'deltas': deltas,
                    'pct_changes': pct_changes,
                    'avg_delta': avg_delta,
                    'avg_pct': avg_pct,
                    'v2_wins': v2_wins,
                    'awq_wins': awq_wins,
                    'ties': ties
                }

        return {}

    def analyze_results(self, comparison):
        """Analyze and print final verdict."""
        if not comparison:
            print("❌ Insufficient data for analysis")
            return

        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        v2_wins = comparison['v2_wins']
        awq_wins = comparison['awq_wins']
        ties = comparison['ties']
        avg_pct = comparison['avg_pct']

        print(f"\nWin Record:")
        print(f"  GWH-PRAQ-V2: {v2_wins}/3 datasets")
        print(f"  GW-AWQ:      {awq_wins}/3 datasets")
        print(f"  Ties:        {ties}/3 datasets")

        print(f"\nAverage Performance:")
        print(f"  V2 Average:  {comparison['v2_data']['average']:.4f}")
        print(f"  AWQ Average: {comparison['awq_data']['average']:.4f}")
        print(f"  Difference:  {avg_pct:+.3f}%")

        print("\n" + "="*80)

        if v2_wins > awq_wins:
            print("🏆 WINNER: GWH-PRAQ-V2 (Asym)")
            print(f"   V2 wins {v2_wins}/3 datasets")
            print(f"   Average improvement: {abs(avg_pct):.3f}%")
            print("\n   ✅ V2 ENHANCEMENTS SUCCESSFUL!")
            print("      Adaptive quantization beats pure asymmetric")
            winner = "GWH-PRAQ-V2 (Asym)"
        elif awq_wins > v2_wins:
            print("🏆 WINNER: GW-AWQ (Asym)")
            print(f"   AWQ wins {awq_wins}/3 datasets")
            print(f"   V2 is worse by {abs(avg_pct):.3f}%")
            print("\n   ❌ V2 enhancements did not improve performance")
            print("      Pure asymmetric AWQ remains champion")
            winner = "GW-AWQ (Asym)"
        else:
            if abs(avg_pct) < 0.1:
                print("🤝 STATISTICAL TIE")
                print(f"   Difference: {abs(avg_pct):.3f}% (negligible)")
                print("\n   Both methods are equivalent")
                print("   Recommendation: Use simpler GW-AWQ (Asym)")
                winner = "Tie (use GW-AWQ)"
            elif avg_pct < 0:
                print("🏆 WINNER: GWH-PRAQ-V2 (Asym)")
                print(f"   Better average: {abs(avg_pct):.3f}%")
                winner = "GWH-PRAQ-V2 (Asym)"
            else:
                print("🏆 WINNER: GW-AWQ (Asym)")
                print(f"   Better average: {abs(avg_pct):.3f}%")
                winner = "GW-AWQ (Asym)"

        print("="*80)

        return {
            'winner': winner,
            'v2_wins': v2_wins,
            'awq_wins': awq_wins,
            'ties': ties,
            'avg_pct_improvement': avg_pct
        }

    def save_results(self, comparison, analysis, output_dir="./v2_vs_awq_results"):
        """Save results to JSON and markdown."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        filename = f"v2_vs_awq_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output = {
            'timestamp': timestamp,
            'seed': self.seed,
            'comparison': {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                          for k, v in comparison.items()},
            'analysis': analysis,
            'detailed_results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✅ Results saved to: {filepath}")

        # Save markdown summary
        readme_path = os.path.join(output_dir, "V2_VS_AWQ_SUMMARY.md")
        with open(readme_path, 'w') as f:
            f.write("# Head-to-Head: GWH-PRAQ-V2 vs GW-AWQ\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Winner:** {analysis['winner']}\n\n")

            f.write("## Results\n\n")
            f.write("| Dataset | GWH-PRAQ-V2 | GW-AWQ | Delta | Winner |\n")
            f.write("|---------|------------|--------|-------|--------|\n")

            datasets = ['WikiText-2', 'C4', 'AG News']
            for i, dataset in enumerate(datasets):
                v2_ppl = comparison['v2_data']['perplexities'][i]
                awq_ppl = comparison['awq_data']['perplexities'][i]
                delta_pct = comparison['pct_changes'][i]
                winner_mark = "V2" if delta_pct < -0.01 else ("AWQ" if delta_pct > 0.01 else "Tie")
                f.write(f"| {dataset} | {v2_ppl:.4f} | {awq_ppl:.4f} | {delta_pct:+.3f}% | {winner_mark} |\n")

            f.write(f"\n**Average:** V2={comparison['v2_data']['average']:.4f}, ")
            f.write(f"AWQ={comparison['awq_data']['average']:.4f}, ")
            f.write(f"Δ={comparison['avg_pct']:+.3f}%\n\n")

            f.write(f"**Win Count:** V2={analysis['v2_wins']}/3, AWQ={analysis['awq_wins']}/3, Ties={analysis['ties']}/3\n\n")

            f.write("## Recommendation\n\n")
            f.write(f"**Deploy:** {analysis['winner']}\n")

        print(f"✅ Summary saved to: {readme_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Head-to-head: GWH-PRAQ-V2 vs GW-AWQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--v2-path",
        type=str,
        default="./quantized_models/minicpm_gwh_praq_asym_v2",
        help="Path to GWH-PRAQ-V2 model"
    )
    parser.add_argument(
        "--awq-path",
        type=str,
        default="./quantized_models/minicpm_gw_awq_asym",
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
    validator = HeadToHeadValidator(device=device, seed=args.seed)

    # Run comparison
    validator.run_comparison(
        v2_path=args.v2_path,
        awq_path=args.awq_path,
        n_samples=args.n_samples
    )

    # Generate comparison table
    comparison = validator.generate_comparison_table()

    # Analyze results
    analysis = validator.analyze_results(comparison)

    # Save results if requested
    if args.save_results:
        validator.save_results(comparison, analysis)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    if analysis:
        print(f"\n🏆 Winner: {analysis['winner']}")
        print(f"📊 V2 wins: {analysis['v2_wins']}/3")
        print(f"📊 AWQ wins: {analysis['awq_wins']}/3")
        print(f"📈 Avg change: {analysis['avg_pct_improvement']:+.3f}%")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
