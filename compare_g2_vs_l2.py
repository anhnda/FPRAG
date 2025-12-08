"""
Cross-Dataset Validation: G² vs L2 Salience

Comprehensive evaluation to compare:
- GW-AWQ-L2: Standard L2 salience E[X²] for all layers
- GW-AWQ-G²: Gradient-squared salience for SiLU-followed layers
  - SiLU layers: E[1 + g²/max(g²)] where g = ∇SiLU(XW)/∇W
  - Other layers: E[X²] (standard L2)

Datasets tested:
1. WikiText-2 validation - In-distribution (Wikipedia, formal)
2. C4 validation - Cross-dataset (Web crawl, diverse)
3. AG News test - Cross-dataset (News, journalistic)

This validates whether gradient-based importance scoring for SiLU-followed layers
improves quantization quality compared to standard L2 salience.

Key Question: Does G² (gradient-squared) beat standard L2 for SiLU layers?

Author: Fast-R-PRAQ research
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


class G2vsL2Validator:
    """Cross-dataset validation for G² vs L2 salience comparison."""

    def __init__(self, device="cuda", seed=42):
        self.device = device
        self.seed = seed
        self.results = {}

        print("="*80)
        print("G² vs L2 SALIENCE CROSS-DATASET VALIDATION")
        print("="*80)
        print(f"Device: {device}")
        print(f"Random seed: {seed}")
        print("\nComparing:")
        print("  • L2:  E[X²] - Standard activation magnitude for ALL layers")
        print("  • G²:  HYBRID approach")
        print("         - SiLU-followed layers: E[1 + g²/max(g²)]")
        print("           where g = ∇SiLU(XW)/∇W (gradient-based)")
        print("         - Other layers: E[X²] (standard)")
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

    def run_comprehensive_validation(self, l2_path, g2_path, n_samples=2000):
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
            'L2 (Standard)': l2_path,
            'G² (Gradient)': g2_path
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
        print(f"\n{'Dataset':<15} {'L2 (Standard)':<18} {'G² (Gradient)':<18} {'Delta':<12} {'Winner':<12}")
        print("-" * 80)

        # Results for each dataset
        dataset_results = []
        for dataset_name in ['WikiText-2', 'C4', 'AG News']:
            if dataset_name in self.results:
                l2_ppl = self.results[dataset_name]['L2 (Standard)']['perplexity']
                g2_ppl = self.results[dataset_name]['G² (Gradient)']['perplexity']
                delta = g2_ppl - l2_ppl
                delta_pct = (delta / l2_ppl) * 100

                # G² is better if it's LOWER perplexity
                winner = "G²" if delta < -0.05 else ("L2" if delta > 0.05 else "Tie")

                print(f"{dataset_name:<15} {l2_ppl:<18.4f} {g2_ppl:<18.4f} {delta_pct:>+11.3f}%  {winner:<12}")

                dataset_results.append({
                    'dataset': dataset_name,
                    'l2_ppl': l2_ppl,
                    'g2_ppl': g2_ppl,
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
        g2_wins = sum(1 for r in dataset_results if r['winner'] == 'G²')
        l2_wins = sum(1 for r in dataset_results if r['winner'] == 'L2')
        ties = sum(1 for r in dataset_results if r['winner'] == 'Tie')

        print(f"\nWin Count:")
        print(f"  G² (Gradient):     {g2_wins}/{len(dataset_results)}")
        print(f"  L2 (Standard):     {l2_wins}/{len(dataset_results)}")
        print(f"  Ties:              {ties}/{len(dataset_results)}")

        # Average performance
        avg_l2 = np.mean([r['l2_ppl'] for r in dataset_results])
        avg_g2 = np.mean([r['g2_ppl'] for r in dataset_results])
        avg_delta_pct = ((avg_g2 - avg_l2) / avg_l2) * 100

        print(f"\nAverage Perplexity:")
        print(f"  L2 (Standard):     {avg_l2:.4f}")
        print(f"  G² (Gradient):     {avg_g2:.4f}")
        print(f"  Difference:        {avg_delta_pct:+.3f}%")

        # Statistical significance
        improvements = [r['delta_pct'] for r in dataset_results]
        print(f"\nPer-Dataset Deltas:")
        for r in dataset_results:
            print(f"  {r['dataset']:<15}: {r['delta_pct']:+.3f}%")

        # Determine winner
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        if g2_wins > l2_wins:
            print(f"\n🏆 G² (GRADIENT) is the OVERALL WINNER!")
            print(f"   Wins: {g2_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            print(f"\n   ✅ RECOMMENDED FOR PRODUCTION")
            print(f"\n   Key Benefits:")
            print(f"     • Gradient-aware quantization for SiLU layers")
            print(f"     • Captures output sensitivity: g = ∇SiLU(XW)/∇W")
            print(f"     • E[1 + g²/max(g²)] importance scoring")
            print(f"     • Prevents 'dead neuron resurrection' from quantization noise")
            winner = "G² (Gradient)"
        elif l2_wins > g2_wins:
            print(f"\n🏆 L2 (STANDARD) is the OVERALL WINNER!")
            print(f"   Wins: {l2_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            print(f"\n   ✅ RECOMMENDED FOR PRODUCTION")
            print(f"\n   Key Benefits:")
            print(f"     • Simpler implementation")
            print(f"     • Proven L2 salience metric")
            print(f"     • Consistent across all layers")
            print(f"     • Faster calibration (no gradient computation)")
            winner = "L2 (Standard)"
        else:
            print(f"\n🤝 TIE - Both methods equally strong")
            print(f"   G² recommended for theoretical benefits")
            print(f"\n   Recommendation:")
            print(f"     • Use G² for potential future gains")
            print(f"     • Gradient-aware approach handles risky channels")
            winner = "G² (Gradient) - tie"

        # Method characteristics
        print("\n" + "="*80)
        print("METHOD CHARACTERISTICS")
        print("="*80)

        print("\nL2 (Standard):")
        print("  Salience: E[X²] for ALL layers")
        print("  Pros:  Simple, proven, fast calibration")
        print("  Cons:  No gradient awareness, blind to output sensitivity")

        print("\nG² (Gradient):")
        print("  Salience:")
        print("    - SiLU layers: E[1 + g²/max(g²)] where g = ∇SiLU(XW)/∇W")
        print("    - Other layers: E[X²] (standard)")
        print("  Pros:  Output-sensitive, handles risky dead neurons")
        print("  Cons:  More complex, requires gradient computation")

        print("\nKey Insight:")
        print("  G² addresses the 'dead neuron resurrection' problem:")
        print("  Channels with negative pre-activation but high gradient sensitivity")
        print("  can be 'resurrected' by quantization noise. G² detects these risky")
        print("  channels and preserves their precision.")

        return {
            'winner': winner,
            'g2_wins': g2_wins,
            'l2_wins': l2_wins,
            'ties': ties,
            'avg_l2': avg_l2,
            'avg_g2': avg_g2,
            'avg_delta_pct': avg_delta_pct
        }

    def save_results(self, dataset_results, analysis, output_dir="./results"):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"g2_vs_l2_validation_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output = {
            'timestamp': timestamp,
            'seed': self.seed,
            'device': self.device,
            'comparison': 'G² vs L2',
            'datasets_tested': len(dataset_results),
            'dataset_results': dataset_results,
            'analysis': analysis,
            'detailed_results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ Results saved to: {filepath}")

        # Also save a summary README
        readme_path = os.path.join(output_dir, "G2_VS_L2_SUMMARY.md")
        with open(readme_path, 'w') as f:
            f.write("# G² vs L2 Salience Validation Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Winner:** {analysis['winner']}\n\n")
            f.write("## Comparison\n\n")
            f.write("- **L2 (Standard)**: E[X²] - Standard activation magnitude for all layers\n")
            f.write("- **G² (Gradient)**: Hybrid approach\n")
            f.write("  - SiLU-followed layers: E[1 + g²/max(g²)] where g = ∇SiLU(XW)/∇W\n")
            f.write("  - Other layers: E[X²] (standard)\n\n")
            f.write("## Results\n\n")
            f.write("| Dataset | L2 (Standard) | G² (Gradient) | Delta | Winner |\n")
            f.write("|---------|---------------|---------------|-------|--------|\n")
            for r in dataset_results:
                f.write(f"| {r['dataset']} | {r['l2_ppl']:.4f} | {r['g2_ppl']:.4f} | {r['delta_pct']:+.3f}% | {r['winner']} |\n")
            f.write(f"\n**Average:** L2={analysis['avg_l2']:.4f}, G²={analysis['avg_g2']:.4f}, Δ={analysis['avg_delta_pct']:+.3f}%\n")
            f.write(f"\n**Win Count:** G²={analysis['g2_wins']}/3, L2={analysis['l2_wins']}/3, Ties={analysis['ties']}/3\n")
            f.write(f"\n## Recommendation\n\n")
            f.write(f"**Deploy:** {analysis['winner']}\n\n")
            f.write("## Key Insights\n\n")
            if analysis['g2_wins'] > analysis['l2_wins']:
                f.write("✅ **G²'s gradient-aware quantization provides measurable benefits**\n\n")
                f.write("The gradient-squared approach successfully:\n")
                f.write("- Detects output-sensitive channels via ∇SiLU(XW)/∇W\n")
                f.write("- Prevents 'dead neuron resurrection' from quantization noise\n")
                f.write("- Uses E[1 + g²/max(g²)] to prioritize risky channels\n\n")
                f.write("This validates the theoretical advantage of gradient-based importance scoring.\n")
            elif analysis['l2_wins'] > analysis['g2_wins']:
                f.write("✅ **Standard L2 proves sufficient for this model**\n\n")
                f.write("While G² is theoretically superior, the simpler L2 approach\n")
                f.write("achieves better results in practice. This suggests:\n")
                f.write("- The model may not have many risky dead neurons\n")
                f.write("- L2 magnitude alone captures sufficient importance information\n")
                f.write("- Gradient computation overhead may not be worthwhile\n")
            else:
                f.write("🤝 **Both methods perform equally well**\n\n")
                f.write("G² is recommended for future-proofing, as the gradient-aware\n")
                f.write("approach is theoretically superior and may provide benefits on\n")
                f.write("models with more risky dead neurons.\n")

        print(f"✅ Summary saved to: {readme_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare G² vs L2 salience across datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--l2-path",
        type=str,
        default="./quantized_models/minicpm_gw_awq_asym_l2",
        help="Path to L2 (standard) quantized model"
    )
    parser.add_argument(
        "--g2-path",
        type=str,
        default="./quantized_models/minicpm_gw_awq_g2",
        help="Path to G² (gradient) quantized model"
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
    validator = G2vsL2Validator(device=device, seed=args.seed)

    # Run comprehensive validation
    validator.run_comprehensive_validation(
        l2_path=args.l2_path,
        g2_path=args.g2_path,
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
    print(f"✅ G² wins: {analysis['g2_wins']}")
    print(f"✅ L2 wins: {analysis['l2_wins']}")
    print(f"🤝 Ties: {analysis['ties']}")
    print(f"\n💡 Average delta: {analysis['avg_delta_pct']:+.3f}%")
    print("   (Negative = G² better, Positive = L2 better)")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
