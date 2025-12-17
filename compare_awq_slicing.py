"""
Cross-Dataset Validation: AWQ Methods with Sliding Window Evaluation

Uses sliding window (rolling window) evaluation instead of standard perplexity.
This approach follows the lm-evaluation-harness loglikelihood_rolling methodology.

Key differences from standard perplexity:
1. Uses sliding window with stride=512 (configurable)
2. Computes log-likelihood on overlapping windows
3. More realistic evaluation on long documents
4. Tests model's ability to predict with longer context

Datasets tested:
1. WikiText-2 test - In-distribution (Wikipedia, formal)
2. C4 validation - Cross-dataset (Web crawl, diverse)
3. AG News test - Cross-dataset (News, journalistic)

Comparison:
- Standard AWQ (gw_awq_asym_l2.py): Uses E[X²] salience + min/max quantization
- Heuristic AWQ (awq_op.py): Uses E[X²] salience + E[Xs]-guided quantization

Method: Sliding Window (Rolling Window)
Stride: 512 tokens (default)
Metric Type: loglikelihood_rolling (similar to lm-evaluation-harness)

Author: AWQ sliding window validation
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


class AWQSlidingWindowValidator:
    """Comprehensive cross-dataset validation using sliding window evaluation."""

    def __init__(self, device="cuda", seed=42, stride=512, max_length=1024):
        self.device = device
        self.seed = seed
        self.stride = stride
        self.max_length = max_length
        self.results = {}

        print("="*80)
        print("AWQ SLIDING WINDOW CROSS-DATASET VALIDATION")
        print("="*80)
        print(f"Device: {device}")
        print(f"Random seed: {seed}")
        print(f"Sliding window stride: {stride}")
        print(f"Maximum sequence length: {max_length}")
        print("="*80)

    def load_wikitext2_test(self, n_samples=500):
        """Load WikiText-2 test set.

        Note: Fewer samples needed for sliding window since each sample
        produces multiple windows.
        """
        print("\n[1/3] Loading WikiText-2 test...")
        random.seed(self.seed)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # For sliding window, we want longer documents
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 500]

        random.seed(self.seed)
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)

        print(f"  ✅ Loaded {len(texts)} samples (longer documents for sliding window)")
        return texts

    def load_c4_validation(self, n_samples=500):
        """Load C4 validation set."""
        print("\n[2/3] Loading C4 validation...")
        random.seed(self.seed)

        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

        texts = []
        for i, item in enumerate(tqdm(dataset, desc="  Collecting C4", total=n_samples*2)):
            if len(texts) >= n_samples:
                break
            text = item['text']
            # Prefer longer documents for sliding window
            if len(text.strip()) > 500:
                texts.append(text)

        random.seed(self.seed)
        random.shuffle(texts)

        print(f"  ✅ Loaded {len(texts)} samples (longer documents for sliding window)")
        return texts[:n_samples]

    def load_ag_news_test(self, n_samples=500):
        """Load AG News test set."""
        print("\n[3/3] Loading AG News test...")
        random.seed(self.seed)

        dataset = load_dataset("ag_news", split="test")
        # AG News articles are shorter, but we still filter
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 200]

        random.seed(self.seed)
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)

        print(f"  ✅ Loaded {len(texts)} samples")
        return texts
    @torch.no_grad()
    def evaluate_sliding_window(self, model, tokenizer, texts):
        """
        CORRECTED: Evaluate using Sliding Window (Rolling Likelihood).
        """
        model.eval()
        nlls = [] 
        total_tokens = 0
        successful_docs = 0

        for text in tqdm(texts, desc="  Evaluating (sliding window)", leave=False):
            try:
                # FIX 2: Explicitly add special tokens (BOS)
                encodings = tokenizer(
                    text, 
                    return_tensors="pt", 
                    add_special_tokens=True 
                )
                input_ids = encodings.input_ids[:, :self.max_length * 10] 
                
                seq_len = input_ids.size(1)
                prev_end_loc = 0
                
                for begin_loc in range(0, seq_len, self.stride):
                    end_loc = min(begin_loc + self.max_length, seq_len)
                    trg_len = end_loc - prev_end_loc  
                    
                    # Prepare inputs
                    input_ids_chunk = input_ids[:, begin_loc:end_loc].to(self.device)
                    target_ids_chunk = input_ids_chunk.clone()
                    
                    # FIX 3: MASKING THE CONTEXT
                    # We set the labels of the previous tokens to -100
                    # This ensures we only calculate loss on the NEW tokens (the stride)
                    target_ids_chunk[:, :-trg_len] = -100

                    with torch.no_grad():
                        outputs = model(input_ids_chunk, labels=target_ids_chunk)
                        
                        # Calculate Sum of NLL (not average)
                        # outputs.loss is the mean NLL of the unmasked tokens
                        # We multiply by trg_len to get the sum
                        neg_log_likelihood = outputs.loss * trg_len

                    nlls.append(neg_log_likelihood)
                    
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

                successful_docs += 1
                total_tokens += seq_len

            except Exception as e:
                continue

        if not nlls:
            return None

        # FIX 4: Correct PPL Aggregation
        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / total_tokens).item()
        
        return {
            "perplexity": perplexity,
            "avg_loss": (total_nll / total_tokens).item(),
            "num_documents": successful_docs,
            "total_windows": len(nlls),
            "total_tokens": total_tokens
        }

    def evaluate_model_on_dataset(self, model_path, model_name, texts, dataset_name):
        """Evaluate a model on a specific dataset using sliding window."""
        print(f"\n  Evaluating {model_name} on {dataset_name}...")

        if not os.path.exists(model_path):
            print(f"  ❌ Model not found: {model_path}")
            return None

        try:
            # FIX: Explicitly handle the Llama-3/Mistral regex issue
            # The warning requires passing this flag to fix tokenization
            tokenizer_kwargs = {}
            if "Llama-3" in model_path or "Mistral" in model_path:
                tokenizer_kwargs["fix_mistral_regex"] = True

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=True,
                **tokenizer_kwargs  # <--- PASSING THE FIX HERE
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Llama 3 explicit BOS check
            if tokenizer.bos_token_id is None:
                print("  ⚠️ WARNING: Tokenizer missing BOS token. Forcing add_bos_token=True")
                tokenizer.add_bos_token = True

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )

            results = self.evaluate_sliding_window(model, tokenizer, texts)

            print(f"  ✅ Perplexity: {results['perplexity']:.4f}")
            print(f"     Documents: {results['num_documents']}, Windows: {results['total_windows']}")

            del model
            torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            # If the flag causes a crash (rare on older transformers), fallback:
            if "unexpected keyword argument" in str(e):
                print("  ⚠️ 'fix_mistral_regex' not supported. Trying without...")
                # ... (You could add fallback logic here, but usually upgrading transformers is better)
            import traceback
            traceback.print_exc()
            return None
    def run_comprehensive_validation(self, heuristic_path, standard_path, n_samples=500):
        """Run validation on all datasets using sliding window."""
        print("\n" + "="*80)
        print("LOADING DATASETS")
        print("="*80)

        datasets = {
            'WikiText-2': self.load_wikitext2_test(n_samples),
            'C4': self.load_c4_validation(n_samples),
            'AG News': self.load_ag_news_test(n_samples)
        }

        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)

        models = {
            'Heuristic AWQ': heuristic_path,
            'Standard AWQ': standard_path
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
        print("COMPREHENSIVE RESULTS (Sliding Window Evaluation)")
        print("="*80)

        # Table header
        print(f"\n{'Dataset':<15} {'Heuristic AWQ':<15} {'Standard AWQ':<15} {'Delta':<12} {'Winner':<10}")
        print("-" * 80)

        # Results for each dataset
        dataset_results = []
        for dataset_name in ['WikiText-2', 'C4', 'AG News']:
            if dataset_name in self.results:
                heur_ppl = self.results[dataset_name]['Heuristic AWQ']['perplexity']
                std_ppl = self.results[dataset_name]['Standard AWQ']['perplexity']
                delta = heur_ppl - std_ppl
                delta_pct = (delta / std_ppl) * 100

                # Winner: Heuristic is better if delta < -0.05 (lower perplexity)
                winner = "Heuristic" if delta < -0.05 else ("Standard" if delta > 0.05 else "Tie")

                print(f"{dataset_name:<15} {heur_ppl:<15.4f} {std_ppl:<15.4f} {delta_pct:>+11.3f}%  {winner:<10}")

                dataset_results.append({
                    'dataset': dataset_name,
                    'heuristic_ppl': heur_ppl,
                    'standard_ppl': std_ppl,
                    'delta_pct': delta_pct,
                    'winner': winner,
                    'heuristic_windows': self.results[dataset_name]['Heuristic AWQ']['total_windows'],
                    'standard_windows': self.results[dataset_name]['Standard AWQ']['total_windows']
                })

        return dataset_results

    def analyze_results(self, dataset_results):
        """Comprehensive analysis of results."""
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)

        # Count wins
        heur_wins = sum(1 for r in dataset_results if r['winner'] == 'Heuristic')
        std_wins = sum(1 for r in dataset_results if r['winner'] == 'Standard')
        ties = sum(1 for r in dataset_results if r['winner'] == 'Tie')

        print(f"\nWin Count:")
        print(f"  Heuristic AWQ: {heur_wins}/{len(dataset_results)}")
        print(f"  Standard AWQ:  {std_wins}/{len(dataset_results)}")
        print(f"  Ties:          {ties}/{len(dataset_results)}")

        # Average performance
        avg_heur = np.mean([r['heuristic_ppl'] for r in dataset_results])
        avg_std = np.mean([r['standard_ppl'] for r in dataset_results])
        avg_delta_pct = ((avg_heur - avg_std) / avg_std) * 100

        print(f"\nAverage Perplexity (Sliding Window):")
        print(f"  Heuristic AWQ: {avg_heur:.4f}")
        print(f"  Standard AWQ:  {avg_std:.4f}")
        print(f"  Difference:    {avg_delta_pct:+.3f}%")

        # Window statistics
        total_heur_windows = sum(r['heuristic_windows'] for r in dataset_results)
        total_std_windows = sum(r['standard_windows'] for r in dataset_results)
        print(f"\nTotal Windows Evaluated:")
        print(f"  Heuristic AWQ: {total_heur_windows:,}")
        print(f"  Standard AWQ:  {total_std_windows:,}")

        # Determine winner
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        if heur_wins > std_wins:
            print(f"\n🏆 HEURISTIC AWQ is the OVERALL WINNER!")
            print(f"   Wins: {heur_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            print(f"\n   ✅ RECOMMENDED: Use awq_op.py with heuristic guidance")
            print(f"   Benefit: E[Xs]-guided quantization performs better on long-context evaluation")
            winner = "Heuristic AWQ"
        elif std_wins > heur_wins:
            print(f"\n🏆 STANDARD AWQ is the OVERALL WINNER!")
            print(f"   Wins: {std_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            print(f"\n   ✅ RECOMMENDED: Use gw_awq_asym_l2.py (simpler, faster)")
            print(f"   Benefit: Standard min/max quantization is sufficient even for long-context")
            winner = "Standard AWQ"
        else:
            print(f"\n🤝 TIE - Both methods equally strong")
            print(f"   Standard AWQ recommended (simpler implementation)")
            print(f"   Heuristic adds complexity without clear benefit")
            winner = "Standard AWQ (tie)"

        # Method characteristics
        print("\n" + "="*80)
        print("EVALUATION METHOD")
        print("="*80)

        print("\nSliding Window (Rolling Window) Evaluation:")
        print(f"  Stride:        {self.stride} tokens")
        print(f"  Max length:    {self.max_length} tokens")
        print(f"  Metric:        loglikelihood_rolling (lm-evaluation-harness style)")
        print(f"  Advantages:")
        print(f"    - Tests model with longer context")
        print(f"    - Overlapping windows for better coverage")
        print(f"    - More realistic than short-sequence perplexity")
        print(f"    - Matches practical deployment scenarios")

        print("\n" + "="*80)
        print("METHOD CHARACTERISTICS")
        print("="*80)

        print("\nStandard AWQ (gw_awq_asym_l2.py):")
        print("  Salience:     E[X²] for scaling")
        print("  Quantization: Min/max asymmetric per group")
        print("  Complexity:   O(n) - simple nearest rounding")
        print("  Speed:        Fast")

        print("\nHeuristic AWQ (awq_op.py):")
        print("  Salience:     E[X²] for scaling")
        print("  Quantization: E[Xs]-guided greedy refinement")
        print("  Complexity:   O(n²) - iterative flip selection")
        print("  Speed:        Slower (heuristic search)")
        print("  Innovation:   Minimizes output error dot(Xs, W-W_quant)")

        print("\n" + "="*80)
        print("DATASET CHARACTERISTICS")
        print("="*80)

        print("\nWikiText-2 (Test):")
        print("  Source: Wikipedia articles")
        print("  Style:  Formal, encyclopedic")
        print("  Domain: General knowledge")
        print("  Length: Long documents (good for sliding window)")

        print("\nC4:")
        print("  Source: Common Crawl web scrape")
        print("  Style:  Diverse, noisy, real-world")
        print("  Domain: Mixed web content")
        print("  Length: Variable (medium to long)")

        print("\nAG News:")
        print("  Source: News articles")
        print("  Style:  Journalistic, factual")
        print("  Domain: World, sports, business, sci/tech")
        print("  Length: Shorter articles")

        print("\nConclusion:")
        print("  Sliding window evaluation tests model quality on longer contexts,")
        print("  providing more realistic assessment than standard perplexity.")
        print("  Testing on 3 diverse datasets validates generalization across")
        print("  different domains, styles, and document lengths.")

        return {
            'winner': winner,
            'heuristic_wins': heur_wins,
            'standard_wins': std_wins,
            'ties': ties,
            'avg_heuristic': avg_heur,
            'avg_standard': avg_std,
            'avg_delta_pct': avg_delta_pct,
            'stride': self.stride,
            'max_length': self.max_length
        }

    def save_results(self, dataset_results, analysis, output_dir="./results"):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"awq_sliding_window_validation_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output = {
            'timestamp': timestamp,
            'seed': self.seed,
            'device': self.device,
            'stride': self.stride,
            'max_length': self.max_length,
            'evaluation_method': 'sliding_window_rolling',
            'datasets_tested': len(dataset_results),
            'dataset_results': dataset_results,
            'analysis': analysis,
            'detailed_results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ Results saved to: {filepath}")

        # Also save a summary README
        readme_path = os.path.join(output_dir, "AWQ_SLIDING_WINDOW_SUMMARY.md")
        with open(readme_path, 'w') as f:
            f.write("# AWQ Sliding Window Validation Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Evaluation Method:** Sliding Window (Rolling Window)\n")
            f.write(f"**Stride:** {self.stride} tokens\n")
            f.write(f"**Max Length:** {self.max_length} tokens\n\n")
            f.write(f"**Winner:** {analysis['winner']}\n\n")
            f.write("## Results\n\n")
            f.write("| Dataset | Heuristic AWQ | Standard AWQ | Delta | Winner | Windows |\n")
            f.write("|---------|---------------|--------------|-------|--------|----------|\n")
            for r in dataset_results:
                f.write(f"| {r['dataset']} | {r['heuristic_ppl']:.4f} | {r['standard_ppl']:.4f} | {r['delta_pct']:+.3f}% | {r['winner']} | {r['heuristic_windows']:,} |\n")
            f.write(f"\n**Average:** Heuristic={analysis['avg_heuristic']:.4f}, Standard={analysis['avg_standard']:.4f}, Δ={analysis['avg_delta_pct']:+.3f}%\n")
            f.write(f"\n**Win Count:** Heuristic={analysis['heuristic_wins']}/3, Standard={analysis['standard_wins']}/3, Ties={analysis['ties']}/3\n")

            f.write(f"\n## Evaluation Method\n\n")
            f.write("**Sliding Window (Rolling Window) Evaluation:**\n\n")
            f.write("This approach uses overlapping windows to evaluate model quality on longer contexts:\n\n")
            f.write(f"- **Stride:** {self.stride} tokens (window shift)\n")
            f.write(f"- **Max Length:** {self.max_length} tokens (window size)\n")
            f.write("- **Metric:** loglikelihood_rolling (lm-evaluation-harness methodology)\n\n")
            f.write("**Advantages:**\n")
            f.write("- Tests prediction quality with longer context\n")
            f.write("- Overlapping windows provide comprehensive coverage\n")
            f.write("- More realistic than short-sequence perplexity\n")
            f.write("- Matches how models are used in practice\n\n")

            f.write(f"\n## Methods Compared\n\n")
            f.write("### Standard AWQ (gw_awq_asym_l2.py)\n")
            f.write("- Salience: E[X²] for activation-aware scaling\n")
            f.write("- Quantization: Min/max asymmetric per group\n")
            f.write("- Complexity: O(n) - simple and fast\n\n")

            f.write("### Heuristic AWQ (awq_op.py)\n")
            f.write("- Salience: E[X²] for activation-aware scaling\n")
            f.write("- Quantization: E[Xs]-guided greedy refinement\n")
            f.write("- Complexity: O(n²) - slower but aims to minimize output error\n")
            f.write("- Innovation: Uses dot(Xs, W-W_quant) to guide rounding\n\n")

            f.write(f"\n## Recommendation\n\n")
            f.write(f"**Deploy:** {analysis['winner']}\n\n")

            if analysis['heuristic_wins'] > analysis['standard_wins']:
                f.write("The heuristic approach provides measurable improvement even on long-context\n")
                f.write("evaluation. The E[Xs]-guided quantization maintains better prediction quality\n")
                f.write("across sliding windows, justifying the additional computational cost.\n")
            elif analysis['standard_wins'] > analysis['standard_wins']:
                f.write("The standard AWQ approach performs well even on long-context evaluation.\n")
                f.write("The heuristic refinement does not provide enough benefit to justify the\n")
                f.write("additional complexity and computational cost.\n")
            else:
                f.write("Both methods perform equivalently on long-context evaluation. Standard AWQ\n")
                f.write("is recommended for its simplicity and faster quantization time.\n")

        print(f"✅ Summary saved to: {readme_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-dataset validation with sliding window: Heuristic AWQ vs Standard AWQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--heuristic-path",
        type=str,
        default="./quantized_models/awq_heuristic",
        help="Path to Heuristic AWQ model (awq_op.py output)"
    )
    parser.add_argument(
        "--standard-path",
        type=str,
        default="./quantized_models/minicpm_gw_awq_asym_l2",
        help="Path to Standard AWQ model (gw_awq_asym_l2.py output)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples per dataset (fewer needed due to sliding window)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride (tokens)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length per window (tokens)"
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
    validator = AWQSlidingWindowValidator(
        device=device,
        seed=args.seed,
        stride=args.stride,
        max_length=args.max_length
    )

    # Run comprehensive validation
    validator.run_comprehensive_validation(
        heuristic_path=args.heuristic_path,
        standard_path=args.standard_path,
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
    print(f"✅ Heuristic wins: {analysis['heuristic_wins']}")
    print(f"✅ Standard wins: {analysis['standard_wins']}")
    print(f"🤝 Ties: {analysis['ties']}")
    print(f"\n📏 Evaluation: Sliding window (stride={args.stride}, max_length={args.max_length})")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
