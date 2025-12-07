"""
Comprehensive Comparison: Symmetric vs Asymmetric Quantization

This script compares 4 quantization methods:
1. GW-AWQ (Symmetric) - Group-Wise AWQ with symmetric INT4 [-8, 7]
2. GW-AWQ-Asym (Asymmetric) - Group-Wise AWQ with asymmetric INT4 [0, 15]
3. GWH-PRAQ (Symmetric) - Hybrid AWQ+PRAQ with symmetric INT4 [-8, 7]
4. GWH-PRAQ-Asym (Asymmetric) - Hybrid AWQ+PRAQ with asymmetric INT4 [0, 15]

Evaluation Metrics:
- Perplexity on WikiText-2 validation set
- Model size (MB)
- Throughput (tokens/second)
- Memory usage (MB)

Output:
- Comparison table
- Bar charts for perplexity
- Statistical analysis
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """Evaluate quantized models on WikiText-2."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def compute_perplexity(self, texts, max_samples=2000, max_length=512):
        """
        Compute perplexity on a set of texts.

        Args:
            texts: List of text strings
            max_samples: Maximum number of samples to evaluate
            max_length: Maximum sequence length

        Returns:
            perplexity: Float
        """
        total_loss = 0.0
        total_tokens = 0
        successful_samples = 0

        print(f"\nComputing perplexity on {min(max_samples, len(texts))} samples...")

        for i, text in enumerate(tqdm(texts[:max_samples], desc="Evaluating")):
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Skip very short sequences
                if inputs['input_ids'].shape[1] < 2:
                    continue

                # Forward pass
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

                # Accumulate loss
                n_tokens = inputs['input_ids'].shape[1]
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
                successful_samples += 1

            except Exception as e:
                if i % 500 == 0 and i > 0:
                    print(f"\nWarning: Some samples failed")
                continue

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        print(f"Evaluated {successful_samples} samples, {total_tokens} tokens")
        print(f"Average loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

        return perplexity

    @torch.no_grad()
    def measure_throughput(self, texts, n_samples=50, max_length=512):
        """
        Measure inference throughput (tokens/second).

        Args:
            texts: List of text strings
            n_samples: Number of samples to benchmark
            max_length: Maximum sequence length

        Returns:
            throughput: Tokens per second (or None if failed)
        """
        print(f"\nMeasuring throughput on {n_samples} samples...")

        total_tokens = 0
        total_time = 0.0
        failed_count = 0

        try:
            # Warmup
            for text in texts[:5]:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = self.model(**inputs, use_cache=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            for text in tqdm(texts[:n_samples], desc="Throughput"):
                try:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.time()

                    # Use use_cache=False to avoid cache compatibility issues
                    _ = self.model(**inputs, use_cache=False)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()

                    total_time += (end_time - start_time)
                    total_tokens += inputs['input_ids'].shape[1]

                except Exception as e:
                    failed_count += 1
                    if failed_count > n_samples // 2:
                        # Too many failures, abort throughput measurement
                        print(f"\n⚠️  Throughput measurement failed (too many errors)")
                        return None
                    continue

            if total_time > 0 and total_tokens > 0:
                throughput = total_tokens / total_time
                print(f"Throughput: {throughput:.2f} tokens/second")
                return throughput
            else:
                print(f"⚠️  Throughput measurement failed (no successful samples)")
                return None

        except Exception as e:
            print(f"⚠️  Throughput measurement failed: {e}")
            return None

    def get_model_size_mb(self):
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    def get_memory_usage_mb(self):
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0


def load_wikitext2(split="validation", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts


def evaluate_model(model_path, model_name, device, eval_texts, benchmark_texts):
    """
    Evaluate a single model.

    Args:
        model_path: Path to model directory
        model_name: Name for display
        device: Device to use
        eval_texts: Texts for perplexity evaluation
        benchmark_texts: Texts for throughput benchmark

    Returns:
        results: Dictionary of metrics
    """
    print("\n" + "=" * 80)
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print("=" * 80)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print(f"   Please run the corresponding quantization script first.")
        return None

    try:
        # Load model
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        # Create evaluator
        evaluator = ModelEvaluator(model, tokenizer, device)

        # Get model size
        model_size = evaluator.get_model_size_mb()
        print(f"Model size: {model_size:.2f} MB")

        # Compute perplexity
        perplexity = evaluator.compute_perplexity(eval_texts, max_samples=2000)

        # Measure throughput (may fail due to compatibility issues)
        throughput = evaluator.measure_throughput(benchmark_texts, n_samples=50)
        if throughput is None:
            print("⚠️  Skipping throughput measurement for this model")

        # Get memory usage
        memory_usage = evaluator.get_memory_usage_mb()
        print(f"Memory usage: {memory_usage:.2f} MB")

        results = {
            'name': model_name,
            'path': model_path,
            'perplexity': perplexity,
            'model_size_mb': model_size,
            'throughput_tokens_per_sec': throughput if throughput is not None else 0.0,
            'memory_usage_mb': memory_usage
        }

        # Cleanup
        del model, tokenizer, evaluator
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_visualizations(results, output_dir):
    """
    Create comparison visualizations.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Extract data
    names = [r['name'] for r in results]
    perplexities = [r['perplexity'] for r in results]
    model_sizes = [r['model_size_mb'] for r in results]
    throughputs = [r['throughput_tokens_per_sec'] for r in results]

    # Color scheme: symmetric vs asymmetric
    colors = []
    for name in names:
        if 'Asym' in name:
            colors.append('#E74C3C')  # Red for asymmetric
        else:
            colors.append('#3498DB')  # Blue for symmetric

    # 1. Perplexity comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), perplexities, color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('Quantization Method', fontsize=12, fontweight='bold')
    plt.ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
    plt.title('Perplexity Comparison: Symmetric vs Asymmetric', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, perplexities)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perplexity_comparison.png'), dpi=300)
    print(f"Saved: {output_dir}/perplexity_comparison.png")
    plt.close()

    # 2. Throughput comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), throughputs, color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('Quantization Method', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (tokens/sec, higher is better)', fontsize=12, fontweight='bold')
    plt.title('Throughput Comparison: Symmetric vs Asymmetric', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, throughputs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=300)
    print(f"Saved: {output_dir}/throughput_comparison.png")
    plt.close()

    # 3. Multi-metric comparison (normalized)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ('Perplexity\n(lower=better)', perplexities, True),
        ('Model Size (MB)\n(lower=better)', model_sizes, True),
        ('Throughput (tok/s)\n(higher=better)', throughputs, False)
    ]

    for ax, (title, values, lower_is_better) in zip(axes, metrics):
        # Normalize values
        if lower_is_better:
            normalized = [min(values) / v for v in values]  # Lower is better
        else:
            normalized = [v / max(values) for v in values]  # Higher is better

        bars = ax.bar(range(len(names)), normalized, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Normalized Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val, orig_val in zip(bars, normalized, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}\n({orig_val:.1f})', ha='center', va='bottom', fontsize=8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498DB', edgecolor='black', label='Symmetric'),
        Patch(facecolor='#E74C3C', edgecolor='black', label='Asymmetric')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11,
              bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_comparison.png'), dpi=300)
    print(f"Saved: {output_dir}/normalized_comparison.png")
    plt.close()


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS: Symmetric vs Asymmetric Quantization")
    print("=" * 100)

    # Header
    print(f"{'Method':<25} {'Perplexity':>12} {'Size (MB)':>12} {'Throughput':>15} {'Memory (MB)':>12}")
    print("-" * 100)

    # Rows
    for r in results:
        throughput_str = f"{r['throughput_tokens_per_sec']:>12.1f} tok/s" if r['throughput_tokens_per_sec'] > 0 else "         N/A"
        print(f"{r['name']:<25} {r['perplexity']:>12.2f} {r['model_size_mb']:>12.2f} "
              f"{throughput_str:>15} {r['memory_usage_mb']:>12.2f}")

    print("=" * 100)

    # Analysis
    print("\nANALYSIS:")
    print("-" * 100)

    # Find best performers
    best_perplexity = min(results, key=lambda x: x['perplexity'])
    smallest_size = min(results, key=lambda x: x['model_size_mb'])

    print(f"✓ Best Perplexity: {best_perplexity['name']} ({best_perplexity['perplexity']:.2f})")

    # Only show throughput if any model has valid throughput
    valid_throughputs = [r for r in results if r['throughput_tokens_per_sec'] > 0]
    if valid_throughputs:
        best_throughput = max(valid_throughputs, key=lambda x: x['throughput_tokens_per_sec'])
        print(f"✓ Best Throughput: {best_throughput['name']} ({best_throughput['throughput_tokens_per_sec']:.1f} tok/s)")
    else:
        print(f"✓ Best Throughput: N/A (measurement failed for all models)")

    print(f"✓ Smallest Size: {smallest_size['name']} ({smallest_size['model_size_mb']:.2f} MB)")

    # Compare symmetric vs asymmetric
    print("\nSYMMETRIC vs ASYMMETRIC:")
    print("-" * 100)

    # Group by method type
    awq_sym = next((r for r in results if 'AWQ' in r['name'] and 'Asym' not in r['name'] and 'Hybrid' not in r['name']), None)
    awq_asym = next((r for r in results if 'AWQ' in r['name'] and 'Asym' in r['name'] and 'Hybrid' not in r['name']), None)
    praq_sym = next((r for r in results if 'Hybrid' in r['name'] and 'Asym' not in r['name']), None)
    praq_asym = next((r for r in results if 'Hybrid' in r['name'] and 'Asym' in r['name']), None)

    if awq_sym and awq_asym:
        ppl_diff = ((awq_asym['perplexity'] - awq_sym['perplexity']) / awq_sym['perplexity']) * 100
        thr_diff = ((awq_asym['throughput_tokens_per_sec'] - awq_sym['throughput_tokens_per_sec']) / awq_sym['throughput_tokens_per_sec']) * 100
        print(f"GW-AWQ: Asymmetric vs Symmetric")
        print(f"  Perplexity: {ppl_diff:+.2f}% ({'better' if ppl_diff < 0 else 'worse'})")
        print(f"  Throughput: {thr_diff:+.2f}% ({'better' if thr_diff > 0 else 'worse'})")

    if praq_sym and praq_asym:
        ppl_diff = ((praq_asym['perplexity'] - praq_sym['perplexity']) / praq_sym['perplexity']) * 100
        thr_diff = ((praq_asym['throughput_tokens_per_sec'] - praq_sym['throughput_tokens_per_sec']) / praq_sym['throughput_tokens_per_sec']) * 100
        print(f"\nGWH-PRAQ: Asymmetric vs Symmetric")
        print(f"  Perplexity: {ppl_diff:+.2f}% ({'better' if ppl_diff < 0 else 'worse'})")
        print(f"  Throughput: {thr_diff:+.2f}% ({'better' if thr_diff > 0 else 'worse'})")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Symmetric vs Asymmetric Quantization Methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--eval-samples", type=int, default=2000,
                       help="Number of samples for perplexity evaluation")
    parser.add_argument("--benchmark-samples", type=int, default=50,
                       help="Number of samples for throughput benchmark")
    parser.add_argument("--output-dir", type=str, default="./comparison_results",
                       help="Output directory for results")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate comparison visualizations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load evaluation data
    eval_texts = load_wikitext2(split="validation", n_samples=args.eval_samples)
    benchmark_texts = load_wikitext2(split="validation", n_samples=args.benchmark_samples)

    # Define models to compare
    models_to_compare = [
        {
            'path': './quantized_models/minicpm_gw_awq',
            'name': 'GW-AWQ (Sym)'
        },
        {
            'path': './quantized_models/minicpm_gw_awq_asym',
            'name': 'GW-AWQ (Asym)'
        },
        {
            'path': './quantized_models/minicpm_gwh_praq',
            'name': 'GWH-PRAQ (Sym)'
        },
        {
            'path': './quantized_models/minicpm_gwh_praq_asym',
            'name': 'GWH-PRAQ (Asym)'
        }
    ]

    # Evaluate all models
    results = []
    for model_config in models_to_compare:
        result = evaluate_model(
            model_config['path'],
            model_config['name'],
            device,
            eval_texts,
            benchmark_texts
        )
        if result:
            results.append(result)

    if not results:
        print("\n❌ No models were successfully evaluated.")
        print("   Please run the quantization scripts first:")
        print("   - python gw_awq.py")
        print("   - python gw_awq_asym.py")
        print("   - python gwh_praq.py")
        print("   - python gwh_praq_asym.py")
        return

    # Print comparison table
    print_comparison_table(results)

    # Save results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    # Create visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        create_comparison_visualizations(results, args.output_dir)
        print(f"✓ Visualizations saved to: {args.output_dir}/")

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()
