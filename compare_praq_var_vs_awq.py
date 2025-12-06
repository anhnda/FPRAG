"""
Compare Variance-Penalized PRAQ vs AWQ

This script evaluates and compares the variance-penalized PRAQ quantization
against the AWQ baseline to verify if variance penalization improves
error propagation and perplexity.

Supports comparing multiple variance penalty values to find the optimal setting.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
import random
import os
import argparse
import matplotlib.pyplot as plt


def load_wikitext2_validation(n_samples=2000, seed=42):
    """Load WikiText-2 validation set with random sampling."""
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
    """
    Evaluate perplexity on a list of text samples.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_time = 0
    successful_samples = 0
    failed_samples = 0

    losses = []

    for text in tqdm(texts, desc="Evaluating", leave=False):
        try:
            # Tokenize
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

            # Forward pass (disable cache to avoid DynamicCache issues)
            start_time = time.time()
            outputs = model(input_ids, labels=input_ids, use_cache=False)
            elapsed = time.time() - start_time

            loss = outputs.loss.item()
            n_tokens = input_ids.shape[1]

            total_loss += loss * n_tokens
            total_tokens += n_tokens
            total_time += elapsed
            losses.append(loss)
            successful_samples += 1

        except Exception as e:
            failed_samples += 1
            # Only print first few errors to avoid spam
            if failed_samples <= 3:
                print(f"\nError processing sample: {e}")
            continue

    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss) if total_tokens > 0 else float('inf')
    avg_time_per_sample = total_time / successful_samples if successful_samples > 0 else 0
    throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "avg_time_per_sample": avg_time_per_sample,
        "throughput_tokens_per_sec": throughput,
        "total_tokens": total_tokens,
        "num_samples": successful_samples,
        "failed_samples": failed_samples,
        "losses": losses
    }


def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def evaluate_model(model_path, model_name, eval_texts, device="cuda"):
    """Evaluate a single model and return results."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None

    try:
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)

        model_size = get_model_size(model)
        print(f"Model size: {model_size:.2f} MB")

        results = evaluate_perplexity(model, tokenizer, eval_texts, device=device)
        results['model_size_mb'] = model_size
        results['model_name'] = model_name

        print(f"\nResults:")
        print(f"  Perplexity: {results['perplexity']:.4f}")
        print(f"  Avg Loss: {results['avg_loss']:.4f}")
        print(f"  Throughput: {results['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"  Model Size: {model_size:.2f} MB")
        print(f"  Processed: {results['num_samples']} samples")
        if results.get('failed_samples', 0) > 0:
            print(f"  ⚠️  Failed samples: {results['failed_samples']}")

        # Clean up
        del model
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"\n❌ Error evaluating {model_name}:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_table(results_dict):
    """Create a formatted comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")

    # Header
    models = list(results_dict.keys())
    header = f"{'Metric':<30}"
    for model_name in models:
        header += f" {model_name:<18}"
    print(header)
    print("-" * (30 + 20 * len(models)))

    # Perplexity
    row = f"{'Perplexity':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['perplexity']:>18.4f}"
        else:
            row += f" {'N/A':>18}"
    print(row)

    # Avg Loss
    row = f"{'Avg Loss':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['avg_loss']:>18.4f}"
        else:
            row += f" {'N/A':>18}"
    print(row)

    # Model Size
    row = f"{'Model Size (MB)':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['model_size_mb']:>18.2f}"
        else:
            row += f" {'N/A':>18}"
    print(row)

    # Throughput
    row = f"{'Throughput (tok/s)':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['throughput_tokens_per_sec']:>18.2f}"
        else:
            row += f" {'N/A':>18}"
    print(row)

    print("-" * (30 + 20 * len(models)))


def analyze_improvements(results_dict, baseline="AWQ"):
    """Analyze improvements relative to baseline."""
    print(f"\n{'='*80}")
    print(f"IMPROVEMENTS RELATIVE TO {baseline}")
    print(f"{'='*80}\n")

    if baseline not in results_dict or not results_dict[baseline]:
        print(f"❌ Baseline model '{baseline}' not found or failed to evaluate")
        return

    baseline_result = results_dict[baseline]
    baseline_ppl = baseline_result['perplexity']

    improvements = []

    for model_name, result in results_dict.items():
        if model_name == baseline or not result:
            continue

        ppl = result['perplexity']
        delta_ppl = ppl - baseline_ppl
        delta_pct = (delta_ppl / baseline_ppl) * 100

        improvements.append({
            'model': model_name,
            'perplexity': ppl,
            'delta': delta_ppl,
            'delta_pct': delta_pct
        })

    # Sort by perplexity (lower is better)
    improvements.sort(key=lambda x: x['perplexity'])

    print(f"Baseline: {baseline} (Perplexity = {baseline_ppl:.4f})\n")

    for imp in improvements:
        sign = "+" if imp['delta'] > 0 else ""
        symbol = "❌" if imp['delta'] > 0 else "✅"
        print(f"{symbol} {imp['model']:<20} Perplexity: {imp['perplexity']:.4f} "
              f"({sign}{imp['delta']:.4f}, {sign}{imp['delta_pct']:.2f}%)")

    # Find best model
    if improvements:
        best = improvements[0]
        if best['delta'] < 0:
            print(f"\n🏆 BEST: {best['model']}")
            print(f"   Achieves {abs(best['delta_pct']):.2f}% lower perplexity than {baseline}")
        elif best['delta'] > 0:
            print(f"\n⚠️  All variants perform worse than {baseline}")
        else:
            print(f"\n🤝 All variants achieve similar perplexity to {baseline}")


def visualize_results(results_dict, output_dir="./visualizations/praq_var_comparison"):
    """Create visualizations comparing the models."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}

    if len(valid_results) < 2:
        print("\n⚠️  Not enough models to visualize")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(valid_results.keys())
    perplexities = [valid_results[m]['perplexity'] for m in models]
    sizes = [valid_results[m]['model_size_mb'] for m in models]

    # Plot 1: Perplexity comparison
    colors = ['#2ecc71' if 'AWQ' in m else '#e74c3c' if 'original' in m.lower() else '#3498db'
              for m in models]

    axes[0].bar(range(len(models)), perplexities, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Perplexity (lower is better)')
    axes[0].set_title('Perplexity Comparison')
    axes[0].grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (ppl, model) in enumerate(zip(perplexities, models)):
        axes[0].text(i, ppl + 0.5, f'{ppl:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Perplexity vs Model Size
    axes[1].scatter(sizes, perplexities, s=200, alpha=0.7, c=colors, edgecolors='black', linewidths=2)

    # Annotate points
    for model, size, ppl in zip(models, sizes, perplexities):
        axes[1].annotate(model, (size, ppl), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

    axes[1].set_xlabel('Model Size (MB)')
    axes[1].set_ylabel('Perplexity (lower is better)')
    axes[1].set_title('Perplexity vs Model Size')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved visualization: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Variance-Penalized PRAQ vs AWQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--awq-path",
        type=str,
        default="./quantized_models/minicpm_awq_custom",
        help="Path to AWQ quantized model"
    )
    parser.add_argument(
        "--praq-var-paths",
        type=str,
        nargs='+',
        default=["./quantized_models/minicpm_praq_var"],
        help="Paths to variance-penalized PRAQ models (can specify multiple)"
    )
    parser.add_argument(
        "--praq-var-names",
        type=str,
        nargs='+',
        default=None,
        help="Names for variance-penalized PRAQ models (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=2000,
        help="Number of evaluation samples"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comparison visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations/praq_var_comparison",
        help="Output directory for visualizations"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("Variance-Penalized PRAQ vs AWQ Comparison")
    print("="*80)
    print(f"Device: {device}")
    print(f"Evaluation samples: {args.n_eval}")
    print(f"AWQ model: {args.awq_path}")
    print(f"PRAQ-Var models: {args.praq_var_paths}")
    print("="*80)

    # Load validation data
    eval_texts = load_wikitext2_validation(n_samples=args.n_eval)

    # Results storage
    results = {}

    # Evaluate AWQ baseline
    results['AWQ'] = evaluate_model(
        args.awq_path,
        "AWQ (Baseline)",
        eval_texts,
        device=device
    )

    # Evaluate variance-penalized PRAQ models
    for i, praq_path in enumerate(args.praq_var_paths):
        # Generate name
        if args.praq_var_names and i < len(args.praq_var_names):
            model_name = args.praq_var_names[i]
        else:
            # Try to extract variance penalty from path
            if 'p0' in praq_path or 'var' in praq_path:
                # Extract penalty value from path if possible
                import re
                match = re.search(r'p(\d+\.?\d*)', praq_path)
                if match:
                    penalty = match.group(1)
                    model_name = f"PRAQ-Var (p={penalty})"
                else:
                    model_name = f"PRAQ-Var-{i+1}"
            else:
                model_name = f"PRAQ-Var-{i+1}"

        results[model_name] = evaluate_model(
            praq_path,
            model_name,
            eval_texts,
            device=device
        )

    # Create comparison table
    create_comparison_table(results)

    # Analyze improvements
    analyze_improvements(results, baseline="AWQ")

    # Visualize if requested
    if args.visualize:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")
        visualize_results(results, output_dir=args.output_dir)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
