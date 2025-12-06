"""
Cross-Dataset Evaluation: C4 Validation

This script evaluates quantized models on C4 dataset for cross-dataset validation:
1. Original FP16 - Baseline (no quantization)
2. Full-AWQ - Pre-activation importance
3. Full-PRAQ - Post-activation importance
4. Robust-PRAQ - Post-activation + noise augmentation

Key Differences from standard comparison:
- Uses C4 dataset instead of WikiText-2 (more diverse, web-crawled data)
- Fixed random seed (42) for reproducibility
- 2000 validation samples
- Tests generalization to different data distribution
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


def load_c4_validation(n_samples=2000, seed=42):
    """
    Load C4 validation set with fixed random seed.

    C4 (Colossal Clean Crawled Corpus) is a large web-crawled dataset
    that's more diverse than WikiText-2, making it a good test of
    generalization to different data distributions.
    """
    print(f"Loading C4 validation dataset...")

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load C4 validation set (en subset)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    # Collect samples
    texts = []
    print(f"Collecting {n_samples} samples from C4 validation (seed={seed})...")

    for i, item in enumerate(tqdm(dataset, desc="Loading C4", total=n_samples)):
        if len(texts) >= n_samples:
            break

        text = item['text']
        # Filter out very short texts
        if len(text.strip()) > 100:
            texts.append(text)

    print(f"Loaded {len(texts)} samples from C4 validation")

    # Shuffle with fixed seed for consistent sampling
    random.seed(seed)
    random.shuffle(texts)

    return texts[:n_samples]


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, texts, max_length=512, device="cuda"):
    """Evaluate perplexity on a list of text samples."""
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

            # Forward pass
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


def evaluate_model(model_path, model_name, eval_texts, device="cuda", is_huggingface=False):
    """Evaluate a single model and return results."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    # Check if it's a HuggingFace model or local path
    if not is_huggingface and not os.path.exists(model_path):
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
    print("COMPARISON TABLE - C4 VALIDATION (Cross-Dataset)")
    print(f"{'='*80}\n")

    # Header
    models = list(results_dict.keys())
    header = f"{'Metric':<30}"
    for model_name in models:
        header += f" {model_name:<20}"
    print(header)
    print("-" * (30 + 22 * len(models)))

    # Perplexity
    row = f"{'Perplexity':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['perplexity']:>20.4f}"
        else:
            row += f" {'N/A':>20}"
    print(row)

    # Avg Loss
    row = f"{'Avg Loss':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['avg_loss']:>20.4f}"
        else:
            row += f" {'N/A':>20}"
    print(row)

    # Model Size
    row = f"{'Model Size (MB)':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['model_size_mb']:>20.2f}"
        else:
            row += f" {'N/A':>20}"
    print(row)

    # Throughput
    row = f"{'Throughput (tok/s)':<30}"
    for model_name in models:
        if results_dict[model_name]:
            row += f" {results_dict[model_name]['throughput_tokens_per_sec']:>20.2f}"
        else:
            row += f" {'N/A':>20}"
    print(row)

    print("-" * (30 + 22 * len(models)))


def analyze_comparison(results_dict):
    """Analyze the comparison results."""
    print(f"\n{'='*80}")
    print("CROSS-DATASET ANALYSIS (C4 Validation)")
    print(f"{'='*80}\n")

    # Find baseline (original FP16 model)
    baseline = results_dict.get('Original')
    if not baseline:
        print("⚠️  No baseline (Original) model found for comparison")
        return

    baseline_ppl = baseline['perplexity']
    print(f"Baseline (FP16) on C4: Perplexity = {baseline_ppl:.4f}\n")

    # Compare quantized models
    quantized_models = []
    for name, result in results_dict.items():
        if name != 'Original' and result:
            ppl = result['perplexity']
            delta_ppl = ppl - baseline_ppl
            delta_pct = (delta_ppl / baseline_ppl) * 100
            quantized_models.append({
                'name': name,
                'perplexity': ppl,
                'delta': delta_ppl,
                'delta_pct': delta_pct
            })

    # Sort by perplexity
    quantized_models.sort(key=lambda x: x['perplexity'])

    print("Quantized Model Performance on C4:")
    for model in quantized_models:
        sign = "+" if model['delta'] > 0 else ""
        if model['delta_pct'] < 5.0:
            symbol = "✅"  # Excellent
        elif model['delta_pct'] < 10.0:
            symbol = "🟢"  # Good
        elif model['delta_pct'] < 20.0:
            symbol = "🟡"  # Acceptable
        else:
            symbol = "❌"  # Poor

        print(f"{symbol} {model['name']:<20} Perplexity: {model['perplexity']:>8.4f} "
              f"(Δ={sign}{model['delta']:>6.4f}, {sign}{model['delta_pct']:>6.2f}%)")

    # Winner
    if quantized_models:
        best = quantized_models[0]
        print(f"\n🏆 BEST METHOD ON C4: {best['name']}")
        print(f"   Perplexity degradation: {best['delta_pct']:+.2f}%")
        print(f"   Quality retention: {100 - best['delta_pct']:.2f}%")

    # Method comparison
    print(f"\n{'='*80}")
    print("METHOD COMPARISON ON C4")
    print(f"{'='*80}")

    full_awq_ppl = results_dict.get('Full-AWQ', {}).get('perplexity')
    full_praq_ppl = results_dict.get('Full-PRAQ', {}).get('perplexity')
    robust_praq_ppl = results_dict.get('Robust-PRAQ', {}).get('perplexity')

    if full_awq_ppl and full_praq_ppl:
        delta = ((full_praq_ppl - full_awq_ppl) / full_awq_ppl) * 100

        print(f"\nFull-AWQ vs Full-PRAQ:")
        print(f"  Full-AWQ:  {full_awq_ppl:.4f}")
        print(f"  Full-PRAQ: {full_praq_ppl:.4f}")

        if full_praq_ppl < full_awq_ppl:
            print(f"\n  ✅ Full-PRAQ wins by {abs(delta):.2f}%")
            print(f"  → Post-activation importance generalizes better!")
        elif full_awq_ppl < full_praq_ppl:
            print(f"\n  ❌ Full-AWQ wins by {abs(delta):.2f}%")
            print(f"  → Pre-activation salience more robust")
        else:
            print(f"\n  🤝 Tie")

    if full_praq_ppl and robust_praq_ppl:
        delta = ((robust_praq_ppl - full_praq_ppl) / full_praq_ppl) * 100

        print(f"\nFull-PRAQ vs Robust-PRAQ:")
        print(f"  Full-PRAQ:   {full_praq_ppl:.4f}")
        print(f"  Robust-PRAQ: {robust_praq_ppl:.4f}")

        if robust_praq_ppl < full_praq_ppl:
            print(f"\n  ✅ Robust-PRAQ wins by {abs(delta):.2f}%")
            print(f"  → Noise augmentation helps on C4!")
        elif full_praq_ppl < robust_praq_ppl:
            print(f"\n  ⚠️  Full-PRAQ wins by {abs(delta):.2f}%")
            print(f"  → Noise augmentation hurts on C4")
        else:
            print(f"\n  🤝 Tie")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS - CROSS-DATASET VALIDATION")
    print(f"{'='*80}")
    print("Dataset Characteristics:")
    print("  - WikiText-2: Wikipedia text (formal, structured)")
    print("  - C4: Web-crawled text (diverse, noisy, real-world)")
    print("\nWhy Cross-Dataset Validation Matters:")
    print("  ✓ Tests generalization beyond training distribution")
    print("  ✓ Reveals overfitting to calibration data")
    print("  ✓ Simulates real-world deployment scenarios")
    print("  ✓ More rigorous evaluation of quantization quality")


def visualize_results(results_dict, output_dir="./visualizations/c4_validation"):
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

    # Color scheme
    colors = []
    for m in models:
        if 'Robust-PRAQ' in m:
            colors.append('#e74c3c')  # Red
        elif 'Full-PRAQ' in m:
            colors.append('#f39c12')  # Orange
        elif 'AWQ' in m:
            colors.append('#2ecc71')  # Green
        else:
            colors.append('#3498db')  # Blue

    # Plot 1: Perplexity comparison
    axes[0].bar(range(len(models)), perplexities, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Perplexity (lower is better)')
    axes[0].set_title('C4 Validation - Perplexity Comparison')
    axes[0].grid(alpha=0.3, axis='y')

    # Add value labels
    for i, (ppl, model) in enumerate(zip(perplexities, models)):
        axes[0].text(i, ppl + 0.5, f'{ppl:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # Plot 2: Degradation from baseline
    baseline_ppl = valid_results.get('Original', {}).get('perplexity')
    if baseline_ppl:
        degradations = [(valid_results[m]['perplexity'] - baseline_ppl) / baseline_ppl * 100
                       for m in models]
        axes[1].bar(range(len(models)), degradations, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('Perplexity Degradation (%)')
        axes[1].set_title('Quality Loss on C4 (vs FP16 Baseline)')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(alpha=0.3, axis='y')

        # Add value labels
        for i, (deg, model) in enumerate(zip(degradations, models)):
            axes[1].text(i, deg + 0.5, f'{deg:+.1f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'c4_validation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved visualization: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Evaluation on C4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--original-model",
        type=str,
        default="openbmb/MiniCPM-2B-sft-bf16",
        help="Original FP16 model (baseline)"
    )
    parser.add_argument(
        "--full-awq-path",
        type=str,
        default="./quantized_models/minicpm_full_awq",
        help="Path to Full-AWQ quantized model"
    )
    parser.add_argument(
        "--full-praq-path",
        type=str,
        default="./quantized_models/minicpm_full_praq",
        help="Path to Full-PRAQ quantized model"
    )
    parser.add_argument(
        "--robust-praq-path",
        type=str,
        default="./quantized_models/minicpm_robust_praq",
        help="Path to Robust-PRAQ quantized model"
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=2000,
        help="Number of C4 validation samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comparison visualizations"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("CROSS-DATASET EVALUATION: C4 VALIDATION")
    print("="*80)
    print("Comparing quantization methods on diverse web-crawled data:")
    print("  1. Original (FP16) - Baseline")
    print("  2. Full-AWQ - Pre-activation importance")
    print("  3. Full-PRAQ - Post-activation importance")
    print("  4. Robust-PRAQ - Post-activation + noise augmentation")
    print("="*80)
    print(f"Device: {device}")
    print(f"Evaluation dataset: C4 (Colossal Clean Crawled Corpus)")
    print(f"Evaluation samples: {args.n_eval}")
    print(f"Random seed: {args.seed} (fixed for reproducibility)")
    print("="*80)

    # Load C4 validation data
    eval_texts = load_c4_validation(n_samples=args.n_eval, seed=args.seed)

    # Results storage
    results = {}

    # Evaluate original model (baseline)
    results['Original'] = evaluate_model(
        args.original_model,
        "Original (FP16)",
        eval_texts,
        device=device,
        is_huggingface=True
    )

    # Evaluate Full-AWQ
    results['Full-AWQ'] = evaluate_model(
        args.full_awq_path,
        "Full-AWQ",
        eval_texts,
        device=device
    )

    # Evaluate Full-PRAQ
    results['Full-PRAQ'] = evaluate_model(
        args.full_praq_path,
        "Full-PRAQ",
        eval_texts,
        device=device
    )

    # Evaluate Robust-PRAQ
    results['Robust-PRAQ'] = evaluate_model(
        args.robust_praq_path,
        "Robust-PRAQ",
        eval_texts,
        device=device
    )

    # Create comparison table
    create_comparison_table(results)

    # Analyze results
    analyze_comparison(results)

    # Visualize if requested
    if args.visualize:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")
        visualize_results(results)

    print(f"\n{'='*80}")
    print("C4 VALIDATION COMPLETE!")
    print(f"{'='*80}")
    print("\nCross-dataset validation provides:")
    print("  ✓ More rigorous evaluation than single dataset")
    print("  ✓ Tests generalization to real-world diverse text")
    print("  ✓ Reveals method robustness across distributions")
    print("  ✓ Fixed seed ensures reproducible results")


if __name__ == "__main__":
    main()
