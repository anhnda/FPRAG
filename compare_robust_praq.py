"""
Compare Robust-PRAQ vs Full-PRAQ vs Original FP16

This script evaluates the impact of noise augmentation on PRAQ quantization:
1. Original FP16 - Baseline (no quantization)
2. Full-PRAQ - Post-activation importance (standard)
3. Robust-PRAQ - Post-activation importance with noise augmentation

Key Questions:
- Does noise augmentation improve robustness?
- How much does Robust-PRAQ help with limited calibration data?
- Is there a quality-robustness tradeoff?
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
    print("COMPARISON TABLE - PRAQ VARIANTS")
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
    print("ANALYSIS")
    print(f"{'='*80}\n")

    # Find baseline (original FP16 model)
    baseline = results_dict.get('Original')
    if not baseline:
        print("⚠️  No baseline (Original) model found for comparison")
        return

    baseline_ppl = baseline['perplexity']
    print(f"Baseline (FP16): Perplexity = {baseline_ppl:.4f}\n")

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

    print("Quantized Model Performance:")
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
        print(f"\n🏆 BEST QUANTIZATION METHOD: {best['name']}")
        print(f"   Perplexity degradation: {best['delta_pct']:+.2f}%")
        print(f"   Quality retention: {100 - best['delta_pct']:.2f}%")

    # Method comparison: Robust-PRAQ vs Full-PRAQ
    print(f"\n{'='*80}")
    print("METHOD COMPARISON: NOISE AUGMENTATION IMPACT")
    print(f"{'='*80}")

    full_praq_ppl = results_dict.get('Full-PRAQ', {}).get('perplexity')
    robust_praq_ppl = results_dict.get('Robust-PRAQ', {}).get('perplexity')

    if full_praq_ppl and robust_praq_ppl:
        delta = robust_praq_ppl - full_praq_ppl
        delta_pct = (delta / full_praq_ppl) * 100

        print(f"\nFull-PRAQ vs Robust-PRAQ:")
        print(f"  Full-PRAQ:   {full_praq_ppl:.4f}")
        print(f"  Robust-PRAQ: {robust_praq_ppl:.4f}")

        if robust_praq_ppl < full_praq_ppl:
            print(f"\n  ✅ Robust-PRAQ wins by {abs(delta_pct):.2f}%")
            print(f"  → Noise augmentation improves quality!")
            print(f"  → Better robustness to calibration set selection")
        elif full_praq_ppl < robust_praq_ppl:
            print(f"\n  ⚠️  Full-PRAQ wins by {abs(delta_pct):.2f}%")
            print(f"  → Noise augmentation may be too aggressive")
            print(f"  → Consider reducing noise_std or n_noise_samples")
        else:
            print(f"\n  🤝 Tie - Both perform equally")

        # Analyze relative to baseline
        full_praq_degradation = ((full_praq_ppl - baseline_ppl) / baseline_ppl) * 100
        robust_praq_degradation = ((robust_praq_ppl - baseline_ppl) / baseline_ppl) * 100

        print(f"\nDegradation from FP16 baseline:")
        print(f"  Full-PRAQ:   {full_praq_degradation:+.2f}%")
        print(f"  Robust-PRAQ: {robust_praq_degradation:+.2f}%")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    print("Method Characteristics:")
    print("  - Original:     FP16 baseline (no quantization)")
    print("  - Full-PRAQ:    Post-activation importance (standard)")
    print("  - Robust-PRAQ:  Post-activation importance + noise augmentation")
    print("\nNoise Augmentation Benefits:")
    print("  ✓ More robust importance estimates with small calibration sets")
    print("  ✓ Averages over multiple noise samples (reduces overfitting)")
    print("  ✓ Better generalization to unseen data")
    print("  ✓ Particularly useful when calibration budget is limited")


def visualize_results(results_dict, output_dir="./visualizations/robust_praq"):
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
            colors.append('#e74c3c')  # Red for Robust-PRAQ
        elif 'Full-PRAQ' in m:
            colors.append('#f39c12')  # Orange for Full-PRAQ
        else:
            colors.append('#3498db')  # Blue for Original

    # Plot 1: Perplexity comparison
    axes[0].bar(range(len(models)), perplexities, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Perplexity (lower is better)')
    axes[0].set_title('Perplexity Comparison - Noise Augmentation Impact')
    axes[0].grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (ppl, model) in enumerate(zip(perplexities, models)):
        axes[0].text(i, ppl + 0.5, f'{ppl:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Perplexity degradation from baseline
    baseline_ppl = valid_results.get('Original', {}).get('perplexity')
    if baseline_ppl:
        degradations = [(valid_results[m]['perplexity'] - baseline_ppl) / baseline_ppl * 100
                       for m in models]
        axes[1].bar(range(len(models)), degradations, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('Perplexity Degradation (%)')
        axes[1].set_title('Quality Loss from FP16 Baseline')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(alpha=0.3, axis='y')

        # Add value labels
        for i, (deg, model) in enumerate(zip(degradations, models)):
            axes[1].text(i, deg + 0.5, f'{deg:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'robust_praq_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved visualization: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Robust-PRAQ vs Full-PRAQ vs Original",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--original-model",
        type=str,
        default="openbmb/MiniCPM-2B-sft-bf16",
        help="Original FP16 model (baseline)"
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
        help="Number of evaluation samples"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comparison visualizations"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("ROBUST-PRAQ COMPARISON")
    print("="*80)
    print("Evaluating impact of noise augmentation on PRAQ:")
    print("  1. Original (FP16) - Baseline")
    print("  2. Full-PRAQ - Standard post-activation importance")
    print("  3. Robust-PRAQ - Post-activation + noise augmentation")
    print("="*80)
    print(f"Device: {device}")
    print(f"Evaluation samples: {args.n_eval}")
    print("="*80)

    # Load validation data
    eval_texts = load_wikitext2_validation(n_samples=args.n_eval)

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
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
