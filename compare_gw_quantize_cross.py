"""
Cross-Dataset Evaluation for Group-Wise Quantization: C4 Validation

This script evaluates group-wise quantized models on C4 dataset for cross-dataset validation:
1. Original FP16 - Baseline (no quantization)
2. GW-AWQ - Group-wise AWQ (pre-activation importance)
3. GW-PRAQ - Group-wise PRAQ (post-activation importance)
4. GWH-PRAQ - Group-wise Hybrid PRAQ (AWQ scaling + PRAQ error weighting)

Key Features:
- Uses C4 dataset instead of WikiText-2 (more diverse, web-crawled data)
- Fixed random seed (42) for reproducibility
- Tests generalization to different data distribution
- Group-wise quantization (group_size=128) for hardware efficiency

Why C4 Cross-Validation:
- WikiText-2: Wikipedia text (formal, structured, similar to training data)
- C4: Web-crawled text (diverse, noisy, real-world scenarios)
- Cross-validation reveals which method generalizes better
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
import pandas as pd


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
        print(f"   Please run the corresponding quantization script first")
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
    print("COMPARISON TABLE - GROUP-WISE QUANTIZATION ON C4")
    print(f"{'='*80}\n")

    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}

    if not valid_results:
        print("❌ No valid results to compare")
        return

    # Create pandas dataframe for easier formatting
    data = []
    for name, result in valid_results.items():
        data.append({
            'Model': name,
            'Perplexity': result['perplexity'],
            'Avg Loss': result['avg_loss'],
            'Size (MB)': result['model_size_mb'],
            'Throughput (tok/s)': result['throughput_tokens_per_sec'],
            'Samples': result['num_samples']
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()


def analyze_comparison(results_dict):
    """Analyze the comparison results."""
    print(f"\n{'='*80}")
    print("CROSS-DATASET ANALYSIS - C4 VALIDATION")
    print(f"{'='*80}\n")

    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}

    # Find baseline (original FP16 model)
    baseline = valid_results.get('Original')
    if not baseline:
        print("⚠️  No baseline (Original) model found for comparison")
        baseline_ppl = None
    else:
        baseline_ppl = baseline['perplexity']
        print(f"Baseline (FP16) on C4: Perplexity = {baseline_ppl:.4f}\n")

    # Compare quantized models
    quantized_models = []
    for name, result in valid_results.items():
        if name != 'Original' and result:
            ppl = result['perplexity']
            if baseline_ppl:
                delta_ppl = ppl - baseline_ppl
                delta_pct = (delta_ppl / baseline_ppl) * 100
            else:
                delta_ppl = None
                delta_pct = None

            quantized_models.append({
                'name': name,
                'perplexity': ppl,
                'delta': delta_ppl,
                'delta_pct': delta_pct,
                'throughput': result['throughput_tokens_per_sec']
            })

    # Sort by perplexity
    quantized_models.sort(key=lambda x: x['perplexity'])

    if baseline_ppl:
        print("Group-Wise Quantized Model Performance on C4:")
        for model in quantized_models:
            if model['delta_pct'] is not None:
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
        print(f"\n🏆 BEST GROUP-WISE METHOD ON C4: {best['name']}")
        if best['delta_pct'] is not None:
            print(f"   Perplexity degradation: {best['delta_pct']:+.2f}%")
            print(f"   Quality retention: {100 - best['delta_pct']:.2f}%")

    # Head-to-head comparisons
    print(f"\n{'='*80}")
    print("HEAD-TO-HEAD COMPARISONS ON C4")
    print(f"{'='*80}")

    gw_awq = valid_results.get('GW-AWQ')
    gw_praq = valid_results.get('GW-PRAQ')
    gwh_praq = valid_results.get('GWH-PRAQ')

    # GW-AWQ vs GW-PRAQ
    if gw_awq and gw_praq:
        awq_ppl = gw_awq['perplexity']
        praq_ppl = gw_praq['perplexity']
        delta_ppl = praq_ppl - awq_ppl
        delta_pct = (delta_ppl / awq_ppl) * 100

        print(f"\nGW-AWQ vs GW-PRAQ on C4:")
        print(f"  GW-AWQ:  {awq_ppl:.4f}")
        print(f"  GW-PRAQ: {praq_ppl:.4f}")
        print(f"  Delta:   {delta_ppl:+.4f} ({delta_pct:+.2f}%)")

        if abs(delta_pct) < 1.0:
            print(f"\n  🤝 Essentially tied (< 1% difference)")
            print(f"  → Both methods generalize well to C4!")
        elif praq_ppl < awq_ppl:
            print(f"\n  ✅ GW-PRAQ wins by {abs(delta_pct):.2f}%")
            print(f"  → Post-activation importance generalizes better to real-world data!")
        else:
            print(f"\n  ❌ GW-AWQ wins by {abs(delta_pct):.2f}%")
            print(f"  → Pre-activation salience more robust on diverse data")

    # GW-PRAQ vs GWH-PRAQ
    if gw_praq and gwh_praq:
        praq_ppl = gw_praq['perplexity']
        hybrid_ppl = gwh_praq['perplexity']
        delta_ppl = hybrid_ppl - praq_ppl
        delta_pct = (delta_ppl / praq_ppl) * 100

        print(f"\nGW-PRAQ vs GWH-PRAQ on C4:")
        print(f"  GW-PRAQ:  {praq_ppl:.4f}")
        print(f"  GWH-PRAQ: {hybrid_ppl:.4f}")
        print(f"  Delta:    {delta_ppl:+.4f} ({delta_pct:+.2f}%)")

        if abs(delta_pct) < 0.5:
            print(f"\n  🤝 Essentially tied")
        elif hybrid_ppl < praq_ppl:
            print(f"\n  ✅ Hybrid wins by {abs(delta_pct):.2f}%")
            print(f"  → Combining AWQ scaling with PRAQ weighting helps on C4!")
        else:
            print(f"\n  ⚠️  Pure PRAQ wins by {abs(delta_pct):.2f}%")
            print(f"  → Hybrid approach doesn't improve on C4")

    # GW-AWQ vs GWH-PRAQ
    if gw_awq and gwh_praq:
        awq_ppl = gw_awq['perplexity']
        hybrid_ppl = gwh_praq['perplexity']
        delta_ppl = hybrid_ppl - awq_ppl
        delta_pct = (delta_ppl / awq_ppl) * 100

        print(f"\nGW-AWQ vs GWH-PRAQ on C4:")
        print(f"  GW-AWQ:   {awq_ppl:.4f}")
        print(f"  GWH-PRAQ: {hybrid_ppl:.4f}")
        print(f"  Delta:    {delta_ppl:+.4f} ({delta_pct:+.2f}%)")

        if abs(delta_pct) < 0.5:
            print(f"\n  🤝 Essentially tied")
        elif hybrid_ppl < awq_ppl:
            print(f"\n  ✅ Hybrid wins by {abs(delta_pct):.2f}%")
            print(f"  → Hybrid is better than pure AWQ on C4!")
        else:
            print(f"\n  ⚠️  AWQ wins by {abs(delta_pct):.2f}%")

    # Overall winner among all methods
    if len(quantized_models) >= 3:
        print(f"\n{'='*80}")
        print("OVERALL BEST METHOD ON C4")
        print(f"{'='*80}")

        best_overall = min(quantized_models, key=lambda x: x['perplexity'])
        print(f"\n🏆 Winner: {best_overall['name']}")
        print(f"   Perplexity: {best_overall['perplexity']:.4f}")
        if best_overall['delta_pct'] is not None:
            print(f"   Degradation: {best_overall['delta_pct']:+.2f}%")
            print(f"   Throughput: {best_overall['throughput']:.1f} tokens/sec")

    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS - C4 CROSS-VALIDATION")
    print(f"{'='*80}")
    print("\nDataset Characteristics:")
    print("  - WikiText-2: Wikipedia text (formal, structured)")
    print("  - C4: Web-crawled text (diverse, noisy, real-world)")
    print("\nWhy Cross-Dataset Validation Matters:")
    print("  ✓ Tests generalization beyond calibration distribution")
    print("  ✓ Reveals overfitting to WikiText-2 patterns")
    print("  ✓ Simulates real-world deployment scenarios")
    print("  ✓ More rigorous evaluation of quantization quality")
    print("\nGroup-Wise Quantization Benefits:")
    print("  ✓ Hardware-efficient (group_size=128)")
    print("  ✓ Better memory access patterns for GPU")
    print("  ✓ Closer to real INT4 deployment")
    print("  ✓ Practical for production use")


def visualize_results(results_dict, output_dir="./visualizations/gw_c4_validation"):
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
        if 'GWH-PRAQ' in m:
            colors.append('#e74c3c')  # Red for Hybrid
        elif 'GW-PRAQ' in m:
            colors.append('#f39c12')  # Orange for PRAQ
        elif 'GW-AWQ' in m:
            colors.append('#2ecc71')  # Green for AWQ
        else:
            colors.append('#3498db')  # Blue for Original

    # Plot 1: Perplexity comparison
    axes[0].bar(range(len(models)), perplexities, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Perplexity (lower is better)')
    axes[0].set_title('C4 Validation - Group-Wise Quantization Comparison')
    axes[0].grid(alpha=0.3, axis='y')

    # Add value labels
    for i, (ppl, model) in enumerate(zip(perplexities, models)):
        axes[0].text(i, ppl + max(perplexities) * 0.02, f'{ppl:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

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
    save_path = os.path.join(output_dir, 'gw_c4_validation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved visualization: {save_path}")


def save_results_csv(results_dict, output_dir="./visualizations/gw_c4_validation"):
    """Save results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    valid_results = {k: v for k, v in results_dict.items() if v is not None}

    data = []
    for name, result in valid_results.items():
        data.append({
            'Model': name,
            'Perplexity': result['perplexity'],
            'Avg_Loss': result['avg_loss'],
            'Model_Size_MB': result['model_size_mb'],
            'Throughput_Tokens_Per_Sec': result['throughput_tokens_per_sec'],
            'Num_Samples': result['num_samples'],
            'Failed_Samples': result.get('failed_samples', 0)
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'gw_c4_validation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results to CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Evaluation for Group-Wise Quantization on C4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--original-model",
        type=str,
        default="openbmb/MiniCPM-2B-sft-bf16",
        help="Original FP16 model (baseline)"
    )
    parser.add_argument(
        "--gw-awq-path",
        type=str,
        default="./quantized_models/minicpm_gw_awq",
        help="Path to Group-Wise AWQ quantized model"
    )
    parser.add_argument(
        "--gw-praq-path",
        type=str,
        default="./quantized_models/minicpm_gw_praq",
        help="Path to Group-Wise PRAQ quantized model"
    )
    parser.add_argument(
        "--gwh-praq-path",
        type=str,
        default="./quantized_models/minicpm_gwh_praq",
        help="Path to Hybrid Group-Wise PRAQ quantized model"
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
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results to CSV file"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("CROSS-DATASET EVALUATION: GROUP-WISE QUANTIZATION ON C4")
    print("="*80)
    print("Comparing group-wise quantization methods on diverse web-crawled data:")
    print("  1. Original (FP16) - Baseline")
    print("  2. GW-AWQ - Group-wise + pre-activation importance")
    print("  3. GW-PRAQ - Group-wise + post-activation risk-aware importance")
    print("  4. GWH-PRAQ - Hybrid (AWQ scaling + PRAQ error weighting)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Evaluation dataset: C4 (Colossal Clean Crawled Corpus)")
    print(f"Evaluation samples: {args.n_eval}")
    print(f"Random seed: {args.seed} (fixed for reproducibility)")
    print(f"Group size: 128 (hardware-efficient)")
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

    # Evaluate GW-AWQ
    results['GW-AWQ'] = evaluate_model(
        args.gw_awq_path,
        "GW-AWQ",
        eval_texts,
        device=device
    )

    # Evaluate GW-PRAQ
    results['GW-PRAQ'] = evaluate_model(
        args.gw_praq_path,
        "GW-PRAQ",
        eval_texts,
        device=device
    )

    # Evaluate GWH-PRAQ
    results['GWH-PRAQ'] = evaluate_model(
        args.gwh_praq_path,
        "GWH-PRAQ",
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

    # Save CSV if requested
    if args.save_csv:
        print(f"\n{'='*80}")
        print("SAVING RESULTS TO CSV")
        print(f"{'='*80}")
        save_results_csv(results)

    print(f"\n{'='*80}")
    print("C4 CROSS-VALIDATION COMPLETE!")
    print(f"{'='*80}")
    print("\nKey Takeaways:")
    print("  ✓ C4 provides more rigorous evaluation than WikiText-2 alone")
    print("  ✓ Tests generalization to real-world diverse, noisy text")
    print("  ✓ Reveals which method is more robust across distributions")
    print("  ✓ Group-wise quantization (128) balances quality and hardware efficiency")
    print("  ✓ Fixed seed ensures reproducible cross-validation results")
    print("\nNext Steps:")
    print("  - Compare C4 results with WikiText-2 results")
    print("  - Identify which method generalizes better")
    print("  - Consider deployment based on cross-validation performance")
    print("="*80)


if __name__ == "__main__":
    main()
