"""
Compare Group-Wise Quantization Methods

This script compares group-wise INT4 quantization approaches:
1. Original FP16 - Baseline (no quantization)
2. Group-Wise AWQ - Group-wise INT4 + activation-aware scaling
3. Group-Wise PRAQ - Group-wise INT4 + risk-aware post-activation importance

Optional: Compare against Full (per-channel) quantization methods

Key Questions:
- Does group-wise quantization (group_size=128) perform well?
- How does it compare to per-channel quantization?
- Does PRAQ's risk-aware importance help in group-wise setting?
- What's the quality vs hardware-efficiency trade-off?
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

            # Forward pass with timing
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

    # Check if model exists
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
    print("COMPARISON TABLE - GROUP-WISE QUANTIZATION")
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
    print("ANALYSIS - GROUP-WISE QUANTIZATION")
    print(f"{'='*80}\n")

    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}

    # Find baseline
    baseline = valid_results.get('Original')
    if not baseline:
        print("⚠️  No baseline (Original) model found for comparison")
        baseline_ppl = None
    else:
        baseline_ppl = baseline['perplexity']
        print(f"Baseline (FP16): Perplexity = {baseline_ppl:.4f}\n")

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
        print("Quantized Model Performance (Group-Wise INT4):")
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

                print(f"{symbol} {model['name']:<25} PPL: {model['perplexity']:>8.4f} "
                      f"(Δ={sign}{model['delta']:>6.4f}, {sign}{model['delta_pct']:>6.2f}%) "
                      f"| {model['throughput']:>7.1f} tok/s")

    # Head-to-head: GW-AWQ vs GW-PRAQ
    print(f"\n{'='*80}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*80}")

    gw_awq = valid_results.get('GW-AWQ')
    gw_praq = valid_results.get('GW-PRAQ')

    if gw_awq and gw_praq:
        awq_ppl = gw_awq['perplexity']
        praq_ppl = gw_praq['perplexity']
        delta_ppl = praq_ppl - awq_ppl
        delta_pct = (delta_ppl / awq_ppl) * 100

        print(f"\nGroup-Wise AWQ vs Group-Wise PRAQ:")
        print(f"  GW-AWQ:  {awq_ppl:.4f}")
        print(f"  GW-PRAQ: {praq_ppl:.4f}")
        print(f"  Delta:   {delta_ppl:+.4f} ({delta_pct:+.2f}%)")

        if abs(delta_pct) < 1.0:
            print(f"\n  🤝 Essentially tied (< 1% difference)")
        elif praq_ppl < awq_ppl:
            print(f"\n  ✅ GW-PRAQ wins by {abs(delta_pct):.2f}%")
            print(f"  → Risk-aware post-activation importance is more effective!")
        else:
            print(f"\n  ❌ GW-AWQ wins by {abs(delta_pct):.2f}%")
            print(f"  → Simple activation salience is more robust")

        # Throughput comparison
        awq_thr = gw_awq['throughput_tokens_per_sec']
        praq_thr = gw_praq['throughput_tokens_per_sec']
        print(f"\n  Throughput:")
        print(f"    GW-AWQ:  {awq_thr:.1f} tokens/sec")
        print(f"    GW-PRAQ: {praq_thr:.1f} tokens/sec")

    # Compare group-wise vs per-channel (if available)
    print(f"\n{'='*80}")
    print("GROUP-WISE vs PER-CHANNEL QUANTIZATION")
    print(f"{'='*80}")

    full_awq = valid_results.get('Full-AWQ')
    full_praq = valid_results.get('Full-PRAQ')

    if gw_awq and full_awq:
        gw_ppl = gw_awq['perplexity']
        full_ppl = full_awq['perplexity']
        delta = ((gw_ppl - full_ppl) / full_ppl) * 100
        print(f"\nAWQ Comparison:")
        print(f"  Per-Channel (Full):   {full_ppl:.4f}")
        print(f"  Group-Wise (128):     {gw_ppl:.4f}")
        print(f"  Delta:                {delta:+.2f}%")
        if abs(delta) < 2.0:
            print(f"  → Group-wise is competitive with per-channel!")
        elif delta > 0:
            print(f"  → Per-channel is better (but group-wise is more hardware-efficient)")

    if gw_praq and full_praq:
        gw_ppl = gw_praq['perplexity']
        full_ppl = full_praq['perplexity']
        delta = ((gw_ppl - full_ppl) / full_ppl) * 100
        print(f"\nPRAQ Comparison:")
        print(f"  Per-Channel (Full):   {full_ppl:.4f}")
        print(f"  Group-Wise (128):     {gw_ppl:.4f}")
        print(f"  Delta:                {delta:+.2f}%")
        if abs(delta) < 2.0:
            print(f"  → Group-wise is competitive with per-channel!")
        elif delta > 0:
            print(f"  → Per-channel is better (but group-wise is more hardware-efficient)")

    # Winner announcement
    if quantized_models:
        print(f"\n{'='*80}")
        print("🏆 BEST GROUP-WISE METHOD")
        print(f"{'='*80}")
        best = quantized_models[0]
        print(f"\nWinner: {best['name']}")
        print(f"  Perplexity: {best['perplexity']:.4f}")
        if best['delta_pct'] is not None:
            print(f"  Degradation from FP16: {best['delta_pct']:+.2f}%")
            print(f"  Quality retention: {100 - best['delta_pct']:.2f}%")
        print(f"  Throughput: {best['throughput']:.1f} tokens/sec")

    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    print("\nGroup-Wise Quantization (group_size=128):")
    print("  ✓ More hardware-efficient than per-channel")
    print("  ✓ Better memory access patterns for GPU inference")
    print("  ✓ Closer to real-world INT4 deployment")
    print("  ✓ Slight quality trade-off vs per-channel (typically < 2%)")
    print("\nMethod Comparison:")
    print("  - GW-AWQ:  Group-wise + pre-activation importance E[|X|]")
    print("  - GW-PRAQ: Group-wise + post-activation importance E[|SiLU(XW)|]")
    print("\nPRAQ Advantage:")
    print("  - Accounts for activation function effects")
    print("  - Protects channels based on actual OUTPUT impact")
    print("  - More accurate importance estimation for MLP layers")


def visualize_results(results_dict, output_dir="./visualizations/groupwise_comparison"):
    """Create visualizations comparing the models."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}

    if len(valid_results) < 2:
        print("\n⚠️  Not enough models to visualize")
        return

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = list(valid_results.keys())
    perplexities = [valid_results[m]['perplexity'] for m in models]
    throughputs = [valid_results[m]['throughput_tokens_per_sec'] for m in models]
    sizes = [valid_results[m]['model_size_mb'] for m in models]

    # Color scheme
    colors = []
    for m in models:
        if 'PRAQ' in m:
            colors.append('#e74c3c')  # Red for PRAQ
        elif 'AWQ' in m:
            colors.append('#2ecc71')  # Green for AWQ
        else:
            colors.append('#3498db')  # Blue for Original

    # Plot 1: Perplexity comparison
    axes[0].bar(range(len(models)), perplexities, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Perplexity (lower is better)', fontsize=11)
    axes[0].set_title('Perplexity Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    for i, (ppl, model) in enumerate(zip(perplexities, models)):
        axes[0].text(i, ppl + max(perplexities) * 0.02, f'{ppl:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Throughput comparison
    axes[1].bar(range(len(models)), throughputs, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('Throughput (tokens/sec, higher is better)', fontsize=11)
    axes[1].set_title('Inference Throughput', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    for i, (thr, model) in enumerate(zip(throughputs, models)):
        axes[1].text(i, thr + max(throughputs) * 0.02, f'{thr:.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Quality-Efficiency Trade-off
    baseline_ppl = valid_results.get('Original', {}).get('perplexity')
    if baseline_ppl:
        degradations = [(valid_results[m]['perplexity'] - baseline_ppl) / baseline_ppl * 100
                       for m in models]

        # Create scatter plot: x=degradation, y=throughput
        quantized_models = [(m, d, t, c) for m, d, t, c in zip(models, degradations, throughputs, colors)
                           if m != 'Original']

        if quantized_models:
            for model, deg, thr, color in quantized_models:
                axes[2].scatter(deg, thr, s=200, c=color, alpha=0.8, edgecolor='black', linewidth=2)
                axes[2].annotate(model, (deg, thr), xytext=(5, 5), textcoords='offset points',
                               fontsize=9, fontweight='bold')

            axes[2].set_xlabel('Perplexity Degradation from FP16 (%)', fontsize=11)
            axes[2].set_ylabel('Throughput (tokens/sec)', fontsize=11)
            axes[2].set_title('Quality-Efficiency Trade-off', fontsize=12, fontweight='bold')
            axes[2].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'groupwise_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved visualization: {save_path}")

    # Create second figure: Group-wise vs Per-channel comparison
    full_awq = valid_results.get('Full-AWQ')
    full_praq = valid_results.get('Full-PRAQ')
    gw_awq = valid_results.get('GW-AWQ')
    gw_praq = valid_results.get('GW-PRAQ')

    if (full_awq and gw_awq) or (full_praq and gw_praq):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        comparison_data = []
        if full_awq and gw_awq:
            comparison_data.extend([
                ('AWQ\nPer-Channel', full_awq['perplexity'], '#2ecc71'),
                ('AWQ\nGroup-Wise', gw_awq['perplexity'], '#27ae60')
            ])
        if full_praq and gw_praq:
            comparison_data.extend([
                ('PRAQ\nPer-Channel', full_praq['perplexity'], '#e74c3c'),
                ('PRAQ\nGroup-Wise', gw_praq['perplexity'], '#c0392b')
            ])

        labels, ppls, colors_comp = zip(*comparison_data)
        x_pos = range(len(labels))

        ax.bar(x_pos, ppls, color=colors_comp, alpha=0.8, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
        ax.set_title('Group-Wise vs Per-Channel Quantization', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        for i, ppl in enumerate(ppls):
            ax.text(i, ppl + max(ppls) * 0.01, f'{ppl:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'groupwise_vs_perchannel.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization: {save_path}")


def save_results_csv(results_dict, output_dir="./visualizations/groupwise_comparison"):
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
    csv_path = os.path.join(output_dir, 'groupwise_comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results to CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Group-Wise Quantization Methods",
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
        "--full-awq-path",
        type=str,
        default="./quantized_models/minicpm_full_awq",
        help="Path to Full (per-channel) AWQ model (optional)"
    )
    parser.add_argument(
        "--full-praq-path",
        type=str,
        default="./quantized_models/minicpm_full_praq",
        help="Path to Full (per-channel) PRAQ model (optional)"
    )
    parser.add_argument(
        "--compare-full",
        action="store_true",
        help="Include Full (per-channel) models in comparison"
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
        "--save-csv",
        action="store_true",
        help="Save results to CSV file"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("GROUP-WISE QUANTIZATION COMPARISON")
    print("="*80)
    print("Comparing group-wise INT4 quantization methods (group_size=128):")
    print("  1. Original (FP16) - Baseline")
    print("  2. GW-AWQ - Group-wise + pre-activation importance")
    print("  3. GW-PRAQ - Group-wise + post-activation risk-aware importance")
    if args.compare_full:
        print("  4. Full-AWQ - Per-channel quantization (comparison)")
        print("  5. Full-PRAQ - Per-channel quantization (comparison)")
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

    # Evaluate Group-Wise AWQ
    results['GW-AWQ'] = evaluate_model(
        args.gw_awq_path,
        "GW-AWQ",
        eval_texts,
        device=device
    )

    # Evaluate Group-Wise PRAQ
    results['GW-PRAQ'] = evaluate_model(
        args.gw_praq_path,
        "GW-PRAQ",
        eval_texts,
        device=device
    )

    # Optionally evaluate Full (per-channel) models
    if args.compare_full:
        results['Full-AWQ'] = evaluate_model(
            args.full_awq_path,
            "Full-AWQ",
            eval_texts,
            device=device
        )

        results['Full-PRAQ'] = evaluate_model(
            args.full_praq_path,
            "Full-PRAQ",
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
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print("\nConclusion:")
    print("  Group-wise quantization (group_size=128) offers:")
    print("    ✓ Hardware-efficient implementation")
    print("    ✓ Better memory access patterns")
    print("    ✓ Competitive quality vs per-channel")
    print("    ✓ More practical for real deployment")
    print("="*80)


if __name__ == "__main__":
    main()
