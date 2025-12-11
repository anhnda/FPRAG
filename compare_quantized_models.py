"""
Simple Comparison Script: Load and Compare Pre-Quantized Models

This script LOADS already-quantized models and compares their perplexity.
No re-quantization is performed.

Usage:
    python compare_quantized_models.py \
        --model1-path ./quantized_models/minicpm_gw_awq_asym_l2 \
        --model1-name "E[X²]" \
        --model2-path ./quantized_models/minicpm_gw_awq_asym_eps_l2 \
        --model2-name "E[X²]+ε*mean" \
        --n-eval 500
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
import os


def compute_perplexity(model, tokenizer, texts, device, max_samples=500):
    """
    Compute perplexity on a given set of texts.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    print(f"Computing perplexity on {min(max_samples, len(texts))} samples...")

    with torch.no_grad():
        for text in tqdm(texts[:max_samples], desc="Evaluating"):
            if len(text.strip()) == 0:
                continue

            try:
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=False
                ).to(device)

                if inputs.input_ids.shape[1] < 2:
                    continue

                outputs = model(**inputs, use_cache=False, labels=inputs.input_ids)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  Warning: Invalid loss detected (NaN/Inf), skipping sample")
                    continue

                total_loss += loss.item() * inputs.input_ids.shape[1]
                total_tokens += inputs.input_ids.shape[1]

            except Exception as e:
                print(f"\n⚠️  Error processing sample: {e}")
                continue

    if total_tokens == 0:
        print("⚠️  No valid tokens processed!")
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def get_model_size(model):
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def load_wikitext2(split="validation", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Compare pre-quantized models by loading and evaluating them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model1-path", type=str, required=True,
                       help="Path to first quantized model")
    parser.add_argument("--model1-name", type=str, default="Model 1",
                       help="Display name for first model")
    parser.add_argument("--model2-path", type=str, required=True,
                       help="Path to second quantized model")
    parser.add_argument("--model2-name", type=str, default="Model 2",
                       help="Display name for second model")
    parser.add_argument("--original-model", type=str, default=None,
                       help="Optional: Path or name of original FP16 model for reference")
    parser.add_argument("--n-eval", type=int, default=500,
                       help="Number of evaluation samples")
    parser.add_argument("--output-file", type=str, default="comparison_results.txt",
                       help="Output file for results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("COMPARISON: Pre-Quantized Models")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model 1: {args.model1_name} ({args.model1_path})")
    print(f"Model 2: {args.model2_name} ({args.model2_path})")
    if args.original_model:
        print(f"Original model: {args.original_model}")
    print(f"Evaluation samples: {args.n_eval}")
    print("=" * 80)

    # Load evaluation data
    eval_texts = load_wikitext2(split="validation", n_samples=args.n_eval)

    results = {}

    # Evaluate original model if provided
    if args.original_model:
        print("\n" + "=" * 80)
        print(f"Loading Original Model: {args.original_model}")
        print("=" * 80)

        try:
            tokenizer_orig = AutoTokenizer.from_pretrained(args.original_model, trust_remote_code=True)
            model_orig = AutoModelForCausalLM.from_pretrained(
                args.original_model,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )

            size_orig = get_model_size(model_orig)
            print(f"Model size: {size_orig:.2f} MB")

            ppl_orig = compute_perplexity(model_orig, tokenizer_orig, eval_texts, device, args.n_eval)
            print(f"Perplexity: {ppl_orig:.2f}")

            results['original'] = {'ppl': ppl_orig, 'size_mb': size_orig}

            del model_orig, tokenizer_orig
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️  Error loading original model: {e}")

    # Evaluate Model 1
    print("\n" + "=" * 80)
    print(f"Loading Model 1: {args.model1_name}")
    print("=" * 80)

    try:
        if not os.path.exists(args.model1_path):
            print(f"❌ Error: Model path not found: {args.model1_path}")
            return

        tokenizer1 = AutoTokenizer.from_pretrained(args.model1_path, trust_remote_code=True)
        model1 = AutoModelForCausalLM.from_pretrained(
            args.model1_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        size1 = get_model_size(model1)
        print(f"Model size: {size1:.2f} MB")

        ppl1 = compute_perplexity(model1, tokenizer1, eval_texts, device, args.n_eval)
        print(f"Perplexity: {ppl1:.2f}")

        results['model1'] = {'name': args.model1_name, 'ppl': ppl1, 'size_mb': size1}

        del model1, tokenizer1
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error loading Model 1: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate Model 2
    print("\n" + "=" * 80)
    print(f"Loading Model 2: {args.model2_name}")
    print("=" * 80)

    try:
        if not os.path.exists(args.model2_path):
            print(f"❌ Error: Model path not found: {args.model2_path}")
            return

        tokenizer2 = AutoTokenizer.from_pretrained(args.model2_path, trust_remote_code=True)
        model2 = AutoModelForCausalLM.from_pretrained(
            args.model2_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        size2 = get_model_size(model2)
        print(f"Model size: {size2:.2f} MB")

        ppl2 = compute_perplexity(model2, tokenizer2, eval_texts, device, args.n_eval)
        print(f"Perplexity: {ppl2:.2f}")

        results['model2'] = {'name': args.model2_name, 'ppl': ppl2, 'size_mb': size2}

        del model2, tokenizer2
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error loading Model 2: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if 'original' in results:
        print(f"\nOriginal Model:")
        print(f"  Perplexity: {results['original']['ppl']:.2f}")
        print(f"  Size: {results['original']['size_mb']:.2f} MB")

    print(f"\n{results['model1']['name']}:")
    print(f"  Perplexity: {results['model1']['ppl']:.2f}")
    print(f"  Size: {results['model1']['size_mb']:.2f} MB")
    if 'original' in results:
        ppl_delta1 = results['model1']['ppl'] - results['original']['ppl']
        print(f"  Δ Perplexity: {ppl_delta1:+.2f}")
        compression1 = results['original']['size_mb'] / results['model1']['size_mb']
        print(f"  Compression: {compression1:.2f}x")

    print(f"\n{results['model2']['name']}:")
    print(f"  Perplexity: {results['model2']['ppl']:.2f}")
    print(f"  Size: {results['model2']['size_mb']:.2f} MB")
    if 'original' in results:
        ppl_delta2 = results['model2']['ppl'] - results['original']['ppl']
        print(f"  Δ Perplexity: {ppl_delta2:+.2f}")
        compression2 = results['original']['size_mb'] / results['model2']['size_mb']
        print(f"  Compression: {compression2:.2f}x")

    print(f"\nDirect Comparison:")
    ppl_diff = results['model2']['ppl'] - results['model1']['ppl']
    print(f"  {results['model2']['name']} - {results['model1']['name']}: {ppl_diff:+.2f}")

    if results['model2']['ppl'] < results['model1']['ppl']:
        improvement = (results['model1']['ppl'] - results['model2']['ppl']) / results['model1']['ppl'] * 100
        print(f"  → {results['model2']['name']} is BETTER by {improvement:.2f}%")
        winner = results['model2']['name']
    elif results['model2']['ppl'] > results['model1']['ppl']:
        improvement = (results['model2']['ppl'] - results['model1']['ppl']) / results['model1']['ppl'] * 100
        print(f"  → {results['model1']['name']} is BETTER by {improvement:.2f}%")
        winner = results['model1']['name']
    else:
        print(f"  → Both methods perform equally")
        winner = "Tie"

    print("=" * 80)

    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")

        if 'original' in results:
            f.write(f"Original Model:\n")
            f.write(f"  Perplexity: {results['original']['ppl']:.2f}\n")
            f.write(f"  Size: {results['original']['size_mb']:.2f} MB\n\n")

        f.write(f"{results['model1']['name']}:\n")
        f.write(f"  Path: {args.model1_path}\n")
        f.write(f"  Perplexity: {results['model1']['ppl']:.2f}\n")
        f.write(f"  Size: {results['model1']['size_mb']:.2f} MB\n")
        if 'original' in results:
            f.write(f"  Δ Perplexity: {ppl_delta1:+.2f}\n")
            f.write(f"  Compression: {compression1:.2f}x\n")
        f.write("\n")

        f.write(f"{results['model2']['name']}:\n")
        f.write(f"  Path: {args.model2_path}\n")
        f.write(f"  Perplexity: {results['model2']['ppl']:.2f}\n")
        f.write(f"  Size: {results['model2']['size_mb']:.2f} MB\n")
        if 'original' in results:
            f.write(f"  Δ Perplexity: {ppl_delta2:+.2f}\n")
            f.write(f"  Compression: {compression2:.2f}x\n")
        f.write("\n")

        f.write(f"Winner: {winner}\n")

    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
