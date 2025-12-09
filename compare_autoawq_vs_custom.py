"""
Compare AutoAWQ Library vs Custom gw_awq_asym_l2 Implementation

This script evaluates and compares:
1. AutoAWQ Library (official implementation)
2. Custom gw_awq_asym_l2.py (asymmetric quantization + L2 salience)
3. Original FP16 model (baseline)

Metrics:
- Perplexity on WikiText-2 validation set
- Model size (compression ratio)
- Inference throughput (tokens/second)
- Memory usage

Key Differences:
- AutoAWQ: Symmetric quantization, standard AWQ importance scoring
- Custom gw_awq_asym_l2: Asymmetric quantization [0,15], L2-based importance
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
import os
import json
import time


def load_wikitext2_validation(n_samples=None):
    """Load WikiText-2 validation set."""
    print("Loading WikiText-2 validation set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts


def compute_perplexity(model, tokenizer, texts, device, max_length=512, stride=256):
    """
    Compute perplexity on a dataset using sliding window approach.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text strings
        device: Device to run on
        max_length: Maximum sequence length
        stride: Stride for sliding window

    Returns:
        perplexity: The computed perplexity
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            try:
                # Tokenize
                encodings = tokenizer(text, return_tensors="pt", truncation=False)
                input_ids = encodings.input_ids.to(device)

                seq_len = input_ids.size(1)

                # Sliding window approach for long sequences
                prev_end_loc = 0
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc

                    input_ids_slice = input_ids[:, begin_loc:end_loc]
                    target_ids = input_ids_slice.clone()
                    target_ids[:, :-trg_len] = -100  # Ignore context tokens

                    outputs = model(input_ids_slice, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len

                    total_nll += neg_log_likelihood.item()
                    total_tokens += trg_len

                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

            except Exception as e:
                print(f"Warning: Skipping text due to error: {e}")
                continue

    perplexity = torch.exp(torch.tensor(total_nll / total_tokens))
    return perplexity.item()


def measure_inference_speed(model, tokenizer, device, num_samples=50, seq_length=512):
    """
    Measure inference throughput (tokens/second).

    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        num_samples: Number of samples to average over
        seq_length: Sequence length for testing

    Returns:
        tokens_per_second: Average throughput
    """
    model.eval()

    # Create dummy inputs
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_length)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)

    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(input_ids)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = num_samples * seq_length
    tokens_per_second = total_tokens / total_time

    return tokens_per_second


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def load_autoawq_model(model_path, device):
    """Load AutoAWQ quantized model."""
    print(f"\nLoading AutoAWQ model from {model_path}...")
    try:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            trust_remote_code=True,
            safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load AutoAWQ model: {e}")
        return None, None


def load_custom_model(model_path, device):
    """Load custom quantized model (standard HuggingFace format)."""
    print(f"\nLoading custom model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load custom model: {e}")
        return None, None


def load_original_model(model_path, device):
    """Load original FP16 model."""
    print(f"\nLoading original FP16 model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load original model: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Compare AutoAWQ Library vs Custom gw_awq_asym_l2 Implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--original-model", type=str, default="openbmb/MiniCPM-2B-sft-bf16",
                       help="Original FP16 model path")
    parser.add_argument("--autoawq-model", type=str, default="./quantized_models/minicpm_autoawq",
                       help="AutoAWQ quantized model path")
    parser.add_argument("--custom-model", type=str, default="./quantized_models/minicpm_gw_awq_asym_l2",
                       help="Custom quantized model path")
    parser.add_argument("--n-samples", type=int, default=100,
                       help="Number of validation samples for perplexity")
    parser.add_argument("--output-json", type=str, default="./comparison_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--skip-original", action="store_true",
                       help="Skip evaluation of original model (saves time)")
    parser.add_argument("--skip-speed", action="store_true",
                       help="Skip throughput measurement (saves time)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("Model Comparison: AutoAWQ Library vs Custom Implementation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Validation samples: {args.n_samples}")
    print("=" * 80)

    # Load validation data
    val_texts = load_wikitext2_validation(n_samples=args.n_samples)
    print(f"Loaded {len(val_texts)} validation texts\n")

    results = {}

    # Evaluate Original Model (if not skipped)
    if not args.skip_original and os.path.exists(args.original_model):
        print("\n" + "=" * 80)
        print("Evaluating Original FP16 Model")
        print("=" * 80)

        model, tokenizer = load_original_model(args.original_model, device)
        if model is not None:
            # Model size
            size_mb = get_model_size_mb(model)
            print(f"Model size: {size_mb:.2f} MB")

            # Perplexity
            print("Computing perplexity...")
            ppl = compute_perplexity(model, tokenizer, val_texts, device)
            print(f"Perplexity: {ppl:.4f}")

            # Throughput
            throughput = None
            if not args.skip_speed:
                print("Measuring throughput...")
                throughput = measure_inference_speed(model, tokenizer, device)
                print(f"Throughput: {throughput:.2f} tokens/second")

            results['original_fp16'] = {
                'model_path': args.original_model,
                'size_mb': size_mb,
                'perplexity': ppl,
                'throughput_tokens_per_sec': throughput,
                'compression_ratio': 1.0
            }

            # Clean up
            del model
            torch.cuda.empty_cache()

    # Evaluate AutoAWQ Model
    if os.path.exists(args.autoawq_model):
        print("\n" + "=" * 80)
        print("Evaluating AutoAWQ Library Model")
        print("=" * 80)

        model, tokenizer = load_autoawq_model(args.autoawq_model, device)
        if model is not None:
            # Model size
            size_mb = get_model_size_mb(model)
            print(f"Model size: {size_mb:.2f} MB")

            # Perplexity
            print("Computing perplexity...")
            ppl = compute_perplexity(model, tokenizer, val_texts, device)
            print(f"Perplexity: {ppl:.4f}")

            # Throughput
            throughput = None
            if not args.skip_speed:
                print("Measuring throughput...")
                throughput = measure_inference_speed(model, tokenizer, device)
                print(f"Throughput: {throughput:.2f} tokens/second")

            # Compression ratio
            orig_size = results.get('original_fp16', {}).get('size_mb', None)
            compression = orig_size / size_mb if orig_size else None

            results['autoawq_library'] = {
                'model_path': args.autoawq_model,
                'size_mb': size_mb,
                'perplexity': ppl,
                'throughput_tokens_per_sec': throughput,
                'compression_ratio': compression,
                'quantization_method': 'AutoAWQ Library (symmetric, standard importance)'
            }

            # Clean up
            del model
            torch.cuda.empty_cache()
    else:
        print(f"\n⚠️  AutoAWQ model not found at {args.autoawq_model}")
        print("   Run quantize_autoawq_library.py first to generate this model")

    # Evaluate Custom Model
    if os.path.exists(args.custom_model):
        print("\n" + "=" * 80)
        print("Evaluating Custom gw_awq_asym_l2 Model")
        print("=" * 80)

        model, tokenizer = load_custom_model(args.custom_model, device)
        if model is not None:
            # Model size
            size_mb = get_model_size_mb(model)
            print(f"Model size: {size_mb:.2f} MB")

            # Perplexity
            print("Computing perplexity...")
            ppl = compute_perplexity(model, tokenizer, val_texts, device)
            print(f"Perplexity: {ppl:.4f}")

            # Throughput
            throughput = None
            if not args.skip_speed:
                print("Measuring throughput...")
                throughput = measure_inference_speed(model, tokenizer, device)
                print(f"Throughput: {throughput:.2f} tokens/second")

            # Compression ratio
            orig_size = results.get('original_fp16', {}).get('size_mb', None)
            compression = orig_size / size_mb if orig_size else None

            results['custom_gw_awq_asym_l2'] = {
                'model_path': args.custom_model,
                'size_mb': size_mb,
                'perplexity': ppl,
                'throughput_tokens_per_sec': throughput,
                'compression_ratio': compression,
                'quantization_method': 'Custom (asymmetric [0,15], L2 salience)'
            }

            # Clean up
            del model
            torch.cuda.empty_cache()
    else:
        print(f"\n⚠️  Custom model not found at {args.custom_model}")
        print("   Run gw_awq_asym_l2.py first to generate this model")

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if len(results) == 0:
        print("No models were successfully evaluated!")
        return

    # Create comparison table
    print(f"\n{'Model':<30} {'Size (MB)':<12} {'PPL':<10} {'Throughput':<15} {'Compression':<12}")
    print("-" * 80)

    for name, data in results.items():
        size_str = f"{data['size_mb']:.2f}"
        ppl_str = f"{data['perplexity']:.4f}"
        throughput_str = f"{data['throughput_tokens_per_sec']:.2f}" if data['throughput_tokens_per_sec'] else "N/A"
        compression_str = f"{data['compression_ratio']:.2f}x" if data['compression_ratio'] else "N/A"

        print(f"{name:<30} {size_str:<12} {ppl_str:<10} {throughput_str:<15} {compression_str:<12}")

    # Calculate deltas if we have both quantized models
    if 'autoawq_library' in results and 'custom_gw_awq_asym_l2' in results:
        print("\n" + "=" * 80)
        print("AutoAWQ vs Custom Comparison")
        print("=" * 80)

        autoawq_ppl = results['autoawq_library']['perplexity']
        custom_ppl = results['custom_gw_awq_asym_l2']['perplexity']
        ppl_diff = custom_ppl - autoawq_ppl
        ppl_diff_pct = (ppl_diff / autoawq_ppl) * 100

        print(f"Perplexity difference: {ppl_diff:+.4f} ({ppl_diff_pct:+.2f}%)")

        if ppl_diff < 0:
            print("✅ Custom implementation achieves LOWER perplexity (better)")
        elif ppl_diff > 0:
            print("⚠️  Custom implementation has HIGHER perplexity (worse)")
        else:
            print("➡️  Both implementations achieve same perplexity")

        if results['autoawq_library']['throughput_tokens_per_sec'] and results['custom_gw_awq_asym_l2']['throughput_tokens_per_sec']:
            autoawq_speed = results['autoawq_library']['throughput_tokens_per_sec']
            custom_speed = results['custom_gw_awq_asym_l2']['throughput_tokens_per_sec']
            speed_diff = custom_speed - autoawq_speed
            speed_diff_pct = (speed_diff / autoawq_speed) * 100

            print(f"\nThroughput difference: {speed_diff:+.2f} tokens/sec ({speed_diff_pct:+.2f}%)")

            if speed_diff > 0:
                print("✅ Custom implementation is FASTER")
            elif speed_diff < 0:
                print("⚠️  Custom implementation is SLOWER")
            else:
                print("➡️  Both implementations have same speed")

    # Save results to JSON
    print(f"\nSaving results to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {args.output_json}")
    print("\nKey Insights:")
    print("  - AutoAWQ uses symmetric quantization with standard importance")
    print("  - Custom uses asymmetric [0,15] quantization with L2 salience")
    print("  - Lower perplexity = better model quality")
    print("  - Higher throughput = faster inference")
    print("=" * 80)


if __name__ == "__main__":
    main()
