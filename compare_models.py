import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
import random
import os


def load_wikitext2_validation(n_samples=2000, seed=42):
    """
    Load WikiText-2 validation set with random sampling.

    Args:
        n_samples: Number of samples to randomly select
        seed: Random seed for reproducibility

    Returns:
        List of text samples
    """
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

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of text samples
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_time = 0

    losses = []

    for text in tqdm(texts, desc="Evaluating"):
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
            outputs = model(input_ids, labels=input_ids)
            elapsed = time.time() - start_time

            loss = outputs.loss.item()
            n_tokens = input_ids.shape[1]

            total_loss += loss * n_tokens
            total_tokens += n_tokens
            total_time += elapsed
            losses.append(loss)

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    avg_time_per_sample = total_time / len(texts) if len(texts) > 0 else 0
    throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "avg_time_per_sample": avg_time_per_sample,
        "throughput_tokens_per_sec": throughput,
        "total_tokens": total_tokens,
        "num_samples": len(texts)
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


def compare_models():
    """Compare PRAQ, AWQ, and original models."""

    # Configuration
    original_model_name = "openbmb/MiniCPM-2B-sft-bf16"
    praq_model_path = "./quantized_models/minicpm_praq"
    awq_model_path = "./quantized_models/minicpm_awq"
    n_eval_samples = 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("MiniCPM-2.4 Quantization Comparison: Fast-R-PRAQ vs AWQ")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Evaluation samples: {n_eval_samples}")
    print("=" * 80)

    # Load validation data
    eval_texts = load_wikitext2_validation(n_samples=n_eval_samples)

    results = {}

    # ===========================
    # 1. Evaluate Original Model
    # ===========================
    print("\n" + "=" * 80)
    print("1. Evaluating Original Model (FP16 Baseline)")
    print("=" * 80)

    try:
        print(f"Loading original model: {original_model_name}")
        tokenizer_orig = AutoTokenizer.from_pretrained(original_model_name, trust_remote_code=True)
        model_orig = AutoModelForCausalLM.from_pretrained(
            original_model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        model_size_orig = get_model_size(model_orig)
        print(f"Model size: {model_size_orig:.2f} MB")

        results['original'] = evaluate_perplexity(model_orig, tokenizer_orig, eval_texts, device=device)
        results['original']['model_size_mb'] = model_size_orig

        print(f"\nResults:")
        print(f"  Perplexity: {results['original']['perplexity']:.4f}")
        print(f"  Avg Loss: {results['original']['avg_loss']:.4f}")
        print(f"  Throughput: {results['original']['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"  Model Size: {model_size_orig:.2f} MB")

        # Clean up
        del model_orig
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating original model: {e}")
        results['original'] = None

    # ===========================
    # 2. Evaluate PRAQ Model
    # ===========================
    print("\n" + "=" * 80)
    print("2. Evaluating Fast-R-PRAQ Quantized Model")
    print("=" * 80)

    if os.path.exists(praq_model_path):
        try:
            print(f"Loading PRAQ model: {praq_model_path}")
            tokenizer_praq = AutoTokenizer.from_pretrained(praq_model_path, trust_remote_code=True)
            model_praq = AutoModelForCausalLM.from_pretrained(
                praq_model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )

            model_size_praq = get_model_size(model_praq)
            print(f"Model size: {model_size_praq:.2f} MB")

            results['praq'] = evaluate_perplexity(model_praq, tokenizer_praq, eval_texts, device=device)
            results['praq']['model_size_mb'] = model_size_praq

            print(f"\nResults:")
            print(f"  Perplexity: {results['praq']['perplexity']:.4f}")
            print(f"  Avg Loss: {results['praq']['avg_loss']:.4f}")
            print(f"  Throughput: {results['praq']['throughput_tokens_per_sec']:.2f} tokens/sec")
            print(f"  Model Size: {model_size_praq:.2f} MB")

            # Clean up
            del model_praq
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating PRAQ model: {e}")
            results['praq'] = None
    else:
        print(f"PRAQ model not found at {praq_model_path}")
        print("Please run quantize_minicpm_PRAQ.py first")
        results['praq'] = None

    # ===========================
    # 3. Evaluate AWQ Model
    # ===========================
    print("\n" + "=" * 80)
    print("3. Evaluating AWQ Quantized Model")
    print("=" * 80)

    if os.path.exists(awq_model_path):
        try:
            print(f"Loading AWQ model: {awq_model_path}")
            tokenizer_awq = AutoTokenizer.from_pretrained(awq_model_path, trust_remote_code=True)
            model_awq = AutoAWQForCausalLM.from_quantized(
                awq_model_path,
                fuse_layers=True,
                trust_remote_code=True
            )
            model_awq = model_awq.to(device)

            model_size_awq = get_model_size(model_awq)
            print(f"Model size: {model_size_awq:.2f} MB")

            results['awq'] = evaluate_perplexity(model_awq, tokenizer_awq, eval_texts, device=device)
            results['awq']['model_size_mb'] = model_size_awq

            print(f"\nResults:")
            print(f"  Perplexity: {results['awq']['perplexity']:.4f}")
            print(f"  Avg Loss: {results['awq']['avg_loss']:.4f}")
            print(f"  Throughput: {results['awq']['throughput_tokens_per_sec']:.2f} tokens/sec")
            print(f"  Model Size: {model_size_awq:.2f} MB")

            # Clean up
            del model_awq
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating AWQ model: {e}")
            results['awq'] = None
    else:
        print(f"AWQ model not found at {awq_model_path}")
        print("Please run quantize_minicpm_awq.py first")
        results['awq'] = None

    # ===========================
    # 4. Summary Comparison
    # ===========================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Create comparison table
    print(f"\n{'Metric':<30} {'Original':<15} {'PRAQ':<15} {'AWQ':<15}")
    print("-" * 80)

    if results.get('original'):
        orig = results['original']
        praq = results.get('praq')
        awq = results.get('awq')

        # Perplexity
        print(f"{'Perplexity':<30} {orig['perplexity']:>14.4f} ", end="")
        if praq:
            print(f"{praq['perplexity']:>14.4f} ", end="")
        else:
            print(f"{'N/A':>14} ", end="")
        if awq:
            print(f"{awq['perplexity']:>14.4f}")
        else:
            print(f"{'N/A':>14}")

        # Model Size
        print(f"{'Model Size (MB)':<30} {orig['model_size_mb']:>14.2f} ", end="")
        if praq:
            print(f"{praq['model_size_mb']:>14.2f} ", end="")
        else:
            print(f"{'N/A':>14} ", end="")
        if awq:
            print(f"{awq['model_size_mb']:>14.2f}")
        else:
            print(f"{'N/A':>14}")

        # Throughput
        print(f"{'Throughput (tokens/sec)':<30} {orig['throughput_tokens_per_sec']:>14.2f} ", end="")
        if praq:
            print(f"{praq['throughput_tokens_per_sec']:>14.2f} ", end="")
        else:
            print(f"{'N/A':>14} ", end="")
        if awq:
            print(f"{awq['throughput_tokens_per_sec']:>14.2f}")
        else:
            print(f"{'N/A':>14}")

        print("-" * 80)

        # Perplexity degradation
        if praq:
            praq_degradation = ((praq['perplexity'] - orig['perplexity']) / orig['perplexity']) * 100
            print(f"\nPRAQ Perplexity Degradation: {praq_degradation:+.2f}%")

        if awq:
            awq_degradation = ((awq['perplexity'] - orig['perplexity']) / orig['perplexity']) * 100
            print(f"AWQ Perplexity Degradation: {awq_degradation:+.2f}%")

        # Winner
        if praq and awq:
            print("\n" + "=" * 80)
            if praq['perplexity'] < awq['perplexity']:
                improvement = ((awq['perplexity'] - praq['perplexity']) / awq['perplexity']) * 100
                print(f"✅ WINNER: Fast-R-PRAQ")
                print(f"   PRAQ achieves {improvement:.2f}% lower perplexity than AWQ")
            elif awq['perplexity'] < praq['perplexity']:
                improvement = ((praq['perplexity'] - awq['perplexity']) / praq['perplexity']) * 100
                print(f"✅ WINNER: AWQ")
                print(f"   AWQ achieves {improvement:.2f}% lower perplexity than PRAQ")
            else:
                print("🤝 TIE: Both methods achieve similar perplexity")
            print("=" * 80)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    compare_models()
