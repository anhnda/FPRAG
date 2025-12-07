"""
Automated V2 Hyperparameter Search & Comparison

This script:
1. Quantizes model with 4 different V2 configurations
2. Evaluates each on C4 validation
3. Finds the best V2 configuration
4. Compares: Best V2 vs V1 vs GW-AWQ
5. Outputs comprehensive results and recommendations

Configurations tested:
- Balanced: β=0.5, τ=2.0 (equal AWQ+PRAQ)
- AWQ-leaning: β=0.3, τ=2.5 (robust)
- PRAQ-leaning: β=0.7, τ=1.5 (intelligent)
- Conservative: β=0.4, τ=3.0 (safe generalization)
"""

import subprocess
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
import random
import json


# V2 Configurations to test
CONFIGS = [
    {
        "name": "Balanced",
        "blend_beta": 0.5,
        "temp_tau": 2.0,
        "output_dir": "./quantized_models/minicpm_gwh_v2_balanced",
        "description": "Equal AWQ+PRAQ blend, moderate temperature"
    },
    {
        "name": "AWQ-Leaning",
        "blend_beta": 0.3,
        "temp_tau": 2.5,
        "output_dir": "./quantized_models/minicpm_gwh_v2_awq_lean",
        "description": "More AWQ (robust), softer weighting"
    },
    {
        "name": "PRAQ-Leaning",
        "blend_beta": 0.7,
        "temp_tau": 1.5,
        "output_dir": "./quantized_models/minicpm_gwh_v2_praq_lean",
        "description": "More PRAQ (intelligent), sharper weighting"
    },
    {
        "name": "Conservative",
        "blend_beta": 0.4,
        "temp_tau": 3.0,
        "output_dir": "./quantized_models/minicpm_gwh_v2_conservative",
        "description": "Slightly AWQ-leaning, very soft weighting"
    }
]


def load_c4_validation(n_samples=2000, seed=42):
    """Load C4 validation set."""
    print(f"Loading C4 validation dataset (seed={seed})...")
    random.seed(seed)
    np.random.seed(seed)

    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    texts = []
    for i, item in enumerate(tqdm(dataset, desc="Loading C4", total=n_samples)):
        if len(texts) >= n_samples:
            break
        text = item['text']
        if len(text.strip()) > 100:
            texts.append(text)

    random.seed(seed)
    random.shuffle(texts)
    return texts[:n_samples]


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, texts, max_length=512, device="cuda"):
    """Evaluate perplexity on text samples."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    successful = 0

    for text in tqdm(texts, desc="Evaluating", leave=False):
        try:
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                             truncation=True, padding=False)
            input_ids = inputs["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss.item()
            n_tokens = input_ids.shape[1]

            total_loss += loss * n_tokens
            total_tokens += n_tokens
            successful += 1

        except Exception:
            continue

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss) if total_tokens > 0 else float('inf')

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "num_samples": successful
    }


def evaluate_model(model_path, model_name, eval_texts, device="cuda"):
    """Evaluate a single model."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        results = evaluate_perplexity(model, tokenizer, eval_texts, device=device)
        results['model_name'] = model_name

        print(f"\nResults:")
        print(f"  Perplexity: {results['perplexity']:.4f}")
        print(f"  Avg Loss: {results['avg_loss']:.4f}")
        print(f"  Samples: {results['num_samples']}")

        del model
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def quantize_v2_config(config):
    """Quantize model with a V2 configuration."""
    print(f"\n{'='*80}")
    print(f"Quantizing V2 - {config['name']}")
    print(f"{'='*80}")
    print(f"  β={config['blend_beta']}, τ={config['temp_tau']}")
    print(f"  {config['description']}")
    print(f"{'='*80}")

    cmd = [
        "python", "gwh_praq_v2.py",
        "--n-calib", "128",
        "--blend-beta", str(config['blend_beta']),
        "--temp-tau", str(config['temp_tau']),
        "--output-dir", config['output_dir']
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Quantization complete: {config['name']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Quantization failed: {config['name']}")
        return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_eval_samples = 2000

    print("="*80)
    print("V2 HYPERPARAMETER SEARCH & COMPARISON")
    print("="*80)
    print(f"Device: {device}")
    print(f"Evaluation samples: {n_eval_samples}")
    print(f"Configurations to test: {len(CONFIGS)}")
    print("="*80)

    # Step 1: Load C4 validation data
    print("\n[Step 1/4] Loading C4 validation data...")
    eval_texts = load_c4_validation(n_samples=n_eval_samples, seed=42)
    print(f"✅ Loaded {len(eval_texts)} samples")

    # Step 2: Quantize all V2 configurations
    # print("\n[Step 2/4] Quantizing all V2 configurations...")
    # for i, config in enumerate(CONFIGS, 1):
    #     print(f"\n--- Config {i}/{len(CONFIGS)} ---")
    #     success = quantize_v2_config(config)
    #     if not success:
    #         print(f"⚠️  Skipping {config['name']} due to quantization failure")

    # Step 3: Evaluate all models
    print("\n[Step 3/4] Evaluating all models on C4...")
    results = {}

    # Evaluate V2 configurations
    for config in CONFIGS:
        result = evaluate_model(
            config['output_dir'],
            f"V2-{config['name']}",
            eval_texts,
            device=device
        )
        if result:
            results[config['name']] = {
                'result': result,
                'config': config
            }

    # Evaluate V1 (if exists)
    v1_path = "./quantized_models/minicpm_gwh_praq"
    if os.path.exists(v1_path):
        v1_result = evaluate_model(v1_path, "V1-Original", eval_texts, device=device)
        if v1_result:
            results['V1-Original'] = {
                'result': v1_result,
                'config': {'blend_beta': None, 'temp_tau': None}
            }

    # Evaluate GW-AWQ (if exists)
    awq_path = "./quantized_models/minicpm_gw_awq"
    if os.path.exists(awq_path):
        awq_result = evaluate_model(awq_path, "GW-AWQ", eval_texts, device=device)
        if awq_result:
            results['GW-AWQ'] = {
                'result': awq_result,
                'config': {'blend_beta': 0.0, 'temp_tau': None}
            }

    # Step 4: Analyze and report
    print("\n[Step 4/4] Analysis & Comparison")
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    if not results:
        print("❌ No valid results to compare")
        return

    # Sort by perplexity
    sorted_results = sorted(results.items(), key=lambda x: x[1]['result']['perplexity'])

    print("\nPerplexity Ranking (lower is better):")
    print(f"{'Rank':<6} {'Model':<20} {'Perplexity':<12} {'β':<8} {'τ':<8}")
    print("-" * 70)

    for rank, (name, data) in enumerate(sorted_results, 1):
        ppl = data['result']['perplexity']
        beta = data['config'].get('blend_beta')
        tau = data['config'].get('temp_tau')

        beta_str = f"{beta:.2f}" if beta is not None else "N/A"
        tau_str = f"{tau:.2f}" if tau is not None else "N/A"

        symbol = "🏆" if rank == 1 else "  "
        print(f"{symbol} {rank:<4} {name:<20} {ppl:<12.4f} {beta_str:<8} {tau_str:<8}")

    # Find best V2
    v2_results = [(name, data) for name, data in sorted_results
                  if name.startswith('V2-')]

    if v2_results:
        best_v2_name, best_v2_data = v2_results[0]
        best_v2_ppl = best_v2_data['result']['perplexity']
        best_v2_config = best_v2_data['config']

        print("\n" + "="*80)
        print("BEST V2 CONFIGURATION")
        print("="*80)
        print(f"Winner: {best_v2_name}")
        print(f"  Perplexity: {best_v2_ppl:.4f}")
        print(f"  β (blend): {best_v2_config['blend_beta']:.2f}")
        print(f"  τ (temp): {best_v2_config['temp_tau']:.2f}")
        print(f"  Description: {best_v2_config['description']}")

        # Compare with V1
        v1_data = results.get('V1-Original')
        if v1_data:
            v1_ppl = v1_data['result']['perplexity']
            delta = best_v2_ppl - v1_ppl
            delta_pct = (delta / v1_ppl) * 100

            print(f"\nV2 vs V1:")
            print(f"  V1:      {v1_ppl:.4f}")
            print(f"  V2:      {best_v2_ppl:.4f}")
            print(f"  Delta:   {delta:+.4f} ({delta_pct:+.3f}%)")

            if abs(delta_pct) < 0.05:
                print(f"  → V2 ≈ V1 (essentially tied)")
            elif delta < 0:
                print(f"  → ✅ V2 WINS by {abs(delta_pct):.3f}%!")
            else:
                print(f"  → V1 better by {delta_pct:.3f}%")

        # Compare with GW-AWQ
        awq_data = results.get('GW-AWQ')
        if awq_data:
            awq_ppl = awq_data['result']['perplexity']
            delta = best_v2_ppl - awq_ppl
            delta_pct = (delta / awq_ppl) * 100

            print(f"\nBest V2 vs GW-AWQ:")
            print(f"  GW-AWQ:  {awq_ppl:.4f}")
            print(f"  Best V2: {best_v2_ppl:.4f}")
            print(f"  Delta:   {delta:+.4f} ({delta_pct:+.3f}%)")

            if abs(delta_pct) < 0.05:
                print(f"  → Essentially tied")
            elif delta < 0:
                print(f"  → ✅ V2 WINS by {abs(delta_pct):.3f}%!")
            else:
                print(f"  → AWQ better by {delta_pct:.3f}%")

    # Overall winner
    print("\n" + "="*80)
    print("OVERALL WINNER")
    print("="*80)
    winner_name, winner_data = sorted_results[0]
    winner_ppl = winner_data['result']['perplexity']
    print(f"🏆 {winner_name}")
    print(f"   Perplexity: {winner_ppl:.4f}")

    if 'description' in winner_data['config']:
        print(f"   Config: {winner_data['config']['description']}")

    # Save results to JSON
    results_file = "./results/v2_hyperparameter_search.json"
    os.makedirs("./results", exist_ok=True)

    save_data = {
        name: {
            'perplexity': data['result']['perplexity'],
            'avg_loss': data['result']['avg_loss'],
            'blend_beta': data['config'].get('blend_beta'),
            'temp_tau': data['config'].get('temp_tau'),
            'description': data['config'].get('description', '')
        }
        for name, data in results.items()
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if winner_name.startswith('V2-'):
        print(f"✅ Use V2 with {winner_name[3:]} configuration")
        print(f"   Path: {winner_data['config']['output_dir']}")
    elif winner_name == 'V1-Original':
        print("✅ Stick with V1 - V2 doesn't improve significantly")
        print("   V1 is simpler with fewer hyperparameters")
    else:
        print(f"✅ {winner_name} is best overall")

    print("="*80)


if __name__ == "__main__":
    main()
