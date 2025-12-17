"""
Cross-Dataset Validation: AWQ Methods with Sliding Window Evaluation (FINAL CORRECTED)

Fixes:
1. Sliding Window Math: Now uses correct context masking (labels=-100).
2. Llama 3 BOS: Manually handles BOS to prevent "Double BOS" (PPL 15.5 -> 6.2).
3. Tokenizer: Handles Mistral/Llama-3 regex warnings.

Metric: loglikelihood_rolling (Standard lm-evaluation-harness methodology)
Stride: 512 tokens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import os
import json
from datetime import datetime

class AWQSlidingWindowValidator:
    def __init__(self, device="cuda", seed=42, stride=512, max_length=2048): # Increased to 2048 for Llama 3
        self.device = device
        self.seed = seed
        self.stride = stride
        self.max_length = max_length
        self.results = {}

        print("="*80)
        print("AWQ SLIDING WINDOW CROSS-DATASET VALIDATION (FINAL)")
        print("="*80)
        print(f"Device: {device}")
        print(f"Stride: {stride}")
        print(f"Max Seq Length: {max_length}")
        print("="*80)

    def load_wikitext2_test(self, n_samples=None):
        """Load WikiText-2 test set."""
        print("\n[1/3] Loading WikiText-2 test...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [text for text in dataset['text'] if len(text) > 0] # Filter empty lines

        # Wikitext is often evaluated as one long stream, but for sample-based:
        if n_samples and n_samples < len(texts):
            random.seed(self.seed)
            texts = random.sample(texts, n_samples)

        print(f"  ✅ Loaded {len(texts)} samples")
        return texts

    def load_c4_validation(self, n_samples=500):
        print("\n[2/3] Loading C4 validation...")
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in tqdm(dataset, total=n_samples, desc="  Collecting C4"):
            if len(texts) >= n_samples: break
            if len(item['text'].strip()) > 500: texts.append(item['text'])
        return texts

    def load_ag_news_test(self, n_samples=500):
        print("\n[3/3] Loading AG News test...")
        dataset = load_dataset("ag_news", split="test")
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 200]
        if n_samples < len(texts):
            random.seed(self.seed)
            texts = random.sample(texts, n_samples)
        return texts

    @torch.no_grad()
    def evaluate_sliding_window(self, model, tokenizer, texts):
        """
        Final Corrected Sliding Window Evaluation.
        """
        model.eval()
        nlls = []
        total_tokens = 0

        # Debug flag to print tokens once
        debug_printed = False

        for text in tqdm(texts, desc="  Evaluating", leave=False):
            # 1. Tokenize WITHOUT adding special tokens automatically
            #    This prevents the [BOS][BOS] issue (PPL 15.5)
            encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encodings.input_ids

            # 2. Manual BOS Injection
            #    Llama 3 MUST start with ID 128000.
            if tokenizer.bos_token_id is not None:
                if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                    bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

            # Limit length for massive docs to prevent OOM
            if input_ids.size(1) > self.max_length * 20:
                input_ids = input_ids[:, :self.max_length * 20]

            input_ids = input_ids.to(self.device)
            seq_len = input_ids.size(1)

            # Skip if too short
            if seq_len < 2: continue

            # DEBUG: Print first tokens of first doc to verify BOS
            if not debug_printed:
                print(f"\n  🔍 DEBUG: First 5 tokens sent to model: {input_ids[0, :5].tolist()}")
                debug_printed = True

            prev_end_loc = 0

            # Sliding Window Loop
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.max_length, seq_len)

                # The tokens we actually want to score in this pass
                trg_len = end_loc - prev_end_loc

                # Extract window
                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()

                # MASKING (The Fix for PPL 11.5)
                # Set context tokens (everything before the target stride) to -100
                # If begin_loc == 0, trg_len == end_loc, so :-trg_len is empty (correct)
                if begin_loc > 0:
                    target_chunk[:, :-trg_len] = -100

                # Check for empty target (can happen at very end)
                if target_chunk.size(1) == 0: break

                with torch.no_grad():
                    outputs = model(input_chunk, labels=target_chunk)

                    # Convert Mean NLL back to Sum NLL
                    # (outputs.loss is average over unmasked tokens)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            total_tokens += seq_len

        if not nlls: return None

        # Final PPL Calculation
        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / total_tokens).item()

        return {
            "perplexity": perplexity,
            "total_tokens": total_tokens
        }

    def evaluate_model_on_dataset(self, model_path, model_name, texts, dataset_name):
        print(f"\n  Evaluating {model_name} on {dataset_name}...")

        try:
            # FIX: Tokenizer Regex Handling
            tokenizer_kwargs = {}
            if "Llama-3" in model_path or "Mistral" in model_path:
                tokenizer_kwargs["fix_mistral_regex"] = True

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                **tokenizer_kwargs
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )

            results = self.evaluate_sliding_window(model, tokenizer, texts)

            if results:
                print(f"  ✅ Perplexity: {results['perplexity']:.4f}")
            else:
                print("  ❌ Evaluation failed (no results)")

            del model
            torch.cuda.empty_cache()
            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_validation(self, heuristic_path, standard_path, n_samples=500):
        datasets = {
            'WikiText-2': self.load_wikitext2_test(n_samples),
            # 'C4': self.load_c4_validation(n_samples),
            # 'AG News': self.load_ag_news_test(n_samples)
        }

        models = {
            'Heuristic AWQ': heuristic_path,
            # 'Standard AWQ': standard_path
        }

        for dataset_name, texts in datasets.items():
            for model_name, model_path in models.items():
                self.evaluate_model_on_dataset(model_path, model_name, texts, dataset_name)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic-path", type=str, required=True)
    parser.add_argument("--standard-path", type=str, default="")
    args = parser.parse_args()

    validator = AWQSlidingWindowValidator()
    validator.run_validation(args.heuristic_path, args.standard_path)

if __name__ == "__main__":
    main()
