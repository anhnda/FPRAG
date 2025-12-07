"""
Test L1 (E[|X|]) vs L2 (E[X²]) salience for AWQ quantization.
Quick comparison on WikiText-2 only.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np


class AWQQuantizer:
    """AWQ quantizer with configurable salience metric."""

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, salience_mode="l1"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.salience_mode = salience_mode  # "l1" or "l2"

        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[AWQ Quantizer - {salience_mode.upper()} Salience]")
        print(f"  Grid points: {n_grid}")
        print(f"  Salience metric: {'E[|X|]' if salience_mode == 'l1' else 'E[X²]'}")

    def register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                inp = input[0].detach().cpu() if isinstance(input, tuple) else input.detach().cpu()
                self.activation_data[name].append(inp)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(get_hook(name)))

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @torch.no_grad()
    def get_activation_salience(self, name):
        """Get salience using L1 or L2 norm."""
        if name not in self.activation_data or not self.activation_data[name]:
            return None

        X_list = self.activation_data[name]
        in_features = X_list[0].shape[-1]

        salience_sum = torch.zeros(in_features)
        total_samples = 0

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])

            if self.salience_mode == "l1":
                salience_sum += x_flat.abs().sum(dim=0)
            else:  # l2
                salience_sum += x_flat.pow(2).sum(dim=0)

            total_samples += x_flat.shape[0]

        return salience_sum / total_samples

    @torch.no_grad()
    def quantize_weight_groupwise(self, W):
        """Asymmetric group-wise quantization."""
        out_features, in_features = W.shape

        if in_features % self.group_size == 0:
            W_grouped = W.view(out_features, -1, self.group_size)
        else:
            pad_len = self.group_size - (in_features % self.group_size)
            W_padded = torch.nn.functional.pad(W, (0, pad_len))
            W_grouped = W_padded.view(out_features, -1, self.group_size)

        # Asymmetric quantization
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, 15)
        W_dequant_grouped = (W_int - zero_point) * scale

        W_dequant_flat = W_dequant_grouped.reshape(out_features, -1)
        return W_dequant_flat[:, :in_features]

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """Grid search for optimal scaling factor."""
        salience = self.get_activation_salience(name)
        if salience is None:
            return torch.ones(module.weight.shape[1]).to(self.device), 0.0, 0.0

        salience = salience.to(self.device)
        salience_norm = salience / (salience.mean() + 1e-8)

        # Prepare calibration data
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)
        max_samples = min(2048, X_cpu.shape[0])

        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)
        del X_cpu

        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        Y_orig = torch.matmul(X_search, W.t())
        if b is not None:
            Y_orig = Y_orig + b

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Grid search
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            scales = salience_norm.pow(alpha).clamp(min=1e-5)
            W_scaled = W * scales.unsqueeze(0)
            W_quant = self.quantize_weight_groupwise(W_scaled)

            X_compensated = X_search / scales.unsqueeze(0)
            Y_quant = torch.matmul(X_compensated, W_quant.t())
            if b is not None:
                Y_quant = Y_quant + b

            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Quantize a single layer."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        W_scaled = W * best_scales.unsqueeze(0)
        W_quant = self.quantize_weight_groupwise(W_scaled)
        W_final = W_quant / best_scales.unsqueeze(0)

        module.weight.data = W_final

        self.layer_scales[name] = {
            'alpha': best_alpha,
            'error': best_error
        }

        torch.cuda.empty_cache()

    def calibrate(self, calibration_data, n_samples=128):
        """Calibrate with data."""
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
        self.model.eval()
        self.register_hooks()

        successful = 0
        for text in tqdm(calibration_data[:n_samples], desc="Calibration"):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    _ = self.model(**inputs, use_cache=False)
                successful += 1
            except Exception:
                continue

        self.remove_hooks()
        print(f"Calibration complete! {successful}/{n_samples} samples")

    def quantize_model(self):
        """Quantize all linear layers."""
        print("\n" + "=" * 80)
        print(f"Quantizing with {self.salience_mode.upper()} salience")
        print("=" * 80)

        quantized = 0
        layer_names = [(n, m) for n, m in self.model.named_modules() if isinstance(m, nn.Linear)]

        for name, module in tqdm(layer_names, desc="Quantizing"):
            try:
                self.quantize_layer(name, module)
                quantized += 1

                if name in self.activation_data:
                    del self.activation_data[name]

                if quantized % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️ Error at {name}: {e}")
                continue

        print(f"\n✅ Quantized {quantized} layers")
        self.activation_data = {}
        torch.cuda.empty_cache()


def load_wikitext2(split="train", n_samples=None):
    """Load WikiText-2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
    if n_samples:
        random.seed(42)
        texts = random.sample(texts, min(n_samples, len(texts)))
    return texts


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, texts, device, max_samples=500):
    """Evaluate perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    successful = 0

    for text in tqdm(texts[:max_samples], desc="Evaluating"):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs['input_ids'], use_cache=False)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * inputs['input_ids'].numel()
            total_tokens += inputs['input_ids'].numel()
            successful += 1

        except Exception:
            continue

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openbmb/MiniCPM-2B-sft-bf16"

    print("=" * 80)
    print("L1 vs L2 Salience Comparison")
    print("=" * 80)
    print(f"Device: {device}")
    print("=" * 80)

    # Load data
    print("\nLoading datasets...")
    calib_texts = load_wikitext2(split="train", n_samples=128)
    eval_texts = load_wikitext2(split="validation", n_samples=500)

    results = {}

    for salience_mode in ["l1", "l2"]:
        print(f"\n{'='*80}")
        print(f"Testing {salience_mode.upper()} Salience: {'E[|X|]' if salience_mode == 'l1' else 'E[X²]'}")
        print(f"{'='*80}")

        # Load fresh model
        print(f"\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(device)

        # Quantize
        quantizer = AWQQuantizer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_grid=20,
            group_size=128,
            salience_mode=salience_mode
        )

        quantizer.calibrate(calib_texts, n_samples=128)
        quantizer.quantize_model()

        # Evaluate
        print(f"\nEvaluating {salience_mode.upper()}...")
        perplexity = evaluate_perplexity(model, tokenizer, eval_texts, device, max_samples=500)

        results[salience_mode] = perplexity

        print(f"\n{salience_mode.upper()} Perplexity: {perplexity:.4f}")

        # Cleanup
        del model, tokenizer, quantizer
        torch.cuda.empty_cache()

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"L1 (E[|X|]):  {results['l1']:.4f}")
    print(f"L2 (E[X²]):   {results['l2']:.4f}")
    print("=" * 80)

    if results['l2'] < results['l1']:
        improvement = ((results['l1'] - results['l2']) / results['l1']) * 100
        print(f"✅ L2 WINS by {improvement:.2f}% (lower perplexity)")
    elif results['l1'] < results['l2']:
        improvement = ((results['l2'] - results['l1']) / results['l2']) * 100
        print(f"✅ L1 WINS by {improvement:.2f}% (lower perplexity)")
    else:
        print("🤝 TIE")
    print("=" * 80)


if __name__ == "__main__":
    main()
