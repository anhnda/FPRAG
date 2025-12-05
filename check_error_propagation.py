"""
Test error propagation through sequential transformer blocks.

This tests whether PRAQ's errors compound worse than AWQ's errors when
propagating through multiple layers in sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from copy import deepcopy


def load_wikitext2(split="train", n_samples=None, seed=42):
    """Load WikiText-2 dataset with optional sampling."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    if n_samples:
        random.seed(seed)
        if split == "validation" and n_samples < len(texts):
            texts = random.sample(texts, n_samples)
        else:
            texts = texts[:n_samples]

    return texts


class BlockCascadeTester:
    """Test error propagation through sequential transformer blocks."""

    def __init__(self, model, tokenizer, num_blocks=3, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.num_blocks = num_blocks
        self.device = device

        # Store original weights and create quantized copies
        self.original_weights = {}
        self.awq_weights = {}
        self.praq_weights = {}

    def collect_activations(self, texts, n_samples=100):
        """Collect activations for each MLP layer in the first N blocks."""
        activations = {}  # layer_name -> list of activation tensors

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple):
                    inp = input[0].detach()
                else:
                    inp = input.detach()

                # Flatten to [batch*seq_len, hidden_dim] to handle variable sequence lengths
                inp_flat = inp.reshape(-1, inp.shape[-1])

                if name not in activations:
                    activations[name] = []
                activations[name].append(inp_flat.cpu().float())
            return hook_fn

        # Register hooks for MLP layers in first N blocks
        hooks = []
        for block_idx in range(self.num_blocks):
            for layer_type in ['gate_proj', 'up_proj', 'down_proj']:
                layer_name = f'model.layers.{block_idx}.mlp.{layer_type}'
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        hooks.append(module.register_forward_hook(make_hook(layer_name)))

        # Run forward passes
        self.model.eval()
        with torch.no_grad():
            for text in tqdm(texts[:n_samples], desc="Collecting activations"):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False)
                except:
                    continue

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Concatenate activations
        for name in activations:
            activations[name] = torch.cat(activations[name], dim=0)

        return activations

    @torch.no_grad()
    def compute_awq_importance(self, X, W, b):
        """Compute AWQ importance: E[|XW^T + b|]"""
        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b
        return Z.abs().mean(dim=0)

    @torch.no_grad()
    def compute_praq_importance(self, X, W, b, beta=3.0, tau=-3.0, noise_factor=0.2):
        """Compute FastRPRAQ importance."""
        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b

        z_mean = Z.mean(dim=0)
        z_std = Z.std(dim=0) + 1e-8
        z_upper = z_mean + 3 * z_std

        x_mag = X.abs().mean()
        w_mag = W.abs().mean(dim=1)
        estimated_noise_impact = x_mag * w_mag * noise_factor

        z_risk_upper = z_upper + estimated_noise_impact
        prob_active = torch.sigmoid(beta * (z_risk_upper - tau))
        magnitude = Z.abs().mean(dim=0) + z_std

        return prob_active * magnitude

    @torch.no_grad()
    def quantize_weights(self, layer_name, importance_scores, keep_ratio=0.2):
        """Quantize a layer's weights based on importance scores."""
        # Get the module
        module = None
        for name, mod in self.model.named_modules():
            if name == layer_name:
                module = mod
                break

        if module is None or not isinstance(module, nn.Linear):
            raise ValueError(f"Layer {layer_name} not found or not Linear")

        W = module.weight.data.cpu().float().clone()
        b = module.bias.data.cpu().float().clone() if module.bias is not None else None

        out_features = W.shape[0]
        k = max(1, int(out_features * keep_ratio))
        top_k_indices = torch.topk(importance_scores, k).indices

        mask_keep = torch.zeros(out_features, dtype=torch.bool)
        mask_keep[top_k_indices] = True

        # Quantize non-kept channels
        W_quantized = W.clone()
        for c in range(out_features):
            if not mask_keep[c]:
                w_channel = W[c, :]
                w_range = w_channel.abs().max()
                if w_range > 0:
                    scale = w_range / 7.0
                    w_quant = torch.round(w_channel / scale).clamp(-8, 7)
                    W_quantized[c, :] = w_quant * scale
                    # Add quantization noise
                    noise = torch.randn_like(w_channel) * scale * 0.1
                    W_quantized[c, :] += noise

        return W_quantized, b

    def quantize_all_layers(self, activations, keep_ratio=0.2):
        """Quantize all MLP layers in first N blocks with both AWQ and PRAQ."""
        print("\nQuantizing all layers...")

        for block_idx in range(self.num_blocks):
            for layer_type in ['gate_proj', 'up_proj', 'down_proj']:
                layer_name = f'model.layers.{block_idx}.mlp.{layer_type}'

                if layer_name not in activations:
                    continue

                # Get module
                module = None
                for name, mod in self.model.named_modules():
                    if name == layer_name:
                        module = mod
                        break

                if module is None:
                    continue

                X = activations[layer_name]
                W = module.weight.data.cpu().float()
                b = module.bias.data.cpu().float() if module.bias is not None else None

                # Store original
                self.original_weights[layer_name] = (W.clone(), b.clone() if b is not None else None)

                # Compute importance and quantize
                awq_importance = self.compute_awq_importance(X, W, b)
                praq_importance = self.compute_praq_importance(X, W, b)

                W_awq, b_awq = self.quantize_weights(layer_name, awq_importance, keep_ratio)
                W_praq, b_praq = self.quantize_weights(layer_name, praq_importance, keep_ratio)

                self.awq_weights[layer_name] = (W_awq, b_awq)
                self.praq_weights[layer_name] = (W_praq, b_praq)

                print(f"  Quantized: {layer_name}")

    def set_weights(self, weight_dict):
        """Set model weights from a weight dictionary."""
        for layer_name, (W, b) in weight_dict.items():
            for name, module in self.model.named_modules():
                if name == layer_name and isinstance(module, nn.Linear):
                    module.weight.data = W.to(self.device)
                    if b is not None:
                        module.bias.data = b.to(self.device)
                    break

    @torch.no_grad()
    def measure_cascade_error(self, texts, n_samples=100):
        """Measure error after each block for original, AWQ, and PRAQ."""
        results = {
            'block_0': {'awq': [], 'praq': []},
            'block_1': {'awq': [], 'praq': []},
            'block_2': {'awq': [], 'praq': []}
        }

        # Collect hidden states at each block
        for method_name, weight_dict in [('awq', self.awq_weights), ('praq', self.praq_weights)]:
            print(f"\nTesting {method_name.upper()} cascade...")
            self.set_weights(weight_dict)

            for block_idx in range(self.num_blocks):
                block_outputs = []

                def make_hook(outputs_list):
                    def hook_fn(module, input, output):
                        # Capture block output
                        if isinstance(output, tuple):
                            out = output[0].detach()
                        else:
                            out = output.detach()
                        # Flatten to handle variable sequence lengths
                        out_flat = out.reshape(-1, out.shape[-1])
                        outputs_list.append(out_flat.cpu().float())
                    return hook_fn

                # Hook the block output
                block_name = f'model.layers.{block_idx}'
                hook = None
                for name, module in self.model.named_modules():
                    if name == block_name:
                        hook = module.register_forward_hook(make_hook(block_outputs))
                        break

                # Run inference
                self.model.eval()
                for text in tqdm(texts[:n_samples], desc=f"Block {block_idx} ({method_name})", leave=False):
                    try:
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        self.model(**inputs, use_cache=False)
                    except:
                        continue

                if hook:
                    hook.remove()

                # Store outputs
                if block_outputs:
                    results[f'block_{block_idx}'][method_name] = torch.cat(block_outputs, dim=0)

        # Restore original weights and get ground truth
        print("\nCollecting ground truth...")
        self.set_weights(self.original_weights)

        for block_idx in range(self.num_blocks):
            block_outputs = []

            def make_hook(outputs_list):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].detach()
                    else:
                        out = output.detach()
                    # Flatten to handle variable sequence lengths
                    out_flat = out.reshape(-1, out.shape[-1])
                    outputs_list.append(out_flat.cpu().float())
                return hook_fn

            block_name = f'model.layers.{block_idx}'
            hook = None
            for name, module in self.model.named_modules():
                if name == block_name:
                    hook = module.register_forward_hook(make_hook(block_outputs))
                    break

            self.model.eval()
            for text in tqdm(texts[:n_samples], desc=f"Block {block_idx} (original)", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False)
                except:
                    continue

            if hook:
                hook.remove()

            if block_outputs:
                results[f'block_{block_idx}']['original'] = torch.cat(block_outputs, dim=0)

        return results

    def compute_errors(self, results):
        """Compute MSE at each block for AWQ and PRAQ."""
        errors = {
            'awq': [],
            'praq': []
        }

        for block_idx in range(self.num_blocks):
            block_key = f'block_{block_idx}'
            original = results[block_key]['original']
            awq_out = results[block_key]['awq']
            praq_out = results[block_key]['praq']

            mse_awq = F.mse_loss(awq_out, original).item()
            mse_praq = F.mse_loss(praq_out, original).item()

            errors['awq'].append(mse_awq)
            errors['praq'].append(mse_praq)

            print(f"\nBlock {block_idx} cumulative error:")
            print(f"  AWQ MSE:  {mse_awq:.6f}")
            print(f"  PRAQ MSE: {mse_praq:.6f}")
            print(f"  Ratio (PRAQ/AWQ): {mse_praq/mse_awq:.3f}x")

        return errors


def visualize_error_propagation(errors, output_dir="./visualizations/error_propagation"):
    """Visualize how errors accumulate through blocks."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    blocks = list(range(len(errors['awq'])))

    # Absolute MSE
    axes[0].plot(blocks, errors['awq'], 'o-', label='AWQ', linewidth=2, markersize=8)
    axes[0].plot(blocks, errors['praq'], 's-', label='PRAQ', linewidth=2, markersize=8)
    axes[0].set_xlabel('Block Number', fontsize=12)
    axes[0].set_ylabel('Cumulative MSE', fontsize=12)
    axes[0].set_title('Error Accumulation Through Blocks', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Relative error (PRAQ/AWQ ratio)
    ratios = [errors['praq'][i] / errors['awq'][i] for i in range(len(blocks))]
    axes[1].plot(blocks, ratios, 'o-', linewidth=2, markersize=8, color='red')
    axes[1].axhline(1.0, color='black', linestyle='--', linewidth=1, label='Equal (ratio=1)')
    axes[1].set_xlabel('Block Number', fontsize=12)
    axes[1].set_ylabel('Error Ratio (PRAQ / AWQ)', fontsize=12)
    axes[1].set_title('Relative Error Growth', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(blocks, 1.0, ratios, where=[r > 1.0 for r in ratios],
                         alpha=0.3, color='red', label='PRAQ worse')
    axes[1].fill_between(blocks, ratios, 1.0, where=[r < 1.0 for r in ratios],
                         alpha=0.3, color='green', label='PRAQ better')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_propagation.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/error_propagation.png")
    plt.close()


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_blocks = 3
    n_calib = 100
    n_val = 100
    keep_ratio = 0.2

    print("=" * 80)
    print("ERROR PROPAGATION TEST: AWQ vs FastRPRAQ")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Testing {num_blocks} sequential transformer blocks")
    print(f"Calibration samples: {n_calib}")
    print(f"Validation samples: {n_val}")
    print(f"Keep ratio: {keep_ratio} ({int(keep_ratio*100)}% FP16, {int((1-keep_ratio)*100)}% INT4)")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Load data
    print("Loading datasets...")
    calib_texts = load_wikitext2("train", n_samples=n_calib)
    val_texts = load_wikitext2("validation", n_samples=n_val)

    # Create tester
    tester = BlockCascadeTester(model, tokenizer, num_blocks=num_blocks, device=device)

    # Collect activations for quantization
    print("\nCollecting activations for calibration...")
    activations = tester.collect_activations(calib_texts, n_samples=n_calib)

    # Quantize all layers
    tester.quantize_all_layers(activations, keep_ratio=keep_ratio)

    # Measure cascade error
    print("\nMeasuring error propagation...")
    results = tester.measure_cascade_error(val_texts, n_samples=n_val)

    # Compute and display errors
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    errors = tester.compute_errors(results)

    # Visualize
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)
    visualize_error_propagation(errors)

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    final_ratio = errors['praq'][-1] / errors['awq'][-1]
    if final_ratio > 1.05:
        print(f"⚠ ERROR PROPAGATION PROBLEM DETECTED!")
        print(f"  After {num_blocks} blocks, PRAQ's cumulative error is {final_ratio:.2f}x worse than AWQ")
        print(f"  This explains why PRAQ has better per-layer MSE but worse perplexity!")
    elif final_ratio < 0.95:
        print(f"✓ PRAQ propagates better!")
        print(f"  After {num_blocks} blocks, PRAQ's cumulative error is {1/final_ratio:.2f}x better than AWQ")
        print(f"  Error propagation does NOT explain the perplexity paradox.")
    else:
        print(f"≈ Similar error propagation")
        print(f"  After {num_blocks} blocks, error ratio is {final_ratio:.2f}x (similar)")
        print(f"  Error propagation is not the main factor.")


if __name__ == "__main__":
    main()
