"""
Test AWQ vs FastRPRAQ on multiple layers to see if the pattern holds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random


def load_wikitext2(split="train", n_samples=None, seed=42):
    """Load WikiText-2 dataset with optional sampling."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    if n_samples and split == "validation":
        random.seed(seed)
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)
    elif n_samples:
        texts = texts[:n_samples]

    return texts


class LayerTester:
    """Test importance scoring methods on a single layer."""

    def __init__(self, model, tokenizer, layer_name, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        self.device = device

        # Find target module
        self.target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.target_module = module
                break

        if not isinstance(self.target_module, nn.Linear):
            raise ValueError(f"Layer {layer_name} is not a Linear layer!")

        self.input_activations = []
        self.output_ground_truth = []
        self.hook = None

    def register_hook(self, store_output=False):
        """Register hook to collect activation data."""
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                inp = input[0].detach()
            else:
                inp = input.detach()

            inp_flat = inp.reshape(-1, inp.shape[-1])
            self.input_activations.append(inp_flat.cpu().float())

            if store_output:
                if isinstance(output, tuple):
                    out = output[0].detach()
                else:
                    out = output.detach()
                out_flat = out.reshape(-1, out.shape[-1])
                self.output_ground_truth.append(out_flat.cpu().float())

        self.hook = self.target_module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    @torch.no_grad()
    def collect_data(self, texts, n_samples, store_output=False):
        """Collect activation data."""
        self.input_activations = []
        self.output_ground_truth = []
        self.model.eval()

        for text in tqdm(texts[:n_samples], desc=f"Collecting data", leave=False, disable=True):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                self.model(**inputs, use_cache=False)
            except:
                continue

    @torch.no_grad()
    def compute_awq_importance(self, X):
        """
        Compute AWQ importance using original AWQ metric: E[|XW^T + b|]
        """
        W = self.target_module.weight.data.cpu().float()
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None

        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b

        # AWQ importance: E[|Z|] per channel (original AWQ paper)
        return Z.abs().mean(dim=0)

    @torch.no_grad()
    def compute_praq_importance(self, X, beta=3.0, tau=-3.0, noise_factor=0.2):
        """Compute FastRPRAQ importance."""
        W = self.target_module.weight.data.cpu().float()
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None

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
    def apply_quantization(self, X, importance_scores, keep_ratio=0.5):
        """Apply quantization and compute output."""
        W = self.target_module.weight.data.cpu().float().clone()
        b = self.target_module.bias.data.cpu().float().clone() if self.target_module.bias is not None else None

        out_features = W.shape[0]
        k = max(1, int(out_features * keep_ratio))
        top_k_indices = torch.topk(importance_scores, k).indices

        mask_keep = torch.zeros(out_features, dtype=torch.bool)
        mask_keep[top_k_indices] = True

        W_quantized = W.clone()
        for c in range(out_features):
            if not mask_keep[c]:
                w_channel = W[c, :]
                w_range = w_channel.abs().max()
                if w_range > 0:
                    scale = w_range / 7.0
                    w_quant = torch.round(w_channel / scale).clamp(-8, 7)
                    W_quantized[c, :] = w_quant * scale
                    noise = torch.randn_like(w_channel) * scale * 0.1
                    W_quantized[c, :] += noise

        Z_quantized = torch.matmul(X, W_quantized.t())
        if b is not None:
            Z_quantized = Z_quantized + b

        return Z_quantized

    def test_layer(self, calib_texts, val_texts, n_calib=500, n_val=2000, keep_ratio=0.5):
        """Test both methods on this layer."""
        # Collect calibration data
        self.register_hook(store_output=False)
        self.collect_data(calib_texts, n_calib, store_output=False)
        self.remove_hook()
        calib_inputs = self.input_activations.copy()

        # Collect validation data
        self.register_hook(store_output=True)
        self.collect_data(val_texts, n_val, store_output=True)
        self.remove_hook()
        val_inputs = self.input_activations.copy()
        val_outputs = self.output_ground_truth.copy()

        # Compute importance from calibration data
        X_calib = torch.cat(calib_inputs, dim=0)
        awq_importance = self.compute_awq_importance(X_calib)
        praq_importance = self.compute_praq_importance(X_calib)

        # Evaluate on validation data
        X_val = torch.cat(val_inputs, dim=0)
        Y_gt = torch.cat(val_outputs, dim=0)

        Y_awq = self.apply_quantization(X_val, awq_importance, keep_ratio)
        Y_praq = self.apply_quantization(X_val, praq_importance, keep_ratio)

        mse_awq = F.mse_loss(Y_awq, Y_gt).item()
        mse_praq = F.mse_loss(Y_praq, Y_gt).item()

        return {
            'mse_awq': mse_awq,
            'mse_praq': mse_praq,
            'improvement': ((mse_awq - mse_praq) / mse_awq) * 100 if mse_awq > 0 else 0
        }


def find_mlp_layers(model, num_blocks=3):
    """
    Find MLP layers to test. Tests all 3 components (gate_proj, up_proj, down_proj)
    from multiple transformer blocks.

    Args:
        num_blocks: Number of transformer blocks to sample

    Returns:
        List of layer names covering all MLP components
    """
    # Group by transformer block
    blocks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name_lower = name.lower()
            if 'mlp' in name_lower:
                # Extract block number (e.g., "model.layers.5.mlp.gate_proj" -> 5)
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        block_id = int(parts[i + 1])
                        if block_id not in blocks:
                            blocks[block_id] = []
                        blocks[block_id].append(name)
                        break

    # Sample blocks evenly across model depth
    sorted_blocks = sorted(blocks.keys())
    if len(sorted_blocks) > num_blocks:
        step = len(sorted_blocks) // num_blocks
        selected_blocks = [sorted_blocks[i * step] for i in range(num_blocks)]
    else:
        selected_blocks = sorted_blocks

    # Collect all layers from selected blocks
    selected_layers = []
    for block_id in selected_blocks:
        # Sort to get consistent order: gate_proj, up_proj, down_proj
        block_layers = sorted(blocks[block_id])
        selected_layers.extend(block_layers)

    return selected_layers


def main():
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_calib = 500
    n_val = 2000
    keep_ratio = 0.5

    print("=" * 80)
    print("MULTI-LAYER MSE COMPARISON: AWQ vs FastRPRAQ")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Calibration samples: {n_calib}")
    print(f"Validation samples: {n_val}")
    print(f"Keep ratio: {keep_ratio}")
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

    # Find layers to test (3 blocks × 3 layers each = 9 layers)
    mlp_layers = find_mlp_layers(model, num_blocks=3)
    print(f"\nTesting {len(mlp_layers)} MLP layers from 3 blocks:")
    print(f"(Each block has 3 layers: gate_proj, up_proj, down_proj)")
    for layer in mlp_layers:
        print(f"  - {layer}")

    # Load data
    print("\nLoading datasets...")
    calib_texts = load_wikitext2("train", n_samples=n_calib)
    val_texts = load_wikitext2("validation", n_samples=n_val)

    # Test each layer
    results = []
    print("\n" + "=" * 80)
    print("TESTING LAYERS")
    print("=" * 80)

    for layer_name in mlp_layers:
        print(f"\nTesting: {layer_name}")
        try:
            tester = LayerTester(model, tokenizer, layer_name, device)
            result = tester.test_layer(calib_texts, val_texts, n_calib, n_val, keep_ratio)
            result['layer'] = layer_name
            results.append(result)

            print(f"  AWQ MSE:     {result['mse_awq']:.6f}")
            print(f"  PRAQ MSE:    {result['mse_praq']:.6f}")
            print(f"  Improvement: {result['improvement']:+.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        praq_wins = sum(1 for r in results if r['improvement'] > 0)

        print(f"\nLayers tested: {len(results)}")
        print(f"PRAQ wins: {praq_wins}/{len(results)}")
        print(f"Average improvement: {avg_improvement:+.2f}%")

        print("\n" + "-" * 80)
        print(f"{'Layer':<50} {'AWQ MSE':<12} {'PRAQ MSE':<12} {'Improvement':<12}")
        print("-" * 80)
        for r in results:
            layer_short = r['layer'].split('.')[-2] + '.' + r['layer'].split('.')[-1]
            print(f"{layer_short:<50} {r['mse_awq']:<12.6f} {r['mse_praq']:<12.6f} {r['improvement']:+11.2f}%")
        print("-" * 80)


if __name__ == "__main__":
    main()
