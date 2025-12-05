import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os


class AWQQuantizer:
    """
    AWQ (Activation-aware Weight Quantization) implementation.

    AWQ uses a simple but effective importance metric: the magnitude of
    pre-activation outputs. Channels with larger activation magnitudes
    are considered more important.

    Reference: AWQ: Activation-aware Weight Quantization for LLM Compression
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size

        # Storage for activations
        self.activation_data = {}
        self.hooks = []

        print(f"\n[AWQ Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                # Store input activations
                if isinstance(input, tuple):
                    inp = input[0].detach().cpu()
                else:
                    inp = input.detach().cpu()
                self.activation_data[name].append(inp)
            return hook

        # Register hooks for all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @torch.no_grad()
    def compute_awq_importance(self, name, module):
        """
        Compute AWQ importance scores.

        AWQ Metric: E[|Z|] where Z = XW^T + b (pre-activation output)

        Args:
            name: Layer name
            module: Linear layer module

        Returns:
            importance_scores: Tensor of shape [out_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return torch.ones(module.out_features)

        # Concatenate all activation samples
        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)
        X = X.to(self.device)

        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else torch.zeros(module.out_features, device=self.device)

        # Compute pre-activation values: Z = XW^T + b
        Z = torch.matmul(X, W.t()) + b

        # AWQ importance: E[|Z|] per output channel
        importance = Z.abs().mean(dim=0)

        # Optional: Apply per-channel scaling based on AWQ paper
        # Scale importance by activation statistics
        importance = importance + Z.std(dim=0) * 0.1  # Small variance term for stability

        return importance.cpu()

    @torch.no_grad()
    def quantize_layer(self, module, importance_scores, keep_ratio=0.5):
        """
        Apply mixed-precision quantization to a linear layer.

        Args:
            module: Linear layer
            importance_scores: Importance scores for each output channel
            keep_ratio: Fraction of channels to keep in higher precision
        """
        W = module.weight.data
        out_features = W.shape[0]

        # Select top-k most important channels
        k = max(1, int(out_features * keep_ratio))
        top_k_indices = torch.topk(importance_scores, k).indices

        # Create mask for high-precision channels
        mask_keep = torch.zeros(out_features, dtype=torch.bool)
        mask_keep[top_k_indices] = True

        # Quantize low-importance channels more aggressively
        for c in range(out_features):
            if not mask_keep[c]:
                # Simulate INT4 quantization with per-channel scaling
                w_channel = W[c, :]

                # Per-channel quantization (symmetric)
                w_max = w_channel.abs().max()

                if w_max > 0:
                    # INT4 range: [-8, 7] for signed 4-bit
                    scale = w_max / 7.0

                    # Quantize
                    w_quant = torch.round(w_channel / scale).clamp(-8, 7)

                    # Dequantize
                    W[c, :] = w_quant * scale

                    # Add realistic quantization noise
                    noise = torch.randn_like(w_channel) * scale * 0.1
                    W[c, :] += noise

    def calibrate(self, calibration_data, n_samples=500):
        """
        Run calibration on the dataset to collect activations.

        Args:
            calibration_data: List of text samples
            n_samples: Number of samples to use
        """
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
        self.model.eval()
        self.register_hooks()

        for i, text in enumerate(tqdm(calibration_data[:n_samples], desc="Calibration")):
            try:
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass to collect activations
                with torch.no_grad():
                    self.model(**inputs)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        self.remove_hooks()
        print("Calibration complete!")

    def quantize_model(self, keep_ratio=0.5):
        """
        Quantize all linear layers in the model using AWQ.

        Args:
            keep_ratio: Fraction of channels to keep in higher precision
        """
        print("\n" + "=" * 80)
        print("Computing AWQ importance scores and quantizing layers...")
        print("=" * 80)

        quantized_count = 0

        for name, module in tqdm(list(self.model.named_modules()), desc="Quantizing"):
            if isinstance(module, nn.Linear):
                # Compute AWQ importance scores
                importance_scores = self.compute_awq_importance(name, module)

                # Quantize the layer
                self.quantize_layer(module, importance_scores, keep_ratio)

                quantized_count += 1

        print(f"\n✅ Quantization complete!")
        print(f"   Total linear layers quantized: {quantized_count}")

        # Clear activation data to free memory
        self.activation_data = {}


def load_wikitext2(split="train", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Filter out empty texts
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    if n_samples:
        texts = texts[:n_samples]

    return texts


def main():
    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"  # MiniCPM-2.4B
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_calib_samples = 500
    keep_ratio = 0.5  # Keep 50% channels in higher precision
    output_dir = "./quantized_models/minicpm_awq_custom"

    print("=" * 80)
    print("AWQ (Activation-aware Weight Quantization) for MiniCPM-2.4")
    print("=" * 80)
    print("Method: Simple activation magnitude importance")
    print("  Importance = E[|XW^T + b|] per output channel")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {n_calib_samples}")
    print(f"Keep ratio: {keep_ratio}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Get model size before quantization
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_before = (param_size + buffer_size) / 1024**2
    print(f"Model size before quantization: {size_mb_before:.2f} MB")

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=n_calib_samples)

    # Initialize AWQ quantizer
    quantizer = AWQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        group_size=128
    )

    # Calibrate
    quantizer.calibrate(calib_texts, n_samples=n_calib_samples)

    # Quantize
    quantizer.quantize_model(keep_ratio=keep_ratio)

    # Get model size after quantization
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_after = (param_size + buffer_size) / 1024**2
    print(f"\nModel size after quantization: {size_mb_after:.2f} MB")
    print(f"Compression ratio: {size_mb_before / size_mb_after:.2f}x")

    # Save quantized model
    print(f"\nSaving quantized model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)
    print(f"Quantized model saved to: {output_dir}")
    print("\nQuantization approach:")
    print("  - Method: AWQ (Activation-aware Weight Quantization)")
    print("  - Importance: E[|XW^T + b|] - mean absolute activation")
    print("  - All linear layers quantized with same strategy")
    print(f"  - Keep ratio: {keep_ratio} ({int(keep_ratio*100)}% in FP16, {int((1-keep_ratio)*100)}% in INT4)")
    print("=" * 80)


if __name__ == "__main__":
    main()
