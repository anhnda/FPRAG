import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os


class FastRPRAQQuantizer:
    """
    Fast-R-PRAQ v3 Quantizer for Transformer Models.
    Applies risk-aware importance scoring to linear layers.
    """

    def __init__(self, model, tokenizer, device="cuda", beta=3.0, tau=-3.0,
                 noise_factor=0.2, group_size=32, bits=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.tau = tau
        self.noise_factor = noise_factor
        self.group_size = group_size
        self.bits = bits

        # Storage for activations
        self.activation_data = {}
        self.hooks = []

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
    def compute_importance_scores(self, name, module):
        """
        Compute Fast-R-PRAQ v3 importance scores for a linear layer.

        Args:
            name: Layer name
            module: Linear layer module

        Returns:
            importance_scores: Tensor of shape [out_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            # No calibration data, use uniform importance
            return torch.ones(module.out_features)

        # Concatenate all activation samples
        X_list = self.activation_data[name]
        X = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)
        X = X.to(self.device)

        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else torch.zeros(module.out_features, device=self.device)

        # Compute pre-activation values: Z = XW^T + b
        Z = torch.matmul(X, W.t()) + b

        # Step A: Compute signal statistics
        z_mean = Z.mean(dim=0)
        z_std = Z.std(dim=0) + 1e-8
        z_upper = z_mean + 3 * z_std  # 3-sigma safety margin

        # Step B: Sensitivity check (estimate quantization noise impact)
        x_mag = X.abs().mean()
        w_mag = W.abs().mean(dim=1)  # Per output channel

        estimated_noise_impact = x_mag * w_mag * self.noise_factor

        # Risk-adjusted upper bound
        z_risk_upper = z_upper + estimated_noise_impact

        # Step C: Probability of activation (using sigmoid)
        prob_active = torch.sigmoid(self.beta * (z_risk_upper - self.tau))

        # Step D: Utility magnitude
        magnitude = Z.abs().mean(dim=0) + z_std

        raw_importance = prob_active * magnitude

        # Step E: Hardware grouping
        C_out = raw_importance.shape[0]
        if self.group_size > 0 and C_out % self.group_size == 0:
            grouped = raw_importance.view(-1, self.group_size)
            group_scores = grouped.sum(dim=1)
            final_importance = group_scores.repeat_interleave(self.group_size)
        else:
            final_importance = raw_importance

        return final_importance.cpu()

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
                # Simulate INT4 quantization with noise
                w_channel = W[c, :]
                w_range = w_channel.abs().max()

                # Quantize to INT4 range [-8, 7]
                scale = w_range / 7.0
                if scale > 0:
                    w_quant = torch.round(w_channel / scale).clamp(-8, 7)
                    W[c, :] = w_quant * scale

                    # Add quantization noise
                    noise = torch.randn_like(w_channel) * scale * 0.1
                    W[c, :] += noise

    def calibrate(self, calibration_data, n_samples=500):
        """
        Run calibration on the dataset to collect activations.

        Args:
            calibration_data: List of text samples
            n_samples: Number of samples to use
        """
        print(f"Calibrating with {min(n_samples, len(calibration_data))} samples...")
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
        Quantize all linear layers in the model using Fast-R-PRAQ.

        Args:
            keep_ratio: Fraction of channels to keep in higher precision
        """
        print("Computing importance scores and quantizing layers...")

        for name, module in tqdm(list(self.model.named_modules()), desc="Quantizing"):
            if isinstance(module, nn.Linear):
                # Compute importance scores
                importance_scores = self.compute_importance_scores(name, module)

                # Quantize the layer
                self.quantize_layer(module, importance_scores, keep_ratio)

        print("Quantization complete!")

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
    output_dir = "./quantized_models/minicpm_praq"

    print(f"Device: {device}")
    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=n_calib_samples)

    # Initialize quantizer
    quantizer = FastRPRAQQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beta=3.0,
        tau=-3.0,
        noise_factor=0.2,
        group_size=32,
        bits=4
    )

    # Calibrate
    quantizer.calibrate(calib_texts, n_samples=n_calib_samples)

    # Quantize
    quantizer.quantize_model(keep_ratio=keep_ratio)

    # Save quantized model
    print(f"Saving quantized model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done!")
    print(f"Quantized model saved to: {output_dir}")


if __name__ == "__main__":
    main()
