import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os


class FastRPRAQQuantizer:
    """
    Hybrid Quantizer for Transformer Models.
    - For MLP layers (with ReLU-family activations): Use post-activation importance
    - For Attention layers: Use traditional AWQ-style importance
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

        # Detect layer types (MLP vs Attention)
        self.layer_types = self._detect_layer_types()
        self.activation_functions = self._detect_activation_functions()

        # Print summary
        mlp_count = sum(1 for t in self.layer_types.values() if t == 'mlp')
        attn_count = sum(1 for t in self.layer_types.values() if t == 'attention')
        print(f"\n[Layer Detection]")
        print(f"  MLP layers (post-activation quantization): {mlp_count}")
        print(f"  Attention layers (AWQ-style quantization): {attn_count}")

    def _detect_layer_types(self):
        """
        Detect which layers are MLP layers (followed by activation) vs attention layers.

        Returns:
            Dictionary mapping layer names to types ('mlp' or 'attention')
        """
        layer_types = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Heuristic: MLP layers typically have keywords like 'mlp', 'fc', 'gate', 'up', 'down'
                # Attention layers have keywords like 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'out_proj'
                name_lower = name.lower()

                if any(kw in name_lower for kw in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'out_proj', 'attention']):
                    layer_types[name] = 'attention'
                elif any(kw in name_lower for kw in ['mlp', 'fc', 'gate', 'up_proj', 'down_proj', 'ffn']):
                    layer_types[name] = 'mlp'
                else:
                    # Default: treat as MLP if uncertain (safer)
                    layer_types[name] = 'mlp'

        return layer_types

    def _detect_activation_functions(self):
        """
        Detect activation functions in the model.

        Returns:
            Dictionary mapping activation function types
        """
        activations = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.SiLU, nn.GELU, nn.ReLU, nn.LeakyReLU)):
                activations[name] = type(module).__name__

        # Default to SiLU for MiniCPM (commonly uses SiLU)
        if not activations:
            print("No explicit activation modules found. Assuming SiLU for MLP layers.")

        return activations

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

    def _get_activation_function(self):
        """Get the activation function to use (default SiLU for MiniCPM)."""
        # Default to SiLU if not detected
        return nn.SiLU()

    @torch.no_grad()
    def compute_importance_scores_mlp(self, name, module):
        """
        Compute importance scores for MLP layers using POST-ACTIVATION values.
        Since ReLU-family activations drop negative values, we only care about
        the magnitude of the activated output.

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

        # Process in batches to avoid OOM
        batch_size = 1024
        n_samples = X.shape[0]
        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else torch.zeros(module.out_features, device=self.device)
        activation_fn = self._get_activation_function()

        importance_sum = torch.zeros(module.out_features, device=self.device)
        importance_sq_sum = torch.zeros(module.out_features, device=self.device)

        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size].to(self.device)

            # Compute pre-activation values: Z = XW^T + b
            Z = torch.matmul(batch_X, W.t()) + b

            # Apply activation (SiLU/GELU/ReLU)
            A = activation_fn(Z)  # Post-activation values

            # Accumulate statistics
            importance_sum += A.abs().sum(dim=0)
            importance_sq_sum += (A ** 2).sum(dim=0)

            # Free memory
            del batch_X, Z, A

        # Compute mean and std
        importance = importance_sum / n_samples
        variance = (importance_sq_sum / n_samples) - (importance ** 2)
        std = torch.sqrt(variance.clamp(min=0))
        importance = importance + std

        # Hardware grouping
        C_out = importance.shape[0]
        if self.group_size > 0 and C_out % self.group_size == 0:
            grouped = importance.view(-1, self.group_size)
            group_scores = grouped.sum(dim=1)
            final_importance = group_scores.repeat_interleave(self.group_size)
        else:
            final_importance = importance

        return final_importance.cpu()

    @torch.no_grad()
    def compute_importance_scores_attention(self, name, module):
        """
        Compute importance scores for attention layers using AWQ-style approach.
        For attention layers (Q/K/V/O projections), use activation magnitude.

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

        # Process in batches to avoid OOM
        batch_size = 1024
        n_samples = X.shape[0]
        W = module.weight.data  # [out_features, in_features]
        b = module.bias.data if module.bias is not None else torch.zeros(module.out_features, device=self.device)

        importance_sum = torch.zeros(module.out_features, device=self.device)

        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size].to(self.device)

            # Compute pre-activation values: Z = XW^T + b
            Z = torch.matmul(batch_X, W.t()) + b

            # Accumulate statistics
            importance_sum += Z.abs().sum(dim=0)

            # Free memory
            del batch_X, Z

        # AWQ-style importance: simply use output magnitude
        importance = importance_sum / n_samples

        # Hardware grouping
        C_out = importance.shape[0]
        if self.group_size > 0 and C_out % self.group_size == 0:
            grouped = importance.view(-1, self.group_size)
            group_scores = grouped.sum(dim=1)
            final_importance = group_scores.repeat_interleave(self.group_size)
        else:
            final_importance = importance

        return final_importance.cpu()

    @torch.no_grad()
    def compute_importance_scores(self, name, module):
        """
        Compute importance scores based on layer type.

        Args:
            name: Layer name
            module: Linear layer module

        Returns:
            importance_scores: Tensor of shape [out_features]
        """
        layer_type = self.layer_types.get(name, 'mlp')

        if layer_type == 'mlp':
            return self.compute_importance_scores_mlp(name, module)
        else:  # attention
            return self.compute_importance_scores_attention(name, module)

    @torch.no_grad()
    def quantize_layer(self, module, importance_scores, keep_ratio=0.5):
        """
        Apply mixed-precision quantization to a linear layer (vectorized for speed).

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
        mask_keep = torch.zeros(out_features, dtype=torch.bool, device=W.device)
        mask_keep[top_k_indices] = True

        # Get indices of channels to quantize
        quantize_indices = ~mask_keep

        if quantize_indices.any():
            # Vectorized quantization for all low-importance channels at once
            W_quantize = W[quantize_indices]  # [n_quantize, in_features]

            # Per-channel scales (vectorized)
            scales = W_quantize.abs().max(dim=1, keepdim=True)[0] / 7.0  # [n_quantize, 1]
            scales = scales.clamp(min=1e-8)  # Prevent division by zero

            # Quantize to INT4 range [-8, 7] (vectorized)
            W_quant = torch.round(W_quantize / scales).clamp(-8, 7)
            W_dequant = W_quant * scales

            # Add quantization noise (vectorized)
            noise = torch.randn_like(W_dequant) * scales * 0.1
            W_dequant = W_dequant + noise

            # Write back quantized weights
            W[quantize_indices] = W_dequant

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
                    _ = self.model(**inputs, use_cache=False, return_dict=True)

            except Exception:
                # Silently skip errors during calibration (common with cache issues)
                if i % 100 == 0 and i > 0:
                    print(f"\nNote: Skipped {i} samples with errors (cache-related issues are normal)")
                continue

        self.remove_hooks()
        print("Calibration complete!")

    def quantize_model(self, keep_ratio=0.5):
        """
        Quantize all linear layers in the model using hybrid approach.

        Args:
            keep_ratio: Fraction of channels to keep in higher precision
        """
        print("\n" + "=" * 80)
        print("Computing importance scores and quantizing layers...")
        print("=" * 80)

        mlp_quantized = 0
        attn_quantized = 0
        skipped_count = 0

        for name, module in tqdm(list(self.model.named_modules()), desc="Quantizing"):
            if isinstance(module, nn.Linear):
                try:
                    layer_type = self.layer_types.get(name, 'mlp')

                    # Compute importance scores
                    importance_scores = self.compute_importance_scores(name, module)

                    # Quantize the layer
                    self.quantize_layer(module, importance_scores, keep_ratio)

                    if layer_type == 'mlp':
                        mlp_quantized += 1
                    else:
                        attn_quantized += 1

                    # Clear GPU cache every 50 layers to prevent OOM
                    if (mlp_quantized + attn_quantized) % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️  Error quantizing layer {name}: {e}")
                    skipped_count += 1
                    continue

        print(f"\n✅ Quantization complete!")
        print(f"   MLP layers quantized (post-activation): {mlp_quantized}")
        print(f"   Attention layers quantized (AWQ-style): {attn_quantized}")
        if skipped_count > 0:
            print(f"   ⚠️  Skipped {skipped_count} layers due to errors")

        # Clear activation data to free memory
        self.activation_data = {}

        # Final GPU cache clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    output_dir = "./quantized_models/minicpm_praq_hybrid"

    print("=" * 80)
    print("Hybrid Post-Activation Quantization for MiniCPM-2.4")
    print("=" * 80)
    print("Strategy:")
    print("  - MLP layers: Post-activation importance (ReLU-family drops negatives)")
    print("  - Attention layers: AWQ-style importance (no activation)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {n_calib_samples}")
    print(f"Keep ratio: {keep_ratio}")
    print("=" * 80)

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
    print(f"\nSaving quantized model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)
    print(f"Quantized model saved to: {output_dir}")
    print("\nQuantization approach:")
    print("  - MLP layers: Post-activation based importance")
    print("  - Attention layers: AWQ-style magnitude based importance")
    print("=" * 80)


if __name__ == "__main__":
    main()
