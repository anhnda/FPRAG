import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random
import argparse


class LayerMSEComparator:
    """
    Compare MSE on a specific layer's output between AWQ and FastRPRAQ
    quantization strategies.
    """

    def __init__(self, model, tokenizer, layer_name, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        self.device = device

        # Storage for activations
        self.input_activations = []  # Input to the linear layer
        self.output_ground_truth = []  # Original output (ground truth)

        # Find target module
        self.target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.target_module = module
                break

        if self.target_module is None:
            raise ValueError(f"Layer {layer_name} not found!")

        if not isinstance(self.target_module, nn.Linear):
            raise ValueError(f"Layer {layer_name} is not a Linear layer!")

        print(f"Target layer: {layer_name}")
        print(f"  Weight shape: {self.target_module.weight.shape}")
        print(f"  Bias: {self.target_module.bias is not None}")

        self.hook = None

    def register_calibration_hook(self):
        """Register hook to collect calibration data."""
        def hook_fn(module, input, output):
            # Capture input activations
            if isinstance(input, tuple):
                inp = input[0].detach()
            else:
                inp = input.detach()

            # Reshape to [batch * seq, hidden]
            inp_flat = inp.reshape(-1, inp.shape[-1])
            self.input_activations.append(inp_flat.cpu().float())

        self.hook = self.target_module.register_forward_hook(hook_fn)
        print(f"Registered calibration hook on {self.layer_name}")

    def register_validation_hook(self, store_output=True):
        """Register hook to collect validation data (inputs and ground truth outputs)."""
        def hook_fn(module, input, output):
            # Capture input activations
            if isinstance(input, tuple):
                inp = input[0].detach()
            else:
                inp = input.detach()

            # Reshape to [batch * seq, hidden]
            inp_flat = inp.reshape(-1, inp.shape[-1])
            self.input_activations.append(inp_flat.cpu().float())

            # Capture ground truth output
            if store_output:
                if isinstance(output, tuple):
                    out = output[0].detach()
                else:
                    out = output.detach()
                out_flat = out.reshape(-1, out.shape[-1])
                self.output_ground_truth.append(out_flat.cpu().float())

        self.hook = self.target_module.register_forward_hook(hook_fn)
        print(f"Registered validation hook on {self.layer_name}")

    def remove_hook(self):
        """Remove hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    @torch.no_grad()
    def collect_calibration_data(self, texts, n_samples=500):
        """
        Collect calibration data by running texts through the model.

        Args:
            texts: List of text samples
            n_samples: Number of samples to use
        """
        print(f"\nCollecting calibration data ({n_samples} samples)...")
        self.input_activations = []
        self.model.eval()

        successful = 0
        for text in tqdm(texts[:n_samples], desc="Calibration"):
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                self.model(**inputs, use_cache=False)
                successful += 1

            except Exception as e:
                continue

        print(f"Calibration complete! Processed {successful} samples")
        print(f"Collected {len(self.input_activations)} activation batches")

    @torch.no_grad()
    def collect_validation_data(self, texts, n_samples=2000):
        """
        Collect validation data (inputs + ground truth outputs).

        Args:
            texts: List of text samples
            n_samples: Number of samples to use
        """
        print(f"\nCollecting validation data ({n_samples} samples)...")
        self.input_activations = []
        self.output_ground_truth = []
        self.model.eval()

        successful = 0
        for text in tqdm(texts[:n_samples], desc="Validation"):
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                self.model(**inputs, use_cache=False)
                successful += 1

            except Exception as e:
                continue

        print(f"Validation data collection complete! Processed {successful} samples")
        print(f"Collected {len(self.input_activations)} activation batches")
        print(f"Collected {len(self.output_ground_truth)} output batches")

    @torch.no_grad()
    def compute_awq_importance(self, X):
        """
        Compute AWQ-style importance scores.
        Uses the original AWQ metric: E[|XW^T + b|]

        Args:
            X: Input activations [tokens, in_features]

        Returns:
            Importance scores [out_features]
        """
        W = self.target_module.weight.data.cpu().float()  # [out_features, in_features]
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None

        # Compute pre-activation: Z = XW^T + b
        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b

        # AWQ importance: E[|Z|] per channel (original AWQ paper)
        importance = Z.abs().mean(dim=0)

        return importance

    @torch.no_grad()
    def compute_praq_importance(self, X, beta=3.0, tau=-3.0, noise_factor=0.2):
        """
        Compute FastRPRAQ-style importance scores with risk-awareness.

        Args:
            X: Input activations [tokens, in_features]
            beta: Temperature for probability calculation
            tau: Activation threshold (e.g., -3.0 for SiLU)
            noise_factor: Quantization noise ratio

        Returns:
            Importance scores [out_features]
        """
        W = self.target_module.weight.data.cpu().float()  # [out_features, in_features]
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None

        # Compute pre-activation: Z = XW^T + b
        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b

        # Compute signal statistics
        z_mean = Z.mean(dim=0)
        z_std = Z.std(dim=0) + 1e-8
        z_upper = z_mean + 3 * z_std  # 3-sigma safety margin

        # Sensitivity check: estimate quantization noise impact
        x_mag = X.abs().mean()
        w_mag = W.abs().mean(dim=1)  # Per output channel

        estimated_noise_impact = x_mag * w_mag * noise_factor

        # Risk-adjusted upper bound
        z_risk_upper = z_upper + estimated_noise_impact

        # Probability of activation
        prob_active = torch.sigmoid(beta * (z_risk_upper - tau))

        # Utility magnitude
        magnitude = Z.abs().mean(dim=0) + z_std

        # Final importance
        raw_importance = prob_active * magnitude

        return raw_importance

    @torch.no_grad()
    def apply_quantization(self, X, importance_scores, keep_ratio=0.5):
        """
        Apply mixed-precision quantization to the layer and compute output.

        Args:
            X: Input activations [tokens, in_features]
            importance_scores: Importance scores [out_features]
            keep_ratio: Fraction of channels to keep in FP16

        Returns:
            Quantized output [tokens, out_features]
        """
        W = self.target_module.weight.data.cpu().float().clone()
        b = self.target_module.bias.data.cpu().float().clone() if self.target_module.bias is not None else None

        out_features = W.shape[0]

        # Select top-k channels to keep in FP16
        k = max(1, int(out_features * keep_ratio))
        top_k_indices = torch.topk(importance_scores, k).indices

        # Create mask
        mask_keep = torch.zeros(out_features, dtype=torch.bool)
        mask_keep[top_k_indices] = True

        # Quantize low-importance channels
        W_quantized = W.clone()

        for c in range(out_features):
            if not mask_keep[c]:
                # Simulate INT4 quantization
                w_channel = W[c, :]
                w_range = w_channel.abs().max()

                if w_range > 0:
                    # Quantize to INT4 range [-8, 7]
                    scale = w_range / 7.0
                    w_quant = torch.round(w_channel / scale).clamp(-8, 7)
                    W_quantized[c, :] = w_quant * scale

                    # Add quantization noise
                    noise = torch.randn_like(w_channel) * scale * 0.1
                    W_quantized[c, :] += noise

        # Compute output with quantized weights
        Z_quantized = torch.matmul(X, W_quantized.t())
        if b is not None:
            Z_quantized = Z_quantized + b

        return Z_quantized

    def evaluate_mse(self, keep_ratio=0.5):
        """
        Evaluate MSE for both AWQ and FastRPRAQ approaches.

        Args:
            keep_ratio: Fraction of channels to keep in FP16

        Returns:
            Dictionary with MSE results
        """
        print("\n" + "=" * 80)
        print("EVALUATING MSE ON LAYER OUTPUT")
        print("=" * 80)

        # Concatenate all calibration data
        X_calib = torch.cat(self.input_activations, dim=0)
        print(f"\nCalibration data shape: {X_calib.shape}")

        # Compute importance scores
        print("\nComputing AWQ importance scores...")
        awq_importance = self.compute_awq_importance(X_calib)

        print("Computing FastRPRAQ importance scores...")
        praq_importance = self.compute_praq_importance(X_calib)

        # Validation: concatenate ground truth
        Y_gt = torch.cat(self.output_ground_truth, dim=0)
        print(f"Ground truth output shape: {Y_gt.shape}")

        # For validation, we need to recompute inputs through quantized layer
        # We'll use the stored input activations from validation
        print("\n" + "=" * 80)
        print("EVALUATING AWQ QUANTIZATION")
        print("=" * 80)

        # Apply AWQ quantization
        Y_awq = self.apply_quantization(X_calib, awq_importance, keep_ratio)

        # Compute MSE on calibration set (for comparison)
        W_orig = self.target_module.weight.data.cpu().float()
        b_orig = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None
        Y_orig_calib = torch.matmul(X_calib, W_orig.t())
        if b_orig is not None:
            Y_orig_calib = Y_orig_calib + b_orig

        mse_awq_calib = F.mse_loss(Y_awq, Y_orig_calib).item()
        print(f"AWQ MSE (calibration): {mse_awq_calib:.6f}")

        # Now evaluate on validation data
        # We need to apply quantization to validation inputs
        X_val = torch.cat(self.input_activations, dim=0)
        Y_awq_val = self.apply_quantization(X_val, awq_importance, keep_ratio)
        mse_awq_val = F.mse_loss(Y_awq_val, Y_gt).item()
        print(f"AWQ MSE (validation): {mse_awq_val:.6f}")

        print("\n" + "=" * 80)
        print("EVALUATING FastRPRAQ QUANTIZATION")
        print("=" * 80)

        # Apply FastRPRAQ quantization
        Y_praq = self.apply_quantization(X_calib, praq_importance, keep_ratio)
        mse_praq_calib = F.mse_loss(Y_praq, Y_orig_calib).item()
        print(f"FastRPRAQ MSE (calibration): {mse_praq_calib:.6f}")

        # Validation
        Y_praq_val = self.apply_quantization(X_val, praq_importance, keep_ratio)
        mse_praq_val = F.mse_loss(Y_praq_val, Y_gt).item()
        print(f"FastRPRAQ MSE (validation): {mse_praq_val:.6f}")

        # Results
        results = {
            'awq_mse_calibration': mse_awq_calib,
            'awq_mse_validation': mse_awq_val,
            'praq_mse_calibration': mse_praq_calib,
            'praq_mse_validation': mse_praq_val,
            'keep_ratio': keep_ratio,
            'calibration_tokens': X_calib.shape[0],
            'validation_tokens': Y_gt.shape[0],
        }

        return results


def load_wikitext2(split="train", n_samples=None, seed=42):
    """Load WikiText-2 dataset with optional sampling."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Filter out empty texts
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    # Random sampling for validation
    if n_samples and split == "validation":
        random.seed(seed)
        if n_samples < len(texts):
            texts = random.sample(texts, n_samples)
    elif n_samples:
        texts = texts[:n_samples]

    print(f"Loaded {len(texts)} samples")
    return texts


def find_layer_by_id(model, layer_id):
    """
    Find a layer by its ID in the model.

    Args:
        model: The model
        layer_id: Layer number to find (e.g., 28)

    Returns:
        Layer name or None
    """
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    # Try to find MLP layer with this ID
    target_layer_name = None
    for name in layer_names:
        if f".{layer_id}." in name or f"layers.{layer_id}." in name:
            # Prefer gate_proj or up_proj in MLP
            if 'gate' in name.lower() or 'up_proj' in name.lower():
                target_layer_name = name
                break

    # If not found, try any layer with that ID
    if target_layer_name is None:
        for name in layer_names:
            if f".{layer_id}." in name:
                target_layer_name = name
                break

    return target_layer_name


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compare AWQ vs FastRPRAQ on a specific layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        default=28,
        help="Transformer layer ID to analyze (e.g., 0, 16, 28)"
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.5,
        help="Fraction of channels to keep in FP16 (e.g., 0.2, 0.5, 0.7)"
    )
    parser.add_argument(
        "--layer-type",
        type=str,
        default="gate",
        choices=["gate", "up", "down"],
        help="MLP layer type to analyze (gate_proj, up_proj, or down_proj)"
    )
    parser.add_argument(
        "--n-calib",
        type=int,
        default=500,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=2000,
        help="Number of validation samples"
    )
    args = parser.parse_args()

    # Configuration
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    target_layer_id = args.layer_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_calib_samples = args.n_calib
    n_val_samples = args.n_val
    keep_ratio = args.keep_ratio
    seed = 42

    print("=" * 80)
    print(f"LAYER-LEVEL MSE COMPARISON: AWQ vs FastRPRAQ")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Target layer: {target_layer_id} ({args.layer_type}_proj)")
    print(f"Device: {device}")
    print(f"Calibration samples: {n_calib_samples}")
    print(f"Validation samples: {n_val_samples}")
    print(f"Keep ratio: {keep_ratio} ({int(keep_ratio*100)}% FP16, {int((1-keep_ratio)*100)}% INT4)")
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
    model.eval()

    # Find target layer
    print(f"\nSearching for layer {target_layer_id} ({args.layer_type}_proj)...")

    # Find specific layer type
    layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if f".{target_layer_id}." in name and f"{args.layer_type}_proj" in name.lower():
                layer_name = name
                break

    if layer_name is None:
        print(f"ERROR: Could not find layer {target_layer_id} with type {args.layer_type}_proj")
        print(f"Try --layer-type gate, --layer-type up, or --layer-type down")
        return

    print(f"Found target layer: {layer_name}")

    # Load datasets
    calib_texts = load_wikitext2("train", n_samples=n_calib_samples, seed=seed)
    val_texts = load_wikitext2("validation", n_samples=n_val_samples, seed=seed)

    # Initialize comparator
    comparator = LayerMSEComparator(model, tokenizer, layer_name, device)

    # Step 1: Collect calibration data
    comparator.register_calibration_hook()
    comparator.collect_calibration_data(calib_texts, n_samples=n_calib_samples)
    comparator.remove_hook()

    # Store calibration data
    calib_inputs = comparator.input_activations.copy()

    # Step 2: Collect validation data (inputs + ground truth outputs)
    comparator.register_validation_hook(store_output=True)
    comparator.collect_validation_data(val_texts, n_samples=n_val_samples)
    comparator.remove_hook()

    # Store validation data
    val_inputs = comparator.input_activations.copy()
    val_outputs = comparator.output_ground_truth.copy()

    # Step 3: Evaluate MSE
    # Reset data for evaluation
    comparator.input_activations = calib_inputs  # Use calibration for importance scores
    comparator.output_ground_truth = val_outputs  # Use validation for MSE computation

    # But we need validation inputs too
    # Let's restructure: use calibration for computing importance, validation for MSE

    print("\n" + "=" * 80)
    print("COMPUTING IMPORTANCE SCORES FROM CALIBRATION DATA")
    print("=" * 80)

    # Concatenate calibration data
    X_calib = torch.cat(calib_inputs, dim=0)
    print(f"Calibration data shape: {X_calib.shape}")

    # Compute importance scores
    print("\nComputing AWQ importance scores...")
    awq_importance = comparator.compute_awq_importance(X_calib)

    print("Computing FastRPRAQ importance scores...")
    praq_importance = comparator.compute_praq_importance(X_calib)

    # Print importance statistics
    print("\n" + "=" * 80)
    print("IMPORTANCE SCORE STATISTICS")
    print("=" * 80)
    print(f"\nAWQ Importance:")
    print(f"  Min: {awq_importance.min():.4f}")
    print(f"  Max: {awq_importance.max():.4f}")
    print(f"  Mean: {awq_importance.mean():.4f}")
    print(f"  Std: {awq_importance.std():.4f}")

    print(f"\nFastRPRAQ Importance:")
    print(f"  Min: {praq_importance.min():.4f}")
    print(f"  Max: {praq_importance.max():.4f}")
    print(f"  Mean: {praq_importance.mean():.4f}")
    print(f"  Std: {praq_importance.std():.4f}")

    # Compute rank correlation
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(awq_importance.numpy(), praq_importance.numpy())
    print(f"\nRank Correlation (Spearman): {rank_corr:.4f}")

    print("\n" + "=" * 80)
    print("EVALUATING MSE ON VALIDATION DATA")
    print("=" * 80)

    # Concatenate validation data
    X_val = torch.cat(val_inputs, dim=0)
    Y_gt = torch.cat(val_outputs, dim=0)
    print(f"Validation input shape: {X_val.shape}")
    print(f"Validation output (ground truth) shape: {Y_gt.shape}")

    # Apply quantization and compute outputs
    print("\nApplying AWQ quantization...")
    Y_awq = comparator.apply_quantization(X_val, awq_importance, keep_ratio)
    mse_awq = F.mse_loss(Y_awq, Y_gt).item()

    print("Applying FastRPRAQ quantization...")
    Y_praq = comparator.apply_quantization(X_val, praq_importance, keep_ratio)
    mse_praq = F.mse_loss(Y_praq, Y_gt).item()

    # Results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nLayer: {layer_name}")
    print(f"Keep ratio: {keep_ratio} ({int(awq_importance.shape[0] * keep_ratio)}/{awq_importance.shape[0]} channels)")
    print(f"Validation tokens: {Y_gt.shape[0]}")

    print(f"\nAWQ MSE:          {mse_awq:.6f}")
    print(f"FastRPRAQ MSE:    {mse_praq:.6f}")

    print("\n" + "=" * 80)
    if mse_praq < mse_awq:
        improvement = ((mse_awq - mse_praq) / mse_awq) * 100
        print(f"✅ WINNER: FastRPRAQ")
        print(f"   FastRPRAQ achieves {improvement:.2f}% lower MSE than AWQ")
    elif mse_awq < mse_praq:
        degradation = ((mse_praq - mse_awq) / mse_awq) * 100
        print(f"✅ WINNER: AWQ")
        print(f"   AWQ achieves {degradation:.2f}% lower MSE than FastRPRAQ")
    else:
        print("🤝 TIE: Both methods achieve similar MSE")
    print("=" * 80)

    # Additional analysis: per-channel contribution to MSE
    print("\n" + "=" * 80)
    print("PER-CHANNEL MSE ANALYSIS")
    print("=" * 80)

    # Compute per-channel squared error
    se_awq = (Y_awq - Y_gt).pow(2).mean(dim=0)  # [out_features]
    se_praq = (Y_praq - Y_gt).pow(2).mean(dim=0)  # [out_features]

    # Find channels where FastRPRAQ is significantly better
    improvement_per_channel = se_awq - se_praq

    # Top 10 channels where FastRPRAQ wins
    top_praq_wins = torch.topk(improvement_per_channel, min(10, len(improvement_per_channel)))
    print("\nTop 10 channels where FastRPRAQ wins (largest error reduction):")
    for i, (improvement, ch_idx) in enumerate(zip(top_praq_wins.values, top_praq_wins.indices)):
        ch_idx = ch_idx.item()
        print(f"  {i+1}. Channel {ch_idx}: {improvement.item():.6f} "
              f"(AWQ importance rank: {(awq_importance >= awq_importance[ch_idx]).sum().item()}, "
              f"PRAQ importance rank: {(praq_importance >= praq_importance[ch_idx]).sum().item()})")

    # Top 10 channels where AWQ wins
    top_awq_wins = torch.topk(-improvement_per_channel, min(10, len(improvement_per_channel)))
    print("\nTop 10 channels where AWQ wins (largest error reduction):")
    for i, (improvement, ch_idx) in enumerate(zip(top_awq_wins.values, top_awq_wins.indices)):
        ch_idx = ch_idx.item()
        print(f"  {i+1}. Channel {ch_idx}: {-improvement.item():.6f} "
              f"(AWQ importance rank: {(awq_importance >= awq_importance[ch_idx]).sum().item()}, "
              f"PRAQ importance rank: {(praq_importance >= praq_importance[ch_idx]).sum().item()})")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
