"""
Analyze which channels PRAQ and AWQ select, and whether PRAQ's choices are better.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random


def load_wikitext2(split="train", n_samples=None, seed=42):
    """Load WikiText-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples and split == "validation":
        random.seed(seed)
        texts = random.sample(texts, min(n_samples, len(texts)))
    elif n_samples:
        texts = texts[:n_samples]
    return texts


class ChannelDecisionAnalyzer:
    """Analyze channel selection decisions."""

    def __init__(self, model, tokenizer, layer_name, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Find target module
        self.target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.target_module = module
                break

        if not isinstance(self.target_module, nn.Linear):
            raise ValueError(f"Not a Linear layer!")

        self.input_activations = []
        self.output_ground_truth = []
        self.hook = None

    def register_hook(self, store_output=False):
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
        if self.hook:
            self.hook.remove()

    @torch.no_grad()
    def collect_data(self, texts, n_samples, store_output=False):
        self.input_activations = []
        self.output_ground_truth = []
        self.model.eval()

        for text in tqdm(texts[:n_samples], desc="Collecting data", disable=True):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                self.model(**inputs, use_cache=False)
            except:
                continue

    def compute_awq_importance(self, X):
        W = self.target_module.weight.data.cpu().float()
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None
        Z = torch.matmul(X, W.t())
        if b is not None:
            Z = Z + b
        return Z.abs().mean(dim=0)

    def compute_praq_importance(self, X, beta=3.0, tau=-3.0, noise_factor=0.2):
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

    def compute_per_channel_mse(self, X, Y_gt, importance_scores, keep_ratio=0.5):
        """Compute MSE for each channel separately."""
        W_orig = self.target_module.weight.data.cpu().float()
        b = self.target_module.bias.data.cpu().float() if self.target_module.bias is not None else None

        out_features = W_orig.shape[0]
        k = int(out_features * keep_ratio)
        top_k_indices = torch.topk(importance_scores, k).indices

        # Compute per-channel MSE
        per_channel_mse = []

        for c in range(out_features):
            # Get ground truth for this channel
            y_gt_c = Y_gt[:, c]

            # Compute output for this channel (either FP16 or quantized)
            w_c = W_orig[c, :]

            if c in top_k_indices:
                # FP16 - no quantization
                y_pred_c = torch.matmul(X, w_c)
            else:
                # INT4 quantization
                w_range = w_c.abs().max()
                if w_range > 0:
                    scale = w_range / 7.0
                    w_quant = torch.round(w_c / scale).clamp(-8, 7)
                    w_dequant = w_quant * scale
                    noise = torch.randn_like(w_c) * scale * 0.1
                    w_dequant = w_dequant + noise
                    y_pred_c = torch.matmul(X, w_dequant)
                else:
                    y_pred_c = torch.matmul(X, w_c)

            if b is not None:
                y_pred_c = y_pred_c + b[c]

            mse_c = F.mse_loss(y_pred_c, y_gt_c).item()
            per_channel_mse.append(mse_c)

        return torch.tensor(per_channel_mse)

    def analyze(self, calib_texts, val_texts, n_calib=500, n_val=2000, keep_ratio=0.5):
        """Full analysis."""
        print("Collecting calibration data...")
        self.register_hook(store_output=False)
        self.collect_data(calib_texts, n_calib, store_output=False)
        self.remove_hook()
        X_calib = torch.cat(self.input_activations, dim=0)

        print("Collecting validation data...")
        self.register_hook(store_output=True)
        self.collect_data(val_texts, n_val, store_output=True)
        self.remove_hook()
        X_val = torch.cat(self.input_activations, dim=0)
        Y_gt = torch.cat(self.output_ground_truth, dim=0)

        print("\nComputing importance scores...")
        awq_importance = self.compute_awq_importance(X_calib)
        praq_importance = self.compute_praq_importance(X_calib)

        out_features = awq_importance.shape[0]
        k = int(out_features * keep_ratio)

        awq_selected = set(torch.topk(awq_importance, k).indices.tolist())
        praq_selected = set(torch.topk(praq_importance, k).indices.tolist())

        overlap = awq_selected.intersection(praq_selected)
        awq_only = awq_selected - praq_selected
        praq_only = praq_selected - awq_selected

        print("\n" + "=" * 80)
        print("CHANNEL SELECTION ANALYSIS")
        print("=" * 80)
        print(f"Total channels: {out_features}")
        print(f"Keep ratio: {keep_ratio}")
        print(f"Channels kept in FP16: {k} ({100*k/out_features:.0f}%)")
        print(f"Channels quantized to INT4: {out_features - k} ({100*(out_features-k)/out_features:.0f}%)")

        print(f"\nSelection overlap:")
        print(f"  Both methods agree: {len(overlap)} channels ({100*len(overlap)/k:.1f}%)")
        print(f"  AWQ-only selections: {len(awq_only)} channels")
        print(f"  PRAQ-only selections: {len(praq_only)} channels")

        print(f"\nChannel ID distribution:")
        print(f"  AWQ selections - min ID: {min(awq_selected)}, max ID: {max(awq_selected)}")
        print(f"  PRAQ selections - min ID: {min(praq_selected)}, max ID: {max(praq_selected)}")

        # Compute per-channel MSE
        print("\nComputing per-channel MSE (this may take a while)...")
        awq_per_channel_mse = self.compute_per_channel_mse(X_val, Y_gt, awq_importance, keep_ratio)
        praq_per_channel_mse = self.compute_per_channel_mse(X_val, Y_gt, praq_importance, keep_ratio)

        # Analyze disagreement channels
        print("\n" + "=" * 80)
        print("DISAGREEMENT ANALYSIS")
        print("=" * 80)

        awq_only_list = sorted(list(awq_only))
        praq_only_list = sorted(list(praq_only))

        awq_only_mse_awq = awq_per_channel_mse[awq_only_list].mean().item()
        awq_only_mse_praq = praq_per_channel_mse[awq_only_list].mean().item()

        praq_only_mse_awq = awq_per_channel_mse[praq_only_list].mean().item()
        praq_only_mse_praq = praq_per_channel_mse[praq_only_list].mean().item()

        print(f"\nChannels AWQ keeps but PRAQ quantizes ({len(awq_only)} channels):")
        print(f"  MSE with AWQ decision (FP16): {awq_only_mse_awq:.6f}")
        print(f"  MSE with PRAQ decision (INT4): {awq_only_mse_praq:.6f}")
        print(f"  Difference: {awq_only_mse_praq - awq_only_mse_awq:+.6f}")
        if awq_only_mse_praq < awq_only_mse_awq:
            print(f"  → PRAQ is RIGHT to quantize these ({100*(awq_only_mse_awq-awq_only_mse_praq)/awq_only_mse_awq:.1f}% better)")
        else:
            print(f"  → AWQ is RIGHT to keep these ({100*(awq_only_mse_praq-awq_only_mse_awq)/awq_only_mse_awq:.1f}% worse)")

        print(f"\nChannels PRAQ keeps but AWQ quantizes ({len(praq_only)} channels):")
        print(f"  MSE with AWQ decision (INT4): {praq_only_mse_awq:.6f}")
        print(f"  MSE with PRAQ decision (FP16): {praq_only_mse_praq:.6f}")
        print(f"  Difference: {praq_only_mse_praq - praq_only_mse_awq:+.6f}")
        if praq_only_mse_praq < praq_only_mse_awq:
            print(f"  → PRAQ is RIGHT to keep these ({100*(praq_only_mse_awq-praq_only_mse_praq)/praq_only_mse_awq:.1f}% better)")
        else:
            print(f"  → AWQ is RIGHT to quantize these ({100*(praq_only_mse_praq-praq_only_mse_awq)/praq_only_mse_awq:.1f}% worse)")

        print("\n" + "=" * 80)


def main():
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    layer_id = 28
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Find layer 28
    layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and f".{layer_id}." in name and "gate" in name.lower():
            layer_name = name
            break

    if not layer_name:
        print(f"ERROR: Could not find layer {layer_id}")
        return

    print(f"Analyzing layer: {layer_name}\n")

    calib_texts = load_wikitext2("train", n_samples=500)
    val_texts = load_wikitext2("validation", n_samples=2000)

    analyzer = ChannelDecisionAnalyzer(model, tokenizer, layer_name, device)
    analyzer.analyze(calib_texts, val_texts)


if __name__ == "__main__":
    main()
