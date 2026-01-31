"""
AdaRound Quantization - XL Version with Learned Rounding

This version implements AdaRound (Adaptive Rounding) quantization for LLM compression.

Key Features:
- Learned rounding using gradient descent instead of heuristic rules
- Rectified sigmoid for soft rounding decisions
- Regularization to encourage binary (0/1) rounding
- Beta annealing from 20 → 2 over iterations
- Special lm_head chunking to avoid OOM
- Batched sequential quantization for memory efficiency

AdaRound Algorithm:
1. For each layer, initialize learnable rounding parameters V
2. Optimize V to minimize reconstruction error (MSE)
3. Use rectified sigmoid h(V) to get soft rounding in [0, 1]
4. Regularization forces h(V) → 0 or 1 over time
5. Final hard rounding: round(h(V))

Formula:
- W_quant = (W_floor + h(V)) × delta
- h(V) = clamp(sigmoid(V) × (ζ - γ) + γ, 0, 1)
- Regularization: Σ(1 - |2h(V) - 1|^β) where β decreases from 20 to 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  Warning: psutil not installed. Memory monitoring disabled.")

# Try to import calibration utils, fallback if not present
try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️ calibration_utils not found. Using internal fallback loaders.")
    def get_c4_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")


class AdaRoundOptimizer(nn.Module):
    """
    AdaRound wrapper for a single linear layer.

    Learns optimal rounding decisions via gradient descent.
    """
    def __init__(self, layer, weight_delta, weight_floor, iterations=10000, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.layer = layer

        # Register scale and floor as buffers (not parameters, but need proper device handling)
        self.register_buffer('delta', weight_delta)
        self.register_buffer('w_floor', weight_floor)

        # V is the learnable rounding parameter (same shape as weights)
        # Initialized such that sigmoid(V) ≈ 0.5
        # Create new tensor explicitly to avoid copying metadata from layer.weight
        v_init = torch.zeros(
            layer.weight.shape,
            dtype=layer.weight.dtype,
            device=layer.weight.device
        )
        self.v = nn.Parameter(v_init, requires_grad=True)

        # Hyperparameters (Qualcomm AIMET defaults)
        self.iterations = iterations
        self.zeta = zeta      # Rectification high (default: 1.1)
        self.gamma = gamma    # Rectification low (default: -0.1)

    def get_soft_rounding(self):
        """Rectified Sigmoid: Maps V to [0, 1]"""
        return torch.clamp(torch.sigmoid(self.v) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def forward(self, x):
        """Apply the learned rounding to the weights"""
        # W_q = (W_floor + h(V)) × delta
        h_v = self.get_soft_rounding()
        w_q = (self.w_floor + h_v) * self.delta

        return F.linear(x, w_q, self.layer.bias)


def compute_adaround_reg(v_parameter, iter_count, max_iter, zeta=1.1, gamma=-0.1, beta_start=20, beta_end=2):
    """
    Regularization term to force rounding to 0 or 1.

    Args:
        v_parameter: The learnable V tensor
        iter_count: Current iteration
        max_iter: Maximum iterations
        zeta, gamma: Rectified sigmoid parameters
        beta_start, beta_end: Annealing range for beta

    Returns:
        Regularization loss (scalar)
    """
    # Anneal beta from beta_start down to beta_end over iterations
    beta = beta_end + (beta_start - beta_end) * (1 - iter_count / max_iter)

    # h_v is the soft decision
    h_v = torch.clamp(torch.sigmoid(v_parameter) * (zeta - gamma) + gamma, 0, 1)

    # Penalty: 1 - |2*h_v - 1|^beta
    # This is 0 when h_v is 0 or 1, and max when h_v is 0.5
    reg = (1 - (2 * h_v - 1).abs().pow(beta)).mean()  # Use mean instead of sum
    return reg


class AdaRoundQuantizerXL:
    """AdaRound Quantizer with XL model support (chunked lm_head)."""

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 adaround_iters=10000, adaround_lr=1e-3, reg_weight=0.01,
                 max_tokens_per_sample=512, layer_batch_size=16, lmhead_chunks=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size
        self.adaround_iters = adaround_iters
        self.adaround_lr = adaround_lr
        self.reg_weight = reg_weight
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.lmhead_chunks = lmhead_chunks

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[AdaRound Quantizer XL Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  AdaRound iterations: {adaround_iters}")
        print(f"  Learning rate: {adaround_lr}")
        print(f"  Regularization weight: {reg_weight}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Special: lm_head split into {lmhead_chunks} chunks to avoid OOM")

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if sequence is too long (memory optimization)
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]  # Keep temporal order
                inp = inp[:, indices, :]

            # Store on CPU to save GPU memory, use float32 for numerical stability
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_calibration_data(self, name):
        """
        Get concatenated calibration activations for a layer.

        Returns:
            Tensor of shape [total_tokens, in_features]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        # Concatenate all activation batches
        X_all = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)
        return X_all

    @torch.no_grad()
    def compute_quantization_params_groupwise(self, W):
        """
        Compute asymmetric quantization parameters (scale, zero-point) for group-wise quantization.

        Args:
            W: Weight tensor [out_features, in_features]

        Returns:
            tuple: (scale, zero_point, w_floor) all with shape [out_features, in_features]
        """
        out_features, in_features = W.shape
        device = W.device

        # Padding to multiple of group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        # Reshape to groups [out_features, n_groups, group_size]
        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        # Asymmetric Quantization: [0, 2^bits - 1]
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2**self.bits - 1

        # Scale and zero-point
        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        # Expand to full size [out_features, padded_in_features]
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)


        # Compute floor(W / scale)
        W_div = W_padded / scale_flat
        W_floor = torch.floor(W_div + zp_flat).clamp(0, max_int) - zp_flat

        # Trim padding if needed
        if padded_in_features > in_features:
            scale_flat = scale_flat[:, :in_features]
            zp_flat = zp_flat[:, :in_features]
            W_floor = W_floor[:, :in_features]

        return scale_flat, zp_flat, W_floor

    def optimize_layer_adaround(self, name, module, calibration_data_cpu, num_iterations=10000, debug=False):
        """
        Optimize rounding for a single layer using AdaRound.

        Args:
            name: Layer name
            module: The linear module
            calibration_data_cpu: Calibration inputs on CPU [n_samples, in_features]
            num_iterations: Number of optimization iterations
            debug: Print debug information

        Returns:
            Optimized weight tensor
        """
        W = module.weight.data
        original_dtype = W.dtype

        # Compute quantization parameters
        scale, zp, w_floor = self.compute_quantization_params_groupwise(W)

        # Move to device
        w_floor = w_floor.to(self.device)
        scale = scale.to(self.device)

        # Create AdaRound wrapper
        wrapper = AdaRoundOptimizer(module, scale, w_floor, iterations=num_iterations).to(self.device)
        optimizer = torch.optim.Adam([wrapper.v], lr=self.adaround_lr)

        # Subsample calibration data for speed (use fewer samples for AdaRound)
        max_samples = min(1024, calibration_data_cpu.shape[0])
        if calibration_data_cpu.shape[0] > max_samples:
            indices = torch.randperm(calibration_data_cpu.shape[0])[:max_samples]
            calib_data = calibration_data_cpu[indices].to(self.device).to(original_dtype)
        else:
            calib_data = calibration_data_cpu.to(self.device).to(original_dtype)

        # Get ground truth (FP32 output)
        with torch.no_grad():
            target_out = F.linear(calib_data, W, module.bias)

        # Optimization loop
        best_loss = float('inf')
        best_v = None

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Quantized forward pass
            current_out = wrapper(calib_data)

            # MSE Loss + Regularization
            rec_loss = F.mse_loss(current_out, target_out)
            reg_loss = self.reg_weight * compute_adaround_reg(wrapper.v, i, num_iterations)

            total_loss = rec_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_v = wrapper.v.data.clone()

            if debug and i % 1000 == 0:
                print(f"      Iter {i}/{num_iterations}: Total={total_loss.item():.6f}, "
                      f"Rec={rec_loss.item():.6f}, Reg={reg_loss.item():.6f}")

        # Final hard rounding using best V
        with torch.no_grad():
            wrapper.v.data = best_v
            h_v_final = wrapper.get_soft_rounding()
            # Round to 0 or 1 based on optimized soft values
            hard_rounding = (h_v_final > 0.5).float()
            optimized_weights = (w_floor + hard_rounding) * scale

        # Cleanup
        del wrapper, optimizer, calib_data, target_out, current_out, scale, zp, w_floor
        torch.cuda.empty_cache()

        return optimized_weights.to(original_dtype), best_loss

    def quantize_layer(self, name, module, debug=False):
        """Apply AdaRound quantization to a single layer."""
        # Get calibration data
        calib_data_cpu = self.get_calibration_data(name)

        if calib_data_cpu is None or calib_data_cpu.shape[0] == 0:
            print(f"    ⚠️  No calibration data for {name}, skipping")
            return

        # Run AdaRound optimization
        W_optimized, final_loss = self.optimize_layer_adaround(
            name, module, calib_data_cpu,
            num_iterations=self.adaround_iters,
            debug=debug
        )

        # Update weights
        module.weight.data = W_optimized

        # Store statistics
        self.layer_stats[name] = {
            'final_loss': final_loss,
            'shape': list(W_optimized.shape)
        }

        # Cleanup
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def quantize_lmhead_chunked(self, name, module, num_chunks=4, debug=False):
        """
        Quantize lm_head by splitting it into chunks along output dimension.
        This reduces peak memory usage significantly.

        Args:
            num_chunks: Number of chunks to split into
        """
        print(f"\n  🔧 Special handling for {name} (split into {num_chunks} chunks)")

        W = module.weight.data
        original_dtype = W.dtype
        out_features, in_features = W.shape

        print(f"     Shape: {W.shape} ({W.numel() / 1e6:.1f}M parameters)")

        # Get calibration data once
        calib_data_cpu = self.get_calibration_data(name)

        if calib_data_cpu is None or calib_data_cpu.shape[0] == 0:
            print(f"    ⚠️  No calibration data for {name}, skipping")
            return

        # Calculate chunk boundaries
        chunk_size = out_features // num_chunks
        chunk_boundaries = [(i * chunk_size,
                            out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
                           for i in range(num_chunks)]

        W_final_chunks = []
        chunk_stats = []

        # Process each chunk
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_boundaries):
            print(f"     Processing chunk {chunk_idx + 1}/{num_chunks}: rows {start_idx}-{end_idx}")

            # Create a temporary module for this chunk
            chunk_module = nn.Linear(in_features, end_idx - start_idx, bias=module.bias is not None)
            chunk_module.weight.data = W[start_idx:end_idx, :].clone()
            if module.bias is not None:
                chunk_module.bias.data = module.bias.data[start_idx:end_idx].clone()
            chunk_module = chunk_module.to(self.device)

            # Optimize this chunk
            W_chunk_optimized, chunk_loss = self.optimize_layer_adaround(
                f"{name}_chunk{chunk_idx}",
                chunk_module,
                calib_data_cpu,
                num_iterations=self.adaround_iters,
                debug=(debug and chunk_idx == 0)
            )

            W_final_chunks.append(W_chunk_optimized.cpu())
            chunk_stats.append({'loss': chunk_loss})

            # Cleanup
            del chunk_module, W_chunk_optimized
            torch.cuda.empty_cache()

        # Combine all chunks
        W_final = torch.cat(W_final_chunks, dim=0).to(self.device)
        module.weight.data = W_final

        # Store statistics
        avg_loss = np.mean([s['loss'] for s in chunk_stats])
        self.layer_stats[name] = {
            'final_loss': avg_loss,
            'shape': list(W_final.shape),
            'num_chunks': num_chunks
        }

        # Print summary
        loss_str = ', '.join([f'loss_{i+1}={s["loss"]:.6f}' for i, s in enumerate(chunk_stats)])
        print(f"     ✓ Done: {loss_str}")

        # Cleanup
        del W_final_chunks, chunk_stats, calib_data_cpu
        torch.cuda.empty_cache()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """Calibrate a batch of layers simultaneously."""
        print(f"  Calibrating {len(layer_names_batch)} layers...")

        self.model.eval()
        handles = []

        # Register hooks for all layers in this batch
        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        # Run calibration
        successful = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="  Calibration", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1

                    # Periodic cache clearing
                    if (successful + 1) % 32 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        # Remove hooks
        for _, handle in handles:
            handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """Batched sequential quantization with AdaRound."""
        print("\n" + "=" * 80)
        print("Batched Sequential AdaRound Quantization (XL Version)")
        print("=" * 80)
        print(f"  Strategy: Process {self.layer_batch_size} layers per batch")

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"  Initial System RAM: {initial_ram:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        num_layers = len(layer_names)
        num_batches = (num_layers + self.layer_batch_size - 1) // self.layer_batch_size

        print(f"  Total layers: {num_layers}")
        print(f"  Total batches: {num_batches}")
        print("=" * 80)

        quantized_count = 0

        # Process in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.layer_batch_size
            batch_end = min(batch_start + self.layer_batch_size, num_layers)
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Layers {batch_start}-{batch_end-1}")

            # Calibrate this batch
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            # Quantize all layers in this batch
            print(f"  Quantizing {len(batch_layers)} layers with AdaRound...")
            for name, module in tqdm(batch_layers, desc="  Quantization", leave=False):
                try:
                    # Check if this is lm_head (special handling)
                    is_lmhead = 'lm_head' in name.lower() or name.endswith('lm_head')

                    if is_lmhead:
                        # Use chunked processing for lm_head
                        debug = (quantized_count < 2)
                        self.quantize_lmhead_chunked(name, module, num_chunks=self.lmhead_chunks, debug=debug)
                    else:
                        # Standard AdaRound processing
                        debug = (quantized_count < 2)
                        self.quantize_layer(name, module, debug=debug)

                    quantized_count += 1

                except Exception as e:
                    print(f"\n⚠️  Error quantizing {name}: {e}")
                    continue

            # Clear activations for this batch
            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"  Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

        print("\n" + "=" * 80)
        print("✓ Batched Sequential AdaRound Quantization Complete")
        print(f"  Total layers quantized: {quantized_count}/{num_layers}")
        print("=" * 80)

        if self.layer_stats:
            losses = [info['final_loss'] for info in self.layer_stats.values()]
            print(f"\nFinal Loss Statistics:")
            print(f"  Mean: {np.mean(losses):.6f}")
            print(f"  Median: {np.median(losses):.6f}")
            print(f"  Min: {np.min(losses):.6f} | Max: {np.max(losses):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4], help="Quantization bit width (default: 4)")
    parser.add_argument("--adaround-iters", type=int, default=10000,
                       help="Number of AdaRound optimization iterations per layer (default: 10000)")
    parser.add_argument("--adaround-lr", type=float, default=1e-3,
                       help="Learning rate for AdaRound optimization (default: 1e-3)")
    parser.add_argument("--reg-weight", type=float, default=0.01,
                       help="Weight for regularization term (default: 0.01)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample (default: 2048)")
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Number of layers to process per batch (default: 16)")
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                       help="Number of chunks to split lm_head into (default: 4, higher = less memory)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_adaround_xl")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset (default: c4)")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                       help="Directory to cache calibration data (default: ./calibration_cache)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use model path from args
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("AdaRound Quantization (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"AdaRound iterations per layer: {args.adaround_iters}")
    print(f"Learning rate: {args.adaround_lr}")
    print(f"Regularization weight: {args.reg_weight}")
    print(f"Special: lm_head split into {args.lmhead_chunks} chunks")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Fix for Llama/Mistral models lacking pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    quantizer = AdaRoundQuantizerXL(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        group_size=args.group_size,
        adaround_iters=args.adaround_iters,
        adaround_lr=args.adaround_lr,
        reg_weight=args.reg_weight,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks
    )

    # Use batched sequential quantization (optimal memory/speed balance)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
