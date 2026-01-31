"""
AdaRound Quantization - XL Version with Learned Rounding

This version implements AdaRound (Adaptive Rounding) quantization for LLM compression.

Key Features:
- Learned rounding using gradient descent instead of heuristic rules
- Rectified sigmoid for soft rounding decisions
- Regularization to encourage binary (0/1) rounding
- Beta annealing from 2 → 20 over iterations (gentle → hard)
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
- Regularization: Σ(1 - |2h(V) - 1|^β) where β increases from 2 to 20
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
    def __init__(self, layer, scale_g, w_floor_int, zp_g, n_groups, group_size, iterations=10000, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.layer = layer
        self.n_groups = n_groups
        self.group_size = group_size
        out_features, in_features = layer.weight.shape

        # Register COMPACT scale, floor, and zero-point as buffers
        # CRITICAL: scale_g as fp16 for 2× memory savings (safe for multiplicative factor)
        self.register_buffer('scale_g', scale_g.half())  # [out, n_groups] fp16
        self.register_buffer('w_floor_int', w_floor_int)  # [out, in] int16
        self.register_buffer('zp_g', zp_g)  # [out, n_groups] uint8

        # V is the learnable rounding parameter (same shape as weights)
        # CRITICAL: Initialize to fractional part WITH zero-point for asymmetric quantization
        # Directly fill v_init chunk-by-chunk to avoid allocating extra W_frac tensor
        W = layer.weight.data.float()
        v_init = torch.empty_like(W, dtype=torch.float32)

        # Compute v_init group-by-group using compact params
        for g in range(n_groups):
            j0 = g * group_size
            j1 = min((g + 1) * group_size, in_features)
            scale_g_cur = scale_g[:, g:g+1].float()  # [out, 1]
            zp_g_cur = zp_g[:, g:g+1].float()  # [out, 1]

            W_chunk = W[:, j0:j1]
            W_div = W_chunk / scale_g_cur
            W_shift = W_div + zp_g_cur  # Add zero-point (asymmetric!)
            W_frac_chunk = W_shift - torch.floor(W_shift)  # Fractional part [0, 1)

            # Clip away from exact 0/1 to avoid saturation (keep in [0.01, 0.99])
            h_init_chunk = torch.clamp(W_frac_chunk, 0.01, 0.99)

            # Invert rectified sigmoid to get V
            # h = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
            # Solve for V: sigmoid(V) = (h - gamma) / (zeta - gamma)
            sigmoid_target = (h_init_chunk - gamma) / (zeta - gamma)
            sigmoid_target = torch.clamp(sigmoid_target, 0.01, 0.99)  # Avoid log(0)

            # Inverse sigmoid: V = log(s / (1-s))
            v_init[:, j0:j1] = torch.log(sigmoid_target / (1.0 - sigmoid_target))

        # CRITICAL: Keep V in fp32 for stable gradients (even if model is bf16)
        self.v = nn.Parameter(v_init.float(), requires_grad=True)

        # Hyperparameters (Qualcomm AIMET defaults)
        self.iterations = iterations
        self.zeta = zeta      # Rectification high (default: 1.1)
        self.gamma = gamma    # Rectification low (default: -0.1)

    def get_soft_rounding(self):
        """Rectified Sigmoid: Maps V to [0, 1]"""
        return torch.clamp(torch.sigmoid(self.v) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def forward(self, x):
        """
        Apply the learned rounding to the weights.
        Uses block-wise reconstruction to avoid materializing full w_q AND full h_v.
        """
        # Accumulate output group-by-group (block-wise matmul)
        batch_dims = x.shape[:-1]  # Support arbitrary batch dims
        in_features = x.shape[-1]
        out_features = self.layer.weight.shape[0]

        # Initialize output
        out = torch.zeros(*batch_dims, out_features, device=x.device, dtype=torch.float32)

        # Process each group (compute h_v chunk-wise, never materialize full h_v)
        for g in range(self.n_groups):
            j0 = g * self.group_size
            j1 = min((g + 1) * self.group_size, in_features)

            # CRITICAL: Compute h_v for this chunk only (not full [out, in])
            v_chunk = self.v[:, j0:j1]  # [out, chunk_size]
            h_v_chunk = torch.clamp(
                torch.sigmoid(v_chunk) * (self.zeta - self.gamma) + self.gamma,
                0, 1
            ).float()  # [out, chunk_size]

            # Reconstruct weights for this group only
            scale_g_cur = self.scale_g[:, g:g+1].float()  # [out, 1]
            w_floor_chunk = self.w_floor_int[:, j0:j1].float()  # [out, chunk_size]

            # W_q_chunk = (w_floor + h_v) * scale
            w_q_chunk = (w_floor_chunk + h_v_chunk) * scale_g_cur  # [out, chunk_size]

            # Accumulate: out += x[..., j0:j1] @ w_q_chunk.T
            x_chunk = x[..., j0:j1].float()  # [..., chunk_size]
            out += F.linear(x_chunk, w_q_chunk, bias=None)

        # Add bias once at the end
        if self.layer.bias is not None:
            out += self.layer.bias.float()

        return out.to(x.dtype)


def compute_adaround_reg(v_parameter, iter_count, max_iter, zeta=1.1, gamma=-0.1, beta_start=2, beta_end=20):
    """
    Regularization term to force rounding to 0 or 1.

    Args:
        v_parameter: The learnable V tensor
        iter_count: Current iteration
        max_iter: Maximum iterations
        zeta, gamma: Rectified sigmoid parameters
        beta_start, beta_end: Annealing range for beta (2 → 20 for gentle → hard)

    Returns:
        Regularization loss (scalar)
    """
    # Anneal beta from beta_start UP to beta_end over iterations (2 → 20)
    beta = beta_start + (beta_end - beta_start) * (iter_count / max_iter)

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
        print(f"  AdaRound iterations: {adaround_iters} (max, early stopping enabled)")
        print(f"  Learning rate: {adaround_lr}")
        print(f"  Regularization weight: {reg_weight}")
        print(f"  Calibration: 256 samples max, mini-batch size 64")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample (stored as fp16)")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Optimization: V in fp32, weights computed in fp32")
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

            # CRITICAL: Store as fp16 (not fp32) to reduce memory by 50%
            # For 128 samples × 512 tokens × 4096 dim: fp16 = 512MB vs fp32 = 1GB
            self.activation_data[name].append(inp.detach().cpu().half())
        return hook

    @torch.no_grad()
    def get_calibration_data(self, name):
        """
        Get concatenated calibration activations for a layer.

        Returns:
            Tensor of shape [total_tokens, in_features] in fp32
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        # Concatenate all activation batches (stored as fp16, convert to fp32 for stability)
        X_all = torch.cat([x.reshape(-1, x.shape[-1]).float() for x in X_list], dim=0)
        return X_all

    @torch.no_grad()
    def compute_quantization_params_groupwise(self, W):
        """
        Compute asymmetric quantization parameters (scale, zero-point) for group-wise quantization.

        Args:
            W: Weight tensor [out_features, in_features]

        Returns:
            tuple: (scale_g, zp_g, w_floor_int, n_groups, group_size_actual)
                scale_g: [out_features, n_groups] fp32
                zp_g: [out_features, n_groups] uint8
                w_floor_int: [out_features, in_features] int8
                n_groups: int
                group_size_actual: int (the actual group size used)
        """
        out_features, in_features = W.shape
        device = W.device

        # CRITICAL: Compute quantization params in fp32 for numerical stability
        W = W.float()

        # Padding to multiple of group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=torch.float32)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        # Reshape to groups [out_features, n_groups, group_size]
        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        # Asymmetric Quantization: [0, 2^bits - 1]
        w_min = W_g.min(dim=2, keepdim=False)[0]  # [out, n_groups]
        w_max = W_g.max(dim=2, keepdim=False)[0]  # [out, n_groups]
        max_int = 2**self.bits - 1

        # Scale and zero-point (COMPACT: keep as [out, n_groups])
        scale_g = (w_max - w_min) / max_int  # [out, n_groups] fp32
        scale_g = scale_g.clamp(min=1e-8)
        zp_g = torch.round(-w_min / scale_g).clamp(0, max_int)  # [out, n_groups]

        # Compute w_floor using compact params (reconstruct on-the-fly per group)
        w_floor_list = []
        for g in range(n_groups):
            j0 = g * self.group_size
            j1 = min((g + 1) * self.group_size, padded_in_features)
            scale_g_cur = scale_g[:, g:g+1]  # [out, 1]
            zp_g_cur = zp_g[:, g:g+1]  # [out, 1]

            W_chunk = W_padded[:, j0:j1]
            W_div = W_chunk / scale_g_cur
            W_floor_chunk = torch.floor(W_div + zp_g_cur).clamp(0, max_int) - zp_g_cur
            w_floor_list.append(W_floor_chunk)

        W_floor = torch.cat(w_floor_list, dim=1)  # [out, padded_in]

        # Trim padding if needed
        if padded_in_features > in_features:
            W_floor = W_floor[:, :in_features]

        # Convert to integers for memory efficiency
        # CRITICAL: Use int16 for w_floor (not int8) to avoid overflow
        # Range can be [-zp, max_int - zp] which is [-15, 15] for 4-bit
        zp_g_int = zp_g.to(torch.uint8)  # [out, n_groups] uint8
        w_floor_int = W_floor.to(torch.int16)  # [out, in] int16 (safe for all cases)

        return scale_g, zp_g_int, w_floor_int, n_groups, self.group_size

    def optimize_layer_adaround(self, name, module, calibration_data_cpu, num_iterations=10000, debug=False):
        """
        Optimize rounding for a single layer using AdaRound with mini-batch processing.

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

        # CRITICAL: Use the layer's actual device (for multi-GPU support with device_map="auto")
        layer_device = W.device

        # Compute quantization parameters (now returns compact group-wise params)
        scale_g, zp_g, w_floor_int, n_groups, group_size = self.compute_quantization_params_groupwise(W)

        # CRITICAL: Check for invalid scale values
        if (scale_g == 0).any() or torch.isnan(scale_g).any() or torch.isinf(scale_g).any():
            print(f"    🚨 ERROR: Invalid scale values in {name}!")
            print(f"       Zero scales: {(scale_g == 0).sum().item()}")
            print(f"       NaN scales: {torch.isnan(scale_g).sum().item()}")
            print(f"       Inf scales: {torch.isinf(scale_g).sum().item()}")
            # Return original weights without quantization
            return W.clone(), float('inf')

        # Move to layer's device (not self.device!)
        scale_g = scale_g.to(layer_device)
        zp_g = zp_g.to(layer_device)
        w_floor_int = w_floor_int.to(layer_device)

        # Create AdaRound wrapper on the same device as the layer (pass compact params)
        wrapper = AdaRoundOptimizer(module, scale_g, w_floor_int, zp_g, n_groups, group_size,
                                   iterations=num_iterations).to(layer_device)
        optimizer = torch.optim.Adam([wrapper.v], lr=self.adaround_lr)

        # Subsample calibration data for speed (REDUCED from 1024 to 256 for memory)
        max_samples = min(256, calibration_data_cpu.shape[0])
        if calibration_data_cpu.shape[0] > max_samples:
            indices = torch.randperm(calibration_data_cpu.shape[0])[:max_samples]
            calib_data_cpu_subset = calibration_data_cpu[indices]
        else:
            calib_data_cpu_subset = calibration_data_cpu

        # Mini-batch size for processing (MEMORY OPTIMIZATION: process in smaller chunks)
        mini_batch_size = 64
        num_mini_batches = (max_samples + mini_batch_size - 1) // mini_batch_size

        # Optimization loop with early stopping
        best_loss = float('inf')
        best_v = None
        patience_counter = 0
        patience_limit = 200  # Stop if no improvement for 200 iterations (reduced from 500)

        for i in range(num_iterations):
            optimizer.zero_grad()

            total_rec_loss = 0.0

            # Process calibration data in mini-batches to save memory
            for mb_idx in range(num_mini_batches):
                mb_start = mb_idx * mini_batch_size
                mb_end = min(mb_start + mini_batch_size, max_samples)

                # Load only current mini-batch to layer's device
                calib_mb = calib_data_cpu_subset[mb_start:mb_end].to(layer_device).to(original_dtype)

                # Ground truth for this mini-batch
                with torch.no_grad():
                    target_mb = F.linear(calib_mb, W, module.bias)

                # Quantized forward pass
                current_mb = wrapper(calib_mb)

                # Accumulate reconstruction loss
                rec_loss_mb = F.mse_loss(current_mb, target_mb)
                total_rec_loss += rec_loss_mb * (mb_end - mb_start) / max_samples

                # Free mini-batch memory immediately
                del calib_mb, target_mb, current_mb

            # Regularization (computed once, not per mini-batch)
            reg_loss = self.reg_weight * compute_adaround_reg(wrapper.v, i, num_iterations)

            total_loss = total_rec_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            # Track best and early stopping
            current_loss = total_loss.item()

            # Special handling for first iteration
            if best_v is None:
                best_loss = current_loss
                best_v = wrapper.v.data.clone()
                patience_counter = 0
            else:
                # Use relative tolerance for better robustness with fp16/bf16
                improvement_threshold = max(1e-5, best_loss * 1e-4)  # 0.01% relative improvement
                if current_loss < best_loss - improvement_threshold:
                    best_loss = current_loss
                    best_v = wrapper.v.data.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Early stopping
            if patience_counter >= patience_limit:
                print(f"      ⏹️  Early stopping at iter {i}/{num_iterations} (no improvement for {patience_limit} iters)")
                break

            # Progress updates (more frequent for visibility)
            if debug and i % 200 == 0:
                print(f"      Iter {i}/{num_iterations}: Total={total_loss.item():.6f}, "
                      f"Rec={total_rec_loss.item():.6f}, Reg={reg_loss.item():.6f}, "
                      f"Patience={patience_counter}/{patience_limit}")

            # Free gradients and intermediate tensors
            del total_rec_loss, reg_loss, total_loss

            # Periodic cache clearing
            if i % 100 == 0:
                torch.cuda.empty_cache()

        # Final hard rounding using best V (reconstruct group-by-group)
        # CRITICAL: Compute h_v chunk-wise to avoid materializing full [out, in] tensor
        with torch.no_grad():
            # Safety check: if best_v is None (should never happen now), use current V
            if best_v is None:
                print(f"      ⚠️  WARNING: best_v is None! Using current V")
                best_v = wrapper.v.data.clone()

            wrapper.v.data = best_v

            # Reconstruct weights group-by-group using compact params
            _, in_features = W.shape
            optimized_weights = torch.zeros_like(W, dtype=torch.float32)

            for g in range(n_groups):
                j0 = g * group_size
                j1 = min((g + 1) * group_size, in_features)

                # Compute h_v for this chunk only (not full tensor!)
                v_chunk = wrapper.v[:, j0:j1]
                h_v_chunk = torch.clamp(
                    torch.sigmoid(v_chunk) * (wrapper.zeta - wrapper.gamma) + wrapper.gamma,
                    0, 1
                ).float()

                # Round to 0 or 1 based on optimized soft values
                hard_rounding_chunk = (h_v_chunk > 0.5).float()

                # Reconstruct this chunk
                scale_g_cur = scale_g[:, g:g+1].float()  # [out, 1]
                w_floor_chunk = w_floor_int[:, j0:j1].float()  # [out, chunk_size]

                # W_q = (w_floor + hard_rounding) * scale
                optimized_weights[:, j0:j1] = (w_floor_chunk + hard_rounding_chunk) * scale_g_cur

            final_weights = optimized_weights.to(original_dtype)

        # Cleanup ALL temporary tensors (keep final_weights for return)
        del wrapper, optimizer, scale_g, zp_g, w_floor_int, calib_data_cpu_subset
        del best_v, optimized_weights
        torch.cuda.empty_cache()
        gc.collect()

        return final_weights, best_loss

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

        # CRITICAL: Check for NaN/Inf before updating weights
        if torch.isnan(W_optimized).any() or torch.isinf(W_optimized).any():
            print(f"    🚨 ERROR: {name} has NaN/Inf values! Skipping quantization.")
            print(f"       NaN count: {torch.isnan(W_optimized).sum().item()}")
            print(f"       Inf count: {torch.isinf(W_optimized).sum().item()}")
            del W_optimized, calib_data_cpu
            return

        # Update weights
        module.weight.data = W_optimized

        # Diagnostic: Print weight statistics for first few layers
        if debug:
            print(f"       Weight stats: min={W_optimized.min().item():.6f}, "
                  f"max={W_optimized.max().item():.6f}, "
                  f"mean={W_optimized.mean().item():.6f}, "
                  f"std={W_optimized.std().item():.6f}")

        # Store statistics
        self.layer_stats[name] = {
            'final_loss': final_loss,
            'shape': list(W_optimized.shape),
            'weight_min': W_optimized.min().item(),
            'weight_max': W_optimized.max().item()
        }

        # Aggressive cleanup
        del W_optimized, calib_data_cpu
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
        layer_device = W.device  # CRITICAL: Use layer's actual device
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

            # Create a temporary module for this chunk (CRITICAL: match original dtype)
            chunk_module = nn.Linear(in_features, end_idx - start_idx, bias=module.bias is not None)
            chunk_module.weight.data = W[start_idx:end_idx, :].clone()
            if module.bias is not None:
                chunk_module.bias.data = module.bias.data[start_idx:end_idx].clone()
            # Move to layer's device and convert to original dtype
            chunk_module = chunk_module.to(layer_device, dtype=original_dtype)

            # Optimize this chunk
            W_chunk_optimized, chunk_loss = self.optimize_layer_adaround(
                f"{name}_chunk{chunk_idx}",
                chunk_module,
                calib_data_cpu,
                num_iterations=self.adaround_iters,
                debug=(debug and chunk_idx == 0)
            )

            # Check for NaN/Inf in chunk
            if torch.isnan(W_chunk_optimized).any() or torch.isinf(W_chunk_optimized).any():
                print(f"    🚨 ERROR: Chunk {chunk_idx} has NaN/Inf! Using original weights.")
                W_final_chunks.append(W[start_idx:end_idx, :].clone().cpu())
            else:
                W_final_chunks.append(W_chunk_optimized.cpu())
            chunk_stats.append({'loss': chunk_loss})

            # Cleanup
            del chunk_module, W_chunk_optimized
            torch.cuda.empty_cache()

        # Combine all chunks and move to layer's device
        W_final = torch.cat(W_final_chunks, dim=0).to(layer_device)
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

        # Cleanup (W_final is now owned by module.weight.data, safe to delete reference)
        del W_final_chunks, chunk_stats, calib_data_cpu
        torch.cuda.empty_cache()
        gc.collect()

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
    parser.add_argument("--adaround-iters", type=int, default=2000,
                       help="Number of AdaRound optimization iterations per layer (default: 2000, early stopping at 200)")
    parser.add_argument("--adaround-lr", type=float, default=1e-3,
                       help="Learning rate for AdaRound optimization (default: 1e-3)")
    parser.add_argument("--reg-weight", type=float, default=0.001,
                       help="Weight for regularization term (default: 0.001)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=256,
                       help="Max tokens to store per sample (default: 256, CRITICAL for memory)")
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
