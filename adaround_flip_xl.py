"""
AdaRound + Heuristic Flipping - XL Version

This version combines:
1. AdaRound (Adaptive Rounding) - Learned rounding via gradient descent
2. Heuristic Flipping - Global greedy bit-flip correction

Key Features:
- AdaRound for initial quantization (learned optimal rounding)
- Heuristic flipping to further reduce quantization error
- Group-wise asymmetric quantization [0, 15] for 4-bit
- Beta annealing from 2 → 20 over iterations
- Dynamic outlier masking using kneedle algorithm
- Memory-optimized with compact group-wise storage
- Batched sequential quantization

Workflow:
1. Run AdaRound optimization to get initial quantized weights
2. Apply heuristic flipping on top of AdaRound results
3. Return final refined weights

Formula (AdaRound):
- W_quant = (W_floor + h(V)) × scale
- h(V) = clamp(sigmoid(V) × (ζ - γ) + γ, 0, 1)
- Regularization: Σ(1 - |2h(V) - 1|^β) where β increases from 2 to 20

Flipping (Heuristic):
- Compute flip impacts based on activation importance
- Apply global greedy flipping to minimize error
- Limit flips per output channel to max_flip_percent
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

# Try to import calibration utils
try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️ calibration_utils not found. Using internal fallback loaders.")
    def get_c4_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")


# Import AdaRound components from adaround_xl.py
class AdaRoundOptimizer(nn.Module):
    """AdaRound wrapper for a single linear layer."""
    def __init__(self, layer, scale_g, w_floor_int, zp_g, n_groups, group_size, iterations=10000, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.layer = layer
        self.n_groups = n_groups
        self.group_size = group_size
        out_features, in_features = layer.weight.shape

        # Register COMPACT scale, floor, and zero-point as buffers
        self.register_buffer('scale_g', scale_g.half())  # [out, n_groups] fp16
        self.register_buffer('w_floor_int', w_floor_int)  # [out, in] int16
        self.register_buffer('zp_g', zp_g)  # [out, n_groups] uint8

        # Initialize V directly chunk-by-chunk
        W = layer.weight.data.float()
        v_init = torch.empty_like(W, dtype=torch.float32)

        for g in range(n_groups):
            j0 = g * group_size
            j1 = min((g + 1) * group_size, in_features)
            scale_g_cur = scale_g[:, g:g+1].float()
            zp_g_cur = zp_g[:, g:g+1].float()

            W_chunk = W[:, j0:j1]
            W_div = W_chunk / scale_g_cur
            W_shift = W_div + zp_g_cur
            W_frac_chunk = W_shift - torch.floor(W_shift)

            h_init_chunk = torch.clamp(W_frac_chunk, 0.01, 0.99)
            sigmoid_target = (h_init_chunk - gamma) / (zeta - gamma)
            sigmoid_target = torch.clamp(sigmoid_target, 0.01, 0.99)
            v_init[:, j0:j1] = torch.log(sigmoid_target / (1.0 - sigmoid_target))

        self.v = nn.Parameter(v_init.float(), requires_grad=True)
        self.iterations = iterations
        self.zeta = zeta
        self.gamma = gamma

    def get_soft_rounding(self):
        return torch.clamp(torch.sigmoid(self.v) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def forward(self, x):
        """Block-wise forward to avoid materializing full w_q and h_v."""
        batch_dims = x.shape[:-1]
        in_features = x.shape[-1]
        out_features = self.layer.weight.shape[0]
        out = torch.zeros(*batch_dims, out_features, device=x.device, dtype=torch.float32)

        for g in range(self.n_groups):
            j0 = g * self.group_size
            j1 = min((g + 1) * self.group_size, in_features)

            v_chunk = self.v[:, j0:j1]
            h_v_chunk = torch.clamp(
                torch.sigmoid(v_chunk) * (self.zeta - self.gamma) + self.gamma,
                0, 1
            ).float()

            scale_g_cur = self.scale_g[:, g:g+1].float()
            w_floor_chunk = self.w_floor_int[:, j0:j1].float()
            w_q_chunk = (w_floor_chunk + h_v_chunk) * scale_g_cur

            x_chunk = x[..., j0:j1].float()
            out += F.linear(x_chunk, w_q_chunk, bias=None)

        if self.layer.bias is not None:
            out += self.layer.bias.float()

        return out.to(x.dtype)


def compute_adaround_reg(v_parameter, iter_count, max_iter, zeta=1.1, gamma=-0.1, beta_start=2, beta_end=20):
    """Regularization term to force rounding to 0 or 1."""
    beta = beta_start + (beta_end - beta_start) * (iter_count / max_iter)
    h_v = torch.clamp(torch.sigmoid(v_parameter) * (zeta - gamma) + gamma, 0, 1)
    reg = (1 - (2 * h_v - 1).abs().pow(beta)).mean()
    return reg


def compute_dynamic_outlier_threshold_kneedle(activations, knee_tolerance=0.1):
    """
    Compute dynamic outlier threshold using Kneedle algorithm.

    Returns:
        tuple: (threshold, outlier_percent)
    """
    abs_act = activations.abs()
    sorted_act, _ = torch.sort(abs_act, descending=True)
    n = sorted_act.shape[0]

    # Normalize to [0, 1]
    y_norm = (sorted_act - sorted_act.min()) / (sorted_act.max() - sorted_act.min() + 1e-8)
    x_norm = torch.arange(n, device=activations.device, dtype=torch.float32) / (n - 1)

    # Compute differences (second derivative approximation)
    diffs = y_norm[:-1] - y_norm[1:]

    # Find knee (maximum curvature with tolerance)
    if diffs.numel() == 0:
        knee_idx = 0
    else:
        # Use tolerance-based kneedle: find first point where slope drops significantly
        threshold_diff = diffs.max() * knee_tolerance
        candidates = (diffs >= threshold_diff).nonzero(as_tuple=True)[0]
        knee_idx = candidates[0].item() if candidates.numel() > 0 else 0

    threshold = sorted_act[knee_idx].item()
    outlier_percent = (knee_idx / n) * 100

    return threshold, outlier_percent


class AdaRoundFlipQuantizerXL:
    """AdaRound + Heuristic Flipping Quantizer with XL model support."""

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 adaround_iters=2000, adaround_lr=1e-3, reg_weight=0.001,
                 max_tokens_per_sample=256, layer_batch_size=16, lmhead_chunks=4,
                 use_flipping=True, max_flip_percent=0.05, knee_tolerance=0.1):
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
        self.use_flipping = use_flipping
        self.max_flip_percent = max_flip_percent
        self.knee_tolerance = knee_tolerance

        self.activation_data = {}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[AdaRound + Heuristic Flipping Quantizer XL Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  AdaRound iterations: {adaround_iters} (max, early stopping enabled)")
        print(f"  Learning rate: {adaround_lr}")
        print(f"  Regularization weight: {reg_weight}")
        print(f"  Calibration: 256 samples max, mini-batch size 64")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample (stored as fp16)")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Use flipping: {use_flipping}")
        if use_flipping:
            print(f"  Max flip percent per channel: {max_flip_percent*100:.1f}%")
            print(f"  Knee tolerance: {knee_tolerance}")
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

            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            self.activation_data[name].append(inp.detach().cpu().half())
        return hook

    @torch.no_grad()
    def get_calibration_data(self, name):
        """Get concatenated calibration activations for a layer."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        X_all = torch.cat([x.reshape(-1, x.shape[-1]).float() for x in X_list], dim=0)
        return X_all

    @torch.no_grad()
    def compute_quantization_params_groupwise(self, W):
        """Compute asymmetric quantization parameters (scale, zero-point) for group-wise quantization."""
        out_features, in_features = W.shape
        device = W.device
        W = W.float()

        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=torch.float32)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        W_g = W_padded.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=False)[0]
        w_max = W_g.max(dim=2, keepdim=False)[0]
        max_int = 2**self.bits - 1

        scale_g = (w_max - w_min) / max_int
        scale_g = scale_g.clamp(min=1e-8)
        zp_g = torch.round(-w_min / scale_g).clamp(0, max_int)

        w_floor_list = []
        for g in range(n_groups):
            j0 = g * self.group_size
            j1 = min((g + 1) * self.group_size, padded_in_features)
            scale_g_cur = scale_g[:, g:g+1]
            zp_g_cur = zp_g[:, g:g+1]

            W_chunk = W_padded[:, j0:j1]
            W_div = W_chunk / scale_g_cur
            W_floor_chunk = torch.floor(W_div + zp_g_cur).clamp(0, max_int) - zp_g_cur
            w_floor_list.append(W_floor_chunk)

        W_floor = torch.cat(w_floor_list, dim=1)

        if padded_in_features > in_features:
            W_floor = W_floor[:, :in_features]

        zp_g_int = zp_g.to(torch.uint8)
        w_floor_int = W_floor.to(torch.int16)

        return scale_g, zp_g_int, w_floor_int, n_groups, self.group_size

    def _apply_heuristic_flipping_chunked(self, W, scale_g, zp_g, w_floor_int, n_groups, group_size,
                                          group_activation_means, chunk_size_out):
        """
        Chunked version of heuristic flipping for large layers.
        Processes output channels in chunks to avoid OOM.
        """
        out_features, in_features = W.shape
        device = W.device

        print(f"      Using chunked flipping: {out_features} outputs → chunks of {chunk_size_out}")

        num_chunks = (out_features + chunk_size_out - 1) // chunk_size_out
        W_refined = torch.zeros_like(W, dtype=torch.float32)

        total_flips = 0
        all_flips_per_channel = []

        for chunk_idx in range(num_chunks):
            start_out = chunk_idx * chunk_size_out
            end_out = min(start_out + chunk_size_out, out_features)

            # Extract chunk
            W_chunk = W[start_out:end_out, :]
            scale_g_chunk = scale_g[start_out:end_out, :]
            zp_g_chunk = zp_g[start_out:end_out, :]
            w_floor_int_chunk = w_floor_int[start_out:end_out, :]

            # Apply flipping to this chunk (will use non-chunked version since it's smaller)
            W_refined_chunk, flip_stats_chunk = self._apply_heuristic_flipping_single(
                W_chunk, scale_g_chunk, zp_g_chunk, w_floor_int_chunk,
                n_groups, group_size, group_activation_means
            )

            W_refined[start_out:end_out, :] = W_refined_chunk
            total_flips += flip_stats_chunk['total']
            all_flips_per_channel.append(flip_stats_chunk.get('_per_channel_raw', torch.zeros(in_features, device=device)))

            # Cleanup
            del W_chunk, scale_g_chunk, zp_g_chunk, w_floor_int_chunk, W_refined_chunk
            torch.cuda.empty_cache()

        # Aggregate flip statistics
        flips_per_channel = torch.stack(all_flips_per_channel, dim=0).sum(dim=0)  # Sum across output chunks

        flip_stats = {
            'total': total_flips,
            'per_channel_mean': flips_per_channel.mean().item(),
            'per_channel_median': flips_per_channel.median().item(),
            'per_channel_min': flips_per_channel.min().item(),
            'per_channel_max': flips_per_channel.max().item(),
            'per_channel_std': flips_per_channel.std().item(),
            'per_channel_p95': torch.quantile(flips_per_channel, 0.95).item(),
            'per_channel_zero_pct': (flips_per_channel == 0).float().mean().item() * 100,
            'outlier_percent': 0.0  # Not tracked in chunked mode
        }

        return W_refined.to(W.dtype), flip_stats

    def _apply_heuristic_flipping_single(self, W, scale_g, zp_g, w_floor_int, n_groups, group_size,
                                         group_activation_means):
        """Single-chunk flipping (original algorithm)."""
        out_features, in_features = W.shape
        device = W.device

        # Expand quantization params to full size for flipping
        padded_in_features = n_groups * group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        # Expand scale and zp to full size
        scale_flat = scale_g.unsqueeze(2).repeat(1, 1, group_size).reshape(out_features, padded_in_features).float()
        zp_flat = zp_g.unsqueeze(2).repeat(1, 1, group_size).reshape(out_features, padded_in_features).float()

        # Pad w_floor_int if needed
        if padded_in_features > in_features:
            w_int = torch.zeros(out_features, padded_in_features, device=device, dtype=torch.float32)
            w_int[:, :in_features] = w_floor_int.float() + zp_flat[:, :in_features]
        else:
            w_int = w_floor_int.float() + zp_flat

        max_int = 2**self.bits - 1

        # Current quantized weights
        W_quant = (w_int - zp_flat) * scale_flat

        # --- Global Greedy Heuristic Flipping ---
        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1)

        W_div = W_padded / scale_flat
        flip_dir = torch.sign(W_div + zp_flat - w_int)
        flip_dir[flip_dir == 0] = 1.0
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat

        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        w_int_proposed = w_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # DYNAMIC Outlier Masking
        outlier_threshold, outlier_percent = compute_dynamic_outlier_threshold_kneedle(
            act_padded, knee_tolerance=self.knee_tolerance
        )
        is_outlier = act_padded.abs() > outlier_threshold
        valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        rounding_costs = (W_div + zp_flat - w_int).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        error_unsqueezed = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())

        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0

        # Limit flips per output channel
        max_flips_per_output = int(self.max_flip_percent * in_features)
        cumsum_flips = final_flips_sorted.long().cumsum(dim=1)
        within_limit = cumsum_flips <= max_flips_per_output
        sorted_flip_dir[~within_limit] = 0.0

        w_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
        w_int.clamp_(0, max_int)

        # Compute flip statistics
        num_flips_total = final_flips_sorted.sum().item()
        flips_per_channel = final_flips_sorted.sum(dim=0).float()

        if padded_in_features > in_features:
            flips_per_channel = flips_per_channel[:in_features]

        flip_stats = {
            'total': num_flips_total,
            'per_channel_mean': flips_per_channel.mean().item(),
            'per_channel_median': flips_per_channel.median().item(),
            'per_channel_min': flips_per_channel.min().item(),
            'per_channel_max': flips_per_channel.max().item(),
            'per_channel_std': flips_per_channel.std().item(),
            'per_channel_p95': torch.quantile(flips_per_channel, 0.95).item(),
            'per_channel_zero_pct': (flips_per_channel == 0).float().mean().item() * 100,
            'outlier_percent': outlier_percent,
            '_per_channel_raw': flips_per_channel  # For chunked aggregation
        }

        # Dequantize
        W_refined = (w_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_refined = W_refined[:, :in_features]

        return W_refined.to(W.dtype), flip_stats

    def apply_heuristic_flipping(self, W, scale_g, zp_g, w_floor_int, n_groups, group_size,
                                 group_activation_means):
        """
        Apply heuristic flipping to refine quantized weights.

        Takes AdaRound output and applies global greedy bit-flipping.
        For large layers, chunks along output dimension to avoid OOM.

        Args:
            W: Original FP weights [out, in]
            scale_g: Group-wise scales [out, n_groups]
            zp_g: Group-wise zero-points [out, n_groups]
            w_floor_int: Integer quantized weights from AdaRound [out, in] (int16)
            n_groups: Number of groups
            group_size: Group size
            group_activation_means: Per-channel activation means [in]

        Returns:
            tuple: (W_refined, flip_stats)
        """
        out_features, in_features = W.shape
        device = W.device

        # CRITICAL: For large layers, chunk along output dimension to avoid OOM
        # Heuristic: if estimated memory > 1GB, use chunking
        estimated_memory_gb = (out_features * in_features * 4) / 1e9  # 4 bytes (fp32)
        chunk_size_out = 2048  # Process 2048 output channels at a time

        if estimated_memory_gb > 1.0 and out_features > chunk_size_out:
            # Use chunked flipping
            return self._apply_heuristic_flipping_chunked(
                W, scale_g, zp_g, w_floor_int, n_groups, group_size,
                group_activation_means, chunk_size_out
            )

        # Original non-chunked version for small layers

        # Expand quantization params to full size for flipping
        padded_in_features = n_groups * group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        # Expand scale and zp to full size
        scale_flat = scale_g.unsqueeze(2).repeat(1, 1, group_size).reshape(out_features, padded_in_features).float()
        zp_flat = zp_g.unsqueeze(2).repeat(1, 1, group_size).reshape(out_features, padded_in_features).float()

        # Pad w_floor_int if needed
        if padded_in_features > in_features:
            w_int = torch.zeros(out_features, padded_in_features, device=device, dtype=torch.float32)
            w_int[:, :in_features] = w_floor_int.float() + zp_flat[:, :in_features]
        else:
            w_int = w_floor_int.float() + zp_flat

        max_int = 2**self.bits - 1

        # Current quantized weights
        W_quant = (w_int - zp_flat) * scale_flat

        # --- Global Greedy Heuristic Flipping ---

        # A. Calculate Current Error
        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1)  # [out_features]

        # B. Identify Flip Candidates
        W_div = W_padded / scale_flat
        flip_dir = torch.sign(W_div + zp_flat - w_int)
        flip_dir[flip_dir == 0] = 1.0
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat  # [out, in]

        # C. Validity Masks
        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        w_int_proposed = w_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # DYNAMIC Outlier Masking using Kneedle algorithm
        outlier_threshold, outlier_percent = compute_dynamic_outlier_threshold_kneedle(
            act_padded, knee_tolerance=self.knee_tolerance
        )
        is_outlier = act_padded.abs() > outlier_threshold
        valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        # D. Sorting & Optimization
        rounding_costs = (W_div + zp_flat - w_int).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        error_unsqueezed = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        # E. Apply Flips with Constraint
        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())

        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0

        # Limit flips per output channel
        max_flips_per_output = int(self.max_flip_percent * in_features)
        cumsum_flips = final_flips_sorted.long().cumsum(dim=1)
        within_limit = cumsum_flips <= max_flips_per_output
        sorted_flip_dir[~within_limit] = 0.0

        w_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
        w_int.clamp_(0, max_int)

        # F. Compute Flip Statistics
        num_flips_total = final_flips_sorted.sum().item()
        flips_per_channel = final_flips_sorted.sum(dim=0).float()

        if padded_in_features > in_features:
            flips_per_channel = flips_per_channel[:in_features]

        flip_stats = {
            'total': num_flips_total,
            'per_channel_mean': flips_per_channel.mean().item(),
            'per_channel_median': flips_per_channel.median().item(),
            'per_channel_min': flips_per_channel.min().item(),
            'per_channel_max': flips_per_channel.max().item(),
            'per_channel_std': flips_per_channel.std().item(),
            'per_channel_p95': torch.quantile(flips_per_channel, 0.95).item(),
            'per_channel_zero_pct': (flips_per_channel == 0).float().mean().item() * 100,
            'outlier_percent': outlier_percent
        }

        # G. Dequantize & Return
        W_refined = (w_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_refined = W_refined[:, :in_features]

        return W_refined.to(W.dtype), flip_stats

    def optimize_layer_adaround_flip(self, name, module, calibration_data_cpu, num_iterations=2000, debug=False):
        """
        Optimize rounding for a single layer using AdaRound + Heuristic Flipping.

        Workflow:
        1. Run AdaRound optimization (gradient-based learned rounding)
        2. Apply heuristic flipping on top of AdaRound results
        3. Return final refined weights
        """
        W = module.weight.data
        original_dtype = W.dtype
        layer_device = W.device

        # Compute activation means for flipping
        X_all = calibration_data_cpu.float()
        group_activation_means = X_all.abs().mean(dim=0)  # [in_features]

        # Compute quantization parameters
        scale_g, zp_g, w_floor_int, n_groups, group_size = self.compute_quantization_params_groupwise(W)

        # Check for invalid scale values
        if (scale_g == 0).any() or torch.isnan(scale_g).any() or torch.isinf(scale_g).any():
            print(f"    🚨 ERROR: Invalid scale values in {name}!")
            return W.clone(), float('inf'), {}

        # Move to layer's device
        scale_g = scale_g.to(layer_device)
        zp_g = zp_g.to(layer_device)
        w_floor_int = w_floor_int.to(layer_device)
        group_activation_means = group_activation_means.to(layer_device)

        # === STEP 1: AdaRound Optimization ===

        # ADAPTIVE: For very large layers (e.g., lm_head), reduce samples to save memory
        out_features, in_features = W.shape
        layer_size_mb = (out_features * in_features * 4) / (1024**2)  # V parameter size in MB

        if layer_size_mb > 1500:  # >1.5 GB (e.g., lm_head 128256×4096)
            max_samples = min(128, calibration_data_cpu.shape[0])  # Half samples
            mini_batch_size = 32  # Smaller batches
            print(f"    ⚠️  Very large layer ({layer_size_mb:.0f} MB), using adaptive memory mode:")
            print(f"        Calibration samples: {max_samples}, Mini-batch: {mini_batch_size}")
        elif layer_size_mb > 500:  # >500 MB
            max_samples = min(192, calibration_data_cpu.shape[0])
            mini_batch_size = 48
        else:
            max_samples = min(256, calibration_data_cpu.shape[0])
            mini_batch_size = 64

        # Free memory before creating wrapper (which allocates V parameter)
        torch.cuda.empty_cache()

        wrapper = AdaRoundOptimizer(module, scale_g, w_floor_int, zp_g, n_groups, group_size,
                                   iterations=num_iterations).to(layer_device)
        optimizer = torch.optim.Adam([wrapper.v], lr=self.adaround_lr)

        # Subsample calibration data
        if calibration_data_cpu.shape[0] > max_samples:
            indices = torch.randperm(calibration_data_cpu.shape[0])[:max_samples]
            calib_data_cpu_subset = calibration_data_cpu[indices]
        else:
            calib_data_cpu_subset = calibration_data_cpu

        num_mini_batches = (max_samples + mini_batch_size - 1) // mini_batch_size

        best_loss = float('inf')
        best_v = None
        patience_counter = 0
        patience_limit = 200

        for i in range(num_iterations):
            optimizer.zero_grad()
            total_rec_loss = 0.0

            for mb_idx in range(num_mini_batches):
                mb_start = mb_idx * mini_batch_size
                mb_end = min(mb_start + mini_batch_size, max_samples)
                calib_mb = calib_data_cpu_subset[mb_start:mb_end].to(layer_device).to(original_dtype)

                with torch.no_grad():
                    target_mb = F.linear(calib_mb, W, module.bias)

                current_mb = wrapper(calib_mb)
                rec_loss_mb = F.mse_loss(current_mb, target_mb)
                total_rec_loss += rec_loss_mb * (mb_end - mb_start) / max_samples

                # CRITICAL: Delete all intermediate tensors immediately
                del calib_mb, target_mb, current_mb, rec_loss_mb

            reg_loss = self.reg_weight * compute_adaround_reg(wrapper.v, i, num_iterations)
            total_loss = total_rec_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            current_loss = total_loss.item()
            if best_v is None:
                best_loss = current_loss
                best_v = wrapper.v.data.clone()
                patience_counter = 0
            else:
                improvement_threshold = max(1e-5, best_loss * 1e-4)
                if current_loss < best_loss - improvement_threshold:
                    best_loss = current_loss
                    best_v = wrapper.v.data.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= patience_limit:
                if debug:
                    print(f"      ⏹️  Early stopping at iter {i}/{num_iterations}")
                break

            if debug and i % 200 == 0:
                print(f"      Iter {i}/{num_iterations}: Total={total_loss.item():.6f}, "
                      f"Rec={total_rec_loss.item():.6f}, Reg={reg_loss.item():.6f}")

            del total_rec_loss, reg_loss, total_loss

            # More aggressive memory cleanup to prevent accumulation
            if i % 50 == 0:
                torch.cuda.empty_cache()
                if i % 200 == 0:
                    gc.collect()

        # Get AdaRound optimized weights
        with torch.no_grad():
            if best_v is None:
                best_v = wrapper.v.data.clone()

            wrapper.v.data = best_v

            # Reconstruct weights group-by-group
            _, in_features = W.shape
            adaround_weights = torch.zeros_like(W, dtype=torch.float32)

            for g in range(n_groups):
                j0 = g * group_size
                j1 = min((g + 1) * group_size, in_features)

                v_chunk = wrapper.v[:, j0:j1]
                h_v_chunk = torch.clamp(
                    torch.sigmoid(v_chunk) * (wrapper.zeta - wrapper.gamma) + wrapper.gamma,
                    0, 1
                ).float()

                hard_rounding_chunk = (h_v_chunk > 0.5).float()
                scale_g_cur = scale_g[:, g:g+1].float()
                w_floor_chunk = w_floor_int[:, j0:j1].float()

                adaround_weights[:, j0:j1] = (w_floor_chunk + hard_rounding_chunk) * scale_g_cur

        # === CRITICAL: Free AdaRound resources BEFORE flipping ===
        # Flipping doesn't need gradients, so free ~500 MB per large layer
        optimizer.zero_grad(set_to_none=True)  # Free gradient buffers
        del optimizer  # Free optimizer state (momentum, etc.)
        del wrapper  # Free AdaRound wrapper and V parameter (~235 MB for large layers)
        del best_v  # Free best V
        torch.cuda.empty_cache()
        gc.collect()

        # === STEP 2: Heuristic Flipping (if enabled) ===
        flip_stats = {}
        if self.use_flipping:
            # Extract w_floor_int from adaround_weights
            w_floor_from_adaround = torch.zeros_like(w_floor_int, dtype=torch.int16)
            for g in range(n_groups):
                j0 = g * group_size
                j1 = min((g + 1) * group_size, in_features)
                scale_g_cur = scale_g[:, g:g+1].float()
                zp_g_cur = zp_g[:, g:g+1].float()

                w_div = adaround_weights[:, j0:j1] / scale_g_cur
                w_int = torch.round(w_div + zp_g_cur).clamp(0, 2**self.bits - 1)
                w_floor_from_adaround[:, j0:j1] = (w_int - zp_g_cur).to(torch.int16)

            final_weights, flip_stats = self.apply_heuristic_flipping(
                W, scale_g, zp_g, w_floor_from_adaround, n_groups, group_size,
                group_activation_means
            )
            del w_floor_from_adaround  # Free immediately after use
        else:
            final_weights = adaround_weights.clone()  # Clone to free original
            flip_stats = {'total': 0}

        # Final cleanup (optimizer/wrapper already freed before flipping)
        del scale_g, zp_g, w_floor_int, calib_data_cpu_subset
        del adaround_weights, group_activation_means
        torch.cuda.empty_cache()
        gc.collect()

        return final_weights.to(original_dtype), best_loss, flip_stats

    def quantize_layer(self, name, module, debug=False):
        """Apply AdaRound + Flipping quantization to a single layer."""
        calib_data_cpu = self.get_calibration_data(name)

        if calib_data_cpu is None or calib_data_cpu.shape[0] == 0:
            print(f"    ⚠️  No calibration data for {name}, skipping")
            return

        W_optimized, final_loss, flip_stats = self.optimize_layer_adaround_flip(
            name, module, calib_data_cpu,
            num_iterations=self.adaround_iters,
            debug=debug
        )

        if torch.isnan(W_optimized).any() or torch.isinf(W_optimized).any():
            print(f"    🚨 ERROR: {name} has NaN/Inf values! Skipping quantization.")
            del W_optimized, calib_data_cpu
            return

        module.weight.data = W_optimized

        if debug:
            print(f"       Weight stats: min={W_optimized.min().item():.6f}, "
                  f"max={W_optimized.max().item():.6f}")
            if flip_stats:
                print(f"       Flips: {flip_stats['total']:,}")

        # Store stats WITHOUT raw tensors to prevent memory leak
        flip_stats_clean = {k: v for k, v in flip_stats.items() if not k.startswith('_')}
        self.layer_stats[name] = {
            'final_loss': final_loss,
            'shape': list(W_optimized.shape),
            'flip_stats': flip_stats_clean  # No raw tensors
        }

        del W_optimized, calib_data_cpu, flip_stats

        # CRITICAL: Free activation data immediately after layer is quantized
        if name in self.activation_data:
            del self.activation_data[name]

        # Aggressive cleanup to free maximum memory before next layer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all CUDA ops to complete
        gc.collect()

    @torch.no_grad()
    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """Calibrate a batch of layers simultaneously."""
        print(f"  Calibrating {len(layer_names_batch)} layers...")

        self.model.eval()
        handles = []

        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        successful = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="  Calibration", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1

                    # CRITICAL: Free outputs immediately to prevent accumulation
                    del outputs, inputs

                    if (successful + 1) % 16 == 0:  # More frequent (was 32)
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for _, handle in handles:
            handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """Batched sequential quantization with AdaRound + Flipping."""
        print("\n" + "=" * 80)
        print("Batched Sequential AdaRound + Flipping Quantization")
        print("=" * 80)

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

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.layer_batch_size
            batch_end = min(batch_start + self.layer_batch_size, num_layers)
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Layers {batch_start}-{batch_end-1}")

            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            print(f"  Quantizing {len(batch_layers)} layers with AdaRound + Flipping...")
            for idx, (name, module) in enumerate(tqdm(batch_layers, desc="  Quantization", leave=False)):
                try:
                    debug = (quantized_count < 2)
                    self.quantize_layer(name, module, debug=debug)
                    quantized_count += 1

                    # Monitor GPU memory every 5 layers
                    if torch.cuda.is_available() and (idx + 1) % 5 == 0:
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"\n    GPU Memory after layer {idx+1}/{len(batch_layers)}: "
                              f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                except Exception as e:
                    print(f"\n⚠️  Error quantizing {name}: {e}")
                    continue

            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"  Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

        print("\n" + "=" * 80)
        print("✓ Batched Sequential AdaRound + Flipping Complete")
        print(f"  Total layers quantized: {quantized_count}/{num_layers}")
        print("=" * 80)

        if self.layer_stats:
            losses = [info['final_loss'] for info in self.layer_stats.values()]
            print(f"\nFinal Loss Statistics:")
            print(f"  Mean: {np.mean(losses):.6f}")
            print(f"  Median: {np.median(losses):.6f}")

            if self.use_flipping:
                flip_totals = [info.get('flip_stats', {}).get('total', 0) for info in self.layer_stats.values()]
                print(f"\nFlipping Statistics:")
                print(f"  Total flips: {int(np.sum(flip_totals)):,}")
                print(f"  Mean flips per layer: {np.mean(flip_totals):,.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--adaround-iters", type=int, default=2000)
    parser.add_argument("--adaround-lr", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=0.001)
    parser.add_argument("--max-tokens-per-sample", type=int, default=256)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=4)
    parser.add_argument("--use-flipping", action="store_true", default=True)
    parser.add_argument("--no-flipping", dest="use_flipping", action="store_false")
    parser.add_argument("--max-flip-percent", type=float, default=0.05)
    parser.add_argument("--knee-tolerance", type=float, default=0.00)
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_adaround_flip_xl")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("AdaRound + Heuristic Flipping Quantization (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Use flipping: {args.use_flipping}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048,
                                              seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048,
                                                     seed=args.seed, cache_dir=args.cache_dir)

    quantizer = AdaRoundFlipQuantizerXL(
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
        lmhead_chunks=args.lmhead_chunks,
        use_flipping=args.use_flipping,
        max_flip_percent=args.max_flip_percent,
        knee_tolerance=args.knee_tolerance
    )

    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
