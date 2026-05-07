"""
AdaRound + Bias Correction — XL Version

This file extends adaround_xl.py with naive bias correction, ported from the
sign-correct convention used in gptq_bc.py.

Bias Correction Algorithm
-------------------------
After AdaRound writes W_q into module.weight.data, compute:

    bias_correction = (W_orig - W_q) @ x_mean         # shape: [out_features]

This equals E[X (W_orig - W_q)^T] in expectation — the mean output error
introduced by quantization. We then update:

    bias_new = bias_old + bias_correction             # PLUS, not minus

Sign convention (same as gptq_bc.py):
    Y_orig  - Y_quant = X (W_orig - W_q)^T
    mean over samples = (W_orig - W_q) @ x_mean
    bias <- bias + correction        (subtracting would double the error)

Where x_mean comes from
-----------------------
AdaRound already captures all layer inputs in `self.activation_data[name]`
(stored as fp16 tensors on CPU). We compute x_mean directly from the
already-collected calibration tensor — no separate hook needed. The mean
is computed token-wise, in fp32, which mirrors gptq_bc.py exactly:

    x_mean = calib_data_cpu.float().mean(dim=0)       # [in_features]

For lm_head chunking: x_mean is shared across chunks (it's the input to the
*whole* lm_head), but each chunk gets its own slice of (W_orig - W_q) and
hence its own slice of the bias correction.

When bias correction is skipped
-------------------------------
- If --no-bias-correction is passed, behaviour matches adaround_xl.py.
- If --skip-lmhead is set, lm_head weights are untouched, so no correction
  is computed for it (delta_W is zero anyway, but we short-circuit).
- If a layer's quantization is skipped due to NaN/Inf or missing calib
  data, no bias correction is applied for that layer (W_orig == W_q).
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

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️ calibration_utils not found. Using internal fallback loaders.")
    def get_c4_calibration_data(*args, **kwargs):
        raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs):
        raise NotImplementedError("Please provide calibration_utils.py")


# ---------------------------------------------------------------------------
# AdaRound optimizer (unchanged from adaround_xl.py)
# ---------------------------------------------------------------------------

class AdaRoundOptimizer(nn.Module):
    """AdaRound wrapper for a single linear layer."""

    def __init__(self, layer, weight_delta, weight_floor, weight_zp,
                 iterations=10000, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.layer = layer

        self.register_buffer('delta',   weight_delta)
        self.register_buffer('w_floor', weight_floor)
        self.register_buffer('w_zp',    weight_zp)

        W = layer.weight.data.float()
        W_div   = W / weight_delta.float()
        W_shift = W_div + weight_zp.float()
        W_frac  = W_shift - torch.floor(W_shift)

        h_init = torch.clamp(W_frac, 0.01, 0.99)
        sigmoid_target = (h_init - gamma) / (zeta - gamma)
        sigmoid_target = torch.clamp(sigmoid_target, 0.01, 0.99)
        v_init = torch.log(sigmoid_target / (1.0 - sigmoid_target))

        self.v = nn.Parameter(v_init.float(), requires_grad=True)

        self.iterations = iterations
        self.zeta  = zeta
        self.gamma = gamma

    def get_soft_rounding(self):
        return torch.clamp(
            torch.sigmoid(self.v) * (self.zeta - self.gamma) + self.gamma,
            0, 1
        )

    def forward(self, x):
        h_v = self.get_soft_rounding().float()
        w_q = ((self.w_floor.float() + h_v) * self.delta.float()).to(x.dtype)
        return F.linear(x, w_q, self.layer.bias)


def compute_adaround_reg(v_parameter, iter_count, max_iter,
                         zeta=1.1, gamma=-0.1, beta_start=2, beta_end=20):
    beta = beta_start + (beta_end - beta_start) * (iter_count / max_iter)
    h_v = torch.clamp(torch.sigmoid(v_parameter) * (zeta - gamma) + gamma, 0, 1)
    reg = (1 - (2 * h_v - 1).abs().pow(beta)).mean()
    return reg


# ---------------------------------------------------------------------------
# Bias correction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_bias_correction(module, W_orig, W_q, x_mean):
    """
    Apply naive bias correction to `module`, sign-correct.

    Args:
        module: nn.Linear whose weights have already been overwritten with W_q.
        W_orig: original (pre-quantization) weights, [out, in], any dtype/device.
        W_q:    quantized weights as actually written, [out, in], same shape.
        x_mean: per-input-channel mean activation, [in], any dtype/device.

    Returns:
        L2 norm of the bias correction (float, for logging).

    Sign convention:
        bc = (W_orig - W_q) @ x_mean
        bias_new = bias_old + bc
    """
    target_device = module.weight.device
    target_dtype  = module.weight.dtype

    # All math in fp32 on the layer's device for numerical stability.
    W_orig_f = W_orig.to(device=target_device, dtype=torch.float32)
    W_q_f    = W_q.to(device=target_device, dtype=torch.float32)
    x_mean_f = x_mean.to(device=target_device, dtype=torch.float32)

    delta_W = W_orig_f - W_q_f                # [out, in]
    bc      = delta_W.matmul(x_mean_f)        # [out]

    bc_cast = bc.to(dtype=target_dtype, device=target_device)

    if module.bias is None:
        out_features = module.weight.shape[0]
        new_bias = nn.Parameter(
            torch.zeros(out_features, dtype=target_dtype, device=target_device)
        )
        module.bias = new_bias
        module.bias.data.copy_(bc_cast)
    else:
        module.bias.data.add_(bc_cast.to(module.bias.data.dtype))

    return bc.norm().item()


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class AdaRoundBCQuantizerXL:
    """AdaRound Quantizer with XL model support, plus naive bias correction."""

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 adaround_iters=10000, adaround_lr=1e-3, reg_weight=0.01,
                 max_tokens_per_sample=512, layer_batch_size=16,
                 lmhead_chunks=4, skip_lmhead=False, bias_correction=True):
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
        self.skip_lmhead = skip_lmhead
        self.bias_correction = bias_correction

        self.activation_data = {}
        self.hooks = []
        self.layer_stats = {}

        # Bias-correction bookkeeping
        self.total_bc_norm = 0.0
        self.n_bc_applied  = 0

        print(f"\n[AdaRound + BC Quantizer XL Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  AdaRound iterations: {adaround_iters} (max, early stopping)")
        print(f"  Learning rate: {adaround_lr}")
        print(f"  Regularization weight: {reg_weight}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample (fp16)")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Quantization: GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  lm_head: split into {lmhead_chunks} chunks "
              f"(skip={skip_lmhead})")
        print(f"  Bias correction: {'ENABLED (sign: bias += (W_orig - W_q) @ x_mean)' if bias_correction else 'DISABLED'}")

    def get_hook(self, name):
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            inp = input[0] if isinstance(input, tuple) else input

            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            self.activation_data[name].append(inp.detach().cpu().half())
        return hook

    @torch.no_grad()
    def get_calibration_data(self, name):
        """Concatenate stored activations for a layer → [total_tokens, in_features] fp32."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None
        X_list = self.activation_data[name]
        X_all = torch.cat(
            [x.reshape(-1, x.shape[-1]).float() for x in X_list], dim=0
        )
        return X_all

    @torch.no_grad()
    def compute_quantization_params_groupwise(self, W):
        """Compute (scale, zero_point, w_floor) for asymmetric group-wise quantization."""
        out_features, in_features = W.shape
        device = W.device
        W = W.float()

        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features,
                                   device=device, dtype=torch.float32)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2 ** self.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat    = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        W_div   = W_padded / scale_flat
        W_floor = torch.floor(W_div + zp_flat).clamp(0, max_int) - zp_flat

        if padded_in_features > in_features:
            scale_flat = scale_flat[:, :in_features]
            zp_flat    = zp_flat[:, :in_features]
            W_floor    = W_floor[:, :in_features]

        return scale_flat, zp_flat, W_floor

    def optimize_layer_adaround(self, name, module, calibration_data_cpu,
                                num_iterations=10000, debug=False):
        """Optimize rounding for a single layer (unchanged from adaround_xl.py)."""
        W = module.weight.data
        original_dtype = W.dtype
        layer_device = W.device

        scale, zp, w_floor = self.compute_quantization_params_groupwise(W)

        if (scale == 0).any() or torch.isnan(scale).any() or torch.isinf(scale).any():
            print(f"    🚨 ERROR: Invalid scale values in {name}!")
            return W.clone(), float('inf')

        w_floor = w_floor.to(layer_device)
        scale   = scale.to(layer_device)
        zp      = zp.to(layer_device)

        wrapper   = AdaRoundOptimizer(module, scale, w_floor, zp,
                                      iterations=num_iterations).to(layer_device)
        optimizer = torch.optim.Adam([wrapper.v], lr=self.adaround_lr)

        max_samples = min(256, calibration_data_cpu.shape[0])
        if calibration_data_cpu.shape[0] > max_samples:
            indices = torch.randperm(calibration_data_cpu.shape[0])[:max_samples]
            calib_data_cpu_subset = calibration_data_cpu[indices]
        else:
            calib_data_cpu_subset = calibration_data_cpu

        mini_batch_size  = 64
        num_mini_batches = (max_samples + mini_batch_size - 1) // mini_batch_size

        best_loss = float('inf')
        best_v    = None
        patience_counter = 0
        patience_limit   = 200

        for i in range(num_iterations):
            optimizer.zero_grad()
            total_rec_loss = 0.0

            for mb_idx in range(num_mini_batches):
                mb_start = mb_idx * mini_batch_size
                mb_end   = min(mb_start + mini_batch_size, max_samples)

                calib_mb = calib_data_cpu_subset[mb_start:mb_end].to(layer_device).to(original_dtype)

                with torch.no_grad():
                    target_mb = F.linear(calib_mb, W, module.bias)

                current_mb = wrapper(calib_mb)
                rec_loss_mb = F.mse_loss(current_mb, target_mb)
                total_rec_loss += rec_loss_mb * (mb_end - mb_start) / max_samples

                del calib_mb, target_mb, current_mb

            reg_loss = self.reg_weight * compute_adaround_reg(
                wrapper.v, i, num_iterations
            )

            total_loss = total_rec_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            current_loss = total_loss.item()

            if best_v is None:
                best_loss = current_loss
                best_v    = wrapper.v.data.clone()
                patience_counter = 0
            else:
                improvement_threshold = max(1e-5, best_loss * 1e-4)
                if current_loss < best_loss - improvement_threshold:
                    best_loss = current_loss
                    best_v    = wrapper.v.data.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"      ⏹️  Early stopping at iter {i}/{num_iterations}")
                break

            if debug and i % 200 == 0:
                print(f"      Iter {i}/{num_iterations}: Total={total_loss.item():.6f}, "
                      f"Rec={total_rec_loss.item():.6f}, Reg={reg_loss.item():.6f}, "
                      f"Patience={patience_counter}/{patience_limit}")

            del total_rec_loss, reg_loss, total_loss

            if i % 100 == 0:
                torch.cuda.empty_cache()

        with torch.no_grad():
            if best_v is None:
                best_v = wrapper.v.data.clone()
            wrapper.v.data = best_v
            h_v_final = wrapper.get_soft_rounding()
            hard_rounding = (h_v_final > 0.5).float()
            optimized_weights = (w_floor + hard_rounding) * scale
            final_weights = optimized_weights.to(original_dtype)

        del wrapper, optimizer, scale, zp, w_floor, calib_data_cpu_subset
        del best_v, h_v_final, hard_rounding, optimized_weights
        torch.cuda.empty_cache()
        gc.collect()

        return final_weights, best_loss

    def quantize_layer(self, name, module, debug=False):
        """Quantize a single layer with AdaRound, then apply bias correction."""
        calib_data_cpu = self.get_calibration_data(name)

        if calib_data_cpu is None or calib_data_cpu.shape[0] == 0:
            print(f"    ⚠️  No calibration data for {name}, skipping")
            return

        # ---- Snapshot original weights BEFORE overwriting -----------------
        # Kept on CPU in fp32 to avoid burning GPU memory while AdaRound runs.
        # We move it back to device when computing the bias correction.
        W_orig_cpu = module.weight.data.detach().to(torch.float32).cpu().clone()
        # -------------------------------------------------------------------

        W_optimized, final_loss = self.optimize_layer_adaround(
            name, module, calib_data_cpu,
            num_iterations=self.adaround_iters,
            debug=debug
        )

        if torch.isnan(W_optimized).any() or torch.isinf(W_optimized).any():
            print(f"    🚨 ERROR: {name} has NaN/Inf values! Skipping quantization.")
            del W_optimized, calib_data_cpu, W_orig_cpu
            return

        # Write quantized weights.
        module.weight.data = W_optimized

        # ---- Bias correction ----------------------------------------------
        # x_mean is computed token-wise from the SAME calibration tensor used
        # by AdaRound (so the mean is consistent with what the layer actually
        # saw during this calibration pass).
        if self.bias_correction:
            x_mean = calib_data_cpu.mean(dim=0)             # [in_features], fp32, on CPU
            bc_norm = apply_bias_correction(
                module=module,
                W_orig=W_orig_cpu,                          # fp32, CPU
                W_q=W_optimized,                            # original_dtype, device
                x_mean=x_mean,                              # fp32, CPU
            )
            self.total_bc_norm += bc_norm
            self.n_bc_applied  += 1
            if debug:
                print(f"       Bias correction: ||bc||_2 = {bc_norm:.6f}")
            del x_mean
        # -------------------------------------------------------------------

        if debug:
            print(f"       Weight stats: min={W_optimized.min().item():.6f}, "
                  f"max={W_optimized.max().item():.6f}, "
                  f"mean={W_optimized.mean().item():.6f}")

        self.layer_stats[name] = {
            'final_loss': final_loss,
            'shape': list(W_optimized.shape),
            'weight_min': W_optimized.min().item(),
            'weight_max': W_optimized.max().item(),
        }

        del W_optimized, calib_data_cpu, W_orig_cpu
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def quantize_lmhead_chunked(self, name, module, num_chunks=4, debug=False):
        """
        Quantize lm_head in chunks. Bias correction is applied per chunk
        because each chunk has its own quantized weights, but the input
        x_mean is shared across chunks (same input feeds the whole lm_head).
        """
        print(f"\n  🔧 Special handling for {name} (split into {num_chunks} chunks)")

        W = module.weight.data
        original_dtype = W.dtype
        layer_device   = W.device
        out_features, in_features = W.shape
        print(f"     Shape: {W.shape} ({W.numel() / 1e6:.1f}M parameters)")

        calib_data_cpu = self.get_calibration_data(name)
        if calib_data_cpu is None or calib_data_cpu.shape[0] == 0:
            print(f"    ⚠️  No calibration data for {name}, skipping")
            return

        # Shared x_mean for the whole lm_head input
        x_mean_full = calib_data_cpu.mean(dim=0) if self.bias_correction else None

        # Snapshot W_orig for the whole layer (CPU, fp32) before mutation
        W_orig_full_cpu = W.detach().to(torch.float32).cpu().clone()

        chunk_size = out_features // num_chunks
        chunk_boundaries = [
            (i * chunk_size,
             out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
            for i in range(num_chunks)
        ]

        W_final_chunks = []
        chunk_stats = []

        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_boundaries):
            print(f"     Processing chunk {chunk_idx + 1}/{num_chunks}: rows {start_idx}-{end_idx}")

            chunk_module = nn.Linear(in_features, end_idx - start_idx,
                                     bias=module.bias is not None)
            chunk_module.weight.data = W[start_idx:end_idx, :].clone()
            if module.bias is not None:
                chunk_module.bias.data = module.bias.data[start_idx:end_idx].clone()
            chunk_module = chunk_module.to(layer_device, dtype=original_dtype)

            W_chunk_optimized, chunk_loss = self.optimize_layer_adaround(
                f"{name}_chunk{chunk_idx}",
                chunk_module,
                calib_data_cpu,
                num_iterations=self.adaround_iters,
                debug=(debug and chunk_idx == 0),
            )

            if torch.isnan(W_chunk_optimized).any() or torch.isinf(W_chunk_optimized).any():
                print(f"    🚨 ERROR: Chunk {chunk_idx} has NaN/Inf! Using original weights.")
                W_final_chunks.append(W[start_idx:end_idx, :].clone().cpu())
            else:
                W_final_chunks.append(W_chunk_optimized.cpu())
            chunk_stats.append({'loss': chunk_loss})

            del chunk_module, W_chunk_optimized
            torch.cuda.empty_cache()

        # Combine and write back
        W_final = torch.cat(W_final_chunks, dim=0).to(layer_device)
        module.weight.data = W_final

        # ---- Bias correction for lm_head ----------------------------------
        # bc[o] = sum_c (W_orig[o,c] - W_q[o,c]) * x_mean[c], for ALL rows.
        # Since lm_head was chunked along rows, we can compute this in one
        # shot now that W_final is assembled — equivalent to per-chunk and
        # cheaper to write.
        if self.bias_correction:
            bc_norm = apply_bias_correction(
                module=module,
                W_orig=W_orig_full_cpu,
                W_q=W_final,
                x_mean=x_mean_full,
            )
            self.total_bc_norm += bc_norm
            self.n_bc_applied  += 1
            print(f"     Bias correction: ||bc||_2 = {bc_norm:.6f}")
            del x_mean_full
        # -------------------------------------------------------------------

        avg_loss = float(np.mean([s['loss'] for s in chunk_stats]))
        self.layer_stats[name] = {
            'final_loss': avg_loss,
            'shape': list(W_final.shape),
            'num_chunks': num_chunks,
        }

        loss_str = ', '.join([f'loss_{i+1}={s["loss"]:.6f}'
                              for i, s in enumerate(chunk_stats)])
        print(f"     ✓ Done: {loss_str}")

        del W_final_chunks, chunk_stats, calib_data_cpu, W_orig_full_cpu
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
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
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1
                    if (successful + 1) % 32 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for _, handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        print("\n" + "=" * 80)
        print("Batched Sequential AdaRound + BC Quantization (XL Version)")
        print("=" * 80)

        if HAS_PSUTIL:
            print(f"  Initial System RAM: {psutil.virtual_memory().percent:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]
        num_layers  = len(layer_names)
        num_batches = (num_layers + self.layer_batch_size - 1) // self.layer_batch_size

        print(f"  Total layers: {num_layers}")
        print(f"  Total batches: {num_batches}")
        print("=" * 80)

        quantized_count = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.layer_batch_size
            batch_end   = min(batch_start + self.layer_batch_size, num_layers)
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Layers {batch_start}-{batch_end-1}")

            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            print(f"  Quantizing {len(batch_layers)} layers with AdaRound...")
            for name, module in tqdm(batch_layers, desc="  Quantization", leave=False):
                try:
                    is_lmhead = 'lm_head' in name.lower() or name.endswith('lm_head')
                    if is_lmhead and self.skip_lmhead:
                        print(f"\n  ⏭️  Skipping {name} (--skip-lmhead set)")
                        if name in self.activation_data:
                            del self.activation_data[name]
                        quantized_count += 1
                        continue
                    elif is_lmhead:
                        debug = (quantized_count < 2)
                        self.quantize_lmhead_chunked(
                            name, module,
                            num_chunks=self.lmhead_chunks, debug=debug
                        )
                    else:
                        debug = (quantized_count < 2)
                        self.quantize_layer(name, module, debug=debug)
                    quantized_count += 1
                except Exception as e:
                    print(f"\n⚠️  Error quantizing {name}: {e}")
                    continue

            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                print(f"  Batch {batch_idx+1} complete. RAM: {psutil.virtual_memory().percent:.1f}%")

        print("\n" + "=" * 80)
        print("✓ Batched Sequential AdaRound + BC Quantization Complete")
        print(f"  Total layers quantized: {quantized_count}/{num_layers}")
        if self.bias_correction and self.n_bc_applied > 0:
            print(f"  Bias correction: applied to {self.n_bc_applied} layers, "
                  f"mean ||bc||_2 = {self.total_bc_norm / self.n_bc_applied:.4f}")
        print("=" * 80)

        if self.layer_stats:
            losses = [info['final_loss'] for info in self.layer_stats.values()]
            print(f"\nFinal Loss Statistics:")
            print(f"  Mean: {np.mean(losses):.6f}")
            print(f"  Median: {np.median(losses):.6f}")
            print(f"  Min: {np.min(losses):.6f} | Max: {np.max(losses):.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--adaround-iters", type=int, default=2000)
    parser.add_argument("--adaround-lr", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=0.001)
    parser.add_argument("--max-tokens-per-sample", type=int, default=256)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=4)
    parser.add_argument("--output-dir", type=str,
                        default="./quantized_models/model_adaround_bc_xl")
    parser.add_argument("--model-path", type=str,
                        default="./models/Mistral-7B-v0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")
    parser.add_argument("--skip-lmhead", action="store_true")
    parser.add_argument("--bias-correction", action="store_true", default=True,
                        help="Apply naive bias correction after each layer (default: on)")
    parser.add_argument("--no-bias-correction", dest="bias_correction",
                        action="store_false")
    args = parser.parse_args()
    args.skip_lmhead = True  # default to skipping lm_head to avoid OOM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("AdaRound + Bias Correction Quantization (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    print(f"Bits: {args.bits}")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"AdaRound iterations per layer: {args.adaround_iters}")
    print(f"Learning rate: {args.adaround_lr}")
    print(f"Regularization weight: {args.reg_weight}")
    print(f"lm_head chunks: {args.lmhead_chunks} (skip={args.skip_lmhead})")
    print(f"Bias correction: {args.bias_correction}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir
        )
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset
                       if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir
        )

    quantizer = AdaRoundBCQuantizerXL(
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
        skip_lmhead=args.skip_lmhead,
        bias_correction=args.bias_correction,
    )

    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == "__main__":
    main()