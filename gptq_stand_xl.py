"""
GPTQ Implementation for Extra Large Models (XL)
Based on "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

Key Features:
- Layer-wise quantization with Hessian-based weight updates
- Cholesky decomposition for numerical stability
- Batched block processing (B=128)
- Sequential quantization for memory efficiency
- Special handling for large lm_head layers

Algorithm (from GPTQ paper):
1. Accumulate Hessian: H = 2XX^T + λI (dampening)
2. Compute Cholesky decomposition: H^-1 = Cholesky(H)^T
3. For each block of B=128 columns:
   - Quantize column j: Q[:,j] = quant(W[:,j])
   - Compute error: err = (W[:,j] - Q[:,j]) / H^-1[j,j]
   - Update remaining columns in block: W[:,j+1:i+B] -= err * H^-1[j,j+1:i+B]
4. After block, update all remaining weights: W[:,(i+B):] -= E * H^-1[i:(i+B),(i+B):]
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import gc
import sys
import math
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  Warning: psutil not installed. Memory monitoring disabled.")
    print("   Install with: pip install psutil")

# Import calibration utils (assuming they exist in the same folder)
from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data


class GPTQQuantizer:
    """
    GPTQ Quantizer implementing the algorithm from the paper.

    This is the core quantization class that handles one linear layer at a time.
    Based on Algorithm 1 from the GPTQ paper (ICLR 2023).
    """

    def __init__(self, layer, device='cuda'):
        """
        Initialize GPTQ quantizer for a single layer.

        Args:
            layer: nn.Linear layer to quantize
            device: Device to use for computation
        """
        self.layer = layer
        self.dev = device

        W = layer.weight.data.clone()
        self.rows = W.shape[0]  # out_features
        self.columns = W.shape[1]  # in_features

        # Hessian matrix H = 2XX^T
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        """
        Accumulate Hessian from a batch of inputs.

        Args:
            inp: Input activations [batch_size, ..., in_features]
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]

        # Reshape to [in_features, num_samples]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        # Update Hessian: H = 2XX^T
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        # Normalize for numerical stability
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def quantize(self, bits=4, groupsize=-1, blocksize=128, percdamp=0.01):
        """
        Perform GPTQ quantization on the layer.

        Args:
            bits: Number of bits for quantization (default: 4)
            groupsize: Group size for group-wise quantization (-1 = per-channel)
            blocksize: Block size for batch processing (default: 128)
            percdamp: Percentage dampening for numerical stability (default: 0.01)

        Returns:
            Quantization error (sum of squared errors)
        """
        W = self.layer.weight.data.clone().float()

        tick = time.time()

        # Handle dead neurons (zero diagonal in Hessian)
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Dampening for numerical stability
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        # Cholesky decomposition for numerical stability
        # H^-1 = (L L^T)^-1 where L is lower triangular
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # Quantization
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # Quantization parameters
        maxq = 2 ** bits - 1

        # Process in blocks of size `blocksize`
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Process each column in the block
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Compute quantization parameters for this column
                if groupsize != -1:
                    # Group-wise quantization
                    if (i1 + i) % groupsize == 0:
                        # Compute scale and zero point for this group
                        group_start = i1 + i
                        group_end = min(group_start + groupsize, self.columns)
                        w_group = W[:, group_start:group_end]

                        # Asymmetric quantization
                        w_min = w_group.min(dim=1, keepdim=True)[0]
                        w_max = w_group.max(dim=1, keepdim=True)[0]
                        scale = (w_max - w_min) / maxq
                        scale = scale.clamp(min=1e-8)
                        zero = torch.round(-w_min / scale).clamp(0, maxq)

                        # Store for this group
                        self.current_scale = scale[:, 0]
                        self.current_zero = zero[:, 0]
                else:
                    # Per-channel quantization
                    w_min = w.min()
                    w_max = w.max()
                    scale = (w_max - w_min) / maxq
                    scale = max(scale, 1e-8)
                    zero = torch.clamp(torch.round(-w_min / scale), 0, maxq)

                    self.current_scale = torch.tensor([scale], device=w.device).expand(w.shape[0])
                    self.current_zero = torch.tensor([zero], device=w.device).expand(w.shape[0])

                # Quantize
                q = torch.clamp(
                    torch.round(w / self.current_scale + self.current_zero),
                    0, maxq
                )
                q = self.current_scale * (q - self.current_zero)

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                # Compute error and update remaining weights in block
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # Update all remaining weights outside the block
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        elapsed = time.time() - tick
        error = torch.sum(Losses).item()

        # Update layer weights
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        return error, elapsed

    def free(self):
        """Free memory."""
        self.H = None
        torch.cuda.empty_cache()


class GPTQStandXLQuantizer:
    """
    GPTQ Quantizer with interface similar to AWQ (awq_stand_xl.py).
    Handles batched sequential quantization for extra large models.
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, group_size=128,
                 blocksize=128, percdamp=0.01, max_tokens_per_sample=2048, skip_lmhead=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.group_size = group_size
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.max_tokens_per_sample = max_tokens_per_sample
        self.skip_lmhead = skip_lmhead

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[GPTQ Quantizer Initialized - XL Version]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Block size: {blocksize}")
        print(f"  Dampening: {percdamp}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Quantization: {'Group-wise' if group_size > 0 else 'Per-channel'} ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Skip lm_head: {skip_lmhead}")

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # Suppress unused parameter warnings - we need them for hook signature
            _ = module
            _ = output

            if name not in self.activation_data:
                self.activation_data[name] = []

            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if sequence is too long
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len, device=inp.device)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            # Store activation on CPU (use float32 for numerical stability)
            inp_stored = inp.detach().cpu().float().clone()
            self.activation_data[name].append(inp_stored)
            del inp

        return hook

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """
        Calibrate a BATCH of layers simultaneously.

        Args:
            layer_names_batch: List of (name, module) tuples
            calibration_data: Calibration text data
            n_samples: Number of calibration samples
        """
        # Clear any previous activation data
        self.activation_data = {}

        # Register hooks for ALL layers in this batch
        handles = []
        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        # Run calibration data through model ONCE for all layers in batch
        successful_passes = 0
        with torch.no_grad():
            for i, text in enumerate(calibration_data[:n_samples]):
                try:
                    inputs = self.tokenizer(
                        text, return_tensors="pt",
                        truncation=True, max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    _ = self.model(**inputs, use_cache=False, return_dict=True)
                    successful_passes += 1
                    del inputs

                    # Aggressive cleanup
                    if (i + 1) % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                except Exception as e:
                    if i == 0:
                        print(f"\n⚠️  Forward pass error: {str(e)[:100]}")
                    continue

        # Remove all hooks
        for name, handle in handles:
            handle.remove()

        if successful_passes == 0:
            print(f"\n❌ FATAL: No successful forward passes for batch!")

        # Verify activations were captured
        for name, _ in layer_names_batch:
            if name not in self.activation_data:
                self.activation_data[name] = []

    def quantize_layer(self, name, module):
        """
        Quantize a single layer using GPTQ.

        Args:
            name: Layer name
            module: nn.Linear module to quantize

        Returns:
            Dictionary with quantization statistics
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            print(f"  ⚠️  No calibration data for {name}, skipping")
            return None

        # Create GPTQ quantizer for this layer
        gptq = GPTQQuantizer(module, device=self.device)

        # Add calibration batches to accumulate Hessian
        for act_batch in self.activation_data[name]:
            act_batch = act_batch.to(self.device)
            gptq.add_batch(act_batch)
            del act_batch

        # Perform quantization
        error, elapsed = gptq.quantize(
            bits=self.bits,
            groupsize=self.group_size,
            blocksize=self.blocksize,
            percdamp=self.percdamp
        )

        # Free memory
        gptq.free()
        del gptq
        torch.cuda.empty_cache()

        return {
            'error': error,
            'time': elapsed,
            'nsamples': sum(x.reshape(-1, x.shape[-1]).shape[0]
                          for x in self.activation_data[name])
        }

    def quantize_model_sequential(self, calibration_data, n_samples=500, layer_batch_size=16):
        """
        BATCHED SEQUENTIAL QUANTIZATION for the entire model.

        Args:
            calibration_data: Text data for calibration
            n_samples: Number of calibration samples
            layer_batch_size: Number of layers to calibrate simultaneously
        """
        try:
            print("\n" + "=" * 80)
            print("GPTQ BATCHED SEQUENTIAL QUANTIZATION (XL Version)")
            print("=" * 80)

            if HAS_PSUTIL:
                initial_ram = psutil.virtual_memory().percent
                print(f"Initial System RAM: {initial_ram:.1f}%")

            # Get all linear layers
            all_layers = [
                (name, module) for name, module in self.model.named_modules()
                if isinstance(module, nn.Linear)
            ]

            # Filter out lm_head if skip_lmhead is True
            if self.skip_lmhead:
                layer_names = [
                    (name, module) for name, module in all_layers
                    if 'lm_head' not in name.lower()
                ]
                skipped_layers = [name for name, _ in all_layers if 'lm_head' in name.lower()]
                if skipped_layers:
                    print(f"\n⚠️  Skipping lm_head quantization (--skip-lmhead enabled)")
                    print(f"   Skipped layers: {', '.join(skipped_layers)}")
            else:
                layer_names = all_layers

            print(f"\nFound {len(all_layers)} linear layers total")
            print(f"Quantizing {len(layer_names)} layers (skipped {len(all_layers) - len(layer_names)})")
            print(f"Batch size: {layer_batch_size} layers per batch")
            num_batches = (len(layer_names) + layer_batch_size - 1) // layer_batch_size
            print(f"Total batches: {num_batches}")

            quantized_count = 0
            total_error = 0
            total_time = 0

            # Process layers in batches
            for batch_idx in range(num_batches):
                batch_start = batch_idx * layer_batch_size
                batch_end = min(batch_start + layer_batch_size, len(layer_names))
                batch_layers = layer_names[batch_start:batch_end]

                print(f"\n{'='*60}")
                print(f"Batch {batch_idx + 1}/{num_batches}: Layers {batch_start}-{batch_end-1}")
                print(f"{'='*60}")

                # STEP 1: Calibrate this BATCH
                self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

                # STEP 2: Quantize each layer in the batch
                for name, module in tqdm(batch_layers, desc=f"Quantizing Batch {batch_idx+1}"):
                    try:
                        stats = self.quantize_layer(name, module)

                        if stats is not None:
                            self.layer_stats[name] = stats
                            total_error += stats['error']
                            total_time += stats['time']

                            # Print stats for first few layers
                            if quantized_count < 3:
                                print(f"\n  Layer: {name}")
                                print(f"    Error: {stats['error']:.6f}")
                                print(f"    Time: {stats['time']:.2f}s")
                                print(f"    Samples: {stats['nsamples']}")

                        quantized_count += 1

                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                            raise
                        print(f"\n⚠️  Error quantizing {name}: {e}")
                        continue
                    except Exception as e:
                        print(f"\n⚠️  Error quantizing {name}: {e}")
                        continue

                # STEP 3: Clear activations
                self.activation_data = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if HAS_PSUTIL:
                    ram_pct = psutil.virtual_memory().percent
                    print(f"Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

            print(f"\n✅ GPTQ Quantization Complete!")
            print(f"   Total layers quantized: {quantized_count}/{len(layer_names)}")
            print(f"   Total error: {total_error:.6f}")
            print(f"   Total time: {total_time:.2f}s")

            # Final cleanup
            self.activation_data = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            print("\n" + "=" * 80)
            print("❌ CUDA OUT OF MEMORY ERROR")
            print("=" * 80)
            print("The GPU ran out of memory during quantization.")
            print("Suggestions:")
            print(f"  1. Reduce --layer-batch-size (current: {layer_batch_size})")
            print("  2. Reduce --n-calib (fewer calibration samples)")
            print("  3. Reduce --max-tokens-per-sample")
            print("  4. Reduce --blocksize")
            print("  5. Use a smaller model or GPU with more VRAM")
            print("=" * 80)
            sys.exit(1)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n" + "=" * 80)
                print("❌ CUDA OUT OF MEMORY ERROR")
                print("=" * 80)
                print(f"Error: {e}")
                print("Suggestions:")
                print(f"  1. Reduce --layer-batch-size (current: {layer_batch_size})")
                print("  2. Reduce --n-calib (fewer calibration samples)")
                print("  3. Reduce --max-tokens-per-sample")
                print("  4. Reduce --blocksize")
                print("=" * 80)
                sys.exit(1)
            else:
                raise


def load_wikitext2_simple(n_samples=128):
    """Simple WikiText-2 loader."""
    from datasets import load_dataset
    print(f"Loading WikiText-2 (simple/fast approach)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    return texts[:n_samples]


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ Post-Training Quantization for XL Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128,
                       help="Calibration samples")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for quantization (-1 for per-channel)")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                       help="Quantization bit width (default: 4)")
    parser.add_argument("--blocksize", type=int, default=128,
                       help="Block size for GPTQ algorithm (default: 128)")
    parser.add_argument("--percdamp", type=float, default=0.01,
                       help="Percentage dampening (default: 0.01)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample")
    parser.add_argument("--output-dir", type=str,
                       default="./quantized_models/model_gptq_xl",
                       help="Output directory")
    parser.add_argument("--model-path", type=str,
                       default="./models/Mistral-7B-v0.3",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset")
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Number of layers to calibrate simultaneously")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                       help="Directory to cache calibration data")
    parser.add_argument("--skip-lmhead", action="store_true", default=True,
                       help="Skip lm_head quantization (default: True, hard to quantize)")
    parser.add_argument("--quantize-lmhead", dest="skip_lmhead", action="store_false",
                       help="Enable lm_head quantization (override --skip-lmhead)")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("GPTQ: Accurate Post-Training Quantization (XL Version)")
    print(f"Target Model: {args.model_path}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Bits: {args.bits}")
    print(f"Group size: {args.group_size}")
    print(f"Block size: {args.blocksize}")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,  # Use float16 for GPTQ (as in paper)
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir
        )
    elif args.calib_dataset == "wikitext2-simple":
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=2048,
            seed=args.seed, cache_dir=args.cache_dir
        )

    # Initialize quantizer
    quantizer = GPTQStandXLQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        group_size=args.group_size,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        max_tokens_per_sample=args.max_tokens_per_sample,
        skip_lmhead=args.skip_lmhead
    )

    # Batched sequential quantization
    quantizer.quantize_model_sequential(
        calib_texts,
        n_samples=args.n_calib,
        layer_batch_size=args.layer_batch_size
    )

    # Save model
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("GPTQ QUANTIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
