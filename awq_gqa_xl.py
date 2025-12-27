"""
AWQ-GQA: Combined AWQ Quantization with GQA-Specific ReFlip Refinement

Pipeline:
1. Apply AWQ quantization to all linear layers (from awq_js_xl.py)
2. Detect Group-Query Attention layers (Q_proj, K_proj, V_proj)
3. Apply ReFlip refinement to GQA layers (from fast_quantize_qkv.py)

This provides:
- General AWQ quantization for all layers
- Specialized attention-aware refinement for GQA layers (Llama 3+, etc.)

Usage:
    python awq_gqa_xl.py \
        --model-path ./models/Llama-3-8B \
        --output-dir ./quantized_models/llama3_awq_gqa \
        --n-calib 128 \
        --apply-gqa-reflip  # Enable GQA ReFlip refinement
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import os
import numpy as np
from tqdm import tqdm
import gc
import copy

# Import AWQ quantizer from awq_js_xl
from awq_js_xl import JamesSteinHeuristicAWQQuantizerXL, find_knee_point, compute_james_stein_mean

# Import ReFlip function from fast_quantize_qkv
from fast_quantize_qkv import quantize_qkv_reflip_fast


def is_gqa_layer(layer_name):
    """
    Detect if a layer is part of Group-Query Attention.

    GQA layers typically have names like:
    - model.layers.0.self_attn.q_proj
    - model.layers.0.self_attn.k_proj
    - model.layers.0.self_attn.v_proj
    """
    gqa_keywords = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']
    layer_lower = layer_name.lower()

    return any(keyword in layer_lower for keyword in gqa_keywords)


def get_layer_group(layer_name):
    """
    Extract layer group from name (e.g., 'model.layers.0.self_attn').

    Returns: (layer_idx, attn_group) or None
    """
    parts = layer_name.split('.')

    # Find layer index
    layer_idx = None
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                break
            except ValueError:
                continue

    if layer_idx is None:
        return None

    # Get attention group (everything before q_proj/k_proj/v_proj)
    if 'self_attn' in parts:
        attn_idx = parts.index('self_attn')
        attn_group = '.'.join(parts[:attn_idx + 1])
        return (layer_idx, attn_group)

    return None


class AWQGQAQuantizer(JamesSteinHeuristicAWQQuantizerXL):
    """
    Extended AWQ Quantizer with GQA ReFlip refinement.

    Inherits from JamesSteinHeuristicAWQQuantizerXL and adds:
    - GQA layer detection
    - Model state preservation (only if ReFlip enabled)
    - ReFlip refinement after AWQ quantization
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, use_heuristic=True, knee_tolerance=0.1,
                 max_tokens_per_sample=512, layer_batch_size=16, lmhead_chunks=4,
                 max_flip_percent=0.05, use_james_stein=True,
                 apply_gqa_reflip=False, gqa_critical_dim_pct=0.15,
                 gqa_max_flip_pct=0.05):
        """
        Initialize AWQ-GQA Quantizer.

        Args:
            model: Pre-loaded model
            tokenizer: Pre-loaded tokenizer
            (other args same as JamesSteinHeuristicAWQQuantizerXL)
            apply_gqa_reflip: Enable GQA ReFlip refinement
            gqa_critical_dim_pct: Percentage of moderate dimensions for ReFlip
            gqa_max_flip_pct: Max flip percentage for ReFlip
        """
        # Initialize parent class (same as awq_js_xl.py)
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            bits=bits,
            n_grid=n_grid,
            group_size=group_size,
            use_heuristic=use_heuristic,
            knee_tolerance=knee_tolerance,
            max_tokens_per_sample=max_tokens_per_sample,
            layer_batch_size=layer_batch_size,
            lmhead_chunks=lmhead_chunks,
            max_flip_percent=max_flip_percent,
            use_james_stein=use_james_stein
        )

        # GQA-specific attributes
        self.apply_gqa_reflip = apply_gqa_reflip
        self.gqa_critical_dim_pct = gqa_critical_dim_pct
        self.gqa_max_flip_pct = gqa_max_flip_pct

        # Storage for GQA refinement (only if enabled)
        self.original_state_dict = None  # Store original weights before quantization

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """
        Override to add GQA ReFlip refinement after AWQ quantization.

        CRITICAL: When apply_gqa_reflip=False, this behaves EXACTLY like parent class.
        """
        if not self.apply_gqa_reflip:
            # NO CHANGES - call parent directly
            super().quantize_model_sequential(calibration_data, n_samples)
            return

        # ===== GQA ReFlip Mode =====
        print("\n" + "=" * 80)
        print("AWQ-GQA: Combined Quantization Pipeline")
        print("=" * 80)
        print(f"  Step 1: Save original model state")
        print(f"  Step 2: AWQ quantization for all layers")
        print(f"  Step 3: GQA ReFlip refinement")
        print("=" * 80)

        # Step 1: Save original state dict (only GQA layers to save memory)
        print("\n[Step 1] Saving original GQA layer weights...")
        self.original_state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                # Store on CPU to save GPU memory
                self.original_state_dict[name + '.weight'] = module.weight.data.clone().cpu()

        num_saved = len(self.original_state_dict)
        print(f"  ✓ Saved {num_saved} GQA layer weights")

        # Step 2: Standard AWQ quantization
        print("\n[Step 2] Running AWQ quantization...")
        super().quantize_model_sequential(calibration_data, n_samples)

        # Step 3: GQA ReFlip refinement
        print("\n[Step 3] Applying GQA ReFlip refinement...")
        self.apply_gqa_reflip_refinement(calibration_data, n_samples)

        # Clean up
        self.original_state_dict = None
        torch.cuda.empty_cache()
        gc.collect()

    def apply_gqa_reflip_refinement(self, calibration_data, n_samples):
        """
        Apply ReFlip refinement to GQA layers.

        For each attention layer:
        1. Group Q, K, V projections
        2. Re-capture activations (on-demand)
        3. Apply ReFlip to correct attention score errors
        4. Update quantized weights

        Args:
            calibration_data: Same calibration data used for AWQ
            n_samples: Number of calibration samples
        """
        print("\n" + "=" * 80)
        print("GQA ReFlip Refinement")
        print("=" * 80)

        # Group GQA layers by attention block
        attn_groups = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                group_info = get_layer_group(name)
                if group_info:
                    layer_idx, attn_group = group_info

                    if attn_group not in attn_groups:
                        attn_groups[attn_group] = {}

                    # Identify projection type
                    if 'q_proj' in name.lower() or 'query' in name.lower():
                        attn_groups[attn_group]['q_proj'] = (name, module)
                    elif 'k_proj' in name.lower() or 'key' in name.lower():
                        attn_groups[attn_group]['k_proj'] = (name, module)
                    elif 'v_proj' in name.lower() or 'value' in name.lower():
                        attn_groups[attn_group]['v_proj'] = (name, module)

        print(f"  Found {len(attn_groups)} attention groups")

        refined_count = 0
        for attn_group, projs in tqdm(attn_groups.items(), desc="  Refining GQA layers"):
            # Check if we have all required projections
            if 'q_proj' not in projs or 'k_proj' not in projs:
                print(f"    ⚠️  Skipping {attn_group}: Missing Q or K projection")
                continue

            try:
                # Re-capture activations for this group only
                activations = self.capture_activations_for_group(projs, calibration_data, n_samples)

                # Apply refinement
                self.refine_attention_group(attn_group, projs, activations)
                refined_count += 1

                # Clear activations immediately
                del activations
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"    ⚠️  Error refining {attn_group}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n  ✓ Refined {refined_count}/{len(attn_groups)} attention groups")
        print("=" * 80)

    def capture_activations_for_group(self, projs, calibration_data, n_samples):
        """
        Re-capture activations for a specific attention group.

        Args:
            projs: Dictionary of {q_proj, k_proj, v_proj} tuples
            calibration_data: Calibration text data
            n_samples: Number of samples to use

        Returns:
            dict: {layer_name: activations}
        """
        activations = {}

        # Create temporary storage for this group
        temp_activation_data = {}

        def get_temp_hook(name):
            def hook(_module, input, _output):
                if name not in temp_activation_data:
                    temp_activation_data[name] = []
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input

                # Subsample tokens if needed
                if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                    seq_len = inp.shape[1]
                    indices = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                    indices = indices.sort()[0]
                    inp = inp[:, indices, :]

                # Store on CPU to save GPU memory
                temp_activation_data[name].append(inp.detach().cpu().float())
            return hook

        # Register hooks for Q and K projections only
        hooks = []
        for proj_type in ['q_proj', 'k_proj']:
            if proj_type in projs:
                name, module = projs[proj_type]
                handle = module.register_forward_hook(get_temp_hook(name))
                hooks.append(handle)

        # Run forward passes to capture activations
        self.model.eval()
        with torch.no_grad():
            for text in calibration_data[:n_samples]:
                if isinstance(text, str):
                    inputs = self.tokenizer(text, return_tensors='pt',
                                          max_length=self.max_tokens_per_sample,
                                          truncation=True).to(self.device)
                    self.model(**inputs, use_cache=False)

        # Remove hooks
        for handle in hooks:
            handle.remove()

        # Return captured activations
        return temp_activation_data

    def refine_attention_group(self, attn_group, projs, activations):
        """
        Apply ReFlip refinement to a single attention group (Q, K, V).

        This method:
        1. Extracts original and AWQ-quantized weights
        2. Prepares activations
        3. Calls fast ReFlip to refine Q projections
        4. Updates model weights with refined versions
        """
        q_name, q_module = projs['q_proj']
        k_name, k_module = projs['k_proj']

        # Check if we have original weights
        q_weight_key = q_name + '.weight'
        k_weight_key = k_name + '.weight'

        if (q_weight_key not in self.original_state_dict or
            k_weight_key not in self.original_state_dict or
            q_name not in activations or
            k_name not in activations):
            print(f"      ⚠️  Missing data for {attn_group}, skipping")
            return

        # ===== 1. Prepare Activations =====
        X_q_list = activations[q_name]
        X_concat = []
        for x in X_q_list:
            if x.dim() == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x
            X_concat.append(x_flat)

        X_all = torch.cat(X_concat, dim=0)  # [total_samples, hidden_dim]

        # Compute raw mean activation
        raw_mean = X_all.mean(dim=0).cpu().float()  # [hidden_dim]

        # Apply James-Stein estimator (same as AWQ)
        if self.use_james_stein:
            js_mean = compute_james_stein_mean(raw_mean)
            X = js_mean.numpy()
        else:
            X = raw_mean.numpy()

        del X_concat, X_all, raw_mean

        # ===== 2. Extract Original Weights =====
        Wq_orig = self.original_state_dict[q_weight_key].cpu().float().numpy()
        Wk_orig = self.original_state_dict[k_weight_key].cpu().float().numpy()

        # ===== 3. Extract AWQ-Quantized Weights (Current State) =====
        Wq_awq = q_module.weight.data.cpu().float().numpy()
        Wk_awq = k_module.weight.data.cpu().float().numpy()

        # Re-quantize AWQ weights to get clean INT4 representation
        from utils_qkv import quantize_weight_groupwise_int4

        Wq_heuristic, Wq_scales_clean, Wq_zp_clean, Wq_int_heuristic = \
            quantize_weight_groupwise_int4(Wq_awq, group_size=self.group_size, method='nearest')
        Wk_heuristic, _, _, _ = quantize_weight_groupwise_int4(Wk_awq, group_size=self.group_size)

        # ===== 4. Reshape to Multi-Head Format =====
        # Simplified: Treat entire weight matrix as single head
        num_heads = 1
        head_dim = Wq_orig.shape[0]
        hidden_dim = Wq_orig.shape[1]

        Wq_orig_3d = Wq_orig.reshape(num_heads, head_dim, hidden_dim)
        Wk_orig_3d = Wk_orig.reshape(num_heads, Wk_orig.shape[0], hidden_dim)
        Wq_heuristic_3d = Wq_heuristic.reshape(num_heads, head_dim, hidden_dim)
        Wk_heuristic_3d = Wk_heuristic.reshape(num_heads, Wk_orig.shape[0], hidden_dim)
        Wq_int_heuristic_3d = Wq_int_heuristic.reshape(num_heads, head_dim, hidden_dim)

        Wq_scales_3d = Wq_scales_clean.reshape(num_heads, head_dim, -1)
        Wq_zp_3d = Wq_zp_clean.reshape(num_heads, head_dim, -1)

        # ===== 5. Compute Q and K Values =====
        Q_orig_all = np.zeros(num_heads * head_dim)
        Q_heuristic_all = np.zeros(num_heads * head_dim)

        for h in range(num_heads):
            Q_orig_all[h*head_dim:(h+1)*head_dim] = X @ Wq_orig_3d[h].T
            Q_heuristic_all[h*head_dim:(h+1)*head_dim] = X @ Wq_heuristic_3d[h].T

        Q_orig_all = Q_orig_all.reshape(num_heads, head_dim)
        Q_heuristic_all = Q_heuristic_all.reshape(num_heads, head_dim)

        K_heuristic = X @ Wk_heuristic.T

        # ===== 6. Apply ReFlip Refinement =====
        try:
            (Wq_refined, Wq_scales_out, Wq_zp_out, Wq_int_out,
             Wk_refined, Wk_scales_out, Wk_zp_out, Wk_int_out,
             reflip_stats) = quantize_qkv_reflip_fast(
                Wq=Wq_orig_3d,
                Wk=Wk_orig_3d,
                X=X,
                Q_orig_all=Q_orig_all,
                Q_heuristic_all=Q_heuristic_all,
                Wq_heuristic=Wq_heuristic_3d,
                Wk_heuristic=Wk_heuristic_3d,
                K_heuristic=K_heuristic,
                Wq_int_heuristic=Wq_int_heuristic_3d,
                Wq_scales_heuristic=Wq_scales_3d,
                Wq_zp_heuristic=Wq_zp_3d,
                critical_dim_pct=self.gqa_critical_dim_pct,
                knee_tolerance=0.0,
                group_size=self.group_size,
                max_flip_pct=self.gqa_max_flip_pct,
                correction_scale=1.0,
                debug=False
            )

            # ===== 7. Update Model Weights =====
            Wq_refined_2d = Wq_refined.reshape(head_dim, hidden_dim)
            q_module.weight.data = torch.from_numpy(Wq_refined_2d).to(
                q_module.weight.dtype
            ).to(q_module.weight.device)

            print(f"      ✓ Refined {attn_group}")
            print(f"        Total flips: {reflip_stats['total_flips']}")
            print(f"        Flip rate: {reflip_stats['flip_rate_pct']:.3f}%")

        except Exception as e:
            print(f"      ⚠️  ReFlip failed for {attn_group}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='AWQ-GQA: Combined Quantization')

    # All arguments from awq_js_xl.py
    parser.add_argument("--n-calib", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4], help="Quantization bit width (default: 4)")
    parser.add_argument("--use-heuristic", action="store_true", default=True,
                        help="Use heuristic flip correction (default: True)")
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false",
                        help="Disable heuristic flip correction")
    parser.add_argument("--use-james-stein", action="store_true", default=True,
                        help="Use James-Stein estimator for activation means (default: True)")
    parser.add_argument("--no-james-stein", dest="use_james_stein", action="store_false",
                        help="Disable James-Stein estimator")
    parser.add_argument("--knee-tolerance", type=float, default=0.000,
                        help="Tolerance offset for Kneedle algorithm (default: 0.0)")
    parser.add_argument("--max-flip-percent", type=float, default=0.05,
                        help="Max percentage of weights to flip per channel (default: 0.05)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                        help="Max tokens per calibration sample (default: 2048)")
    parser.add_argument("--layer-batch-size", type=int, default=16,
                        help="Number of layers to quantize per batch (default: 16)")
    parser.add_argument("--lmhead-chunks", type=int, default=8,
                        help="Number of chunks to split lm_head into (default: 8)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/llama3_awq_gqa")
    parser.add_argument("--model-path", type=str, default="./models/Llama-3-8B",
                        help="Path to model or HuggingFace model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"],
                        help="Calibration dataset to use")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                        help="Directory to cache calibration data")

    # GQA ReFlip options
    parser.add_argument('--apply-gqa-reflip', action='store_true',
                        help='Apply ReFlip refinement to GQA layers')
    parser.add_argument('--gqa-critical-dim-pct', type=float, default=0.15,
                        help='Percentage of moderate dimensions for ReFlip')
    parser.add_argument('--gqa-max-flip-pct', type=float, default=0.05,
                        help='Max flip percentage for ReFlip')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AWQ-GQA: Combined Quantization with GQA ReFlip Refinement")
    print("=" * 80)
    print(f"  Model: {args.model_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Calibration samples: {args.n_calib}")
    print(f"  GQA ReFlip: {'Enabled' if args.apply_gqa_reflip else 'Disabled'}")
    if args.apply_gqa_reflip:
        print(f"    - Critical dim %: {args.gqa_critical_dim_pct}")
        print(f"    - Max flip %: {args.gqa_max_flip_pct}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer (same as awq_js_xl.py)
    print(f"\nLoading model and tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Create quantizer with GQA support
    quantizer = AWQGQAQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        knee_tolerance=args.knee_tolerance,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        max_flip_percent=args.max_flip_percent,
        use_james_stein=args.use_james_stein,
        apply_gqa_reflip=args.apply_gqa_reflip,
        gqa_critical_dim_pct=args.gqa_critical_dim_pct,
        gqa_max_flip_pct=args.gqa_max_flip_pct
    )

    # Load calibration data (exactly as in awq_js_xl.py)
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(quantizer.tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(quantizer.tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    # Quantize model (includes GQA ReFlip if enabled)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    # Save quantized model (exactly as awq_js_xl.py)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
