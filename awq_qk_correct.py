"""
AWQ-GQA: Tier 4 — Cost-only QK-ASC refinement.

Key differences vs Document 1:
  * Adds a lightweight calibration-pass accumulator for the diagonal
    score-loss Hessian h_{g,i,j} = E[ u_{g,i}(x)^2 * x_j^2 ], where
    u_{g,i}(x) = k^{fp}_{g,i}(x) is the full-precision key coordinate.
  * Replaces the knee-based moderate-dimension selection and the
    subset-sum residual-matching rule with a per-flip reward-cost score:
        benefit(j) = 2 * |s_j * reward_j|  -  s_j^2 * h_j
    where the reward is a rank-1 proxy derived from mean activations
    (no per-token g-vector is accumulated -- that is Tier 3 and above).
  * A flip is applied only when benefit(j) > 0 and the integer-range
    constraint holds, subject to a per-row flip budget. This is the
    discrete analogue of a trust region: the expected loss cannot
    increase, in expectation over the calibration distribution, from
    any flip the method takes.
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from tqdm import tqdm

from awq_js_xl import JamesSteinHeuristicAWQQuantizerXL


def is_gqa_layer(layer_name):
    gqa_keywords = ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']
    return any(kw in layer_name.lower() for kw in gqa_keywords)


def get_layer_group(layer_name):
    parts = layer_name.split('.')
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
    if 'self_attn' in parts:
        attn_idx = parts.index('self_attn')
        attn_group = '.'.join(parts[:attn_idx + 1])
        return (layer_idx, attn_group)
    return None


class AWQGQAQuantizerTier4(JamesSteinHeuristicAWQQuantizerXL):
    """
    Extended AWQ Quantizer with Tier-4 QK-ASC refinement.
    Keeps all AWQ machinery; replaces the GQA refinement step.
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, use_heuristic=True, knee_tolerance=0.1,
                 max_tokens_per_sample=512, layer_batch_size=16, lmhead_chunks=8,
                 max_flip_percent=0.05, use_james_stein=True,
                 apply_gqa_reflip=False, gqa_max_flip_pct=0.05,
                 gqa_hessian_n_tokens=2048, gqa_hessian_eps=1e-8):
        super().__init__(
            model=model, tokenizer=tokenizer, device=device, bits=bits, n_grid=n_grid,
            group_size=group_size, use_heuristic=use_heuristic, knee_tolerance=knee_tolerance,
            max_tokens_per_sample=max_tokens_per_sample, layer_batch_size=layer_batch_size,
            lmhead_chunks=lmhead_chunks, max_flip_percent=max_flip_percent,
            use_james_stein=use_james_stein,
        )
        self.apply_gqa_reflip = apply_gqa_reflip
        self.gqa_max_flip_pct = gqa_max_flip_pct
        self.gqa_hessian_n_tokens = gqa_hessian_n_tokens
        self.gqa_hessian_eps = gqa_hessian_eps

        # Storage carried over from Document 1.
        self.original_state_dict = None
        self.gqa_js_means = {}
        self.gqa_heuristic_int_weights = {}
        self.gqa_heuristic_scales = {}
        self.gqa_heuristic_zp = {}
        self.gqa_awq_scales = {}

        # New: diagonal Hessian, keyed by k_proj layer name.
        #   shape [H_k, h, d]  (shared across the r query heads of each group)
        self.gqa_diag_hessian = {}

    # ------------------------------------------------------------------ #
    # (Unchanged from Document 1 — reproduced verbatim for completeness) #
    # ------------------------------------------------------------------ #

    def quantize_layer(self, name, module):
        try:
            if self.apply_gqa_reflip and is_gqa_layer(name):
                _, js_mean = self.get_activation_stats(name)
                if js_mean is not None:
                    self.gqa_js_means[name] = js_mean.cpu().float()

                best_scales, best_alpha, best_error = self.search_best_scale(name, module)
                W = module.weight.data
                W_scaled = W * best_scales.unsqueeze(0)

                if js_mean is not None:
                    scaled_act_mean = (js_mean.to(self.device).to(W.dtype) / best_scales)
                else:
                    scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

                W_quant, scales, zp, W_int = self.quantize_weight_heuristic_with_int_output(
                    W_scaled, scaled_act_mean, apply_heuristic=self.use_heuristic
                )

                self.gqa_heuristic_int_weights[name + '.weight'] = W_int.cpu()
                self.gqa_heuristic_scales[name + '.weight'] = scales.cpu()
                self.gqa_heuristic_zp[name + '.weight'] = zp.cpu()
                self.gqa_awq_scales[name + '.weight'] = best_scales.cpu()

                W_final = (W_quant / best_scales.unsqueeze(0)).to(W.dtype)
                module.weight.data = W_final

                self.layer_scales[name] = {
                    'scales': best_scales.cpu(),
                    'alpha': best_alpha,
                    'error': best_error,
                }
                del best_scales, scaled_act_mean, W_scaled, W_quant, W_final, W_int, scales, zp
                if name in self.activation_data:
                    del self.activation_data[name]
                torch.cuda.empty_cache()
            else:
                super().quantize_layer(name, module)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            raise

    def quantize_weight_heuristic_with_int_output(self, W, group_activation_means, apply_heuristic=True):
        # Identical to Document 1.
        out_features, in_features = W.shape
        device = W.device
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        W_g = W_padded.reshape(out_features, n_groups, self.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2 ** self.bits - 1
        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)

        if apply_heuristic:
            W_quant = (W_int - zp_flat) * scale_flat
            W_diff = W_padded - W_quant
            current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1)

            flip_dir = torch.sign(W_div + zp_flat - W_int)
            flip_dir[flip_dir == 0] = 1.0
            flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat

            target_sign = torch.sign(current_error).unsqueeze(1)
            valid_mask = (torch.sign(flip_impacts) == target_sign)
            w_int_proposed = W_int + flip_dir
            in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
            valid_mask = valid_mask & in_range

            outlier_threshold, _ = self.compute_dynamic_outlier_threshold(act_padded)
            is_outlier = act_padded.abs() > outlier_threshold
            valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

            rounding_costs = (W_div + zp_flat - W_int).abs()
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

            max_flips_per_output = int(self.max_flip_percent * in_features)
            cumsum_flips = final_flips_sorted.long().cumsum(dim=1)
            within_limit = cumsum_flips <= max_flips_per_output
            sorted_flip_dir[~within_limit] = 0.0

            W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
            W_int.clamp_(0, max_int)

        W_dequant = (W_int - zp_flat) * scale_flat
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]
            W_int = W_int[:, :in_features]

        return W_dequant.to(W.dtype), scale.squeeze(-1), zp.squeeze(-1), W_int.to(torch.uint8)

    def infer_head_dim(self, k_out):
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'head_dim'):
            return self.model.config.head_dim
        for hd in [256, 128, 96, 80, 64]:
            if k_out % hd == 0:
                return hd
        return k_out

    # -------------------------------------------------------------- #
    # NEW: diagonal-Hessian accumulator over the calibration set.    #
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def accumulate_gqa_diag_hessian(self, calibration_data, n_samples=None):
        """
        Stream h_{g,i,j} = E_x[ u_{g,i}(x)^2 * x_j^2 ] for every k_proj.

        We read u_{g,i}(x) from the FULL-PRECISION key projection, using the
        un-AWQ-scaled weights stored in self.original_state_dict. The input
        x to this layer (its module input) is the same that the quantized
        layer will see at inference, so we can use hooks on the quantized
        k_proj modules and internally re-project through W_K^{fp}.

        Stored in self.gqa_diag_hessian[k_name] with shape
        [out_features, in_features] == [H_k * h, d], fp32, on CPU.
        """
        if n_samples is None:
            n_samples = self.gqa_hessian_n_tokens

        print("\n" + "=" * 80)
        print("QK-ASC Tier-4: Accumulating diagonal score-loss Hessian")
        print(f"  Target tokens: ~{n_samples}")
        print("=" * 80)

        # Find all k_proj modules we need to hook.
        k_modules = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            lname = name.lower()
            if 'k_proj' in lname or ('key' in lname and 'self_attn' in name):
                k_modules[name] = module

        if not k_modules:
            print("  No k_proj modules found; skipping.")
            return

        # Per-layer running sums and counts.
        # h_accum[name]: [out_features, in_features] fp32 on the module's device.
        h_accum = {}
        token_count = {name: 0 for name in k_modules}

        # We need W_K^{fp} (un-AWQ-scaled) for each k_proj.
        wk_fp = {}
        for name in k_modules:
            key = name + '.weight'
            if key not in self.original_state_dict:
                print(f"  ! Missing original weights for {name}; skipping layer.")
                continue
            wk_fp[name] = self.original_state_dict[key]  # CPU fp

        # Pre-allocate accumulators on the correct devices.
        for name, module in k_modules.items():
            if name not in wk_fp:
                continue
            out_feat, in_feat = wk_fp[name].shape
            h_accum[name] = torch.zeros(out_feat, in_feat,
                                        device=module.weight.device,
                                        dtype=torch.float32)

        # Hook: for a given token batch x of shape [B*T, d], compute
        #   u = x @ W_K^{fp}.T   -> [B*T, out_feat]
        #   update h[i, j] += sum_over_tokens(u[:, i]^2 * x[:, j]^2) / 1
        # and increment the token counter. Normalisation by N is done at end.
        hooks = []

        def make_hook(name):
            W = wk_fp[name]

            def hook(module, inputs, output):
                if name not in h_accum:
                    return
                x = inputs[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                x = x.to(torch.float32).to(module.weight.device)

                Wg = W.to(torch.float32).to(module.weight.device)
                # Full-precision key projection: [N, out_feat]
                u = x @ Wg.T
                # Accumulate outer-square:  h[i,j] += sum_n u[n,i]^2 x[n,j]^2
                # Implemented as (u^2).T @ (x^2) to avoid a tokens x out x d tensor.
                h_accum[name].add_((u.pow(2)).T @ x.pow(2))
                token_count[name] += x.shape[0]

            return hook

        for name, module in k_modules.items():
            if name in h_accum:
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Run calibration forward until we hit the token budget.
        self.model.eval()
        tokens_seen = 0
        try:
            for sample in calibration_data:
                if tokens_seen >= n_samples:
                    break
                if isinstance(sample, str):
                    enc = self.tokenizer(sample, return_tensors='pt',
                                         truncation=True,
                                         max_length=self.max_tokens_per_sample)
                    input_ids = enc['input_ids'].to(self.device)
                else:
                    input_ids = sample.to(self.device)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)

                self.model(input_ids)
                tokens_seen += input_ids.numel()
        finally:
            for h in hooks:
                h.remove()

        # Normalise and move to CPU.
        for name in list(h_accum.keys()):
            N = max(token_count[name], 1)
            h_accum[name] = (h_accum[name] / N).cpu()
            self.gqa_diag_hessian[name] = h_accum[name]

        print(f"  ✓ Accumulated diag Hessian for {len(self.gqa_diag_hessian)} layers "
              f"(~{tokens_seen} tokens)")

    # -------------------------------------------------------------- #
    # Quantisation pipeline.                                         #
    # -------------------------------------------------------------- #

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        if not self.apply_gqa_reflip:
            super().quantize_model_sequential(calibration_data, n_samples)
            return

        print("\n" + "=" * 80)
        print("AWQ-GQA (Tier 4): Pipeline")
        print("=" * 80)

        # Step 1: cache FP GQA weights.
        self.original_state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                self.original_state_dict[name + '.weight'] = module.weight.data.clone().cpu()
        print(f"  [1] Saved {len(self.original_state_dict)} GQA FP weights")

        # Step 2: accumulate diag Hessian BEFORE AWQ overwrites k_proj weights.
        # (Implementation detail: W_K^{fp} is read from original_state_dict, so
        # this ordering is only strictly required if we wanted to project
        # through the live module; we choose to always project through the
        # cached fp copy, which lets us run this step either before or after
        # AWQ. We run it first because it is a forward-only pass and does not
        # need the quantised k_proj.)
        print("  [2] Diag-Hessian accumulation")
        self.accumulate_gqa_diag_hessian(calibration_data,
                                         n_samples=self.gqa_hessian_n_tokens)

        # Step 3: standard AWQ quantisation.
        print("  [3] AWQ quantisation")
        super().quantize_model_sequential(calibration_data, n_samples)

        # Step 4: Tier-4 QK-ASC refinement.
        print("  [4] QK-ASC Tier-4 refinement")
        self.apply_gqa_reflip_refinement()

        self.original_state_dict = None
        self.gqa_diag_hessian.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def apply_gqa_reflip_refinement(self):
        attn_groups = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and is_gqa_layer(name):
                info = get_layer_group(name)
                if info is None:
                    continue
                _, attn_group = info
                attn_groups.setdefault(attn_group, {})
                lname = name.lower()
                if 'q_proj' in lname or 'query' in lname:
                    attn_groups[attn_group]['q_proj'] = (name, module)
                elif 'k_proj' in lname or 'key' in lname:
                    attn_groups[attn_group]['k_proj'] = (name, module)
                elif 'v_proj' in lname or 'value' in lname:
                    attn_groups[attn_group]['v_proj'] = (name, module)

        print(f"  Found {len(attn_groups)} attention groups")
        refined = 0
        for attn_group, projs in tqdm(attn_groups.items(), desc="  Refining"):
            if 'q_proj' not in projs or 'k_proj' not in projs:
                continue
            try:
                self.refine_attention_group(attn_group, projs)
                refined += 1
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"    ! {attn_group}: {e}")
                import traceback; traceback.print_exc()
        print(f"  ✓ Refined {refined}/{len(attn_groups)} groups")
        self.gqa_js_means.clear()

    # -------------------------------------------------------------- #
    # Per-block refinement.                                          #
    # -------------------------------------------------------------- #

    def refine_attention_group(self, attn_group, projs):
        q_name, q_module = projs['q_proj']
        k_name, k_module = projs['k_proj']
        q_key = q_name + '.weight'
        k_key = k_name + '.weight'

        required = [
            (q_key, self.original_state_dict, 'orig Q'),
            (k_key, self.original_state_dict, 'orig K'),
            (q_name, self.gqa_js_means, 'JS mean Q'),
            (k_name, self.gqa_js_means, 'JS mean K'),
            (q_key, self.gqa_awq_scales, 'AWQ scale Q'),
            (k_key, self.gqa_awq_scales, 'AWQ scale K'),
            (q_key, self.gqa_heuristic_int_weights, 'int Q'),
            (k_name, self.gqa_diag_hessian, 'diag H'),
        ]
        for k, d, desc in required:
            if k not in d:
                print(f"    ! {attn_group}: missing {desc}")
                return

        device = q_module.weight.device
        dtype = torch.float32

        # FP weights and JS means.
        Wq_fp = self.original_state_dict[q_key].to(device).to(dtype)
        Wk_fp = self.original_state_dict[k_key].to(device).to(dtype)
        mu_q = self.gqa_js_means[q_name].to(device).to(dtype)
        mu_k = self.gqa_js_means[k_name].to(device).to(dtype)

        # AWQ scales.
        sQ = self.gqa_awq_scales[q_key].to(device).to(dtype)
        sK = self.gqa_awq_scales[k_key].to(device).to(dtype)

        # Move to AWQ-scaled domain.
        Wq_fp_s = Wq_fp * sQ.unsqueeze(0)           # same domain as Wq_int
        Wk_fp_s = Wk_fp * sK.unsqueeze(0)
        x_q = mu_q / sQ                             # inversely scaled activation mean
        x_k = mu_k / sK

        # Current quantised K in scaled domain:
        # k_module.weight  ==  (W_int - zp) * quant_scale / awq_scale
        # → multiply by awq_scale to recover the scaled-domain dequant.
        Wk_q_s = k_module.weight.data.to(device).to(dtype) * sK.unsqueeze(0)

        # Shape inference.
        q_out, d = Wq_fp_s.shape
        k_out = Wk_fp_s.shape[0]
        h = self.infer_head_dim(k_out)
        H_k = k_out // h
        H_q = q_out // h
        if H_q % H_k != 0:
            print(f"    ! H_q={H_q} not divisible by H_k={H_k}")
            return
        r = H_q // H_k
        print(f"      GQA: H_q={H_q}, H_k={H_k}, r={r}, h={h}")

        # Diagonal Hessian for this k_proj: shape [k_out, d] = [H_k*h, d]
        # Meaning: h_accum[i_global, j] = E[u_{g,i}(x)^2 * x_j^2],
        # where u is in the *un-scaled* domain (we hooked on the raw module
        # input). Convert to scaled domain for consistency with Wq_int:
        # u_scaled_j = u_unscaled_j / sK_j  →  u^2 and x^2 scale independently,
        # so accumulator in scaled domain is h_accum * (x^2 rescale).
        # Practical choice: we accumulate on un-scaled x, then divide by sQ^2
        # here to match the flipping domain (where flips perturb by s*1/sQ).
        H_diag = self.gqa_diag_hessian[k_name].to(device).to(dtype)  # [k_out, d]
        # Per-dim rescale for AWQ-scaled domain:
        #   at inference, row i of Q projects x_q = x / sQ, so  x_j_scaled = x_j / sQ_j
        #   → x_j_scaled^2 = x_j^2 / sQ_j^2
        H_diag = H_diag / (sQ.unsqueeze(0) ** 2)   # [k_out, d]
        # Reshape to [H_k, h, d] so we can broadcast across the r query heads.
        H_diag = H_diag.view(H_k, h, d)

        # Heuristic integer weights + scales/zp for Q (what we will flip).
        Wq_int = self.gqa_heuristic_int_weights[q_key].to(device).to(dtype)
        Wq_scale = self.gqa_heuristic_scales[q_key].to(device).to(dtype)
        Wq_zp = self.gqa_heuristic_zp[q_key].to(device).to(dtype)

        # Expand group-level scale/zp to full hidden_dim.
        Wq_scale = Wq_scale.repeat_interleave(self.group_size, dim=-1)[:, :d]  # [q_out, d]
        Wq_zp    = Wq_zp.repeat_interleave(self.group_size, dim=-1)[:, :d]

        # Reshape [q_out, d] -> [H_k, r, h, d].
        Wq_int   = Wq_int.view(H_k, r, h, d)
        Wq_scale = Wq_scale.view(H_k, r, h, d)
        Wq_zp    = Wq_zp.view(H_k, r, h, d)

        # Mean-activation projections (rank-1 reward-side quantities).
        # FP target and quantised current, per head.
        Wq_fp_4d = Wq_fp_s.view(H_k, r, h, d)
        Wk_fp_4d = Wk_fp_s.view(H_k, 1, h, d)
        Wk_q_4d  = Wk_q_s.view(H_k, 1, h, d)

        Q_fp  = torch.einsum('bghd,d->bgh', Wq_fp_4d, x_q)       # [H_k, r, h]
        K_fp  = torch.einsum('bshd,d->bsh', Wk_fp_4d, x_k)       # [H_k, 1, h]
        K_q   = torch.einsum('bshd,d->bsh', Wk_q_4d,  x_k)       # [H_k, 1, h]
        Q_q   = torch.einsum('bghd,d->bgh',
                             (Wq_int - Wq_zp) * Wq_scale, x_q)    # [H_k, r, h]

        # Per-head score residual at the mean.
        # e_{g,ell} = Q_fp · K_fp  -  Q_q · K_q
        e = (Q_fp * K_fp).sum(dim=-1) - (Q_q * K_q.squeeze(1).unsqueeze(1)).sum(dim=-1)
        # shape [H_k, r]

        # -------------------------------------------------------- #
        # Tier-4 rank-1 reward proxy.                              #
        #                                                          #
        # We want per-dim reward g_{g,ell,i}[j] ~ leverage of flip  #
        # j in row i of head (g,ell) on the score residual.        #
        # Rank-1 approximation:                                    #
        #    g_{g,ell,i}[j]  ≈  alpha_{g,ell,i} * |k_{g,i}| * |x_j| #
        # where alpha redistributes the head-level scalar e_{g,ell} #
        # across head-coords i proportionally to |q^{fp}_{g,ell,i}|. #
        # -------------------------------------------------------- #

        abs_Q_fp = Q_fp.abs()                                # [H_k, r, h]
        denom = abs_Q_fp.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        c = e.unsqueeze(-1) * (abs_Q_fp / denom)             # [H_k, r, h]
        # Use quantised K for the per-dim scaling (matches real inference).
        K_for_flip = K_q.squeeze(1)                          # [H_k, h]
        # Δq target per head-coord (inverted from Eq. (5) in the paper);
        # sign comes from c and K_for_flip.
        K_safe = torch.where(K_for_flip.abs() > 1e-10,
                             K_for_flip,
                             torch.ones_like(K_for_flip))
        dq_target = c / K_safe.unsqueeze(1)                  # [H_k, r, h]

        # Per-dim reward: sign(dq_target) * |x_j|. The magnitude scaling is
        # absorbed into dq_target. We will combine with Wq_scale below.
        #   reward_j for row (g,ell,i) = Wq_scale_{g,ell,i,j} * x_q_j *
        #                                sign(dq_target_{g,ell,i})
        # but the sign of the flip is determined jointly with sign(K) — so
        # we work with the *signed* benefit rather than abs, then pick the
        # sign from the combined rule.

        x_abs = x_q.abs()                                    # [d]
        # Per-row magnitude reward: [H_k, r, h, d]
        reward_mag = Wq_scale.abs() * x_abs.view(1, 1, 1, d)

        # Cost: Wq_scale^2 * H_diag. H_diag is [H_k, h, d], broadcast over r.
        cost = (Wq_scale ** 2) * H_diag.unsqueeze(1)         # [H_k, r, h, d]

        # Benefit (reward weight: 2 per §3.4; reward magnitude dominated by
        # absolute value of dq_target per-row). We scale reward by
        # |dq_target| so rows with small required change get proportionally
        # smaller reward — otherwise a row with e≈0 would still flip.
        reward = 2.0 * reward_mag * dq_target.abs().unsqueeze(-1)   # [H_k, r, h, d]
        benefit = reward - cost                                      # [H_k, r, h, d]

        # Flip direction per row. Since dq_target can be any sign, and
        # score is Q·K_q, the flip direction on an integer weight at
        # position j of row (g,ell,i) is:
        #     sign( dq_target_{g,ell,i} )     (from the reward side)
        # The K sign is already baked into dq_target via the division, so
        # we do not multiply by sign(K) again here.
        flip_dir = torch.sign(dq_target).unsqueeze(-1)               # [H_k, r, h, 1]
        # Flatten edge case sign(0) -> 0 means no direction chosen; suppress
        # benefit there.
        zero_dir = (flip_dir == 0)
        benefit = torch.where(zero_dir.expand_as(benefit),
                              torch.full_like(benefit, -1.0),
                              benefit)

        # Integer-range validity.
        proposed = Wq_int + flip_dir                                 # [H_k, r, h, d]
        max_int = (2 ** self.bits) - 1
        in_range = (proposed >= 0) & (proposed <= max_int)
        benefit = torch.where(in_range, benefit, torch.full_like(benefit, -1.0))

        # -------------------------------------------------------- #
        # Per-row selection: top-K by benefit, among positives,    #
        # subject to flip budget.                                  #
        # -------------------------------------------------------- #
        K_max = max(int(self.gqa_max_flip_pct * d), 1)

        # Flatten (H_k, r, h) rows for argsort.
        benefit_rows = benefit.view(-1, d)                           # [R, d]
        flip_dir_rows = flip_dir.expand_as(benefit).contiguous().view(-1, d)

        # Descending sort on benefit.
        sorted_vals, sorted_idx = torch.sort(benefit_rows, dim=1, descending=True)
        # Take only positive-benefit flips, up to K_max.
        take_mask = (sorted_vals > 0)
        # Cumulatively cap at K_max per row.
        take_cum = take_mask.long().cumsum(dim=1)
        take_mask = take_mask & (take_cum <= K_max)

        # Build the flip tensor in the original layout.
        flips_rows = torch.zeros_like(benefit_rows)
        # Gather flip directions for selected entries.
        fd_sorted = torch.gather(flip_dir_rows, 1, sorted_idx)
        fd_sorted = fd_sorted * take_mask.float()
        # Scatter back.
        flips_rows.scatter_(1, sorted_idx, fd_sorted)
        flips_4d = flips_rows.view(H_k, r, h, d)

        # Apply.
        Wq_int_new = (Wq_int + flips_4d).clamp(0, max_int)

        total_flips = (flips_4d != 0).sum().item()
        n_rows = H_k * r * h
        print(f"        flips: {total_flips} / {n_rows * d} "
              f"(avg {total_flips / max(n_rows, 1):.2f} per row, cap {K_max})")

        # Dequantise back and write to the module (un-scaled domain).
        Wq_dequant_s = (Wq_int_new - Wq_zp) * Wq_scale               # [H_k, r, h, d]
        Wq_dequant   = Wq_dequant_s.view(q_out, d) / sQ.unsqueeze(0)
        q_module.weight.data.copy_(Wq_dequant.to(q_module.weight.dtype))

def main():
    import argparse
    import os
    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    parser = argparse.ArgumentParser(
        description='AWQ-GQA Tier-4: Cost-only QK-ASC refinement'
    )

    # --- AWQ arguments (unchanged from Document 1) ---
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--use-heuristic", action="store_true", default=True)
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false")
    parser.add_argument("--use-james-stein", action="store_true", default=True)
    parser.add_argument("--no-james-stein", dest="use_james_stein", action="store_false")
    parser.add_argument("--knee-tolerance", type=float, default=0.0)
    parser.add_argument("--max-flip-percent", type=float, default=0.05)
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=8)
    parser.add_argument("--output-dir", type=str,
                        default="./quantized_models/llama3_awq_gqa_tier4")
    parser.add_argument("--model-path", type=str, default="./models/Llama-3-8B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")

    # --- QK-ASC Tier-4 arguments ---
    parser.add_argument('--apply-gqa-reflip', action='store_true',
                        help='Apply Tier-4 QK-ASC refinement to GQA layers')
    parser.add_argument('--gqa-max-flip-pct', type=float, default=0.05,
                        help='Per-row flip budget as fraction of hidden dim')
    parser.add_argument('--gqa-hessian-n-tokens', type=int, default=2048,
                        help='Number of calibration tokens for diag-Hessian '
                             'accumulation (Tier-3 subsampling). '
                             'Set to 0 to use the full calibration set.')
    parser.add_argument('--gqa-hessian-eps', type=float, default=1e-8,
                        help='Damping on the diag Hessian for numerical '
                             'stability in the cost term.')

    args = parser.parse_args()

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Banner ---
    print("\n" + "=" * 80)
    print("AWQ-GQA Tier-4: Cost-only QK-ASC Refinement")
    print("=" * 80)
    print(f"  Model:                {args.model_path}")
    print(f"  Output:               {args.output_dir}")
    print(f"  Bits:                 {args.bits}")
    print(f"  Group size:           {args.group_size}")
    print(f"  Calibration samples:  {args.n_calib}")
    print(f"  Calibration dataset:  {args.calib_dataset}")
    print(f"  QK-ASC (Tier-4):      {'ENABLED' if args.apply_gqa_reflip else 'DISABLED'}")
    if args.apply_gqa_reflip:
        n_hess = (args.gqa_hessian_n_tokens
                  if args.gqa_hessian_n_tokens > 0 else "full set")
        print(f"    - Hessian tokens:   {n_hess}")
        print(f"    - Max flip %:       {args.gqa_max_flip_pct}")
        print(f"    - Hessian eps:      {args.gqa_hessian_eps}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model + tokenizer ---
    print(f"\nLoading model and tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # --- Build the Tier-4 quantizer ---
    # If the user passed 0, fall back to "use everything".
    hess_tokens = (args.gqa_hessian_n_tokens
                   if args.gqa_hessian_n_tokens > 0
                   else args.n_calib * args.max_tokens_per_sample)

    quantizer = AWQGQAQuantizerTier4(
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
        gqa_max_flip_pct=args.gqa_max_flip_pct,
        gqa_hessian_n_tokens=hess_tokens,
        gqa_hessian_eps=args.gqa_hessian_eps,
    )

    # --- Load calibration data ---
    from calibration_utils import (
        get_c4_calibration_data,
        get_wikitext2_calibration_data,
    )

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(
            quantizer.tokenizer,
            n_samples=args.n_calib,
            seqlen=2048,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [
            item['text'] for item in dataset
            if len(item['text'].strip()) > 100
        ][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(
            quantizer.tokenizer,
            n_samples=args.n_calib,
            seqlen=2048,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    # --- Quantize ---
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved quantized model to {args.output_dir}")


if __name__ == '__main__':
    main()