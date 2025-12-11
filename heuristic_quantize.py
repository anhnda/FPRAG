
def quantize_groupwise_global_greedy(x, w, group_size=128, n_bits=4, outlier_percent=0.05):
    """
    1. Calculate scales per group.
    2. Quantize per group.
    3. Collect flip candidates GLOBALLY (across all groups).
    4. Sort globally by cost and correct the TOTAL error.
    """
    d = x.shape[0]
    assert d % group_size == 0, f"Dimension {d} not divisible by group_size {group_size}"
    num_groups = d // group_size

    # Reshape to (num_groups, group_size)
    w_groups = w.view(num_groups, group_size)
    x_groups = x.view(num_groups, group_size)

    # --- 1. Group-wise Setup ---
    q_max = 2 ** (n_bits - 1) - 1

    # Max per group (keepdims=True for broadcasting)
    max_vals = w_groups.abs().max(dim=1, keepdim=True).values
    max_vals[max_vals == 0] = 1e-9  # Avoid div/0
    scales = max_vals / q_max  # Shape: (num_groups, 1)

    # Broadcast scales to full shape
    scales_expanded = scales.repeat(1, group_size).view(d)  # Flatten back to D

    # --- 2. Initial Group-wise Rounding ---
    w_div = w / scales_expanded
    w_int = torch.round(w_div).clamp(-2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1)
    w_quant = w_int * scales_expanded

    # --- 3. Global Error Calculation ---
    # We look at the SINGLE scalar error of the dot product
    current_error = torch.dot(x, w - w_quant)

    if abs(current_error) < 1e-6: return w_quant

    # --- 4. Global Candidate Selection ---
    target_sign = torch.sign(current_error)

    # Directions
    flip_dir = torch.sign(w_div - w_int)
    flip_dir[flip_dir == 0] = 1.0

    # Impact: x * sign * specific_group_scale
    flip_impacts = x * flip_dir * scales_expanded

    # Outlier Mask (Global logic applied to whole vector)
    k_outliers = int(d * outlier_percent)
    outlier_mask = torch.zeros(d, dtype=torch.bool, device=x.device)
    if k_outliers > 0:
        outlier_mask[torch.topk(x.abs(), k_outliers).indices] = True

    # Validity Mask
    valid_mask = (~outlier_mask) & (torch.sign(flip_impacts) == target_sign)

    # Range check
    w_int_prop = w_int + flip_dir
    in_range = (w_int_prop >= -2 ** (n_bits - 1)) & (w_int_prop <= 2 ** (n_bits - 1) - 1)
    valid_mask = valid_mask & in_range

    if not valid_mask.any(): return w_quant

    # --- 5. Global Sorting & Optimization ---
    valid_indices = torch.nonzero(valid_mask).squeeze()
    if valid_indices.ndim == 0: valid_indices = valid_indices.unsqueeze(0)

    # Cost is distance to boundary
    candidate_costs = (w_div - w_int).abs()[valid_indices]

    # Sort ALL candidates from ALL groups together
    sorted_indices = torch.argsort(candidate_costs, descending=True)
    sorted_impacts = flip_impacts[valid_indices][sorted_indices]

    # Find best K
    cumsum = torch.cumsum(sorted_impacts, dim=0)
    best_k = torch.argmin(torch.abs(current_error - cumsum))

    if best_k == 0: return w_quant

    # Apply Flips
    indices_to_flip = valid_indices[sorted_indices][:best_k]
    w_int[indices_to_flip] += flip_dir[indices_to_flip].long()

    return w_int * scales_expanded


# --- RESULTS (Absolute Dot Product Error) ---
# 1. Non-Group Global Proposed:  0.004118
# 2. Group-Wise Nearest (Base):  0.001875
# 3. Group-Wise Proposed (New):  0.001396