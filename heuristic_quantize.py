import torch
import numpy as np
import matplotlib.pyplot as plt


def quantize_nearest_greedy(x, w, n_bits=4, outlier_percent=0.05):
    d = x.shape[0]

    # --- 0. Setup ---
    max_val = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scale = max_val / q_max

    # Identify Outliers
    k_outliers = int(d * outlier_percent)
    outlier_mask = torch.zeros(d, dtype=torch.bool, device=x.device)
    if k_outliers > 0:
        _, indices = torch.topk(x.abs(), k_outliers)
        outlier_mask[indices] = True

    # --- 1. Nearest Rounding (Baseline Init) ---
    w_div = w / scale
    w_int = torch.round(w_div)
    w_int = torch.clamp(w_int, -2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1)
    w_quant = w_int * scale

    # --- 2. Calculate Residual Error ---
    # current_error = True - Quantized
    current_error = torch.dot(x, w - w_quant)

    if abs(current_error) < 1e-6:
        return w_quant

    # --- 3. Vectorized Greedy Correction ---

    # FIX: We want Correction to match the sign of the Error
    # New_Error = Old_Error - Correction
    # If Error is +10, we need Correction to be +10.
    target_sign = torch.sign(current_error)

    # Calculate flip direction (Up or Down)
    # If w_div > w_int (rounded down), flip goes UP (+1)
    # If w_div < w_int (rounded up), flip goes DOWN (-1)
    flip_direction = torch.sign(w_div - w_int)
    flip_direction[flip_direction == 0] = 1.0  # Handle exact integers

    # Calculate Impact of flipping
    # Impact = X * Delta_W
    flip_impacts = x * flip_direction * scale

    # Filter:
    # 1. Not Outlier
    # 2. Impact must have SAME sign as error (to reduce it)
    valid_mask = (~outlier_mask) & (torch.sign(flip_impacts) == target_sign)

    # Ensure bits don't overflow
    w_int_proposed = w_int + flip_direction
    in_range = (w_int_proposed >= -2 ** (n_bits - 1)) & (w_int_proposed <= 2 ** (n_bits - 1) - 1)
    valid_mask = valid_mask & in_range

    if not valid_mask.any():
        return w_quant

    valid_indices = torch.nonzero(valid_mask).squeeze()
    if valid_indices.ndim == 0: valid_indices = valid_indices.unsqueeze(0)

    candidate_impacts = flip_impacts[valid_indices]

    # Sorting: Prioritize weights closest to the boundary (largest rounding error)
    rounding_error = (w_div - w_int).abs()
    candidate_costs = rounding_error[valid_indices]

    sorted_indices = torch.argsort(candidate_costs, descending=True)
    sorted_impacts = candidate_impacts[sorted_indices]

    # --- 4. Find Optimal k ---
    cumsum_impacts = torch.cumsum(sorted_impacts, dim=0)

    # We want (Error - Correction) -> 0
    # Correction is positive (matching error sign), so we subtract it.
    # Logic: minimize |Error - cumsum|
    # Since 'flip_impacts' has same sign as 'current_error',
    # we effectively look for |Abs(Error) - Abs(Cumsum)|

    # Mathematically: New_Error = current_error - cumsum_impacts
    residuals = torch.abs(current_error - cumsum_impacts)

    all_residuals = torch.cat([torch.abs(current_error).unsqueeze(0), residuals])
    best_k_idx = torch.argmin(all_residuals)

    if best_k_idx == 0:
        return w_quant

    indices_to_flip = valid_indices[sorted_indices][:best_k_idx]
    w_int[indices_to_flip] += flip_direction[indices_to_flip].long()

    return w_int * scale


# --- Simulation Runner ---
def run_simulation(n_trials=1000):
    torch.manual_seed(42)
    D = 1024
    errors_nearest = []
    errors_proposed = []

    print(f"Running {n_trials} Monte Carlo trials...")

    for _ in range(n_trials):
        x = torch.randn(D) * 0.1
        outlier_idx = torch.randperm(D)[:int(0.05 * D)]
        x[outlier_idx] = torch.randn(len(outlier_idx)) * 10.0
        w = torch.randn(D)

        # Baseline
        max_val = w.abs().max()
        scale = max_val / 7.0
        w_n = torch.round(w / scale) * scale
        err_n = torch.dot(x, w - w_n).item()
        errors_nearest.append(err_n)

        # Proposed
        w_p = quantize_nearest_greedy(x, w, n_bits=4, outlier_percent=0.05)
        err_p = torch.dot(x, w - w_p).item()
        errors_proposed.append(err_p)

    return np.array(errors_nearest), np.array(errors_proposed)


if __name__ == "__main__":
    err_n, err_p = run_simulation()
    print("\nRESULTS (Dot Product Error):")
    print(f"Nearest Rounding | MAE: {np.mean(np.abs(err_n)):.4f}")
    print(f"Proposed Method  | MAE: {np.mean(np.abs(err_p)):.4f}")
    print(f"Improvement:     {np.mean(np.abs(err_n)) / np.mean(np.abs(err_p)):.2f}x")

# Running 1000 Monte Carlo trials...
#
# RESULTS (Dot Product Error):
# Nearest Rounding | MAE: 8.0262
# Proposed Method  | MAE: 0.2729
# Improvement:     29.41x
