import torch
import torch.nn.functional as F


# ==========================================
# 1. Synthetic Data: "The Loud Silence"
# ==========================================
def generate_challenging_data(in_dim=128, out_dim=64, n_samples=2048, device="cpu"):
    """
    Generates a scenario where:
    1. Group A: Useful signals (small weights, active).
    2. Group B: Dangerous 'Dead' signals (HUGE weights, deeply negative bias).

    Why this is hard:
    - AWQ sees Group B's huge magnitude and keeps it (Accidentally correct).
    - Naive sparsity methods see Group B is 'Dead' and drop it (Fatal error).
    - Fast-R-PRAQ v3 must detect the 'Risk' of Group B and keep it.
    """
    X = torch.randn(n_samples, in_dim, device=device)
    W = torch.zeros(in_dim, out_dim, device=device)
    b = torch.zeros(out_dim, device=device)

    half = out_dim // 2

    # --- Group A: Useful (Active) ---
    # Moderate weights. These channels provide the actual accuracy.
    W[:, :half] = torch.randn(in_dim, half, device=device) * 0.5
    b[:half] = 0.5

    # --- Group B: The Trap (Saturated High-Energy) ---
    # Massive weights -> Massive Quantization Noise.
    # Deep negative bias -> Looks 'Dead' to standard algorithms.
    W[:, half:] = torch.randn(in_dim, half, device=device) * 3.0  # Huge weights
    b[half:] = -15.0  # Deeply negative (SiLU output is ~0)

    return X, W, b


# ==========================================
# 2. AWQ Strategy (Baseline)
# ==========================================
@torch.no_grad()
def get_awq_importance(X, W, b):
    """
    AWQ Metric: E[ |XW + b| ]
    Simply looks at output magnitude.
    """
    Z = X @ W + b
    return Z.abs().mean(dim=0)


# ==========================================
# 3. Fast-R-PRAQ v3 (The Solution)
# ==========================================
@torch.no_grad()
def fast_rpraq_v3_complete(X, W, b, beta=3.0, tau=-3.0, noise_factor=0.2, group_size=32):
    """
    Fast-R-PRAQ v3: Sensitivity & Hardware Aware.

    Args:
        noise_factor: Estimated noise relative to weight magnitude (e.g., 0.2 for Int4).
        group_size: Hardware block size (e.g., 32).
    """
    # --- Step A: Compute Signal Stats ---
    Z = X @ W + b
    z_mean = Z.mean(dim=0)
    z_std = Z.std(dim=0)
    z_upper = z_mean + 3 * z_std  # 3-sigma safety margin

    # --- Step B: Sensitivity Check (The v3 Fix) ---
    # Estimate how much quantization noise will be added to Z.
    # Noise ~ Mean(|X|) * Mean(|W|) * Scale
    x_mag = X.abs().mean()
    w_mag = W.abs().mean(dim=0)

    # If weights are huge, noise is huge.
    estimated_noise_impact = x_mag * w_mag * noise_factor

    # "Risk-Adjusted" Upper Bound.
    # We pretend the signal is higher by the amount of noise.
    # If (Signal + Noise) crosses the activation threshold, we MUST keep it.
    z_risk_upper = z_upper + estimated_noise_impact

    # --- Step C: Probability of Activation ---
    # tau is -3.0 for SiLU (where it starts activating)
    prob_active = torch.sigmoid(beta * (z_risk_upper - tau))

    # --- Step D: Utility Magnitude ---
    # Basic importance of the channel when it IS active
    magnitude = Z.abs().mean(dim=0) + z_std

    raw_importance = prob_active * magnitude

    # --- Step E: Hardware Grouping (v2 Logic) ---
    # Aggregate scores into blocks of 32
    C_out = raw_importance.shape[0]
    if group_size > 0 and C_out % group_size == 0:
        grouped = raw_importance.view(-1, group_size)
        # Sum importance over the block (if one neuron is critical, save the block)
        group_scores = grouped.sum(dim=1)
        # Broadcast back
        final_importance = group_scores.repeat_interleave(group_size)
    else:
        final_importance = raw_importance

    return final_importance


# ==========================================
# 4. Quantization Simulator
# ==========================================
def quantize_and_eval(X, W, b, importance_scores, keep_ratio=0.5):
    """
    Simulates Mixed-Precision Quantization:
    - Top K% channels -> FP16 (Clean)
    - Bottom K% channels -> Int4 (Noisy)
    """
    k = int(W.shape[1] * keep_ratio)

    # Select Top K
    # Note: Use stable sort or topk
    top_k_indices = torch.topk(importance_scores, k).indices
    mask_keep = torch.zeros(W.shape[1], dtype=torch.bool, device=W.device)
    mask_keep[top_k_indices] = True

    W_q = W.clone()

    # Simulation Parameters for Int4
    # Noise is proportional to the weight magnitude of that specific channel
    int4_noise_scale = 0.15

    for c in range(W.shape[1]):
        if not mask_keep[c]:
            # Apply Noise to "Dropped" channels
            # The noise is huge if W[:, c] is huge!
            w_range = W[:, c].abs().mean()
            noise = torch.randn_like(W[:, c]) * w_range * int4_noise_scale
            W_q[:, c] += noise

    # Forward Pass
    Y_q = F.silu(X @ W_q + b)
    return Y_q


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)  # Seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Fast-R-PRAQ v3 vs AWQ on {device}...")

    # 1. Generate The Trap
    X, W, b = generate_challenging_data(n_samples=4096, device=device)
    Y_gt = F.silu(X @ W + b)

    # 2. Run AWQ
    # AWQ sees huge weights in Group B -> Keeps Group B -> Drops Group A -> Error
    # (Actually, in this specific trap, AWQ keeps B, which prevents resurrection noise,
    # but might lose accuracy on A. Let's see who balances it better.)
    imp_awq = get_awq_importance(X, W, b)
    Y_awq = quantize_and_eval(X, W, b, imp_awq, keep_ratio=0.5)
    mse_awq = F.mse_loss(Y_awq, Y_gt).item()

    # 3. Run Fast-R-PRAQ v3
    # v3 sees Group B is 'dead' but 'risky' -> Keeps B.
    # v3 sees Group A is 'active' -> Keeps A.
    # It realizes BOTH are needed and tries to rank them smarter.
    imp_v3 = fast_rpraq_v3_complete(X, W, b, noise_factor=0.2, group_size=32)
    Y_v3 = quantize_and_eval(X, W, b, imp_v3, keep_ratio=0.5)
    mse_v3 = F.mse_loss(Y_v3, Y_gt).item()

    # 4. Results
    print("\n" + "=" * 40)
    print(f"AWQ MSE:             {mse_awq:.6f}")
    print(f"Fast-R-PRAQ v3 MSE:  {mse_v3:.6f}")
    print("=" * 40)

    if mse_v3 < mse_awq:
        print("\n✅ Fast-R-PRAQ v3 WON.")
        print("Analysis: v3 correctly balanced 'Utility' (Group A) and 'Risk' (Group B).")
    elif abs(mse_v3 - mse_awq) < 1e-5:
        print("\n🤝 TIE (Both prioritized the high-risk channels).")
    else:
        print("\n❌ AWQ WON.")