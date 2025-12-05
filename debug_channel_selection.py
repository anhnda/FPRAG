"""
Debug script to check if there's a channel selection issue.
"""
import torch

def test_channel_selection():
    """Test that topk selects highest importance channels."""

    # Simulate importance scores with known pattern
    n_channels = 100
    importance = torch.randn(n_channels)

    # Keep top 50%
    keep_ratio = 0.5
    k = int(n_channels * keep_ratio)

    # Select top-k
    top_k_indices = torch.topk(importance, k).indices

    print("=" * 80)
    print("CHANNEL SELECTION TEST")
    print("=" * 80)
    print(f"Total channels: {n_channels}")
    print(f"Keep ratio: {keep_ratio}")
    print(f"Channels to keep: {k}")

    # Check if we're keeping the highest importance channels
    top_k_values = importance[top_k_indices]
    not_top_k_mask = torch.ones(n_channels, dtype=torch.bool)
    not_top_k_mask[top_k_indices] = False
    other_values = importance[not_top_k_mask]

    print(f"\nTop-{k} selected channels:")
    print(f"  Min importance: {top_k_values.min():.4f}")
    print(f"  Max importance: {top_k_values.max():.4f}")
    print(f"  Mean importance: {top_k_values.mean():.4f}")

    print(f"\nOther {n_channels - k} channels (to be quantized):")
    print(f"  Min importance: {other_values.min():.4f}")
    print(f"  Max importance: {other_values.max():.4f}")
    print(f"  Mean importance: {other_values.mean():.4f}")

    # Verify correctness
    if top_k_values.min() >= other_values.max():
        print("\n✅ CORRECT: All kept channels have higher importance than quantized channels")
    else:
        print("\n❌ ERROR: Some quantized channels have higher importance than kept channels!")

    # Show some example channel IDs
    sorted_indices = torch.argsort(importance, descending=True)
    print(f"\nTop 10 most important channels (by ID):")
    print(f"  {sorted_indices[:10].tolist()}")

    print(f"\nBottom 10 least important channels (by ID):")
    print(f"  {sorted_indices[-10:].tolist()}")

    print("\n" + "=" * 80)
    print("Key insight: Channel IDs can be ANY value (0-99).")
    print("What matters is importance score, not the channel ID!")
    print("=" * 80)


def test_praq_vs_awq_selection():
    """Test if PRAQ and AWQ select different channels."""

    print("\n\n" + "=" * 80)
    print("PRAQ vs AWQ CHANNEL SELECTION")
    print("=" * 80)

    n_channels = 100

    # Simulate AWQ importance (simple magnitude)
    awq_importance = torch.rand(n_channels) * 10

    # Simulate PRAQ importance (risk-aware, might rank channels differently)
    # Channels with high risk but medium magnitude get boosted
    praq_importance = torch.rand(n_channels) * 8  # Base utility
    risk_factor = torch.rand(n_channels) * 3  # Risk boost
    praq_importance = praq_importance + risk_factor

    k = 50  # Keep top 50%

    awq_top_k = set(torch.topk(awq_importance, k).indices.tolist())
    praq_top_k = set(torch.topk(praq_importance, k).indices.tolist())

    overlap = awq_top_k.intersection(praq_top_k)
    awq_only = awq_top_k - praq_top_k
    praq_only = praq_top_k - awq_top_k

    print(f"\nAWQ selects top-{k} channels: {len(awq_top_k)}")
    print(f"PRAQ selects top-{k} channels: {len(praq_top_k)}")
    print(f"Overlap: {len(overlap)} channels ({100*len(overlap)/k:.1f}%)")
    print(f"AWQ-only selections: {len(awq_only)} channels")
    print(f"PRAQ-only selections: {len(praq_only)} channels")

    if len(overlap) < k:
        print(f"\n⚠️  Methods disagree on {k - len(overlap)} channels!")
        print(f"   AWQ chooses these instead of PRAQ's: {sorted(list(awq_only))[:5]}")
        print(f"   PRAQ chooses these instead of AWQ's: {sorted(list(praq_only))[:5]}")
        print("\nThis is EXPECTED and shows PRAQ's risk-aware selection is different.")
    else:
        print("\n✅ Both methods select the same channels (unlikely in practice)")


if __name__ == "__main__":
    test_channel_selection()
    test_praq_vs_awq_selection()
