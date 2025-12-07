#!/bin/bash

# Hyperparameter Tuning Script for Improved Hybrid PRAQ
# Tests different configurations to find optimal settings for C4 generalization

echo "=========================================="
echo "Hyperparameter Tuning for Hybrid PRAQ V2"
echo "=========================================="

# Default configuration (balanced)
echo ""
echo "1. Testing BALANCED configuration (β=0.5, τ=2.0)..."
python gwh_praq_v2.py \
    --n-calib 128 \
    --blend-beta 0.5 \
    --temp-tau 2.0 \
    --output-dir ./quantized_models/minicpm_gwh_v2_balanced

# AWQ-leaning (more robust to diverse data)
echo ""
echo "2. Testing AWQ-LEANING configuration (β=0.3, τ=2.5)..."
python gwh_praq_v2.py \
    --n-calib 128 \
    --blend-beta 0.3 \
    --temp-tau 2.5 \
    --output-dir ./quantized_models/minicpm_gwh_v2_awq_lean

# PRAQ-leaning (more intelligent optimization)
echo ""
echo "3. Testing PRAQ-LEANING configuration (β=0.7, τ=1.5)..."
python gwh_praq_v2.py \
    --n-calib 128 \
    --blend-beta 0.7 \
    --temp-tau 1.5 \
    --output-dir ./quantized_models/minicpm_gwh_v2_praq_lean

# Conservative (softest weighting, best generalization?)
echo ""
echo "4. Testing CONSERVATIVE configuration (β=0.4, τ=3.0)..."
python gwh_praq_v2.py \
    --n-calib 128 \
    --blend-beta 0.4 \
    --temp-tau 3.0 \
    --output-dir ./quantized_models/minicpm_gwh_v2_conservative

echo ""
echo "=========================================="
echo "Tuning Complete!"
echo "=========================================="
echo ""
echo "Configurations tested:"
echo "  1. Balanced:     β=0.5, τ=2.0 (equal AWQ+PRAQ, moderate weighting)"
echo "  2. AWQ-leaning:  β=0.3, τ=2.5 (more AWQ, softer weighting)"
echo "  3. PRAQ-leaning: β=0.7, τ=1.5 (more PRAQ, sharper weighting)"
echo "  4. Conservative: β=0.4, τ=3.0 (mostly AWQ, very soft weighting)"
echo ""
echo "Next: Evaluate all on C4 to find the best configuration"
echo ""
