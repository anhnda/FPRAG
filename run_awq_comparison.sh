#!/bin/bash

# AutoAWQ vs Custom Implementation - Complete Comparison Pipeline
# This script automates the entire workflow from conversion to comparison

set -e  # Exit on error

echo "=============================================================================="
echo "AutoAWQ vs Custom Implementation - Complete Comparison Pipeline"
echo "=============================================================================="
echo ""

# Configuration
INPUT_MODEL="openbmb/MiniCPM-2B-sft-bf16"
SAFETENSORS_MODEL="./models/MiniCPM-2B-safetensors"
AUTOAWQ_OUTPUT="./quantized_models/minicpm_autoawq"
CUSTOM_OUTPUT="./quantized_models/minicpm_gw_awq_asym_l2"
CALIB_SAMPLES=128
N_GRID=20
GROUP_SIZE=128

# Step 1: Convert to safetensors (if not already done)
echo "Step 1: Checking if safetensors conversion is needed..."
echo "------------------------------------------------------------------------------"
if [ -f "$SAFETENSORS_MODEL/model.safetensors" ]; then
    echo "✅ Safetensors model already exists at $SAFETENSORS_MODEL"
    echo "   Skipping conversion."
else
    echo "Converting model to safetensors format..."
    echo "This will download and convert the model (~4.8GB)"
    echo ""
    python convert_to_safetensors.py \
        --input-model "$INPUT_MODEL" \
        --output-dir "$SAFETENSORS_MODEL"

    if [ $? -eq 0 ]; then
        echo "✅ Conversion successful!"
    else
        echo "❌ Conversion failed!"
        exit 1
    fi
fi
echo ""

# Step 2: Quantize with AutoAWQ
echo "Step 2: Quantizing with AutoAWQ library..."
echo "------------------------------------------------------------------------------"
if [ -f "$AUTOAWQ_OUTPUT/model.safetensors" ]; then
    echo "✅ AutoAWQ model already exists at $AUTOAWQ_OUTPUT"
    echo "   Skipping quantization."
else
    echo "Running AutoAWQ quantization..."
    echo "This will take 5-10 minutes depending on your GPU."
    echo ""
    python quantize_autoawq_library.py \
        --model-path "$SAFETENSORS_MODEL" \
        --output-dir "$AUTOAWQ_OUTPUT" \
        --calib-samples "$CALIB_SAMPLES" \
        --q-group-size "$GROUP_SIZE" \
        --zero-point

    if [ $? -eq 0 ]; then
        echo "✅ AutoAWQ quantization successful!"
    else
        echo "❌ AutoAWQ quantization failed!"
        exit 1
    fi
fi
echo ""

# Step 3: Quantize with custom implementation
echo "Step 3: Quantizing with custom gw_awq_asym_l2..."
echo "------------------------------------------------------------------------------"
if [ -f "$CUSTOM_OUTPUT/pytorch_model.bin" ] || [ -f "$CUSTOM_OUTPUT/model.safetensors" ]; then
    echo "✅ Custom model already exists at $CUSTOM_OUTPUT"
    echo "   Skipping quantization."
else
    echo "Running custom quantization..."
    echo "This will take 10-15 minutes depending on your GPU."
    echo ""
    python gw_awq_asym_l2.py \
        --n-calib "$CALIB_SAMPLES" \
        --n-grid "$N_GRID" \
        --group-size "$GROUP_SIZE" \
        --output-dir "$CUSTOM_OUTPUT"

    if [ $? -eq 0 ]; then
        echo "✅ Custom quantization successful!"
    else
        echo "❌ Custom quantization failed!"
        exit 1
    fi
fi
echo ""

# Step 4: Run comparison
echo "Step 4: Comparing both implementations..."
echo "------------------------------------------------------------------------------"
echo "This will evaluate perplexity on WikiText-2 validation set."
echo "Estimated time: 5-10 minutes"
echo ""
python compare_autoawq_vs_custom.py \
    --original-model "$INPUT_MODEL" \
    --autoawq-model "$AUTOAWQ_OUTPUT" \
    --custom-model "$CUSTOM_OUTPUT" \
    --n-samples 100 \
    --output-json ./comparison_results.json

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "PIPELINE COMPLETED SUCCESSFULLY!"
    echo "=============================================================================="
    echo ""
    echo "Results saved to: ./comparison_results.json"
    echo ""
    echo "Summary of what was created:"
    echo "  1. $SAFETENSORS_MODEL - Converted model in safetensors format"
    echo "  2. $AUTOAWQ_OUTPUT - AutoAWQ quantized model"
    echo "  3. $CUSTOM_OUTPUT - Custom quantized model"
    echo "  4. ./comparison_results.json - Detailed comparison metrics"
    echo ""
    echo "To view results:"
    echo "  cat comparison_results.json | python -m json.tool"
    echo ""
    echo "=============================================================================="
else
    echo "❌ Comparison failed!"
    exit 1
fi
