#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the log file name
LOG_FILE="grid_search.log"

# Redirect stdout (standard output) and stderr (errors) to both the console and the log file
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=========================================================="
    echo "Starting Grid Search. Logging to: $LOG_FILE"
echo "=========================================================="

# Define the grids
KNEE_TOLERANCES=("0.0" "0.01" "0.02")
MAX_FLIP_PERCENTS=("0.005" "0.01" "0.02" "0.05")

# Constant path
MODEL_PATH="/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

# Nested loops for the grid search
for kt in "${KNEE_TOLERANCES[@]}"; do
    for mfp in "${MAX_FLIP_PERCENTS[@]}"; do
        
        echo "=========================================================="
        echo "Running: --knee-tolerance $kt | --max-flip-percent $mfp"
        echo "Timestamp: $(date)"
        echo "=========================================================="
        
        # Dynamically label the output directory based on current parameters
        CURRENT_OUT_DIR="./quantized_models/model_rtn_js_kt_${kt}_mf_${mfp}"
        
        # 1. Run the quantization script
        python rtn_js_xl.py \
            --model-path "$MODEL_PATH" \
            --bits 4 \
            --group-size 128 \
            --knee-tolerance "$kt" \
            --max-flip-percent "$mfp" \
            --output-dir "$CURRENT_OUT_DIR"
            
        # 2. Run the comparison script
        python compare_slicing.py --heuristic-path "$CURRENT_OUT_DIR"
        
        echo "Finished combo: kt=$kt, mfp=$mfp"
        echo "----------------------------------------------------------"
        echo ""
        rm -rf "$CURRENT_OUT_DIR"  # Clean up the output directory after comparison
    done
done

echo "Grid search complete! All logs saved to $LOG_FILE"