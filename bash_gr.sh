#!/bin/bash

# Configuration
MODEL_PATH="/models/Llama-3-8B"
OUTPUT_DIR="./quantized_models/Llama-3-8B_gptq_js"
STANDARD_PATH="./quantized_models/Llama-3-8B_gptq"
LOG_FILE="./grid_search_results.log"

# Parameter Arrays
KNEE_TOL_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)
MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)

# Clear previous log
echo "Grid Search Log - $(date)" > "$LOG_FILE"

# Outer Loop: Knee Tolerance
for knee in "${KNEE_TOL_VALUES[@]}"; do
    echo "===========================================" | tee -a "$LOG_FILE"
    echo "CATEGORY: Knee-Tolerance = $knee" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"

    # Inner Loop: Max-Flip Percent
    for flip in "${MAX_FLIP_VALUES[@]}"; do
        
        echo "[STARTING] Knee: $knee | Flip: $flip" | tee -a "$LOG_FILE"
        
        # Cleanup to save disk
        rm -rf "$OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
        
        # 1. Run Quantization
        python gptq_js.py \
            --model-path "$MODEL_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --asym --act-order \
            --smart-flip \
            --knee-tolerance "$knee" \
            --max-flip-percent "$flip" \
            --true-sequential \
            
        # 2. Run Comparison
        echo "Results for Knee $knee / Flip $flip:" >> "$LOG_FILE"
        python compare_slicing.py \
            --heuristic-path "$OUTPUT_DIR" \
            --standard-path "$STANDARD_PATH" >> "$LOG_FILE" 2>&1
            
        echo "Done. Model files purged." 
        echo "-------------------------------------------" >> "$LOG_FILE"
    done
done

echo "Search finished. Check $LOG_FILE for the comparison data."