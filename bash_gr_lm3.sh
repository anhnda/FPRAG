#!/bin/bash

# Global Config
BASE_OUT="./quantized_models"
LOG_FILE="./quantization_grid_search2.log"

# Grid Search Parameters
KNEE_TOL_VALUES=(0.00)
MAX_FLIP_VALUES=(0.005)

# Mistral Hardcoded Baselines (Standard AWQ)
MISTRAL_WIKI_PPL=4.9373
MISTRAL_C4_PPL=7.7318

# Initialize log
echo "Comprehensive Grid Search Log - $(date)" > "$LOG_FILE"

# --- THE GRID SEARCH FUNCTION ---
run_full_grid_search() {
    local M_NAME=$1
    local M_PATH=$2
    local JS_OUT="${BASE_OUT}/${M_NAME}_gptq_js"
    local STD_OUT="${BASE_OUT}/${M_NAME}_gptq"

    echo "=================================================" | tee -a "$LOG_FILE"
    echo "STARTING MODEL: $M_NAME" | tee -a "$LOG_FILE"
    echo "=================================================" | tee -a "$LOG_FILE"

    # 1. Handle Baseline Generation
    if [ "$M_NAME" == "Mistral-7B-v0.3" ]; then
        echo "[SKIP] Using provided Standard AWQ results for Mistral baseline." | tee -a "$LOG_FILE"
        echo "Baseline Results (WikiText-2: $MISTRAL_WIKI_PPL, C4: $MISTRAL_C4_PPL)" >> "$LOG_FILE"
    else
        echo "[STEP] Generating Standard Baseline for $M_NAME..."
        rm -rf "$STD_OUT" && mkdir -p "$STD_OUT"
        python gptq_stand_xl.py \
            --model-path "$M_PATH" \
            --output-dir "$STD_OUT" \
            --percdamp 0.01 --asym --act-order --true-sequential
    fi

    # 2. Parameter Loops
    for knee in "${KNEE_TOL_VALUES[@]}"; do
        echo ">>> Testing Knee Tolerance: $knee" | tee -a "$LOG_FILE"
        
        for flip in "${MAX_FLIP_VALUES[@]}"; do
            echo "    [RUN] Max-Flip: $flip"
            
            # Wipe and recreate JS output dir to save space
            rm -rf "$JS_OUT" && mkdir -p "$JS_OUT"

            # Run Custom JS Quantization
            python gptq_js.py \
                --model-path "$M_PATH" \
                --output-dir "$JS_OUT" \
                --asym --act-order --smart-flip --true-sequential \
                --knee-tolerance "$knee" \
                --max-flip-percent "$flip"

            # Run Comparison and log results
            echo "Results for $M_NAME (K:$knee, F:$flip):" >> "$LOG_FILE"
            
            # If Mistral, we pass the baseline values as arguments or handle inside the script
            # Adjust flags for compare_slicing.py based on your specific implementation
            python compare_slicing.py \
                --heuristic-path "$JS_OUT" >> "$LOG_FILE" 2>&1
            
            echo "---------------------------------------" >> "$LOG_FILE"
        done
    done
    
    # Final cleanup
    rm -rf "$JS_OUT"
    [ "$M_NAME" != "Mistral-7B-v0.3" ] && rm -rf "$STD_OUT"
}

# --- EXECUTION LIST ---

# 1. Mistral-7B (Will skip standard baseline run)
# run_full_grid_search "Mistral-7B-v0.3" "/models/Mistral-7B-v0.3"
run_full_grid_search "Mistral-7B-v0.3" "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"

# 2. Qwen2.5-7B (Will generate new standard baseline)
#run_full_grid_search "Qwen2.5-7B" "/models/Qwen2.5-7B"

# 3. Llama-3-8B
#run_full_grid_search "Llama-3-8B" "/models/Llama-3-8B"

echo "================================================="
echo "ALL MODELS COMPLETE. Final results in: $LOG_FILE"