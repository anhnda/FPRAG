#!/bin/bash
# AdaRound Grid Search: Baseline (adaround_xl) vs AdaRound + Flipping (adaround_flip_xl)
# Models: Llama-3-8B, Mistral-7B-v0.3
# Sweeps knee-tolerance and max-flip-percent for the flipping variant.

# Global Config
BASE_OUT="./quantized_models"
LOG_FILE="./adaround_grid_search.log"

# Grid Search Parameters (for adaround_flip_xl.py)
KNEE_TOL_VALUES=(0.00 0.01)
MAX_FLIP_VALUES=(0.01 0.05)

# Shared AdaRound hyperparameters
N_CALIB=128
ADAROUND_ITERS=10000
ADAROUND_LR=1e-3
LAYER_BATCH_SIZE=16

# Initialize log
echo "AdaRound Grid Search Log - $(date)" > "$LOG_FILE"
echo "Sweep: knee_tol={${KNEE_TOL_VALUES[*]}}, max_flip={${MAX_FLIP_VALUES[*]}}" >> "$LOG_FILE"
echo "n_calib=$N_CALIB, adaround_iters=$ADAROUND_ITERS, lr=$ADAROUND_LR, layer_batch=$LAYER_BATCH_SIZE" >> "$LOG_FILE"

# --- THE GRID SEARCH FUNCTION ---
run_full_grid_search() {
    local M_NAME=$1
    local M_PATH=$2
    local AR_OUT="${BASE_OUT}/${M_NAME}_ar"      # AdaRound baseline (no flip)
    local ARF_OUT="${BASE_OUT}/${M_NAME}_arf"    # AdaRound + Flipping (per-config)

    echo "=================================================" | tee -a "$LOG_FILE"
    echo "STARTING MODEL: $M_NAME" | tee -a "$LOG_FILE"
    echo "Path: $M_PATH" | tee -a "$LOG_FILE"
    echo "=================================================" | tee -a "$LOG_FILE"

    # 1. Generate AdaRound Baseline (no flipping)
    echo "[STEP] Generating AdaRound baseline for $M_NAME..." | tee -a "$LOG_FILE"
    rm -rf "$AR_OUT" && mkdir -p "$AR_OUT"
    python adaround_xl.py \
        --model-path "$M_PATH" \
        --output-dir "$AR_OUT" \
        --n-calib "$N_CALIB" \
        --adaround-iters "$ADAROUND_ITERS" \
        --adaround-lr "$ADAROUND_LR" \
        --layer-batch-size "$LAYER_BATCH_SIZE"

    # Evaluate AdaRound baseline
    echo "[EVAL] AdaRound baseline for $M_NAME:" >> "$LOG_FILE"
    python compare_slicing.py \
        --heuristic-path "$AR_OUT" >> "$LOG_FILE" 2>&1
    echo "---------------------------------------" >> "$LOG_FILE"

    # 2. Grid Search over (knee_tolerance, max_flip_percent) for AdaRound + Flipping
    for knee in "${KNEE_TOL_VALUES[@]}"; do
        echo ">>> Testing Knee Tolerance: $knee" | tee -a "$LOG_FILE"

        for flip in "${MAX_FLIP_VALUES[@]}"; do
            echo "    [RUN] $M_NAME | knee=$knee | max_flip=$flip" | tee -a "$LOG_FILE"

            # Wipe and recreate flip output dir to save disk space
            rm -rf "$ARF_OUT" && mkdir -p "$ARF_OUT"

            # Run AdaRound + Flipping with current config
            python adaround_flip_xl.py \
                --model-path "$M_PATH" \
                --output-dir "$ARF_OUT" \
                --n-calib "$N_CALIB" \
                --adaround-iters "$ADAROUND_ITERS" \
                --adaround-lr "$ADAROUND_LR" \
                --layer-batch-size "$LAYER_BATCH_SIZE" \
                --knee-tolerance "$knee" \
                --max-flip-percent "$flip"

            # Log results
            echo "Results for $M_NAME (knee:$knee, max_flip:$flip):" >> "$LOG_FILE"
            python compare_slicing.py \
                --heuristic-path "$ARF_OUT" >> "$LOG_FILE" 2>&1
            echo "---------------------------------------" >> "$LOG_FILE"
        done
    done

    # Final cleanup
    rm -rf "$ARF_OUT"
    rm -rf "$AR_OUT"
}

# --- EXECUTION LIST ---
run_full_grid_search "Llama-3-8B" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

run_full_grid_search "Mistral-7B-v0.3" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"

run_full_grid_search "Qwen2.5-7B" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

echo "================================================="
echo "ALL MODELS COMPLETE. Final results in: $LOG_FILE"