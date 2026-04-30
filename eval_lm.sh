#!/bin/bash

set -euo pipefail

# --- Configuration ---
MODEL_PATH="/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"
MODEL_NAME="Mistral-7B-v0.3"

# Define the benchmarks you want to run
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"

BASE_OUT="./eval_quantized_models"
RESULTS_DIR="./eval_results"
BASELINE_OUT="${BASE_OUT}/${MODEL_NAME}_baseline"

mkdir -p "$BASE_OUT"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------
# 1. Run Standard Baseline Quantization + Evaluate
# ---------------------------------------------------------
echo "==> STEP 1: Running Standard Baseline Quantization (gptq_stand_xl.py)"

mkdir -p "$BASELINE_OUT"
python gptq_stand_xl.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$BASELINE_OUT" \
    --asym --act-order --true-sequential

echo "==> STEP 1b: Evaluating Standard Baseline"
python -m lm_eval --model hf \
    --model_args pretrained="$BASELINE_OUT" \
    --tasks "$TASKS" \
    --device cuda:0 \
    --batch_size auto \
    --output_path "${RESULTS_DIR}/baseline_standard.json"

rm -rf "$BASELINE_OUT"

# ---------------------------------------------------------
# 2. Evaluate Top 3 Quantized Configs (Best PPL Results)
# ---------------------------------------------------------

# Config format: "KNEE:FLIP"
BEST_CONFIGS=("0.000:0.010" "0.005:0.010" "-0.020:0.010")

for config in "${BEST_CONFIGS[@]}"; do
    KNEE="${config%%:*}"
    FLIP="${config##*:}"

    CURR_OUT="${BASE_OUT}/${MODEL_NAME}_k${KNEE}_f${FLIP}"

    echo "================================================="
    echo "==> PROCESSING: Knee=$KNEE, Flip=$FLIP"
    echo "================================================="

    # A. Run Quantization
    mkdir -p "$CURR_OUT"
    python gptq_js.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$CURR_OUT" \
        --asym --act-order --smart-flip --true-sequential \
        --knee-tolerance "$KNEE" \
        --max-flip-percent "$FLIP"

    # B. Run LM Evaluation
    echo "==> RUNNING LM_EVAL for K=$KNEE, F=$FLIP"
    python -m lm_eval --model hf \
        --model_args pretrained="$CURR_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/quant_k${KNEE}_f${FLIP}.json"

    # C. Cleanup model folder to save disk space
    rm -rf "$CURR_OUT"

done

echo "All benchmarks complete. Check $RESULTS_DIR for results."