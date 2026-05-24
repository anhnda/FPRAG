#!/bin/bash
set -euo pipefail

# --- Configuration ---
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
BASE_OUT="./eval_quantized_models"
RESULTS_DIR="./eval_ada_3bits"

mkdir -p "$BASE_OUT"
mkdir -p "$RESULTS_DIR"

# --- Shared AdaRound hyperparameters (from grid search) ---
N_CALIB=128
ADAROUND_ITERS=10000
ADAROUND_LR=1e-3
LAYER_BATCH_SIZE=16

# Best config from grid search (same for both models)
BEST_KNEE="-10"
BEST_FLIP="1"

# =============================================================
# HELPER: evaluate one model
# Args: MODEL_NAME MODEL_PATH
# =============================================================
eval_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local BASELINE_OUT="${BASE_OUT}/${MODEL_NAME}_ar_baseline"
    local FLIP_OUT="${BASE_OUT}/${MODEL_NAME}_arf_best"

    echo "================================================="
    echo "MODEL: $MODEL_NAME"
    echo "================================================="

    # ----------------------------------------------------------
    # 1. AdaRound Baseline (no flipping)
    # ----------------------------------------------------------
    echo "==> STEP 1: AdaRound baseline for $MODEL_NAME"
    mkdir -p "$BASELINE_OUT"
    python adaround_xl.py \
        --model-path "$MODEL_PATH" --bits 3 \
        --output-dir "$BASELINE_OUT" \
        --n-calib "$N_CALIB" \
        --adaround-iters "$ADAROUND_ITERS" \
        --adaround-lr "$ADAROUND_LR" \
        --layer-batch-size "$LAYER_BATCH_SIZE"

    echo "==> STEP 1b: Evaluating AdaRound baseline for $MODEL_NAME"
    python -m lm_eval --model hf \
        --model_args pretrained="$BASELINE_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_adaround_baseline.json"

    rm -rf "$BASELINE_OUT"

    # ----------------------------------------------------------
    # 2. AdaRound + Flipping — best config (knee=0.01, flip=0.05)
    # ----------------------------------------------------------
    echo "==> STEP 2: AdaRound+Flip (knee=${BEST_KNEE}, flip=${BEST_FLIP}) for $MODEL_NAME"
    mkdir -p "$FLIP_OUT"
    python adaround_flip_xl.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$FLIP_OUT" \
        --bits 3 \  
        --n-calib "$N_CALIB" \
        --adaround-iters "$ADAROUND_ITERS" \
        --adaround-lr "$ADAROUND_LR" \
        --layer-batch-size "$LAYER_BATCH_SIZE" \
        --knee-tolerance "$BEST_KNEE" \
        --max-flip-percent "$BEST_FLIP"

    echo "==> STEP 2b: Evaluating AdaRound+Flip for $MODEL_NAME"
    python -m lm_eval --model hf \
        --model_args pretrained="$FLIP_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_adaround_flip_k${BEST_KNEE}_f${BEST_FLIP}.json"

    rm -rf "$FLIP_OUT"

    echo "==> DONE: $MODEL_NAME"
    echo ""
}

# =============================================================
# EXECUTION
# =============================================================
#eval_model "Llama-3-8B" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
eval_model "Meta-Llama-3.1-8B" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
eval_model "Mistral-7B-v0.3" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"
#eval_model"Qwen2.5-7B" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

echo "================================================="
echo "ALL EVALUATIONS COMPLETE. Results in: $RESULTS_DIR"