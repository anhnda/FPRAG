#!/bin/bash
set -euo pipefail

# --- Configuration ---
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
BASE_OUT="./eval_quantized_models"
RESULTS_DIR="./eval_results_qwen25"

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
    # 2. AWQ + Flipping — best config (knee=0.01, flip=0.05)
    # ----------------------------------------------------------
    echo "==> STEP 2: AWQ+Flip (knee=${BEST_KNEE}, flip=${BEST_FLIP}) for $MODEL_NAME"
    mkdir -p "$FLIP_OUT"
    python awq_js_xl.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$FLIP_OUT" \
        --n-calib "$N_CALIB" \
        --layer-batch-size "$LAYER_BATCH_SIZE" \
        --knee-tolerance "$BEST_KNEE" \
        --max-flip-percent "$BEST_FLIP"
    python compare_slicing.py --heuristic-path "$FLIP_OUT" 
    
    echo "==> STEP 2b: Evaluating AWQ+Flip for $MODEL_NAME"
    python -m lm_eval --model hf \
        --model_args pretrained="$FLIP_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_awq_flip_k${BEST_KNEE}_f${BEST_FLIP}.json"

    rm -rf "$FLIP_OUT"

    echo "==> DONE: $MODEL_NAME"
    echo ""
}

# =============================================================
# EXECUTION
# =============================================================
#eval_model "Llama-3-8B" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

eval_model "Mistral-7B-v0.3" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"
# eval_model"Qwen2.5-7B" \
#     "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

echo "================================================="
echo "ALL EVALUATIONS COMPLETE. Results in: $RESULTS_DIR"