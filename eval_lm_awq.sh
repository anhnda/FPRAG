#!/bin/bash
set -euo pipefail

# --- Configuration ---
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
BASE_OUT="./eval_quantized_models"
RESULTS_DIR="./eval_results_qwen25"

mkdir -p "$BASE_OUT"
mkdir -p "$RESULTS_DIR"

# --- Shared hyperparameters ---
N_CALIB=128
LAYER_BATCH_SIZE=16

# Flip configs to test
CONFIGS=(
    "-10,0.05"
    "0.03,1"
    "-10,1"
)


# =============================================================
# HELPER: evaluate one model
# Args: MODEL_NAME MODEL_PATH
# =============================================================
eval_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local AWQ_BASE_OUT="${BASE_OUT}/${MODEL_NAME}_awq_base"

    echo "================================================="
    echo "MODEL: $MODEL_NAME"
    echo "================================================="

    # ----------------------------------------------------------
    # STEP 0: Base model (no quantization)
    # ----------------------------------------------------------
    echo "==> STEP 0: Base model PPL + downstream — $MODEL_NAME"
    python compare_slicing.py --heuristic-path "$MODEL_PATH"
    python -m lm_eval --model hf \
        --model_args pretrained="$MODEL_PATH" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_base.json"

    # ----------------------------------------------------------
    # STEP 1: AWQ baseline (no flip)
    # ----------------------------------------------------------
    echo "==> STEP 1: AWQ baseline quantization — $MODEL_NAME"
    mkdir -p "$AWQ_BASE_OUT"
    python awq_stand_xl.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$AWQ_BASE_OUT" --bits 3\
        --n-calib "$N_CALIB" \
        --layer-batch-size "$LAYER_BATCH_SIZE"

    echo "==> STEP 1b: AWQ baseline PPL + downstream — $MODEL_NAME"
    python compare_slicing.py --heuristic-path "$AWQ_BASE_OUT"
    python -m lm_eval --model hf \
        --model_args pretrained="$AWQ_BASE_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_awq_base.json"

    rm -rf "$AWQ_BASE_OUT"

    # ----------------------------------------------------------
    # STEP 2: AWQ + Flip — each config
    # ----------------------------------------------------------
    for CONFIG in "${CONFIGS[@]}"; do
        local KNEE="${CONFIG%%,*}"
        local FLIP="${CONFIG##*,}"
        local FLIP_OUT="${BASE_OUT}/${MODEL_NAME}_arf_k${KNEE}_f${FLIP}"

        echo "==> STEP 2: AWQ+Flip (knee=${KNEE}, flip=${FLIP}) — $MODEL_NAME"
        mkdir -p "$FLIP_OUT"
        python awq_js_xl.py \
            --model-path "$MODEL_PATH" \
            --output-dir "$FLIP_OUT" --bits 3\
            --n-calib "$N_CALIB" \
            --layer-batch-size "$LAYER_BATCH_SIZE" \
            --knee-tolerance "$KNEE" \
            --max-flip-percent "$FLIP"

        echo "==> STEP 2b: AWQ+Flip PPL + downstream (knee=${KNEE}, flip=${FLIP}) — $MODEL_NAME"
        python compare_slicing.py --heuristic-path "$FLIP_OUT"
        python -m lm_eval --model hf \
            --model_args pretrained="$FLIP_OUT" \
            --tasks "$TASKS" \
            --device cuda:0 \
            --batch_size auto \
            --output_path "${RESULTS_DIR}/${MODEL_NAME}_awq_flip_k${KNEE}_f${FLIP}.json"

        rm -rf "$FLIP_OUT"

        echo "==> Done: $MODEL_NAME | knee=${KNEE} flip=${FLIP}"
        echo ""
    done

    echo "==> ALL CONFIGS DONE: $MODEL_NAME"
    echo ""
}

# =============================================================
# EXECUTION
# =============================================================
eval_model "Meta-Llama-3.1-8B" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
eval_model "Mistral-7B-v0.3" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"


echo "================================================="
echo "ALL EVALUATIONS COMPLETE. Results in: $RESULTS_DIR"