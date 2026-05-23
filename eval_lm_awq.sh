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

# Configs to test
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

    echo "================================================="
    echo "MODEL: $MODEL_NAME"
    echo "================================================="

    for CONFIG in "${CONFIGS[@]}"; do
        local KNEE="${CONFIG%%,*}"
        local FLIP="${CONFIG##*,}"
        local OUT_DIR="${BASE_OUT}/${MODEL_NAME}_arf_k${KNEE}_f${FLIP}"

        echo "==> AWQ+Flip (knee=${KNEE}, flip=${FLIP}) for $MODEL_NAME"
        mkdir -p "$OUT_DIR"

        python awq_js_xl.py \
            --model-path "$MODEL_PATH" \
            --output-dir "$OUT_DIR" \
            --n-calib "$N_CALIB" \
            --layer-batch-size "$LAYER_BATCH_SIZE" \
            --knee-tolerance "$KNEE" \
            --max-flip-percent "$FLIP"

        python compare_slicing.py --heuristic-path "$OUT_DIR"

        echo "==> Evaluating AWQ+Flip (knee=${KNEE}, flip=${FLIP}) for $MODEL_NAME"
        python -m lm_eval --model hf \
            --model_args pretrained="$OUT_DIR" \
            --tasks "$TASKS" \
            --device cuda:0 \
            --batch_size auto \
            --output_path "${RESULTS_DIR}/${MODEL_NAME}_awq_flip_k${KNEE}_f${FLIP}.json"

        rm -rf "$OUT_DIR"

        echo "==> Done: $MODEL_NAME | knee=${KNEE} flip=${FLIP}"
        echo ""
    done

    echo "==> ALL CONFIGS DONE: $MODEL_NAME"
    echo ""
}

# =============================================================
# EXECUTION
# =============================================================
#eval_model "Llama-3-8B" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

eval_model "Mistral-7B-v0.3" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"

# eval_model "Qwen2.5-7B" \
#     "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

echo "================================================="
echo "ALL EVALUATIONS COMPLETE. Results in: $RESULTS_DIR"