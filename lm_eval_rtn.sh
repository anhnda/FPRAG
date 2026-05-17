#!/bin/bash
set -euo pipefail

# --- Configuration ---
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
RESULTS_DIR="./eval_results_rtn"

mkdir -p "$RESULTS_DIR"

# =============================================================
# HELPER: evaluate a pre-quantized model
# Args: MODEL_NAME MODEL_PATH
# =============================================================
eval_quantized_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2

    echo "================================================="
    echo "STARTING EVALUATION: $MODEL_NAME"
    echo "PATH: $MODEL_PATH"
    echo "================================================="

    # Run lm-evaluation-harness
    python -m lm_eval --model hf \
        --model_args pretrained="$MODEL_PATH" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_results.json"

    echo "==> DONE: $MODEL_NAME"
    echo "-------------------------------------------------"
    echo ""
}

# =============================================================
# EXECUTION
# =============================================================

# 1. Evaluate the standard RTN model
eval_quantized_model "model_rtn" "./quantized_models/model_rtn"

# 2. Evaluate the RTN JS model
eval_quantized_model "model_rtn_js" "./quantized_models/model_rtn_js_xl"


echo "================================================="
echo "ALL EVALUATIONS COMPLETE. Results saved in: $RESULTS_DIR"