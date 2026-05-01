#!/bin/bash

set -euo pipefail

# =========================================================
# LM evaluation harness for GPTQ + Bias Correction
#
# For each model:
#   1. Run standard baseline quantization (gptq_stand_xl.py) and evaluate
#   2. Run GPTQ + bias correction (gptq_bc.py) and evaluate
#
# Output JSONs land in $RESULTS_DIR with model-prefixed names so the two
# models do not overwrite each other.
# =========================================================

# --- Models to run ---------------------------------------
# Each entry is "MODEL_NAME|MODEL_PATH". Add or remove freely.
MODELS=(
    "Llama-3-8B|/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
    "Mistral-7B-v0.3|/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"
)

# --- Benchmarks ------------------------------------------
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"

# --- Paths -----------------------------------------------
BASE_OUT="./eval_quantized_models"
RESULTS_DIR="./eval_results_gptq_bc"

mkdir -p "$BASE_OUT"
mkdir -p "$RESULTS_DIR"

# =========================================================
# Run pipeline for each model
# =========================================================
for entry in "${MODELS[@]}"; do
    MODEL_NAME="${entry%%|*}"
    MODEL_PATH="${entry##*|}"

    echo ""
    echo "#########################################################"
    echo "# MODEL: $MODEL_NAME"
    echo "# PATH : $MODEL_PATH"
    echo "#########################################################"

    BASELINE_OUT="${BASE_OUT}/${MODEL_NAME}_baseline"
    BC_OUT="${BASE_OUT}/${MODEL_NAME}_bc"

    # -----------------------------------------------------
    # 1. Standard Baseline Quantization + Evaluation
    # -----------------------------------------------------
    echo ""
    echo "==> STEP 1: Standard Baseline Quantization (gptq_stand_xl.py)"
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
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_baseline_standard.json"

    rm -rf "$BASELINE_OUT"

    # -----------------------------------------------------
    # 2. GPTQ + Bias Correction + Evaluation
    # -----------------------------------------------------
    echo ""
    echo "================================================="
    echo "==> STEP 2: GPTQ + Bias Correction (gptq_bc.py)"
    echo "================================================="

    mkdir -p "$BC_OUT"
    python gptq_bc.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$BC_OUT" \
        --asym --act-order --true-sequential \
        --bias-correction

    echo "==> RUNNING LM_EVAL for bias-corrected model"
    python -m lm_eval --model hf \
        --model_args pretrained="$BC_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_quant_bc.json"

    rm -rf "$BC_OUT"

done

echo ""
echo "All benchmarks complete. Check $RESULTS_DIR for results."