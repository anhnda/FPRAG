#!/bin/bash

set -euo pipefail

# =========================================================
# LM evaluation harness for Qwen2.5-7B
#
# Pipeline:
#   1. Evaluate the base (non-quantized) model
#   2. Run standard baseline quantization (gptq_stand_xl.py) and evaluate
#   3. Run GPTQ + bias correction (gptq_bc.py) and evaluate
#   4. Run GPTQ + smart flip (gptq_js.py) for two configs and evaluate
#
# Output JSONs land in $RESULTS_DIR.
# =========================================================

# --- Model -----------------------------------------------
MODEL_NAME="Qwen2.5-7B"
MODEL_PATH="/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

# --- Benchmarks ------------------------------------------
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"

# --- Paths -----------------------------------------------
BASE_OUT="./eval_quantized_models"
RESULTS_DIR="./eval_results_qwen"

mkdir -p "$BASE_OUT"
mkdir -p "$RESULTS_DIR"

echo ""
echo "#########################################################"
echo "# MODEL: $MODEL_NAME"
echo "# PATH : $MODEL_PATH"
echo "#########################################################"

# ---------------------------------------------------------
# 0. Evaluate Base (Non-Quantized) Model
# ---------------------------------------------------------
echo ""
echo "================================================="
echo "==> STEP 0: Evaluating Base (Non-Quantized) Model"
echo "================================================="
python -m lm_eval --model hf \
    --model_args pretrained="$MODEL_PATH" \
    --tasks "$TASKS" \
    --device cuda:0 \
    --batch_size auto \
    --output_path "${RESULTS_DIR}/${MODEL_NAME}_base.json"

# ---------------------------------------------------------
# 1. Standard Baseline Quantization + Evaluation
# ---------------------------------------------------------
BASELINE_OUT="${BASE_OUT}/${MODEL_NAME}_baseline"

echo ""
echo "================================================="
echo "==> STEP 1: Standard Baseline Quantization (gptq_stand_xl.py)"
echo "================================================="
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

# ---------------------------------------------------------
# 2. GPTQ + Bias Correction + Evaluation
# ---------------------------------------------------------
BC_OUT="${BASE_OUT}/${MODEL_NAME}_bc"

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

# ---------------------------------------------------------
# 3. GPTQ + Smart Flip (gptq_js.py) for selected configs
# ---------------------------------------------------------
# Config format: "KNEE:FLIP"
#   - GPTQ+Flip(K0.005, F0.02)
#   - GPTQ+Flip(K0.000, F0.005)
BEST_CONFIGS=("0.005:0.020" "0.000:0.005")

for config in "${BEST_CONFIGS[@]}"; do
    KNEE="${config%%:*}"
    FLIP="${config##*:}"

    CURR_OUT="${BASE_OUT}/${MODEL_NAME}_k${KNEE}_f${FLIP}"

    echo ""
    echo "================================================="
    echo "==> STEP 3: GPTQ + Flip  Knee=$KNEE  Flip=$FLIP"
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
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_quant_k${KNEE}_f${FLIP}.json"

    # C. Cleanup model folder to save disk space
    rm -rf "$CURR_OUT"
done

echo ""
echo "All benchmarks complete. Check $RESULTS_DIR for results."