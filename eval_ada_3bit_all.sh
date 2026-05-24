#!/bin/bash
# AdaRound 3-bit eval: baseline + flipping + bias correction
set -euo pipefail

# --- Configuration ---
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
BASE_OUT="./eval_quantized_models"
RESULTS_DIR="./eval_ada_3bits"
LOG_FILE="./eval_ada_3bits.log"

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

# Initialize log
echo "AdaRound 3-bit Eval Run - $(date)" > "$LOG_FILE"
echo "n_calib=$N_CALIB, adaround_iters=$ADAROUND_ITERS, lr=$ADAROUND_LR, layer_batch=$LAYER_BATCH_SIZE" >> "$LOG_FILE"
echo "best_knee=$BEST_KNEE, best_flip=$BEST_FLIP, bits=3" >> "$LOG_FILE"
echo "tasks=$TASKS" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# =============================================================
# HELPER: evaluate one model
# Args: MODEL_NAME MODEL_PATH
# =============================================================
eval_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local BASELINE_OUT="${BASE_OUT}/${MODEL_NAME}_ar_baseline"
    local FLIP_OUT="${BASE_OUT}/${MODEL_NAME}_arf_best"
    local BC_OUT="${BASE_OUT}/${MODEL_NAME}_arbc"

    echo "=================================================" | tee -a "$LOG_FILE"
    echo "MODEL: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "Path:  $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "=================================================" | tee -a "$LOG_FILE"

    # ----------------------------------------------------------
    # 1. AdaRound Baseline (no flipping)
    # ----------------------------------------------------------
    echo "==> STEP 1: AdaRound baseline for $MODEL_NAME" | tee -a "$LOG_FILE"
    rm -rf "$BASELINE_OUT" && mkdir -p "$BASELINE_OUT"
    python adaround_xl.py \
        --model-path "$MODEL_PATH" --bits 3 \
        --output-dir "$BASELINE_OUT" \
        --n-calib "$N_CALIB" \
        --adaround-iters "$ADAROUND_ITERS" \
        --adaround-lr "$ADAROUND_LR" \
        --layer-batch-size "$LAYER_BATCH_SIZE" 2>&1 | tee -a "$LOG_FILE"

    echo "==> STEP 1b: PPL evaluation for $MODEL_NAME (AdaRound baseline)" | tee -a "$LOG_FILE"
    python compare_slicing.py \
        --heuristic-path "$BASELINE_OUT" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    echo "==> STEP 1c: Evaluating AdaRound baseline for $MODEL_NAME" | tee -a "$LOG_FILE"
    python -m lm_eval --model hf \
        --model_args pretrained="$BASELINE_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_adaround_baseline.json" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    rm -rf "$BASELINE_OUT"

    # ----------------------------------------------------------
    # 2. AdaRound + Flipping — best config
    # ----------------------------------------------------------
    echo "==> STEP 2: AdaRound+Flip (knee=${BEST_KNEE}, flip=${BEST_FLIP}) for $MODEL_NAME" | tee -a "$LOG_FILE"
    rm -rf "$FLIP_OUT" && mkdir -p "$FLIP_OUT"
    python adaround_flip_xl.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$FLIP_OUT" \
        --bits 3 \
        --n-calib "$N_CALIB" \
        --adaround-iters "$ADAROUND_ITERS" \
        --adaround-lr "$ADAROUND_LR" \
        --layer-batch-size "$LAYER_BATCH_SIZE" \
        --knee-tolerance "$BEST_KNEE" \
        --max-flip-percent "$BEST_FLIP" 2>&1 | tee -a "$LOG_FILE"

    echo "==> STEP 2b: PPL evaluation for $MODEL_NAME (AdaRound+Flip)" | tee -a "$LOG_FILE"
    python compare_slicing.py \
        --heuristic-path "$FLIP_OUT" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    echo "==> STEP 2c: Evaluating AdaRound+Flip for $MODEL_NAME" | tee -a "$LOG_FILE"
    python -m lm_eval --model hf \
        --model_args pretrained="$FLIP_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_adaround_flip_k${BEST_KNEE}_f${BEST_FLIP}.json" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    rm -rf "$FLIP_OUT"

    # ----------------------------------------------------------
    # 3. AdaRound + Bias Correction
    # ----------------------------------------------------------
    echo "==> STEP 3: AdaRound + Bias Correction for $MODEL_NAME" | tee -a "$LOG_FILE"
    rm -rf "$BC_OUT" && mkdir -p "$BC_OUT"
    python adaround_bc.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$BC_OUT" \
        --bits 3 \
        --n-calib "$N_CALIB" \
        --adaround-iters "$ADAROUND_ITERS" \
        --adaround-lr "$ADAROUND_LR" \
        --layer-batch-size "$LAYER_BATCH_SIZE" 2>&1 | tee -a "$LOG_FILE"

    echo "==> STEP 3b: PPL evaluation for $MODEL_NAME (AdaRound + BC)" | tee -a "$LOG_FILE"
    python compare_slicing.py \
        --heuristic-path "$BC_OUT" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    echo "==> STEP 3c: Evaluating AdaRound + Bias Correction for $MODEL_NAME" | tee -a "$LOG_FILE"
    python -m lm_eval --model hf \
        --model_args pretrained="$BC_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_adaround_bc.json" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    rm -rf "$BC_OUT"

    echo "==> DONE: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# =============================================================
# EXECUTION
# =============================================================
#eval_model "Llama-3-8B" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
eval_model "Mistral-7B-v0.3" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"
eval_model "Meta-Llama-3.1-8B" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
#eval_model "Qwen2.5-7B" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

echo "=================================================" | tee -a "$LOG_FILE"
echo "ALL EVALUATIONS COMPLETE." | tee -a "$LOG_FILE"
echo "Log:     $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"