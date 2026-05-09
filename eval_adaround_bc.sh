#!/bin/bash
# AdaRound + Bias Correction: quantize -> PPL eval -> lm_eval
# Models: Llama-3-8B, Mistral-7B-v0.3, Qwen2.5-7B

set -euo pipefail

# --- Configuration ---
TASKS="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande"
BASE_OUT="./quantized_models"
RESULTS_DIR="./eval_results"
LOG_FILE="./adaround_bc.log"

mkdir -p "$BASE_OUT"
mkdir -p "$RESULTS_DIR"

# --- Shared AdaRound hyperparameters (match the grid-search config) ---
N_CALIB=128
ADAROUND_ITERS=10000
ADAROUND_LR=1e-3
LAYER_BATCH_SIZE=16

# Initialize log
echo "AdaRound + Bias Correction Run - $(date)" > "$LOG_FILE"
echo "n_calib=$N_CALIB, adaround_iters=$ADAROUND_ITERS, lr=$ADAROUND_LR, layer_batch=$LAYER_BATCH_SIZE" >> "$LOG_FILE"
echo "bias_correction=ON (default in adaround_bc.py)" >> "$LOG_FILE"
echo "tasks=$TASKS" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# =============================================================
# HELPER: quantize + evaluate one model
# Args: MODEL_NAME MODEL_PATH
# =============================================================
run_one_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local BC_OUT="${BASE_OUT}/${MODEL_NAME}_arbc"

    echo "=================================================" | tee -a "$LOG_FILE"
    echo "MODEL: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "Path:  $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "=================================================" | tee -a "$LOG_FILE"

    # ----------------------------------------------------------
    # 1. AdaRound + BC quantization
    # ----------------------------------------------------------
    echo "==> STEP 1: Quantizing $MODEL_NAME with AdaRound + Bias Correction" | tee -a "$LOG_FILE"
    rm -rf "$BC_OUT" && mkdir -p "$BC_OUT"

    python adaround_bc.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$BC_OUT" \
        --n-calib "$N_CALIB" \
        --adaround-iters "$ADAROUND_ITERS" \
        --adaround-lr "$ADAROUND_LR" \
        --layer-batch-size "$LAYER_BATCH_SIZE" 2>&1 | tee -a "$LOG_FILE"

    # ----------------------------------------------------------
    # 2. PPL evaluation via compare_slicing.py
    # ----------------------------------------------------------
    echo "==> STEP 2: PPL evaluation for $MODEL_NAME (AdaRound + BC)" | tee -a "$LOG_FILE"
    python compare_slicing.py \
        --heuristic-path "$BC_OUT" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    # ----------------------------------------------------------
    # 3. lm_eval on standard tasks
    # ----------------------------------------------------------
    echo "==> STEP 3: lm_eval for $MODEL_NAME (AdaRound + BC)" | tee -a "$LOG_FILE"
    python -m lm_eval --model hf \
        --model_args pretrained="$BC_OUT" \
        --tasks "$TASKS" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "${RESULTS_DIR}/${MODEL_NAME}_adaround_bc.json" 2>&1 | tee -a "$LOG_FILE"
    echo "---------------------------------------" >> "$LOG_FILE"

    # ----------------------------------------------------------
    # 4. Cleanup quantized model dir to save disk
    # ----------------------------------------------------------
    rm -rf "$BC_OUT"

    echo "==> DONE: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# =============================================================
# EXECUTION
# =============================================================
#run_one_model "Llama-3-8B" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

#run_one_model "Mistral-7B-v0.3" \
#    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"

run_one_model "Qwen2.5-7B" \
    "/home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

echo "================================================="
echo "ALL MODELS COMPLETE."
echo "Log:     $LOG_FILE"
echo "Results: $RESULTS_DIR"
echo "================================================="