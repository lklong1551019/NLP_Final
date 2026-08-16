#!/bin/bash
set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# === Configuration ===
PRED_MODEL="${PRED_MODEL:-qwen}"
XAI_MODEL="${XAI_MODEL:-deepseek}"
DATA="copa_en"
DEVICE="${DEVICE:-0}"
XAI_ITER="${XAI_ITER:-20}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-50}"
SAVE_PATH="${SAVE_PATH:-./results/local}"
DATA_SPLIT="${DATA_SPLIT:-train}"

mkdir -p "$SAVE_PATH"

echo "============================================="
echo "  FaithLM Local Pipeline — English COPA"
echo "============================================="
echo "  Predictor:  $PRED_MODEL"
echo "  Explainer:  $XAI_MODEL"
echo "  Dataset:    $DATA ($DATA_SPLIT)"
echo "  Questions:  $START_IDX → $END_IDX"
echo "============================================="

python main_local.py \
    --device_num $DEVICE \
    --data "$DATA" \
    --pred_model "$PRED_MODEL" \
    --xai_model "$XAI_MODEL" \
    --xai_iter "$XAI_ITER" \
    --ques_idx_start "$START_IDX" \
    --ques_idx_end "$END_IDX" \
    --save_file_path "$SAVE_PATH" \
    --max_tokens 1000 \
    --temp_exp 0.9 \
    --data_split "$DATA_SPLIT" \
    --load_in_4bit

echo "============ Done! Results in: $SAVE_PATH"
