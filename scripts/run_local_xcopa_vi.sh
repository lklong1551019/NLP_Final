#!/bin/bash
set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# === Configuration (override via environment variables) ===
PRED_MODEL="${PRED_MODEL:-qwen}"
XAI_MODEL="${XAI_MODEL:-deepseek}"
DATA="${DATA:-xcopa_vi}"
DEVICE="${DEVICE:-0}"
XAI_ITER="${XAI_ITER:-20}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-50}"
SAVE_PATH="${SAVE_PATH:-./results/local}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
TEMP_EXP="${TEMP_EXP:-0.9}"
XCOPA_LANG="${XCOPA_LANG:-vi}"
DATA_SPLIT="${DATA_SPLIT:-test}"

mkdir -p "$SAVE_PATH"

echo "============================================="
echo "  FaithLM Local Pipeline"
echo "============================================="
echo "  Predictor:  $PRED_MODEL"
echo "  Explainer:  $XAI_MODEL"
echo "  Dataset:    $DATA ($XCOPA_LANG / $DATA_SPLIT)"
echo "  Questions:  $START_IDX → $END_IDX"
echo "  Iterations: $XAI_ITER"
echo "  Save path:  $SAVE_PATH"
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
    --max_tokens "$MAX_TOKENS" \
    --temp_exp "$TEMP_EXP" \
    --xcopa_lang "$XCOPA_LANG" \
    --data_split "$DATA_SPLIT" \
    --load_in_4bit

echo "============ Done! Results in: $SAVE_PATH"
