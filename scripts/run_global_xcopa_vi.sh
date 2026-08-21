#!/bin/bash
set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# === Configuration ===
PRED_MODEL="${PRED_MODEL:-qwen}"
XAI_MODEL="${XAI_MODEL:-deepseek}"
DATA="${DATA:-xcopa_vi}"
DEVICE="${DEVICE:-0}"
XAI_ITER="${XAI_ITER:-3}"
ROUND_XAI_ITER="${ROUND_XAI_ITER:-10}"
QUES_SAMPLE="${QUES_SAMPLE:-15}"
SAVE_PATH="${SAVE_PATH:-./results/global}"
XCOPA_LANG="${XCOPA_LANG:-vi}"
DATA_SPLIT="${DATA_SPLIT:-validation}"

mkdir -p "$SAVE_PATH"

echo "============================================="
echo "  FaithLM Global Pipeline"
echo "============================================="
echo "  Predictor:   $PRED_MODEL"
echo "  Explainer:   $XAI_MODEL"
echo "  Dataset:     $DATA ($XCOPA_LANG / $DATA_SPLIT)"
echo "  Iterations:  $XAI_ITER x $ROUND_XAI_ITER rounds"
echo "  Sample size: $QUES_SAMPLE questions/iter"
echo "============================================="

python main_global.py \
    --device_num $DEVICE \
    --data "$DATA" \
    --pred_model "$PRED_MODEL" \
    --xai_model "$XAI_MODEL" \
    --xai_iter "$XAI_ITER" \
    --round_xai_iter "$ROUND_XAI_ITER" \
    --ques_sample "$QUES_SAMPLE" \
    --save_file "$SAVE_PATH" \
    --max_tokens 1000 \
    --temp_exp 0.9 \
    --xcopa_lang "$XCOPA_LANG" \
    --data_split "$DATA_SPLIT" \
    --load_in_4bit

echo "============ Done! Results in: $SAVE_PATH"
