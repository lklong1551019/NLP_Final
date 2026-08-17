#!/bin/bash
# Usage: bash scripts/run_experiment.sh [DATASET] [PRED_MODEL] [XAI_MODEL]
# Example: bash scripts/run_experiment.sh xcopa_vi qwen deepseek
set -euo pipefail

if [ -f .env ]; then
    set -a; source .env; set +a
fi

DATASET=${1:-"xcopa_vi"}
PRED_MODEL=${2:-"qwen"}
XAI_MODEL=${3:-"deepseek"}

START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-200}"
XAI_ITER="${XAI_ITER:-15}"
# Global pipeline has its own loop shape: xai_iter steps per round, ques_sample per step.
GLOBAL_XAI_ITER="${GLOBAL_XAI_ITER:-3}"
ROUND_XAI_ITER="${ROUND_XAI_ITER:-10}"
QUES_SAMPLE="${QUES_SAMPLE:-15}"
# Local evaluates on test; global tunes the trigger prompt, so it must not see test.
LOCAL_SPLIT="${LOCAL_SPLIT:-test}"
GLOBAL_SPLIT="${GLOBAL_SPLIT:-validation}"

TEAM_RESULTS_DIR="./results/experiments/${DATASET}_${PRED_MODEL}_${XAI_MODEL}"
mkdir -p "$TEAM_RESULTS_DIR"

echo "============================================================"
echo " Running Local Pipeline for ${DATASET}"
echo " Predictor: ${PRED_MODEL} | Explainer: ${XAI_MODEL}"
echo " Output Dir: ${TEAM_RESULTS_DIR}"
echo "============================================================"

python main_local.py \
    --data "$DATASET" \
    --pred_model "$PRED_MODEL" \
    --xai_model "$XAI_MODEL" \
    --data_split "$LOCAL_SPLIT" \
    --ques_idx_start "$START_IDX" \
    --ques_idx_end "$END_IDX" \
    --xai_iter "$XAI_ITER" \
    --save_file_path "$TEAM_RESULTS_DIR"

echo "============================================================"
echo " Running Global Pipeline for ${DATASET} (split: ${GLOBAL_SPLIT})"
echo "============================================================"

# main_global.py takes --save_file / --ques_sample / --round_xai_iter.
# It has no --ques_idx_start/--ques_idx_end/--save_file_path; passing those
# made argparse exit 2 before any work happened.
python main_global.py \
    --data "$DATASET" \
    --pred_model "$PRED_MODEL" \
    --xai_model "$XAI_MODEL" \
    --data_split "$GLOBAL_SPLIT" \
    --xai_iter "$GLOBAL_XAI_ITER" \
    --round_xai_iter "$ROUND_XAI_ITER" \
    --ques_sample "$QUES_SAMPLE" \
    --save_file "$TEAM_RESULTS_DIR"

echo "Experiment variant complete! Results saved in ${TEAM_RESULTS_DIR}"
