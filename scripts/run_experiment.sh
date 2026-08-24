#!/bin/bash
set -euo pipefail

# Usage: bash scripts/run_experiment.sh [DATASET] [PRED_MODEL] [XAI_MODEL]
# Example: bash scripts/run_experiment.sh xcopa_vi qwen deepseek
#          bash scripts/run_experiment.sh copa_en litellm litellm

DATASET=${1:-"xcopa_vi"}
PRED_MODEL=${2:-"qwen"}
XAI_MODEL=${3:-"deepseek"}

PYTHON="${PYTHON:-python}"

START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-200}"
XAI_ITER="${XAI_ITER:-15}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Create a unique output folder for this team member's experiment variant
TEAM_RESULTS_DIR="${TEAM_RESULTS_DIR:-./results/experiments/${DATASET}_${PRED_MODEL}_${XAI_MODEL}}"
mkdir -p "$TEAM_RESULTS_DIR"

echo "============================================================"
echo " Running Local Pipeline for ${DATASET}"
echo " Predictor: ${PRED_MODEL} | Explainer: ${XAI_MODEL}"
echo " Output Dir: ${TEAM_RESULTS_DIR}"
echo "============================================================"

"$PYTHON" main_local.py \
    --data "$DATASET" \
    --pred_model "$PRED_MODEL" \
    --xai_model "$XAI_MODEL" \
    --ques_idx_start "$START_IDX" \
    --ques_idx_end "$END_IDX" \
    --xai_iter "$XAI_ITER" \
    --save_file_path "$TEAM_RESULTS_DIR" \
    $EXTRA_ARGS

echo "============================================================"
echo " Running Global Pipeline for ${DATASET}"
echo "============================================================"

"$PYTHON" main_global.py \
    --data "$DATASET" \
    --pred_model "$PRED_MODEL" \
    --xai_model "$XAI_MODEL" \
    --xai_iter "$XAI_ITER" \
    --ques_sample "${QUES_SAMPLE:-15}" \
    --save_file "$TEAM_RESULTS_DIR" \
    $EXTRA_ARGS

echo "Experiment variant complete! Results saved in ${TEAM_RESULTS_DIR}"
