#!/bin/bash
set -euo pipefail

# Sweep the EXPLAINER model over a fixed list, everything else held constant.
#
# The predictor, dataset, question range and iteration budget stay identical
# across runs, so the resulting per-model directories form a controlled
# comparison ("does a different explainer improve faithfulness on Vietnamese?").
#
# Each model gets its own OUTDIR. This matters: the default OUTDIR naming uses
# the --xai_model FLAG (always "litellm" here), not the actual model id, so
# without per-model dirs every sweep run would overwrite / falsely-resume the
# previous one.
#
# Usage:
#   bash scripts/run_explainer_sweep.sh [DATASET] [START] [END] [ITER] [NSHARDS]
#
#   XAI_MODELS="deepseek/deepseek-v4-pro google/gemini-3.5-flash" \
#       bash scripts/run_explainer_sweep.sh xcopa_vi 0 50 8 6
#
# Requires .env with LITELLM_API_KEY / LITELLM_BASE_URL / LITELLM_PRED_MODEL.
# Model ids must be written exactly as the gateway exposes them.

DATASET="${1:-xcopa_vi}"
START="${2:-0}"
END="${3:-50}"
ITER="${4:-8}"
NSHARDS="${5:-6}"

# Default sweep list: the team's standard explainer, a strong multilingual one,
# and a cheap English-centric one as contrast. Override with XAI_MODELS.
XAI_MODELS="${XAI_MODELS:-deepseek/deepseek-v4-pro google/gemini-3.5-flash openai/gpt-4o-mini}"

BASE_OUT="${BASE_OUT:-./results/experiments/${DATASET}_xai_sweep}"

echo "============================================="
echo "  Explainer sweep on $DATASET"
echo "  Range   : $START..$END  (iter=$ITER, shards=$NSHARDS)"
echo "  Pred    : ${LITELLM_PRED_MODEL:-<gateway default>} (fixed)"
echo "  XAI list: $XAI_MODELS"
echo "============================================="

fail=0
for model in $XAI_MODELS; do
    slug="$(echo "$model" | tr '/.:' '---')"
    outdir="$BASE_OUT/$slug"
    mkdir -p "$outdir"
    # Record what actually ran - the filenames inside only say "litellm".
    echo "$model" > "$outdir/XAI_MODEL.txt"

    echo ""
    echo ">>> explainer = $model  ->  $outdir"
    if ! LITELLM_XAI_MODEL="$model" bash scripts/run_sharded.sh \
            "$DATASET" litellm litellm "$START" "$END" "$ITER" "$NSHARDS" "$outdir"; then
        echo "!! sweep member failed: $model (continuing with the rest)"
        fail=1
    fi
done

echo ""
echo "============ Sweep done (fail=$fail). Compare with:"
echo "  .venv/bin/python scripts/build_report.py --results_dir $BASE_OUT --output ./docs/xai_sweep_report.md"
exit $fail
