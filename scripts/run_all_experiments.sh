#!/bin/bash
# Run every experiment variant for the paper, then aggregate.
#
#   bash scripts/run_all_experiments.sh
#   N_QUESTIONS=200 XAI_ITER=15 bash scripts/run_all_experiments.sh
#
# Every variant resumes from disk, so re-running after an interruption picks up
# where it stopped rather than starting over.
set -euo pipefail

N_QUESTIONS="${N_QUESTIONS:-200}"
XAI_ITER="${XAI_ITER:-15}"

[ -f .env ] && { set -a; source .env; set +a; }

CONFIGS=(
    configs/xcopa_vi_qwen_deepseek.yaml    # main result
    configs/xcopa_vi_symmetric.yaml        # corrected metric
    configs/copa_en_qwen_deepseek.yaml     # cross-lingual control
    configs/baseline_negation.yaml         # non-LLM baseline
    configs/global_xcopa_vi.yaml           # global pipeline
)

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo " $config"
    echo "============================================================"
    # The global pipeline has no per-question range, so only pass it for local runs.
    if grep -q "pipeline: global" "$config"; then
        python run.py --config "$config"
    else
        python run.py --config "$config" --end "$N_QUESTIONS" --xai_iter "$XAI_ITER"
    fi
done

echo ""
echo "============================================================"
echo " Aggregating"
echo "============================================================"
python scripts/aggregate_results.py --results_dir ./results --output ./docs/experiment_results.md
echo "Report: docs/experiment_results.md"
