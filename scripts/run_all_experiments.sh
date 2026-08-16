#!/bin/bash
set -euo pipefail

# ============================================
# FaithLM — Run All Experiment Variants
# ============================================
# Each variant is: "PRED_MODEL XAI_MODEL DATA"
# Add or comment out variants as needed.

VARIANTS=(
    "qwen deepseek xcopa_vi"    # V1: Qwen3.5-4B + DeepSeek on Vietnamese XCOPA
    "qwen deepseek copa_en"     # V2: Qwen3.5-4B + DeepSeek on English COPA
    # "phi deepseek xcopa_vi"   # V3: Phi-2 + DeepSeek on Vietnamese XCOPA
    # "phi deepseek copa_en"    # V4: Phi-2 + DeepSeek on English COPA
)

# The original paper processed 500 instances for their main evaluation.
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-200}"
XAI_ITER="${XAI_ITER:-15}"

echo "============================================="
echo "  FaithLM — All Experiments (Full Benchmark)"
echo "  Questions: $START_IDX → $END_IDX"
echo "  Variants: ${#VARIANTS[@]}"
echo "============================================="

for variant in "${VARIANTS[@]}"; do
    read -r pred xai data <<< "$variant"
    echo ""
    echo ">>> Running LOCAL Pipeline: pred=$pred xai=$xai data=$data"
    echo "---"

    # Run the LOCAL pipeline
    PRED_MODEL=$pred \
    XAI_MODEL=$xai \
    DATA=$data \
    START_IDX=$START_IDX \
    END_IDX=$END_IDX \
    XAI_ITER=$XAI_ITER \
        bash scripts/run_local_xcopa_vi.sh

    echo ">>> Running GLOBAL Pipeline: pred=$pred xai=$xai data=$data"
    echo "---"

    # Run the GLOBAL pipeline
    PRED_MODEL=$pred \
    XAI_MODEL=$xai \
    DATA=$data \
        bash scripts/run_global_xcopa_vi.sh

    echo "<<< Finished Variant: pred=$pred xai=$xai data=$data"
done

echo ""
echo "============================================="
echo "  Aggregating Results"
echo "============================================="
python scripts/aggregate_results.py \
    --results_dir ./results \
    --output ./docs/experiment_results.md

echo "============ All experiments complete!"
echo "============ Report: docs/experiment_results.md"
