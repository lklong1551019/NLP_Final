#!/bin/bash
set -uo pipefail

# Full reproduction driver: paper's English COPA, then the Vietnamese XCOPA
# transfer. Local pipeline is sharded; global pipeline is single-process because
# it optimises one shared prompt and cannot be split.

PYTHON="${PYTHON:-.venv/bin/python}"
N_QUES="${N_QUES:-100}"
ITER="${ITER:-10}"
NSHARDS="${NSHARDS:-6}"
G_ITER="${G_ITER:-10}"
G_SAMPLE="${G_SAMPLE:-15}"

run_variant () {
    local data="$1" split="$2"
    local outdir="./results/experiments/${data}_litellm_litellm"
    mkdir -p "$outdir/logs"

    echo ""
    echo "#############################################"
    echo "#  VARIANT: $data  (split=$split)"
    echo "#############################################"

    DATA_SPLIT="$split" PYTHON="$PYTHON" \
        bash scripts/run_sharded.sh "$data" litellm litellm 0 "$N_QUES" "$ITER" "$NSHARDS" "$outdir"

    echo ">>> GLOBAL pipeline for $data"
    "$PYTHON" main_global.py \
        --data "$data" \
        --pred_model litellm \
        --xai_model litellm \
        --data_split "$split" \
        --xai_iter "$G_ITER" \
        --ques_sample "$G_SAMPLE" \
        --save_file "$outdir" \
        > "$outdir/logs/global.log" 2>&1
    echo ">>> global exit=$?"
}

date
run_variant copa_en test
run_variant xcopa_vi test
date
echo "============ ALL DONE"
