#!/bin/bash
set -euo pipefail

# Run the LOCAL pipeline over a question range, split across N parallel shards.
#
# The local pipeline treats every question independently and writes one file per
# question (local_{data}_{xai}_{pred}_iter-{N}_sample-{idx}.json), so sharding by
# question index is safe - no shard can overwrite another shard's output.
#
# Usage: bash scripts/run_sharded.sh DATASET PRED XAI START END ITER NSHARDS OUTDIR

DATASET="${1:?dataset}"
PRED="${2:?pred_model}"
XAI="${3:?xai_model}"
START="${4:-0}"
END="${5:-100}"
ITER="${6:-10}"
NSHARDS="${7:-6}"
OUTDIR="${8:-./results/experiments/${DATASET}_${PRED}_${XAI}}"

PYTHON="${PYTHON:-.venv/bin/python}"
SPLIT="${DATA_SPLIT:-test}"

mkdir -p "$OUTDIR" "$OUTDIR/logs"

TOTAL=$(( END - START ))
CHUNK=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))

echo "============================================="
echo "  Sharded LOCAL run"
echo "  Dataset : $DATASET ($SPLIT)"
echo "  Models  : pred=$PRED xai=$XAI"
echo "  Range   : $START..$END  (iter=$ITER)"
echo "  Shards  : $NSHARDS x $CHUNK questions"
echo "  Output  : $OUTDIR"
echo "============================================="

pids=()
for (( s=0; s<NSHARDS; s++ )); do
    s_start=$(( START + s * CHUNK ))
    s_end=$(( s_start + CHUNK ))
    if (( s_start >= END )); then break; fi
    if (( s_end > END )); then s_end=$END; fi

    PYTHONUNBUFFERED=1 "$PYTHON" main_local.py \
        --data "$DATASET" \
        --pred_model "$PRED" \
        --xai_model "$XAI" \
        --data_split "$SPLIT" \
        --ques_idx_start "$s_start" \
        --ques_idx_end "$s_end" \
        --xai_iter "$ITER" \
        --save_file_path "$OUTDIR" \
        > "$OUTDIR/logs/shard_${s}.log" 2>&1 &
    # macOS ships bash 3.2, so no negative array subscripts here.
    shard_pid=$!
    pids[${#pids[@]}]=$shard_pid
    echo "  shard $s -> questions [$s_start, $s_end)  pid $shard_pid"
done

fail=0
if (( ${#pids[@]} == 0 )); then
    echo "  !! no shards launched"; exit 1
fi
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "  !! shard pid $pid exited non-zero"
        fail=1
    fi
done

echo "============ Local shards done (fail=$fail). Files: $(ls -1 "$OUTDIR"/local_*.json 2>/dev/null | wc -l | tr -d ' ')"
exit $fail
