#!/bin/bash
# Sweep the fidelity score modes on one dataset, one predictor, one explainer.
# Everything except --score_mode is held fixed, so the runs are comparable.
#
#   DATASET=copa_en  bash scripts/run_metric_sweep.sh
#   DATASET=xcopa_vi END_IDX=50 MODES="accuracy logprob" bash scripts/run_metric_sweep.sh
set -euo pipefail
[ -f .env ] && { set -a; source .env; set +a; }

DATASET="${DATASET:-copa_en}"
PRED_MODEL="${PRED_MODEL:-phi}"          # local, logits available for the prob modes
XAI_MODEL="${XAI_MODEL:-openai}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-3.5-turbo}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-200}"
XAI_ITER="${XAI_ITER:-15}"
DATA_SPLIT="${DATA_SPLIT:-test}"
# Paper Table 2: explainer temperature 0.9, top-p 0.9.
TEMP_EXP="${TEMP_EXP:-0.9}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-300}"
STOP_THRESHOLD="${STOP_THRESHOLD:-0.5}"
STOP_PATIENCE="${STOP_PATIENCE:-4}"

# accuracy      = published metric, generate-and-parse. The baseline.
# prob_accuracy = same |Δacc| formula but from the choice argmax, no text parsing.
#                 Control: separates "continuous signal helped" from "dropping the parser helped".
# flip          = did the prediction change. Label-free, binary.
# logprob       = signed probability shift. Continuous.
# tv            = total variation. Non-negative divergence (identical to |logprob|
#                 for two choices; differs only on 5-choice ECQA).
MODES="${MODES:-accuracy prob_accuracy flip logprob tv}"

if [ "$XAI_MODEL" = "openai" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set (put it in .env)." >&2; exit 1
fi

ROOT="${ROOT:-./results/metric_sweep/${DATASET}_${PRED_MODEL}_${XAI_MODEL}}"
echo "============================================="
echo "  Metric sweep - $DATASET"
echo "  Predictor $PRED_MODEL (local) | Explainer $XAI_MODEL/$OPENAI_MODEL"
echo "  Questions $START_IDX..$END_IDX | max $XAI_ITER iterations"
echo "  Modes: $MODES"
echo "============================================="

for MODE in $MODES; do
    OUT="$ROOT/$MODE"; mkdir -p "$OUT"
    echo ""; echo ">>> score_mode=$MODE -> $OUT"
    python main_local.py \
        --data "$DATASET" --data_split "$DATA_SPLIT" \
        --pred_model "$PRED_MODEL" \
        --xai_model "$XAI_MODEL" --openai_model "$OPENAI_MODEL" \
        --score_mode "$MODE" \
        --ques_idx_start "$START_IDX" --ques_idx_end "$END_IDX" \
        --xai_iter "$XAI_ITER" \
        --stop_threshold "$STOP_THRESHOLD" --stop_patience "$STOP_PATIENCE" \
        --temp_exp "$TEMP_EXP" --top_p_exp "$TOP_P" --max_tokens "$MAX_TOKENS" \
        --save_file_path "$OUT" --metrics_log "$OUT/metrics.jsonl"
done

echo ""; echo "=== iterations to stop, per mode ==="
python - "$ROOT" <<'PY'
import json, os, sys, statistics as st
root = sys.argv[1]
print(f"{'mode':14s} {'câu':>5s} {'vòng/câu':>9s} {'điểm cuối':>10s}")
print("-" * 44)
for mode in sorted(os.listdir(root)):
    f = os.path.join(root, mode, "metrics.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        continue
    per = {}
    for r in rows:
        per.setdefault(r["question_idx"], []).append(r)
    iters = [len(v) for v in per.values()]
    final = [v[-1].get("optimised_score", 0.0) for v in per.values()]
    print(f"{mode:14s} {len(per):>5d} {st.mean(iters):>9.2f} {st.mean(final):>10.4f}")
PY
