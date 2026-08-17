#!/bin/bash
# Variant: Qwen3.5-4B (local, 4-bit) as predictor + Qwen via OpenRouter as explainer,
# on XCOPA Vietnamese.
#
#   bash scripts/run_qwen_openrouter_xcopa_vi.sh            # local + global
#   MODE=local  END_IDX=20 bash scripts/run_qwen_openrouter_xcopa_vi.sh
#   OPENROUTER_MODEL=qwen/qwen3.7-plus bash scripts/run_qwen_openrouter_xcopa_vi.sh
set -euo pipefail

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set. Put it in .env at the repo root." >&2
    exit 1
fi

MODE="${MODE:-both}"                 # local | global | both
PRED_MODEL="${PRED_MODEL:-qwen}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-qwen/qwen3.7-flash}"
DEVICE="${DEVICE:-0}"

# Local pipeline (Algorithm 1: refine the explanation text per instance)
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-200}"
XAI_ITER="${XAI_ITER:-15}"

# Global pipeline (Algorithm 2: refine the trigger prompt across instances)
GLOBAL_XAI_ITER="${GLOBAL_XAI_ITER:-3}"
ROUND_XAI_ITER="${ROUND_XAI_ITER:-10}"
QUES_SAMPLE="${QUES_SAMPLE:-15}"

# Paper settings: explainer temperature 0.9, top-p 0.9 (Table 2).
TEMP_EXP="${TEMP_EXP:-0.9}"
TOP_P="${TOP_P:-0.9}"
# Measured on XCOPA-vi with reasoning off: ~106 completion tokens per call.
# 500 leaves plenty of head-room without paying for runaway generations.
MAX_TOKENS="${MAX_TOKENS:-500}"
MAX_SPEND="${MAX_SPEND:-5.0}"

RESULTS_DIR="${RESULTS_DIR:-./results/experiments/xcopa_vi_${PRED_MODEL}_openrouter}"
USAGE_LOG="${USAGE_LOG:-${RESULTS_DIR}/usage.jsonl}"
mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "  FaithLM - XCOPA-vi / Qwen + OpenRouter"
echo "============================================="
echo "  Predictor:  $PRED_MODEL (local, 4-bit)"
echo "  Explainer:  $OPENROUTER_MODEL (OpenRouter)"
echo "  Reasoning:  OFF  (billed as output; would return empty completions)"
echo "  Budget cap: \$$MAX_SPEND     usage log: $USAGE_LOG"
echo "  Mode:       $MODE"
echo "============================================="

common_args=(
    --device_num "$DEVICE"
    --data xcopa_vi
    --xcopa_lang vi
    --pred_model "$PRED_MODEL"
    --xai_model openrouter
    --openrouter_model "$OPENROUTER_MODEL"
    --temp_exp "$TEMP_EXP"
    --top_p "$TOP_P"
    --max_tokens "$MAX_TOKENS"
    --max_spend "$MAX_SPEND"
    --usage_log "$USAGE_LOG"
)

if [ "$MODE" = "local" ] || [ "$MODE" = "both" ]; then
    echo ">>> LOCAL pipeline (split: test, questions ${START_IDX}..${END_IDX})"
    python main_local.py "${common_args[@]}" \
        --data_split test \
        --ques_idx_start "$START_IDX" \
        --ques_idx_end "$END_IDX" \
        --xai_iter "$XAI_ITER" \
        --save_file_path "$RESULTS_DIR"
fi

if [ "$MODE" = "global" ] || [ "$MODE" = "both" ]; then
    # validation (100 rows), NOT test: the trigger prompt is tuned here, and tuning
    # on the split you report on is prompt-fitting your own evaluation set.
    echo ">>> GLOBAL pipeline (split: validation)"
    python main_global.py "${common_args[@]}" \
        --data_split validation \
        --xai_iter "$GLOBAL_XAI_ITER" \
        --round_xai_iter "$ROUND_XAI_ITER" \
        --ques_sample "$QUES_SAMPLE" \
        --save_file "$RESULTS_DIR"
fi

echo "============ Done. Results in: $RESULTS_DIR"
if [ -f "$USAGE_LOG" ]; then
    python - "$USAGE_LOG" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if rows:
    cost = sum(r["cost"] for r in rows)
    reas = sum(r.get("reasoning_tokens") or 0 for r in rows)
    print(f"============ {len(rows)} calls, ${cost:.4f} charged, "
          f"{reas} reasoning tokens (should be 0)")
EOF
fi
