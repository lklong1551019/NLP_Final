#!/bin/bash
set -uo pipefail

# Reproduce the paper's COPA setting exactly (Table 2, COPA column):
#   Optimization steps      20
#   Temperature of Predictor 0.7   <- sampling, so FAITHLM_PRED_GREEDY is NOT set
#   Temperature of Explainer 0.9
#   Top-P of Explainer       0.9
#   Results averaged over 3 runs
#
# Each repetition writes to its own directory. They must not share one: the
# result filename does not encode the repetition, and RESUME=1 would make later
# repetitions skip everything the first one already produced.

DATA="${DATA:-copa_en}"
N_QUES="${N_QUES:-500}"
ITER="${ITER:-20}"
NSHARDS="${NSHARDS:-6}"
REPS="${REPS:-3}"
BASE="${BASE:-/mnt/project/productivity/anhnh/results/NLP_Final}"

echo "############################################################"
echo "#  FaithLM paper-exact reproduction"
echo "#  data=$DATA  n=$N_QUES  iter=$ITER  shards=$NSHARDS  reps=$REPS"
echo "#  predictor temp 0.7 (sampling) | explainer temp 0.9 top-p 0.9"
echo "############################################################"
date

for r in $(seq 1 "$REPS"); do
    OUT="$BASE/${DATA}_phi_gpt35_paper_rep${r}"
    echo ""
    echo "===== REPETITION $r / $REPS  ->  $OUT"
    TEMP_EXP=0.9 TOP_P_EXP=0.9 MAX_TOKENS=1000 RESUME=1 \
        bash scripts/run_sharded.sh "$DATA" phi litellm 0 "$N_QUES" "$ITER" "$NSHARDS" "$OUT"
    echo "===== repetition $r exit=$?"
done

date
echo "############ ALL REPETITIONS DONE"
