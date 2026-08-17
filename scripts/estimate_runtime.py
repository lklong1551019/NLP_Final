#!/usr/bin/env python3
"""Measure real throughput on this machine and extrapolate a full run.

The team's "1 minute per iteration" figure came from the pre-refactor code,
which generated 256 tokens for every scoring call. With log-probability
scoring the predictor no longer generates during scoring at all, so the
bottleneck moved to the explainer API. Re-measure before committing to a
question count.

    python scripts/estimate_runtime.py --config configs/xcopa_vi_qwen_deepseek.yaml
    python scripts/estimate_runtime.py --config configs/... --probe 3 --iters 3
"""

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithlm import datasets as datasets_mod
from faithlm import explainers, predictors
from faithlm.config import load_config, load_dotenv_if_present
from faithlm.pipelines import run_local, select_indices


def fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate full-run wall time")
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe", type=int, default=3, help="questions to time")
    parser.add_argument("--iters", type=int, default=3, help="LLM-OPT iterations per probe question")
    parser.add_argument("--target_questions", type=int, default=200)
    parser.add_argument("--target_iters", type=int, default=10)
    args = parser.parse_args()

    load_dotenv_if_present()
    cfg = load_config(args.config)

    print(f"Config    : {args.config}")
    print(f"Variant   : {cfg.variant_id()}")
    print(f"Probe     : {args.probe} questions x {args.iters} iterations\n")

    examples = datasets_mod.load(cfg.dataset.name, cfg.dataset.lang, cfg.dataset.split)
    print(f"Dataset   : {len(examples)} examples\n")

    t0 = time.time()
    predictor = predictors.build(cfg.predictor)
    explainer = explainers.build(cfg.explainer)
    load_time = time.time() - t0
    print(f"Model load: {fmt(load_time)}\n")

    # Probe into a scratch directory so a timing run never pollutes real results.
    tmp = tempfile.mkdtemp(prefix="faithlm_probe_")
    cfg.run.output_dir = tmp
    cfg.run.resume = False
    cfg.run.xai_iter = args.iters
    cfg.run.ques_idx_start = 0
    cfg.run.ques_idx_end = args.probe

    started = time.time()
    records = run_local(cfg, examples, predictor, explainer)
    elapsed = time.time() - started
    shutil.rmtree(tmp, ignore_errors=True)

    timings = [r["elapsed_sec"] for r in records if "elapsed_sec" in r and not r.get("error")]
    failed = sum(1 for r in records if r.get("error"))
    if not timings:
        print("Every probe question failed — check the API key and GPU before estimating.")
        return

    per_question = statistics.mean(timings)
    per_iteration = per_question / max(1, args.iters)

    print(f"\n{'=' * 58}")
    print(f"Measured  : {fmt(per_question)}/question at {args.iters} iterations")
    print(f"          : {fmt(per_iteration)}/iteration")
    if len(timings) > 1:
        print(f"          : spread {fmt(min(timings))} - {fmt(max(timings))}")
    if failed:
        print(f"          : {failed} probe question(s) failed")
    print(f"{'=' * 58}\n")

    print(f"{'questions':>10} {'iters':>6} {'estimate':>12}   {'Kaggle 12h sessions':>20}")
    print("-" * 58)
    for n_q in (50, 100, 200, 500):
        for n_i in (5, 10, 15, 20):
            if (n_q, n_i) not in {(args.target_questions, args.target_iters),
                                  (50, 5), (100, 10), (200, 10), (200, 15), (500, 20)}:
                continue
            total = per_iteration * n_i * n_q
            sessions = total / (12 * 3600)
            marker = "  <-- team plan" if (n_q, n_i) == (args.target_questions, args.target_iters) else ""
            print(f"{n_q:>10} {n_i:>6} {fmt(total):>12}   {sessions:>19.1f}x{marker}")

    print(f"\nThe estimate scales linearly and ignores the one-off {fmt(load_time)} model load.")
    print("Explainer latency dominates, so a shared API key across four people")
    print("running at once will be slower than this probe suggests.")


if __name__ == "__main__":
    main()
