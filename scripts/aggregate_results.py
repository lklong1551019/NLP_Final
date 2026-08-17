#!/usr/bin/env python3
"""Aggregate results across experiment variants into a Markdown report.

    python scripts/aggregate_results.py --results_dir ./results --output ./docs/experiment_results.md

Reads the structured JSON the pipelines write, so unlike the previous version
it does not need `eval()` on model-generated text.
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List


def _read_json(path: str, quiet_if_missing: bool = False):
    if quiet_if_missing and not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] skipping {path}: {exc}")
        return None


def collect_local(variant_dir: str) -> Dict:
    records = []
    for path in sorted(glob.glob(os.path.join(variant_dir, "local", "sample-*.json"))):
        record = _read_json(path)
        if record and not record.get("error"):
            records.append(record)
    if not records:
        return {}

    scores = [r.get("best_score", 0.0) for r in records]
    correct = [bool(r.get("correct")) for r in records]
    improvements = [
        r["iterations"][-1]["score"] - r["iterations"][0]["score"]
        for r in records if len(r.get("iterations", [])) > 1
    ]
    return {
        "n": len(records),
        "accuracy": sum(correct) / len(records),
        "faith_mean": statistics.mean(scores),
        "faith_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "faith_max": max(scores),
        "zero_score_pct": sum(1 for s in scores if s == 0.0) / len(scores),
        "opt_gain": statistics.mean(improvements) if improvements else 0.0,
    }


def collect_global(variant_dir: str) -> Dict:
    summary = _read_json(
        os.path.join(variant_dir, "global", "summary.json"), quiet_if_missing=True
    )
    if not summary:
        return {}
    rounds = summary.get("rounds", [])
    scores = [r["score"] for r in rounds]
    return {
        "rounds": len(rounds),
        "best_score": summary.get("best_score", 0.0),
        "first_score": scores[0] if scores else 0.0,
        "final_score": scores[-1] if scores else 0.0,
        "best_instruction": summary.get("best_instruction", ""),
    }


def generate_report(results_dir: str, output_path: str) -> None:
    variants = sorted(
        d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d)
    )

    local_rows, global_rows = [], []
    for variant_dir in variants:
        name = os.path.basename(variant_dir)
        local = collect_local(variant_dir)
        if local:
            local_rows.append((name, local))
        glob_stats = collect_global(variant_dir)
        if glob_stats:
            global_rows.append((name, glob_stats))

    lines = ["# FaithLM - Experiment Results", ""]
    lines.append(f"> Source: `{results_dir}`")
    lines.append("")

    if local_rows:
        lines += [
            "## Local pipeline",
            "",
            "| Variant | N | Task acc. | Faithfulness (mean +/- sd) | Max | Zero-score | LLM-OPT gain |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, s in local_rows:
            lines.append(
                f"| `{name}` | {s['n']} | {s['accuracy']:.3f} | "
                f"{s['faith_mean']:.4f} +/- {s['faith_stdev']:.4f} | {s['faith_max']:.4f} | "
                f"{s['zero_score_pct']:.1%} | {s['opt_gain']:+.4f} |"
            )
        lines += [
            "",
            "*Zero-score* is the fraction of questions scoring exactly 0.0 - the "
            "failure mode that made single-question exact-match scoring uninformative.",
            "",
        ]

    if global_rows:
        lines += [
            "## Global pipeline",
            "",
            "| Variant | Rounds | First | Final | Best |",
            "|---|---|---|---|---|",
        ]
        for name, s in global_rows:
            lines.append(
                f"| `{name}` | {s['rounds']} | {s['first_score']:.4f} | "
                f"{s['final_score']:.4f} | {s['best_score']:.4f} |"
            )
        lines.append("")

        for name, s in global_rows:
            if s.get("best_instruction"):
                lines += [f"### Best instruction - `{name}`", "", "```",
                          s["best_instruction"][:1200], "```", ""]

    if not local_rows and not global_rows:
        lines += ["No results found. Run an experiment first:", "",
                  "```bash", "python run.py --config configs/xcopa_vi_qwen_deepseek.yaml", "```", ""]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report: {output_path}  ({len(local_rows)} local, {len(global_rows)} global)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate FaithLM results")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output", type=str, default="./docs/experiment_results.md")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)
