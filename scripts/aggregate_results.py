#!/usr/bin/env python3
"""
Aggregate FaithLM experiment results into a markdown report.

Usage:
    python scripts/aggregate_results.py --results_dir ./results --output ./docs/experiment_results.md
"""

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime


def parse_local_result(filepath):
    """Parse a local pipeline result file."""
    filename = os.path.basename(filepath)
    # Format: local_{data}_{xai}_{pred}_iter-{N}_sample-{idx}.json
    parts = filename.replace(".json", "").split("_")

    scores = []
    target_line = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("============"):
                target_line = line
            elif line.startswith("{"):
                try:
                    entry = eval(line)  # Safe-ish since we wrote these files
                    if isinstance(entry.get("Score"), (int, float)):
                        scores.append(entry["Score"])
                except:
                    pass

    correct = "Corrct" in (target_line or "")
    return {
        "filename": filename,
        "scores": scores,
        "correct": correct,
        "target": target_line,
        "max_score": max(scores) if scores else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
    }


def parse_global_result(filepath):
    """Parse a global pipeline result file."""
    filename = os.path.basename(filepath)

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None

    scores = [entry["Score"] for entry in data if isinstance(entry.get("Score"), (int, float))]
    prompts = [entry.get("XAI prompt", "") for entry in data]

    return {
        "filename": filename,
        "scores": scores,
        "best_score": max(scores) if scores else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "best_prompt": prompts[-1] if prompts else "",
        "num_iterations": len(scores),
    }


def extract_variant_info(filename):
    """Extract variant info from filename."""
    name = filename.replace(".json", "")
    parts = name.split("_")

    pipeline = parts[0] if parts else "unknown"
    # Try to extract model names from the filename
    return {
        "pipeline": pipeline,
        "full_name": name,
    }


def generate_report(results_dir, output_path):
    """Generate markdown experiment report."""
    local_results = []
    global_results = []

    # Walk the whole tree. The previous version only looked at results/,
    # results/local and results/global, so anything written by
    # run_experiment.sh into results/experiments/<variant>/ was never picked up.
    for root, dirs, filenames in os.walk(results_dir):
        dirs[:] = [d for d in dirs if d != "logs"]
        subdir = os.path.relpath(root, results_dir)
        if subdir == ".":
            subdir = ""

        for filename in sorted(filenames):
            filepath = os.path.join(root, filename)
            if not os.path.isfile(filepath):
                continue

            if filename.startswith("local_"):
                result = parse_local_result(filepath)
                if result:
                    result["subdir"] = subdir
                    local_results.append(result)
            elif filename.startswith("global_") and filename.endswith(".json"):
                result = parse_global_result(filepath)
                if result:
                    result["subdir"] = subdir
                    global_results.append(result)

    # Generate markdown
    lines = []
    lines.append("# FaithLM Experiment Results")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Results directory: `{results_dir}`")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    total_local = len(local_results)
    total_global = len(global_results)
    lines.append(f"- **Local results**: {total_local} files")
    lines.append(f"- **Global results**: {total_global} files")
    lines.append("")

    # Local results table
    if local_results:
        lines.append("## Local Pipeline Results")
        lines.append("")

        # Group by variant (all fields except sample index)
        variant_groups = defaultdict(list)
        for r in local_results:
            # Group key: everything before _sample-
            key = r["filename"].rsplit("_sample-", 1)[0] if "_sample-" in r["filename"] else r["filename"]
            if r.get("subdir"):
                key = f"{r['subdir']}/{key}"
            variant_groups[key].append(r)

        lines.append("| Variant | Samples | Correct % | Avg Faith. Score | Max Faith. Score |")
        lines.append("|---------|---------|-----------|------------------|------------------|")

        for variant_key, results in sorted(variant_groups.items()):
            n = len(results)
            correct_pct = sum(1 for r in results if r["correct"]) / n * 100 if n > 0 else 0
            avg_score = sum(r["avg_score"] for r in results) / n if n > 0 else 0
            max_score = max(r["max_score"] for r in results) if results else 0
            lines.append(f"| `{variant_key}` | {n} | {correct_pct:.1f}% | {avg_score:.4f} | {max_score:.4f} |")

        lines.append("")

        # Detailed per-sample results
        lines.append("### Per-Sample Details")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand</summary>")
        lines.append("")
        lines.append("| File | Correct | Scores | Max |")
        lines.append("|------|---------|--------|-----|")
        for r in local_results:
            correct_str = "✅" if r["correct"] else "❌"
            scores_str = ", ".join(f"{s:.2f}" for s in r["scores"][:5])
            if len(r["scores"]) > 5:
                scores_str += "..."
            lines.append(f"| `{r['filename']}` | {correct_str} | {scores_str} | {r['max_score']:.4f} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Global results table
    if global_results:
        lines.append("## Global Pipeline Results")
        lines.append("")
        lines.append("| File | Iterations | Avg Score | Best Score |")
        lines.append("|------|-----------|-----------|------------|")
        for r in global_results:
            lines.append(f"| `{r['filename']}` | {r['num_iterations']} | {r['avg_score']:.4f} | {r['best_score']:.4f} |")
        lines.append("")

    # No results
    if not local_results and not global_results:
        lines.append("## No Results Found")
        lines.append("")
        lines.append("No experiment results found in the results directory. Run experiments first:")
        lines.append("")
        lines.append("```bash")
        lines.append("bash scripts/run_local_xcopa_vi.sh")
        lines.append("```")
        lines.append("")

    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to: {output_path}")
    print(f"  Local results:  {total_local}")
    print(f"  Global results: {total_global}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate FaithLM results")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output", type=str, default="./docs/experiment_results.md")
    args = parser.parse_args()

    generate_report(args.results_dir, args.output)
