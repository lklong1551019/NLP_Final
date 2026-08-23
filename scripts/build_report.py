#!/usr/bin/env python3
"""Build the experiment report from a results tree.

Usage:
    python scripts/build_report.py --results_dir ./results/experiments \
        --output ./docs/experiment_report.md

Local pipeline files look like:
    ============ Corrct --> Q:[...] || GT-A:<gold> || LLM-A:<pred>
    {'Score': 1.0, 'XAI prompt': '<EXP>...'}
    {'Score': 0.0, 'XAI prompt': '<EXP>...'}

Global pipeline files are a JSON list of {"Score": ..., "XAI prompt": ...}.
"""

import argparse
import ast
import json
import os
import re
import statistics
from collections import defaultdict


def parse_local_file(path):
    target, scores, controls = None, [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("============"):
                target = line
            elif line.startswith("{"):
                try:
                    entry = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(entry.get("Score"), (int, float)):
                    scores.append(float(entry["Score"]))
                if isinstance(entry.get("ControlScore"), (int, float)):
                    controls.append(float(entry["ControlScore"]))

    if target is None:
        return None

    gold = pred = ""
    m = re.search(r"GT-A:(.*?) \|\| LLM-A:(.*)$", target)
    if m:
        gold, pred = m.group(1).strip(), m.group(2).strip()

    return {
        "file": os.path.basename(path),
        "correct": "Corrct" in target,
        "gold": gold,
        "pred": pred,
        "unparsed": pred == "X",
        "scores": scores,
        "iterations": len(scores),
        "controls": controls,
        "max_control": max(controls) if controls else None,
        "max_score": max(scores) if scores else 0.0,
        "mean_score": statistics.fmean(scores) if scores else 0.0,
        "any_flip": any(s > 0 for s in scores),
    }


def parse_global_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    scores = [float(e["Score"]) for e in data if isinstance(e.get("Score"), (int, float))]
    if not scores:
        return None
    return {
        "file": os.path.basename(path),
        "scores": scores,
        "iterations": len(scores),
        "best": max(scores),
        "first": scores[0],
        "last": scores[-1],
        "final_prompt": data[-1].get("XAI prompt", "") if data else "",
    }


def collect(results_dir):
    variants = defaultdict(lambda: {"local": [], "global": []})
    for root, dirs, files in os.walk(results_dir):
        dirs[:] = [d for d in dirs if d != "logs"]
        variant = os.path.relpath(root, results_dir)
        if variant == ".":
            variant = "(root)"
        for name in sorted(files):
            path = os.path.join(root, name)
            if name.startswith("local_") and name.endswith(".json"):
                parsed = parse_local_file(path)
                if parsed:
                    variants[variant]["local"].append(parsed)
            elif name.startswith("global_") and name.endswith(".json"):
                parsed = parse_global_file(path)
                if parsed:
                    variants[variant]["global"].append(parsed)
    return {k: v for k, v in variants.items() if v["local"] or v["global"]}


def pct(num, den):
    return f"{100.0 * num / den:.1f}%" if den else "n/a"


_STATS_RE = re.compile(
    r"LLM calls: (\d+) \| empty completions: (\d+) \| "
    r"failed calls: (\d+) \| fallback used: (\d+)"
)


def collect_api_stats(results_dir):
    """Sum the per-process API counters printed at the end of every run."""
    totals = {"procs": 0, "calls": 0, "empty": 0, "errors": 0, "fallback": 0}
    for root, _dirs, files in os.walk(results_dir):
        if os.path.basename(root) != "logs":
            continue
        for name in files:
            if not name.endswith(".log"):
                continue
            try:
                text = open(os.path.join(root, name), encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for m in _STATS_RE.finditer(text):
                totals["procs"] += 1
                totals["calls"] += int(m.group(1))
                totals["empty"] += int(m.group(2))
                totals["errors"] += int(m.group(3))
                totals["fallback"] += int(m.group(4))
    return totals


def build(results_dir, output_path):
    variants = collect(results_dir)
    if not variants:
        raise SystemExit(f"No results found under {results_dir}")

    out = []
    out.append("# FaithLM — Experiment Report")
    out.append("")
    out.append(f"> Results directory: `{results_dir}`")
    out.append("")

    # ---- setup ---------------------------------------------------------
    out.append("## 0. Experimental setup")
    out.append("")
    out.append("Both FaithLM roles were served by an OpenAI-compatible LiteLLM gateway; no")
    out.append("local model weights were used. See `docs/changelog_2026-08-17.md` §E.")
    out.append("")
    out.append("| Role | Model | Temperature | max_tokens |")
    out.append("|---|---|---|---|")
    out.append(f"| Predictor — initial answer | `{os.environ.get('LITELLM_PRED_MODEL', 'n/a')}` | 0.0 | 200 |")
    out.append(f"| Predictor — scoring | `{os.environ.get('LITELLM_PRED_MODEL', 'n/a')}` | 0.01 | 200 |")
    out.append(f"| Explainer | `{os.environ.get('LITELLM_XAI_MODEL', 'n/a')}` | `--temp_exp` | 1000 |")
    out.append(f"| Fallback (empty completions) | `{os.environ.get('LITELLM_FALLBACK_MODEL', 'n/a')}` | — | — |")
    out.append("")
    out.append("### Deviations from the paper's Table 2")
    out.append("")
    out.append("These are material and must be quoted alongside any number in this report.")
    out.append("")
    out.append("| Hyper-parameter | Paper (COPA) | This run |")
    out.append("|---|---|---|")
    out.append("| Fidelity optimisation steps (Alg. 1) | 20 | 8 |")
    out.append("| Predictor temperature | 0.7 | 0.0 / 0.01 |")
    out.append("| Explainer temperature | 0.9 | 0.01 (local) / 0.9 (global) |")
    out.append("| Explainer top-p | 0.9 | not set (1.0) |")
    out.append("| Trigger-prompt steps (Alg. 2) | 100 | 8 |")
    out.append("| Sampled instances per step | 30 (Table 2) / 15 (§4.3) | 12 |")
    out.append("| Repetitions | 3 runs averaged, grid search | 1 |")
    out.append("| Instances evaluated | 500 | 100 (indices 0–99, **not** a random sample) |")
    out.append("| Target model | Vicuna-7B, Phi-2 | API model (see above) |")
    out.append("| Explainer | GPT-3.5-Turbo, Claude-2 | API model (see above) |")
    out.append("| Baselines | SelfExp, Self-consistency | **not implemented** |")
    out.append("| Truthfulness metric | GPT-4o + RoBERTa-L + XLNet-L | **not implemented** |")
    out.append("")

    # ---- api stats -----------------------------------------------------
    api = collect_api_stats(results_dir)
    if api["calls"]:
        out.append("### API call statistics")
        out.append("")
        out.append(f"- Processes reporting: {api['procs']}")
        out.append(f"- Total LLM calls: **{api['calls']}**")
        out.append(f"- Calls that failed outright: {api['errors']}")
        out.append(f"- Calls the primary model left empty and that were retried away: {api['empty']}")
        out.append(f"- Calls served by the **fallback** model: **{api['fallback']}** "
                   f"({pct(api['fallback'], api['calls'])})")
        out.append("")
        if api["fallback"]:
            out.append("> The fallback share is not incidental: a non-trivial fraction of the")
            out.append("> Explainer's output came from a different model than the one named above,")
            out.append("> because the primary returned an empty body for prompts its content filter")
            out.append("> rejected. Vietnamese prompts triggered this far more often than English ones.")
            out.append("")

    # ---- summary table -------------------------------------------------
    out.append("## 1. Summary")
    out.append("")
    out.append("| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Random-hint control | Corrected | Questions with any flip | Mean iterations |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for variant in sorted(variants):
        loc = variants[variant]["local"]
        if not loc:
            continue
        n = len(loc)
        acc = sum(r["correct"] for r in loc)
        unp = sum(r["unparsed"] for r in loc)
        flips = sum(r["any_flip"] for r in loc)
        mean_max = statistics.fmean(r["max_score"] for r in loc)
        mean_it = statistics.fmean(r["iterations"] for r in loc)
        ctl = [r["max_control"] for r in loc if r["max_control"] is not None]
        if ctl:
            mean_ctl = statistics.fmean(ctl)
            ctl_txt = f"{mean_ctl:.3f}"
            corr_txt = f"**{mean_max - mean_ctl:+.3f}**"
        else:
            ctl_txt = corr_txt = "not measured"
        out.append(
            f"| `{variant}` | {n} | {pct(acc, n)} | {pct(unp, n)} | "
            f"{mean_max:.3f} | {ctl_txt} | {corr_txt} | {pct(flips, n)} | {mean_it:.2f} |"
        )
    out.append("")
    out.append("**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation "
               "− accuracy with the counterfactual explanation|, per question, over a single "
               "instance. It is therefore 0 or 1 per iteration; the table reports the maximum "
               "reached across that question's optimisation iterations.")
    out.append("")

    # ---- per-variant detail --------------------------------------------
    out.append("## 2. Per-variant detail")
    out.append("")
    for variant in sorted(variants):
        loc = variants[variant]["local"]
        glo = variants[variant]["global"]
        out.append(f"### `{variant}`")
        out.append("")

        if loc:
            n = len(loc)
            dist = defaultdict(int)
            for r in loc:
                dist[r["iterations"]] += 1
            out.append(f"- Questions evaluated: **{n}**")
            out.append(f"- Predictor accuracy (no explanation): **{pct(sum(r['correct'] for r in loc), n)}**")
            out.append(f"- Answers the parser could not resolve (`X`): **{sum(r['unparsed'] for r in loc)}**")
            out.append(f"- Questions where the counterfactual flipped the prediction at least once: "
                       f"**{sum(r['any_flip'] for r in loc)}/{n}**")
            out.append("")
            out.append("Iterations before early-stop:")
            out.append("")
            out.append("| Iterations | Questions |")
            out.append("|---|---|")
            for k in sorted(dist):
                out.append(f"| {k} | {dist[k]} |")
            out.append("")

        if glo:
            out.append("Global pipeline (prompt optimisation):")
            out.append("")
            out.append("| File | Iterations | First score | Best score | Last score |")
            out.append("|---|---|---|---|---|")
            for g in glo:
                out.append(f"| `{g['file']}` | {g['iterations']} | {g['first']:.3f} | "
                           f"{g['best']:.3f} | {g['last']:.3f} |")
            out.append("")

    # ---- error analysis ------------------------------------------------
    out.append("## 3. Error analysis")
    out.append("")
    for variant in sorted(variants):
        loc = variants[variant]["local"]
        wrong = [r for r in loc if not r["correct"]]
        if not wrong:
            continue
        unparsed = [r for r in wrong if r["unparsed"]]
        mischosen = [r for r in wrong if not r["unparsed"]]
        out.append(f"### `{variant}` — {len(wrong)} incorrect predictions")
        out.append("")
        out.append(f"- Genuinely picked the wrong option: **{len(mischosen)}**")
        out.append(f"- Response the parser could not resolve to any option (`X`): **{len(unparsed)}**")
        out.append("")
        if len(unparsed) > len(mischosen):
            out.append("> Most 'errors' are parsing failures, not reasoning failures. What this")
            out.append("> pipeline calls *accuracy* is closer to a **parse-success rate**, and it")
            out.append("> degrades on Vietnamese relative to English. Any accuracy figure taken")
            out.append("> from FaithLM should be read with that in mind.")
            out.append("")
        out.append("| Gold answer | Model answer |")
        out.append("|---|---|")
        for r in wrong[:15]:
            g = r["gold"].replace("|", "\\|")[:70]
            p = r["pred"].replace("|", "\\|")[:70]
            out.append(f"| {g} | {p} |")
        if len(wrong) > 15:
            out.append(f"| … | … ({len(wrong) - 15} more) |")
        out.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")

    print(f"Report written to {output_path}")
    for variant in sorted(variants):
        loc = variants[variant]["local"]
        print(f"  {variant}: {len(loc)} local, {len(variants[variant]['global'])} global")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results/experiments")
    ap.add_argument("--output", default="./docs/experiment_report.md")
    args = ap.parse_args()
    build(args.results_dir, args.output)
