"""Position-bias analysis for a local-pipeline results directory.

Answers: when the predictor's answer parses, is it actually reading the
question, or just picking the first-listed choice? A pure first-picker scores
exactly the rate at which gold sits in position 1, so comparing the two
separates comprehension from format heuristics.

Usage:
    python scripts/analyze_position_bias.py --results_dir ./results/experiments/phi2_xcopa_vi_en_vertexgemini
"""

import argparse
import glob
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    parsed = correct = first_pick = gold_first = total = 0
    for path in sorted(glob.glob(os.path.join(args.results_dir, "local_*.json"))):
        with open(path, encoding="utf-8") as f:
            head = f.read().splitlines()[0]
        total += 1
        m = re.search(r"\[choice\](.*?)@ \[choice\](.*?)@", head)
        ans = head.split("LLM-A:")[-1].strip()
        gt = head.split("GT-A:")[-1].split("||")[0].strip()
        if not m or ans == "X":
            continue
        parsed += 1
        c1 = m.group(1).strip()
        if ans == gt:
            correct += 1
        if ans == c1:
            first_pick += 1
        if gt == c1:
            gold_first += 1

    if not parsed:
        print(f"{total} files, nothing parsed.")
        return
    print(f"Questions: {total} | parsed: {parsed} ({parsed/total:.0%})")
    print(f"Correct among parsed:     {correct}/{parsed} ({correct/parsed:.0%})")
    print(f"Picked first-listed:      {first_pick}/{parsed} ({first_pick/parsed:.0%})")
    print(f"Gold at first position:   {gold_first}/{parsed} ({gold_first/parsed:.0%})")
    print(
        f"-> a pure first-picker would score {gold_first/parsed:.0%} on this subset; "
        f"accuracy above that margin is the comprehension signal."
    )


if __name__ == "__main__":
    main()
