"""Iterations to the first prediction flip, per --score_mode.

This is the comparison the "does a better signal need fewer iterations?" question
actually needs. A flip -- the target's argmax changing once the contrary hint is
injected -- is the same event in every mode, so it is the one quantity that can be
compared across objectives that are otherwise on different scales.

Run every mode with --no_early_stop so each gets the same budget; a mode allowed to
quit on its own terms would otherwise be scored on its stopping rule, not its signal.

Questions that never flip within the budget are censored: they are reported as a
separate count rather than folded into the mean, which would otherwise reward a mode
for failing early.

Usage: python scripts/iters_to_flip.py results/metric_sweep/copa_en_phi_openai
"""
import collections
import json
import os
import statistics as st
import sys


def per_question(path):
    per = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                per[row["question_idx"]].append(row)
    for rows in per.values():
        rows.sort(key=lambda r: r["iter"])
    return per


def main(root):
    modes = {}
    for mode in sorted(os.listdir(root)):
        path = os.path.join(root, mode, "metrics.jsonl")
        if os.path.isfile(path):
            modes[mode] = per_question(path)
    if not modes:
        sys.exit(f"no metrics.jsonl under {root}")

    hdr = (f"{'mode':15s} {'câu':>4s} {'lật được':>9s} {'vòng đến lật':>13s} "
           f"{'trung vị':>9s} {'lật ngay vòng 0':>16s} {'ngân sách dùng':>15s}")
    print(f"root: {root}\n{hdr}\n" + "-" * len(hdr))

    for mode, per in modes.items():
        first, censored, budget = [], 0, []
        for rows in per.values():
            budget.append(len(rows))
            hit = next((r["iter"] for r in rows if r.get("flip", 0.0) >= 1.0), None)
            if hit is None:
                censored += 1
            else:
                first.append(hit)
        n = len(per)
        m = f"{st.mean(first):.2f}" if first else "-"
        med = f"{st.median(first):.1f}" if first else "-"
        zero = sum(1 for x in first if x == 0)
        print(f"{mode:15s} {n:>4d} {n-censored:>4d}/{n:<4d} {m:>13s} {med:>9s} "
              f"{zero:>16d} {st.mean(budget):>15.2f}")

    print("\nCách đọc:")
    print("  'vòng đến lật' chỉ tính trên các câu CÓ lật. 'lật được' cho biết mẫu số.")
    print("  Một mode lật ít hơn nhưng nhanh hơn thì không phải là tốt hơn - đọc hai cột cùng nhau.")
    print("  Nếu 'ngân sách dùng' khác nhau giữa các mode thì early stop chưa tắt,")
    print("  và cột 'vòng đến lật' đang bị cắt cụt khác nhau -> không so được.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/metric_sweep")
