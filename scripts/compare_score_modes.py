"""Compare --score_mode runs on a common yardstick.

Each mode optimises a different objective, so ranking them by "final score" is
meaningless -- the numbers are on different scales. Every run also records the
probability-based metrics regardless of what it optimised, and those are what make
the runs comparable.

Report per mode:
  iterations   how many optimisation steps each question actually took
  |shift|      how far the contrary hint moved the target, best over the run
  tv           same thing as a proper divergence
  flip rate    share of questions where the prediction changed at least once

Usage: python scripts/compare_score_modes.py results/metric_sweep/copa_en_phi_openai
"""
import collections
import json
import os
import statistics as st
import sys


def load(root):
    runs = {}
    for mode in sorted(os.listdir(root)):
        path = os.path.join(root, mode, "metrics.jsonl")
        if not os.path.isfile(path):
            continue
        per = collections.defaultdict(list)
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    per[row["question_idx"]].append(row)
        if per:
            runs[mode] = per
    return runs


def main(root):
    runs = load(root)
    if not runs:
        sys.exit(f"no metrics.jsonl under {root}")

    print(f"root: {root}\n")
    hdr = (f"{'mode':15s} {'câu':>4s} {'vòng/câu':>9s} {'vòng tối đa':>12s} "
           f"{'|shift| tốt nhất':>17s} {'tv tốt nhất':>12s} {'flip':>7s}")
    print(hdr)
    print("-" * len(hdr))
    missing = []
    for mode, per in runs.items():
        iters = [len(v) for v in per.values()]
        shifts, tvs, flips = [], [], []
        for rows in per.values():
            s = [abs(r["prob_shift"]) for r in rows if "prob_shift" in r]
            t = [r["tv"] for r in rows if "tv" in r]
            f = [r["flip"] for r in rows if "flip" in r]
            if s:
                shifts.append(max(s))
            if t:
                tvs.append(max(t))
            if f:
                flips.append(max(f))
        if not shifts:
            missing.append(mode)
        fmt = lambda xs: f"{st.mean(xs):.4f}" if xs else "-"
        print(f"{mode:15s} {len(per):>4d} {st.mean(iters):>9.2f} {max(iters):>12d} "
              f"{fmt(shifts):>17s} {fmt(tvs):>12s} "
              f"{(st.mean(flips) if flips else float('nan')):>7.2f}")

    if missing:
        print(f"\n!! {', '.join(missing)} thiếu chỉ số xác suất -- chạy lại "
              f"không có --no_log_all_metrics thì mới so được.")

    print("\nĐọc bảng này thế nào:")
    print("  'vòng/câu' KHÔNG phải thước đo hiệu quả. Quy tắc dừng của paper")
    print("  (iter%5==0 và sum(scores)!=0) kích hoạt ngay vòng 0 khi điểm nhị phân")
    print("  khác 0, nên baseline dừng non chứ không phải hội tụ nhanh. So hai mode")
    print("  bằng '|shift| tốt nhất' và 'flip' -- đó là fidelity thực sự đạt được.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/metric_sweep")
