"""Random-hint control: is the flip caused by the explanation's meaning, or by any text?

The fidelity measure claims a flip shows the explanation drives the prediction. But
injecting *any* plausible sentence into the prompt also perturbs the target. If a
contrary hint written for a different question flips the answer just as often, the
measure is reading perturbation sensitivity, not faithfulness - and that applies to
the published metric as much as to ours.

Neither the paper nor the released code runs this check.

For each question we score three conditions on the same target:
    no hint          P(a0)
    aligned    ¬E_i  the contrary of THIS question's own explanation
    shuffled   ¬E_j  the contrary of a DIFFERENT question's explanation

Aligned >> shuffled means the effect is semantic. Aligned ≈ shuffled means it is not.
"""
import argparse, ast, collections, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def final_explanations(run_dir):
    """Last explanation each question converged on, from the saved per-question files."""
    out = {}
    for name in os.listdir(run_dir):
        if not (name.startswith("local_") and name.endswith(".json")):
            continue
        idx = int(name.rsplit("sample-", 1)[1].split(".")[0])
        rows = []
        for line in open(os.path.join(run_dir, name)):
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(ast.literal_eval(line))
                except (ValueError, SyntaxError):
                    pass
        if rows:
            out[idx] = rows[-1]["XAI prompt"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/final_qwen/logprob")
    ap.add_argument("--out", default="results/random_hint_control.json")
    ap.add_argument("--pred_model", default="qwen")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args_cli = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    import main_local as M
    from model.predictor import load_model, normalize_answer, parse_choices, _answer_cue_prompt
    from model.fidelity import choice_probs_local
    from model.explainer import reponse_xai_model, generate_counterfact_prompt

    class A:
        xai_model = "openai"; openai_key = None; data = "xcopa_vi"
        temp_exp = 0.9; top_p_exp = 0.9; max_tokens = 300
        openai_model = args_cli.openai_model; pred_model = args_cli.pred_model
    args = A()

    expl = final_explanations(args_cli.run_dir)
    idxs = sorted(expl)[: args_cli.limit]
    print(f"{len(idxs)} câu có giải thích cuối cùng")

    model, tok = load_model(args.pred_model, {0: "14GiB"}, load_in_4bit=False)
    td = M.preprocess_xcopa_vi(lang="vi", split="test")
    TASK = "Hãy chọn phương án đúng cho mỗi câu hỏi. Lưu ý không lặp lại ngữ cảnh đầu vào."

    print("sinh contrary hint...")
    hints = {}
    for n, i in enumerate(idxs):
        hints[i] = reponse_xai_model(generate_counterfact_prompt([expl[i]], args), args)
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(idxs)}")

    # A derangement: every question gets someone else's hint.
    rng = random.Random(args_cli.seed)
    shuffled = idxs[:]
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(idxs, shuffled)):
            break
    partner = dict(zip(idxs, shuffled))

    rows = []
    for i in idxs:
        q, gold = td["question"][i], td["answer"][i]
        ch = parse_choices(q)
        if len(ch) < 2:
            continue
        p0 = choice_probs_local(model, tok, _answer_cue_prompt(TASK, q), ch)
        k = 0 if p0[0] >= p0[1] else 1
        pa = choice_probs_local(model, tok, _answer_cue_prompt(TASK, q, hints[i]), ch)
        ps = choice_probs_local(model, tok, _answer_cue_prompt(TASK, q, hints[partner[i]]), ch)
        rows.append({
            "idx": i, "partner": partner[i],
            "p_before": p0[k], "p_aligned": pa[k], "p_shuffled": ps[k],
            "flip_aligned": int((0 if pa[0] >= pa[1] else 1) != k),
            "flip_shuffled": int((0 if ps[0] >= ps[1] else 1) != k),
            "correct_before": int(normalize_answer(ch[k]) == normalize_answer(gold)),
        })

    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    json.dump(rows, open(args_cli.out, "w"), ensure_ascii=False, indent=1)

    n = len(rows)
    fa = sum(r["flip_aligned"] for r in rows); fs = sum(r["flip_shuffled"] for r in rows)
    sa = sum(r["p_before"] - r["p_aligned"] for r in rows) / n
    ss = sum(r["p_before"] - r["p_shuffled"] for r in rows) / n
    b = sum(1 for r in rows if r["flip_aligned"] and not r["flip_shuffled"])
    c = sum(1 for r in rows if r["flip_shuffled"] and not r["flip_aligned"])
    import math
    p = math.erfc(math.sqrt(((abs(b - c) - 1) ** 2 / (b + c)) / 2)) if b + c else None

    print(f"\n=== random-hint control, {n} câu ===")
    print(f"  tỉ lệ lật, hint ĐÚNG của câu đó : {fa}/{n} = {fa/n:.3f}")
    print(f"  tỉ lệ lật, hint của câu KHÁC    : {fs}/{n} = {fs/n:.3f}")
    print(f"  dịch chuyển xác suất, đúng      : {sa:+.4f}")
    print(f"  dịch chuyển xác suất, xáo trộn  : {ss:+.4f}")
    print(f"  McNemar: chỉ đúng {b}, chỉ xáo {c}" + (f", p={p:.4f}" if p else ""))
    print()
    if p is not None and p < 0.05 and fa > fs:
        print("  -> Hiệu ứng CÓ tính ngữ nghĩa: hint đúng lật nhiều hơn hint ngẫu nhiên.")
    else:
        print("  -> KHÔNG tách được khỏi nhiễu nhiễu loạn: hint của câu khác cũng lật tương đương.")
        print("     Nếu vậy thì phép đo fidelity - của cả paper - đang đọc độ nhạy với nhiễu,")
        print("     chứ không phải mức độ trung thực của lời giải thích.")


if __name__ == "__main__":
    main()
