"""Flip reproducibility: is a 1.0 a property of the explanation, or of the dice?

The pipeline scores 1.0 the first time a counterfactual (negated) explanation,
sampled at temperature 0.9, makes the predictor change its answer - and then
early-stops. That conflates "this explanation is causally load-bearing" with
"this particular negation sample happened to work".

Here we hold the explanation fixed, re-sample the negation exactly once, and
re-ask the predictor. The reproduction rate is how much of the 1.0 signal
survives a second roll of the dice.

Usage:
    python scripts/flip_reproducibility.py --limit 5     # dry run, inspect by hand
    python scripts/flip_reproducibility.py               # full set
"""
import argparse, ast, glob, json, math, os, re, sys, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
os.environ["PROMPT_LANG"] = "en"          # paper's original EN templates

from main_local import split_reply
from model import llm_api
from model.explainer import generate_counterfact_prompt
from model.predictor import contains_answer, parse_choices, select_choice

SRC = "results/experiments/xcopa_vi_xai_sweep_en/vertex-google-gemini-3-5-flash"
OUT = "results/experiments/flip_repro_gemini_en"
XAI_MODEL = "vertex/google/gemini-3.5-flash"
PRED_MODEL = "deepseek/deepseek-v4-flash"
# main_local.py, xcopa_vi + PROMPT_LANG=en branch, verbatim.
TASK_INSTRUCTION = "Please select a correct choice for the each question.                             Make sure not to repeat the input context."
BASE = "Below is an instruction that describes a task. Write a response that appropriately completes the request of input."
HDR = re.compile(r"^=+ (Corrct|Wrong)\s+--> Q:(.*) \|\| GT-A:(.*) \|\| LLM-A:(.*)$")
ARGS = types.SimpleNamespace(data="xcopa_vi")


def load_flips(src):
    """One entry per file whose LLM-OPT loop ever scored 1.0.

    The flip record is the FIRST 1.0, not the last: early-stop only fires at
    iter % 5 == 0, so a flip at iter 1-4 leaves further records behind it.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(src, "local_*.json")),
                       key=lambda p: int(p.rsplit("sample-", 1)[1].split(".")[0])):
        lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
        m = HDR.match(lines[0])
        if not m or m.group(4).strip() == "X":
            continue
        recs = [ast.literal_eval(l) for l in lines[1:]]
        scores = [r["Score"] for r in recs]
        if 1.0 not in scores:
            continue
        i = scores.index(1.0)
        out.append({
            "id": int(path.rsplit("sample-", 1)[1].split(".")[0]),
            "orig_flip_round": i + 1,
            "n_records": len(recs),
            "question": ast.literal_eval(m.group(2))[0],
            "gold": m.group(3).strip(),
            "llm_a_nohint": m.group(4).strip(),
            "explanation": recs[i]["XAI prompt"],
        })
    return out


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_one(item, xai_model):
    reply = llm_api.chat(
        generate_counterfact_prompt([item["explanation"]], ARGS),
        model=xai_model, max_tokens=1000, temperature=0.9, top_p=0.9,
        system="You are an expert at explaining language model behavior.",
    )
    # Same post-processing the pipeline applies to a counterfactual reply, so
    # the hint string is byte-identical in shape to the one that scored 1.0.
    negation = split_reply(reply.split(":\n\n")[-1])[0]
    # generate_counterfact_prompt interpolates the raw list repr into the
    # prompt ("Sentences: ['...']"), and Gemini sometimes mirrors that
    # formatting back. Unwrap so the hint is the sentence, not its repr.
    if negation.startswith("[") and negation.endswith("]"):
        try:
            inner = ast.literal_eval(negation)
            if isinstance(inner, list) and inner:
                negation = str(inner[0]).strip()
        except (ValueError, SyntaxError):
            pass
    # predictor.py::diff_task_score_ecqa, EN branch count_final_prompt, verbatim.
    prompt = (f"{BASE}\n\n### Instruction: {TASK_INSTRUCTION}\n\n### Hint: {negation}"
              f"\n\n### Input: {item['question']}\n\n### Response: Let's think step by step.")
    raw = llm_api.chat(prompt, model=PRED_MODEL, max_tokens=200, temperature=0.0,
                       system="You are a helpful assistant.")
    picked = select_choice(raw, parse_choices(prompt))
    if picked is None and contains_answer(item["gold"], raw):
        picked = item["gold"]
    # Pipeline formula: abs(true_score - count_score) == 1 <=> correctness differs.
    if picked is None:
        reproduced = None
    else:
        reproduced = contains_answer(item["gold"], picked) != contains_answer(
            item["gold"], item["llm_a_nohint"])
    return negation, raw, picked, reproduced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--src", default=SRC,
                    help="sweep-arm results dir to re-sample flips from")
    ap.add_argument("--xai-model", default=XAI_MODEL, dest="xai_model",
                    help="explainer that re-generates the negation")
    a = ap.parse_args()

    items = load_flips(a.src)
    if a.limit:
        items = items[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    jsonl = os.path.join(a.out, "per_question.jsonl")
    rows = []
    with open(jsonl, "w", encoding="utf-8") as fh:
        for n, it in enumerate(items, 1):
            try:
                negation, raw, picked, reproduced = run_one(it, a.xai_model)
            except Exception as exc:
                print(f"[{n}/{len(items)}] id={it['id']} ERROR {exc}", flush=True)
                negation, raw, picked, reproduced = "", "", None, None
            row = {**it, "new_negation": negation, "new_raw": raw,
                   "new_answer": picked, "reproduced": reproduced}
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{n}/{len(items)}] id={it['id']} round={it['orig_flip_round']} "
                  f"reproduced={reproduced}", flush=True)

    def bucket(sel):
        sub = [r for r in rows if sel(r)]
        parsed = [r for r in sub if r["reproduced"] is not None]
        k = sum(1 for r in parsed if r["reproduced"])
        lo, hi = wilson(k, len(parsed))
        return {"n": len(sub), "unparsed": len(sub) - len(parsed), "n_scored": len(parsed),
                "reproduced": k, "rate": k / len(parsed) if parsed else None,
                "ci95_wilson": [lo, hi]}

    summary = {
        "source_dir": a.src, "xai_model": a.xai_model, "pred_model": PRED_MODEL,
        "prompt_lang": "en", "negation_temperature": 0.9, "negation_top_p": 0.9,
        "overall": bucket(lambda r: True),
        "flip_round_1": bucket(lambda r: r["orig_flip_round"] == 1),
        "flip_round_ge2": bucket(lambda r: r["orig_flip_round"] >= 2),
        "api_stats": dict(llm_api.STATS),
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(llm_api.stats_summary())


if __name__ == "__main__":
    main()
