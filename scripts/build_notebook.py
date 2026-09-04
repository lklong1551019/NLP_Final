#!/usr/bin/env python3
"""Generate notebooks/FaithLM_XCOPA_vi.ipynb.

Kept as a script so the notebook is reproducible and diffable. Re-run after
editing:  python scripts/build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
py = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------- 0
md(r"""
# FaithLM on Vietnamese XCOPA — reproduction, controls, and a live demo

**NLP Final Project** · Nguyen Hoang Anh · Ly Kim Long · Nguyen Dang Nhat Minh · Nguyen Quoc Anh

Runs on **Google Colab** (Runtime → Change runtime type → T4 GPU is optional) or **Kaggle**. No local paths.

| # | Section | Needs API key? |
|---|---|---|
| 1 | Install & configure | – |
| 2 | Load data | – |
| 3 | Explore & preprocess | – |
| 4 | Set up models | Explainer: yes · Predictor: no if `PRED_MODEL="phi"` |
| 5 | Evaluate on the test set | 5a (precomputed, 1,560+ results) no · 5b (live) yes |
| 6 | Compare with a baseline | yes |
| 7 | Error analysis | no |
| 8 | Demo on new Vietnamese data | yes |

The Explainer is an LLM behind any OpenAI-compatible endpoint (OpenAI, OpenRouter, a LiteLLM gateway…). Put the key in **Colab Secrets** as `LITELLM_API_KEY` (🔑 icon in the left bar), or you will be prompted. Cells that need the key skip themselves cleanly when it is absent.

Paper: `paper/main.pdf`. Every number below traces to `experiments/<member>/` — see `paper/README.md`.
""")

# ----------------------------------------------------------------- 1
md(r"""
## 1. Install & configure the environment
""")
py(r'''
import os, sys, subprocess, pathlib

REPO_URL = "https://github.com/lklong1551019/NLP_Final.git"
# Already inside a checkout (Kaggle "Add data", local run)? Use it. Otherwise clone.
if (pathlib.Path.cwd() / "main_local.py").exists():
    REPO_DIR = pathlib.Path.cwd()
else:
    REPO_DIR = pathlib.Path("/content/NLP_Final") if pathlib.Path("/content").exists() else pathlib.Path.cwd() / "NLP_Final"

if not (REPO_DIR / "main_local.py").exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("repo:", REPO_DIR, "| commit:", subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip())
''')
py(r'''
# torch ships with Colab/Kaggle. bitsandbytes (4-bit) is only for CUDA and is not needed here.
%pip install -q "transformers>=4.40" "datasets>=2.19" "openai>=1.30" scikit-learn python-dotenv tqdm accelerate anthropic google-auth
import torch, transformers
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| transformers", transformers.__version__)
''')
md(r"""
### Configuration

- `PRED_MODEL`: `"litellm"` runs the target model through the API (works without a GPU). `"phi"` loads **microsoft/phi-2** locally — the paper's own target — and needs the T4 runtime (~5.5 GB, ~2 min to load).
- `PROMPT_LANG`: `"en"` uses the paper's English prompt scaffold verbatim over Vietnamese content; `"vi"` uses the team's Vietnamese rewrite.
- `N_LIVE`: how many test questions the live sections run. Each costs ~1–3 min and a handful of API calls.
""")
py(r'''
PRED_MODEL  = "litellm"     # "litellm" (API) or "phi" (local GPU)
PROMPT_LANG = "en"          # "en" or "vi"
N_LIVE      = 5             # live questions in sections 5b/6
XAI_ITER    = 3             # optimisation steps for the live demo (paper: 20)

# --- API key & endpoint -------------------------------------------------------
def _secret(name, default=""):
    try:
        from google.colab import userdata          # Colab Secrets
        v = userdata.get(name)
        if v: return v
    except Exception:
        pass
    return os.environ.get(name, default)

API_KEY  = _secret("LITELLM_API_KEY")
if not API_KEY:
    import getpass
    API_KEY = getpass.getpass("API key for the Explainer (OpenAI-compatible; leave empty to skip live sections): ")
BASE_URL = _secret("LITELLM_BASE_URL", "https://openrouter.ai/api/v1")
XAI_ID   = _secret("LITELLM_XAI_MODEL",  "openai/gpt-4o-mini")
PRED_ID  = _secret("LITELLM_PRED_MODEL", "openai/gpt-4o-mini")

HAVE_KEY = bool(API_KEY)
os.environ.update({
    "LITELLM_API_KEY": API_KEY, "LITELLM_BASE_URL": BASE_URL,
    "LITELLM_XAI_MODEL": XAI_ID, "LITELLM_PRED_MODEL": PRED_ID,
    "LITELLM_FALLBACK_MODEL": "",          # never silently swap the explainer
    "PROMPT_LANG": PROMPT_LANG, "PYTHONUNBUFFERED": "1",
})
print("API key:", "present" if HAVE_KEY else "ABSENT — sections 4/5b/6/8 will be skipped")
print("endpoint:", BASE_URL, "| explainer:", XAI_ID, "| predictor:", PRED_MODEL if PRED_MODEL=="phi" else PRED_ID)
''')
py(r'''
# Repo modules. main_local is imported for its preprocessing functions only.
import re, ast, glob, statistics, contextlib, io
from types import SimpleNamespace
from model import llm_api
from model.predictor import (load_model, generate_api_predictor_output, generate_predictor_output_ecqa,
                             diff_task_score_ecqa, generate_predictor_reasoning, parse_choices, select_choice, contains_answer)
import model.predictor as predictor
from model.explainer import reponse_xai_model, generate_exp_prompt, generate_counterfact_prompt, generate_local_xai_prompt
from main_local import preprocess_xcopa_vi, preprocess_copa_en
print("imports OK")
''')

# ----------------------------------------------------------------- 2
md(r"""
## 2. Load the datasets

Two public corpora from the Hugging Face Hub:

- **Balanced COPA** (`pkavumba/balanced-copa`, Kavumba et al. 2019) — English causal commonsense, binary choice.
- **XCOPA** (`cambridgeltl/xcopa`, config `vi`, Ponti et al. 2020) — professional Vietnamese translation.
""")
py(r'''
from datasets import load_dataset
copa  = load_dataset("pkavumba/balanced-copa")
xcopa = load_dataset("cambridgeltl/xcopa", "vi")
print("Balanced COPA:", {k: len(v) for k, v in copa.items()})
print("XCOPA-vi     :", {k: len(v) for k, v in xcopa.items()})
''')

# ----------------------------------------------------------------- 3
md(r"""
## 3. Explore & preprocess

The two **test** splits are the same 500 items — XCOPA was translated from COPA's test set. That makes English-vs-Vietnamese a controlled comparison in which language is the only variable.
""")
py(r'''
import pandas as pd
en, vi = copa["test"], xcopa["test"]
side = pd.DataFrame({
    "type":      en["question"][:6],
    "premise_en": en["premise"][:6],
    "premise_vi": vi["premise"][:6],
    "gold_en":   [en[i]["choice1"] if en[i]["label"]==0 else en[i]["choice2"] for i in range(6)],
    "gold_vi":   [vi[i]["choice1"] if vi[i]["label"]==0 else vi[i]["choice2"] for i in range(6)],
})
display(side)
print("cause/effect split (vi):", pd.Series(vi["question"]).value_counts().to_dict())
print("gold at position 0 (vi):", f'{sum(1 for l in vi["label"] if l==0)/len(vi):.1%}',
      "— balanced, so a first-option bias scores ~50%")
print("mean premise length (chars) en/vi:", round(pd.Series(en["premise"]).str.len().mean()),
      "/", round(pd.Series(vi["premise"]).str.len().mean()))
''')
py(r'''
# The repo's preprocessing formats each item as the prompt the target model sees.
# PROMPT_LANG decides whether the scaffold (### Question / ### Câu hỏi ...) is English or Vietnamese.
xvi = preprocess_xcopa_vi(lang="vi", split="test")
print(xvi["question"][0]); print("gold:", xvi["answer"][0])
''')

# ----------------------------------------------------------------- 4
md(r"""
## 4. Set up the models

FaithLM has **two** roles (paper §2.1):

| Role | Here | Does |
|---|---|---|
| **Target / predictor** $f(\cdot)$ | Phi-2 (local) or an API model | answers the question; is the thing being explained |
| **Explainer** $g_E(\cdot)$ | API model | writes the explanation, its *contrary hint*, and acts as the LLM optimiser |

The paper's explainers (GPT-3.5-Turbo, Claude-2) are retired; any OpenAI-compatible model stands in.
""")
py(r'''
# Mirrors the argparse defaults of main_local.py so repo functions behave identically.
args = SimpleNamespace(
    data="xcopa_vi", pred_model=PRED_MODEL, xai_model="litellm",
    temp_exp=0.9, top_p_exp=0.9, max_tokens=1000,
    litellm_pred_model=None, deepseek_model=None, load_in_4bit=False,
    use_predictor_reasoning=False, device_num="0", score_mode="accuracy",
)
# Instructions, verbatim from the repo (docs/prompts_translation.md).
if llm_api.vi_prompts(args):
    TASK_INS = "Hãy chọn đáp án đúng cho mỗi câu hỏi. Lưu ý không lặp lại phần ngữ cảnh đầu vào."
    EXP_INS  = ("Dựa trên suy luận của bạn, hãy giải thích một cách khách quan lý do mô hình đưa ra câu trả lời cho các câu hỏi này. "
                "Hãy đưa ra lý do bất kể câu trả lời đó đúng hay sai. Tuyệt đối không tự trả lời câu hỏi hay đưa ra gợi ý để trả lời tốt hơn. "
                "Mỗi câu giải thích phải bắt đầu bằng <EXP>. Không lặp lại câu hỏi hay câu trả lời đầu vào. Lưu ý: Chỉ xuất ra các câu giải thích, không thêm bất kỳ nội dung nào khác.")
else:
    TASK_INS = "Please select a correct choice for the each question. Make sure not to repeat the input context."
    EXP_INS  = ("Please provide the objective explanations of why model generates the answers toward the given questions based on your thoughts. "
                "Guess the reasons of why model provides the answers whether they are correct or not. Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. "
                "Every explanations should begin with <EXP>. Make sure not to repeat the input questions and answers. Please only output the explanation sentences.")

pred_model = pred_tok = None
if PRED_MODEL == "phi":
    pred_model, pred_tok = load_model("phi", {0: "45GB"})
    generate_ans = generate_predictor_output_ecqa
    print("Phi-2 loaded on", next(pred_model.parameters()).device)
else:
    pred_model, pred_tok = "litellm", None
    generate_ans = generate_api_predictor_output
    print("predictor via API:", PRED_ID)
''')
py(r'''
# Sanity check: one prediction.
if HAVE_KEY or PRED_MODEL == "phi":
    q, g = [xvi["question"][0]], [xvi["answer"][0]]
    print("prediction:", generate_ans(pred_model, pred_tok, TASK_INS, q, g, args)[0], "| gold:", g[0])
else:
    print("skipped — no API key")
''')

# ----------------------------------------------------------------- 5
md(r"""
## 5. Evaluate on the test set

### 5a. Full-scale results (precomputed, no API needed)

Running 500 questions × up to 20 optimisation steps takes hours, so the notebook loads the raw per-question outputs committed under `experiments/` and rebuilds the tables. `fidelity` is FaithLM's $S_E=|f(X)-f(X\mid\neg E)|$, maximum over a question's iterations (the paper reports the explanation after optimisation converges).
""")
py(r'''
def parse_run(rdir):
    rows = []
    for f in sorted(glob.glob(f"{rdir}/local_*.json")):
        L = [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
        if not L: continue
        m = re.search(r"LLM-A:(.*)$", L[0]); pred = m.group(1).strip() if m else ""
        sc, ct = [], []
        for l in L[1:]:
            if l.startswith("{"):
                try:
                    e = ast.literal_eval(l)
                    if isinstance(e.get("Score"), (int, float)): sc.append(float(e["Score"]))
                    if isinstance(e.get("ControlScore"), (int, float)): ct.append(float(e["ControlScore"]))
                except Exception: pass
        rows.append(dict(correct="Corrct" in L[0], unparsed=pred=="X",
                         fid=max(sc) if sc else 0.0, ctrl=max(ct) if ct else None, iters=len(sc)))
    return rows

def summarise(rows):
    n = len(rows); ctl = [r["ctrl"] for r in rows if r["ctrl"] is not None]
    return dict(N=n, accuracy=f'{sum(r["correct"] for r in rows)/n:.1%}',
                unparsed=f'{sum(r["unparsed"] for r in rows)/n:.1%}',
                fidelity=round(statistics.fmean(r["fid"] for r in rows), 3),
                control=round(statistics.fmean(ctl), 3) if ctl else None,
                mean_iters=round(statistics.fmean(r["iters"] for r in rows), 2))

RUNS = {
  "Reproduction · COPA-en · Phi-2 + GPT-3.5 (paper Table 2 config)": "experiments/anhnh/copa_en_phi_gpt35_paper_rep1",
  "Control A · COPA-en · Phi-2":                                      "experiments/anhnh/copa_en_phi_gpt35_control",
  "Control A · XCOPA-vi · Phi-2":                                     "experiments/anhnh/xcopa_vi_phi_gpt35_control",
  "XCOPA-vi · Phi-2 · Gemini (prompt EN)":                            "experiments/minhndn/xcopa_vi_phi2_gemini35_promptEN",
  "XCOPA-vi · Qwen3.5-4B · Gemini (prompt EN)":                       "experiments/minhndn/xcopa_vi_qwen35_gemini35_promptEN",
  "XCOPA-vi · DeepSeek-v4-flash · Gemini (prompt EN)":                "experiments/minhndn/xcopa_vi_dsflash_gemini35_promptEN",
  "XCOPA-vi · Qwen3.5-4B · DeepSeek-pro (prompt VI)":                 "experiments/longlk/xcopa_vi_qwen_deepseek",
}
tbl = pd.DataFrame({k: summarise(parse_run(v)) for k, v in RUNS.items()}).T
display(tbl)
''')
md(r"""
**Reading the table.**
- Row 1 reproduces the paper: fidelity **0.872** against the ~0.85 reported for COPA (Figure 3), with the paper's own target and explainer.
- *Control A* scores a third prompt whose hint is **unrelated to the question**. On English it flips the target on **0.554** of instances by itself — ~64% of the headline score is the model following any hint at all. Corrected effect: +0.306.
- Across targets on Vietnamese, fidelity tracks the **parse rate** (`unparsed` column) more closely than model capability.
""")
md(r"""
### 5b. Live run on a few test questions

A compact version of `main_local.py`'s loop (Algorithm 1 of the paper): predict → explain → negate → re-predict with the contrary hint → score → rewrite the explanation → repeat. Stops at the first non-zero score.
""")
py(r'''
def _split(reply):
    return reply.split(":\n\n")[-1].split("\n\n")

def faithlm_one(question, gold, xai_iter=XAI_ITER, verbose=False):
    """Returns dict(pred, correct, explanation, contrary, scores, fidelity)."""
    q, g = [question], [gold]
    pred = generate_ans(pred_model, pred_tok, TASK_INS, q, g, args)
    exp = _split(reponse_xai_model(generate_exp_prompt(EXP_INS, q, pred, args), args))
    xai_list, scores, last_ctr = list(exp), [], None
    for it in range(xai_iter):
        ctr = _split(reponse_xai_model(generate_counterfact_prompt(exp, args), args))
        s = diff_task_score_ecqa(pred_model, pred_tok, TASK_INS, q, g, exp, ctr, args)
        scores.append(s); last_ctr = ctr
        if verbose: print(f"  iter {it}: score={s}")
        if s > 0: break                                   # eager stop (paper checks every 5th iter)
        exp = _split(reponse_xai_model(generate_local_xai_prompt(xai_list, scores, question, pred, args), args))
        xai_list.extend(exp)
    return dict(pred=pred[0], correct=contains_answer(gold, pred[0]) and pred[0]!="X",
                explanation=xai_list[-1], contrary=last_ctr[0] if last_ctr else "",
                scores=scores, fidelity=max(scores) if scores else 0.0)

live = []
if HAVE_KEY:
    for i in range(N_LIVE):
        with contextlib.redirect_stdout(io.StringIO()):     # the scorer prints debug blocks
            r = faithlm_one(xvi["question"][i], xvi["answer"][i])
        live.append(dict(i=i, gold=xvi["answer"][i][:40], **{k: r[k] for k in ("pred","correct","fidelity","scores")}))
        print(f"q{i}: pred={r['pred'][:40]!r} correct={r['correct']} fidelity={r['fidelity']} scores={r['scores']}")
    display(pd.DataFrame(live))
    print("\nlive fidelity:", round(statistics.fmean(x["fidelity"] for x in live), 3),
          "| API:", llm_api.stats_summary())
else:
    print("skipped — no API key")
''')

# ----------------------------------------------------------------- 6
md(r"""
## 6. Compare with a baseline

The paper compares against **SelfExp** (Madsen et al. 2024) and **Self-consistency**. Neither is in the released code, so we implement a **minimal SelfExp** here and state plainly that it is our own reduction, not the published method:

- **SelfExp (one-shot):** ask the *target itself* to explain its answer once (`generate_predictor_reasoning`), negate that explanation, score once. **No optimisation loop.**
- **FaithLM:** the loop from 5b.
- **Irrelevant-hint control:** the same scoring with a hint that has nothing to do with the question — the null baseline the paper lacks.

If FaithLM's advantage over SelfExp is real it should survive subtracting the control.
""")
py(r'''
def selfexp_one(question, gold):
    q, g = [question], [gold]
    pred = generate_ans(pred_model, pred_tok, TASK_INS, q, g, args)
    self_exp = generate_predictor_reasoning(pred_model, pred_tok, q, pred, args) or [""]
    ctr = _split(reponse_xai_model(generate_counterfact_prompt(self_exp, args), args))
    s = diff_task_score_ecqa(pred_model, pred_tok, TASK_INS, q, g, self_exp, ctr, args)
    return dict(self_explanation=self_exp[0], fidelity=s)

if HAVE_KEY:
    os.environ["FAITHLM_RANDOM_CONTROL"] = "1"         # scorer also scores an off-topic hint → predictor.CONTROL["last"]
    rows = []
    for i in range(N_LIVE):
        with contextlib.redirect_stdout(io.StringIO()):
            f = faithlm_one(xvi["question"][i], xvi["answer"][i]); ctrl_f = (predictor.CONTROL.get("last") or {}).get("diff_random")
            s = selfexp_one(xvi["question"][i], xvi["answer"][i]);  ctrl_s = (predictor.CONTROL.get("last") or {}).get("diff_random")
        rows.append(dict(i=i, FaithLM=f["fidelity"], SelfExp=s["fidelity"], control=ctrl_f if ctrl_f is not None else ctrl_s))
    os.environ.pop("FAITHLM_RANDOM_CONTROL", None)
    df = pd.DataFrame(rows); display(df)
    m = df[["FaithLM","SelfExp","control"]].mean()
    print(f"mean fidelity — FaithLM {m.FaithLM:.2f} | SelfExp {m.SelfExp:.2f} | irrelevant hint {m.control:.2f}")
    print(f"corrected (minus control) — FaithLM {m.FaithLM-m.control:+.2f} | SelfExp {m.SelfExp-m.control:+.2f}")
    print(f"\nN={N_LIVE} is a demonstration. At N=500 (experiments/anhnh) the control alone is 0.554 on English; see paper §6.3.")
else:
    print("skipped — no API key")
''')

# ----------------------------------------------------------------- 7
md(r"""
## 7. Error analysis

Three things the full-scale data show (paper §7). All from committed results; no API needed.
""")
py(r'''
# (1) What "wrong" means: genuinely wrong choice vs. an answer the parser could not read.
def err_split(rdir):
    rows = parse_run(rdir); wrong = [r for r in rows if not r["correct"]]
    return dict(N=len(rows), wrong=len(wrong), unparsed_X=sum(r["unparsed"] for r in wrong),
                wrong_choice=sum(1 for r in wrong if not r["unparsed"]))
print(pd.DataFrame({
  "COPA-en (Phi-2)":  err_split("experiments/anhnh/copa_en_phi_gpt35_control"),
  "XCOPA-vi (Phi-2)": err_split("experiments/anhnh/xcopa_vi_phi_gpt35_control"),
}).T)
print("\n→ on English 13 of 14 errors are parse failures, not reasoning failures. 'Accuracy' here is closer to a parse-success rate.")
''')
py(r'''
# (2) The cascade: an unreadable answer becomes the literal string "X", and the explainer explains *that*.
hits = []
for f in glob.glob("experiments/anhnh/xcopa_vi_phi_gpt35_control/local_*.json"):
    L = [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
    if "LLM-A:X" in L[0]:
        for l in L[1:3]:
            if l.startswith("{"):
                try: hits.append(ast.literal_eval(l)["XAI prompt"][:220])
                except Exception: pass
        if len(hits) >= 2: break
for h in hits: print("•", h, "…\n")
print("→ the explanation, its contrary hint, and the score are all about an answer that was never produced.")
''')
py(r'''
# (3) Position bias (Control C, minhndn): on XCOPA-vi with English prompts, Phi-2's 83 correct answers among 117
#     parseable ones are exactly the 83 on which it picked the FIRST-listed option — yet it still scores 0.755.
r = summarise(parse_run("experiments/minhndn/xcopa_vi_phi2_gemini35_promptEN"))
print("Phi-2 · XCOPA-vi · prompt EN:", r)
print("→ a target that does no reasoning is rated as producing faithful explanations. Full analysis: docs/experiment_explainer_sweep.md §4")
''')
py(r'''
# (4) From the live run: raw predictor output behind any unparseable answer.
if HAVE_KEY and live:
    bad = [x for x in live if x["pred"] == "X"]
    if bad:
        os.environ["FAITHLM_DEBUG"] = "1"
        i = bad[0]["i"]
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            generate_ans(pred_model, pred_tok, TASK_INS, [xvi["question"][i]], [xvi["answer"][i]], args)
        raw = [l for l in buf.getvalue().splitlines() if "RAW" in l]
        print(f"q{i} raw output:", raw[0][:500] if raw else "(none captured)")
        os.environ.pop("FAITHLM_DEBUG", None)
    else:
        print(f"all {len(live)} live predictions were parseable.")
else:
    print("skipped — no live run")
''')

# ----------------------------------------------------------------- 8
md(r"""
## 8. Demo on new Vietnamese data

Type any Vietnamese premise with two alternatives. These examples are **not** in XCOPA. The pipeline predicts, explains, negates, and reports whether the negation moved the target — plus the irrelevant-hint control so you can see how much of a flip is just suggestibility.
""")
py(r'''
DEMO = [
    dict(kind="cause",  premise="Cả tòa nhà mất điện.",
         c1="Một cơn bão đã làm đứt đường dây điện.", c2="Có người vừa mở cửa sổ."),
    dict(kind="effect", premise="Cô ấy bỏ quên ví ở nhà.",
         c1="Cô ấy phải mượn tiền bạn để ăn trưa.", c2="Cô ấy được thăng chức."),
]
# Or edit this one:
DEMO.append(dict(kind="effect", premise="Trời mưa suốt cả tuần.",
                 c1="Con đường đất trở nên lầy lội.", c2="Nhà hàng mở thêm chi nhánh."))

def fmt(d):
    purp = ("nguyên nhân" if d["kind"]=="cause" else "kết quả") if llm_api.vi_prompts(args) else d["kind"]
    if llm_api.vi_prompts(args):
        return f"### Câu hỏi: Đâu là {purp} của Tiền đề?\n### Tiền đề: {d['premise']}\n### Lựa chọn: [choice]{d['c1']}@ [choice]{d['c2']}@"
    return f"###Question: What is the {purp} of the Premise?\n### Premise: {d['premise']}\n### Choices: [choice]{d['c1']}@ [choice]{d['c2']}@"

if HAVE_KEY:
    os.environ["FAITHLM_RANDOM_CONTROL"] = "1"
    for d in DEMO:
        q = fmt(d)
        with contextlib.redirect_stdout(io.StringIO()):
            r = faithlm_one(q, d["c1"], xai_iter=XAI_ITER)      # c1 is treated as "gold" only for scoring bookkeeping
            ctrl = (predictor.CONTROL.get("last") or {}).get("diff_random")
        print("="*80); print(d["premise"], f"({d['kind']})"); print("  A:", d["c1"]); print("  B:", d["c2"])
        print("  target answered :", r["pred"])
        print("  explanation     :", r["explanation"][:300])
        print("  contrary hint   :", r["contrary"][:300])
        print(f"  flipped by contrary hint: {'YES' if r['fidelity']>0 else 'no'}   | flipped by irrelevant hint: {'YES' if ctrl else 'no'}")
    os.environ.pop("FAITHLM_RANDOM_CONTROL", None)
    print("\nAPI:", llm_api.stats_summary())
else:
    print("skipped — no API key. Add LITELLM_API_KEY to Colab Secrets and re-run from section 1.")
''')
md(r"""
---
**What to take away.** A flip proves the target *moved*; it does not by itself prove the explanation was *why*. The control on the last line is the difference. Details, four controls, and every number's provenance: `paper/main.pdf`, `paper/README.md`.
""")

nb.cells = C
out = "notebooks/FaithLM_XCOPA_vi.ipynb"
nbf.validate(nb)
nbf.write(nb, out)
print(f"wrote {out}: {len(C)} cells")
