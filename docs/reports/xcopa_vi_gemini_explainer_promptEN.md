# FaithLM — Experiment Report

> Results directory: `./results/experiments/xcopa_vi_xai_sweep_en/vertex-google-gemini-3-5-flash`

## 0. Experimental setup

Both FaithLM roles were served by an OpenAI-compatible LiteLLM gateway; no
local model weights were used. See `docs/changelog_2026-08-17.md` §E.

| Role | Model | Temperature | max_tokens |
|---|---|---|---|
| Predictor — initial answer | `n/a` | 0.0 | 200 |
| Predictor — scoring | `n/a` | 0.01 | 200 |
| Explainer | `n/a` | `--temp_exp` | 1000 |
| Fallback (empty completions) | `n/a` | — | — |

### Deviations from the paper's Table 2

These are material and must be quoted alongside any number in this report.

| Hyper-parameter | Paper (COPA) | This run |
|---|---|---|
| Fidelity optimisation steps (Alg. 1) | 20 | 8 |
| Predictor temperature | 0.7 | 0.0 / 0.01 |
| Explainer temperature | 0.9 | 0.01 (local) / 0.9 (global) |
| Explainer top-p | 0.9 | not set (1.0) |
| Trigger-prompt steps (Alg. 2) | 100 | 8 |
| Sampled instances per step | 30 (Table 2) / 15 (§4.3) | 12 |
| Repetitions | 3 runs averaged, grid search | 1 |
| Instances evaluated | 500 | 100 (indices 0–99, **not** a random sample) |
| Target model | Vicuna-7B, Phi-2 | API model (see above) |
| Explainer | GPT-3.5-Turbo, Claude-2 | API model (see above) |
| Baselines | SelfExp, Self-consistency | **not implemented** |
| Truthfulness metric | GPT-4o + RoBERTa-L + XLNet-L | **not implemented** |

### API call statistics

- Processes reporting: 6
- Total LLM calls: **1865**
- Calls that failed outright: 0
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **0** (0.0%)

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `(root)` | 100 | 93.0% | 3.0% | 0.960 | 96.0% | 4.40 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `(root)`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **93.0%**
- Answers the parser could not resolve (`X`): **3**
- Questions where the counterfactual flipped the prediction at least once: **96/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 34 |
| 6 | 61 |
| 8 | 5 |

## 3. Error analysis

### `(root)` — 7 incorrect predictions

- Genuinely picked the wrong option: **4**
- Response the parser could not resolve to any option (`X`): **3**

| Gold answer | Model answer |
|---|---|
| Tôi lấy một cuống vé. | Tôi tìm thấy một vũ khí. |
| Cô đã thuê một người quản lý chiến dịch. | Cô làm chứng trước tòa. |
| Phi công điều hướng khỏi cơn bão. | X |
| Đứa bé làm bẩn tã của mình. | Đứa bé chảy nước dãi trên chiếc yếm của cô. |
| Người cha nắm lấy tay con trai mình. | X |
| Thuyền kayak đến bờ. | X |
| Họ đã nhìn thấy một con cá mập. | Họ ướt đẫm. |

