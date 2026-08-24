# FaithLM — Experiment Report

> Results directory: `./results/experiments/xcopa_vi_target_qwen35/qwen-qwen3-7-max`

## 0. Experimental setup

The Explainer was served by an OpenAI-compatible gateway; the Predictor ran
from local Hugging Face weights on the GPU. See `docs/changelog_2026-08-17.md` §E.

| Role | Model | Temperature | max_tokens |
|---|---|---|---|
| Predictor — initial answer | `Qwen/Qwen3.5-4B (4-bit NF4, local)` | 0.0 | 200 |
| Predictor — scoring | `Qwen/Qwen3.5-4B (4-bit NF4, local)` | 0.01 | 200 |
| Explainer | `qwen/qwen3.7-max` | `--temp_exp` | 1000 |
| Fallback (empty completions) | `` | — | — |

### Deviations from the paper's Table 2

These are material and must be quoted alongside any number in this report.

| Hyper-parameter | Paper (COPA) | This run |
|---|---|---|
| Fidelity optimisation steps (Alg. 1) | 20 | 8 |
| Predictor temperature | 0.7 | 0.0 / 0.01 |
| Explainer temperature | 0.9 | 0.9 |
| Explainer top-p | 0.9 | 0.9 |
| Trigger-prompt steps (Alg. 2) | 100 | 8 |
| Sampled instances per step | 30 (Table 2) / 15 (§4.3) | 12 |
| Repetitions | 3 runs averaged, grid search | 1 |
| Instances evaluated | 500 | 200 (indices 0–199, **not** a random sample) |
| Target model | Vicuna-7B, Phi-2 | API model (see above) |
| Explainer | GPT-3.5-Turbo, Claude-2 | API model (see above) |
| Baselines | SelfExp, Self-consistency | **not implemented** |
| Truthfulness metric | GPT-4o + RoBERTa-L + XLNet-L | **not implemented** |

### API call statistics

- Processes reporting: 1
- Total LLM calls: **1405**
- Calls that failed outright: 0
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **0** (0.0%)

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `(root)` | 200 | 92.0% | 5.0% | 0.975 | 97.5% | 3.50 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `(root)`

- Questions evaluated: **200**
- Predictor accuracy (no explanation): **92.0%**
- Answers the parser could not resolve (`X`): **10**
- Questions where the counterfactual flipped the prediction at least once: **195/200**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 101 |
| 6 | 96 |
| 8 | 3 |

## 3. Error analysis

### `(root)` — 16 incorrect predictions

- Genuinely picked the wrong option: **6**
- Response the parser could not resolve to any option (`X`): **10**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Chuông báo cháy reng. | X |
| Người cha đuổi con trai ra khỏi nhà. | X |
| Đầu chúng va vào nhau. | X |
| Cô bị mất biên lai. | Chiếc váy không vừa. |
| Ông ấy đang nghĩ về những lời nói của bạn mình. | 1**: " |
| Rèm cửa rung rinh. | Chuông cửa reo. |
| Tôi rất tức giận. | Tôi rút phích cắm đèn. |
| Anh nâng thanh xà trên đầu. | Anh uốn cong cơ bắp trong gương. |
| Cô muốn rời khỏi bữa tiệc. | X |
| Bụi bay vào mắt ông ấy. | X |
| Cô bầm tím đầu gối. | X |
| Tôi đã thay áo. | Tôi đeo tạp dề. (I put on an apron.) |
| Cô bước ra khỏi hàng. | X |
| Đứa bé làm bẩn tã của mình. | X |
| Anh ấy đi học đại học. | X |
| … | … (1 more) |

