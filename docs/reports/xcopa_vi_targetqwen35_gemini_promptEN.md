# FaithLM — Experiment Report

> Results directory: `./results/experiments/xcopa_vi_target_qwen35/vertex-google-gemini-3-5-flash`

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
| Instances evaluated | 500 | 200 (indices 0–199, **not** a random sample) |
| Target model | Vicuna-7B, Phi-2 | API model (see above) |
| Explainer | GPT-3.5-Turbo, Claude-2 | API model (see above) |
| Baselines | SelfExp, Self-consistency | **not implemented** |
| Truthfulness metric | GPT-4o + RoBERTa-L + XLNet-L | **not implemented** |

### API call statistics

- Processes reporting: 1
- Total LLM calls: **1220**
- Calls that failed outright: 0
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **0** (0.0%)

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `(root)` | 200 | 90.5% | 6.5% | 0.945 | 94.5% | 3.04 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `(root)`

- Questions evaluated: **200**
- Predictor accuracy (no explanation): **90.5%**
- Answers the parser could not resolve (`X`): **13**
- Questions where the counterfactual flipped the prediction at least once: **189/200**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 120 |
| 6 | 76 |
| 8 | 4 |

## 3. Error analysis

### `(root)` — 19 incorrect predictions

- Genuinely picked the wrong option: **6**
- Response the parser could not resolve to any option (`X`): **13**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Một mùi thơm ấm áp tràn ngập nhà bếp. | X |
| Chuông báo cháy reng. | Thang máy ngừng hoạt động. |
| Anh cầu xin cô quay trở lại. | X |
| Thật là buồn tẻ. | Nó rẻ tiền. |
| Ánh sáng mặt trời chói lóa. | X |
| Mối ăn gỗ trong nhà. | Mối biến mất khỏi nhà. |
| Đối thủ của cô buộc tội cô gian lận. | X |
| Mọi người khen anh. | Mọi người cô lập anh. |
| Cô trèo lên một sợi dây. | X |
| Nước trái cây tràn ra. | X |
| Cô nhảy dây. | Cô chơi cờ đam. |
| Cô bước ra khỏi hàng. | X |
| Tôi muốn kiểm tra thời gian. | X |
| Đứa bé làm bẩn tã của mình. | X |
| Cô quên cài đồng hồ báo thức. | X |
| … | … (4 more) |

