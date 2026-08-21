# FaithLM — Experiment Report

> Results directory: `./results/experiments/phi2_xcopa_vi_en_vertexgemini`

## 0. Experimental setup

Both FaithLM roles were served by an OpenAI-compatible LiteLLM gateway; no
local model weights were used. See `docs/changelog_2026-08-17.md` §E.

| Role | Model | Temperature | max_tokens |
|---|---|---|---|
| Predictor — initial answer | `deepseek/deepseek-v4-flash` | 0.0 | 200 |
| Predictor — scoring | `deepseek/deepseek-v4-flash` | 0.01 | 200 |
| Explainer | `vertex/google/gemini-3.5-flash` | `--temp_exp` | 1000 |
| Fallback (empty completions) | `deepseek/deepseek-v3.2` | — | — |

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
- Total LLM calls: **490**
- Calls that failed outright: 0
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **0** (0.0%)

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `(root)` | 200 | 41.5% | 41.5% | 0.755 | 75.5% | 2.58 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `(root)`

- Questions evaluated: **200**
- Predictor accuracy (no explanation): **41.5%**
- Answers the parser could not resolve (`X`): **83**
- Questions where the counterfactual flipped the prediction at least once: **151/200**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 139 |
| 6 | 56 |
| 8 | 5 |

## 3. Error analysis

### `(root)` — 117 incorrect predictions

- Genuinely picked the wrong option: **34**
- Response the parser could not resolve to any option (`X`): **83**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Nó dễ vỡ. | X |
| Tôi lấy một cuống vé. | Tôi lụy một cuống vé. |
| Nó đã chết. | X |
| Anh ấy bị cảm. | Anh bỏ thuốc lá. |
| Bố mẹ cậu nhốt cậu anh. | X |
| Người chủ xích con chó con. | Người chủ mặc cho con chó con một cổ áo. |
| Anh đứng trên mặt hồ tĩnh lặng. | X |
| Anh ấy bị chấn động. | Anh lạc trong suy nghĩ. |
| Con chó của họ bỏ nhà ra đi. | Đồ trang sức đắt tiền đã bị mất khỏi nhà. |
| Thời tiết se lạnh. | Ngực tôi cảm thấy căng cứng. |
| Một mùi thơm ấm áp tràn ngập nhà bếp. | X |
| Cô đi giày cao gót. | X |
| Nước sôi. | . |
| Tôi đi dép. | X |
| Ông mất sự ủng hộ của cử tri. | X |
| … | … (102 more) |

