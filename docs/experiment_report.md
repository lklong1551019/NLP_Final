# FaithLM — Experiment Report

> Results directory: `./results/experiments`

## 0. Experimental setup

Both FaithLM roles were served by an OpenAI-compatible LiteLLM gateway; no
local model weights were used. See `docs/changelog_2026-08-17.md` §E.

| Role | Model | Temperature | max_tokens |
|---|---|---|---|
| Predictor — initial answer | `deepseek/deepseek-v4-flash` | 0.0 | 200 |
| Predictor — scoring | `deepseek/deepseek-v4-flash` | 0.01 | 200 |
| Explainer | `deepseek/deepseek-v4-pro` | `--temp_exp` | 1000 |
| Fallback (empty completions) | `openrouter/deepseek/deepseek-v3.2` | — | — |

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

- Processes reporting: 26
- Total LLM calls: **4287**
- Calls that failed outright: 2
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **840** (19.6%)

> The fallback share is not incidental: a non-trivial fraction of the
> Explainer's output came from a different model than the one named above,
> because the primary returned an empty body for prompts its content filter
> rejected. Vietnamese prompts triggered this far more often than English ones.

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `copa_en_litellm_litellm` | 100 | 96.0% | 4.0% | 0.880 | 88.0% | 4.25 |
| `xcopa_vi_litellm_litellm` | 100 | 90.0% | 9.0% | 0.920 | 92.0% | 3.48 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `copa_en_litellm_litellm`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **96.0%**
- Answers the parser could not resolve (`X`): **4**
- Questions where the counterfactual flipped the prediction at least once: **88/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 41 |
| 6 | 44 |
| 8 | 15 |

Global pipeline (prompt optimisation):

| File | Iterations | First score | Best score | Last score |
|---|---|---|---|---|
| `global_copa_en_litellm_litellm_iter-80_sample-12.json` | 8 | 0.333 | 0.583 | 0.500 |

### `xcopa_vi_litellm_litellm`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **90.0%**
- Answers the parser could not resolve (`X`): **9**
- Questions where the counterfactual flipped the prediction at least once: **92/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 52 |
| 6 | 44 |
| 8 | 4 |

Global pipeline (prompt optimisation):

| File | Iterations | First score | Best score | Last score |
|---|---|---|---|---|
| `global_xcopa_vi_litellm_litellm_iter-80_sample-12.json` | 8 | 0.500 | 0.667 | 0.417 |

## 3. Error analysis

### `copa_en_litellm_litellm` — 4 incorrect predictions

- Genuinely picked the wrong option: **0**
- Response the parser could not resolve to any option (`X`): **4**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| The glass became full. | X |
| He caught the student cheating. | X |
| He screamed for help. | X |
| He was going to college. | X |

### `xcopa_vi_litellm_litellm` — 10 incorrect predictions

- Genuinely picked the wrong option: **1**
- Response the parser could not resolve to any option (`X`): **9**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Tôi lấy một cuống vé. | Tôi tìm thấy một vũ khí. |
| Cô đã thuê một người quản lý chiến dịch. | X |
| Khán giả reo hò trong sự ngạc nhiên. | X |
| Các sản phẩm được sản xuất thông qua lao động trẻ em. | X |
| Phi công điều hướng khỏi cơn bão. | X |
| Trọng tài ra quyết định sai. | X |
| Người cha nắm lấy tay con trai mình. | X |
| Họ đã nhìn thấy một con cá mập. | X |
| Cô muốn tìm hiểu về các nền văn hóa khác. | X |
| Tôi  hâm nóng nó trong lò vi sóng. | X |

