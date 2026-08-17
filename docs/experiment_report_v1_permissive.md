# FaithLM — Experiment Report

> Results directory: `./results/experiments_v1_permissive_parser`

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
- Total LLM calls: **4811**
- Calls that failed outright: 0
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **957** (19.9%)

> The fallback share is not incidental: a non-trivial fraction of the
> Explainer's output came from a different model than the one named above,
> because the primary returned an empty body for prompts its content filter
> rejected. Vietnamese prompts triggered this far more often than English ones.

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `copa_en_litellm_litellm` | 100 | 98.0% | 2.0% | 0.810 | 81.0% | 5.02 |
| `xcopa_vi_litellm_litellm` | 100 | 90.0% | 10.0% | 0.880 | 88.0% | 3.98 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `copa_en_litellm_litellm`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **98.0%**
- Answers the parser could not resolve (`X`): **2**
- Questions where the counterfactual flipped the prediction at least once: **81/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 28 |
| 6 | 51 |
| 8 | 21 |

Global pipeline (prompt optimisation):

| File | Iterations | First score | Best score | Last score |
|---|---|---|---|---|
| `global_copa_en_litellm_litellm_iter-80_sample-12.json` | 8 | 0.167 | 0.500 | 0.500 |

### `xcopa_vi_litellm_litellm`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **90.0%**
- Answers the parser could not resolve (`X`): **10**
- Questions where the counterfactual flipped the prediction at least once: **88/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 46 |
| 6 | 40 |
| 8 | 14 |

Global pipeline (prompt optimisation):

| File | Iterations | First score | Best score | Last score |
|---|---|---|---|---|
| `global_xcopa_vi_litellm_litellm_iter-80_sample-12.json` | 8 | 0.500 | 0.500 | 0.250 |

## 3. Error analysis

### `copa_en_litellm_litellm` — 2 incorrect predictions

- Genuinely picked the wrong option: **0**
- Response the parser could not resolve to any option (`X`): **2**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Juice spilled out. | X |
| She crashed into a fence. | X |

### `xcopa_vi_litellm_litellm` — 10 incorrect predictions

- Genuinely picked the wrong option: **0**
- Response the parser could not resolve to any option (`X`): **10**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Cố vấn trại của họ kể cho họ một câu chuyện ma. | X |
| Đối thủ của cô buộc tội cô gian lận. | X |
| Cậu quyết định chọc em gái mình. | X |
| Bố nó dừng xe ở trạm xăng. | X |
| Ông đóng kín phong bì. | X |
| Trọng tài ra quyết định sai. | X |
| Khán giả vỗ tay theo điệu nhạc. | X |
| Tôi hét to tên anh. | X |
| Cô gái mang cho cô giáo một quả táo. | X |
| Ông ấy bỏ thuốc tẩy lên tóc. | X |

