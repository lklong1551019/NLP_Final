# FaithLM — Experiment Report

> Results directory: `./results/experiments/phi2_xcopa_vi_en_vertexgemini`

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

- Processes reporting: 1
- Total LLM calls: **425**
- Calls that failed outright: 0
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **0** (0.0%)

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `(root)` | 100 | 46.0% | 40.0% | 0.750 | 75.0% | 2.71 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `(root)`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **46.0%**
- Answers the parser could not resolve (`X`): **40**
- Questions where the counterfactual flipped the prediction at least once: **75/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 67 |
| 6 | 30 |
| 8 | 3 |

## 3. Error analysis

### `(root)` — 54 incorrect predictions

- Genuinely picked the wrong option: **14**
- Response the parser could not resolve to any option (`X`): **40**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Nó dễ vỡ. | X |
| Tôi lấy một cuống vé. | Tôi lụy một cuống vé. |
| Anh ấy bị chấn động. | Anh lạc trong suy nghĩ. |
| Tài khoản ngân hàng của tôi trống rỗng. | X |
| Ông ấy đang nghĩ về những lời nói của bạn mình. | Ông ấy muốn hỗ trợ bạn mình. |
| Có một liên kết bị hỏng trong dây. | Dây xích bị quấn quanh một lốp xe. |
| Mối ăn gỗ trong nhà. | Mối biến mất khỏi nhà. |
| Cô đã thuê một người quản lý chiến dịch. | X |
| Anh ấy bắt được học sinh gian lận. | X |
| Tôi hết hơi. | Tôi mất giọng. |
| Anh ta muốn một ngày nghỉ. | X |
| Cô nhảy khỏi bảng lặn. | X |
| Đó là ngày khai mạc cho bộ phim. | X |
| Mọi người khen anh. | Mọi người cô lập anh. |
| Bạn tôi làm tôi chờ. | X |
| … | … (39 more) |

