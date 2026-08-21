# FaithLM — Experiment Report

> Results directory: `./results/experiments/xcopa_vi_xai_sweep_en/qwen-qwen3-7-max`

## 0. Experimental setup

Both FaithLM roles were served by an OpenAI-compatible LiteLLM gateway; no
local model weights were used. See `docs/changelog_2026-08-17.md` §E.

| Role | Model | Temperature | max_tokens |
|---|---|---|---|
| Predictor — initial answer | `deepseek/deepseek-v4-flash` | 0.0 | 200 |
| Predictor — scoring | `deepseek/deepseek-v4-flash` | 0.01 | 200 |
| Explainer | `qwen/qwen3.7-max` | `--temp_exp` | 1000 |
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

- Processes reporting: 2
- Total LLM calls: **1921**
- Calls that failed outright: 0
- Calls the primary model left empty and that were retried away: 0
- Calls served by the **fallback** model: **0** (0.0%)

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `(root)` | 200 | 92.5% | 1.5% | 0.945 | 94.5% | 4.59 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `(root)`

- Questions evaluated: **200**
- Predictor accuracy (no explanation): **92.5%**
- Answers the parser could not resolve (`X`): **3**
- Questions where the counterfactual flipped the prediction at least once: **189/200**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 64 |
| 6 | 117 |
| 8 | 19 |

## 3. Error analysis

### `(root)` — 15 incorrect predictions

- Genuinely picked the wrong option: **12**
- Response the parser could not resolve to any option (`X`): **3**

| Gold answer | Model answer |
|---|---|
| Tôi lấy một cuống vé. | Tôi tìm thấy một vũ khí. |
| Bố mẹ cậu nhốt cậu anh. | Cậu nói dối bố mẹ. |
| Người chủ xích con chó con. | Người chủ mặc cho con chó con một cổ áo. |
| Một cuộc bạo loạn đã nổ ra trước tòa án. | X |
| Tôi lao vào trong. | Cơn bão trở nên tồi tệ hơn. |
| Cái nắp rơi ra khỏi thùng rác. | Có thùng các tông trong thùng rác. |
| Cô bị mất biên lai. | X |
| Tôi mở  bản đồ. | Tôi đếm tiền mặt của tôi. |
| Nó co rúm lại. | Nó đã bị ăn. |
| Tôi đậu gần lối vào. | Tôi đậu xe bên kia đường. |
| Tôi rất tức giận. | Tôi rút phích cắm đèn. |
| Người cha nhẹ nhàng đá đứa bé. | Người cha đã thay tã cho em bé. |
| Họ đã thương lượng một hiệp ước. | X |
| Đứa bé làm bẩn tã của mình. | Đứa bé chảy nước dãi trên chiếc yếm của cô. |
| Tôi hét to tên anh. | Tôi vẫy tay. |

