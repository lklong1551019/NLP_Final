# FaithLM — Experiment Report

> Results directory: `./experiments/anhnh`

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

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Random-hint control | Corrected | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|---|---|
| `copa_en_phi_gpt35_control` | 84 | 73.8% | 20.2% | 0.869 | 0.607 | **+0.262** | 86.9% | 3.32 |
| `copa_en_phi_gpt35_greedy` | 30 | 86.7% | 13.3% | 0.767 | not measured | not measured | 76.7% | 2.47 |
| `copa_en_phi_gpt35_paper_rep1` | 500 | 70.4% | 22.6% | 0.872 | not measured | not measured | 87.2% | 3.14 |
| `xcopa_vi_phi_gpt35_greedy` | 30 | 33.3% | 40.0% | 0.233 | not measured | not measured | 23.3% | 3.00 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `copa_en_phi_gpt35_control`

- Questions evaluated: **84**
- Predictor accuracy (no explanation): **73.8%**
- Answers the parser could not resolve (`X`): **17**
- Questions where the counterfactual flipped the prediction at least once: **73/84**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 50 |
| 6 | 29 |
| 11 | 5 |

### `copa_en_phi_gpt35_greedy`

- Questions evaluated: **30**
- Predictor accuracy (no explanation): **86.7%**
- Answers the parser could not resolve (`X`): **4**
- Questions where the counterfactual flipped the prediction at least once: **23/30**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 19 |
| 5 | 11 |

### `copa_en_phi_gpt35_paper_rep1`

- Questions evaluated: **500**
- Predictor accuracy (no explanation): **70.4%**
- Answers the parser could not resolve (`X`): **113**
- Questions where the counterfactual flipped the prediction at least once: **436/500**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 307 |
| 6 | 177 |
| 11 | 13 |
| 16 | 1 |
| 20 | 2 |

### `xcopa_vi_phi_gpt35_greedy`

- Questions evaluated: **30**
- Predictor accuracy (no explanation): **33.3%**
- Answers the parser could not resolve (`X`): **12**
- Questions where the counterfactual flipped the prediction at least once: **7/30**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 15 |
| 5 | 15 |

## 3. Error analysis

### `copa_en_phi_gpt35_control` — 22 incorrect predictions

- Genuinely picked the wrong option: **5**
- Response the parser could not resolve to any option (`X`): **17**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| I took a detour. | X |
| The mother gave birth to twins. | X |
| She went to the library. | X |
| Their house caught fire. | X |
| The can got crushed. | X |
| I slammed the door upon leaving the house. | X |
| The paper creased. | X |
| Leaders of other countries sent emergency relief. | Leaders of other countries formed an alliance. |
| It was due to be returned to the library. | He borrowed it from a friend. |
| He was convicted of murder. | X |
| Her wig came off. | She went bald. |
| He deemed the sentence unclear. | X |
| She realized the card was missing. | X |
| The girl ruffled it. | X |
| I stepped on the bug. | __. |
| … | … (7 more) |

### `copa_en_phi_gpt35_greedy` — 4 incorrect predictions

- Genuinely picked the wrong option: **0**
- Response the parser could not resolve to any option (`X`): **4**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| The glass became full. | X |
| I ran out of breath. | X |
| I put it in the microwave. | X |
| He wanted a day off. | X |

### `copa_en_phi_gpt35_paper_rep1` — 148 incorrect predictions

- Genuinely picked the wrong option: **35**
- Response the parser could not resolve to any option (`X`): **113**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| It was fragile. | X |
| A drop of blood formed on my finger. | X |
| His parents grounded him. | He lied to his parents. |
| The owner kept the puppy on a leash. | X |
| They wanted better working conditions. | X |
| She wore high heels. | X |
| I wore sandals. | X |
| The couple eloped. | X |
| She left the cookies in the oven. | She took the cookies out of the jar. |
| I dashed to get inside. | X |
| The fire alarm went off. | X |
| The elevator was out of order. | X |
| He begged her to take him back. | X |
| Their heads collided. | X |
| The lid was off the garbage can. | X |
| … | … (133 more) |

### `xcopa_vi_phi_gpt35_greedy` — 20 incorrect predictions

- Genuinely picked the wrong option: **8**
- Response the parser could not resolve to any option (`X`): **12**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Nó dễ vỡ. | X |
| Anh ấy bị chấn động. | Anh lạc trong suy nghĩ. |
| Tài khoản ngân hàng của tôi trống rỗng. | X |
| Ông ấy xóa thư rác. | X |
| Anh đã tiếp xúc với dịch bệnh. | X |
| Chiếc ly trở nên đầy. | Nước làm dịu cơn khát của tôi. |
| Ông ấy đang nghĩ về những lời nói của bạn mình. | X |
| Có một liên kết bị hỏng trong dây. | Dây xích bị đứt. |
| Mối ăn gỗ trong nhà. | Mối biến mất khỏi nhà. |
| Cô đã thuê một người quản lý chiến dịch. | X |
| Cậu ấy giơ tay. | X |
| Anh ấy bắt được học sinh gian lận. | X |
| Tôi hết hơi. | X |
| Anh ta muốn một ngày nghỉ. | Anh ta bị đau bụng. |
| Các nhân viên tuần tra kiểm tra hộ chiếu của họ. | X |
| … | … (5 more) |

