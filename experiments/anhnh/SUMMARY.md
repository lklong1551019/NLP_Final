# FaithLM — Experiment Report

> Results directory: `./experiments/anhnh`

<!-- inlined from FINDINGS.md - edit that file, not this one -->

## Findings

Predictor `microsoft/phi-2`, explainer `gpt-3.5-turbo` — the paper's own models
(§4.2). 500 instances, Table 2 COPA column. One repetition, not three.

### 1. The paper reproduces

Fidelity **0.872**, against the ~0.85 reported for COPA in Figure 3.

### 2. But the metric sits at its ceiling

The contrary hint flipped **every** parseable prediction — all 354 correct
answers and all 34 incorrect ones score exactly 1.0000. A measure at its maximum
cannot rank explanations, and is equally consistent with "the target model just
follows whatever hint it is given".

### 3. About two thirds of that is suggestibility

The paper has no control for this. Scoring a third prompt per instance carrying
a hint unrelated to the question ("The weather forecast mentions scattered
clouds tomorrow afternoon.") separates the two:

| | copa_en | xcopa_vi |
|---|---|---|
| Accuracy | 70.8% | 11.8% |
| Unparseable (`X`) | 22.4% | 45.4% |
| Fidelity | 0.860 | 0.336 |
| Irrelevant hint | 0.554 | 0.122 |
| **Corrected** | **+0.306** | **+0.214** |
| Mean iterations | 3.11 | 7.82 |

The measure is not empty — the contrary explanation beats an irrelevant sentence
by +0.306 — but 64% of the raw 0.860 is the model following any hint at all.

### 4. In Vietnamese the parser fails, not the metric

Balanced COPA test and XCOPA-vi test are the **same 500 items**, so language is
the only variable. Splitting by whether the predictor answered at all reverses
the aggregate picture:

| Group | copa_en corrected | xcopa_vi corrected |
|---|---|---|
| Correct prediction | +0.4011 (n=354) | **+0.4746** (n=59) |
| Incorrect prediction | +0.2647 (n=34) | +0.3224 (n=214) |
| Unparseable (`X`) | +0.0179 (n=112) | +0.0441 (n=227) |

Where the model produces an answer the corrected signal **survives the language
change and is larger**. The drop from +0.306 to +0.214 is driven almost entirely
by the 227 unparseable instances, which contribute +0.044.

Vietnamese accuracy is 11.8% (95% CI [9.3%, 14.9%]); over the 273 parseable
answers it is 21.6% ([17.1%, 26.9%]). Chance on a two-choice task is 50%, well
outside both, so Phi-2 picks the wrong option systematically rather than
guessing. The random-hint control collapses too (0.554 → 0.122), consistent with
a model that cannot read the hint either.

This points at generate-then-parse scoring as the defect, not at the fidelity
measure — which is what likelihood-based scoring would fix.

### 5. Cost scales inversely with success

Vietnamese needed 7.82 optimisation iterations per question against 3.11, and
~82s against ~19s. Early stop fires on the first non-zero score, so when the
measurement yields nothing every instance runs to the 20-iteration cap. The
paper's Limitations section mentions the carbon cost of iterating, but not that
the cost is highest exactly where the framework works least.

### Caveats

- **One repetition.** The paper averages three; no variance is reported here.
  That is the same gap we criticise in `docs/paper_limitations.md` (L10).
- **Aggregation changes the answer.** Mean over every scoring event on the
  500-instance run is 0.4707; maximum per question is 0.8720. The latter is what
  compares to the paper. Tables here use maximum per question throughout.
- **`diff_score` carries one bit per instance.** Accuracy over a single example
  is 0 or 1. A 30-instance estimate put Vietnamese accuracy at 33.3% with a
  confidence interval that still contained chance; the 500-instance value is
  11.8%. Nothing here is safe to conclude from small samples.

---

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
| `copa_en_phi_gpt35_control` | 500 | 70.8% | 22.4% | 0.860 | 0.554 | **+0.306** | 86.0% | 3.11 |
| `copa_en_phi_gpt35_greedy` | 30 | 86.7% | 13.3% | 0.767 | not measured | not measured | 76.7% | 2.47 |
| `copa_en_phi_gpt35_paper_rep1` | 500 | 70.4% | 22.6% | 0.872 | not measured | not measured | 87.2% | 3.14 |
| `xcopa_vi_phi_gpt35_control` | 500 | 11.8% | 45.4% | 0.336 | 0.122 | **+0.214** | 33.6% | 7.82 |
| `xcopa_vi_phi_gpt35_greedy` | 30 | 33.3% | 40.0% | 0.233 | not measured | not measured | 23.3% | 3.00 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `copa_en_phi_gpt35_control`

- Questions evaluated: **500**
- Predictor accuracy (no explanation): **70.8%**
- Answers the parser could not resolve (`X`): **112**
- Questions where the counterfactual flipped the prediction at least once: **430/500**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 306 |
| 6 | 180 |
| 11 | 11 |
| 16 | 3 |

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

### `xcopa_vi_phi_gpt35_control`

- Questions evaluated: **500**
- Predictor accuracy (no explanation): **11.8%**
- Answers the parser could not resolve (`X`): **227**
- Questions where the counterfactual flipped the prediction at least once: **168/500**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 275 |
| 6 | 48 |
| 11 | 18 |
| 16 | 8 |
| 20 | 151 |

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

### `copa_en_phi_gpt35_control` — 146 incorrect predictions

- Genuinely picked the wrong option: **34**
- Response the parser could not resolve to any option (`X`): **112**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| I retrieved a ticket stub. | X |
| A drop of blood formed on my finger. | X |
| It was dead. | X |
| His parents grounded him. | X |
| The owner kept the puppy on a leash. | The owner put a collar on the puppy. |
| I called her back. | I met her for dinner. |
| Their dog ran away from home. | Expensive jewelry was missing from their home. |
| She wore high heels. | X |
| I wore sandals. | I wore boots. |
| The couple eloped. | The couple got pregnant. |
| My bank account was empty. | X |
| A riot broke loose in front of the courthouse. | X |
| She left the cookies in the oven. | X |
| I dashed to get inside. | X |
| They disappointed their fans. | X |
| … | … (131 more) |

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

### `xcopa_vi_phi_gpt35_control` — 441 incorrect predictions

- Genuinely picked the wrong option: **214**
- Response the parser could not resolve to any option (`X`): **227**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Nó dễ vỡ. | Các bắt dòng về hướng trong điều phòng. |
| Tôi lấy một cuống vé. | Tôi lấy một thấy đường. |
| Cố vấn trại của họ kể cho họ một câu chuyện ma. | Số trường yêu phản hồi của họ kể cho nơi trên lửa trại. |
| Điện trong khu phố đã tắt. | Mãy việc lớn của các hàng xây trường. |
| Anh xúc phạm khán giả. | X |
| Một giọt máu hình thành trên ngón tay của tôi. | Chúng ta để hướng dẫn. |
| Nó đã chết. | Nó đáp án. |
| Thang máy đã đến tầng được chỉ định. | Đủ đồng bị ở một thời gian. |
| Bố mẹ cậu nhốt cậu anh. | X |
| Nước nhà vệ sinh tràn ra. | Nước hỏi những giá trình về máy bị hỏng. |
| Người chủ xích con chó con. | Vạn bắp rời được giao. |
| Anh đứng trên mặt hồ tĩnh lặng. | Mô tả không một tây âm năm ảnh hoàn vũ. |
| Anh ấy bị chấn động. | X |
| Tôi gọi lại cho cô ấy. | X |
| Con chó của họ bỏ nhà ra đi. | Đồ trang sức đắt tiền đã bị mất khỏi nhà gồm của tranh. |
| … | … (426 more) |

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

