# FaithLM — Experiment Report

> Results directory: `./experiments/minhndn`

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

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `copa_en_phi2_gemini35_promptEN` | 200 | 72.0% | 19.0% | 0.860 | 86.0% | 3.63 |
| `xcopa_vi_dsflash_gemini35_promptEN` | 200 | 92.5% | 2.0% | 0.930 | 93.0% | 4.47 |
| `xcopa_vi_dsflash_gpt56luna_promptEN` | 200 | 93.0% | 1.0% | 0.955 | 95.5% | 4.45 |
| `xcopa_vi_dsflash_qwen37max_promptEN` | 200 | 92.5% | 1.5% | 0.945 | 94.5% | 4.59 |
| `xcopa_vi_phi2_gemini35_promptEN` | 200 | 41.5% | 41.5% | 0.755 | 75.5% | 2.58 |
| `xcopa_vi_phi2_gemini35_promptVI_n20` | 20 | 10.0% | 55.0% | 0.300 | 30.0% | 3.50 |
| `xcopa_vi_qwen35_gemini35_promptEN` | 200 | 90.5% | 6.5% | 0.945 | 94.5% | 3.04 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `copa_en_phi2_gemini35_promptEN`

- Questions evaluated: **200**
- Predictor accuracy (no explanation): **72.0%**
- Answers the parser could not resolve (`X`): **38**
- Questions where the counterfactual flipped the prediction at least once: **172/200**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 100 |
| 6 | 87 |
| 8 | 13 |

### `xcopa_vi_dsflash_gemini35_promptEN`

- Questions evaluated: **200**
- Predictor accuracy (no explanation): **92.5%**
- Answers the parser could not resolve (`X`): **4**
- Questions where the counterfactual flipped the prediction at least once: **186/200**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 70 |
| 6 | 108 |
| 8 | 22 |

### `xcopa_vi_dsflash_gpt56luna_promptEN`

- Questions evaluated: **200**
- Predictor accuracy (no explanation): **93.0%**
- Answers the parser could not resolve (`X`): **2**
- Questions where the counterfactual flipped the prediction at least once: **191/200**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 68 |
| 6 | 117 |
| 8 | 15 |

### `xcopa_vi_dsflash_qwen37max_promptEN`

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

### `xcopa_vi_phi2_gemini35_promptEN`

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

### `xcopa_vi_phi2_gemini35_promptVI_n20`

- Questions evaluated: **20**
- Predictor accuracy (no explanation): **10.0%**
- Answers the parser could not resolve (`X`): **11**
- Questions where the counterfactual flipped the prediction at least once: **6/20**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 12 |
| 6 | 3 |
| 8 | 5 |

### `xcopa_vi_qwen35_gemini35_promptEN`

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

### `copa_en_phi2_gemini35_promptEN` — 56 incorrect predictions

- Genuinely picked the wrong option: **18**
- Response the parser could not resolve to any option (`X`): **38**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| It was fragile. | X |
| A drop of blood formed on my finger. | X |
| It was dead. | X |
| His parents grounded him. | X |
| He stood over the calm lake. | X |
| I called her back. | X |
| The weather was chilly. | My chest felt tight. |
| She wore high heels. | X |
| I wore sandals. | I wore boots. |
| The couple eloped. | X |
| The sales associate saw the girl put merchandise in her purse. | X |
| I dashed to get inside. | The storm worsened. |
| The elevator was out of order. | X |
| The lid was off the garbage can. | X |
| He was talking to himself. | He was studying for an exam. |
| … | … (41 more) |

### `xcopa_vi_dsflash_gemini35_promptEN` — 15 incorrect predictions

- Genuinely picked the wrong option: **11**
- Response the parser could not resolve to any option (`X`): **4**

| Gold answer | Model answer |
|---|---|
| Tôi lấy một cuống vé. | Tôi tìm thấy một vũ khí. |
| Bố mẹ cậu nhốt cậu anh. | Cậu nói dối bố mẹ. |
| Tôi lao vào trong. | Cơn bão trở nên tồi tệ hơn. |
| Tôi mở  bản đồ. | Tôi đếm tiền mặt của tôi. |
| Nó co rúm lại. | X |
| Tôi đậu gần lối vào. | Tôi đậu xe bên kia đường. |
| Tôi rất tức giận. | Tôi rút phích cắm đèn. |
| Người cha nhẹ nhàng đá đứa bé. | Người cha đã thay tã cho em bé. |
| Tôi không có nhà. | Tôi cô đơn. |
| Cô đã thuê một người quản lý chiến dịch. | Cô làm chứng trước tòa. |
| Phi công điều hướng khỏi cơn bão. | X |
| Đứa bé làm bẩn tã của mình. | Đứa bé chảy nước dãi trên chiếc yếm của cô. |
| Người cha nắm lấy tay con trai mình. | X |
| Thuyền kayak đến bờ. | X |
| Họ đã nhìn thấy một con cá mập. | Họ ướt đẫm. |

### `xcopa_vi_dsflash_gpt56luna_promptEN` — 14 incorrect predictions

- Genuinely picked the wrong option: **12**
- Response the parser could not resolve to any option (`X`): **2**

| Gold answer | Model answer |
|---|---|
| Tôi lấy một cuống vé. | Tôi tìm thấy một vũ khí. |
| Bố mẹ cậu nhốt cậu anh. | Cậu nói dối bố mẹ. |
| Người chủ xích con chó con. | Người chủ mặc cho con chó con một cổ áo. |
| Tôi mở  bản đồ. | Tôi đếm tiền mặt của tôi. |
| Tôi đậu gần lối vào. | Tôi đậu xe bên kia đường. |
| Tôi rất tức giận. | Tôi rút phích cắm đèn. |
| Người cha nhẹ nhàng đá đứa bé. | Người cha đã thay tã cho em bé. |
| Tôi không có nhà. | Tôi cô đơn. |
| Tôi hết hơi. | Tôi mất giọng. |
| Phi công điều hướng khỏi cơn bão. | X |
| Đứa bé làm bẩn tã của mình. | Đứa bé chảy nước dãi trên chiếc yếm của cô. |
| Người cha nắm lấy tay con trai mình. | X |
| Tôi hét to tên anh. | Tôi vẫy tay. |
| Tôi cài đặt lại bộ ngắt mạch. | Tôi bật đèn lên. |

### `xcopa_vi_dsflash_qwen37max_promptEN` — 15 incorrect predictions

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

### `xcopa_vi_phi2_gemini35_promptEN` — 117 incorrect predictions

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

### `xcopa_vi_phi2_gemini35_promptVI_n20` — 18 incorrect predictions

- Genuinely picked the wrong option: **7**
- Response the parser could not resolve to any option (`X`): **11**

> Most 'errors' are parsing failures, not reasoning failures. What this
> pipeline calls *accuracy* is closer to a **parse-success rate**, and it
> degrades on Vietnamese relative to English. Any accuracy figure taken
> from FaithLM should be read with that in mind.

| Gold answer | Model answer |
|---|---|
| Nó dễ vỡ. | Hãy không đầu vào. |
| Tôi lấy một cuống vé. | X |
| Cố vấn trại của họ kể cho họ một câu chuyện ma. | Không như họ lời đến đấy. |
| Anh ấy bị chấn động. | X |
| Tài khoản ngân hàng của tôi trống rỗng. | X |
| Anh đã tiếp xúc với dịch bệnh. | X |
| Cô tự đọc nhẩm lại. | X |
| Chiếc ly trở nên đầy. | Nước làm dịu cơn khát của tôi. |
| Ông ấy đang nghĩ về những lời nói của bạn mình. | Ông ấy muốn hỗ trợ bạn mình. |
| Tôi cảm thấy tội lỗi. | X |
| Có một liên kết bị hỏng trong dây. | Tình dây đã để lại báo. |
| Các nhân viên tuần tra kiểm tra hộ chiếu của họ. | X |
| Đang vào đợt nghỉ. | X |
| Cô nhảy dây. | X |
| Cô bước ra khỏi hàng. | Hãy hãy bị một nhiều người tốt bản trường từ khỏi hàng. |
| … | … (3 more) |

### `xcopa_vi_qwen35_gemini35_promptEN` — 19 incorrect predictions

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

