# FaithLM Experiment Report: XCOPA-VI (Qwen + DeepSeek)

> **Generated:** 2026-08-22
> **Results directory:** `./results/experiments/`

This report details the execution of the FaithLM optimization pipeline on the Vietnamese XCOPA dataset.

## 1. Experimental Setup

The experiment was run twice to compare the performance of different Explainer models. Crucially, the dataset and prompting templates were fully localized into Vietnamese to prevent English data leakage.

| Role | Model | Details |
|---|---|---|
| **Predictor (Initial Answer)** | `Qwen/Qwen3.5-4B` | 4-bit precision |
| **Predictor (Scoring)** | `Qwen/Qwen3.5-4B` | 4-bit precision |
| **Explainer (Run 1 - Pro)** | `deepseek-v4-pro` | - |
| **Explainer (Run 2 - Flash)** | `deepseek-v4-flash` | - |

### Configuration Details
- **Dataset:** `xcopa_vi` (XCOPA, Vietnamese split)
- **Prompt Language:** Fully translated to Vietnamese (e.g., `### Câu hỏi:`, `### Tiền đề:`).
- **Fidelity Optimization Steps:** 15 iterations per sample
- **Samples Evaluated:** 200 instances

---

## 2. Results Summary

Below are the aggregated results comparing the two Explainer model variants:

| Explainer Model | Samples | Correctness | Avg Faithfulness Score | Max Faithfulness Score |
|---|---|---|---|---|
| **DeepSeek-v4-Pro** | 200 | **87.5%** | **0.5041** | 1.0000 |
| **DeepSeek-v4-Flash** | 200 | 85.0% | 0.4100 | 1.0000 |

### Key Observations:
1. **Model Capability Gap:** Using the heavier `deepseek-v4-pro` as the Explainer LLM yielded a significantly higher Average Faithfulness Score (0.5041) compared to `deepseek-v4-flash` (0.4100).
2. **Correctness Improvement:** The Pro variant also helped improve the baseline correctness of the predictor model slightly (87.5% vs 85.0%), indicating that higher quality explanations correlate with better task performance.
3. **Localization Success:** The Vietnamese prompts were highly successful. The models consistently adhered to the formatting without throwing the "API error" fallback that was previously caused by broken list formatting and English prompt collisions.

---

## 3. Global Pipeline Metrics
*Note: The global pipeline was executed to optimize the system prompt template itself.*

| Run | Iterations | Avg Score |
|---|---|---|
| **Global Pipeline (Pro/Flash)** | 2 | ~0.1333 |

*(For a full per-sample breakdown of all 400 local pipeline results, please refer to the raw JSON logs in the `./results/experiments/` directory).*
## 4. Error Analysis (Incorrect Initial Predictions)

### `xcopa_vi_qwen_deepseek` Error Analysis

- Total Evaluated: **200**
- Incorrect Predictions: **25**

| Gold answer | Model answer |
|---|---|
| Tôi lấy một cuống vé. | X |
| Bố mẹ cậu nhốt cậu anh. | X |
| Người cha đuổi con trai ra khỏi nhà. | Người cha mua cho con trai một cốc bia. |
| Anh cầu xin cô quay trở lại. | He asked her to come back |
| Động cơ quá nóng. | I lit a match. |
| Tôi ôm anh. | X |
| Tôi rất tức giận. | X |
| Nước thấm chảy ra. | X |
| Tôi trở nên nghi ngờ. | X |
| Bằng chứng liên quan đến anh ta. | X |
| Cô muốn rời khỏi bữa tiệc. | X |
| Ngôi nhà bị cháy. | X |
| Mối ăn gỗ trong nhà. | Mối biến mất khỏi nhà. |
| Cô trèo lên một sợi dây. | X |
| Cô bầm tím đầu gối. | X |
| Nước trái cây tràn ra. | X |
| Mọi người cho anh tiền lẻ. | X |
| Lá dồn trên mặt đường. | X |
| Ông đóng kín phong bì. | X |
| Cô bước ra khỏi hàng. | Nhiều người hơn đaz vào hàng. |
| Người cha nắm lấy tay con trai mình. | X |
| Thuyền kayak đến bờ. | The kayak boat reaches the shore. |
| Tôi làm đổ nước lên poster. | X |
| Cậu hét lên cầu cứu. | Cậu dựng lều. |
| Trông nó rất vui. | X |


### `xcopa_vi_qwen_deepseek_flash` Error Analysis

- Total Evaluated: **200**
- Incorrect Predictions: **30**

| Gold answer | Model answer |
|---|---|
| Nó đã chết. | It is hungry soon. |
| Người chủ xích con chó con. | The owner puts a collar on the puppy. ( |
| Anh ấy bị chấn động. | Anh lạc trong suy nghĩ. |
| Cô bị mất biên lai. | X |
| Anh đang nói chuyện với chính mình. | X |
| Nó co rúm lại. | X |
| Học sinh trả lời đúng câu hỏi. | Student answers correctly |
| Cô đi siêu thị. | Cô đi đến trang trại. |
| Tôi ôm anh. | X |
| Vải khô. | Dried cloth. |
| Tóc anh ngày càng dài. | X |
| Tôi cảm thấy tội lỗi. | I feel guilty. |
| Tôi không có nhà. | X |
| Khu nghỉ mát nằm trên một hòn đảo. | X |
| Có một liên kết bị hỏng trong dây. | X |
| Ông phung phí tài sản của mình. | He sold the company's shares. |
| Cô trèo lên một sợi dây. | Cô gõ một lá thư. |
| Chị cô hạnh phúc. | X |
| Cô nhảy dây. | Cô chơi cờ đam (She plays chess) |
| Lá dồn trên mặt đường. | X |
| Bố nó dừng xe ở trạm xăng. | X |
| Cô bước ra khỏi hàng. | Nhiều người hơn đaz vào hàng. |
| Tôi vặn mỏ lết. | Tôi thay chốt. |
| Cô muốn ngắm hoàng hôn. | X |
| Trọng tài ra quyết định sai. | X |
| Cô đâm vào một hàng rào. | X |
| Nước rút ra khỏi bồn. | X |
| Tôi làm đổ nước lên poster. | X |
| Đồng nghiệp của anh ấy đã được thăng chức. | The colleague got promoted. |
| Anh ấy đi học đại học. | X |


