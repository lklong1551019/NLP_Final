# Tổng kết thực nghiệm

**1.027 lượt-câu · 5.112 vòng lặp tối ưu · 13 commit trên 3 nhánh**

## Các lần chạy có dữ liệu lưu

| Lần chạy | Cấu hình | Quy mô |
|---|---|---|
| `pilot_validate` | Phi-2 + gpt-3.5, 2 chế độ | 5 câu |
| `pilot2` | Phi-2 + gpt-3.5, 4 chế độ | 10 câu |
| `fair` | Phi-2 + gpt-3.5, 5 chế độ, **ngân sách bằng nhau** | 10 câu × 150 vòng |
| `xcopa_vi_200` | Phi-2 + gpt-3.5, 2 chế độ | **200 câu**, 2.312 vòng |
| `pilot_qwen` | Qwen3.5-4B + gpt-4o-mini, 2 chế độ | 20 câu |
| `final_qwen` | Qwen3.5-4B + gpt-4o-mini, 3 chế độ | **200 câu**, 1.678 vòng |

Ngoài ra các phép đo không lưu vào `results/`: đo giá OpenRouter từ API, pilot reasoning-token
(60 lượt gọi), so 4 explainer (32 lượt), walkthrough 6 câu đầy-đủ-pipeline, so 3 cách chấm
trên cùng 6 câu, và ba lần kiểm tra năng lực model (Phi-2 trên XCOPA-vi và COPA-en, Qwen3.5-4B
trên XCOPA-vi, mỗi lần 100 câu) — chạy cục bộ nên miễn phí.

## Kết quả chính, theo độ chắc chắn

### Tất định — không phụ thuộc thống kê

| Phát hiện | Số liệu |
|---|---|
| Quy tắc `iter%5` chạy thừa sau khi đã đủ điều kiện dừng | **105/631 vòng = 16.6%** (n=200) |
| Parser khớp chuỗi tuyệt đối lệch so với argmax chuẩn hoá | **88/150 lượt = 59%** |
| Qwen3.5-4B: parser của paper không đọc được output | **106/200 câu = 53%** ra `"X"` |
| Chấm bằng xác suất thay vì sinh 256 token | **~20×** nhanh hơn (88 ms/câu) |
| §3.1 nói *"logits or calibrated probabilities"*, code dùng khớp chuỗi rời rạc | nguyên văn |
| Theorem 1 đòi *"strictly proper divergence"*, `\|Δacc\|` không phải | nguyên văn |
| `qwen3.8-max` không tắt được reasoning → completion rỗng | lỗi 400 nguyên văn |
| GPT-5.6 không nhận `temperature`/`top_p` → không tái lập được Table 2 | lỗi 400 nguyên văn |

### Đo được, có ý nghĩa thống kê

**Qwen3.5-4B + gpt-4o-mini, 200 câu XCOPA-vi, `xai_iter 5`, `stop_rule flip`:**

| | vòng/câu | tỉ lệ lật (94 câu chung) | KTC 95% |
|---|---|---|---|
| `accuracy` (baseline) | 2.67 | 0.330 | [0.235, 0.425] |
| `prob_accuracy` (control) | 3.64 | 0.415 | [0.315, 0.514] |
| `logprob` | 3.50 | **0.489** | [0.388, 0.590] |

- `accuracy` vs `logprob`: **p = 0.0051**, có ý nghĩa
- `prob_accuracy` vs `logprob`: p = 0.0637, **chưa có ý nghĩa** → phần lớn cải thiện đến từ
  **bỏ parser text**, không phải từ tín hiệu liên tục
- Số vòng chênh **0.8** (2.67 → 3.50), so với 2.7× trước khi sửa quy tắc dừng

### Đã bác bỏ

| Giả thuyết | Bằng chứng |
|---|---|
| Metric liên tục giảm số vòng lặp | Phi-2 + gpt-3.5, n=200: tăng 2.7× (3.15 → 8.40); khớp ngân sách thì lật **ít hơn** (0.416 vs 0.649) |
| Tín hiệu `margin` (khoảng cách tới ranh giới) cứu được | n=10 ngân sách bằng: 5/10 lật, 2.20 vòng — vẫn trên cùng đường đánh đổi |
| Phi-2 dùng được làm target tiếng Việt | 0.520 so với sàn đa số 0.530; chọn A 87% |

## Năng lực model đã đo

| Model | XCOPA-vi accuracy | Vượt sàn | ‖Δp‖ khi tiêm hint |
|---|---|---|---|
| Qwen3.5-4B | **0.730** | **+0.200** | **0.0208** |
| Phi-2 | 0.520 | −0.010 | 0.0051 |
| Phi-2 trên COPA-en | 0.750 | +0.220 | 0.0203 |

## Đường chặn đã gỡ

| Vấn đề | Cách xử lý |
|---|---|
| `Qwen3_5Config has no attribute vocab_size` | Dùng `AutoModelForImageTextToText`; checkpoint khai `ForConditionalGeneration`, `vocab_size` nằm trong `text_config` |
| `cannot import name HybridCache` — không import nổi module nào | Bỏ import chết `PREFIX_CHECKPOINT_DIR` (kéo theo `peft`) |
| `No module named anthropic` | Cho `anthropic`/`ipdb` thành import tuỳ chọn |
| Completion rỗng bị chấm điểm âm thầm | Raise thay vì trả chuỗi |
| Gateway Qualgo chặn Cloudflare 403 | VPN cert-based, `AUTH_FAILED` — chưa gỡ |

## Chi phí

| | |
|---|---|
| OpenRouter | **$0.19** (đo trực tiếp từ API) |
| OpenAI | ~$3 (ước tính — backend `openai` chưa ghi usage log, đây là lỗ hổng nên vá) |
