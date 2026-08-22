# Báo cáo: thực nghiệm sửa metric fidelity

Nhánh `feat/logprob-fidelity-metric` · dữ liệu thô trong `docs/evidence/`

> **Cập nhật 22/08.** Bản trước kết luận "metric không cải thiện gì". Kết luận đó dựa trên
> target Phi-2, mà đo lại cho thấy Phi-2 **không đọc được tiếng Việt** (0.520 so với sàn đa
> số 0.530, chọn phương án A 87% — thiên lệch vị trí, không phải hiểu). Toàn bộ phép đo khi
> đó nằm trên nền nhiễu. Chạy lại với Qwen3.5-4B cho kết quả khác hẳn.

## Kết quả chính

**Qwen3.5-4B (cục bộ) + gpt-4o-mini, 200 câu XCOPA-vi, prompt tiếng Việt, `xai_iter 5`, `stop_rule flip`**

| | vòng/câu | tỉ lệ lật | KTC 95% |
|---|---|---|---|
| `accuracy` — code phát hành | 2.67 | 0.330 | [0.235, 0.425] |
| `prob_accuracy` — công thức paper, tính sạch | 3.64 | 0.415 | [0.315, 0.514] |
| `logprob` — đúng thứ §3.1 mô tả | 3.50 | **0.489** | [0.388, 0.590] |

Tính trên 94 câu mà baseline đọc được — cách thuận cho baseline nhất.

- `accuracy` vs `logprob`: **McNemar p = 0.0051**, có ý nghĩa. 20 câu chỉ `logprob` lật, 5 câu chỉ baseline lật.
- **Số vòng gần như không tăng**: 2.67 → 3.50, chênh 0.8 vòng.

## Nhưng control nói phần lớn công không thuộc về metric

`prob_accuracy` vs `logprob`: **p = 0.0637, chưa có ý nghĩa**.

`prob_accuracy` dùng **đúng công thức của paper**, chỉ khác ở chỗ đọc từ xác suất thay vì
parse văn bản — và nó đã đạt 0.415 so với baseline 0.330. Tín hiệu liên tục đẩy thêm lên
0.489 nhưng phần thêm đó chưa vượt ngưỡng ý nghĩa.

Nói cách khác: **cải thiện chủ yếu đến từ việc bỏ parser text, không phải từ tín hiệu liên tục.**
Không có chế độ control này thì sẽ quy nhầm toàn bộ 0.330 → 0.489 cho metric mới.

## Về mục tiêu "giảm số vòng lặp"

Không đạt bằng metric. Nhưng đạt bằng **quy tắc dừng**, và đó là hai thứ độc lập:

Code paper chỉ kiểm tra điều kiện dừng ở vòng 0, 5, 10, 15:

```python
if iter%5 == 0 and sum(scores_list) != 0:
```

Lật ở vòng 2 thì vẫn chạy tới vòng 5. Trên 200 câu: **105/631 vòng chạy thừa = 16.6%**,
43 câu thừa 1–4 vòng. Sửa một dòng, tất định, không đánh đổi chất lượng.

Trước khi sửa quy tắc dừng, `logprob` chạy 8.40 vòng so với baseline 3.15 — tăng 2.7 lần.
Sau khi đổi sang `stop_rule flip` và hạ `xai_iter` xuống 5: 3.50 so với 2.67.

## Bốn điều cả nhóm cần biết

1. **Parser của baseline sai lệch 59%** — `accuracy_score` khớp chuỗi tuyệt đối trên đoạn cắt
   thô giữa `]` và `@`, không `strip()`. Bất đồng với argmax chuẩn hoá ở 88/150 lượt chấm.

2. **Với Qwen3.5-4B, parser bỏ qua 53% dữ liệu** — 106/200 câu ra `"X"` vì Qwen sinh văn bản
   suy luận không khớp `[choice]…@`. Phương pháp của paper không dùng được với target hiện đại.

3. **Phi-2 không đọc được tiếng Việt** — ai dùng `phi` trên `xcopa_vi` cần biết. Trên COPA-en
   thì nó bình thường (+0.220 trên sàn).

4. **Code không khớp lý thuyết của chính paper** — §3.1 nguyên văn *"estimates S_E from output
   logits or calibrated probabilities"*; Theorem 1 đòi `D` là *"any strictly proper divergence"*.
   `|Δaccuracy|` trên argmax rời rạc không phải divergence.

## Cách định vị đóng góp

> Hiện thực hoá đúng công thức mà §3.1 và Theorem 1 của paper mô tả. Kết quả: tỉ lệ lật tăng
> từ 0.330 lên 0.489 (p = 0.0051) với số vòng lặp gần như không đổi. Phân tách bằng chế độ
> control cho thấy phần lớn cải thiện đến từ việc thay parser văn bản bằng argmax xác suất,
> chứ không phải từ tín hiệu liên tục.

Kèm bốn phát hiện tất định ở trên, không phụ thuộc thống kê.

## Việc tiếp theo

- **Vá lỗ hổng đo chi phí**: backend `openai` chưa ghi usage log (backend OpenRouter thì có).
  Chi phí OpenAI ~$3 hiện là ước tính, không phải số đo.
- **Thử explainer mạnh hơn**: dòng GPT-5.6 vừa ra (`luna` $0.20/$1.20, `terra` $2/$12,
  `sol` $4/$20). Giả thuyết chưa kiểm chứng là explainer mới là mắt xích yếu — cả gpt-3.5 lẫn
  gpt-4o-mini đều diễn đạt lại cùng một nội dung qua các vòng thay vì tạo phương án khác biệt.
  Lưu ý GPT-5.6 **không nhận `temperature`/`top_p`**, nên không tái lập được Table 2 của paper.
- **Tăng cỡ mẫu cho control**: `prob_accuracy` vs `logprob` đang ở p = 0.0637. Với 500 câu
  (toàn bộ test set) may ra kết luận được phần đóng góp riêng của tín hiệu liên tục.

## Chi phí

OpenRouter **$0.19** (đo trực tiếp). OpenAI ~**$3** (ước tính).
