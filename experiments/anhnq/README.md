# anhnq — thay tầng đọc kết quả bằng xác suất

| File | Nội dung |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | **Kết luận và toàn bộ số liệu.** Nguồn duy nhất — viết tay. |
| `*/` | Kết quả thô, một JSON mỗi câu, kèm `metrics.jsonl` và `usage.jsonl`. |
| `random_hint_control.json` | Control kiểm tra tính hợp lệ, 200 câu × 3 điều kiện. |

Cách chấm được chọn bằng `--score_mode`; cơ chế mô tả trong
[`docs/metric_experiment.md`](../../docs/metric_experiment.md).

## Các lần chạy

Predictor chạy cục bộ, explainer qua OpenAI API trực tiếp (`--xai_model openai`).

| Thư mục | N | Cấu hình |
|---|---|---|
| `xcopa_vi_qwen_gpt4omini_{accuracy,prob_accuracy,logprob}` | 500 | Qwen3.5-4B bf16 + gpt-4o-mini, 5 bước, `stop_rule flip` — **kết quả chính** |
| `xcopa_vi_phi_gpt35_{accuracy,logprob}` | 200 | Phi-2 + gpt-3.5-turbo, 15 bước. Ngân sách hai chế độ **lệch nhau** (3.15 vs 8.40 vòng) nên không so trực tiếp được; giữ vì là nguồn của phép đo `iter%5` |
| `xcopa_vi_phi_gpt56luna_accuracy` | 179 | GPT-5.6 Luna, dừng vì completion rỗng — giữ làm bằng chứng lỗi |

## Điểm khác biệt so với các phần khác trong nhóm

Mọi lần chạy ghi **đủ sáu chỉ số** mỗi vòng lặp bất kể `--score_mode` nào đang được
tối ưu. Nếu không có điều đó thì mỗi chế độ chỉ được đo bằng chính mục tiêu của nó,
và câu hỏi "chế độ nào tốt hơn" không có nghĩa vì các con số ở thang khác nhau.

## Tái tạo bảng số

```bash
python scripts/compare_score_modes.py experiments/anhnq
python scripts/iters_to_flip.py experiments/anhnq
```
