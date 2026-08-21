# Dữ liệu thô của thực nghiệm metric

Mỗi file là `metrics.jsonl` của một lần chạy: một dòng cho mỗi vòng lặp của mỗi câu.
Mọi lần chạy đều ghi **đủ cả sáu chỉ số** bất kể `--score_mode` nào đang được tối ưu,
nên các lần chạy so sánh được trực tiếp với nhau.

| File | Lần chạy |
|---|---|
| `xcopa_vi_200__accuracy.jsonl` | 200 câu XCOPA-vi, baseline (code paper), early stop bật |
| `xcopa_vi_200__logprob.jsonl` | 200 câu XCOPA-vi, metric liên tục, early stop bật |
| `fair__*.jsonl` | 10 câu COPA-en, 5 chế độ, `--no_early_stop` (ngân sách bằng nhau) |

Cấu hình chung: target Phi-2 cục bộ, explainer gpt-3.5-turbo, tối đa 15 vòng,
temperature 0.9 / top-p 0.9 (Table 2 của paper).

Tái tạo bảng số:

```bash
python scripts/compare_score_modes.py results/xcopa_vi_200
python scripts/iters_to_flip.py results/xcopa_vi_200
```

## Trường dữ liệu

| Trường | Ý nghĩa |
|---|---|
| `question_idx`, `iter` | câu thứ mấy, vòng thứ mấy |
| `optimised_score` | giá trị mà optimizer thực sự đi theo ở chế độ đó |
| `accuracy_parsed` | `\|Δacc\|` của code phát hành (sinh 256 token rồi cắt chuỗi, khớp tuyệt đối) |
| `accuracy` | cùng công thức nhưng từ argmax xác suất, có chuẩn hoá |
| `prob_shift` | `P_trước(a₀) − P_sau(a₀)`, có dấu |
| `margin` | `0.5 − P_sau(a₀)`, khoảng cách vượt ranh giới |
| `tv` | total variation giữa hai phân phối |
| `flip` | argmax có đổi không — sự kiện duy nhất giống nhau ở mọi chế độ |
| `p_before`, `p_after` | xác suất đặt lên lựa chọn ban đầu, trước và sau khi tiêm hint |
| `pred_before`, `pred_after` | lựa chọn target chọn, trước và sau |

`accuracy_parsed` và `accuracy` bất đồng ở **88/150 lượt (59%)** — đây là bằng chứng
cho phần nói về lỗi parser trong báo cáo.
