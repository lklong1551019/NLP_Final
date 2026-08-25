# anhnq — thay tầng đọc kết quả bằng xác suất

| File | Nội dung |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | **Kết luận và toàn bộ số liệu.** Nguồn duy nhất — viết tay. |
| `*/` | Kết quả thô, một JSON mỗi câu, kèm `metrics.jsonl` và `usage.jsonl`. |
| `random_hint_control.json` | Control kiểm tra tính hợp lệ, 200 câu × 3 điều kiện. |

Cách chấm được chọn bằng `--score_mode`; cơ chế mô tả trong
[`docs/metric_experiment.md`](../../docs/metric_experiment.md).

## Không có `SUMMARY.md`

Các thành viên khác sinh `SUMMARY.md` bằng `scripts/build_report.py`. Ở đây thì
không, vì script tính cột "Mean faithfulness" bằng cách lấy trung bình trường
`Score` — mà `Score` là **đại lượng mà chế độ đó đang tối ưu**, và các chế độ tối ưu
những thứ ở thang khác nhau: `accuracy` và `prob_accuracy` cho `|Δacc| ∈ {0,1}`, còn
`logprob` cho dịch chuyển xác suất với biên độ khoảng 0.1.

Đặt cạnh nhau, bảng đó cho `logprob` **0.104** so với `accuracy` **0.234** — tức
trông như chế độ chính tệ nhất, trong khi cột "Questions with any flip" của chính
bảng đó cho **88.6%** so với **23.4%**, ngược hẳn lại. Chú thích của script cũng
khẳng định điểm số "is therefore 0 or 1 per iteration", đúng với chế độ của nhóm
nhưng sai với `logprob` và `margin`.

Một bảng đọc ngược kết luận thì nguy hiểm hơn là không có bảng. Số liệu nằm trong
[`FINDINGS.md`](FINDINGS.md), so bằng **tỉ lệ lật** — đại lượng duy nhất giống nhau ở
mọi chế độ.

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
