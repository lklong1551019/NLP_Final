# anhnq — thay tầng đọc kết quả bằng xác suất

Predictor cục bộ, explainer qua OpenAI API. Log từng vòng lặp trong `metrics.jsonl`
của mỗi thư mục run; mỗi lần chạy ghi **đủ cả sáu chỉ số** bất kể `--score_mode` nào
đang được tối ưu, nên các lần chạy so sánh được trực tiếp với nhau.

## Vấn đề

§3.1 của paper: *"estimates S_E from output logits or calibrated probabilities"*.
Theorem 1 đòi `D` là *"any strictly proper divergence"*. Code phát hành tính
`|Δaccuracy|` trên đáp án argmax **parse ra từ văn bản sinh**. Không phải logits,
không phải divergence.

Hệ quả đo được trên XCOPA-vi:

- Parser vứt **46–47%** số câu thành `"X"`. Phi-2 và Qwen3.5-4B cho cùng tỉ lệ →
  lỗi lệch pha prompt–parser, không phải lỗi target. Prompt kết thúc ở
  `### Phản hồi:` mà không nêu định dạng, còn parser đi tìm `[choice]…@`.
- Trên phần giữ lại, parser bất đồng với argmax chuẩn hoá ở **88/150 lượt (59%)**.

## Kết quả chính

Qwen3.5-4B + gpt-4o-mini, 500 câu, tính trên 265 câu baseline đọc được:

| `--score_mode` | vòng/câu | tỉ lệ lật | KTC 95% |
|---|---|---|---|
| `accuracy` (code phát hành) | 2.67 | 0.374 | [0.315, 0.432] |
| `prob_accuracy` (công thức paper, đọc bằng xác suất) | 3.53 | 0.460 | [0.400, 0.520] |
| `logprob` (tín hiệu liên tục) | 3.43 | **0.509** | [0.449, 0.570] |

McNemar so với baseline: **p = 0.00011** (`logprob`), **p = 0.01003** (`prob_accuracy`).

Tập 265 câu **không thiên lệch**: hai chế độ lật với tỉ lệ như nhau trên 235 câu
baseline bỏ qua (p = 0.83).

## Control quan trọng hơn con số chính

`prob_accuracy` vs `logprob`: **p = 0.103** ở n = 500. Mà 500 là toàn bộ test set,
nên đây là **trần cứng**, không phải thiếu mẫu.

**Phần lớn cải thiện đến từ bỏ parser văn bản, không phải từ tín hiệu liên tục.**
Báo cáo con số chính mà không tách phần này là quy công sai chỗ.

## Kiểm tra tính hợp lệ (paper không có)

Tiêm contrary hint viết cho **câu khác**: lật 0.140, so với 0.370 khi dùng hint của
chính câu đó (p < 0.0001, 200 câu). Hiệu ứng có tính ngữ nghĩa — nhưng **nền 0.140
nghĩa là khoảng 1 trong 7 ca lật là nhiễu**, nên mọi tỉ lệ fidelity nên trừ nền
hoặc ít nhất nêu ra.

Đây là biến thể **chặt hơn** control hint-lạc-đề của `anhnh`: control đó dùng câu
cố định ngoài chủ đề (dễ bỏ qua), còn cái này khớp văn phong, độ dài, hình thức, chỉ
khác ở chỗ thuộc về câu hỏi nào. Hai control **xác nhận chéo nhau** dù khác thiết kế
và khác target: nền 0.122 vs 0.140, hiệu ứng đã hiệu chỉnh +0.214 vs +0.230, tỉ lệ
không parse được 45.4% vs 47%.

## Quy tắc dừng

Quy tắc phát hành chỉ kiểm tra điều kiện ở vòng 0/5/10/15, nên câu thoả điều kiện ở
vòng 2 vẫn chạy tới vòng 5: **105/631 vòng (16.6%)** trên 200 câu là chạy thừa.

`--stop_rule` thêm `eager` (cùng điều kiện, kiểm tra mỗi vòng) và `flip` (dừng khi
đáp án lật — sự kiện Algorithm 1 nêu, và là quy tắc **duy nhất** khiến số vòng so
sánh được giữa các chế độ).

## Những gì KHÔNG hiệu quả

Ghi lại thay vì bỏ đi:

- **Metric không giảm số vòng lặp.** Đã thử ba tín hiệu (`prob_shift`, `margin`,
  `flip`), đều trên cùng đường đánh đổi: nhanh hơn thì lật được ít câu hơn. Thứ giảm
  số vòng là **quy tắc dừng**, và hai thứ đó độc lập nhau.
- **Phi-2 không đọc được tiếng Việt**: 0.520 so với sàn đa số 0.530, chọn phương án
  A 87% — thiên lệch vị trí. Các lần chạy dùng nó đo trên nền nhiễu, nên kết quả
  Qwen ở trên thay thế chúng.

## Đang chạy — sẽ bổ sung sau

Một lần chạy nữa chưa xong tại thời điểm merge này, ghi ra đây để không ai chạy trùng
và để người đọc biết vì sao chưa có bảng khớp cấu hình baseline.

**`xcopa_vi_phi_gpt35_matched_{accuracy,prob_accuracy,logprob}`** — 200 câu, 20 bước,
`stop_rule flip`, explainer temp 0.9 / top-p 0.9. Khớp cấu hình baseline của
`anhnh` (Phi-2 + gpt-3.5-turbo, Table 2) trừ cỡ mẫu: 200 thay vì 500, vì ở 20 bước
thì 500 câu mất khoảng 56 giờ.

Nó trả lời câu hỏi mà kết quả chính **không** trả lời được: bảng chính chạy trên
Qwen3.5-4B + gpt-4o-mini, tức đổi metric **và** đổi target **và** đổi explainer cùng
lúc. Lần chạy này giữ nguyên mọi thứ theo baseline và chỉ đổi cách chấm, nên bảng
của nó đặt cạnh bảng của `anhnh` được.

Kỳ vọng nên đặt thấp: cấu hình baseline dùng Phi-2, mà Phi-2 không đọc được tiếng
Việt (xem mục trên), nên nhiều khả năng cả ba chế độ đều ra số nhỏ và sát nhau. Đó
vẫn là kết quả đáng báo cáo — *trên chính cấu hình của paper thì cách chấm mới cho
bao nhiêu* — chỉ là nó không thay thế được bảng Qwen.

Lưu ý sẵn: lần chạy `xcopa_vi_phi_gpt35_{accuracy,logprob}` đã có trong thư mục này
**ngân sách lệch nhau** (3.15 vòng so với 8.40) vì mỗi chế độ dừng theo luật riêng,
nên số của nó không so trực tiếp được. Lần đang chạy dùng `stop_rule flip` cho cả ba
chế độ để sửa đúng chỗ đó.

## Các thư mục run

| Thư mục | Cấu hình |
|---|---|
| `xcopa_vi_qwen_gpt4omini_{accuracy,prob_accuracy,logprob}` | 500 câu, `xai_iter 5`, `stop_rule flip` — **kết quả chính** |
| `xcopa_vi_phi_gpt35_{accuracy,logprob}` | 200 câu, cấu hình baseline; ngân sách lệch nhau nên chỉ dùng tham khảo |
| `copa_en_phi_gpt35_fairbudget_*` | 10 câu × 5 chế độ, `--no_early_stop` — ngân sách bằng nhau để so số vòng |
| `xcopa_vi_qwen_gpt4omini_pilot_*` | 20 câu, pilot |
| `random_hint_control.json` | 200 câu × 3 điều kiện (không hint / hint đúng / hint câu khác) |

Chi phí API đo thật trên toàn bộ: **$5.07** qua 11.334 lượt gọi.

## Tái tạo bảng số

```bash
python scripts/compare_score_modes.py experiments/anhnq
python scripts/iters_to_flip.py experiments/anhnq
python scripts/random_hint_control.py --run_dir experiments/anhnq/xcopa_vi_qwen_gpt4omini_logprob
```
