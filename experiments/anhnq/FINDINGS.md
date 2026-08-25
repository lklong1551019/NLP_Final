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

## Lần chạy khớp cấu hình baseline — dừng giữa chừng vì hết credit

`xcopa_vi_phi_gpt35_matched_*` — 200 câu, 20 bước, `stop_rule flip`, Phi-2 +
gpt-3.5-turbo, khớp cấu hình baseline của `anhnh` trừ cỡ mẫu. Dừng ở lượt 360/600 vì
tài khoản OpenAI hết credit (`429 credit_balance_exhausted`), **không phải lỗi code**.

| Chế độ | Đã chấm | vòng/câu |
|---|---|---|
| `accuracy` | 109/200 (91 câu bị parser bỏ) | 9.07 |
| `prob_accuracy` | 160/200 | 14.76 |
| `logprob` | **0 — chưa kịp chạy** | — |

Trên 87 câu cả hai chế độ cùng có: `accuracy` 0.402, `prob_accuracy` 0.391,
McNemar **p = 1.00**. Không phân biệt được.

**Bảng này chưa dùng để kết luận được**, vì hai lý do:

1. Thiếu `logprob` — đúng chế độ chính cần so.
2. Cấu hình baseline dùng Phi-2, mà Phi-2 không đọc được tiếng Việt (xem mục trên),
   nên cả hai chế độ đều đo trên nền nhiễu. Kết quả p = 1.00 phù hợp với dự đoán đã
   ghi trước khi chạy, chứ không phải phát hiện mới.

Muốn hoàn tất cần thêm khoảng **$4.90** (40 câu `prob_accuracy` + 200 câu `logprob` ở
$0.0204/câu). Kết quả chính trên Qwen3.5-4B **không phụ thuộc bảng này**.

### Chi phí đo được

| Lần chạy | Model | Lượt gọi | Chi phí |
|---|---|---|---|
| `matched200` | gpt-3.5-turbo | 6.933 | **$5.50** |
| `full500` | gpt-4o-mini | 5.493 | $0.81 |
| `luna200` | gpt-5.6-luna | 577 | $0.13 |
| | | **13.003** | **$6.43** |

`matched200` chiếm 85% tổng chi phí vì ba yếu tố nhân nhau: 20 bước thay vì 5 (4×),
gpt-3.5-turbo thay vì gpt-4o-mini (3.3× mỗi token), và `stop_rule flip` hiếm khi kích
hoạt trên Phi-2 tiếng Việt nên thực tế chạy 9–15 vòng/câu thay vì ~3. Tổng cộng đắt
hơn khoảng 13 lần mỗi câu so với các lần chạy trước.

Các lần chạy sớm hơn (`final_qwen`, `xcopa_vi_200`, các pilot) không có usage log nên
chi phí của chúng không nằm trong bảng — usage log chỉ được thêm về sau.

## Các thư mục run

| Thư mục | Cấu hình |
|---|---|
| `xcopa_vi_qwen_gpt4omini_{accuracy,prob_accuracy,logprob}` | 500 câu, `xai_iter 5`, `stop_rule flip` — **kết quả chính** |
| `xcopa_vi_phi_gpt35_{accuracy,logprob}` | 200 câu, cấu hình baseline; ngân sách lệch nhau nên chỉ dùng tham khảo |
| `copa_en_phi_gpt35_fairbudget_*` | 10 câu × 5 chế độ, `--no_early_stop` — ngân sách bằng nhau để so số vòng |
| `xcopa_vi_qwen_gpt4omini_pilot_*` | 20 câu, pilot |
| `random_hint_control.json` | 200 câu × 3 điều kiện (không hint / hint đúng / hint câu khác) |
| `xcopa_vi_phi_gpt56luna_accuracy` | 179 câu, GPT-5.6 Luna — xem ghi chú bên dưới |
| `copa_en_phi_gpt35_pilot4mode_*` | 10 câu × 4 chế độ, pilot thiết kế |

### Ghi chú về `xcopa_vi_phi_gpt56luna_accuracy`

Lần chạy này định thử explainer mạnh hơn. Chế độ `accuracy` chạy được 179 câu rồi
`logprob` chết ngay với `RuntimeError: gpt-5.6-luna returned empty content`.

Nguyên nhân: GPT-5.6 tiêu hết `max_completion_tokens` vào reasoning, không còn chỗ
cho nội dung. Khi test một câu ngắn thì `reasoning_tokens = 0` nên không lộ; dưới
tải thật với prompt tiếng Việt dài thì nó reasoning nhiều hơn. Cách tắt là
`reasoning_effort="none"` — đã kiểm chứng: reasoning về 0, nội dung đầy đủ, và output
giảm từ 145 xuống 72 token nên rẻ hơn.

Đáng chú ý là lá chắn đã hoạt động đúng: backend **raise** thay vì âm thầm chấm điểm
trên chuỗi rỗng. Nếu không có nó thì sẽ có 200 câu dữ liệu rác trông như hợp lệ —
đúng lỗi mà `mignh` cũng gặp và vá bằng `split_reply()`.

Chưa chạy lại vì `mignh` đã có nhánh sweep GPT-5.6 riêng; giữ lại phần này làm bằng
chứng cho lỗi, không phải làm kết quả.


Chi phí API đo thật trên toàn bộ: **$5.07** qua 11.334 lượt gọi.

## Tái tạo bảng số

```bash
python scripts/compare_score_modes.py experiments/anhnq
python scripts/iters_to_flip.py experiments/anhnq
python scripts/random_hint_control.py --run_dir experiments/anhnq/xcopa_vi_qwen_gpt4omini_logprob
```
