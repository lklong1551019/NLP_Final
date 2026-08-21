# Báo cáo: thực nghiệm sửa metric fidelity

Nhánh `feat/logprob-fidelity-metric` · 200 câu đầu XCOPA-vi · target Phi-2 (cục bộ), explainer gpt-3.5-turbo

## Kết luận về mục tiêu được giao

**Sửa metric KHÔNG giảm được số vòng lặp.** Đã thử ba tín hiệu (`prob_shift`, `margin`, `flip`), đều rơi vào cùng một đánh đổi: nhanh hơn thì lật được ít câu hơn.

| | vòng/câu | tỉ lệ lật |
|---|---|---|
| baseline (code paper) | 3.15 | 0.26 |
| metric liên tục | 8.40 | 0.39 |

Số thô trông như thắng (McNemar p=0.0018) nhưng **ngân sách lệch 2.7 lần**. Khớp ngân sách lại thì đảo chiều: cùng 3 vòng đầu, baseline lật 0.649 còn metric mới 0.416.

Nguyên nhân: metric tối ưu *độ dịch chuyển xác suất*, còn điều kiện dừng lại là *vượt ranh giới quyết định*. Hai đại lượng khác nhau. Thêm nữa, gpt-3.5-turbo diễn đạt lại cùng một câu qua các vòng nên optimizer không có phương án khác biệt để chọn — mắt xích yếu là explainer, không phải metric.

## Thứ thật sự giảm số vòng lặp

Không phải công thức tính điểm mà là **quy tắc dừng**. Code paper chỉ kiểm tra điều kiện dừng ở vòng 0, 5, 10, 15:

```python
if iter%5 == 0 and sum(scores_list) != 0:
```

Lật ở vòng 2 thì vẫn chạy tới vòng 5. Trên 200 câu: **105/631 vòng là chạy thừa = −16.6%**, 43 câu thừa từ 1–4 vòng. Sửa một dòng, tất định, không đánh đổi chất lượng.

## Bốn điều cả nhóm cần biết

1. **Parser của baseline sai lệch 59%.** `accuracy_score` khớp chuỗi tuyệt đối trên đoạn cắt thô giữa `]` và `@`, không `strip()`. Bất đồng với argmax chuẩn hoá ở **88/150 lượt chấm**. Ảnh hưởng tới số liệu của mọi người, kể cả bản baseline.

2. **Phi-2 không đọc được tiếng Việt.** XCOPA-vi: accuracy 0.520 so với sàn đa số 0.530, chọn phương án A 87% (thiên lệch vị trí, không phải hiểu). Trên COPA-en thì +0.220 trên sàn. Ai dùng `phi` trên `xcopa_vi` cần biết.

3. **Code không khớp với chính lý thuyết của paper.** §3.1 nguyên văn: *"estimates S_E from output logits or calibrated probabilities"*. Theorem 1 đòi `D` là *"any strictly proper divergence"*. Code dùng `|Δaccuracy|` trên argmax rời rạc — không phải divergence. Nên bảo đảm của Theorem 1 không áp dụng cho đại lượng code thực sự đo.

4. **Chấm bằng xác suất nhanh ~20×** — 88 ms/câu thay vì sinh 256 token. Đáng dùng cho mọi biến thể nếu target chạy cục bộ.

## Đề xuất định vị lại

Không nên trình bày là *"metric mới của tôi tốt hơn"* — kết quả không đỡ được. Nên trình bày là:

> Hiện thực hoá đúng công thức mà §3.1 và Theorem 1 của paper mô tả, rồi đo xem điều đó thay đổi kết luận ra sao. Kết quả: không giảm số vòng lặp, nhưng phát hiện phép đo gốc lệch 59% và không khớp với định lý của chính nó.

Một kết quả âm về mệnh đề chính + bốn phát hiện tất định. Đủ nội dung cho một phần thực nghiệm và không có chỗ nào để reviewer bắt lỗi.

## Việc tiếp theo

- Nếu muốn theo tiếp hướng metric: cần đổi explainer (gpt-4o-mini đảo nghĩa sạch hơn rõ rệt, rẻ hơn 3×) và đổi target sang model đọc được tiếng Việt. Metric hiện tại đang đo trên nền nhiễu.
- Nếu muốn kết quả chắc ngay: chốt phần sửa quy tắc dừng + parser, đó là hai cải tiến tất định.

## Chi phí đã dùng

gpt-3.5-turbo: khoảng $2 cho toàn bộ pilot và lần chạy 200 câu.
