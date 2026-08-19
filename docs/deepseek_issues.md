# Báo Cáo: Các Vấn Đề (Issues) Khi Sử Dụng DeepSeek Trong Pipeline FaithLM

Tài liệu này tổng hợp các vấn đề và lỗi hành vi phát sinh khi sử dụng DeepSeek làm mô hình **Explainer** (thay thế cho Claude/GPT-4 trong bài báo gốc), kèm theo các ví dụ thực tế thu thập được trong quá trình chạy thử nghiệm.

---

## 1. Lỗ Hổng "Hack Phần Thưởng" (Reward Hacking) và Ảo Giác Học Bậy

Đây là vấn đề nghiêm trọng nhất liên quan đến **In-Context Learning**. Khi DeepSeek gặp lỗi (ví dụ: timeout) và trả về một chuỗi rỗng `""` (empty string), nếu pipeline vô tình ghi nhận chuỗi rỗng này vào lịch sử và gán cho nó điểm số cao (ví dụ: `1.0`), DeepSeek sẽ học được một quy luật sai lệch ở các vòng lặp tiếp theo.

**Ví dụ thực tế đã xảy ra (Log):**
DeepSeek nhận được Prompt chứa lịch sử sau:
```text
Văn bản:

Điểm số:
1.0

Văn bản:

Điểm số:
1.0
```
**Phân tích của DeepSeek:** "À, hóa ra để được điểm tuyệt đối (1.0), mình chỉ cần trả về một chuỗi trống trơn không có chữ nào!".
**Hành vi lỗi:** Từ đó trở đi, DeepSeek liên tục trả về `''` (empty string), dẫn đến việc kích hoạt cơ chế Retry liên tục vô ích và làm sụp đổ toàn bộ quá trình tối ưu.

> [!TIP]
> **Giải pháp (Đã áp dụng):** Khi DeepSeek thất bại trong việc sinh ra câu giải thích (trả về chuỗi rỗng), thay vì lưu chuỗi rỗng vào bộ nhớ, script nay sử dụng lệnh `break` hoặc `continue` để hủy bỏ hoàn toàn vòng lặp hiện tại, giữ cho lịch sử (Prompt history) luôn sạch.

---

## 2. Bỏ Quên Thẻ (Tags) Quy Định `<EXP>`

Mặc dù trong Prompt có lệnh cực kỳ rõ ràng: *"Mọi lời giải thích phải bắt đầu bằng `<EXP>`"*. Tuy nhiên, DeepSeek đôi khi bỏ qua lệnh này hoặc tự ý thêm các ký tự thừa như dấu nháy đơn `'...'`.

**Ví dụ thực tế (Log):**
```text
[DEEPSEEK RESPONSE]
'Mô hình liên hệ việc dùng bọc bong bóng với nhu cầu chống va đập, nên coi đặc tính dễ vỡ là nguyên nhân hợp lý hơn.'
```
**Hậu quả:** 
DeepSeek bọc câu trả lời trong dấu nháy đơn `''` và hoàn toàn KHÔNG CÓ thẻ `<EXP>`. 
Trong pipeline gốc, script cắt chuỗi bằng `.split(":\n\n")[-1]`. Việc DeepSeek không tuân thủ định dạng khiến việc tách (parse) câu trả lời trở nên rủi ro, đôi khi Predictor không hiểu được đâu là phần giải thích.

---

## 3. Cắt Ngang Câu Trả Lời (Generation Truncation)

Đôi khi, API của DeepSeek trả về một câu bị cắt ngang một cách khó hiểu (có thể do lỗi tokenization hoặc lỗi mạng từ phía máy chủ).

**Ví dụ thực tế (Log):**
```text
[DEEPSEEK REQUEST]
Vui lòng tạo một ví dụ mang ý nghĩa trái ngược với câu đã cho...

[DEEPSEEK RESPONSE]
'Mô hình su'
```
**Hậu quả:**
Câu giải thích "Mô hình su" hoàn toàn vô nghĩa. Khi đưa câu này vào cho Qwen (Predictor) làm gợi ý (Hint), Qwen sẽ bị bối rối và trả lời sai (điểm số bị sụt giảm không đáng có).

---

## 4. Quá "Nhiệt Tình" (Chatty Behavior)

Bài báo OPRO yêu cầu mô hình sinh ra **đúng một câu duy nhất** (Ví dụ: Đối với Counterfactual, prompt ghi rõ *"Đảm bảo bạn chỉ xuất ra câu, không kèm giải thích thêm"*).
Tuy nhiên, vì DeepSeek là mô hình Chat (Instruction-tuned), đôi khi nó rất thích nói chuyện theo kiểu:

**Ví dụ hành vi:**
```text
[DEEPSEEK RESPONSE]
Dưới đây là câu trái ngược bạn yêu cầu:
<EXP>Kích thước nhỏ mới là nguyên nhân chính chứ không phải dễ vỡ.</EXP>
Chúc bạn một ngày tốt lành!
```
**Hậu quả:**
Sự dư thừa văn bản này làm thay đổi cấu trúc của Hint khi đưa vào Predictor, gây xáo trộn khả năng dự đoán của Qwen (Qwen nhạy cảm vô cùng với các ký tự lạ hoặc định dạng lệch chuẩn).

> [!IMPORTANT]
> **Kết Luận Tổng Quan:**
> DeepSeek rất mạnh về logic ngôn ngữ tự nhiên, nhưng lại bộc lộ sự thiếu ổn định về mặt **tuân thủ định dạng khắt khe (Strict formatting adherence)**. Khi kết hợp với vòng lặp OPRO (vốn rất nhạy cảm với dữ liệu rác), các lỗi nhỏ của DeepSeek có xu hướng khuếch đại (snowball) thành sự sụp đổ của toàn bộ pipeline nếu không có các lớp bảo vệ bằng code Python (`try-except`, `continue`, `break` và hàm làm sạch chuỗi) cực kỳ cứng rắn.
