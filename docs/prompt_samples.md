# Tài Liệu: Kiến Trúc FaithLM & Mẫu Prompt Tối Ưu Hóa (OPRO)

Tài liệu này giải thích chi tiết luồng hoạt động của FaithLM, các mô hình tham gia trong từng bước (Explainer vs Predictor/Target LLM), và cung cấp các ví dụ thực tế về cách Prompt tiến hóa qua các vòng lặp (0 hint, 1 hint, 2 hint).

---

## I. Tổng Quan Các Mô Hình Tham Gia

Trong FaithLM, có hai mô hình làm việc song song với các vai trò tách biệt:
1. **Target LLM (Predictor)** (VD: Qwen, Vicuna): Đóng vai trò là người trả lời câu hỏi. Nhiệm vụ của nó là đọc Prompt (có hoặc không có gợi ý) và đưa ra đáp án Multiple Choice. Chúng ta coi nó là một "Hộp Đen" (Black-box) và muốn tìm hiểu xem vì sao nó chọn đáp án đó.
2. **Explainer LLM** (VD: DeepSeek, Claude, GPT-4): Đóng vai trò là nhà phân tích. Nhiệm vụ của nó là sinh ra (hoặc tối ưu hóa) các câu giải thích `<EXP>` hoặc lệnh hướng dẫn `<INS>` để phỏng đoán tư duy của Target LLM.

---

## II. Các Bước Trong Vòng Lặp Tối Ưu (Local Pipeline)

Dưới đây là chi tiết những gì diễn ra ở mỗi câu hỏi khi bạn chạy `main_local.py`:

### Bước 1: Tạo Lời Giải Thích Ban Đầu (Initial True Explanation)
- **Mô hình tham gia:** Explainer LLM (DeepSeek)
- **Input:** Câu hỏi + Đáp án gốc mà Target LLM đã chọn.
- **Output:** Một câu giải thích phỏng đoán lý do (ví dụ: `<EXP>Nó chọn đáp án A vì...</EXP>`).

### Bước 2.1: Tạo Câu Trái Ngược (Counterfactual)
- **Mô hình tham gia:** Explainer LLM (DeepSeek)
- **Input:** Câu giải thích đúng ở Bước 1.
- **Output:** Một câu giải thích mang ý nghĩa ngược lại (nhằm đánh lừa mô hình).

### Bước 2.2: Đánh Giá Điểm Số (Score Calculation)
- **Mô hình tham gia:** Target LLM (Qwen)
- **Input:** (Câu hỏi + Gợi ý đúng) **VÀ** (Câu hỏi + Gợi ý trái ngược).
- **Output:** Điểm số `diff_score = Score(True) - Score(Counterfactual)`. Điểm cao (gần 1.0) chứng tỏ lời giải thích đúng có tác động mạnh mẽ đến Target LLM, tức là nó có độ trung thành cao.

### Bước 2.3: Tối Ưu Hóa (LLM Optimizer / OPRO)
- **Mô hình tham gia:** Explainer LLM (DeepSeek)
- **Input:** Lịch sử các câu giải thích ở các vòng trước kèm điểm số + Lệnh yêu cầu viết câu mới tốt hơn.
- **Output:** Lời giải thích mới và sắc bén hơn `<EXP>...</EXP>`. (Sau đó lặp lại từ Bước 2.1).

---

## III. Quá Trình Tiến Hóa Của Prompt (LLM Optimizer)

Ở Bước 2.3, Prompt gửi cho **Explainer LLM** sẽ thay đổi sau mỗi vòng lặp vì nó cần đọc lịch sử để học hỏi (In-context learning). Dưới đây là mô phỏng Prompt với 0, 1 và 2 bản ghi lịch sử (hints).

### 1. Vòng lặp đầu tiên (0 Hint / Không có lịch sử)
Lúc này, chưa có điểm số nào được tính. Prompt sẽ chỉ chứa phần hướng dẫn (meta instruction) và không có bất kỳ ví dụ lịch sử nào.

```text
Tôi có một số văn bản cùng với điểm số tương ứng của chúng. Các văn bản này là lời giải thích có thể có cho câu hỏi và câu trả lời đã cho dưới đây. Các văn bản được sắp xếp theo thứ tự ngẫu nhiên dựa trên điểm số của chúng, trong đó điểm cao hơn cho thấy chất lượng tốt hơn. Điểm số được tính bằng mức độ liên quan của các văn bản đối với câu hỏi và câu trả lời đã cho như là lời giải thích. Điểm số dao động từ 0 đến 1 dựa trên văn bản đầu ra của bạn.

### Đầu vào:
['### Câu hỏi: Nguyên nhân của Tiền đề là gì?\n### Tiền đề: Các mặt hàng đã được đóng gói trong bọc bong bóng.\n### Lựa chọn: [choice]Nó dễ vỡ.@ [choice]Nó nhỏ.@']
### Trả lời: ['Nó dễ vỡ.']

Các ví dụ sau đây cho thấy cách áp dụng văn bản của bạn: Bạn thay thế <EXP> bằng văn bản của mình. Chúng tôi nói đầu ra của bạn là tệ nếu đầu ra của bạn đạt điểm thấp hơn văn bản trước đó, và chúng tôi nói đầu ra của bạn là tốt nếu đầu ra của bạn đạt điểm cao hơn văn bản trước đó. Đầu ra phải bắt đầu bằng <EXP>.

(Danh sách lịch sử đang trống rỗng)

Vui lòng cung cấp văn bản khách quan mới để mô tả lý do tại sao các câu trả lời được đưa ra cho các câu hỏi dựa trên suy nghĩ của bạn. Đoán lý do dù đúng hay sai. Đảm bảo không tự trả lời các câu hỏi hoặc cung cấp bất kỳ đề xuất nào để trả lời các câu hỏi tốt hơn. Mọi lời giải thích phải bắt đầu bằng <EXP>. Đảm bảo không lặp lại các câu hỏi và câu trả lời đầu vào. Vui lòng chỉ xuất ra các câu giải thích.
```

### 2. Vòng lặp thứ hai (1 Hint / 1 Bản ghi lịch sử)
Explainer LLM đã sinh ra được 1 câu, và câu đó đã được Target LLM chấm điểm. Lịch sử được chèn vào giữa prompt.

```text
... (phần meta instruction và input giống như trên) ...

Văn bản:
<EXP>Mô hình suy luận rằng bọc bong bóng thường được sử dụng để bảo vệ các vật phẩm, vì vậy nó chọn “Nó dễ vỡ”.</EXP>
Điểm số:
0.0

Vui lòng cung cấp văn bản khách quan mới để mô tả lý do tại sao các câu trả lời được đưa ra cho các câu hỏi dựa trên suy nghĩ của bạn... (như trên)
```

### 3. Vòng lặp thứ ba (2 Hints / 2 Bản ghi lịch sử)
Explainer LLM sẽ thấy được cả câu thất bại (điểm thấp) và câu cải tiến (điểm cao hơn) để đối chiếu, từ đó tìm ra quy luật để sinh ra câu thứ 3 xuất sắc hơn.

```text
... (phần meta instruction và input giống như trên) ...

Văn bản:
<EXP>Mô hình suy luận rằng bọc bong bóng thường được sử dụng để bảo vệ các vật phẩm, vì vậy nó chọn “Nó dễ vỡ”.</EXP>
Điểm số:
0.0

Văn bản:
<EXP>Bọc bong bóng chống va đập, nên món hàng chắc chắn là dễ vỡ chứ không phải vì nó nhỏ.</EXP>
Điểm số:
0.5

Vui lòng cung cấp văn bản khách quan mới để mô tả lý do tại sao các câu trả lời được đưa ra cho các câu hỏi dựa trên suy nghĩ của bạn... (như trên)
```
