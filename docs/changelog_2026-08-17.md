# Changelog — 2026-08-17

Refactor toàn diện trên nhánh `feat/faithlm-refactor`. Thay thế bản ghi ngày 2026-08-16.

---

## A. Lỗi ảnh hưởng đến số liệu

### A1. Điểm faithfulness chỉ nhận giá trị 0.0 hoặc 1.0

**Vấn đề.** `_ecqa_score()` gọi `accuracy_score()` trên **một** câu hỏi, nên chỉ trả về 0.0 hoặc 1.0. Hiệu số của hai giá trị như vậy gần như luôn bằng 0, khiến LLM-OPT không có tín hiệu nào để tối ưu — optimizer nhận toàn 0.0 rồi thỉnh thoảng nhảy lên 1.0.

**Sửa.** Chấm bằng log-probability chuẩn hóa theo độ dài mà predictor gán cho từng lựa chọn, rồi lấy softmax để ra P(đáp án đúng). Cho điểm liên tục trong [0,1] chỉ với một forward pass — không tốn thêm thời gian so với sinh văn bản.

Xem `faithlm/predictors.py::choice_logprobs` và `faithlm/metrics.py::_softmax_gold_prob`. Vẫn giữ `scorer: exact_match` cho dữ liệu không có lựa chọn (TriviaQA) và cho predictor qua API.

### A2. Nhánh "giải thích thật" không hề nhận giải thích

**Vấn đề.** Trong `diff_task_score_ecqa()`, biến `true_exp_pair` được `zip()` rồi **vứt bỏ** phần explanation:

```python
ture_final_prompt = [f"...### Input: {ques}..." for _, ques in true_exp_pair]
#                                                    ^^^ exp bị bỏ
```

Nên chỉ số thực tế là `|acc(không gợi ý) − acc(gợi ý đối nghịch)|`, không phải như `docs/architecture.md` mô tả. Nó đo *tác động của việc thêm gợi ý*, lẫn với *tác động của việc đảo nghĩa*.

**Sửa.** Cài hai chỉ số, chọn qua config:

- `metric.name: paper` — giữ nguyên hành vi bản gốc, để tái lập.
- `metric.name: symmetric` — đưa gợi ý vào **cả hai** nhánh, tách bạch tác động đảo nghĩa.

Test `test_metrics_disagree_when_hint_itself_helps` chứng minh hai chỉ số cho kết quả khác nhau trên cùng dữ liệu.

### A3. `zip()` cắt cụt khi explainer trả chuỗi rỗng

**Vấn đề.** `exp_reply.split("\n\n")` có thể ra list rỗng; `zip(exp_reply, question)` khi đó ra rỗng, và cả vòng lặp chấm trên list rỗng mà không báo lỗi.

**Sửa.** `parse_explanations()` luôn trả về ít nhất một phần tử. Có test riêng cho các đầu vào rỗng.

### A4. `preprocess_copa()` gán đè thay vì thêm vào

**Vấn đề.** `train_dict['answer'] = [...]` (gán, không `.append()`) bên trong vòng lặp — mỗi vòng ghi đè toàn bộ danh sách đáp án.

**Sửa.** Toàn bộ loader viết lại, trả về `List[Example]` có kiểu rõ ràng.

---

## B. Lỗi logic

### B1. Điều kiện dừng sớm sai độ ưu tiên toán tử

```python
if iter%5 == 0 and ("apologize" in exp_reply[0]) or ("Unfortunately" in exp_reply[0]):
```

Python đọc thành `(A and B) or C`, nên vế `C` luôn được kiểm tra bất kể `iter%5`. Ngoài ra `iter%5 == 0` chỉ đúng ở vòng 0, 5, 10, 15 — hiếm khi kích hoạt.

**Sửa.** Tách thành hàm `is_refusal()` riêng, kiểm tra ở mọi vòng.

### B2. `try/except` bao cả vòng lặp global nuốt mọi lỗi

**Vấn đề.** `except: continue` bắt tất cả rồi in "API Error". Nếu `updated_xai_prompt` chưa từng được gán, dòng lưu file cuối cùng ném `NameError` — chạy 10 vòng rồi crash lúc lưu, mất sạch kết quả.

**Sửa.** Bắt lỗi ở phạm vi từng câu hỏi, ghi rõ nguyên nhân, và ghi kết quả xuống đĩa sau **mỗi** vòng.

### B3. `pbar.update(1)` nằm sau `break`

Thanh tiến trình luôn thiếu ít nhất một nhịp. Nay dùng `tqdm` bao trực tiếp vòng lặp.

### B4. Model ID không tồn tại

`Qwen/Qwen3.5-4B` không có trên Hugging Face Hub — sẽ crash ngay khi nạp mô hình. Tên đúng: `Qwen/Qwen3-4B-Instruct-2507`.

Đã kiểm chứng `deepseek-v4-pro` và `deepseek-v4-flash` là tên hợp lệ, giữ nguyên.

### B5. `run_experiment.sh` truyền tham số không tồn tại cho `main_global.py`

Script truyền `--ques_idx_start/--ques_idx_end/--save_file_path`, nhưng `main_global.py` không định nghĩa các tham số này → argparse thoát ngay. Nửa sau của script được nêu trong README chưa bao giờ chạy được.

### B6. `run_all_experiments.sh` bỏ qua biến `DATA`

Biến thể `copa_en` vẫn chạy với `--xcopa_lang vi --data_split test`, tức sai split.

### B7. `.gitignore` chứa `*.json` nuốt toàn bộ kết quả

Nay chỉ bỏ qua thư mục `results/`.

### B8. `eval()` trên nội dung do mô hình sinh

`aggregate_results.py` dùng `eval(line)` để đọc file kết quả. Nay pipeline ghi JSON thật và script dùng `json.load()`.

---

## C. Tính năng mới

### C1. Registry + YAML config

Mọi dataset/predictor/explainer/metric đăng ký theo tên. Thêm mô hình mới = thêm một hàm có decorator, thay vì sửa 4 nhánh `if/elif` trải khắp `main_local.py`, `main_global.py`, `predictor.py`, `explainer.py`.

Config sai khóa sẽ báo lỗi ngay thay vì im lặng bỏ qua — lỗi gõ nhầm trong một lần chạy nhiều giờ rất khó phát hiện.

### C2. Chạy tiếp sau khi mất session

Mỗi câu hỏi (local) hoặc mỗi vòng (global) ghi ra một file JSON riêng, ghi theo kiểu atomic (`os.replace`) nên session bị kill không để lại file hỏng. Bật `run.resume` thì lần chạy sau bỏ qua phần đã xong.

Cần thiết vì Kaggle giới hạn 12 giờ/session, trong khi một biến thể đầy đủ mất lâu hơn thế.

### C3. Baseline không dùng LLM

Ba baseline thay thế bước sinh đối nghịch, dùng chung giao diện với explainer:

- `baseline_negation` — phủ định theo luật.
- `baseline_identity` — trả lại nguyên văn; điểm phải tiến về 0, nếu không thì chỉ số đang đo nhiễu prompt chứ không phải đảo nghĩa.
- `baseline_shuffle` — câu ngẫu nhiên không liên quan.

Trả lời câu hỏi: bước sinh đối nghịch có thực sự cần LLM không?

### C4. Notebook cho Colab/Kaggle

`notebooks/FaithLM_Vietnamese.ipynb`, đủ 8 phần theo yêu cầu đồ án, không phụ thuộc đường dẫn cục bộ. Tự phát hiện VRAM để chọn 4-bit hay bf16, tự đọc khóa API từ Kaggle Secrets / Colab userdata.

### C5. Bộ kiểm thử

26 unit test chạy không cần GPU hay API key. Bao gồm test hồi quy cho lỗi so khớp chuỗi với đầu ra Chain-of-Thought dài dòng — chính là lỗi mục 1 trong changelog trước.

---

## E. Port từ thực nghiệm độc lập `final_thesis` (bổ sung cùng ngày)

Các thành phần dưới đây được chuyển từ bộ thực nghiệm chạy tay (Ollama, RTX 3060)
đã có kết quả thật trên ECQA/XCOPA-vi vào kiến trúc registry của package.

### E1. Backend Ollama (`predictor: ollama`, `explainer: ollama`)

Chạy toàn bộ pipeline cục bộ, không cần GPU CUDA hay khóa API. Gọi API native
`/api/chat` với `think: false` và lọc block `<think>` — mô hình reasoning
(Qwen3.5) nếu không sẽ đốt hết token trong phần suy nghĩ. Config mẫu:
`configs/xcopa_vi_ollama.yaml`. Đã kiểm chứng end-to-end trên XCOPA-vi thật.

### E2. Predictor `api` tổng quát

Mọi endpoint OpenAI-compatible làm mô hình đích; khóa và base URL đặt qua
`predictor.api_key_env` / `base_url_env` — thêm nhà cung cấp mới chỉ cần sửa YAML.

### E3. Bộ prompt tiếng Việt (`run.prompt_lang: vi`)

Toàn bộ mẫu prompt (instruction, counterfactual, optimizer local/global) có bản
tiếng Việt, dịch tương đương từ Appendix H/J và đã dùng trong các run thật của
`final_thesis`. Bộ tiếng Anh giữ nguyên văn làm mặc định nên kết quả cũ không đổi;
`variant_id` chỉ thêm hậu tố `_vi` khi dùng pack mới.

### E4. Chỉ số `flip` — flipping answer rate black-box

`1.0` nếu gợi ý đối nghịch làm đổi **đáp án của chính mô hình** (không so với
nhãn vàng). Không cần log-prob → dùng được với predictor API/Ollama và với dữ
liệu không nhãn (phần demo tiếng Việt mới của notebook). Đây chính là "score"
trong trajectory prompt của paper (Figure 13).

### E5. Pipeline `selfcons` — baseline self-consistency

Lấy chuỗi CoT của chính predictor làm lời giải thích rồi chấm cùng quy trình
counterfactual. Đo trực tiếp luận điểm mục 2.2 của paper (CoT không phải giải
thích trung thực); `final_thesis` đo được 0.06 (en) / 0.33 (vi) so với 0.75 /
0.85 của FaithLM.

### E6. Hold-out + transfer eval cho pipeline global

`run.holdout_split` tách tập tối ưu prompt khỏi tập đánh giá (đúng tinh thần
Algorithm 2); sau khi tối ưu, prompt tốt nhất được chấm one-shot trên test và so
với prompt viết tay (`transfer_eval` trong `summary.json`). `final_thesis` cho
thấy hold-out nhỏ thắng trên chính nó nhưng **thua trên test** — khoảng chênh
transfer là con số trung thực cần báo cáo.

---

## D. Đã gỡ bỏ

| File | Thay bằng |
|---|---|
| `main_local.py` | `faithlm/pipelines.py::run_local` |
| `main_global.py` | `faithlm/pipelines.py::run_global` |
| `model/predictor.py` | `faithlm/predictors.py` |
| `model/explainer.py` | `faithlm/explainers.py` |
| `scripts/run_*.sh` (4 file) | `configs/*.yaml` + `run.py` |

Nhánh cũ vẫn còn trên `main` và `origin/refactor-framework` để đối chiếu.
