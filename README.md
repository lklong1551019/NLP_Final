# FaithLM cho tiếng Việt

Tái lập và mở rộng **FaithLM: Towards Faithful Explanations for Large Language Models** (EACL 2026) trên **XCOPA tiếng Việt**.

Một giải thích được coi là *trung thực* nếu việc đảo ngược ý nghĩa của nó làm mô hình đổi câu trả lời. Điểm faithfulness là chênh lệch độ chính xác giữa hai nhánh, nằm trong [0, 1].

## Chạy nhanh

```bash
conda create -n nlp_final python=3.10 && conda activate nlp_final
pip install -r requirements.txt
echo 'DEEPSEEK_API_KEY="sk-..."' > .env

python run.py --config configs/xcopa_vi_qwen_deepseek.yaml --end 5   # thử 5 câu
python run.py --config configs/xcopa_vi_qwen_deepseek.yaml           # chạy đầy đủ
python run.py --list                                                 # xem thành phần đã đăng ký
```

**Notebook** (chạy trên Colab/Kaggle, không cần cài đặt gì trên máy): [`notebooks/FaithLM_Vietnamese.ipynb`](notebooks/FaithLM_Vietnamese.ipynb)

## Đổi mô hình

Mọi thành phần được chọn bằng tên trong file YAML. Không cần sửa code khi đổi mô hình:

```yaml
predictor:
  name: qwen                              # qwen | phi | vicuna | hf | deepseek
  model_id: Qwen/Qwen3-4B-Instruct-2507   # bất kỳ model nào trên HF Hub
  load_in_4bit: false                     # true nếu GPU dưới 12GB

explainer:
  name: deepseek                          # deepseek | openai | claude | hf | baseline_*

metric:
  name: paper                             # paper | symmetric
  scorer: logprob                         # logprob | exact_match
```

Ghi đè nhanh từ dòng lệnh mà không cần sửa file:

```bash
python run.py --config configs/xcopa_vi_qwen_deepseek.yaml \
    --predictor hf --predictor_model_id Qwen/Qwen3-8B --metric symmetric
```

Thêm một mô hình mới = thêm một hàm có decorator trong [`faithlm/predictors.py`](faithlm/predictors.py):

```python
@register_predictor("my_model")
def _build(cfg):
    return HFPredictor(model_id=cfg.model_id or "org/my-model")
```

## Khác biệt so với mã nguồn gốc

| | Bản gốc | Bản này |
|---|---|---|
| Chấm điểm | So khớp chuỗi trên 1 câu → chỉ ra 0.0 hoặc 1.0 | Log-prob các lựa chọn → liên tục trong [0,1] |
| Chỉ số | Nhánh "thật" không hề nhận giải thích | Thêm `symmetric`: cả hai nhánh đều có gợi ý |
| Chọn mô hình | Chuỗi `if/elif` ở 4 chỗ | Registry + YAML |
| Mất session | Chạy lại từ đầu | Tiếp tục từ câu dang dở |
| Baseline | Không có | Phủ định theo luật, đồng nhất, ngẫu nhiên |
| Kiểm thử | Không có | 26 unit test, không cần GPU |

Chi tiết kỹ thuật: [`docs/changelog_2026-08-17.md`](docs/changelog_2026-08-17.md)

## Cấu trúc

```
faithlm/
├── registry.py      # đăng ký thành phần theo tên
├── config.py        # config có kiểu, nạp từ YAML hoặc dict
├── datasets.py      # XCOPA-vi, COPA-en, ECQA, Social IQa, TriviaQA
├── predictors.py    # mô hình đích + chấm điểm log-prob
├── explainers.py    # mô hình sinh giải thích (API/cục bộ)
├── baselines.py     # baseline không dùng LLM
├── metrics.py       # chỉ số paper / symmetric
├── prompts.py       # toàn bộ mẫu prompt
├── pipelines.py     # vòng lặp local & global, có checkpoint
└── run.py           # entrypoint dùng chung cho CLI và notebook
configs/             # 5 biến thể thực nghiệm
tests/               # 26 unit test
notebooks/           # notebook nộp bài
```

## Kiểm thử

```bash
python -m pytest tests/ -v
```

Không cần GPU hay API key — dùng stub model.

## Chia việc trong nhóm

Bốn config dưới đây **chỉ khác nhau đúng một biến**, dùng chung `seed: 42` và
`sampling: random`, nên ghép lại được thành một bảng kết quả so sánh có kiểm soát.

| Người | Config | Biến thay đổi | Vai trò trong báo cáo |
|---|---|---|---|
| A | `configs/xcopa_vi_qwen_deepseek.yaml` | — | Kết quả chính (tái lập paper) |
| B | `configs/xcopa_vi_symmetric.yaml` | `metric` | Đóng góp của nhóm |
| C | `configs/copa_en_qwen_deepseek.yaml` | ngôn ngữ | So sánh Việt–Anh |
| D | `configs/baseline_negation.yaml` + `configs/global_xcopa_vi.yaml` | explainer | Baseline + pipeline global |

Người D nhẹ nhất (`xai_iter: 1`, không gọi API) nên gánh thêm pipeline global.

**Không cần push file dữ liệu đã lọc.** `sampling: random` + `seed: 42` cho ra
đúng cùng 200 câu trên máy của mọi người. Kiểm tra bằng:

```bash
python -c "from faithlm import load_config; from faithlm.pipelines import select_indices; \
print(select_indices(load_config('configs/xcopa_vi_qwen_deepseek.yaml'), 500)[:10])"
```

Nếu ai đó ra dãy số khác thì config đã bị sửa lệch.

### Chốt số câu và số vòng lặp bằng số đo thật

Ước tính "1 phút/iteration" là đo trên code cũ, khi mỗi lần chấm điểm đều phải
sinh 256 token. Với `scorer: logprob` predictor không sinh token khi chấm nữa,
nên nút cổ chai chuyển sang độ trễ API của explainer. Đo lại trước khi chốt:

```bash
python scripts/estimate_runtime.py --config configs/xcopa_vi_qwen_deepseek.yaml --probe 3 --iters 3
```

Script chạy thử vài câu rồi ngoại suy ra thời gian cho các mức 200/500 câu và
5/10/15/20 vòng lặp, kèm số session Kaggle 12 giờ cần dùng.

## Chạy toàn bộ thực nghiệm

```bash
bash scripts/run_all_experiments.sh
python scripts/aggregate_results.py --results_dir ./results --output ./docs/experiment_results.md
```

Bị ngắt giữa chừng thì chạy lại đúng lệnh đó; những câu đã xong sẽ được bỏ qua.

## Lưu ý khi chạy trên Kaggle

- Bật **GPU** và **Internet** trong Settings (Internet cần xác thực số điện thoại).
- Thêm `DEEPSEEK_API_KEY` qua *Add-ons → Secrets*.
- Session giới hạn 12 giờ. Ghi kết quả vào `/kaggle/working` để tải về, rồi upload lại làm Dataset cho lần chạy tiếp theo.
- T4/P100 16GB đủ chạy `load_in_4bit: false` (bf16), nhanh hơn 4-bit.

## Tài liệu

- [Kiến trúc](docs/architecture.md)
- [Tham chiếu cấu hình](docs/configuration.md)
- [Changelog](docs/changelog_2026-08-17.md)
