# FaithLM — Tham chiếu cấu hình

Tài liệu chuẩn cho mọi tùy chọn. Mọi thành phần chọn bằng **tên**, nên đổi mô hình không cần sửa code.

Xem danh sách thành phần đã đăng ký:

```bash
python run.py --list
```

---

## Biến môi trường

Đặt trong `.env` ở thư mục gốc, hoặc export trong shell. Trên Kaggle dùng *Add-ons → Secrets*; trên Colab dùng *userdata*.

| Biến | Mô tả | Bắt buộc khi |
|---|---|---|
| `DEEPSEEK_API_KEY` | Khóa API DeepSeek | `explainer.name: deepseek` |
| `DEEPSEEK_BASE_URL` | Mặc định `https://api.deepseek.com` | không |
| `DEEPSEEK_MODEL` | Mặc định `deepseek-v4-pro` | không |
| `OPENAI_API_KEY` | Khóa OpenAI | `explainer.name: openai` |
| `ANTHROPIC_API_KEY` | Khóa Anthropic | `explainer.name: claude` |
| `GEMINI_API_KEY` | Khóa Google AI Studio | `explainer.name: gemini` |
| `GEMINI_BASE_URL` | Mặc định endpoint OpenAI-compat của Gemini | không |
| `OLLAMA_BASE_URL` | Mặc định `http://localhost:11434` | không |
| *(tùy chọn trong config)* | `predictor.api_key_env` / `base_url_env` trỏ tới biến bất kỳ | `predictor.name: api` |

Backend `ollama` **không cần khóa API** — chỉ cần Ollama đang chạy trên máy.

---

## Cấu trúc file YAML

```yaml
dataset:
  name: xcopa_vi           # xcopa_vi | copa_en | ecqa | social | trivaqa
  lang: vi                 # mã ngôn ngữ XCOPA
  split: test              # train | test | validation

predictor:
  name: qwen               # qwen | phi | vicuna | hf | ollama | api | deepseek
  model_id: null           # null -> mặc định của backend
  load_in_4bit: false      # true nếu GPU dưới 12GB
  max_new_tokens: 256
  temperature: 0.7
  max_memory_per_gpu: 14GiB
  device_num: [0]
  api_key_env: DEEPSEEK_API_KEY    # chỉ dùng với name: api
  base_url_env: DEEPSEEK_BASE_URL  # chỉ dùng với name: api

explainer:
  name: deepseek           # deepseek | openai | claude | ollama | hf | baseline_*
  model_id: null
  max_tokens: 1000
  temperature: 0.9

metric:
  name: paper              # paper | symmetric | flip
  scorer: logprob          # logprob | exact_match (flip bỏ qua trường này)
  k_samples: 1             # chỉ dùng với exact_match

run:
  pipeline: local          # local | global | selfcons
  ques_idx_start: 0
  ques_idx_end: 200
  sampling: random         # sequential | random
  xai_iter: 15             # số vòng LLM-OPT mỗi câu (local)
  round_xai_iter: 10       # số vòng tối ưu (global)
  ques_sample: 15          # số câu lấy mẫu mỗi vòng (global)
  prompt_lang: en          # en | vi — ngôn ngữ toàn bộ mẫu prompt
  holdout_split: null      # global: tối ưu trên split này, transfer-eval trên split chính
  holdout_size: 15         # số câu lấy từ holdout split
  output_dir: ./results
  resume: true             # bỏ qua phần đã chạy xong
  seed: 42
```

Gõ sai tên khóa hoặc tên mục sẽ báo lỗi ngay khi nạp config, không âm thầm bỏ qua.

---

## Predictor

| Tên | Mô hình mặc định | Loại | Hỗ trợ log-prob |
|---|---|---|---|
| `qwen` | `Qwen/Qwen3-4B-Instruct-2507` | HF cục bộ | có |
| `phi` | `microsoft/phi-2` | HF cục bộ | có |
| `vicuna` | `lmsys/vicuna-7b-v1.5` | HF cục bộ | có |
| `hf` | *(bắt buộc đặt `model_id`)* | HF cục bộ | có |
| `ollama` | `qwen3.5:4b` | Ollama cục bộ | **không** |
| `api` | *(bắt buộc `model_id`)* | OpenAI-compatible bất kỳ | **không** |
| `deepseek` | `deepseek-v4-pro` | API | **không** |

Predictor qua API/Ollama không tính được log-prob, phải dùng `scorer: exact_match` hoặc `metric: flip`. Chương trình kiểm tra và báo lỗi ngay từ đầu thay vì để chạy rồi mới hỏng.

Backend `api` nhận mọi endpoint OpenAI-compatible mà không cần thêm code — đặt `api_key_env`/`base_url_env` trỏ tới biến môi trường tương ứng của nhà cung cấp.

Dùng bất kỳ mô hình nào trên Hub mà không cần sửa code:

```yaml
predictor:
  name: hf
  model_id: Qwen/Qwen3-8B
```

### VRAM tham khảo

| Mô hình | bf16 | 4-bit NF4 |
|---|---|---|
| Qwen3-4B | ~9 GB | ~4 GB |
| Phi-2 (2.7B) | ~6 GB | ~3 GB |
| Vicuna-7B | ~14 GB | ~5 GB |

Kaggle T4/P100 (16GB) chạy được Qwen3-4B ở bf16. GPU 8GB nên bật `load_in_4bit: true`.

---

## Explainer

| Tên | Mặc định | Loại | Ghi chú |
|---|---|---|---|
| `deepseek` | `deepseek-v4-pro` | API | Chính. Có thể đổi `deepseek-v4-flash` cho rẻ hơn |
| `openai` | `gpt-4o-mini` | API | |
| `claude` | `claude-sonnet-5` | API | |
| `gemini` | `gemini-3.5-flash` | API | Cần `GEMINI_API_KEY`; endpoint OpenAI-compat dựng sẵn |
| `ollama` | `qwen3.5:9b` | Ollama cục bộ | Không cần khóa API; explainer nên mạnh hơn predictor |
| `hf` | *(bắt buộc `model_id`)* | Cục bộ | Tốn thêm VRAM cùng lúc với predictor |
| `baseline_negation` | — | Luật | Phủ định theo mẫu |
| `baseline_identity` | — | Luật | Trả lại nguyên văn |
| `baseline_shuffle` | — | Luật | Câu ngẫu nhiên |

Các explainer qua API tự thử lại 3 lần với backoff lũy thừa.

---

## Dataset

| Tên | Nguồn | Ngôn ngữ | Split |
|---|---|---|---|
| `xcopa_vi` | `cambridgeltl/xcopa` | 11 ngôn ngữ qua `lang` | validation (100), test (500) |
| `copa_en` | `pkavumba/balanced-copa` | Anh | train (1000), test (500) |
| `ecqa` | `yangdong/ecqa` | Anh | train |
| `social` | `tasksource/bigbench` | Anh | validation |
| `trivaqa` | `THUDM/LongBench` | Anh | test — **chỉ dùng `exact_match`** |

Mã ngôn ngữ XCOPA: `et`, `ht`, `id`, `it`, `qu`, `sw`, `ta`, `th`, `tr`, `vi`, `zh`.

---

## Chỉ số

| `metric.name` | Công thức | Dùng khi |
|---|---|---|
| `paper` | \|acc(không gợi ý) − acc(gợi ý đối nghịch)\| | Tái lập bản gốc |
| `symmetric` | \|acc(giải thích thật) − acc(giải thích đối nghịch)\| | Chỉ số đã sửa |
| `flip` | 1 nếu gợi ý đối nghịch làm đổi **đáp án của chính mô hình** | Black-box: predictor API/Ollama, dữ liệu không nhãn |

`flip` so với đáp án mô hình tự đưa ra (không cần nhãn vàng) nên dùng được cho phần demo trên dữ liệu tiếng Việt mới. Nó luôn tự sinh văn bản để chấm, bỏ qua `scorer`.

| `metric.scorer` | Cách chấm | Giới hạn |
|---|---|---|
| `logprob` | Log-prob các lựa chọn → điểm liên tục | Cần dữ liệu có lựa chọn + mô hình cục bộ |
| `exact_match` | So khớp chuỗi, trung bình `k_samples` lần | Điểm rời rạc khi `k_samples: 1` |

---

## Dòng lệnh

Mọi tùy chọn đều ghi đè được mà không cần sửa file YAML:

```bash
python run.py --config configs/xcopa_vi_qwen_deepseek.yaml --end 5
python run.py --config configs/xcopa_vi_qwen_deepseek.yaml --metric symmetric
python run.py --config configs/xcopa_vi_qwen_deepseek.yaml \
    --predictor hf --predictor_model_id Qwen/Qwen3-8B
python run.py --config configs/global_xcopa_vi.yaml --rounds 5 --ques_sample 10
python run.py --config configs/xcopa_vi_qwen_deepseek.yaml --no_resume
```

| Cờ | Ghi đè |
|---|---|
| `--pipeline` | `run.pipeline` |
| `--dataset`, `--lang`, `--split` | mục `dataset` |
| `--predictor`, `--predictor_model_id` | mục `predictor` |
| `--explainer`, `--explainer_model_id` | mục `explainer` |
| `--metric`, `--scorer` | mục `metric` |
| `--start`, `--end`, `--xai_iter` | phạm vi câu hỏi (local) |
| `--rounds`, `--ques_sample` | tham số global |
| `--prompt_lang` | `run.prompt_lang` (en \| vi) |
| `--holdout_split`, `--holdout_size` | hold-out cho global |
| `--output_dir`, `--no_resume` | ghi kết quả |
| `--load_in_4bit` | lượng tử hóa |
| `--list` | in danh sách thành phần rồi thoát |

---

## Dùng trong notebook

Không cần file YAML — dựng config trực tiếp:

```python
from faithlm import run_experiment, from_dict

cfg = from_dict({
    "dataset":   {"name": "xcopa_vi", "lang": "vi", "split": "test"},
    "predictor": {"name": "qwen", "load_in_4bit": False},
    "explainer": {"name": "deepseek"},
    "metric":    {"name": "symmetric", "scorer": "logprob"},
    "run":       {"ques_idx_end": 30, "xai_iter": 5},
})
summary = run_experiment(cfg)
```

Chạy nhiều biến thể mà chỉ nạp mô hình một lần:

```python
from faithlm import predictors, explainers

predictor = predictors.build(cfg.predictor)
explainer = explainers.build(cfg.explainer)

for metric in ("paper", "symmetric"):
    cfg.metric.name = metric
    run_experiment(cfg, predictor=predictor, explainer=explainer)
```

---

## Đường dẫn kết quả

```
results/{pipeline}_{dataset}_{predictor}_{explainer}_{metric}_{scorer}/
├── local/sample-{idx}.json      # một file mỗi câu hỏi
├── global/round-{n}.json        # một file mỗi vòng
├── global/summary.json
└── summary_{pipeline}.json      # kèm toàn bộ config đã dùng
```

Pipeline nằm trong tên thư mục, nên chạy local và global cùng cấu hình không ghi đè lên nhau.

---

## Thêm thành phần mới

Thêm một predictor:

```python
# faithlm/predictors.py
@register_predictor("my_model")
def _build(cfg):
    return HFPredictor(model_id=cfg.model_id or "org/my-model",
                       load_in_4bit=cfg.load_in_4bit)
```

Thêm một chỉ số:

```python
# faithlm/metrics.py
@register_metric("my_metric")
def my_metric(predictor, example, task_instruction, true_exp, counter_exp,
              scorer="logprob", **kwargs):
    ...
    return ScoreDetail(faithfulness=..., true_arm=..., counter_arm=...)
```

Sau đó dùng ngay bằng tên trong YAML. Không phải sửa chỗ nào khác.
