# FaithLM — Configuration Reference

> This document serves as the **single source of truth** for all CLI arguments, model identifiers, dataset options, and environment variables used in the FaithLM pipeline. Reference this file when setting up new experiments or switching models/datasets.

---

## Environment Variables

Set these in the `.env` file at the project root, or export them in your shell.

| Variable | Description | Default | Example |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API authentication key | *(required)* | `sk-0de6a9...` |
| `DEEPSEEK_BASE_URL` | DeepSeek API base URL | `https://api.deepseek.com` | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | Default DeepSeek model to use | `deepseek-v4-pro` | `deepseek-v4-flash` |

---

## CLI Arguments

### Common Arguments (both `main_local.py` and `main_global.py`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--device_num` | int (nargs) | `0` | GPU device ID(s) for model loading |
| `--data` | str | `ecqa` | Dataset to use (see [Dataset Options](#dataset-options)) |
| `--pred_model` | str | `vicuna` | Target/Predictor LLM (see [Predictor Models](#predictor-models)) |
| `--xai_model` | str | `claude` | Explainer LLM (see [Explainer Models](#explainer-models)) |
| `--max_tokens` | int | `1000` | Max tokens for explainer generation |
| `--temp_exp` | float | `0.9` | Temperature for explainer generation |
| `--deepseek_key` | str | `None` | DeepSeek API key (overrides `DEEPSEEK_API_KEY` env var) |
| `--deepseek_model` | str | `deepseek-v4-pro` | DeepSeek model name |
| `--xcopa_lang` | str | `vi` | XCOPA language subset code |
| `--data_split` | str | `test` | Dataset split to use (`train`, `test`, `validation`) |
| `--load_in_4bit` | flag | `True` | Enable 4-bit quantization (for ≤ 8GB GPUs) |
| `--no_4bit` | flag | — | Disable 4-bit quantization |
| `--gpt_key` | str | `None` | OpenAI/Azure GPT API key (legacy) |
| `--claude_key` | str | `None` | Anthropic Claude API key (legacy) |
| `--openrouter_key` | str | `None` | OpenRouter API key (overrides `OPENROUTER_API_KEY`) |
| `--openrouter_model` | str | `qwen/qwen3.7-flash` | OpenRouter model slug |
| `--or_reasoning` | flag | *off* | Allow reasoning tokens. **Leave off** — see below |
| `--top_p` | float | `0.9` | Top-p for the explainer (paper's Table 2 setting) |
| `--max_spend` | float | `5.0` | Hard-stop once this many USD are charged (`0` = no limit) |
| `--usage_log` | str | `None` | JSONL of per-call tokens and charged cost |
| `--verbose` | flag | *off* | Print full scoring prompts (thousands of lines at scale) |

> **`--load_in_4bit` / `--no_4bit` currently do nothing.** `load_model()` receives only
> `(model_name, max_memory)`, so the flag is never read and the `qwen` branch always
> quantizes. Left as-is to avoid changing the other variants' behaviour.

### Local Pipeline Only (`main_local.py`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--xai_iter` | int | `20` | Number of LLM-OPT iterations per question |
| `--ques_idx_start` | int | `40` | Start index for question processing |
| `--ques_idx_end` | int | `40` | End index for question processing (exclusive) |
| `--save_cf_file_path` | str | `None` | Path to save counterfactual explanations |
| `--save_file_path` | str | `./results` | Output directory for results |

### Global Pipeline Only (`main_global.py`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--xai_iter` | int | `3` | Number of LLM-OPT iterations per round |
| `--round_xai_iter` | int | `10` | Number of optimization rounds |
| `--ques_sample` | int | `15` | Number of questions sampled per iteration |
| `--save_file` | str | `./results/global` | Output directory for results |

---

## Predictor Models

The `--pred_model` argument selects the target LLM whose predictions we want to explain.

| Value | Model | Source | Type | VRAM Needed | Notes |
|---|---|---|---|---|---|
| `qwen` | Qwen3.5-4B | `Qwen/Qwen3.5-4B` | Local (HF) | ~4–5 GB (4-bit) | **Primary**. Uses 4-bit NF4 quantization by default. |
| `phi` | Phi-2 | `microsoft/phi-2` | Local (HF) | ~5 GB (bf16) | Alternative small model. 2.7B params. |
| `vicuna` | Vicuna-7B | `lmsys/vicuna-7b-v1.5` | Local (HF) | ~14 GB (bf16) | Original paper's predictor. Too large for 8GB GPU. |
| `claude` | Claude-2 | Anthropic API | API | — | Legacy. Requires `--claude_key`. |
| `gpt35` | GPT-3.5 Turbo | Azure OpenAI API | API | — | Legacy. Requires `--gpt_key`. |

### Adding a New Predictor Model

1. Add a new `elif model_name == "your_model"` block in [`model/predictor.py` → `load_model()`](file:///home/long/Master/FaithLM/model/predictor.py#L10-L53)
2. Add corresponding branches in `generate_predictor_output_ecqa()` and `_ecqa_score()` (same file)
3. Update this table

---

## Explainer Models

The `--xai_model` argument selects the LLM used for generating and refining explanations.

| Value | Model | Source | Type | Notes |
|---|---|---|---|---|
| `deepseek` | DeepSeek v4 Pro/Flash | DeepSeek API | API | OpenAI-compatible. Set model via `--deepseek_model`. |
| `openrouter` | any OpenRouter model | OpenRouter | API | Qwen variant. Set slug via `--openrouter_model`. |
| `phi` | Phi-2 | `microsoft/phi-2` | Local (HF) | Can be used as both predictor and explainer. |
| `claude` | Claude-2 | Anthropic API | API | Legacy. Requires `--claude_key`. |
| `gpt35` | GPT-3.5 Turbo | Azure OpenAI API | API | Legacy. Requires `--gpt_key`. |

### DeepSeek Model Variants

| `--deepseek_model` Value | Description | Use Case |
|---|---|---|
| `deepseek-v4-pro` | High capability, complex reasoning | Default — best quality explanations |
| `deepseek-v4-flash` | Cheaper, faster, lower latency | Budget-conscious experiments or large-scale runs |

### OpenRouter Explainer (`--xai_model openrouter`)

Set `OPENROUTER_API_KEY` in `.env`, then:

```bash
bash scripts/run_qwen_openrouter_xcopa_vi.sh                 # local + global
MODE=local END_IDX=20 bash scripts/run_qwen_openrouter_xcopa_vi.sh   # short smoke run
```

**Reasoning tokens must stay off.** They are billed as output *and* count against
`max_tokens`. Measured on 10 XCOPA-vi instances at `max_tokens=400`:

| Config | completion | reasoning | usable | $/call |
|---|---|---|---|---|
| `qwen3.7-flash`, reasoning off | 105.6 | 0 | **105.6** | $0.000021 |
| `qwen3.7-flash`, reasoning on | 402.0 | 400.0 | **2.0** | $0.000057 |
| `qwen3.8-max`, reasoning excluded | 457.9 | 431.0 | **26.9** | $0.003211 |

With reasoning on the model returns an **empty string**, which the pipeline would
otherwise split into a blank "explanation" and score silently. `openrouter_client.py`
raises `EmptyCompletion` instead. The flag is off by default; `--or_reasoning` re-enables it.

**Do not use `qwen/qwen3.8-max`** — it rejects `reasoning.enabled=false` with
*"Reasoning is mandatory for this endpoint"*, so ~94% of every call is unavoidable
reasoning overhead, and a 500×2-target run costs ~$27 instead of ~$0.18.

Reasonable slugs: `qwen/qwen3.7-flash` ($0.03/$0.13 per 1M), `qwen/qwen3.7-plus`
($0.32/$1.28), `qwen/qwen3.7-max` ($1.475/$4.425). All three allow disabling reasoning.

Spend is tracked per call, written to `--usage_log`, and hard-stopped at `--max_spend`.

### Adding a New Explainer Model

1. Add a new `elif model_name == "your_model"` block in [`model/explainer.py` → `reponse_xai_model()`](file:///home/long/Master/FaithLM/model/explainer.py#L10-L65)
2. Update this table

---

## Dataset Options

The `--data` argument selects the dataset and associated preprocessing/scoring logic.

| Value | Dataset | Source | Language | Task | Splits Available |
|---|---|---|---|---|---|
| `xcopa_vi` | XCOPA Vietnamese | `cambridgeltl/xcopa` (`vi`) | Vietnamese | 2-choice causal reasoning | `validation` (100), `test` (500) |
| `copa_en` | Balanced COPA | `pkavumba/balanced-copa` | English | 2-choice causal reasoning | `train` (1000), `test` (500) |
| `copa` | Balanced COPA | `pkavumba/balanced-copa` | English | 2-choice causal reasoning | Legacy — uses original preprocessing |
| `ecqa` | ECQA | `yangdong/ecqa` | English | 5-choice commonsense QA | `train` |
| `trivaqa` | TriviaQA | `THUDM/LongBench` (`triviaqa_e`) | English | Open-ended QA with passage | `test` |
| `social` | Social IQa | `tasksource/bigbench` (`social_iqa`) | English | Multi-choice social reasoning | `validation` |
| `xcopa` | XCOPA Italian | `xcopa` (`it`) | Italian | 2-choice causal reasoning | Legacy — original code |

### XCOPA Language Codes

When using `--data xcopa_vi`, the language is controlled by `--xcopa_lang`. Available codes from `cambridgeltl/xcopa`:

`et` (Estonian), `ht` (Haitian), `id` (Indonesian), `it` (Italian), `qu` (Quechua), `sw` (Swahili), `ta` (Tamil), `th` (Thai), `tr` (Turkish), `vi` (Vietnamese), `zh` (Chinese)

---

## Typical Experiment Commands

### Quick Debug (2 questions, Vietnamese)
```bash
python main_local.py \
    --data xcopa_vi --pred_model qwen --xai_model deepseek \
    --xai_iter 2 --ques_idx_start 0 --ques_idx_end 2 --device_num 0
```

### Full Local Run (50 questions, Vietnamese)
```bash
python main_local.py \
    --data xcopa_vi --pred_model qwen --xai_model deepseek \
    --xai_iter 20 --ques_idx_start 0 --ques_idx_end 50 --device_num 0
```

### English COPA Baseline
```bash
python main_local.py \
    --data copa_en --data_split train --pred_model qwen --xai_model deepseek \
    --xai_iter 20 --ques_idx_start 0 --ques_idx_end 50 --device_num 0
```

### Global Pipeline (Vietnamese)
```bash
python main_global.py \
    --data xcopa_vi --pred_model qwen --xai_model deepseek \
    --xai_iter 3 --round_xai_iter 10 --ques_sample 15 --device_num 0
```

### Switch to DeepSeek Flash
```bash
python main_local.py \
    --data xcopa_vi --pred_model qwen --xai_model deepseek \
    --deepseek_model deepseek-v4-flash \
    --xai_iter 20 --ques_idx_start 0 --ques_idx_end 50
```

---

## Output File Naming Convention

**Local**: `local_{data}_{xai_model}_{pred_model}_iter-{xai_iter}_sample-{question_idx}.json`
**Global**: `global_{data}_{xai_model}_{pred_model}_iter-{total_iters}_sample-{ques_sample}.json`

Example: `local_xcopa_vi_deepseek_qwen_iter-20_sample-0.json`
