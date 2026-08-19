# Implementation Plan: FaithLM with Qwen3.5-4B + DeepSeek + XCOPA-Vi

## Goal

Adapt the FaithLM framework to run experiments on the **Vietnamese XCOPA** dataset (`cambridgeltl/xcopa`, `vi` subset) using:
- **Target LLM (Predictor)**: `Qwen/Qwen3.5-4B` (local, HuggingFace) — swappable with `microsoft/phi-2`
- **Explainer LLM**: DeepSeek API (`deepseek-v4-pro` via OpenAI-compatible endpoint) — swappable with flash or other APIs
- **Flexibility**: Support both XCOPA-Vi and the original English COPA (`pkavumba/balanced-copa`) for comparison experiments

---

## Resolved Questions

### GPU: 3050Ti with 8GB VRAM
Qwen3.5-4B in bfloat16 needs ~8–10GB — too tight for a 3050Ti. We'll use **4-bit quantization** via `bitsandbytes`:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
# ~4–5 GB VRAM usage, fits comfortably on 8GB
```

This requires adding `bitsandbytes>=0.43.0` to `requirements.txt`.

### Dataset Splits
Verified from HuggingFace:

| Dataset | Splits | Size |
|---|---|---|
| `cambridgeltl/xcopa` → `vi` | `validation` (100), `test` (500) | **No train split** |
| `pkavumba/balanced-copa` (English) | `train` (1000), `test` (500) | Has train split |

**Strategy**:
- **Global pipeline** (needs sampling): Use `balanced-copa` English `train` split (1000 examples) OR XCOPA-Vi `test` (500 examples) depending on experiment
- **Local pipeline** (per-question): Use XCOPA-Vi `test` (500 examples) or `validation` (100 examples) for debugging
- Configurable via `--data_split` CLI arg

### DeepSeek Model
Default: `deepseek-v4-pro`. Switchable to `deepseek-v4-flash` via `--deepseek_model` CLI arg.

### COPA ↔ XCOPA Flexibility
Add `--data` options: `"copa"` (original English balanced-copa) and `"xcopa_vi"` (Vietnamese). Both share the same 2-choice format and scoring logic.

---

## Analysis: The `preprocess_copa()` / `preprocess_xcopa()` Answer Code

> [!NOTE]
> **User asked**: "Are you sure about the overwrite bug? What part of the paper mentions this, or is there another branch?"

**Findings**:
- There is **only one branch** (`main`) in the repo — no other branches exist.
- The paper does not discuss implementation details at this level.
- The same code pattern appears in **both** `main_local.py` and `main_global.py`, and in **both** `preprocess_copa()` and `preprocess_xcopa()`.

**The code in question** ([main_local.py:L126](file:///home/long/Master/FaithLM/main_local.py#L126)):
```python
for idx, ques_txt in enumerate(question_text):
    train_dict['question'].append(question)       # ← append (correct)
    train_dict['answer'] = [opt[answer[idx]] for opt in option]  # ← full replacement each iteration
```

**What this does**:
- On each iteration, it creates a list of `len(option)` elements: `[option[0][label_idx], option[1][label_idx], ...]`
- Uses the label (`answer[idx]`) of the **current** question and applies it to **ALL** option pairs
- On the next iteration, the entire list is **replaced**
- After the loop, `train_dict['answer']` contains `[option[i][answer[LAST_IDX]] for i in range(N)]`

**Downstream** ([main_local.py:L282](file:///home/long/Master/FaithLM/main_local.py#L282)):
```python
answer = [train_dict['answer'][idx]]  # Gets option[idx][answer[LAST_IDX]]
```
This gives the **last question's label** applied to each question's options — not each question's own label.

**Verdict**: This is likely an unintended behavior, but since COPA is binary (labels are 0 or 1), ~50% of answers happen to be correct by coincidence, and the pipeline doesn't crash. The authors' main experiments in the paper focused on ECQA and TriviaQA (which don't have this issue), so COPA/XCOPA may have been added later without thorough testing.

**Our approach**: We'll implement `preprocess_xcopa()` correctly from scratch:
```python
train_dict['answer'].append(option[idx][answer[idx]])  # Correct: each question's own answer
```
We'll also add a `preprocess_copa_en()` with the same fix for English COPA experiments.

> [!IMPORTANT]
> If you'd prefer to keep the original behavior for reproducibility with the paper's COPA results, let me know and I'll add a `--legacy_preprocessing` flag.

---

## Proposed Changes

### Configuration & Environment

#### [NEW] `requirements.txt`

```
torch>=2.1.0
transformers>=4.40.0
datasets>=2.19.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
openai>=1.30.0
scikit-learn>=1.4.0
tqdm>=4.66.0
numpy>=1.26.0
python-dotenv>=1.0.0
```

- `bitsandbytes` — for 4-bit quantization on 3050Ti
- `openai` — for DeepSeek API (OpenAI-compatible)
- `python-dotenv` — for loading API keys from `.env`
- Python version: **3.10+**

#### [NEW] `.env`

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

---

### Data Preprocessing

#### [MODIFY] `main_local.py` — Rewrite `preprocess_xcopa()` + Add `preprocess_copa_en()`

```python
def preprocess_xcopa(lang="vi", split="test"):
    """Load XCOPA dataset for cross-lingual experiments."""
    train_dict = collections.defaultdict(list)

    all_na_data = load_dataset("cambridgeltl/xcopa", lang)
    hg_data = all_na_data[split]
    question_text = hg_data["premise"]
    question_purp = hg_data["question"]
    labels = hg_data["label"]
    op1 = hg_data["choice1"]
    op2 = hg_data["choice2"]
    option = list(zip(op1, op2))
    choice = [f"[choice]{opt[0]}@ [choice]{opt[1]}@" for opt in option]

    for idx, ques_txt in enumerate(question_text):
        question = (
            f"###Question: What is the {question_purp[idx]} of the Premise?\n"
            f"### Premise: {ques_txt}\n"
            f"### Choices: {choice[idx]}"
        )
        train_dict['question'].append(question)
        train_dict['answer'].append(option[idx][labels[idx]])
    return train_dict


def preprocess_copa_en(split="train"):
    """Load original English balanced-COPA dataset."""
    train_dict = collections.defaultdict(list)

    all_na_data = load_dataset("pkavumba/balanced-copa")
    hg_data = all_na_data[split]
    question_text = hg_data["premise"]
    question_purp = hg_data["question"]
    labels = hg_data["label"]
    op1 = hg_data["choice1"]
    op2 = hg_data["choice2"]
    option = list(zip(op1, op2))
    choice = [f"[choice]{opt[0]}@ [choice]{opt[1]}@" for opt in option]

    for idx, ques_txt in enumerate(question_text):
        question = (
            f"###Question: What is the {question_purp[idx]} of the Premise?\n"
            f"### Premise: {ques_txt}\n"
            f"### Choices: {choice[idx]}"
        )
        train_dict['question'].append(question)
        train_dict['answer'].append(option[idx][labels[idx]])
    return train_dict
```

#### [MODIFY] `main_global.py` — Add `xcopa_vi` and `copa_en` data branches

Same preprocessing functions, integrated into the data selection `if/elif` chain.

---

### Predictor Module

#### [MODIFY] [`model/predictor.py`](file:///home/long/Master/FaithLM/model/predictor.py)

**Add Qwen3.5-4B support** in `load_model()` with 4-bit quantization:

```python
elif model_name == "qwen":
    print("============ Predictor: Qwen3.5-4B (4-bit)")
    from transformers import BitsAndBytesConfig
    model_id = "Qwen/Qwen3.5-4B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
```

**Add `"qwen"` branches** in:
- `generate_predictor_output_ecqa()` — reuses similar logic to `"phi"` branch
- `_ecqa_score()` — same pattern
- `diff_task_score_ecqa()` — no change needed (already generic)

---

### Explainer Module

#### [MODIFY] [`model/explainer.py`](file:///home/long/Master/FaithLM/model/explainer.py)

**Add DeepSeek API support** in `reponse_xai_model()`:

```python
elif model_name == "deepseek":
    from openai import OpenAI
    import os

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", args.deepseek_key),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    deepseek_model = getattr(args, 'deepseek_model', None) \
        or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

    try:
        input_text = prompt if isinstance(prompt, str) else prompt[0]
        completion = client.chat.completions.create(
            model=deepseek_model,
            temperature=args.temp_exp,
            max_tokens=args.max_tokens,
            messages=[
                {"role": "system", "content": "You are an expert at explaining language model behavior."},
                {"role": "user", "content": input_text},
            ],
        )
        response = completion.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] DeepSeek API: {e}")
        response = "API error."
```

---

### CLI Arguments

#### [MODIFY] Both `main_global.py` and `main_local.py` — `get_args()`

New arguments:

```python
# API config
parser.add_argument('--deepseek_key', type=str, default=None)
parser.add_argument('--deepseek_model', type=str, default='deepseek-v4-pro',
                    choices=['deepseek-v4-pro', 'deepseek-v4-flash'])

# Dataset config
parser.add_argument('--xcopa_lang', type=str, default='vi')
parser.add_argument('--data_split', type=str, default='test',
                    choices=['train', 'test', 'validation'])

# Quantization
parser.add_argument('--load_in_4bit', action='store_true', default=True,
                    help='Use 4-bit quantization (required for 8GB GPUs)')
parser.add_argument('--no_4bit', dest='load_in_4bit', action='store_false')
```

Updated `--data` choices: `"ecqa"`, `"trivaqa"`, `"copa"`, `"copa_en"`, `"xcopa_vi"`, `"social"`

Updated `--pred_model` choices: `"vicuna"`, `"phi"`, `"qwen"`, `"claude"`, `"gpt35"`

Updated `--xai_model` choices: `"claude"`, `"gpt35"`, `"phi"`, `"deepseek"`

---

### Shell Scripts

#### [NEW] `scripts/run_local_xcopa_vi.sh`

```bash
#!/bin/bash
set -euo pipefail

# Load .env
if [ -f .env ]; then source .env; fi

PRED_MODEL="${PRED_MODEL:-qwen}"
XAI_MODEL="${XAI_MODEL:-deepseek}"
DATA="xcopa_vi"
DEVICE="${DEVICE:-0}"
XAI_ITER="${XAI_ITER:-20}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-50}"
SAVE_PATH="./results/local"

mkdir -p "$SAVE_PATH"

echo "=== FaithLM Local Pipeline ==="
echo "  Predictor: $PRED_MODEL | Explainer: $XAI_MODEL | Data: $DATA"

python main_local.py \
    --device_num $DEVICE \
    --data "$DATA" \
    --pred_model "$PRED_MODEL" \
    --xai_model "$XAI_MODEL" \
    --xai_iter "$XAI_ITER" \
    --ques_idx_start "$START_IDX" \
    --ques_idx_end "$END_IDX" \
    --save_file_path "$SAVE_PATH" \
    --max_tokens 1000 \
    --temp_exp 0.9 \
    --load_in_4bit
```

#### [NEW] `scripts/run_local_copa_en.sh`

Same but with `--data copa_en --data_split train` for English COPA experiments.

#### [NEW] `scripts/run_global_xcopa_vi.sh`

Global pipeline variant.

#### [NEW] `scripts/run_all_experiments.sh`

```bash
#!/bin/bash
# Master script: all experiment variants

VARIANTS=(
    "qwen deepseek xcopa_vi"    # V1: Qwen + DeepSeek on Vietnamese
    "qwen deepseek copa_en"     # V2: Qwen + DeepSeek on English COPA
    "phi deepseek xcopa_vi"     # V3: Phi-2 + DeepSeek on Vietnamese
    "phi deepseek copa_en"      # V4: Phi-2 + DeepSeek on English COPA
)

for variant in "${VARIANTS[@]}"; do
    read -r pred xai data <<< "$variant"
    echo "=== Variant: pred=$pred xai=$xai data=$data ==="
    PRED_MODEL=$pred XAI_MODEL=$xai DATA=$data bash scripts/run_local_xcopa_vi.sh
done

python scripts/aggregate_results.py --results_dir ./results --output ./docs/experiment_results.md
echo "=== Results saved to docs/experiment_results.md ==="
```

---

### Experiment Tracking

#### [NEW] `scripts/aggregate_results.py`

Reads all result JSON files, computes averages per variant, generates markdown:

#### [NEW] `docs/experiment_results.md` (auto-generated template)

```markdown
# FaithLM Experiment Results

## Summary Table
| Variant | Predictor | Explainer | Dataset | Pipeline | Avg Score | N |
|---------|-----------|-----------|---------|----------|-----------|---|
| V1 | Qwen3.5-4B (4bit) | DeepSeek-v4-pro | XCOPA-Vi | Local | — | — |
| V2 | Qwen3.5-4B (4bit) | DeepSeek-v4-pro | COPA-En  | Local | — | — |
| V3 | Phi-2              | DeepSeek-v4-pro | XCOPA-Vi | Local | — | — |
| V4 | Phi-2              | DeepSeek-v4-pro | COPA-En  | Local | — | — |
```

---

## File Summary

| File | Action | Description |
|---|---|---|
| `requirements.txt` | **NEW** | Python deps incl. bitsandbytes for 4-bit |
| `.env` | **NEW** | DeepSeek API key + config |
| `model/predictor.py` | **MODIFY** | Add Qwen3.5-4B w/ 4-bit quantization, new branches |
| `model/explainer.py` | **MODIFY** | Add DeepSeek API via OpenAI SDK |
| `main_local.py` | **MODIFY** | New `preprocess_xcopa()` + `preprocess_copa_en()`, new CLI args, data branches |
| `main_global.py` | **MODIFY** | Same preprocessing + CLI additions |
| `scripts/run_local_xcopa_vi.sh` | **NEW** | Local pipeline runner (XCOPA-Vi) |
| `scripts/run_local_copa_en.sh` | **NEW** | Local pipeline runner (English COPA) |
| `scripts/run_global_xcopa_vi.sh` | **NEW** | Global pipeline runner |
| `scripts/run_all_experiments.sh` | **NEW** | Master experiment orchestrator |
| `scripts/aggregate_results.py` | **NEW** | Results → markdown report |
| `docs/experiment_results.md` | **NEW** | Auto-generated experiment report |

---

## Verification Plan

### Automated Tests

1. **Dataset smoke test**:
   ```bash
   python -c "from main_local import preprocess_xcopa; d = preprocess_xcopa('vi', 'test'); print(len(d['question']), d['answer'][0])"
   ```

2. **Qwen loading** (4-bit):
   ```bash
   python -c "from model.predictor import load_model; m, t = load_model('qwen', {0: '8GB'}); print('VRAM OK')"
   ```

3. **DeepSeek API**:
   ```bash
   python -c "
   from openai import OpenAI; import os
   c = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com')
   r = c.chat.completions.create(model='deepseek-v4-pro', messages=[{'role':'user','content':'Say hi'}], max_tokens=10)
   print(r.choices[0].message.content)
   "
   ```

4. **End-to-end** (2 questions):
   ```bash
   python main_local.py --data xcopa_vi --pred_model qwen --xai_model deepseek \
       --xai_iter 2 --ques_idx_start 0 --ques_idx_end 2 --device_num 0
   ```

### Manual Verification
- Inspect result JSON files in `./results/local/`
- Verify faithfulness scores in [0, 1]
- Review `docs/experiment_results.md` output
