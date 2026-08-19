# FaithLM Codebase Changelog
**Date**: 2026-08-17
**Branch**: `fix/litellm-repro`

Follow-up to [`changelog_2026-08-16.md`](changelog_2026-08-16.md). This round came
out of an audit of that changelog against the code: three of its claims did not
match what the repository actually contained, and verifying them surfaced four
further defects, two of which silently corrupted the results.

## A. Corrections to the previous changelog

| Previous claim | Reality |
|---|---|
| §1 accuracy-parsing fix applied to `main_local.py` **and `main_global.py`** | `main_global.py` has no correctness check at all — no `Corrct`/`Wrong` branch, before or after that commit. Only `main_local.py` was changed. |
| §3 "Integrated `BitsAndBytesConfig` into the `load_model` function" | The NF4 config existed only inside the `qwen` branch, and `load_model()` took no quantization parameter — so `--load_in_4bit` / `--no_4bit` were parsed and then ignored. |
| §6 "Files modified: `scripts/run_all_experiments.sh`" | Six scripts were created, not one. `run_all_experiments.sh` also never forwards `XAI_ITER` to the global stage, so global ran at its default `xai_iter=3` rather than the documented 15. |

## B. Result-corrupting defects fixed

### B1. Answer matching was case- and punctuation-sensitive
**Files**: `model/predictor.py`

`_ecqa_score` and `generate_api_predictor_output` decided whether the predictor
was right with a literal `gold.strip() in text` test. Modern chat models restate
the choice inside a sentence, so the *same correct answer* scored differently
depending on phrasing:

| Prompt | Model output | Old score |
|---|---|---|
| with explanation | `...Correct choice: It was fragile.` | 1.0 |
| with counterfactual | `The item was packaged in bubble wrap because it was fragile.` | **0.0** |

Both answers are correct. The lowercase `i` and the moved full stop produced
`diff_score = 1.0` — a fabricated faithfulness signal. Since `diff_score` is the
paper's entire measurement, every affected instance was noise.

Replaced with `contains_answer()`, which casefolds, strips punctuation and
collapses whitespace via a unicode-aware regex, so Vietnamese diacritics survive.
`_trivaqa_score` already compared case-insensitively, so ECQA-style scoring is
now consistent with it rather than newly invented.

### B2. `[choice]…@` slicing produced garbage when the terminator was missing
**Files**: `model/predictor.py`

The parser did `if index != -1: ans = ans[index+1:end_index]` where
`end_index = ans.find("@")`. When a model emitted `]` but no `@`, `end_index` was
`-1` and the slice silently dropped the last character, yielding an unmatchable
string. Now guarded by `index != -1 and end_index > index`, so such responses
fall through to the normalized containment test instead. Applied at all 7 sites.

### B3. Bare `except: continue` hid API failures behind a valid-looking file
**Files**: `main_global.py`

The global optimisation loop was wrapped in `except: continue`, printing
`API Error: Skip Step`. A wrong key or a model id with the wrong prefix made the
script finish in seconds and write an empty results file that looked like a
successful run. It now logs the exception type, message and traceback, counts
failures, and aborts once they exceed `max(3, xai_iter // 2)`.

### B4. Empty completions were scored as wrong answers
**Files**: `model/llm_api.py`

The gateway returns an empty body (not an error) for prompts its content filter
dislikes — e.g. the COPA instance containing the word *weapon* returned empty on
4 consecutive attempts. An empty response scores as a wrong answer, which again
manufactures a faithfulness difference. Empty completions are now detected,
retried, then routed to `LITELLM_FALLBACK_MODEL`, and counted in `STATS` so the
report can disclose how many instances were affected instead of hiding them.

## C. Crashes fixed

### C1. `run_experiment.sh` passed arguments `main_global.py` does not define
**Files**: `scripts/run_experiment.sh`, `main_global.py`

The script passed `--save_file_path`, `--ques_idx_start` and `--ques_idx_end` to
`main_global.py`, which defines none of them (it has `--save_file` and
`--ques_sample`). `argparse` exited with code 2, so the global half of the
documented workflow never ran. The script now passes the correct flags, and
`--save_file_path` is registered as an alias so both entry points accept it.

### C2. Unknown model names raised `UnboundLocalError`
**Files**: `model/predictor.py`, `model/explainer.py`

`load_model` and `reponse_xai_model` fell off the end of their if/elif chains
without assigning `model` / `response`. Both now raise a `ValueError` naming the
valid options.

### C3. `--deepseek_model` default overrode the environment
**Files**: `main_local.py`, `main_global.py`

The default was the literal string `deepseek-v4-pro`. Being truthy, it always
won over `$DEEPSEEK_MODEL`. On a LiteLLM gateway the id needs a provider prefix
(`deepseek/deepseek-v4-pro`), so every run would have 404'd. Default is now
`None` so the environment is consulted.

## D. Behavioural fix

### D1. Correct answers were logged as the ground truth
**Files**: `main_local.py`

On the correct branch the log line wrote `LLM-A:{answer[0]}` — the gold answer —
instead of `output_ans[0]`. Whenever the model was right, its actual output was
discarded, making it impossible to study "right answer, wrong reasoning" cases.
That is precisely the category the assignment's *Error Analysis* section wants.

## E. New capability: LiteLLM for both model roles

**Files**: `model/llm_api.py` (new), `model/predictor.py`, `model/explainer.py`

Previously only the Explainer could use an API (`deepseek` branch); the Predictor
was limited to local Hugging Face weights or a hardcoded `claude-2` / Azure
`gpt35turbo`. Since 4-bit `bitsandbytes` requires CUDA, none of it runs on Apple
Silicon.

`model/llm_api.py` adds one OpenAI-compatible client shared by both roles, with
retries, exponential backoff, empty-completion handling, a fallback model and
call statistics. Selecting `--pred_model litellm --xai_model litellm` runs the
whole pipeline through the gateway, with the two roles pointed at different
models:

```env
LITELLM_BASE_URL=https://<gateway>/v1
LITELLM_PRED_MODEL=deepseek/deepseek-v4-flash   # the model being explained
LITELLM_XAI_MODEL=deepseek/deepseek-v4-pro      # explainer + LLM optimizer
LITELLM_FALLBACK_MODEL=openrouter/deepseek/deepseek-v3.2
```

This also removes the GPU dependency for the Colab/Kaggle notebook the
assignment requires (§II.2): no 4 GB download, no CUDA, no local paths.

## F. Tooling

- **`scripts/run_sharded.sh`** (new) — the local pipeline treats every question
  independently and writes one file per question, so it shards cleanly by
  question index. Wall-clock drops roughly linearly with shard count.
  Uses `pids[${#pids[@]}]=` rather than `${pids[-1]}` because macOS ships
  bash 3.2, where negative subscripts are a syntax error.
- **`scripts/run_paper_repro.sh`** (new) — drives the full reproduction:
  English COPA, then Vietnamese XCOPA, local (sharded) then global for each.
- **`scripts/build_report.py`** (new) — produces `docs/experiment_report.md`
  with accuracy, faithfulness distribution, early-stop histogram and an error
  table.
- **`scripts/aggregate_results.py`** — now walks the results tree. It previously
  scanned only `results/`, `results/local` and `results/global`, so everything
  `run_experiment.sh` wrote into `results/experiments/<variant>/` was invisible
  to it.

## G. Known issues left in place deliberately

These are upstream behaviours we did **not** change, because changing them would
alter the method rather than fix a defect. They are documented so the report can
discuss them.

1. **The "true" prompt carries no explanation — and this matches the paper.**
   In `diff_task_score_ecqa`, `ture_final_prompt` is built with
   `for _, ques in true_exp_pair`, so the explanation is zipped in and then
   discarded. This is *not* an implementation error: the paper defines fidelity
   as `S_E := f(X) − f(X | ¬E_NL)` (p. 3805), i.e. the shift between the
   unconditioned prediction and the prediction under the contrary hint. The
   explanation is never inserted into any prompt. The consequence is therefore a
   property of the method, not of this code: the score never tests whether the
   explanation *helps*, only whether its negation *hurts*.
2. **`diff_score` is measured on a single instance**, so it can only be 0 or 1.
   The "faithfulness score" is therefore a coin-flip-grained signal per
   iteration, not a continuous measure.
3. **Early stopping triggers on the first non-zero score**
   (`if iter%5 == 0 and sum(scores_list) != 0: break`), so most questions run a
   single iteration. This makes the real runtime far lower than the 50 h/variant
   estimated previously, but also means the LLM optimiser rarely gets to iterate.
4. **`preprocess_copa` assigns instead of appending**
   (`train_dict['answer'] = [...]` inside the loop), leaving only the last
   value. Unused by our variants, which go through `preprocess_copa_en` /
   `preprocess_xcopa_vi`.
