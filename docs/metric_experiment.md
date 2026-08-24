# Fidelity metric experiment — `--score_mode`

Does replacing the published fidelity score with a continuous one reduce the number
of optimisation iterations, and does it beat the baseline?

The new signal is opt-in. `--score_mode accuracy` is the default and runs the
published code path unchanged, so existing results stay reproducible.

## Why the baseline signal is hard to optimise against

`diff_task_score_ecqa` returns `abs(acc(f(X)) - acc(f(X|¬E)))`. On a single
instance each accuracy is 0 or 1, so the trajectory the explainer sees is a list of
zeros and ones. Three problems follow:

| Problem | Effect on the loop |
|---|---|
| Binary — no gradient | `[E₁,0] [E₂,0] [E₃,0]` cannot say which rewrite moved the target |
| `abs()` — no sign | "hint pushed the model away" and "hint pushed it further in" score alike |
| Measures Δcorrectness, not Δprediction | if the target is wrong both times the score is 0 even when the answer flipped |

The third one is not hypothetical. On XCOPA-vi instance 4 the contrary hint flipped
the prediction from `Đang vào đợt nghỉ.` to `Đang mùa hè.` and the published metric
recorded `0.0`, because neither answer matched the gold string exactly.

## What `--score_mode logprob` measures

The target's probability over the answer choices, read in both conditions:

```
prob_shift = P_before(a₀) − P_after(a₀)        signed, continuous, ∈ [−1, 1]
```

`a₀` is the choice the target picked on its own, so **no gold label is involved** —
which is what faithfulness is about: whether the explanation's content drives the
model, including when the model is wrong.

Nothing is parsed out of free text, so the `[choice]…@` extraction, its `"X"`
sentinel and the spurious "Bad Answer" early stops all drop out. The target is not
sampled either — one forward pass per choice instead of 256 generated tokens.

All four numbers come from the same two probability vectors, at no extra cost:

| Field | Meaning | Use |
|---|---|---|
| `prob_shift` | signed probability shift | what the optimiser follows |
| `tv` | total variation between the two distributions | non-negative divergence — the quantity Theorem 1 defines |
| `flip` | did argmax change | report |
| `accuracy` | baseline metric, from choice labels not strings | compare against the paper |

## Measured on the first 6 XCOPA-vi instances

Same instances, same explanations, same contrary hints — only the readout changes.

| # | prob_shift | tv | flip | accuracy | P(a₀) before → after |
|---|---|---|---|---|---|
| 0 | +0.4647 | 0.4647 | 0 | 0 | 0.9959 → 0.5312 |
| 1 | +0.9511 | 0.9511 | 1 | 1 | 0.9740 → 0.0230 |
| 2 | +0.2078 | 0.2078 | 0 | 0 | 0.9627 → 0.7549 |
| 3 | +0.0092 | 0.0092 | 0 | 0 | 0.9933 → 0.9841 |
| 4 | +0.8572 | 0.8572 | 1 | 1 | 0.9526 → 0.0953 |
| 5 | +0.4120 | 0.4120 | 1 | 1 | 0.8808 → 0.4688 |

Non-zero signal: **accuracy 3/6, flip 3/6, logprob 5/6**, graded from 0.0092 to 0.9511.
Instance 3 is the useful case — a near-zero shift says "this explanation barely moves
the model", which a binary metric cannot express.

## Early stopping

The baseline stops at `iter % 5 == 0 and sum(scores) != 0`. A continuous score is
almost never exactly zero, so that rule would fire on iteration 0 and there would be
no optimisation left. In `logprob` mode the loop instead stops when

* the best shift reaches `--stop_threshold` (default 0.5), or
* `--stop_patience` rewrites in a row fail to beat the best (default 4).

**Iteration count is not comparable unless you report the stopping rule**, since the
two modes use different ones. Report iterations-to-stop alongside the final score.

## Running the comparison

```bash
# baseline — published metric, unchanged code path
python main_local.py --data xcopa_vi --pred_model qwen --xai_model deepseek \
    --score_mode accuracy --xai_iter 15 --ques_idx_start 0 --ques_idx_end 50 \
    --save_file_path ./results/metric_ab/accuracy \
    --metrics_log ./results/metric_ab/accuracy/metrics.jsonl

# new signal
python main_local.py --data xcopa_vi --pred_model qwen --xai_model deepseek \
    --score_mode logprob --xai_iter 15 --ques_idx_start 0 --ques_idx_end 50 \
    --save_file_path ./results/metric_ab/logprob \
    --metrics_log ./results/metric_ab/logprob/metrics.jsonl
```

Both runs write every metric each iteration, so the two are directly comparable
whichever one was being optimised.

What to report: iterations to stop, final `tv` (paper-comparable), flip rate, and
wall-clock per instance.

## API targets

`choice_probs_local` uses the model's logits directly, so a local predictor needs
nothing extra. An API target needs a provider that returns logprobs — many do not,
and a gateway may route the same model to several. Pin one:

```bash
LITELLM_PROVIDER_ORDER=Parasail   # comma separated, fallbacks disabled
```

If logprobs are unavailable the run raises rather than scoring on a uniform prior.

## Known blocker, unrelated to this branch

`load_model()` still loads `Qwen/Qwen3.5-4B` with `AutoModelForCausalLM`. That model
is `Qwen3_5ForConditionalGeneration` (vision-language, config nests `text_config`),
and the call fails with `'Qwen3_5Config' object has no attribute 'vocab_size'`. The
local predictor cannot start until that is fixed; the numbers above were produced
with an API target standing in.
