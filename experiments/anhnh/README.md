# anhnh — FaithLM reproduction

Predictor `microsoft/phi-2`, explainer `openrouter/openai/gpt-3.5-turbo` via a
LiteLLM gateway. Both are the paper's own models (§4.2). The fallback model was
disabled for every run, so the explainer identity is never silently substituted.

| Directory | N | Config | Purpose |
|---|---|---|---|
| `copa_en_phi_gpt35_paper_rep1` | 500 | Paper Table 2 (COPA): 20 steps, predictor temp 0.7, explainer temp 0.9, top-p 0.9 | Reproduction |
| `copa_en_phi_gpt35_control` | 500 | As above **+ irrelevant-hint control** | Metric validity |
| `xcopa_vi_phi_gpt35_control` | 500 | Paper config **+ irrelevant-hint control** | Cross-lingual, matched |
| `copa_en_phi_gpt35_greedy` | 30 | 5 steps, greedy decoding | Decoding ablation |
| `xcopa_vi_phi_gpt35_greedy` | 30 | 5 steps, greedy decoding | Cross-lingual comparison |

## Read this before quoting a number

- **One repetition, not three.** The paper averages over 3 runs; these are
  single runs, so no variance is reported. That is the same gap we criticise the
  paper for.
- **Two aggregations give very different answers.** Mean over every scoring
  event on the 500-instance run is 0.4707; maximum per question is 0.8720. The
  latter is what compares to the paper, which reports the explanation after
  optimisation converges. `SUMMARY.md` uses maximum per question throughout.
- **`diff_score` is binary per instance.** Accuracy over a single example is 0
  or 1, so each observation carries one bit. We drew wrong conclusions twice in
  this project from n=2.

## The finding that matters

On both 500-instance runs the contrary hint flipped **every** parseable
prediction. Fidelity is exactly 1.0000 for all 354 correct answers *and* all 34
incorrect ones. A metric pinned at its ceiling cannot rank explanations, and is
equally consistent with "the target model just follows whatever hint it is
given".

The control run separates the two. A hint with nothing to do with the question —
"The weather forecast mentions scattered clouds tomorrow afternoon." — is scored
the same way:

| Group | n | Fidelity | Irrelevant hint | Corrected |
|---|---|---|---|---|
| Correct prediction | 354 | 1.0000 | 0.5989 | **+0.4011** |
| Incorrect prediction | 34 | 1.0000 | 0.7353 | +0.2647 |
| Unparseable (`X`) | 112 | 0.3750 | 0.3571 | +0.0179 |
| **All** | **500** | **0.8600** | **0.5540** | **+0.3060** |

Two readings, and both matter:

- **The metric is not empty.** The contrary explanation beats an irrelevant
  sentence by +0.306. Something real is being measured.
- **The uncorrected number overstates it ~2.8x.** 64% of the raw 0.860 is the
  model following any hint at all, not the explanation carrying its reason.

The `X` row is an internal check on the control itself. Where the predictor
produced no usable answer, there is nothing for an explanation to be faithful
to — and there the contrary hint beats random by only +0.018. The control
behaves as a baseline should rather than as noise.

Incorrect predictions are flipped by a random hint more often (0.735) than
correct ones (0.599), which is what you would expect: a model that is already
wrong needs less of a nudge to move to the other option.

## Cross-lingual, matched at 500 instances

Balanced COPA test and XCOPA-vi test are the **same 500 items**, so language is
the only variable. Same predictor, explainer, config and control.

| | copa_en | xcopa_vi |
|---|---|---|
| Accuracy | 70.8% | **11.8%** |
| Unparseable (`X`) | 22.4% | **45.4%** |
| Fidelity | 0.860 | 0.336 |
| Irrelevant hint | 0.554 | 0.122 |
| **Corrected** | **+0.306** | **+0.214** |
| Mean iterations | 3.11 | **7.82** |

Vietnamese accuracy is 11.8% (95% CI [9.3%, 14.9%]); restricted to the 273
instances that produced a parseable answer it is 21.6% ([17.1%, 26.9%]). Chance
on a two-choice task is 50%, well outside both intervals, so Phi-2 is not merely
uninformed in Vietnamese - it picks the wrong option systematically.

An earlier 30-instance estimate put this at 33.3%. The 500-instance value is
11.8%, off by 21 points, and at n=30 the confidence interval [19.2%, 51.2%]
still contained chance. Nothing here is safe to conclude from small samples.

### The conditional result

Splitting by whether the predictor answered at all changes the picture:

| Group | copa_en corrected | xcopa_vi corrected |
|---|---|---|
| Correct prediction | +0.4011 (n=354) | **+0.4746** (n=59) |
| Incorrect prediction | +0.2647 (n=34) | +0.3224 (n=214) |
| Unparseable (`X`) | +0.0179 (n=112) | +0.0441 (n=227) |

Where the model actually produces an answer, the corrected signal **survives the
language change and is if anything larger**. The aggregate drop from +0.306 to
+0.214 is driven almost entirely by the 227 unparseable instances, which
contribute +0.044.

The random-hint control also collapses in Vietnamese (0.554 → 0.122), which is
consistent: a model that cannot read the question cannot read the hint either,
so it has nothing to be suggestible to.

### Cost

Vietnamese needed 7.82 optimisation iterations per question against 3.11 for
English, and ~82s/question against ~19s. FaithLM's early stop fires on the first
non-zero score, so when the measurement yields nothing the loop runs to the
20-iteration cap on every instance. The framework spends the most compute
exactly where it works least - the paper's Limitations section mentions the
carbon cost of iterating, but not that the cost scales inversely with success.
