# anhnh — FaithLM reproduction

Predictor `microsoft/phi-2`, explainer `openrouter/openai/gpt-3.5-turbo` via a
LiteLLM gateway. Both are the paper's own models (§4.2). The fallback model was
disabled for every run, so the explainer identity is never silently substituted.

| Directory | N | Config | Purpose |
|---|---|---|---|
| `copa_en_phi_gpt35_paper_rep1` | 500 | Paper Table 2 (COPA): 20 steps, predictor temp 0.7, explainer temp 0.9, top-p 0.9 | Reproduction |
| `copa_en_phi_gpt35_control` | 500 | As above **+ irrelevant-hint control** | Metric validity |
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
