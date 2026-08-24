## Findings

Predictor `microsoft/phi-2`, explainer `gpt-3.5-turbo` — the paper's own models
(§4.2). 500 instances, Table 2 COPA column. One repetition, not three.

### 1. The paper reproduces

Fidelity **0.872**, against the ~0.85 reported for COPA in Figure 3.

### 2. But the metric sits at its ceiling

The contrary hint flipped **every** parseable prediction — all 354 correct
answers and all 34 incorrect ones score exactly 1.0000. A measure at its maximum
cannot rank explanations, and is equally consistent with "the target model just
follows whatever hint it is given".

### 3. About two thirds of that is suggestibility

The paper has no control for this. Scoring a third prompt per instance carrying
a hint unrelated to the question ("The weather forecast mentions scattered
clouds tomorrow afternoon.") separates the two:

| | copa_en | xcopa_vi |
|---|---|---|
| Accuracy | 70.8% | 11.8% |
| Unparseable (`X`) | 22.4% | 45.4% |
| Fidelity | 0.860 | 0.336 |
| Irrelevant hint | 0.554 | 0.122 |
| **Corrected** | **+0.306** | **+0.214** |
| Mean iterations | 3.11 | 7.82 |

The measure is not empty — the contrary explanation beats an irrelevant sentence
by +0.306 — but 64% of the raw 0.860 is the model following any hint at all.

### 4. In Vietnamese the parser fails, not the metric

Balanced COPA test and XCOPA-vi test are the **same 500 items**, so language is
the only variable. Splitting by whether the predictor answered at all reverses
the aggregate picture:

| Group | copa_en corrected | xcopa_vi corrected |
|---|---|---|
| Correct prediction | +0.4011 (n=354) | **+0.4746** (n=59) |
| Incorrect prediction | +0.2647 (n=34) | +0.3224 (n=214) |
| Unparseable (`X`) | +0.0179 (n=112) | +0.0441 (n=227) |

Where the model produces an answer the corrected signal **survives the language
change and is larger**. The drop from +0.306 to +0.214 is driven almost entirely
by the 227 unparseable instances, which contribute +0.044.

Vietnamese accuracy is 11.8% (95% CI [9.3%, 14.9%]); over the 273 parseable
answers it is 21.6% ([17.1%, 26.9%]). Chance on a two-choice task is 50%, well
outside both, so Phi-2 picks the wrong option systematically rather than
guessing. The random-hint control collapses too (0.554 → 0.122), consistent with
a model that cannot read the hint either.

This points at generate-then-parse scoring as the defect, not at the fidelity
measure — which is what likelihood-based scoring would fix.

### 5. Cost scales inversely with success

Vietnamese needed 7.82 optimisation iterations per question against 3.11, and
~82s against ~19s. Early stop fires on the first non-zero score, so when the
measurement yields nothing every instance runs to the 20-iteration cap. The
paper's Limitations section mentions the carbon cost of iterating, but not that
the cost is highest exactly where the framework works least.

### Caveats

- **One repetition.** The paper averages three; no variance is reported here.
  That is the same gap we criticise in `docs/paper_limitations.md` (L10).
- **Aggregation changes the answer.** Mean over every scoring event on the
  500-instance run is 0.4707; maximum per question is 0.8720. The latter is what
  compares to the paper. Tables here use maximum per question throughout.
- **`diff_score` carries one bit per instance.** Accuracy over a single example
  is 0 or 1. A 30-instance estimate put Vietnamese accuracy at 33.3% with a
  confidence interval that still contained chance; the 500-instance value is
  11.8%. Nothing here is safe to conclude from small samples.
