# minhndn — explainer sweep and metric reliability

Paper's two explainers are gone: GPT-3.5-Turbo and Claude-2 are both retired. No
reproduction can call them, so the question this axis answers is what happens
when they are replaced by the 2026 generation — four explainers from four labs,
everything else held fixed.

Predictor `deepseek/deepseek-v4-flash` (OpenRouter) for the sweep, prompts in the
paper's original **English**, verbatim (diffed against the upstream commit
`a8061cd`; only whitespace differs). Vietnamese prompts are a separate axis owned
by another member and are not mixed in here. Reasoning is disabled on every model
— Gemini `thinking_budget=0`, OpenRouter `reasoning.enabled=false` — which also
puts these models closer to the paper's non-thinking originals.

| Directory | N | Target | Explainer | Purpose |
|---|---|---|---|---|
| `xcopa_vi_dsflash_gemini35_promptEN` | 200 | v4-flash | Gemini-3.5-flash | Main run |
| `xcopa_vi_dsflash_gpt56luna_promptEN` | 200 | v4-flash | GPT-5.6-luna | Sweep (OpenAI line) |
| `xcopa_vi_dsflash_qwen37max_promptEN` | 200 | v4-flash | Qwen3.7-max | Sweep (open-weight line) |
| `xcopa_vi_qwen35_gemini35_promptEN` | 200 | Qwen3.5-4B 4-bit, local | Gemini-3.5-flash | Grid, synced with the prompt-VI member |
| `xcopa_vi_qwen35_gpt56luna_promptEN` | 200 | Qwen3.5-4B 4-bit, local | GPT-5.6-luna | Grid — replication on a second target |
| `xcopa_vi_qwen35_qwen37max_promptEN` | 200 | Qwen3.5-4B 4-bit, local | Qwen3.7-max | Grid — replication on a second target |
| `copa_en_phi2_gemini35_promptEN` | 200 | Phi-2, local | Gemini-3.5-flash | Paper's own target, English |
| `xcopa_vi_phi2_gemini35_promptEN` | 200 | Phi-2, local | Gemini-3.5-flash | Why Phi-2 is not the Vietnamese target |
| `xcopa_vi_phi2_gemini35_promptVI_n20` | 20 | Phi-2, local | Gemini-3.5-flash | Qualitative only — Phi-2 emits word salad |
| `flip_repro_gemini_en`, `flip_repro_luna_en` | 178 / 187 | — | — | Re-sampling control, one JSONL row per question |

`xai_iter=8`, explainer temp 0.9 / top-p 0.9, questions 0–199 in dataset order.
Full write-up with z-tests: [`docs/experiment_explainer_sweep.md`](../../docs/experiment_explainer_sweep.md).

## Read this before quoting a number

- **N=100 numbers from this axis are stale.** The first hundred questions were
  favourable: Gemini measured 0.960 there and 0.930 at N=200, luna 0.970 → 0.955.
  Anything quoting "+0.04 over baseline" predates the extension.
- **Faithfulness here is maximum per question**, matching the paper, which reports
  the explanation after optimisation converges.
- **The `_CONTAMINATED` run is excluded on purpose.** A first luna attempt had 23%
  of its calls silently served by the fallback model when rate limits hit; the
  directory is kept out of this folder and the fix is in `llm_api.py` — 429 now
  waits instead of substituting a different model.
- **The baseline row belongs to another member** and ran through the internal
  gateway at N=100. Cross-axis comparisons carry that caveat.

## What the sweep found

Nothing, and that is the result.

| Explainer | N | Faithfulness | 95% CI |
|---|---|---|---|
| DeepSeek-v4-pro (baseline, other member) | 100 | 0.920 | [0.867–0.973] |
| Gemini-3.5-flash | 200 | 0.930 | [0.895–0.965] |
| Qwen3.7-max | 200 | 0.945 | [0.913–0.977] |
| GPT-5.6-luna | 200 | 0.955 | [0.926–0.984] |

Six two-proportion z-tests, none significant (p = 0.216–0.754). All four
confidence intervals overlap completely. Doubling the sample from 100 to 200
tightened the standard error from ~2.0% to 1.5–1.8% and still found no signal —
the metric is at its ceiling and cannot rank explainers.

What did change is measurement quality: unparseable answers fell from 9% to
1.0–2.0%, and empty completions from ~19.6% in the team's first round to 0 across
5,924 calls.

## Two controls that say what the ceiling means

**Position bias.** Phi-2 on Vietnamese answers 41.5% correct — below chance on a
two-choice task — and of the 117 answers that parse, accuracy and first-listed
picks are the *same 83*. Every correct answer is explained by "pick the option
printed first". A target doing no reasoning at all still scores 0.755 faithfulness,
with explanations that never mention position.

**Re-sampling.** Keep the winning explanation, redraw the contrary hint once, ask
the target again:

| Arm | Original flip | Reproduced | 95% CI |
|---|---|---|---|
| Gemini-3.5-flash | 0.930 | **0.607** | [0.533–0.676] |
| GPT-5.6-luna | 0.955 | **0.684** | [0.615–0.747] |
| Pooled | — | **0.647** | [0.596–0.694] |

A third of all flips disappear on a single redraw, with the explanation, question,
target and temperature unchanged. The two arms do not differ significantly
(p = 0.12), so this is a property of the measurement rather than of one explainer.
It also answers the obvious objection to the sweep — "run more questions" — because
the noise lives inside each single-question measurement: ~35 points of it, against
gaps of ≤3.5 points between explainers.

Read together: 0.93 means "flipped on one draw", not "93% of explanations are
faithful".

## Explainer × target grid

Three explainers × two targets, N=200 each, prompt and every other setting fixed.
The second target answers one question: does the ceiling replicate, or was it a
property of `deepseek-v4-flash`?

| Explainer | v4-flash | Qwen3.5-4B 4-bit |
|---|---|---|
| Gemini-3.5-flash | 0.930 | 0.945 |
| GPT-5.6-luna | 0.955 | 0.930 |
| Qwen3.7-max | 0.945 | **0.975** |

All six cells land in 0.930–0.975 — a 4.5-point band. Pooled by target the two are
indistinguishable (0.943 vs 0.950, z=−0.51, p=0.607), and no explainer keeps its
rank across targets: luna is top on v4-flash and bottom on Qwen3.5-4B. **The
ceiling replicates independently on a second target.**

One pairwise exception, reported rather than buried: luna (0.930) vs Qwen3.7-max
(0.975) on Qwen3.5-4B gives z=−2.12, **p=0.034**. It does not survive scrutiny:

- **Multiple comparisons.** It is one of six within-target pairs; the Bonferroni
  threshold is p<0.0083. With six tests, seeing at least one p<0.05 under a true
  null happens ~26% of the time.
- **Wrong test for the design.** Both arms ran the *same* 200 questions, so the
  data are paired and the two-proportion z-test's independence assumption fails.
  McNemar on the same data gives **p=0.052** — the whole effect is 13 questions
  that only Qwen3.7-max flipped against 4 that only luna flipped.
- **Sign flips across targets.** On v4-flash the same pair runs the other way
  (Qwen3.7-max 0.945 *below* luna 0.955, p=0.646). An effect that reverses
  direction between targets is not a property of the explainer.

The z-tests are kept as the headline figure for consistency with the six pairs
already published in the sweep doc; McNemar is reported alongside because it is
the correct test here, not because of which p-value it produces.

### Explanation language is uncontrolled

`PROMPT_LANG=en` sets the *instruction* and *scaffold* in English; the premise and
choices stay Vietnamese, and the paper's `exp_instruction` never names an output
language. Each explainer therefore picks its own, and they disagree wildly —
measured over every explanation record by diacritic detection:

| Explainer | v4-flash | Qwen3.5-4B | Follows |
|---|---|---|---|
| Qwen3.7-max | 1.2% VI | 0.4% VI | the instruction's language |
| GPT-5.6-luna | 34.9% VI | 23.7% VI | mixed |
| Gemini-3.5-flash | 69.7% VI | 75.3% VI | the content's language |

The split is a property of the explainer, not the target (Qwen3.7-max lands at
~1% on both). Explanation language varies from 0.4% to 75.3% Vietnamese while
faithfulness stays inside 0.930–0.975 — the metric cannot see a difference this
coarse. This is an observation under the paper's *unmodified* prompt: the
specification simply does not determine output language once the data is not
English. Forcing Vietnamese would mean editing `exp_instruction`, which would
break comparability with every cell above and is deliberately not done.

### Target axis (earlier framing)

| Target | Accuracy | `X` | Faithfulness (Gemini) |
|---|---|---|---|
| Phi-2 | 41.5% | 41.5% | 0.755 |
| Qwen3.5-4B 4-bit | 90.5% | 6.5% | 0.945 |
| deepseek-v4-flash | 92.5% | 2.0% | 0.930 |

Not monotone in model strength, and Qwen3.5-4B was meant to be the weak middle
point — at 90.5% accuracy it is a second strong target instead, so the hypothesis
that a weak target breaks the ceiling is untested rather than refuted. What
faithfulness does track is the parse rate. Testing the hypothesis properly needs a
target landing between 41.5% and 90.5%; that run is cheap because the predictor is
local, and it is not in this folder.
