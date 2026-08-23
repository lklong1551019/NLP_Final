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
| `xcopa_vi_qwen35_gemini35_promptEN` | 200 | Qwen3.5-4B 4-bit, local | Gemini-3.5-flash | Target axis, synced with the prompt-VI member |
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

## Target axis

Explainer and prompt fixed, target varied:

| Target | Accuracy | `X` | Faithfulness |
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
