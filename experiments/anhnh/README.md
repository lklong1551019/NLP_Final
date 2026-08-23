# anhnh — FaithLM reproduction

Predictor `microsoft/phi-2`, explainer `openrouter/openai/gpt-3.5-turbo` via a
LiteLLM gateway. Both are the paper's own models (§4.2). The fallback model was
disabled for every run, so the explainer identity is never silently substituted.

| Directory | N | Config | Purpose |
|---|---|---|---|
| `copa_en_phi_gpt35_paper_rep1` | 500 | Paper Table 2 (COPA): 20 steps, predictor temp 0.7, explainer temp 0.9, top-p 0.9 | Reproduction |
| `copa_en_phi_gpt35_control` | see SUMMARY | As above **+ irrelevant-hint control** | Metric validity |
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

On the 500-instance reproduction the contrary hint flipped **352/352** correct
predictions and **34/35** incorrect ones. A metric at its ceiling cannot rank
explanations.

The control run explains why. An *irrelevant* hint — "The weather forecast
mentions scattered clouds tomorrow afternoon." — flips the model on a large
fraction of questions on its own. Subtracting that baseline is what separates
the explanation's contribution from the target model simply following hints.
Current numbers are in [`SUMMARY.md`](SUMMARY.md).
