# Experiment outputs

Raw per-question results from the FaithLM reproduction. One JSON file per
question; the first line is the prediction record, the remaining lines are one
record per optimisation iteration.

Regenerate the summary with:

```bash
python scripts/build_report.py --results_dir ./experiments --output ./experiments/SUMMARY.md
```

## Runs

| Directory | N | Config | Purpose |
|---|---|---|---|
| `copa_en_phi_gpt35_paper_rep1` | 500 | Paper Table 2 (COPA): 20 steps, predictor temp 0.7, explainer temp 0.9, top-p 0.9 | Reproduction |
| `copa_en_phi_gpt35_control` | **84** | As above **+ irrelevant-hint control** | Metric validity |
| `copa_en_phi_gpt35_greedy` | 30 | 5 steps, greedy decoding | Decoding ablation |
| `xcopa_vi_phi_gpt35_greedy` | 30 | 5 steps, greedy decoding | Cross-lingual comparison |

Predictor: `microsoft/phi-2`. Explainer: `openrouter/openai/gpt-3.5-turbo` via a
LiteLLM gateway. Both are the paper's own models (§4.2). The fallback model was
disabled for every run so the explainer identity is never silently substituted.

## Read this before quoting a number

- **One repetition, not three.** The paper averages over 3 runs; these are
  single runs, so no variance is reported. Ironically this is the same gap we
  criticise the paper for.
- **`copa_en_phi_gpt35_control` is incomplete: 84/500.** Five of six shards died
  with `torch.OutOfMemoryError` when another user's process filled the shared
  GPU. `RESUME=1` will continue it; the numbers here are provisional.
- **Two aggregations give very different answers.** Mean over every scoring
  event on the 500-instance run is 0.4707; maximum per question is 0.8720. The
  latter is what compares to the paper, which reports the explanation after
  optimisation converges. `SUMMARY.md` uses maximum per question throughout.
- **`diff_score` is binary per instance.** Accuracy over a single example is 0
  or 1, so each observation carries one bit. Small samples are meaningless - we
  drew wrong conclusions twice in this project from n=2.

## The finding that matters

On the 500-instance reproduction the contrary hint flipped **352/352** correct
predictions and **34/35** incorrect ones. A metric at its ceiling cannot rank
explanations.

The control run explains why. An *irrelevant* hint - "The weather forecast
mentions scattered clouds tomorrow afternoon." - flips the model on 60.7% of
questions, against 86.9% for the contrary explanation:

```
raw fidelity        0.869
irrelevant hint     0.607
corrected           +0.262
```

So roughly 70% of the headline fidelity score is the target model following
whatever hint it is given, not the explanation carrying its reason. The measure
is not empty - the corrected margin is real - but the uncorrected number
overstates faithfulness by about 3x.
