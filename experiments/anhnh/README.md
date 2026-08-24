# anhnh — FaithLM reproduction

| File | What it is |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | **Conclusions.** Hand-written — edit this one. |
| [`SUMMARY.md`](SUMMARY.md) | Generated report: findings inlined, then setup, tables, per-run detail, error analysis. **Do not edit** — `build_report.py` overwrites it. |
| `*/` | Raw results, one JSON per question. |

```bash
python scripts/build_report.py \
    --results_dir ./experiments/anhnh \
    --output ./experiments/anhnh/SUMMARY.md
```

`FINDINGS.md` is inlined into `SUMMARY.md` at the top, so a reader sees the
conclusions next to the numbers without needing to know a second file exists,
and regenerating the report cannot destroy the analysis.

## Runs

Predictor `microsoft/phi-2`, explainer `openrouter/openai/gpt-3.5-turbo` through
a LiteLLM gateway — both the paper's own models (§4.2). The fallback model was
disabled for every run, so the explainer identity is never silently substituted.

| Directory | N | Config | Purpose |
|---|---|---|---|
| `copa_en_phi_gpt35_paper_rep1` | 500 | Paper Table 2 (COPA): 20 steps, predictor temp 0.7, explainer temp 0.9, top-p 0.9 | Reproduction |
| `copa_en_phi_gpt35_control` | 500 | As above **+ irrelevant-hint control** | Metric validity |
| `xcopa_vi_phi_gpt35_control` | 500 | As above, Vietnamese | Cross-lingual, matched |
| `copa_en_phi_gpt35_greedy` | 30 | 5 steps, greedy decoding | Decoding ablation |
| `xcopa_vi_phi_gpt35_greedy` | 30 | 5 steps, greedy decoding | Early Vietnamese probe |

Balanced COPA test and XCOPA-vi test are the same 500 items, so the two 500-run
control variants differ only in language.
