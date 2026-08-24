# Experiment outputs

One subdirectory per team member, so runs from different people never collide:

```
experiments/
  <username>/
    <dataset>_<predictor>_<explainer>_<variant>/   one JSON per question
    SUMMARY.md                                     generated, do not hand-edit
```

Inside a run directory there is one file per question. The first line is the
prediction record; each remaining line is one optimisation iteration.

Regenerate a summary for your own runs:

```bash
python scripts/build_report.py \
    --results_dir ./experiments/<username> \
    --output ./experiments/<username>/SUMMARY.md
```

`.gitignore` blocks `*.json` repository-wide; the rule is negated for
`experiments/**` so results here are tracked on purpose.

## Members

| Directory | Runs |
|---|---|
| [`anhnh/`](anhnh/) | FaithLM reproduction on Phi-2 + GPT-3.5, metric-validity control, decoding ablation, English/Vietnamese comparison |
| [`minhndn/`](minhndn/) | Explainer sweep across four 2026 models, explainer × target grid, position-bias and re-sampling controls |
