# ACL short paper

`main.pdf` — ~4.5 pages of body plus references, official ACL format
(`acl.sty`, `acl_natbib.bst` from [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)).

## Build

```bash
tectonic -X compile main.tex --outdir .
```

Or with a standard toolchain: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Provenance

Every number traces to a member's raw outputs under [`../experiments/`](../experiments/)
or to their written analysis. Nothing in the paper is estimated.

| Claim | Value | Owner | Source |
|---|---|---|---|
| Reproduction vs paper | 0.872 vs ~0.85 | anhnh | `anhnh/copa_en_phi_gpt35_paper_rep1` |
| Ceiling: every parseable flip | 354/354, 34/35 | anhnh | `anhnh/copa_en_phi_gpt35_control` |
| Control A, English | 0.860 / 0.554 / +0.306 | anhnh | `anhnh/copa_en_phi_gpt35_control` |
| Control A, Vietnamese | 0.336 / 0.122 / +0.214 | anhnh | `anhnh/xcopa_vi_phi_gpt35_control` |
| Control B (mismatched hint) | 0.370 vs 0.140, p<0.0001 | anhnq | `anhnq/random_hint_control.json`, `anhnq/FINDINGS.md` |
| Control C (position bias) | 83/117 correct = 83/117 first-pick | minhndn | `docs/experiment_explainer_sweep.md` §4 |
| Position-bias floor / fidelity | 62% / 0.755 | minhndn | same |
| Phi-2 at chance under logprob | 0.520 vs floor 0.530, 87% option A | anhnq | `anhnq/FINDINGS.md` |
| Control D (flip reproducibility) | 0.647 [0.596, 0.694], n=365 | minhndn | `docs/experiment_explainer_sweep.md` §5 |
| Control D per arm | 0.607 / 0.684, p=0.12 | minhndn | same |
| Goodhart signature | 0.715 vs 0.609, p=0.041 | minhndn | same §5.3 |
| Explainer sweep, 4 explainers | p=0.216–0.754, none significant | minhndn | same §3 |
| 3×2 grid, pooled by target | 0.943 vs 0.950, z=−0.51, p=0.607 | minhndn | same §8.4 |
| Score modes | 0.374 / 0.460 / 0.509 | anhnq | `anhnq/xcopa_vi_qwen_gpt4omini_*` |
| McNemar | p=0.00011, p=0.01003, p=0.103 | anhnq | `anhnq/FINDINGS.md` |
| Parser discards / disagrees | 45–47% / 59% of 150 | anhnq | same |
| Δcorrectness ≠ Δprediction example | instance 4 scored 0.0 on a real flip | anhnq | `docs/metric_experiment.md` |
| Target sweep (Table 4) | 11.8 / 41.5 / 90.5 / 92.5% | anhnh, minhndn | `anhnh/xcopa_vi_phi_gpt35_control`, `minhndn/*_promptEN` |
| Explanation language spread | 0.4%–75.3% Vietnamese | minhndn | `docs/experiment_explainer_sweep.md` §8.5 |
| Algorithm 2: seed vs best | 0.467→0.533 / 0.467→**0.400** / 0.333→0.600 | longlk | `longlk/*/global_*.json` |
| Algorithm 2: error string as prompt | 5/16 records, incl. final | longlk | `longlk/xcopa_vi_qwen_deepseek/global_*.json` |
| Explainer pro vs flash (same target) | 0.905 vs 0.850 | longlk | `longlk/xcopa_vi_qwen_deepseek{,_flash}` |
| Reasoning-flow variant | 0.905 → 0.425, iter 4.87 → 10.20 | longlk | `longlk/SUMMARY.md`, `longlk/xcopa_vi_qwen_deepseek*` |
| Iterations 7.82 vs 3.11 | — | anhnh | `anhnh/*_control` |
| 16.6% wasted iterations | — | anhnq | `anhnq/FINDINGS.md` |
| Reasoning-token exhaustion | 179/200, 827 tokens | minhndn | `docs/experiment_explainer_sweep.md` §6.1 |
| Silent fallback contamination | 23% of one arm | minhndn | same §6.3 |
| English answers to Vietnamese questions | error tables | longlk | `longlk/SUMMARY.md` §4 |

## Deliberate omissions

Stated in §8 of the paper and repeated here so they are not read as oversights:

- **One repetition per configuration.** The original averages three. No variance
  is reported.
- **No truthfulness metric.** It needs gold explanations; COPA and XCOPA have
  none. ECQA would be required.
- **Baselines not re-run.** `SelfExp` and `Self-consistency` are absent from the
  released code; we compare against published values and say so.
- **Control D is an upper bound.** One redraw, two of four arms, and only on
  instances that already flipped.
- **The weak-target hypothesis is untested**, not refuted: the intended middle
  target measured 90.5% accuracy and is a second strong target.
