# ACL short paper

`main.pdf` — 4 pages of body plus references, in the official ACL format
(`acl.sty`, `acl_natbib.bst` from [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)).

## Build

```bash
tectonic -X compile main.tex --outdir .
```

Any LaTeX toolchain works; `tectonic` is convenient because it fetches packages
and runs BibTeX automatically. With a standard install:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Where every number comes from

Each claim in the paper traces to raw per-instance outputs under
[`../experiments/`](../experiments/). Regenerate the aggregate with
`scripts/build_report.py`.

| Paper claim | Source |
|---|---|
| Reproduction, fidelity 0.872 vs ~0.85 | `anhnh/copa_en_phi_gpt35_paper_rep1` (500) |
| Ceiling: 354/354 and 34/35 at 1.0 | `anhnh/copa_en_phi_gpt35_control` |
| Control A: 0.860 / 0.554 / +0.306 | `anhnh/copa_en_phi_gpt35_control` |
| Control A, Vietnamese: 0.336 / 0.122 / +0.214 | `anhnh/xcopa_vi_phi_gpt35_control` |
| Control B: 0.370 vs 0.140, p<0.0001 | `anhnq/random_hint_control.json`, `anhnq/FINDINGS.md` |
| Score modes: 0.374 / 0.460 / 0.509 | `anhnq/xcopa_vi_qwen_gpt4omini_{accuracy,prob_accuracy,logprob}` |
| McNemar p=0.00011, p=0.01003, p=0.103 | `anhnq/FINDINGS.md` |
| Parser discards 45–47%; disagrees on 59% | `anhnq/FINDINGS.md` |
| Target sweep on XCOPA-vi (Table 3) | `minhndn/xcopa_vi_*_promptEN`, `anhnh/xcopa_vi_phi_gpt35_control` |
| Explainer sweep 0.930–0.975 | `minhndn/xcopa_vi_{dsflash,qwen35}_*` |
| Iterations 7.82 vs 3.11 | `anhnh/*_control` |
| 16.6% wasted iterations | `anhnq/FINDINGS.md` |

## Deliberate omissions

Stated in §8 of the paper, repeated here so they are not mistaken for oversights:

- **One repetition per configuration.** The original averages three. We report
  no variance and say so.
- **No truthfulness metric.** It needs gold explanations; COPA and XCOPA have
  none. ECQA would be required.
- **Baselines not re-run.** `SelfExp` and `Self-consistency` are absent from the
  released code. We compare against the original's published values instead of
  re-implementing them, and label that clearly.
