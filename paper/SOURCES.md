# Source audit

Every document in the repository, what was read, and what it contributed to
`main.tex`. Written because an earlier draft was assembled from section headings
and had to be rewritten once the material was actually read.

**Read in full** = end to end. **Verified** = numbers recomputed from raw data,
not taken from prose. **Skimmed** = structure and headline numbers checked, body
not read line by line, with the reason given.

## Member analyses

| Source | Lines | Status | Used in |
|---|---|---|---|
| `experiments/anhnq/FINDINGS.md` | 160 | Read in full | §4 score modes, §6.5 Table 3, §6.6, §5(ii), Control B |
| `experiments/anhnq/README.md` | 51 | Read in full | Why no `SUMMARY.md`; run inventory |
| `experiments/minhndn/README.md` | 161 | Read in full | §6.2 sweep, §6.3 Control C, §6.4 Control D |
| `experiments/minhndn/SUMMARY.md` | 440 | Read in full | Per-variant accuracy/unparsed for Table 4 |
| `experiments/longlk/SUMMARY.md` | 128 | Read in full | §6.7 reasoning flow, §7 English-answer failure |
| `experiments/anhnh/FINDINGS.md` | 78 | Read in full | §6.1, §6.3 Table 1, §6.6 |
| `docs/experiment_explainer_sweep.md` | 495 | Read in full | §6.2, §6.3 position bias, §6.4, §5(iii), §6.6 language spread, §8 ethics |
| `docs/metric_experiment.md` | 104 | Read in full | §4 score modes, §6.5 Δcorrectness example |
| `docs/paper_limitations.md` | 170 | Read in full | Framing of §6.3–§6.5 |

## Raw data, recomputed rather than quoted

| Source | Check | Result |
|---|---|---|
| `experiments/*/*/local_*.json` (1,560+) | Aggregated accuracy, unparsed, fidelity, control, iterations per run | Matches every member's reported table |
| `experiments/longlk/*/global_*.json` (3) | Parsed all 16 records per run | §6.7 Table 5 — seed vs best, and the `"API error."` prompts |
| `experiments/minhndn/flip_repro_*/summary.json` | Recomputed pooled rate | (108+128)/(178+187) = 0.6466, matches 0.647 |
| `experiments/anhnq/*/metrics.jsonl` | Field schema | 13 fields incl. all six metrics — confirms modes are comparable |
| Confidence intervals | Wilson, recomputed | Vietnamese accuracy 11.8% → [9.3, 14.9]; n=30 → [19.2, 51.2], contains chance |

## Context documents

| Source | Lines | Status | Used in |
|---|---|---|---|
| `docs/2026.eacl-long.177.pdf` | 23 pp. | Read pp. 1–16 | Eq. 1, Table 2 config, §4.2 metrics, RQ2, Appendix B/C/E |
| `docs/prompts_translation.md` | 16 | Read in full | §5 caveat: VI prompts are fluent rewrites, not literal |
| `docs/changelog_2026-08-19.md` | 19 | Read in full | Same caveat |
| `docs/changelog_2026-08-17.md` | 173 | Read in full | Own bug fixes behind §5(i)–(ii) |
| `docs/Yeu-cau.txt` | 41 | Read in full | Report structure |

## Read but not cited

| Source | Reason |
|---|---|
| `docs/reports/*.md` (8 files) | `build_report.py` output; verified duplicates of `minhndn/SUMMARY.md` — e.g. `xcopa_vi_phi2_target_promptEN.md` gives the same 41.5 / 41.5 / 0.755 / 2.58. Their §0 is a hardcoded template and is wrong for local-predictor runs (noted in the sweep doc §8.6). |
| `experiments/minhndn/*/report.md` (2) | Same generator, same duplication |
| `docs/experiment_report{,_v1_permissive}.md` | Superseded by `experiments/anhnh/SUMMARY.md` |
| `docs/architecture.md`, `docs/configuration.md` | Repository mechanics, no experimental claims |
| `docs/prompt_samples.md` | Prompt illustrations already covered by `prompts_translation.md` |
| `experiments/*/logs/*.log` (up to 24k lines) | Progress-bar output; API counters already summarised per run |

## Not read

| Source | Reason |
|---|---|
| `experiments/anhnq/random_hint_control.json` (2,001 lines) | Control B is quoted from `FINDINGS.md` at the aggregate the author computed (0.370 vs 0.140, p<0.0001). The per-instance file was not re-aggregated independently. |
| `experiments/anhnq/*/metrics.jsonl` bodies (6,700+ lines) | Schema checked; the per-iteration values behind Table 3 are quoted from `FINDINGS.md` rather than recomputed. |
| `docs/2026.eacl-long.177.pdf` pp. 17–23 | Appendices G.4 onward — additional truthfulness figures we do not report. |

The two "not read" rows are the places where the paper trusts a member's own
aggregate instead of recomputing it. Every other number was either recomputed
from raw outputs or checked against them.
