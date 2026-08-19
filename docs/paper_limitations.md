# Limitations of FaithLM observed while reproducing it

Source: Chuang et al., *FaithLM: Towards Faithful Explanations for Large Language
Models*, EACL 2026, pp. 3802–3824.

Every item below is either stated in the paper or was observed directly in our
reproduction (100 instances each on COPA-En and XCOPA-vi, API-served predictor
and explainer). Evidence is cited so each claim can be checked.

---

## L1. The paper's own Limitations section discusses only carbon footprint

Section 6 (p. 3810) is entirely about energy consumption and CO₂ from the
iterative optimisation. It raises nothing about measurement validity, language
coverage, metric granularity, or the behaviour of the released code. For an
EACL long paper proposing a *new metric*, that is a narrow self-assessment, and
it leaves the items below unacknowledged.

## L2. The fidelity score never conditions on the explanation itself

The paper defines fidelity as

    S_E := f(X) − f(X | ¬E_NL)      (p. 3805)

that is, the shift between the *unconditioned* prediction and the prediction
under the **contrary** hint. The explanation `E_NL` is never inserted into a
prompt; only its negation is. The implementation matches this exactly
(`diff_task_score_ecqa` builds the "true" prompt with `for _, ques in
true_exp_pair`, discarding the explanation).

Consequence: the metric measures whether *negating* an explanation hurts, never
whether the explanation *helps*. An explanation can be a fabrication and still
score as faithful, provided its negation confuses the model.

## L3. No control condition separating faithfulness from hint-susceptibility

The design has no arm in which the model receives an **irrelevant or random**
contrary hint. Without it, a high score cannot distinguish

  (a) "this explanation encodes the decision-relevant content", from
  (b) "this model flips whenever it is contradicted" (sycophancy / hint-following).

RQ3 (§4.5) only verifies that contrary hints are *semantically dissimilar* to the
explanations — it never tests whether an arbitrary distractor hint produces the
same flip rate.

Our data makes this concrete: **88% (COPA-En) and 92% (XCOPA-vi)** of instances
flipped at least once. When nearly every hint flips the prediction, the measure
cannot discriminate between a good and a bad explanation.

## L4. The per-instance score is binary, not continuous

Fidelity is computed per instance, and accuracy on a single instance is 0 or 1,
so `|f(X) − f(X | ¬E_NL)| ∈ {0, 1}`. The smooth curves in Figures 3 and 8 are
averages over instances; the underlying signal per instance is a coin flip.
Any claim about an individual explanation's fidelity rests on one bit.

## L5. The metric saturates

Reported COPA fidelity reaches ≈0.85 after 20 steps (Figure 3, right). We
measured mean 0.880 (COPA-En) and 0.920 (XCOPA-vi) after only 8 steps with a
modern predictor. A metric already near ceiling has little headroom to rank
methods, which weakens the comparison against the baselines.

## L6. "Accuracy" is closer to a parse-success rate, and the parser is format-coupled

Scoring extracts the answer by looking for the `[choice]…@` markup that the
prompt itself injects, falling back to substring matching. This was written for
2023-era models (Vicuna-7B, Phi-2) that echo the format.

In our run, essentially every "wrong" prediction was a parsing failure, not a
reasoning failure:

| Dataset | Incorrect | Genuinely wrong choice | Unparseable |
|---|---|---|---|
| COPA-En | 4 | 0 | 4 |
| XCOPA-vi | 10 | 1 | 9 |

The framework is described as model-agnostic, but the measurement is coupled to
an output format that modern instruction-tuned models do not follow.

## L7. Early stopping means the advertised 20 optimisation steps are rarely reached

The loop stops at the first non-zero score, but the condition is only *evaluated*
every fifth iteration (`if iter%5 == 0 and sum(scores_list) != 0: break`). So the
number of iterations actually executed collapses onto a few values:

| Iterations run | COPA-En | XCOPA-vi |
|---|---|---|
| 1 | 41 | 52 |
| 6 | 44 | 44 |
| 8 (cap) | 15 | 4 |

Two problems. First, ~half the instances stop after a single iteration, so the
claim that "20 rounds of optimization are sufficient to converge" (§4.3) is
untested for them — they never iterate. Second, an instance that succeeds at
iteration 1 still runs to iteration 5 before the check fires, wasting four
optimisation rounds. Figure 3's averaged curve hides both effects.

## L8. Cross-lingual results are underspecified and degrade in practice

RQ2 (§4.4) transfers COPA → **XCOPA**, but the paper never states which of
XCOPA's 11 languages were used, reports no per-language breakdown, and gives no
translation-quality control. Language is treated as a single transfer axis.

Our Vietnamese run degraded measurably against English under identical settings:

- parse failures **9% vs 4%**,
- the global pipeline ran roughly **3× slower**, because Vietnamese prompts
  triggered provider content filtering far more often.

Neither effect is visible anywhere in the paper.

## L9. API refusals and empty completions are unaccounted for

The paper's explainers are API models (GPT-3.5-Turbo, Claude-2) and its datasets
contain violence and crime items — COPA includes premises about weapons. It
reports no refusal rate, no empty-response rate, and no handling policy.

This matters because an empty completion scores as a wrong answer, which
*manufactures* a fidelity difference. In our run **840 of 4287 calls (19.6%)**
returned empty from the primary model and had to be served by a fallback. Left
unhandled, that is a ~20% contamination channel flowing straight into the metric.

## L10. Variance is never reported

§4.2 states results are "the average scores of 3 times repetitions with the grid
search". No figure or table in the paper carries error bars, standard deviations,
or confidence intervals. Given L4 (binary per-instance signal) and a sampling
step of 15–30 instances, run-to-run variance is likely to be substantial and is
exactly what a reader needs to judge the gap against baselines.

## L11. The paper contradicts itself on the sampling size

§4.3 states "we randomly select **15** instances from the training dataset in
each optimization round". Appendix C and Table 2 both state **30**. The released
code defaults to `ques_sample=15`. A reproduction cannot tell which number the
reported results used.

## L12. The released code does not implement the paper's own baselines or its second metric

The repository at `github.com/ynchuang/FaithLM` contains no implementation of
SelfExp or Self-consistency, and no truthfulness evaluation (the GPT-4o /
RoBERTa-Large / XLNet-Large NLI comparison against gold explanations). Only the
fidelity half of the paper is runnable as published, so Figures 4, 7, 10 and 11
cannot be reproduced from the artifact.

## L13. Conclusions are drawn from small, now-superseded target models

Target models are Vicuna-7B and Phi-2 (2.7B). Both predate instruction-tuned
models with strong commonsense performance. On COPA our modern predictor was
effectively at ceiling — it never once selected the wrong option unaided. When
the target model already solves the task, "does a contrary hint flip it" becomes
a question about prompt-following rather than about explanation faithfulness,
and it is not obvious the paper's conclusions transfer.

---

## Which of these are usable as contributions

L3, L7, L8 and L9 are gaps a short paper can close cheaply and defensibly:

- **L3** — add a random-hint control arm; it is a few lines and directly tests
  whether the metric measures faithfulness or susceptibility.
- **L7** — move the early-stop check out of the `iter%5` gate and report the
  corrected iteration distribution.
- **L8** — report XCOPA per language, with Vietnamese named explicitly, plus a
  native Vietnamese dataset the paper never touches.
- **L9** — report the refusal/empty-completion rate as a first-class number.
