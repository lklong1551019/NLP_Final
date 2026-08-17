"""Faithfulness metrics.

Two variants, selected by `metric.name`:

  paper      |acc(no hint) - acc(counterfactual hint)|
             Reproduces the released implementation. Note this compares an
             unhinted baseline against a counterfactual-hinted run: the true
             explanation is generated but never actually shown to the predictor.

  symmetric  |acc(true-explanation hint) - acc(counterfactual hint)|
             Both arms receive a hint, so the score isolates the effect of
             flipping the explanation's meaning rather than the effect of
             adding a hint at all. This is what the paper's prose describes.

Two scorers, selected by `metric.scorer`:

  logprob      P(gold choice) under a softmax over the choices' length-normalised
               log-probabilities. Continuous on a single question.
  exact_match  1.0/0.0 by string matching, as in the original code.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from .datasets import Example
from .predictors import Predictor, extract_choice
from .prompts import task_prompt
from .registry import register_metric


@dataclass
class ScoreDetail:
    """Everything a single faithfulness computation produced, for error analysis."""
    faithfulness: float
    true_arm: float
    counter_arm: float
    predicted_choice: Optional[str] = None
    abstained: bool = False


def _softmax_gold_prob(logprobs: List[float], gold_index: int) -> float:
    """Convert choice log-probs into P(gold), which is continuous in [0, 1]."""
    finite = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite:
        return 0.0
    peak = max(finite)
    exps = [math.exp(lp - peak) if math.isfinite(lp) else 0.0 for lp in logprobs]
    total = sum(exps)
    if total <= 0:
        return 0.0
    return exps[gold_index] / total


def score_arm(predictor: Predictor, example: Example, task_instruction: str,
              hint: Optional[str], scorer: str, k_samples: int = 1,
              max_new_tokens: int = 256, temperature: float = 0.7) -> ScoreDetail:
    """Score one arm of the comparison: how well the predictor does under `hint`."""
    prompt = task_prompt(
        task_instruction, example.question, hint=hint,
        passage=example.passage, cot=(scorer != "logprob"),
    )

    if scorer == "logprob":
        if not example.is_multiple_choice:
            raise ValueError(
                "logprob scoring needs answer choices; use metric.scorer='exact_match' "
                f"for open-ended datasets."
            )
        gold_index = example.choices.index(example.answer)
        logprobs = predictor.choice_logprobs(prompt, example.choices)
        prob = _softmax_gold_prob(logprobs, gold_index)
        best = example.choices[max(range(len(logprobs)), key=lambda i: logprobs[i])]
        return ScoreDetail(faithfulness=0.0, true_arm=prob, counter_arm=0.0,
                           predicted_choice=best)

    # exact_match: optionally average over k stochastic samples.
    hits, predicted, abstentions = 0, None, 0
    for _ in range(max(1, k_samples)):
        output = predictor.generate([prompt], max_new_tokens=max_new_tokens,
                                    temperature=temperature)[0]
        if example.is_multiple_choice:
            choice = extract_choice(output, example.choices)
            predicted = choice if choice else predicted
            if choice is None:
                abstentions += 1
            elif choice == example.answer:
                hits += 1
        else:
            aliases = example.answer_aliases or [example.answer]
            if any(a.lower() in output.lower() for a in aliases if a):
                hits += 1

    n = max(1, k_samples)
    return ScoreDetail(faithfulness=0.0, true_arm=hits / n, counter_arm=0.0,
                       predicted_choice=predicted, abstained=(abstentions == n))


@register_metric("paper")
def paper_metric(predictor, example, task_instruction, true_exp, counter_exp,
                 scorer="logprob", **kwargs) -> ScoreDetail:
    """|acc(no hint) - acc(counterfactual hint)| — the released implementation."""
    baseline = score_arm(predictor, example, task_instruction, None, scorer, **kwargs)
    counter = score_arm(predictor, example, task_instruction, counter_exp, scorer, **kwargs)
    return ScoreDetail(
        faithfulness=abs(baseline.true_arm - counter.true_arm),
        true_arm=baseline.true_arm,
        counter_arm=counter.true_arm,
        predicted_choice=baseline.predicted_choice,
        abstained=baseline.abstained,
    )


@register_metric("symmetric")
def symmetric_metric(predictor, example, task_instruction, true_exp, counter_exp,
                     scorer="logprob", **kwargs) -> ScoreDetail:
    """|acc(true-exp hint) - acc(counterfactual hint)| — both arms hinted."""
    true = score_arm(predictor, example, task_instruction, true_exp, scorer, **kwargs)
    counter = score_arm(predictor, example, task_instruction, counter_exp, scorer, **kwargs)
    return ScoreDetail(
        faithfulness=abs(true.true_arm - counter.true_arm),
        true_arm=true.true_arm,
        counter_arm=counter.true_arm,
        predicted_choice=true.predicted_choice,
        abstained=true.abstained,
    )


def compute(cfg_metric, predictor, example, task_instruction, true_exp, counter_exp,
            **kwargs) -> ScoreDetail:
    from .registry import get

    fn = get("metric", cfg_metric.name)
    return fn(predictor, example, task_instruction, true_exp, counter_exp,
              scorer=cfg_metric.scorer, k_samples=cfg_metric.k_samples, **kwargs)
