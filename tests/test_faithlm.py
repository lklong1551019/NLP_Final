"""Unit tests covering the logic that does not need a GPU or an API key.

Run with:  python -m pytest tests/ -v
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithlm.baselines import IdentityBaseline, NegationBaseline, ShuffleBaseline
from faithlm.config import Config, from_dict
from faithlm.datasets import Example
from faithlm.explainers import is_refusal, parse_explanations, parse_instruction
from faithlm.metrics import _softmax_gold_prob, paper_metric, symmetric_metric
from faithlm.predictors import extract_choice
from faithlm.prompts import task_prompt


# ----------------------------------------------------------------- config


def test_unknown_section_rejected():
    with pytest.raises(KeyError, match="Unknown config section"):
        from_dict({"predictorr": {"name": "qwen"}})


def test_unknown_key_rejected():
    with pytest.raises(KeyError, match="Unknown key"):
        from_dict({"predictor": {"nmae": "qwen"}})


def test_variant_id_is_stable():
    cfg = Config()
    assert cfg.variant_id() == "local_xcopa_vi_qwen_deepseek_paper_logprob"


def test_variant_id_separates_pipelines():
    """A local and a global run must not overwrite each other's results."""
    local = from_dict({"run": {"pipeline": "local"}})
    glob = from_dict({"run": {"pipeline": "global"}})
    assert local.variant_id() != glob.variant_id()


def test_overrides_merge():
    cfg = from_dict({"run": {"xai_iter": 3}}, predictor={"name": "phi"})
    assert cfg.run.xai_iter == 3
    assert cfg.predictor.name == "phi"


# ----------------------------------------------------------------- prompts


def test_hint_appears_only_when_given():
    without = task_prompt("Pick one.", "Q?")
    with_hint = task_prompt("Pick one.", "Q?", hint="Because X.")
    assert "### Hint:" not in without
    assert "### Hint: Because X." in with_hint
    # The hint must precede the input, matching the original layout.
    assert with_hint.index("### Hint:") < with_hint.index("### Input:")


# --------------------------------------------------------------- parsing


def test_parse_explanations_never_returns_empty():
    for bad in ["", "   ", "\n\n\n"]:
        assert parse_explanations(bad) == [""]


def test_parse_explanations_strips_markers_and_preamble():
    reply = "Here is my explanation:\n\n<EXP>The model chose A because of X.</EXP>"
    assert parse_explanations(reply) == ["The model chose A because of X."]


def test_parse_explanations_splits_paragraphs():
    assert len(parse_explanations("First one.\n\nSecond one.")) == 2


def test_parse_instruction_extracts_ins_body():
    assert parse_instruction("blah <INS>Do the thing.</INS> trailing") == "Do the thing."
    assert parse_instruction("no markers here") == "no markers here"


def test_refusal_detection():
    assert is_refusal("I apologize, but I cannot help with that.")
    assert not is_refusal("The model selected the second choice.")


# ---------------------------------------------------------- choice parsing


def test_extract_choice_from_markup():
    choices = ["he ate lunch", "he went home"]
    assert extract_choice("The answer is [choice]he went home@", choices) == "he went home"


def test_extract_choice_from_verbose_cot():
    """Regression: the pre-refactor bug was verbose CoT never matching."""
    choices = ["anh ấy ăn trưa", "anh ấy về nhà"]
    verbose = "Let's think step by step. The premise suggests... Therefore anh ấy về nhà."
    assert extract_choice(verbose, choices) == "anh ấy về nhà"


def test_extract_choice_returns_none_when_absent():
    assert extract_choice("I am not sure about this.", ["alpha", "beta"]) is None


# ------------------------------------------------------------- log-probs


def test_softmax_gold_prob_is_continuous():
    """The whole point of log-prob scoring: values strictly between 0 and 1."""
    p = _softmax_gold_prob([-1.0, -2.0], gold_index=0)
    assert 0.0 < p < 1.0
    assert math.isclose(p, math.exp(-1.0) / (math.exp(-1.0) + math.exp(-2.0)))


def test_softmax_gold_prob_handles_ties():
    assert math.isclose(_softmax_gold_prob([-1.0, -1.0], 0), 0.5)


def test_softmax_gold_prob_survives_infinities():
    p = _softmax_gold_prob([-math.inf, -1.0], gold_index=1)
    assert math.isclose(p, 1.0)
    assert _softmax_gold_prob([-math.inf, -math.inf], 0) == 0.0


# ---------------------------------------------------------------- metrics


class _StubPredictor:
    """Returns a preset probability per hint, so metric wiring can be tested."""

    supports_logprobs = True

    def __init__(self, by_hint):
        self.by_hint = by_hint
        self.seen = []

    def choice_logprobs(self, prompt, choices):
        hint = None
        if "### Hint:" in prompt:
            hint = prompt.split("### Hint:", 1)[1].split("\n", 1)[0].strip()
        self.seen.append(hint)
        prob = self.by_hint[hint]
        # Encode the target probability as a two-choice log-prob pair.
        return [math.log(max(prob, 1e-9)), math.log(max(1 - prob, 1e-9))]


def _example():
    return Example(question="Q?", answer="A", choices=["A", "B"])


def test_paper_metric_leaves_true_arm_unhinted():
    """Documents the original behaviour: the true explanation is never shown."""
    predictor = _StubPredictor({None: 0.9, "CF": 0.2})
    detail = paper_metric(predictor, _example(), "Pick one.", "TRUE", "CF", scorer="logprob")
    assert None in predictor.seen        # baseline arm got no hint at all
    assert "TRUE" not in predictor.seen  # the true explanation went unused
    assert math.isclose(detail.faithfulness, 0.7, abs_tol=1e-6)


def test_symmetric_metric_hints_both_arms():
    predictor = _StubPredictor({"TRUE": 0.9, "CF": 0.2})
    detail = symmetric_metric(predictor, _example(), "Pick one.", "TRUE", "CF", scorer="logprob")
    assert set(predictor.seen) == {"TRUE", "CF"}
    assert math.isclose(detail.faithfulness, 0.7, abs_tol=1e-6)


def test_metrics_disagree_when_hint_itself_helps():
    """The two metrics are genuinely different, which is why we report both."""
    predictor_paper = _StubPredictor({None: 0.5, "CF": 0.5})
    predictor_symm = _StubPredictor({"TRUE": 0.95, "CF": 0.5})
    paper = paper_metric(predictor_paper, _example(), "T", "TRUE", "CF", scorer="logprob")
    symm = symmetric_metric(predictor_symm, _example(), "T", "TRUE", "CF", scorer="logprob")
    assert math.isclose(paper.faithfulness, 0.0, abs_tol=1e-6)
    assert symm.faithfulness > 0.4


# -------------------------------------------------------------- baselines


def test_negation_baseline_flips_meaning():
    out = NegationBaseline().respond("Sentences: The model is confident about this.")
    assert "is not" in out


def test_negation_baseline_falls_back_when_no_pattern():
    out = NegationBaseline().respond("Sentences: Rain fell steadily.")
    assert out.startswith("It is not the case that")


def test_identity_baseline_returns_input_unchanged():
    out = IdentityBaseline().respond("Sentences: The model chose A.")
    assert out == "The model chose A."


def test_shuffle_baseline_is_deterministic_per_seed():
    assert ShuffleBaseline(seed=1).respond("x") == ShuffleBaseline(seed=1).respond("x")


# --------------------------------------------------------------- registry


def test_all_expected_components_registered():
    import faithlm  # noqa: F401  (import triggers registration)
    from faithlm.registry import available

    assert {"xcopa_vi", "copa_en", "ecqa"} <= set(available("dataset"))
    assert {"qwen", "phi", "hf"} <= set(available("predictor"))
    assert {"deepseek", "baseline_negation", "baseline_identity"} <= set(available("explainer"))
    assert {"paper", "symmetric"} <= set(available("metric"))


def test_duplicate_registration_is_rejected():
    from faithlm.registry import register_metric

    with pytest.raises(KeyError, match="already registered"):
        register_metric("paper")(lambda *a, **k: None)
