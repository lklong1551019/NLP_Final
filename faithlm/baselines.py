"""Non-LLM baselines for the counterfactual step.

The assignment asks for a comparison against a baseline model, and these also
answer a real question about the method: does the faithfulness score actually
require an LLM to write the counterfactual, or would a cheap negation do?

Each baseline exposes the same `respond(prompt)` interface as an explainer, so
it drops straight into the pipelines without special-casing.
"""

import random
import re
from typing import List

from .registry import register_explainer

_NEGATION_MAP = [
    (r"\bis\b", "is not"),
    (r"\bare\b", "are not"),
    (r"\bwas\b", "was not"),
    (r"\bwere\b", "were not"),
    (r"\bcan\b", "cannot"),
    (r"\bwill\b", "will not"),
    (r"\bdoes\b", "does not"),
    (r"\bdid\b", "did not"),
    (r"\bhas\b", "has not"),
    (r"\bhave\b", "have not"),
    (r"\bbecause\b", "despite the fact that"),
    (r"\bcaused\b", "did not cause"),
    (r"\bled to\b", "prevented"),
]


class NegationBaseline:
    """Rule-based negation — flips meaning with no model call at all."""

    def respond(self, prompt: str) -> str:
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        # Recover the sentence the prompt is asking us to transform.
        text = prompt
        for marker in ("Sentences: ", "Sentence-1: "):
            if marker in prompt:
                text = prompt.split(marker, 1)[1]
                break
        text = text.strip()

        for pattern, replacement in _NEGATION_MAP:
            new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            if n:
                return new_text
        return f"It is not the case that {text[0].lower()}{text[1:]}" if text else ""


class ShuffleBaseline:
    """A random unrelated sentence — the floor any real method must beat."""

    _POOL = [
        "The weather was pleasant throughout the afternoon.",
        "Several books were arranged neatly on the wooden shelf.",
        "The train arrived at the station ahead of schedule.",
        "A small crowd gathered near the entrance of the building.",
        "The recipe calls for three tablespoons of olive oil.",
    ]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def respond(self, prompt: str) -> str:
        return self._rng.choice(self._POOL)


class IdentityBaseline:
    """Returns the explanation unchanged — the counterfactual is not a counterfactual.

    Faithfulness should collapse toward zero here. If it does not, the metric is
    measuring prompt perturbation rather than meaning reversal, which is worth
    reporting either way.
    """

    def respond(self, prompt: str) -> str:
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        for marker in ("Sentences: ", "Sentence-1: "):
            if marker in prompt:
                return prompt.split(marker, 1)[1].strip()
        return prompt.strip()


@register_explainer("baseline_negation")
def _build_negation(cfg):
    return NegationBaseline()


@register_explainer("baseline_shuffle")
def _build_shuffle(cfg):
    return ShuffleBaseline()


@register_explainer("baseline_identity")
def _build_identity(cfg):
    return IdentityBaseline()
