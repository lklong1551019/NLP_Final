"""Dataset loaders.

Each loader returns an Example list rather than the old defaultdict-of-lists.
Carrying the choices alongside the rendered prompt is what makes log-probability
scoring possible: the scorer needs the option strings, not just the text blob
they were formatted into.

The rendered `question` text keeps the exact `[choice]...@` layout of the
original code so that reproduction numbers stay comparable.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .registry import register_dataset


def _hf_load(*args, **kwargs):
    """Import `datasets` lazily.

    Keeping the heavy dependency out of module import means the registry, the
    metrics and the prompt code can be imported (and unit-tested) without a
    full deep-learning stack installed.
    """
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


@dataclass
class Example:
    question: str                       # fully rendered prompt body
    answer: str                         # gold answer text
    choices: List[str] = field(default_factory=list)
    passage: Optional[str] = None       # open-ended QA only
    answer_aliases: List[str] = field(default_factory=list)

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)


def _render_choices(choices: List[str]) -> str:
    return " ".join(f"[choice]{c}@" for c in choices)


def _render_copa_style(premise: str, question_type: str, choices: List[str]) -> str:
    return (
        f"###Question: What is the {question_type} of the Premise?\n"
        f"### Premise: {premise}\n"
        f"### Choices: {_render_choices(choices)}"
    )


@register_dataset("xcopa_vi")
def load_xcopa(lang: str = "vi", split: str = "test", **_) -> List[Example]:
    """XCOPA — 2-choice causal reasoning, 11 languages.

    Splits: validation (100), test (500).
    """
    data = _hf_load("cambridgeltl/xcopa", lang)[split]
    examples = []
    for row in data:
        choices = [row["choice1"], row["choice2"]]
        examples.append(
            Example(
                question=_render_copa_style(row["premise"], row["question"], choices),
                answer=choices[row["label"]],
                choices=choices,
            )
        )
    return examples


@register_dataset("copa_en")
def load_copa_en(split: str = "train", **_) -> List[Example]:
    """Balanced COPA — the English counterpart used as our cross-lingual control.

    Splits: train (1000), test (500).
    """
    data = _hf_load("pkavumba/balanced-copa")[split]
    examples = []
    for row in data:
        choices = [row["choice1"], row["choice2"]]
        examples.append(
            Example(
                question=_render_copa_style(row["premise"], row["question"], choices),
                answer=choices[row["label"]],
                choices=choices,
            )
        )
    return examples


@register_dataset("ecqa")
def load_ecqa(split: str = "train", **_) -> List[Example]:
    """ECQA — 5-choice commonsense QA."""
    data = _hf_load("yangdong/ecqa", "rc")[split]
    examples = []
    for row in data:
        choices = [row[f"q_op{i}"] for i in range(1, 6)]
        examples.append(
            Example(
                question=f"{row['q_text']}\n### Choices: {_render_choices(choices)}",
                answer=row["q_ans"],
                choices=choices,
            )
        )
    return examples


@register_dataset("social")
def load_social_iqa(split: str = "validation", **_) -> List[Example]:
    """Social IQa via BIG-bench — multi-choice social reasoning."""
    data = _hf_load("tasksource/bigbench", "social_iqa")[split]
    examples = []
    for row in data:
        choices = list(row["multiple_choice_targets"])
        examples.append(
            Example(
                question=f"{row['inputs']}\n### Choices: {_render_choices(choices)}",
                answer=row["targets"][0],
                choices=choices,
            )
        )
    return examples


@register_dataset("trivaqa")
def load_triviaqa(split: str = "test", **_) -> List[Example]:
    """TriviaQA via LongBench — open-ended QA over a passage.

    Has no choice list, so it can only be scored by exact match.
    """
    data = _hf_load("THUDM/LongBench", "triviaqa_e")[split]
    examples = []
    for row in data:
        text = row["input"].split("Passage:\n")[-1].split("Question:\n")
        passage = text[0]
        question = text[-1].split("Answer:\n")[0]
        aliases = list(row["answers"])
        examples.append(
            Example(
                question=question,
                answer=aliases[0] if aliases else "",
                answer_aliases=aliases,
                passage=passage,
            )
        )
    return examples


def load(name: str, lang: str = "vi", split: str = "test") -> List[Example]:
    from .registry import get

    return get("dataset", name)(lang=lang, split=split)
