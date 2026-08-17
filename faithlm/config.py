"""Typed configuration objects, loadable from YAML or from a plain dict.

The dict path matters as much as the YAML one: notebooks on Colab/Kaggle build
a config inline without touching the filesystem, which keeps them free of the
local paths the assignment forbids.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import copy
import os


@dataclass
class DatasetConfig:
    name: str = "xcopa_vi"
    lang: str = "vi"
    split: str = "test"


@dataclass
class PredictorConfig:
    name: str = "qwen"
    model_id: Optional[str] = None      # None -> the backend's default
    load_in_4bit: bool = False          # Kaggle T4 has 16GB; keep fp16 by default
    max_new_tokens: int = 256
    temperature: float = 0.7
    max_memory_per_gpu: str = "14GiB"
    device_num: list = field(default_factory=lambda: [0])


@dataclass
class ExplainerConfig:
    name: str = "deepseek"
    model_id: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.9
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url_env: str = "DEEPSEEK_BASE_URL"


@dataclass
class MetricConfig:
    # "paper" reproduces the original |acc(no-hint) - acc(cf-hint)|;
    # "symmetric" is our corrected |acc(true-exp) - acc(cf-exp)|.
    name: str = "paper"
    # "logprob" scores continuously from choice log-probabilities;
    # "exact_match" keeps the original string-matching behaviour.
    scorer: str = "logprob"
    k_samples: int = 1                  # only used by the exact_match scorer


@dataclass
class RunConfig:
    pipeline: str = "local"             # "local" or "global"
    ques_idx_start: int = 0
    ques_idx_end: int = 50
    # "sequential" takes examples[start:end]; "random" draws (end - start)
    # examples from the whole split under `seed`. Sequential inherits whatever
    # ordering the source dataset has, so a subset of a few hundred questions
    # should normally be drawn at random to stay representative.
    sampling: str = "sequential"
    xai_iter: int = 15
    round_xai_iter: int = 10            # global only
    ques_sample: int = 15               # global only
    output_dir: str = "./results"
    resume: bool = True                 # skip questions whose result file exists
    seed: int = 42


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def variant_id(self) -> str:
        """Stable identifier used for output directories and result filenames.

        The pipeline is part of the id: a local and a global run of otherwise
        identical settings are different experiments and must not share a
        results directory.
        """
        return (
            f"{self.run.pipeline}_{self.dataset.name}_{self.predictor.name}"
            f"_{self.explainer.name}_{self.metric.name}_{self.metric.scorer}"
        )


_SECTIONS = {
    "dataset": DatasetConfig,
    "predictor": PredictorConfig,
    "explainer": ExplainerConfig,
    "metric": MetricConfig,
    "run": RunConfig,
}


def from_dict(raw: Optional[Dict[str, Any]] = None, **overrides) -> Config:
    """Build a Config from a nested dict, rejecting unknown keys.

    Typos in a config file are silent killers in long experiment runs, so an
    unrecognised key is an error rather than something quietly ignored.
    """
    raw = copy.deepcopy(raw or {})
    for section, value in overrides.items():
        if section not in _SECTIONS:
            raise KeyError(f"Unknown config section '{section}'. Valid: {sorted(_SECTIONS)}")
        raw.setdefault(section, {}).update(value)

    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise KeyError(
            f"Unknown config section(s): {sorted(unknown_sections)}. Valid: {sorted(_SECTIONS)}"
        )

    sections = {}
    for section, cls in _SECTIONS.items():
        values = raw.get(section) or {}
        if not isinstance(values, dict):
            raise TypeError(f"Config section '{section}' must be a mapping, got {type(values).__name__}")
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(values) - valid_fields
        if unknown:
            raise KeyError(
                f"Unknown key(s) in '{section}': {sorted(unknown)}. Valid: {sorted(valid_fields)}"
            )
        sections[section] = cls(**values)

    return Config(**sections)


def load_config(path: str, **overrides) -> Config:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return from_dict(raw, **overrides)


def load_dotenv_if_present() -> None:
    """Load a .env file when python-dotenv is installed; a no-op on Kaggle."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def require_env(name: str, hint: str = "") -> str:
    value = os.environ.get(name)
    if not value:
        suffix = f" {hint}" if hint else ""
        raise EnvironmentError(f"Environment variable {name} is not set.{suffix}")
    return value
