"""FaithLM — faithfulness evaluation for LLM explanations.

One entry point serves both the CLI and the notebook:

    from faithlm import run_experiment, load_config, Config

    run_experiment(load_config("configs/xcopa_vi_qwen_deepseek.yaml"))
    run_experiment(Config())                       # all defaults
    run_experiment(from_dict({"run": {"ques_idx_end": 5}}))
"""

from .config import Config, from_dict, load_config, load_dotenv_if_present
from .registry import available

# Import for side effects: each module registers its components on import.
from . import baselines, datasets, explainers, metrics, predictors  # noqa: F401
from .run import run_experiment

__all__ = [
    "Config",
    "from_dict",
    "load_config",
    "load_dotenv_if_present",
    "run_experiment",
    "available",
]

__version__ = "0.2.0"
