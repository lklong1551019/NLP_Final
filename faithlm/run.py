"""Single entry point shared by the CLI and the notebooks."""

import argparse
import json
import os
import random
from typing import Dict, List, Optional

from . import datasets as datasets_mod
from . import explainers as explainer_mod
from . import pipelines
from . import predictors as predictor_mod
from .config import Config, from_dict, load_config, load_dotenv_if_present


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _validate(cfg: Config, examples: List) -> None:
    """Fail fast on combinations that cannot work, before any model loads."""
    from .prompts import check_lang

    check_lang(cfg.run.prompt_lang)
    if cfg.metric.scorer == "logprob" and cfg.metric.name != "flip":
        if examples and not examples[0].is_multiple_choice:
            raise ValueError(
                f"metric.scorer='logprob' needs answer choices, but dataset "
                f"'{cfg.dataset.name}' is open-ended. Use scorer='exact_match'."
            )
    if cfg.metric.name == "flip" and examples and not examples[0].is_multiple_choice:
        raise ValueError(
            f"metric 'flip' needs answer choices, but dataset "
            f"'{cfg.dataset.name}' is open-ended."
        )
    if cfg.run.pipeline not in ("local", "global", "selfcons"):
        raise ValueError(
            f"run.pipeline must be 'local', 'global' or 'selfcons', got '{cfg.run.pipeline}'"
        )
    if cfg.run.holdout_split and cfg.run.pipeline != "global":
        raise ValueError("run.holdout_split only applies to the global pipeline")


def run_experiment(cfg: Optional[Config] = None, predictor=None, explainer=None) -> Dict:
    """Run one experiment variant end to end and return a summary dict.

    Pass `predictor`/`explainer` to reuse already-loaded models across several
    variants — a notebook sweeping four metric settings should load the 4B model
    once, not four times.
    """
    cfg = cfg or Config()
    load_dotenv_if_present()
    _set_seed(cfg.run.seed)

    print(f"[run] variant: {cfg.variant_id()}")
    print(f"[run] pipeline={cfg.run.pipeline} metric={cfg.metric.name} scorer={cfg.metric.scorer}")

    examples = datasets_mod.load(cfg.dataset.name, cfg.dataset.lang, cfg.dataset.split)
    print(f"[run] loaded {len(examples)} examples from {cfg.dataset.name}/{cfg.dataset.split}")
    _validate(cfg, examples)

    if predictor is None:
        predictor = predictor_mod.build(cfg.predictor)
    # `flip` never touches log-probabilities, whatever the scorer field says.
    if (cfg.metric.scorer == "logprob" and cfg.metric.name != "flip"
            and not predictor.supports_logprobs):
        raise ValueError(
            f"predictor '{cfg.predictor.name}' cannot produce log-probabilities. "
            "Use metric.scorer='exact_match', metric.name='flip', or a local HF predictor."
        )
    if explainer is None:
        explainer = explainer_mod.build(cfg.explainer)

    if cfg.run.pipeline == "local":
        results = pipelines.run_local(cfg, examples, predictor, explainer)
        scored = [r for r in results if not r.get("error")]
        summary = {
            "variant": cfg.variant_id(),
            "pipeline": "local",
            "questions": len(results),
            "failed": len(results) - len(scored),
            "task_accuracy": (
                sum(1 for r in scored if r.get("correct")) / len(scored) if scored else 0.0
            ),
            "mean_best_faithfulness": (
                sum(r.get("best_score", 0.0) for r in scored) / len(scored) if scored else 0.0
            ),
        }
    elif cfg.run.pipeline == "selfcons":
        results = pipelines.run_selfcons(cfg, examples, predictor, explainer)
        scored = [r for r in results if not r.get("error")]
        summary = {
            "variant": cfg.variant_id(),
            "pipeline": "selfcons",
            "questions": len(results),
            "failed": len(results) - len(scored),
            "task_accuracy": (
                sum(1 for r in scored if r.get("correct")) / len(scored) if scored else 0.0
            ),
            "mean_faithfulness": (
                sum(r.get("score", 0.0) for r in scored) / len(scored) if scored else 0.0
            ),
        }
    else:
        holdout = None
        if cfg.run.holdout_split:
            holdout = datasets_mod.load(cfg.dataset.name, cfg.dataset.lang,
                                        cfg.run.holdout_split)
            # A fixed, seed-independent slice keeps the pool identical across
            # machines; the per-round subsample is what the seed controls.
            holdout = holdout[:max(cfg.run.holdout_size, cfg.run.ques_sample)]
            print(f"[run] holdout: {len(holdout)} examples from "
                  f"{cfg.dataset.name}/{cfg.run.holdout_split}")
        rounds = pipelines.run_global(cfg, examples, predictor, explainer,
                                      holdout=holdout)
        summary = {
            "variant": cfg.variant_id(),
            "pipeline": "global",
            "rounds": len(rounds),
            "best_score": max((r["score"] for r in rounds), default=0.0),
        }

    out_dir = os.path.join(cfg.run.output_dir, cfg.variant_id())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"summary_{cfg.run.pipeline}.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "summary": summary}, f, ensure_ascii=False, indent=2)

    print(f"[run] done: {json.dumps(summary, ensure_ascii=False)}")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run a FaithLM experiment variant.")
    parser.add_argument("--config", type=str, help="Path to a YAML config file")
    parser.add_argument("--pipeline", choices=["local", "global", "selfcons"])
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--predictor", type=str)
    parser.add_argument("--predictor_model_id", type=str)
    parser.add_argument("--explainer", type=str)
    parser.add_argument("--explainer_model_id", type=str)
    parser.add_argument("--metric", type=str, help="paper | symmetric")
    parser.add_argument("--scorer", type=str, help="logprob | exact_match")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--sampling", choices=["sequential", "random"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--xai_iter", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--ques_sample", type=int)
    parser.add_argument("--prompt_lang", type=str, help="en | vi — language of all prompt templates")
    parser.add_argument("--holdout_split", type=str,
                        help="global only: optimise on this split, then transfer-eval on the main split")
    parser.add_argument("--holdout_size", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--load_in_4bit", action="store_true", default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--list", action="store_true", help="List registered components and exit")
    args = parser.parse_args(argv)

    if args.list:
        from .registry import available

        for kind in ("dataset", "predictor", "explainer", "metric"):
            print(f"{kind:10s}: {', '.join(available(kind))}")
        return

    cfg = load_config(args.config) if args.config else Config()

    # CLI flags override the file, so a config can be reused with small tweaks.
    overrides = {
        "dataset": {"name": args.dataset, "lang": args.lang, "split": args.split},
        "predictor": {"name": args.predictor, "model_id": args.predictor_model_id,
                      "load_in_4bit": args.load_in_4bit},
        "explainer": {"name": args.explainer, "model_id": args.explainer_model_id},
        "metric": {"name": args.metric, "scorer": args.scorer},
        "run": {"pipeline": args.pipeline, "ques_idx_start": args.start,
                "ques_idx_end": args.end, "xai_iter": args.xai_iter,
                "round_xai_iter": args.rounds, "ques_sample": args.ques_sample,
                "prompt_lang": args.prompt_lang, "holdout_split": args.holdout_split,
                "holdout_size": args.holdout_size,
                "output_dir": args.output_dir, "sampling": args.sampling,
                "seed": args.seed,
                "resume": False if args.no_resume else None},
    }
    for section, values in overrides.items():
        target = getattr(cfg, section)
        for key, value in values.items():
            if value is not None:
                setattr(target, key, value)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
