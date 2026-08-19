"""The local, global and self-consistency FaithLM pipelines.

All pipelines write one JSON file per unit of work (a question for local and
selfcons, a round for global) and skip units that already have a result file
when `run.resume` is on. That keeps a run restartable across Kaggle's 12-hour
session limit without losing finished work.
"""

import json
import os
import random
import time
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from . import explainers as explainer_mod
from . import metrics as metrics_mod
from . import predictors as predictor_mod
from .datasets import Example
from .prompts import (
    counterfactual_prompt,
    exp_instruction_for,
    explanation_prompt,
    global_optimizer_prompt,
    local_optimizer_prompt,
    task_instruction_for,
    task_prompt,
)


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic: a killed session never leaves half a file


def select_indices(cfg, n_examples: int) -> List[int]:
    """Choose which questions to run.

    Result files are named by the dataset index, so a resumed run must produce
    the same indices as the run it continues. Both modes are deterministic
    given the config, and `random` additionally depends only on `run.seed` —
    everyone sharing a config and a seed evaluates exactly the same questions
    without anyone shipping a filtered copy of the data.
    """
    start = max(0, cfg.run.ques_idx_start)
    end = min(cfg.run.ques_idx_end, n_examples)
    count = max(0, end - start)

    if cfg.run.sampling == "sequential":
        return list(range(start, end))
    if cfg.run.sampling == "random":
        rng = random.Random(cfg.run.seed)
        return sorted(rng.sample(range(n_examples), min(count, n_examples)))
    raise ValueError(
        f"run.sampling must be 'sequential' or 'random', got '{cfg.run.sampling}'"
    )


def _answer_question(cfg, example: Example, predictor, task_instruction: str,
                     lang: str, cot: bool = True):
    """Ask the predictor and parse the answer.

    Returns (raw_output, model_answer, parsed_choice, is_correct); for
    open-ended datasets parsed_choice is None and correctness is alias-based.
    """
    prompt = task_prompt(task_instruction, example.question,
                         passage=example.passage, cot=cot, lang=lang)
    raw = predictor.generate(
        [prompt],
        max_new_tokens=cfg.predictor.max_new_tokens,
        temperature=cfg.predictor.temperature,
    )[0]

    if example.is_multiple_choice:
        parsed = predictor_mod.extract_choice(raw, example.choices)
        model_answer = parsed if parsed else raw.strip()[:200]
        is_correct = parsed == example.answer
        return raw, model_answer, parsed, is_correct

    aliases = example.answer_aliases or [example.answer]
    is_correct = any(a.lower() in raw.lower() for a in aliases if a)
    return raw, raw.strip()[:200], None, is_correct


# ------------------------------------------------------------------- local


def run_local(cfg, examples: List[Example], predictor, explainer) -> List[Dict]:
    """Refine the explanation text per question, maximising faithfulness."""
    out_dir = os.path.join(cfg.run.output_dir, cfg.variant_id(), "local")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    indices = select_indices(cfg, len(examples))

    for idx in tqdm(indices, desc="questions"):
        result_path = os.path.join(out_dir, f"sample-{idx}.json")

        if cfg.run.resume and os.path.isfile(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        example = examples[idx]
        started = time.time()

        try:
            record = _run_local_one(cfg, example, idx, predictor, explainer)
        except Exception as exc:
            # One bad question must not end a multi-hour run.
            print(f"[local] question {idx} failed: {exc}")
            record = {"index": idx, "error": str(exc), "iterations": []}

        record["elapsed_sec"] = round(time.time() - started, 2)
        _write_json(result_path, record)
        results.append(record)

    return results


def _run_local_one(cfg, example: Example, idx: int, predictor, explainer) -> Dict:
    lang = cfg.run.prompt_lang
    task_instruction = task_instruction_for(example.is_multiple_choice, lang)

    # 1. What does the predictor actually answer?
    _, model_answer, parsed_choice, is_correct = _answer_question(
        cfg, example, predictor, task_instruction, lang
    )

    # 2. Explain that answer, then iteratively improve the explanation.
    exp_prompt = explanation_prompt(exp_instruction_for(lang), example.question,
                                    model_answer, passage=example.passage)
    explanations = explainer_mod.parse_explanations(explainer.respond(exp_prompt))
    current = explanations[0]

    history_exps: List[str] = []
    history_scores: List[float] = []
    iterations: List[Dict] = []

    for step in range(cfg.run.xai_iter):
        counter_reply = explainer.respond(
            counterfactual_prompt(current, open_ended=not example.is_multiple_choice,
                                  lang=lang)
        )
        counter = explainer_mod.parse_explanations(counter_reply)[0]

        detail = metrics_mod.compute(
            cfg.metric, predictor, example, task_instruction, current, counter,
            max_new_tokens=cfg.predictor.max_new_tokens,
            temperature=cfg.predictor.temperature,
            base_answer=parsed_choice, lang=lang,
        )

        history_exps.append(current)
        history_scores.append(detail.faithfulness)
        iterations.append({
            "step": step,
            "score": round(detail.faithfulness, 4),
            "true_arm": round(detail.true_arm, 4),
            "counter_arm": round(detail.counter_arm, 4),
            "explanation": current,
            "counterfactual": counter,
        })

        if step == cfg.run.xai_iter - 1:
            break

        optimizer_prompt = local_optimizer_prompt(
            history_exps, history_scores, example.question, model_answer, lang=lang
        )
        reply = explainer.respond(optimizer_prompt)
        if explainer_mod.is_refusal(reply):
            iterations[-1]["stopped"] = "explainer_refusal"
            break
        current = explainer_mod.parse_explanations(reply)[0]
        if not current:
            iterations[-1]["stopped"] = "empty_explanation"
            break

    best = max(iterations, key=lambda it: it["score"]) if iterations else None
    return {
        "index": idx,
        "question": example.question,
        "gold_answer": example.answer,
        "model_answer": model_answer,
        "correct": is_correct,
        "metric": cfg.metric.name,
        "scorer": cfg.metric.scorer,
        "iterations": iterations,
        "best_score": best["score"] if best else 0.0,
        "best_explanation": best["explanation"] if best else "",
    }


# --------------------------------------------------------------- selfcons


def run_selfcons(cfg, examples: List[Example], predictor, explainer) -> List[Dict]:
    """Self-consistency baseline: the predictor's own CoT as the explanation.

    The paper's section 2.2 argues a chain-of-thought is not a faithful account
    of why the model answered; this pipeline measures that claim directly by
    scoring the CoT with the same counterfactual procedure as FaithLM. One
    score per question, no optimisation loop.
    """
    out_dir = os.path.join(cfg.run.output_dir, cfg.variant_id(), "selfcons")
    os.makedirs(out_dir, exist_ok=True)
    lang = cfg.run.prompt_lang

    results = []
    for idx in tqdm(select_indices(cfg, len(examples)), desc="questions"):
        result_path = os.path.join(out_dir, f"sample-{idx}.json")
        if cfg.run.resume and os.path.isfile(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        example = examples[idx]
        task_instruction = task_instruction_for(example.is_multiple_choice, lang)
        started = time.time()
        try:
            raw, model_answer, parsed_choice, is_correct = _answer_question(
                cfg, example, predictor, task_instruction, lang, cot=True
            )
            # The whole CoT is the "explanation"; cap it so the counterfactual
            # prompt stays inside the explainer's context.
            explanation = raw.strip()[:1500]
            counter = explainer_mod.parse_explanations(
                explainer.respond(counterfactual_prompt(
                    explanation, open_ended=not example.is_multiple_choice, lang=lang))
            )[0]
            detail = metrics_mod.compute(
                cfg.metric, predictor, example, task_instruction, explanation, counter,
                max_new_tokens=cfg.predictor.max_new_tokens,
                temperature=cfg.predictor.temperature,
                base_answer=parsed_choice, lang=lang,
            )
            record = {
                "index": idx,
                "question": example.question,
                "gold_answer": example.answer,
                "model_answer": model_answer,
                "correct": is_correct,
                "metric": cfg.metric.name,
                "scorer": cfg.metric.scorer,
                "score": round(detail.faithfulness, 4),
                "abstained": detail.abstained,
                "explanation": explanation,
                "counterfactual": counter,
            }
        except Exception as exc:
            print(f"[selfcons] question {idx} failed: {exc}")
            record = {"index": idx, "error": str(exc)}

        record["elapsed_sec"] = round(time.time() - started, 2)
        _write_json(result_path, record)
        results.append(record)

    return results


# ------------------------------------------------------------------ global


def _score_instruction_once(cfg, example: Example, instruction: str,
                            predictor, explainer, lang: str) -> float:
    """One-shot faithfulness of `instruction` on one question (no refinement)."""
    task_instruction = task_instruction_for(example.is_multiple_choice, lang)
    _, answer, parsed_choice, _ = _answer_question(
        cfg, example, predictor, task_instruction, lang
    )
    exp = explainer_mod.parse_explanations(
        explainer.respond(explanation_prompt(
            instruction, example.question, answer, passage=example.passage))
    )[0]
    counter = explainer_mod.parse_explanations(
        explainer.respond(counterfactual_prompt(
            exp, open_ended=not example.is_multiple_choice, lang=lang))
    )[0]
    detail = metrics_mod.compute(
        cfg.metric, predictor, example, task_instruction, exp, counter,
        max_new_tokens=cfg.predictor.max_new_tokens,
        temperature=cfg.predictor.temperature,
        base_answer=parsed_choice, lang=lang,
    )
    return detail.faithfulness


def _mean_instruction_score(cfg, pool: List[Example], indices: List[int],
                            instruction: str, predictor, explainer, lang: str,
                            tag: str) -> Optional[float]:
    total, counted = 0.0, 0
    for idx in indices:
        try:
            total += _score_instruction_once(cfg, pool[idx], instruction,
                                             predictor, explainer, lang)
            counted += 1
        except Exception as exc:
            print(f"[global] {tag}, question {idx} failed: {exc}")
    return total / counted if counted else None


def run_global(cfg, examples: List[Example], predictor, explainer,
               holdout: Optional[List[Example]] = None) -> List[Dict]:
    """Search for one explanation instruction that is faithful across questions.

    When `holdout` is given, the search only ever sees the holdout pool; the
    best instruction is then evaluated one-shot on the `examples` selection,
    against the human-written instruction it started from. Small holdouts
    overfit — the transfer gap is the honest number to report.
    """
    out_dir = os.path.join(cfg.run.output_dir, cfg.variant_id(), "global")
    os.makedirs(out_dir, exist_ok=True)
    lang = cfg.run.prompt_lang
    initial_instruction = exp_instruction_for(lang)

    pool = holdout if holdout is not None else examples
    pool_size = min(cfg.run.ques_sample, len(pool))

    rng = random.Random(cfg.run.seed)
    instructions: List[str] = [initial_instruction]
    scores: List[float] = []
    rounds: List[Dict] = []

    for round_idx in tqdm(range(cfg.run.round_xai_iter), desc="rounds"):
        round_path = os.path.join(out_dir, f"round-{round_idx}.json")

        if cfg.run.resume and os.path.isfile(round_path):
            with open(round_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            rounds.append(record)
            scores.append(record["score"])
            if record.get("next_instruction"):
                instructions.append(record["next_instruction"])
            # Keep the RNG stream aligned with the run being resumed.
            rng.sample(range(len(pool)), pool_size)
            continue

        sample = rng.sample(range(len(pool)), pool_size)
        current_instruction = instructions[-1]

        avg = _mean_instruction_score(cfg, pool, sample, current_instruction,
                                      predictor, explainer, lang,
                                      tag=f"round {round_idx}")
        if avg is None:
            print(f"[global] round {round_idx} scored no questions; stopping.")
            break
        scores.append(avg)

        # Ask the optimizer for a better instruction for the next round.
        next_instruction = ""
        try:
            reply = explainer.respond(global_optimizer_prompt(instructions, scores, lang=lang))
            next_instruction = explainer_mod.parse_instruction(reply)
            if next_instruction:
                instructions.append(next_instruction)
        except Exception as exc:
            print(f"[global] optimizer failed at round {round_idx}: {exc}")

        record = {
            "round": round_idx,
            "score": round(avg, 4),
            "scored_questions": pool_size,
            "instruction": current_instruction,
            "next_instruction": next_instruction,
        }
        _write_json(round_path, record)
        rounds.append(record)

    best = max(rounds, key=lambda r: r["score"]) if rounds else None
    summary = {
        "variant": cfg.variant_id(),
        "metric": cfg.metric.name,
        "scorer": cfg.metric.scorer,
        "optimised_on": "holdout" if holdout is not None else "eval split",
        "rounds": rounds,
        "best_score": best["score"] if best else 0.0,
        "best_instruction": best["instruction"] if best else initial_instruction,
    }

    # Transfer check: does the searched instruction beat the hand-written one
    # outside the pool it was optimised on?
    if holdout is not None and best is not None:
        test_indices = select_indices(cfg, len(examples))
        print(f"[global] transfer eval on {len(test_indices)} test questions...")
        summary["transfer_eval"] = {
            "test_questions": len(test_indices),
            "optimized_instruction": _mean_instruction_score(
                cfg, examples, test_indices, best["instruction"],
                predictor, explainer, lang, tag="transfer/optimized"),
            "initial_instruction": _mean_instruction_score(
                cfg, examples, test_indices, initial_instruction,
                predictor, explainer, lang, tag="transfer/initial"),
        }

    _write_json(os.path.join(out_dir, "summary.json"), summary)
    return rounds
