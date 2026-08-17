"""The local and global FaithLM pipelines.

Both write one JSON file per unit of work (a question for local, a round for
global) and skip units that already have a result file when `run.resume` is on.
That keeps a run restartable across Kaggle's 12-hour session limit without
losing finished work.
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
    EXP_INSTRUCTION,
    TASK_INSTRUCTION_MC,
    TASK_INSTRUCTION_QA,
    counterfactual_prompt,
    explanation_prompt,
    global_optimizer_prompt,
    local_optimizer_prompt,
)


def _task_instruction(example: Example) -> str:
    return TASK_INSTRUCTION_MC if example.is_multiple_choice else TASK_INSTRUCTION_QA


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
        task_instruction = _task_instruction(example)
        started = time.time()

        try:
            record = _run_local_one(cfg, example, idx, predictor, explainer, task_instruction)
        except Exception as exc:
            # One bad question must not end a multi-hour run.
            print(f"[local] question {idx} failed: {exc}")
            record = {"index": idx, "error": str(exc), "iterations": []}

        record["elapsed_sec"] = round(time.time() - started, 2)
        _write_json(result_path, record)
        results.append(record)

    return results


def _run_local_one(cfg, example: Example, idx: int, predictor, explainer,
                   task_instruction: str) -> Dict:
    # 1. What does the predictor actually answer?
    from .prompts import task_prompt

    base_prompt = task_prompt(task_instruction, example.question,
                              passage=example.passage, cot=True)
    raw_answer = predictor.generate(
        [base_prompt],
        max_new_tokens=cfg.predictor.max_new_tokens,
        temperature=cfg.predictor.temperature,
    )[0]

    if example.is_multiple_choice:
        parsed = predictor_mod.extract_choice(raw_answer, example.choices)
        model_answer = parsed if parsed else raw_answer.strip()[:200]
        is_correct = parsed == example.answer
    else:
        aliases = example.answer_aliases or [example.answer]
        is_correct = any(a.lower() in raw_answer.lower() for a in aliases if a)
        model_answer = raw_answer.strip()[:200]

    # 2. Explain that answer, then iteratively improve the explanation.
    exp_prompt = explanation_prompt(EXP_INSTRUCTION, example.question,
                                    model_answer, passage=example.passage)
    explanations = explainer_mod.parse_explanations(explainer.respond(exp_prompt))
    current = explanations[0]

    history_exps: List[str] = []
    history_scores: List[float] = []
    iterations: List[Dict] = []

    for step in range(cfg.run.xai_iter):
        counter_reply = explainer.respond(
            counterfactual_prompt(current, open_ended=not example.is_multiple_choice)
        )
        counter = explainer_mod.parse_explanations(counter_reply)[0]

        detail = metrics_mod.compute(
            cfg.metric, predictor, example, task_instruction, current, counter,
            max_new_tokens=cfg.predictor.max_new_tokens,
            temperature=cfg.predictor.temperature,
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
            history_exps, history_scores, example.question, model_answer
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


def run_global(cfg, examples: List[Example], predictor, explainer) -> List[Dict]:
    """Search for one explanation instruction that is faithful across questions."""
    out_dir = os.path.join(cfg.run.output_dir, cfg.variant_id(), "global")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(cfg.run.seed)
    instructions: List[str] = [EXP_INSTRUCTION]
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
            continue

        sample = rng.sample(range(len(examples)), min(cfg.run.ques_sample, len(examples)))
        current_instruction = instructions[-1]

        total, counted = 0.0, 0
        for idx in sample:
            example = examples[idx]
            task_instruction = _task_instruction(example)
            try:
                from .prompts import task_prompt

                raw = predictor.generate(
                    [task_prompt(task_instruction, example.question,
                                 passage=example.passage, cot=True)],
                    max_new_tokens=cfg.predictor.max_new_tokens,
                    temperature=cfg.predictor.temperature,
                )[0]
                answer = predictor_mod.extract_choice(raw, example.choices) or raw.strip()[:200]

                exp = explainer_mod.parse_explanations(
                    explainer.respond(explanation_prompt(
                        current_instruction, example.question, answer, passage=example.passage))
                )[0]
                counter = explainer_mod.parse_explanations(
                    explainer.respond(counterfactual_prompt(
                        exp, open_ended=not example.is_multiple_choice))
                )[0]

                detail = metrics_mod.compute(
                    cfg.metric, predictor, example, task_instruction, exp, counter,
                    max_new_tokens=cfg.predictor.max_new_tokens,
                    temperature=cfg.predictor.temperature,
                )
                total += detail.faithfulness
                counted += 1
            except Exception as exc:
                print(f"[global] round {round_idx}, question {idx} failed: {exc}")

        if counted == 0:
            print(f"[global] round {round_idx} scored no questions; stopping.")
            break

        avg = total / counted
        scores.append(avg)

        # Ask the optimizer for a better instruction for the next round.
        next_instruction = ""
        try:
            reply = explainer.respond(global_optimizer_prompt(instructions, scores))
            next_instruction = explainer_mod.parse_instruction(reply)
            if next_instruction:
                instructions.append(next_instruction)
        except Exception as exc:
            print(f"[global] optimizer failed at round {round_idx}: {exc}")

        record = {
            "round": round_idx,
            "score": round(avg, 4),
            "scored_questions": counted,
            "instruction": current_instruction,
            "next_instruction": next_instruction,
        }
        _write_json(round_path, record)
        rounds.append(record)

    summary_path = os.path.join(out_dir, "summary.json")
    best = max(rounds, key=lambda r: r["score"]) if rounds else None
    _write_json(summary_path, {
        "variant": cfg.variant_id(),
        "metric": cfg.metric.name,
        "scorer": cfg.metric.scorer,
        "rounds": rounds,
        "best_score": best["score"] if best else 0.0,
        "best_instruction": best["instruction"] if best else EXP_INSTRUCTION,
    })
    return rounds
