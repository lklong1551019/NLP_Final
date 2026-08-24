"""Continuous, label-free fidelity scoring.

The baseline metric is `abs(acc(f(X)) - acc(f(X|!E)))`. On a single instance each
accuracy is 0 or 1, so the score the LLM optimiser sees is binary. Three
consequences, all of which make the optimisation loop wander:

  * No gradient. A trajectory of `[E1, 0] [E2, 0] [E3, 0]` cannot tell the
    explainer which rewrite moved the target and which did nothing.
  * No sign. `abs()` collapses "the hint pushed the model away from its answer"
    (faithful) and "the hint pushed it further in" (the hint was not contrary).
  * It measures a change in *correctness*, not a change in *prediction*. If the
    target is wrong both times the score is 0 even when the answer flipped, so a
    real intervention effect is recorded as no effect.

This module reads the target's probability over the answer choices instead, in
both conditions:

    prob_shift = P_before(a0) - P_after(a0)        signed, continuous, in [-1, 1]

where `a0` is the choice the target picked on its own. No gold label is involved,
which matches what faithfulness is about - whether the explanation's content
drives the model, including when the model is wrong.

`tv` (total variation) is also returned: it is a proper divergence, so it is the
quantity Theorem 1 of the paper actually defines, and is non-negative like the
published metric. Use `prob_shift` to optimise and `tv` / `accuracy` to report.

Nothing here parses free text, so the `[choice]...@` extraction and its "X"
sentinel - and with them the spurious "Bad Answer" early stops - drop out.
"""

import torch

from model.predictor import normalize_answer


@torch.no_grad()
def choice_probs_local(model, tokenizer, prompt, choices):
    """P(choice | prompt) for a local causal LM, via one forward pass per choice.

    Length-normalised: choices differ in token count and the raw sum would favour
    short ones. No generation happens, which is also why this is far cheaper than
    the baseline's 256 sampled tokens.
    """
    scores = []
    for choice in choices:
        enc_prompt = tokenizer(prompt, return_tensors="pt")
        enc_full = tokenizer(prompt + " " + choice, return_tensors="pt")
        ids = enc_full.input_ids.to(model.device)
        n_prompt = enc_prompt.input_ids.shape[1]
        if ids.shape[1] <= n_prompt:            # tokenizer merged the boundary
            scores.append(-1e9)
            continue
        logits = model(ids).logits[:, :-1].float()
        targets = ids[:, 1:]
        token_lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        cont = token_lp[n_prompt - 1:]
        scores.append(cont.sum().item() / max(1, cont.numel()))
    return torch.softmax(torch.tensor(scores), dim=0).tolist()


def choice_probs_api(prompt, choices, args):
    """P(choice | prompt) for an API target: constrain the reply to one letter
    and read that token's probability. Requires a provider that returns logprobs."""
    from model import llm_api

    letters = [chr(ord("A") + i) for i in range(len(choices))]
    listing = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    ask = (f"{prompt}\n\n### Options:\n{listing}\n\n"
           f"### Answer with one letter ({'/'.join(letters)}) only:")
    top = llm_api.chat_token_logprobs(ask, llm_api.pred_model_id(args),
                                      top_logprobs=max(5, min(20, len(choices) * 4)))
    probs = []
    for letter in letters:
        p = sum(entry["prob"] for entry in top if entry["token"].strip().upper() == letter)
        probs.append(p)
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else [1.0 / len(choices)] * len(choices)


def choice_probs(model, tokenizer, prompt, choices, args):
    if tokenizer is None or getattr(args, "pred_model", None) in ("litellm", "gpt35", "claude"):
        return choice_probs_api(prompt, choices, args)
    return choice_probs_local(model, tokenizer, prompt, choices)


def fidelity_metrics(p_before, p_after, choices, gold):
    """All three scores from the same two probability vectors - no extra calls."""
    i0 = max(range(len(p_before)), key=lambda i: p_before[i])
    i1 = max(range(len(p_after)), key=lambda i: p_after[i])
    gold_norm = normalize_answer(gold)

    acc_before = float(normalize_answer(choices[i0]) == gold_norm)
    acc_after = float(normalize_answer(choices[i1]) == gold_norm)
    # Distance past the decision boundary, after the hint. Positive means the target
    # no longer picks a0 -- the flip has happened -- and the size says by how much.
    #
    # Note this is NOT the difference of the two margins: (p_before - 0.5) minus
    # (p_after - 0.5) cancels the 0.5 and collapses back to prob_shift exactly. The
    # point of a margin signal is to stop rewarding movement that does not approach
    # the boundary, so it has to be anchored at the boundary rather than at the
    # starting point. prob_shift scores 0.40 for 0.95 -> 0.55 (a lot of movement, no
    # flip) and 0.02 for 0.51 -> 0.49 (a flip); margin ranks those the other way round.
    margin = 0.5 - p_after[i0]

    return {
        "prob_shift": p_before[i0] - p_after[i0],                     # signed movement
        "margin": margin,                                             # signed distance past the boundary
        "tv": 0.5 * sum(abs(a - b) for a, b in zip(p_before, p_after)),  # divergence, paper-aligned
        "flip": float(i0 != i1),
        "accuracy": abs(acc_before - acc_after),                      # baseline, for comparison
        "p_before": p_before[i0],
        "p_after": p_after[i0],
        "pred_before": choices[i0],
        "pred_after": choices[i1],
    }
