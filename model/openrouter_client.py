"""OpenRouter explainer backend for FaithLM.

Three things here exist because of measured failures, not style preference:

1. Reasoning is disabled by default. Reasoning tokens are billed as output and, on a
   small max_tokens budget, they consume the whole allowance and the model returns an
   EMPTY string. A pilot on XCOPA-vi produced 400 reasoning tokens and 2 usable tokens
   per call. Note that qwen3.8-max rejects `reasoning.enabled=false` outright
   ("Reasoning is mandatory for this endpoint") -- prefer qwen3.7-flash / qwen3.7-plus.
2. Empty or whitespace-only completions raise. The FaithLM loop does
   `reply.split(":\\n\\n")[-1].split("\\n\\n")` on whatever comes back, so an empty
   string silently becomes an "explanation" and every downstream fidelity score is
   computed on nothing. Failing loudly is the only way to notice.
3. Spend is tracked per call and hard-stops at --max_spend. OpenRouter reports the
   charged cost per request when you ask for it, so this is the real number, not an
   estimate.
"""
import json
import os
import time

_BASE_URL = "https://openrouter.ai/api/v1"
_spent = 0.0
_calls = 0


class BudgetExceeded(RuntimeError):
    pass


class EmptyCompletion(RuntimeError):
    pass


def spend():
    return _spent


def call_count():
    return _calls


def _resolve_key(args):
    key = (getattr(args, "openrouter_key", None)
           or os.environ.get("OPENROUTER_API_KEY"))
    if not key:
        raise RuntimeError(
            "No OpenRouter key. Pass --openrouter_key or set OPENROUTER_API_KEY "
            "(a .env file at the repo root is loaded automatically)."
        )
    return key


def _log_usage(args, model, usage, cost, reasoning_tokens):
    path = getattr(args, "usage_log", None)
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "cost": cost,
            "cumulative_cost": _spent,
            "call": _calls,
        }) + "\n")


def generate(prompt, args, retries=4):
    """Single-turn completion. Returns the text, or raises."""
    global _spent, _calls
    from openai import OpenAI

    max_spend = float(getattr(args, "max_spend", 0) or 0)
    if max_spend and _spent >= max_spend:
        raise BudgetExceeded(
            f"Spent ${_spent:.4f} of the ${max_spend:.2f} budget after {_calls} calls. "
            f"Raise --max_spend to continue."
        )

    client = OpenAI(api_key=_resolve_key(args), base_url=_BASE_URL)
    model = getattr(args, "openrouter_model", "qwen/qwen3.7-flash")
    text = prompt if isinstance(prompt, str) else prompt[0]

    extra_body = {"usage": {"include": True}}
    if not getattr(args, "or_reasoning", False):
        extra_body["reasoning"] = {"enabled": False}

    last_err = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert at explaining language model behavior."},
                    {"role": "user", "content": text},
                ],
                temperature=float(args.temp_exp),
                top_p=float(getattr(args, "top_p", 0.9)),
                max_tokens=int(args.max_tokens),
                extra_body=extra_body,
            )
        except Exception as e:                       # transient: rate limit, 5xx, network
            last_err = e
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        raw_usage = getattr(r, "usage", None)
        if raw_usage is None:
            usage = {}
        elif hasattr(raw_usage, "model_dump"):
            usage = raw_usage.model_dump()
        else:
            usage = dict(raw_usage)
        cost = float(usage.get("cost") or 0.0)
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        _spent += cost
        _calls += 1
        _log_usage(args, model, usage, cost, reasoning_tokens)

        content = (r.choices[0].message.content or "").strip()
        if content:
            return content

        # Empty content is nearly always the reasoning budget eating the whole allowance.
        last_err = EmptyCompletion(
            f"{model} returned empty content "
            f"(completion={usage.get('completion_tokens')}, reasoning={reasoning_tokens}, "
            f"max_tokens={args.max_tokens}). "
            + ("Reasoning is ON -- disable it or raise --max_tokens."
               if getattr(args, "or_reasoning", False) else
               "Raise --max_tokens.")
        )
        if attempt == retries - 1:
            raise last_err
        time.sleep(1)

    raise last_err if last_err else RuntimeError("OpenRouter call exhausted retries")
