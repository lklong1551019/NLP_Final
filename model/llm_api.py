"""Shared OpenAI-compatible chat helper.

Both the Predictor and the Explainer can be served by a LiteLLM gateway (or any
OpenAI-compatible endpoint). Keeping the client here means predictor.py and
explainer.py share one implementation of retries and configuration.

Configuration is read from the environment (see .env):
    LITELLM_API_KEY / DEEPSEEK_API_KEY / API_KEY   - bearer token
    LITELLM_BASE_URL / DEEPSEEK_BASE_URL           - gateway base url
    LITELLM_PRED_MODEL                             - model id for the Predictor
    LITELLM_XAI_MODEL / DEEPSEEK_MODEL             - model id for the Explainer
"""

import os
import time

from openai import OpenAI

_CLIENT = None

DEFAULT_PRED_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_XAI_MODEL = "deepseek/deepseek-v4-pro"

# Call-level bookkeeping. An empty completion (provider-side content filtering)
# would otherwise be scored as a wrong answer and silently bias the results, so
# we count these and report them alongside the scores.
STATS = {"calls": 0, "empty": 0, "errors": 0}


def stats_summary():
    return (
        f"LLM calls: {STATS['calls']} | "
        f"empty completions: {STATS['empty']} | "
        f"failed calls: {STATS['errors']} | "
        f"fallback used: {STATS.get('fallback_used', 0)}"
    )


def _first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        api_key = _first_env("LITELLM_API_KEY", "DEEPSEEK_API_KEY", "API_KEY")
        base_url = _first_env("LITELLM_BASE_URL", "DEEPSEEK_BASE_URL")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set LITELLM_API_KEY (or DEEPSEEK_API_KEY) in .env"
            )
        if not base_url:
            raise RuntimeError(
                "No base url found. Set LITELLM_BASE_URL (or DEEPSEEK_BASE_URL) in .env"
            )
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT


def pred_model_id(args=None):
    return (
        getattr(args, "litellm_pred_model", None)
        or _first_env("LITELLM_PRED_MODEL")
        or DEFAULT_PRED_MODEL
    )


def xai_model_id(args=None):
    return (
        getattr(args, "deepseek_model", None)
        or _first_env("LITELLM_XAI_MODEL", "DEEPSEEK_MODEL")
        or DEFAULT_XAI_MODEL
    )


def chat(prompt, model, max_tokens=1000, temperature=0.0, system=None, retries=4, top_p=None):
    """Single chat completion. Raises after `retries` failed attempts.

    Callers that want the original fail-soft behaviour should catch the
    exception themselves - we deliberately do not swallow it here, because a
    silently-empty response corrupts the faithfulness scores downstream.
    """
    if not isinstance(prompt, str):
        # The pipeline passes single-element lists around in several places.
        prompt = prompt[0] if len(prompt) else ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    STATS["calls"] += 1
    for attempt in range(retries):
        try:
            kwargs = {}
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
            completion = get_client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs,
            )
            content = completion.choices[0].message.content
            if content is None or not content.strip():
                # An empty completion scores as a wrong answer downstream, which
                # manufactures a fake faithfulness difference. Treat it as a
                # transient failure and retry instead.
                raise ValueError("empty completion")
            return content
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    # Some providers return an empty body rather than an error for prompts their
    # content filter dislikes. Retrying the same model does not help, so fall
    # back to a different one once before giving up.
    fallback = _first_env("LITELLM_FALLBACK_MODEL")
    if fallback and fallback != model:
        try:
            completion = get_client().chat.completions.create(
                model=fallback,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
            content = completion.choices[0].message.content
            if content and content.strip():
                STATS["fallback_used"] = STATS.get("fallback_used", 0) + 1
                return content
        except Exception:  # noqa: BLE001 - fall through to the error below
            pass

    if isinstance(last_error, ValueError) and "empty completion" in str(last_error):
        STATS["empty"] += 1
    else:
        STATS["errors"] += 1
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def chat_token_logprobs(prompt, model, top_logprobs=20, retries=3):
    """Probabilities of the candidate first answer tokens.

    Used by the logprob fidelity metric: the reply is capped at one token and we
    read the distribution over it, so nothing has to be parsed out of free text.

    Reasoning is disabled explicitly. Reasoning tokens are billed as output *and*
    count against max_tokens, so with max_tokens=1 a thinking model returns an
    empty completion and no logprobs at all. Not every provider serves logprobs
    even when the model advertises them - callers get a RuntimeError so the run
    stops rather than silently scoring on a uniform prior.

    Returns: [{"token": str, "logprob": float, "prob": float}, ...]
    """
    import math

    STATS["calls"] += 1
    last_error = None
    for attempt in range(retries):
        try:
            extra_body = {"reasoning": {"enabled": False}}
            # A gateway may route the same model to several providers, and only
            # some of them return logprobs. Pin the ones that do via
            # LITELLM_PROVIDER_ORDER (comma separated) so scoring is reproducible.
            order = _first_env("LITELLM_PROVIDER_ORDER")
            if order:
                extra_body["provider"] = {
                    "order": [o.strip() for o in order.split(",") if o.strip()],
                    "allow_fallbacks": False,
                }
            completion = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
                extra_body=extra_body,
            )
            logprobs = completion.choices[0].logprobs
            if not logprobs or not logprobs.content:
                raise ValueError("provider returned no logprobs")
            return [{"token": t.token, "logprob": t.logprob, "prob": math.exp(t.logprob)}
                    for t in logprobs.content[0].top_logprobs]
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    STATS["errors"] += 1
    raise RuntimeError(
        f"logprob call failed after {retries} attempts: {last_error}. "
        f"Not all providers serve logprobs; pin one that does, or use "
        f"--score_mode accuracy. Pin with LITELLM_PROVIDER_ORDER."
    )
