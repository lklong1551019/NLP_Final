"""Shared OpenAI-compatible chat helper.

Both the Predictor and the Explainer can be served by a LiteLLM gateway (or any
OpenAI-compatible endpoint). Keeping the client here means predictor.py and
explainer.py share one implementation of retries and configuration.

Configuration is read from the environment (see .env):
    LITELLM_API_KEY / DEEPSEEK_API_KEY / API_KEY   - bearer token
    LITELLM_BASE_URL / DEEPSEEK_BASE_URL           - gateway base url
    LITELLM_PRED_MODEL                             - model id for the Predictor
    LITELLM_XAI_MODEL / DEEPSEEK_MODEL             - model id for the Explainer

Vertex AI route: a model id prefixed with "vertex/" is sent to Vertex AI's
OpenAI-compatible endpoint instead of the gateway, authenticated with a
service-account JSON. Everything after the prefix is the Vertex model id,
e.g. "vertex/google/gemini-3.5-flash". Extra environment variables:
    VERTEX_CREDENTIALS - path to the service-account JSON
                         (default: ./config/gen-lang-client.json)
    VERTEX_LOCATION    - region, default "global"
    VERTEX_PROJECT     - GCP project id, default read from the JSON
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


VERTEX_PREFIX = "vertex/"
_VERTEX_CLIENT = None
_VERTEX_CREDS = None


def _vertex_client():
    """OpenAI client for Vertex AI, recreated whenever the OAuth token renews.

    Vertex has no static API keys - the bearer token comes from the service
    account and expires after ~1h, so long runs must refresh it mid-flight.
    """
    global _VERTEX_CLIENT, _VERTEX_CREDS
    import json

    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    cred_path = os.environ.get("VERTEX_CREDENTIALS", "./config/gen-lang-client.json")
    if _VERTEX_CREDS is None:
        if not os.path.isfile(cred_path):
            raise RuntimeError(
                f"Vertex credentials not found at '{cred_path}'. "
                "Set VERTEX_CREDENTIALS in .env"
            )
        _VERTEX_CREDS = service_account.Credentials.from_service_account_file(
            cred_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    if not _VERTEX_CREDS.valid or _VERTEX_CREDS.expired:
        _VERTEX_CREDS.refresh(Request())
        _VERTEX_CLIENT = None  # force a client with the fresh token

    if _VERTEX_CLIENT is None:
        project = os.environ.get("VERTEX_PROJECT")
        if not project:
            with open(cred_path, "r", encoding="utf-8") as f:
                project = json.load(f)["project_id"]
        location = os.environ.get("VERTEX_LOCATION", "global")
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        _VERTEX_CLIENT = OpenAI(
            api_key=_VERTEX_CREDS.token,
            base_url=(
                f"https://{host}/v1/projects/{project}"
                f"/locations/{location}/endpoints/openapi"
            ),
        )
    return _VERTEX_CLIENT


def _vertex_extra_body():
    """Thinking budget for Gemini on Vertex.

    Gemini 2.5+/3.x are thinking models: unconstrained, a one-word answer can
    burn ~800 reasoning tokens, which multiplies cost and latency and can eat
    the whole max_tokens budget before any visible text is produced. Default
    is 0 (thinking off); set VERTEX_THINKING_BUDGET to a token count, or to
    "off" to send no thinking config at all.
    """
    budget = os.environ.get("VERTEX_THINKING_BUDGET", "0")
    if budget.lower() in ("off", "none", ""):
        return {}
    return {
        "google": {
            "thinking_config": {
                "thinking_budget": int(budget),
                "include_thoughts": False,
            }
        }
    }


def resolve_client(model):
    """Map a model id to (client, actual model id sent on the wire).

    "vertex/<id>" goes to Vertex AI; anything else keeps the gateway path.
    """
    if model and model.startswith(VERTEX_PREFIX):
        return _vertex_client(), model[len(VERTEX_PREFIX):]
    return get_client(), model


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
            client, wire_model = resolve_client(model)
            if model and model.startswith(VERTEX_PREFIX):
                kwargs["extra_body"] = _vertex_extra_body()
            completion = client.chat.completions.create(
                model=wire_model,
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
            fb_client, fb_model = resolve_client(fallback)
            completion = fb_client.chat.completions.create(
                model=fb_model,
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
