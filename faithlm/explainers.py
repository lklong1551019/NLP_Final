"""Explainer backends — the LLMs that write and revise explanations.

Every backend returns a plain string. Splitting that string into explanation
segments is the caller's job and lives in `parse_explanations`, because the
original code did it inline in three places and disagreed with itself.
"""

import os
import time
from typing import List, Optional

from .registry import register_explainer


class Explainer:
    def respond(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatExplainer:
    """Any OpenAI-compatible chat endpoint (DeepSeek, OpenAI, local vLLM...)."""

    def __init__(self, model_id: str, api_key_env: str, base_url: Optional[str],
                 max_tokens: int, temperature: float, max_retries: int = 3):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} is not set. On Kaggle, add it via Add-ons > Secrets "
                "and enable Internet in the notebook settings."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def respond(self, prompt: str) -> str:
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": "You are an expert at explaining language model behavior."},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = resp.choices[0].message.content
                if content:
                    return content
                last_error = "empty response"
            except Exception as exc:
                last_error = exc
                # Transient rate limits are common on long runs; back off and retry.
                wait = 2 ** attempt
                print(f"[explainer] attempt {attempt + 1}/{self.max_retries} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Explainer failed after {self.max_retries} attempts: {last_error}")


class OllamaExplainer(Explainer):
    """A local Ollama model writing the explanations — fully offline runs.

    Reuses OllamaPredictor's transport, including the `think: false` flag and
    the <think> stripping that reasoning models need.
    """

    def __init__(self, model_id: str, max_tokens: int, temperature: float):
        from .predictors import OllamaPredictor

        self.backend = OllamaPredictor(model_id=model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def respond(self, prompt: str) -> str:
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        return self.backend._chat(prompt, self.max_tokens, self.temperature)


class HFExplainer:
    """A local hub model acting as the explainer."""

    def __init__(self, model_id: str, max_tokens: int, temperature: float,
                 load_in_4bit: bool = False):
        from .predictors import HFPredictor

        self.backend = HFPredictor(model_id=model_id, load_in_4bit=load_in_4bit)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def respond(self, prompt: str) -> str:
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        out = self.backend.generate(
            [prompt], max_new_tokens=self.max_tokens, temperature=self.temperature
        )
        return out[0] if out else ""


# ---------------------------------------------------------------- registry


@register_explainer("deepseek")
def _build_deepseek(cfg):
    return OpenAICompatExplainer(
        model_id=cfg.model_id or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        api_key_env=cfg.api_key_env,
        base_url=os.environ.get(cfg.base_url_env, "https://api.deepseek.com"),
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


@register_explainer("openai")
def _build_openai(cfg):
    return OpenAICompatExplainer(
        model_id=cfg.model_id or "gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


@register_explainer("claude")
def _build_claude(cfg):
    """Anthropic models through their OpenAI-compatible endpoint."""
    return OpenAICompatExplainer(
        model_id=cfg.model_id or "claude-sonnet-5",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1/",
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


@register_explainer("gemini")
def _build_gemini(cfg):
    """Google Gemini through its OpenAI-compatible endpoint.

    gemini-3.5-flash is the stable series as of 2026-08; the 2.5 series is
    scheduled for shutdown in October 2026.
    """
    return OpenAICompatExplainer(
        model_id=cfg.model_id or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        api_key_env="GEMINI_API_KEY",
        base_url=os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


@register_explainer("api")
def _build_generic_api_explainer(cfg):
    """Any OpenAI-compatible endpoint as the explainer (Gemini, Groq, vLLM...).

    Point `api_key_env`/`base_url_env` at the provider's variables in YAML —
    e.g. Gemini: GEMINI_API_KEY + GEMINI_BASE_URL
    (https://generativelanguage.googleapis.com/v1beta/openai/).
    """
    if not cfg.model_id:
        raise ValueError("explainer.name='api' requires explainer.model_id to be set")
    return OpenAICompatExplainer(
        model_id=cfg.model_id,
        api_key_env=cfg.api_key_env,
        base_url=os.environ.get(cfg.base_url_env) or None,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


@register_explainer("ollama")
def _build_ollama_explainer(cfg):
    return OllamaExplainer(
        model_id=cfg.model_id or "qwen3.5:9b",
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


@register_explainer("hf")
def _build_hf_explainer(cfg):
    if not cfg.model_id:
        raise ValueError("explainer.name='hf' requires explainer.model_id to be set")
    return HFExplainer(
        model_id=cfg.model_id,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


def build(cfg):
    from .registry import get

    return get("explainer", cfg.name)(cfg)


# ---------------------------------------------------------------- parsing


def parse_explanations(reply: str) -> List[str]:
    """Split an explainer reply into explanation segments.

    Always returns at least one non-empty string. The original code could
    produce an empty list here, which then silently emptied the zip() in the
    scorer and made the whole iteration score nothing at all.
    """
    if not reply or not reply.strip():
        return [""]

    # Drop a leading preamble such as "Here is the explanation:\n\n".
    text = reply.split(":\n\n")[-1] if ":\n\n" in reply else reply
    segments = [seg.strip() for seg in text.split("\n\n") if seg.strip()]

    if not segments:
        stripped = reply.strip()
        return [stripped] if stripped else [""]

    # Strip the <EXP> markers the prompts ask for.
    cleaned = []
    for seg in segments:
        seg = seg.replace("<EXP>", "").replace("</EXP>", "").strip()
        if seg:
            cleaned.append(seg)
    return cleaned or [""]


def parse_instruction(reply: str) -> str:
    """Pull the <INS>...</INS> body out of a global-optimizer reply."""
    if not reply:
        return ""
    if "<INS>" in reply and "</INS>" in reply:
        return reply.split("<INS>", 1)[1].split("</INS>", 1)[0].strip()
    return reply.strip()


def is_refusal(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in ("i apologize", "unfortunately", "i cannot", "as an ai"))
