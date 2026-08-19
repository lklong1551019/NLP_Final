"""Predictor backends — the target LLMs whose behaviour we explain.

The important addition over the original code is `choice_logprobs`. The original
scorer compared generated text against the gold string, which on a single
question can only ever return 0.0 or 1.0; the difference of two such values is
almost always exactly 0, leaving LLM-OPT with no signal to optimise. Scoring by
length-normalised log-probability over the answer choices gives a continuous
value from one forward pass, so the optimiser sees real gradients in the score
and the run stays within Kaggle's session limit.
"""

import math
import os
import re
from typing import List, Optional

from .registry import register_predictor


class Predictor:
    """Common interface: generate free text, and (optionally) score choices."""

    supports_logprobs = False

    def generate(self, prompts: List[str], max_new_tokens: int = 256,
                 temperature: float = 0.7) -> List[str]:
        raise NotImplementedError

    def choice_logprobs(self, prompt: str, choices: List[str]) -> List[float]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support log-probability scoring. "
            "Set metric.scorer='exact_match' in the config."
        )


class HFPredictor(Predictor):
    """Any causal LM from the Hugging Face hub."""

    supports_logprobs = True

    def __init__(self, model_id: str, load_in_4bit: bool = False,
                 max_memory_per_gpu: str = "14GiB", device_num: Optional[List[int]] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        device_num = device_num or [0]
        max_memory = {int(i): max_memory_per_gpu for i in device_num}

        kwargs = {
            "device_map": "auto",
            "max_memory": max_memory,
            "trust_remote_code": True,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print(f"[predictor] {model_id} (4-bit NF4)")
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            print(f"[predictor] {model_id} (bf16)")

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate(self, prompts, max_new_tokens=256, temperature=0.7):
        torch = self.torch
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(self.model.device)

        # temperature=0 is invalid for sampling; switch to greedy instead.
        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tokenizer.pad_token_id}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the continuation, so the prompt never leaks into the answer.
        prompt_len = inputs["input_ids"].shape[1]
        completions = output_ids[:, prompt_len:]
        return self.tokenizer.batch_decode(completions, skip_special_tokens=True)

    def choice_logprobs(self, prompt: str, choices: List[str]) -> List[float]:
        """Length-normalised log P(choice | prompt) for each choice.

        Normalising by token count keeps longer choices from being penalised
        purely for their length.
        """
        torch = self.torch
        scores = []
        for choice in choices:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            full_ids = self.tokenizer(prompt + " " + choice, return_tensors="pt").input_ids
            full_ids = full_ids.to(self.model.device)

            n_prompt = prompt_ids.shape[1]
            n_choice = full_ids.shape[1] - n_prompt
            if n_choice <= 0:
                scores.append(-math.inf)
                continue

            with torch.no_grad():
                logits = self.model(full_ids).logits

            # logits[t] predicts token t+1, so the choice tokens are scored by
            # the logits sitting one position to their left.
            logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            targets = full_ids[0, 1:]
            choice_lp = logprobs[n_prompt - 1:, :].gather(
                1, targets[n_prompt - 1:].unsqueeze(-1)
            ).squeeze(-1)
            scores.append((choice_lp.sum() / n_choice).item())
        return scores


class OllamaPredictor(Predictor):
    """A model served by a local Ollama instance — no GPU setup, no API key.

    Talks to the native /api/chat endpoint rather than Ollama's OpenAI shim so
    that `think: false` can be sent; reasoning models otherwise burn their
    token budget inside a think block and return an empty answer.
    """

    def __init__(self, model_id: str, base_url: Optional[str] = None, timeout: int = 600):
        self.model_id = model_id
        base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.endpoint = base.rstrip("/") + "/api/chat"
        self.timeout = timeout

    def _chat(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        import json
        import urllib.request

        body = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            content = json.loads(resp.read())["message"]["content"]
        return strip_think(content)

    def generate(self, prompts, max_new_tokens=256, temperature=0.7):
        outputs = []
        for prompt in prompts:
            try:
                outputs.append(self._chat(prompt, max_new_tokens, temperature))
            except Exception as exc:
                print(f"[predictor] Ollama error: {exc}")
                outputs.append("")
        return outputs


def strip_think(text: str) -> str:
    """Remove a <think>...</think> block some reasoning models still emit."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class APIPredictor(Predictor):
    """OpenAI-compatible chat endpoint used as the target model."""

    def __init__(self, model_id: str, api_key_env: str, base_url: Optional[str] = None):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} is not set — required for API predictor '{model_id}'."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    def generate(self, prompts, max_new_tokens=256, temperature=0.7):
        outputs = []
        for prompt in prompts:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                outputs.append(resp.choices[0].message.content or "")
            except Exception as exc:
                print(f"[predictor] API error: {exc}")
                outputs.append("")
        return outputs


# ---------------------------------------------------------------- registry


@register_predictor("qwen")
def _build_qwen(cfg):
    # Qwen3.5-4B in the earlier revision does not exist on the hub; the real
    # instruction-tuned 4B checkpoint is Qwen3-4B-Instruct-2507.
    return HFPredictor(
        model_id=cfg.model_id or "Qwen/Qwen3-4B-Instruct-2507",
        load_in_4bit=cfg.load_in_4bit,
        max_memory_per_gpu=cfg.max_memory_per_gpu,
        device_num=cfg.device_num,
    )


@register_predictor("phi")
def _build_phi(cfg):
    return HFPredictor(
        model_id=cfg.model_id or "microsoft/phi-2",
        load_in_4bit=cfg.load_in_4bit,
        max_memory_per_gpu=cfg.max_memory_per_gpu,
        device_num=cfg.device_num,
    )


@register_predictor("vicuna")
def _build_vicuna(cfg):
    return HFPredictor(
        model_id=cfg.model_id or "lmsys/vicuna-7b-v1.5",
        load_in_4bit=cfg.load_in_4bit,
        max_memory_per_gpu=cfg.max_memory_per_gpu,
        device_num=cfg.device_num,
    )


@register_predictor("hf")
def _build_generic_hf(cfg):
    """Escape hatch: any hub model id, no code change needed."""
    if not cfg.model_id:
        raise ValueError("predictor.name='hf' requires predictor.model_id to be set")
    return HFPredictor(
        model_id=cfg.model_id,
        load_in_4bit=cfg.load_in_4bit,
        max_memory_per_gpu=cfg.max_memory_per_gpu,
        device_num=cfg.device_num,
    )


@register_predictor("deepseek")
def _build_deepseek_predictor(cfg):
    return APIPredictor(
        model_id=cfg.model_id or "deepseek-v4-pro",
        api_key_env="DEEPSEEK_API_KEY",
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


@register_predictor("api")
def _build_generic_api(cfg):
    """Any OpenAI-compatible endpoint as the target model.

    The key and base URL come from the env vars named in the config, so a new
    provider is a YAML change, not a code change.
    """
    if not cfg.model_id:
        raise ValueError("predictor.name='api' requires predictor.model_id to be set")
    return APIPredictor(
        model_id=cfg.model_id,
        api_key_env=cfg.api_key_env,
        base_url=os.environ.get(cfg.base_url_env) or None,
    )


@register_predictor("ollama")
def _build_ollama_predictor(cfg):
    return OllamaPredictor(model_id=cfg.model_id or "qwen3.5:4b")


def build(cfg):
    from .registry import get

    return get("predictor", cfg.name)(cfg)


# ---------------------------------------------------------------- parsing


def extract_choice(text: str, choices: List[str]) -> Optional[str]:
    """Recover which choice a free-text answer refers to.

    Tries the `[choice]...@` markup first, then a direct substring match, then
    a normalised match. Returns None when nothing matches, which the caller
    records as an abstention rather than silently scoring it wrong.
    """
    if not text:
        return None

    marked = re.search(r"\[choice\](.*?)@", text, flags=re.DOTALL)
    if marked:
        candidate = marked.group(1).strip()
        for choice in choices:
            if candidate and candidate.lower() == choice.lower().strip():
                return choice

    for choice in choices:
        if choice.strip() and choice.strip().lower() in text.lower():
            return choice

    def norm(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower()).strip()

    normalised = norm(text)
    for choice in choices:
        if norm(choice) and norm(choice) in normalised:
            return choice
    return None
