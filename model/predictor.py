import os
import re
import unicodedata
import zlib

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig
import torch
import openai
try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
except ImportError:  # anthropic >=1.0 dropped the legacy completions constants
    # Only the (unused) claude path needs these. Importing them unconditionally
    # made the whole module unimportable on a modern SDK.
    HUMAN_PROMPT, AI_PROMPT = "\n\nHuman:", "\n\nAssistant:"

    class Anthropic:  # pragma: no cover - legacy path
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "The claude path needs an anthropic SDK that still exports "
                "HUMAN_PROMPT/AI_PROMPT (pre-1.0). Install one, or select "
                "--pred_model/--xai_model litellm."
            )
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from model import llm_api

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(text):
    """Casefold, drop punctuation, collapse whitespace.

    `\\w` is unicode-aware, so Vietnamese diacritics survive - important for the
    XCOPA-vi / ViMMRC runs.
    """
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def contains_answer(gold, text):
    """True if `gold` appears in `text` ignoring case and punctuation.

    The original code used a literal `gold.strip() in text` test. Modern chat
    models restate the choice inside a sentence ("...because it was fragile."),
    so a capital letter or a moved full stop made a correct answer score 0.
    That produced spurious faithfulness scores: the same correct answer counted
    as right in one prompt and wrong in the other. `_trivaqa_score` already
    compared case-insensitively, so this makes ECQA-style scoring consistent
    with it.
    """
    gold_norm = normalize_answer(gold)
    return bool(gold_norm) and gold_norm in normalize_answer(text)


_CHOICE_RE = re.compile(r"\[choice\](.*?)@", re.S)

# Markers a chat model uses to introduce its final pick. Checked last-first, so
# reasoning that mentions a choice before rejecting it does not win.
_ANSWER_MARKERS = (
    "**answer:**", "answer:", "correct choice is", "correct choice:",
    "the correct answer is", "correct answer:", "đáp án đúng là", "đáp án:",
    "câu trả lời đúng là", "lựa chọn đúng là",
)


def parse_choices(prompt):
    """Pull the candidate answers out of a prompt's `[choice]...@` markers."""
    return [c.strip() for c in _CHOICE_RE.findall(str(prompt)) if c.strip()]


def select_choice(response, choices):
    """Return the choice the response actually settles on, or None.

    A plain "is the gold answer mentioned?" test is not enough: models routinely
    name a choice in order to reject it ("The other choice, X, is also possible
    but less directly implied ... **Answer:** Y"). Scoring that as X inflates
    accuracy. We therefore look inside the response's final-answer region when a
    marker is present, and otherwise take the choice mentioned last.
    """
    if not choices:
        return None

    norm_response = normalize_answer(response)

    # Prefer whatever follows the last answer marker.
    cut = -1
    for marker in _ANSWER_MARKERS:
        pos = norm_response.rfind(normalize_answer(marker))
        if pos > cut:
            cut = pos
    regions = []
    if cut != -1:
        regions.append(norm_response[cut:])
    regions.append(norm_response)

    for region in regions:
        best, best_pos = None, -1
        for choice in choices:
            norm_choice = normalize_answer(choice)
            if not norm_choice:
                continue
            pos = region.rfind(norm_choice)
            if pos > best_pos:
                best, best_pos = choice, pos
        if best is not None:
            return best
    return None


def _gen_kwargs(temperature):
    """Make the requested temperature actually take effect.

    transformers ignores `temperature` unless `do_sample=True`; the upstream code
    never set it, so every predictor generation ran greedily and the paper's
    "Temperature of Predictor" row (Table 2: 0.7/0.5/0.7) had no effect. Sampling
    is enabled when a non-zero temperature is asked for, and suppressed
    otherwise so scoring stays deterministic.
    """
    if os.environ.get("FAITHLM_PRED_GREEDY"):
        # Reproduce what the released code actually did: it never set
        # do_sample, so transformers ignored `temperature` and decoded greedily.
        # The paper's Table 2 nonetheless specifies 0.7/0.5/0.7, so the two are
        # not the same experiment. This switch makes the difference measurable.
        return {"do_sample": False}
    if temperature and temperature > 0.0:
        return {"do_sample": True, "temperature": float(temperature)}
    return {"do_sample": False}


def _placement_kwargs(max_memory, load_in_4bit=False):
    """Device/dtype kwargs that work on CUDA, Apple Silicon (MPS) and CPU.

    The upstream code always passed device_map="auto" together with
    max_memory={0: '45GB'}, which assumes an A40-class CUDA device. On a machine
    without CUDA that key refers to a GPU that does not exist, and bfloat16 is
    poorly supported on MPS. Detect the backend instead.
    """
    if torch.cuda.is_available():
        return {"device_map": "auto", "max_memory": max_memory,
                "torch_dtype": torch.bfloat16}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS has patchy bfloat16 kernels; float16 is the supported half type.
        return {"device_map": "mps", "torch_dtype": torch.float16}
    return {"device_map": "cpu", "torch_dtype": torch.float32}


# --- Random-hint control -----------------------------------------------------
# FaithLM scores an explanation by whether contradicting it flips the
# prediction. That conflates two things: the explanation carrying the model's
# actual reason, and the model simply following whatever hint it is given.
# Scoring an irrelevant hint the same way separates them:
#     fidelity_corrected = flip(contrary hint) - flip(irrelevant hint)
# Enabled with FAITHLM_RANDOM_CONTROL=1. Results land in CONTROL["last"].
CONTROL = {"last": None}

_RANDOM_HINTS_EN = [
    "The weather forecast mentions scattered clouds tomorrow afternoon.",
    "The train timetable was revised at the start of the quarter.",
    "A new bakery opened two streets away from the library.",
    "The stadium roof was repainted during the off-season.",
]
_RANDOM_HINTS_VI = [
    "Dự báo thời tiết cho biết ngày mai trời có mây rải rác.",
    "Lịch tàu chạy đã được điều chỉnh từ đầu quý này.",
    "Một tiệm bánh mới khai trương cách thư viện hai con phố.",
    "Mái sân vận động được sơn lại trong kỳ nghỉ giữa mùa.",
]


def random_control_enabled():
    return bool(os.environ.get("FAITHLM_RANDOM_CONTROL"))


def pick_random_hint(question, args):
    """Deterministic irrelevant hint, so a rerun reproduces the same control."""
    pool = _RANDOM_HINTS_VI if getattr(args, "data", "") == "xcopa_vi" else _RANDOM_HINTS_EN
    # zlib.crc32 instead of hash(): PYTHONHASHSEED randomises hash() per
    # process, so sharded runs would each pick a different hint.
    key = normalize_answer(question).encode("utf-8")
    return pool[zlib.crc32(key) % len(pool)]


def load_model(model_name, max_memory, load_in_4bit=True):

    '''
    Setting Device 
    device = "cuda:1"
    max_memory = {0: '0GB', 1:'0GB', 2:'0GB', 3:'0GB', 4:'25GB', 5:'25GB', 6:'25GB', 7:'25GB'}#
    max_memory = {0: '45GB', 1:'45GB', 2:'0GB', 3:'0GB'}
    '''

    # Load Reason Model
    if model_name == "vicuna":
        print("============ Predictor: Vicuna-7B")
        model_id = "lmsys/vicuna-7b-v1.5"
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        model.eval()

    elif model_name == "phi":
        print("============ Predictor: Phi-2")
        model_id = "microsoft/phi-2"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_placement_kwargs(max_memory),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        # Upstream set pad_token to the literal '[PAD]', which is not in phi-2's
        # vocabulary, so padded positions decoded to <unk>. Reuse EOS instead.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    
    elif model_name == "qwen":
        model_id = "Qwen/Qwen3.5-4B"
        quant_kwargs = {}
        if load_in_4bit:
            print("============ Predictor: Qwen3.5-4B (4-bit)")
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            print("============ Predictor: Qwen3.5-4B (bf16)")
            quant_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            **quant_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

    elif model_name == "claude":
        model = "claude"
        tokenizer = None

    elif model_name in ("litellm", "gpt35"):
        # API-served predictor: no weights to load. A None tokenizer is the
        # signal the callers use to route through generate_api_predictor_output.
        model = model_name
        tokenizer = None

    else:
        raise ValueError(
            f"Unknown pred_model '{model_name}'. "
            "Expected one of: vicuna, phi, qwen, claude, gpt35, litellm."
        )

    return model, tokenizer

def generate_api_predictor_output(pred_model, pred_tokenzier, task_instruction, question, ans_gt, args):

    ans_llm = []

    instruction = f"Below is an instruction that describes a task. \
                    Write a response that appropriately completes the request of input."
    final_prompt = [ f"{instruction}\n\n \
                        ### Instruction: {task_instruction}\n\n \
                        ### Input: {ques}\n\n \
                        ### Response: Let's think step by step." for ques in question ]
    
    if args.pred_model == "claude":
        anthropic = Anthropic(
            api_key=args.claude_key,
        )
        model_id = 'claude-2'
        completion = anthropic.completions.create(
                model=model_id,
                max_tokens_to_sample=200,
                prompt=f"{HUMAN_PROMPT} {final_prompt} {AI_PROMPT}",
        )
        ans = completion.completion

    elif args.pred_model == "gpt35":
        openai.api_type = "azure"
        openai.api_base = "https://openai-datalab.openai.azure.com/"
        openai.api_version = "2023-05-15"
        openai.api_key = args.gpt_key
        # ipdb.set_trace()
        response = openai.ChatCompletion.create(
            engine="gpt35turbo", # engine = "deployment_name".
            temperature = 0.9,
            max_tokens = 200,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": f"{final_prompt}"}
            ]
        )
        try:
            ans = response['choices'][0]['message']['content']
        except:
            ans = "X"

    elif args.pred_model == "litellm":
        try:
            ans = llm_api.chat(
                final_prompt,
                model=llm_api.pred_model_id(args),
                max_tokens=200,
                temperature=0.0,
                system="You are a helpful assistant.",
            )
        except Exception as e:
            print(f"[ERROR] Predictor API: {e}")
            ans = "X"

    else:
        raise ValueError(f"generate_api_predictor_output: unsupported pred_model '{args.pred_model}'")

    if os.environ.get("FAITHLM_DEBUG"):
        print(f"[RAW INIT] {ans!r}")

    picked = select_choice(ans, parse_choices(final_prompt[0] if final_prompt else ""))
    if picked is None:
        picked = ans_gt[0].strip() if contains_answer(ans_gt[0], ans) else "X"
    ans_llm.append(picked if picked else "X")

    return ans_llm

def generate_predictor_output_ecqa(model, tokenizer, task_instruction, input_zip, ans_gt, args):

    temperature_cot = 0.7
    ans_llm = []

    if args.pred_model in ["phi", "qwen"]:
        if getattr(args, 'data', '') == "xcopa_vi":
            instruction = "Dưới đây là mô tả về một tác vụ. Hãy viết một phản hồi để hoàn thành yêu cầu được giao."
            final_prompt = [ f"{instruction}\n\n### Hướng dẫn: {task_instruction}\n\n### Đầu vào: {ques}\n\n### Phản hồi:" for ques in input_zip ]
        else:
            instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request of input."
            final_prompt = [ f"{instruction}\n\n### Instruction: {task_instruction}\n\n### Input: {ques}\n\n### Response:" for ques in input_zip ]
        
        model_inputs_all = tokenizer(final_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            split_str = "### Phản hồi:" if getattr(args, 'data', '') == "xcopa_vi" else "### Response:"
            ans = ans_tkn[i].split(split_str)[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            try:
                end_index = ans.find("@")
                index = ans.find("]")
                if index != -1 and end_index > index:
                    if ans[index+1] == " ":
                        ans = ans[index+2:end_index]
                    else:
                        ans = ans[index+1:end_index]
                elif "The correct answer is:\n\n" in ans:
                    ans = ans.split("The correct answer is:\n\n")[-1]
                    ans = ans.replace("*", "").strip()
                else:
                    if contains_answer(ans_gt[i], ans):
                        ans = ans_gt[i]
                    else:
                        ans = "X"
            except:
                ans = "X"
            if len(ans) == 0 or ans is None:
                ans = "X"
            ans_llm.append(ans)

    elif args.pred_model == "vicuna":
        instruction = f"Below is an instruction that describes a task. \
                        Write a response that appropriately completes the request of input."
        final_prompt = [ f"{instruction}\n\n### Instruction: {task_instruction}\n\n \
                            ### Input: {ques}\n\n \
                            ### Response: Let's think step by step." for ques in input_zip ]

        model_inputs_all = tokenizer(final_prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            try:
                index = ans.find("]")
                end_index = ans.find("@")
                if end_index != -1 and index != -1:
                    if ans[index+1] == " ":
                        ans = ans[index+2:end_index]
                    else:
                        ans = ans[index+1:end_index]
                else:
                    if contains_answer(ans_gt[i], ans):
                        ans = ans_gt[i].strip()
                    else:
                        ans = "X"
            except:
                ans = "X"
            if len(ans) == 0 or ans is None:
                ans = "X"
            ans_llm.append(ans)

    return ans_llm

def generate_predictor_output_trivaqa(model, tokenizer, task_instruction, input_zip, ans_gt, args):

    temperature_cot = 0.0
    ans_llm = []

    instruction = f"Below is an instruction that describes a task. \
                    Write a response that appropriately completes the request of input."
    final_prompt = [ f"{instruction}\n\n### Instruction: {task_instruction}\n\n \
                    ### Context: {ques[1]}\n\n### Input: {ques[0]}?\n\n### Response:" for ques in input_zip ]

    if args.pred_model in ["phi", "qwen"]:
        model_inputs_all = tokenizer(final_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=20)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            ans_llm.append(ans)

    elif args.pred_model == "vicuna":
        model_inputs_all = tokenizer(final_prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=20)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            ans_llm.append(ans)

    return ans_llm

def _ecqa_score(model, tokenizer, input_prompt, ans_gt, args):
    temperature_cot = 0.01
    ans_short = []

    if args.pred_model in ["claude"]:
        anthropic = Anthropic(
            api_key=args.claude_key,
        )
        model_id = 'claude-2'
        completion = anthropic.completions.create(
            model=model_id,
            max_tokens_to_sample=200,
            prompt=f"{HUMAN_PROMPT} {input_prompt} {AI_PROMPT}",
        )
        
        ans = completion.completion
        try:
            index = ans.find("]")
            end_index = ans.find("@")
            if index != -1 and end_index > index:
                if ans[index+1] == " ":
                    ans = ans[index+2:end_index]
                else:
                    ans = ans[index+1:end_index]
            else:
                if contains_answer(ans_gt[0], ans):
                    ans = ans_gt[0].strip()
                else:
                    ans = "X"
        except:
            ans = "X"
        if len(ans) == 0 or ans is None:
            ans = "X"
        ans_short.append(ans)
    
    elif args.pred_model == "gpt35":
        openai.api_type = "azure"
        openai.api_base = "https://openai-datalab.openai.azure.com/"
        openai.api_version = "2023-05-15"
        openai.api_key = args.gpt_key
        # ipdb.set_trace()
        response = openai.ChatCompletion.create(
            engine="gpt35turbo",
            temperature = 0.9,
            max_tokens = 200,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": f"{input_prompt}"}
            ]
        )
        try:
            ans = response['choices'][0]['message']['content']
        except:
            ans = "X"
        try:
            index = ans.find("]")
            end_index = ans.find("@")
            if index != -1 and end_index > index:
                if ans[index+1] == " ":
                    ans = ans[index+2:end_index]
                else:
                    ans = ans[index+1:end_index]
            else:
                if contains_answer(ans_gt[0], ans):
                    ans = ans_gt[0].strip()
                else:
                    ans = "X"
        except:
            ans = "X"
        if len(ans) == 0 or ans is None:
            ans = "X"
        ans_short.append(ans)

    elif args.pred_model == "litellm":
        prompts = input_prompt if isinstance(input_prompt, list) else [input_prompt]
        for i, one_prompt in enumerate(prompts):
            try:
                ans = llm_api.chat(
                    one_prompt,
                    model=llm_api.pred_model_id(args),
                    max_tokens=200,
                    temperature=temperature_cot,
                    system="You are a helpful assistant.",
                )
            except Exception as e:
                print(f"[ERROR] Predictor API (score): {e}")
                ans = "X"
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW ANS] {ans!r}")
            # Resolve which option the response settles on, rather than asking
            # whether the gold string appears anywhere in it.
            picked = select_choice(ans, parse_choices(one_prompt))
            if picked is None:
                picked = ans_gt[i] if contains_answer(ans_gt[i], ans) else "X"
            ans_short.append(picked if picked else "X")

    elif args.pred_model in ["phi", "qwen"]:
        model_inputs_all = tokenizer(input_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            split_str = "### Phản hồi:" if getattr(args, 'data', '') == "xcopa_vi" else "### Response:"
            ans = ans_tkn[i].split(split_str)[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            try:
                end_index = ans.find("@")
                index = ans.find("]")

                if index != -1 and end_index > index:
                    ans = ans[index+1:end_index]
                elif "The correct answer is:\n\n" in ans:
                    ans = ans.split("The correct answer is:\n\n")[-1]
                    ans = ans.replace("*", "").strip()
                else:
                    if contains_answer(ans_gt[i], ans):
                        ans = ans_gt[i]
                    else:
                        ans = "X"
            except:
                ans = "X"
            if len(ans) == 0 or ans is None:
                ans = "X"
            ans_short.append(ans)

    else:
        model_inputs_all = tokenizer(input_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            try:
                end_index = ans.find("@")
                index = ans.find("]")
                if index != -1 and end_index > index:
                    if ans[index+1] == " ":
                        ans = ans[index+2:end_index]
                    else:
                        ans = ans[index+1:end_index]
                elif "The correct answer is:\n\n" in ans:
                    ans = ans.split("The correct answer is:\n\n")[-1]
                    ans = ans.replace("*", "").strip()
                else:
                    if contains_answer(ans_gt[i], ans):
                        ans = ans_gt[i]
                    else:
                        ans = "X"
            except:
                ans = "X"
            if len(ans) == 0 or ans is None:
                ans = "X"
            ans_short.append(ans)

    short_acc = accuracy_score(ans_short, ans_gt)
    return short_acc

def diff_task_score_ecqa(model, tokenizer, task_instruction, question, answer, exp_reply, counter_exp_reply, args):

    true_exp_pair = zip(exp_reply, question)
    count_exp_pair = zip(counter_exp_reply, question)

    if getattr(args, 'data', '') == "xcopa_vi":
        instruction = "Dưới đây là mô tả về một tác vụ. Hãy viết một phản hồi để hoàn thành yêu cầu được giao."
        ture_final_prompt = [f"{instruction}\n\n### Hướng dẫn: {task_instruction}\n\n### Đầu vào: {ques}\n\n### Phản hồi: Hãy suy nghĩ từng bước một." for _, ques in true_exp_pair ]
        count_final_prompt = [f"{instruction}\n\n### Hướng dẫn: {task_instruction}\n\n### Gợi ý: {exp}\n\n### Đầu vào: {ques}\n\n### Phản hồi: Hãy suy nghĩ từng bước một." for exp, ques in count_exp_pair ]
    else:
        instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request of input."
        ture_final_prompt = [f"{instruction}\n\n### Instruction: {task_instruction}\n\n### Input: {ques}\n\n### Response: Let's think step by step." for _, ques in true_exp_pair ]
        count_final_prompt = [f"{instruction}\n\n### Instruction: {task_instruction}\n\n### Hint: {exp}\n\n### Input: {ques}\n\n### Response: Let's think step by step." for exp, ques in count_exp_pair ]
    ture_score = _ecqa_score(model, tokenizer, ture_final_prompt, answer, args)
    count_score = _ecqa_score(model, tokenizer, count_final_prompt, answer, args)

    CONTROL["last"] = None
    if random_control_enabled():
        # Same prompt shape as the counterfactual, but the hint has nothing to
        # do with the question. Any flip here is pure suggestibility.
        rand_hint = pick_random_hint(question[0] if question else "", args)
        if getattr(args, "data", "") == "xcopa_vi":
            rand_prompt = [f"{instruction}\n\n### Hướng dẫn: {task_instruction}\n\n### Gợi ý: {rand_hint}\n\n### Đầu vào: {q}\n\n### Phản hồi: Hãy suy nghĩ từng bước một." for q in question]
        else:
            rand_prompt = [f"{instruction}\n\n### Instruction: {task_instruction}\n\n### Hint: {rand_hint}\n\n### Input: {q}\n\n### Response: Let's think step by step." for q in question]
        rand_score = _ecqa_score(model, tokenizer, rand_prompt, answer, args)
        CONTROL["last"] = {
            "rand_score": rand_score,
            "diff_random": abs(ture_score - rand_score),
            "hint": rand_hint,
        }
    
    print("\n--- DEBUG INFO ---")
    print("TRUE PROMPT:")
    print(ture_final_prompt[0] if len(ture_final_prompt) > 0 else "EMPTY")
    print("COUNT PROMPT:")
    print(count_final_prompt[0] if len(count_final_prompt) > 0 else "EMPTY")
    print(f"TRUE SCORE: {ture_score}")
    print(f"COUNT SCORE: {count_score}")
    print("------------------\n")

    diff_score = abs(ture_score - count_score)
    
    return diff_score

def _trivaqa_score(model, tokenizer, input_prompt, ans_gt, args):

    def _accuracy_score(output_ans, gt_ans):
        acc = 0.0
        for idx, llm_ans in enumerate(output_ans):
            for ans in gt_ans[idx]:
                if ans.lower() in llm_ans.lower().strip():
                    acc += 1
                    break
        return acc / len(output_ans)
    
    ans_short = []
    temperature_cot = 0.0
    if args.pred_model in ["phi", "qwen"]:
        model_inputs_all = tokenizer(input_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=10)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            ans_short.append(ans)

    elif args.pred_model == "vicuna":
        model_inputs_all = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, **_gen_kwargs(temperature_cot), max_new_tokens=10)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            if os.environ.get("FAITHLM_DEBUG"):
                print(f"[RAW LOCAL] full={ans_tkn[i]!r}")
            ans_short.append(ans)

    elif args.pred_model == "claude":
        anthropic = Anthropic(
            api_key=args.claude_key,
        )
        model_id = 'claude-2'
        completion = anthropic.completions.create(
            model=model_id,
            max_tokens_to_sample=200,
            prompt=f"{HUMAN_PROMPT} {input_prompt} {AI_PROMPT}",
        )
        
        ans = completion.completion
        ans = ans.split("### Response:")[-1]
        ans_short.append(ans)

    elif args.pred_model == "gpt35":
        openai.api_type = "azure"
        openai.api_base = "https://openai-datalab.openai.azure.com/"
        openai.api_version = "2023-05-15"
        openai.api_key = args.gpt_key
        # ipdb.set_trace()
        response = openai.ChatCompletion.create(
            engine="gpt35turbo", # engine = "deployment_name".
            temperature = 0.9,
            max_tokens = 200,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": f"{input_prompt}"}
            ]
        )
        try:
            ans = response['choices'][0]['message']['content']
            ans = ans.split("### Response:")[-1]
            ans_short.append(ans)
        except:
            ans_short.append("X")

    score = _accuracy_score(ans_short, ans_gt)
    return score

def diff_task_score_trivaqa(model, tokenizer, task_instruction, input_zip, answer, exp_reply, counter_exp_reply, args):

    question = [x[0] for x in input_zip]
    passage = [x[1] for x in input_zip]

    true_exp_pair = zip(exp_reply, question, passage)
    count_exp_pair = zip(counter_exp_reply, question, passage)

    instruction = f"Below is an instruction that describes a task. \
                    Write a response that appropriately completes the request of input."
    
    ture_final_prompt = [f"{instruction}\n\n### Instruction: {task_instruction}\n\n \
                            ### Context: {ques[2]}\n\n \
                            ### Input: {ques[1]}?\n\n \
                            ### Response:" for ques in true_exp_pair ]
    count_final_prompt = [f"{instruction}\n\n### Instruction: {task_instruction}\n\n \
                            ### Goal: {ques[0]}\n\n \
                            ### Context: {ques[2]}\n\n \
                            ### Input: {ques[1]}?\n\n \
                            ### Response:" for ques in count_exp_pair ]
    
    ture_score = _trivaqa_score(model, tokenizer, ture_final_prompt, answer, args)
    count_score = _trivaqa_score(model, tokenizer, count_final_prompt, answer, args)
    diff_score = abs(ture_score - count_score)
    
    return diff_score