"""Prompt templates, kept in one place so the paper's Appendix can quote them.

Two language packs, selected by `run.prompt_lang`:

  en  Wording preserved verbatim from the original implementation, so
      reproduction numbers stay comparable.
  vi  A Vietnamese translation of the same templates. XCOPA-vi questions are
      Vietnamese, so an all-Vietnamese prompt removes the language mismatch
      between the task text and the instructions around it.

Every function takes `lang="en"`, so existing callers keep their behaviour.
The dataset-side question rendering (the `[choice]...@` markup) is not part of
a pack: it is a parsing contract shared with `extract_choice`, identical in
both languages.
"""

BASE_INSTRUCTION_BY_LANG = {
    "en": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request of input."
    ),
    "vi": (
        "Dưới đây là một chỉ dẫn mô tả nhiệm vụ. "
        "Hãy viết câu trả lời hoàn thành đúng yêu cầu của phần đầu vào."
    ),
}

TASK_INSTRUCTION_MC_BY_LANG = {
    "en": (
        "Please select a correct choice for the each question. "
        "Make sure not to repeat the input context."
    ),
    "vi": (
        "Hãy chọn một phương án đúng cho câu hỏi. "
        "Không lặp lại nội dung đề bài."
    ),
}

TASK_INSTRUCTION_QA_BY_LANG = {
    "en": (
        "Please answer the question based on the context. "
        "Be sure to provid precise answer and make sure not to repeat the question. "
        "Answer the question directly and precisely and provide the answer only without context supports."
    ),
    "vi": (
        "Hãy trả lời câu hỏi dựa trên ngữ cảnh. "
        "Trả lời chính xác, trực tiếp, không lặp lại câu hỏi, "
        "và chỉ đưa ra câu trả lời mà không kèm phần ngữ cảnh."
    ),
}

EXP_INSTRUCTION_BY_LANG = {
    "en": (
        "Please provide the objective explanations of why model generates "
        "the answers of the given questions based on your thoughts. "
        "Guess the reason why model provides answer no matter it is wrong or correct. "
        "Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. "
        "Every explanations should begin with <EXP>. "
        "Make sure not to repeat the input questions and answers. "
        "Please only output the explanation sentences."
    ),
    "vi": (
        "Hãy đưa ra lời giải thích khách quan vì sao mô hình đưa ra câu trả lời như vậy "
        "cho câu hỏi bên dưới, dựa trên suy nghĩ của bạn. "
        "Giải thích lý do mô hình trả lời như thế, bất kể câu trả lời đúng hay sai. "
        "Tuyệt đối không tự trả lời câu hỏi và không gợi ý cách trả lời tốt hơn. "
        "Mỗi lời giải thích phải bắt đầu bằng <EXP>. "
        "Không lặp lại câu hỏi và câu trả lời. "
        "Chỉ xuất ra các câu giải thích."
    ),
}

# Backward-compatible aliases: the English pack is the historical default.
BASE_INSTRUCTION = BASE_INSTRUCTION_BY_LANG["en"]
TASK_INSTRUCTION_MC = TASK_INSTRUCTION_MC_BY_LANG["en"]
TASK_INSTRUCTION_QA = TASK_INSTRUCTION_QA_BY_LANG["en"]
EXP_INSTRUCTION = EXP_INSTRUCTION_BY_LANG["en"]

_COT_SUFFIX = {
    "en": "### Response: Let's think step by step.",
    "vi": "### Response: Hãy suy luận từng bước.",
}


def check_lang(lang: str) -> str:
    if lang not in BASE_INSTRUCTION_BY_LANG:
        raise ValueError(
            f"prompt_lang must be one of {sorted(BASE_INSTRUCTION_BY_LANG)}, got '{lang}'"
        )
    return lang


def task_instruction_for(is_multiple_choice: bool, lang: str = "en") -> str:
    check_lang(lang)
    table = TASK_INSTRUCTION_MC_BY_LANG if is_multiple_choice else TASK_INSTRUCTION_QA_BY_LANG
    return table[lang]


def exp_instruction_for(lang: str = "en") -> str:
    check_lang(lang)
    return EXP_INSTRUCTION_BY_LANG[lang]


def task_prompt(task_instruction: str, question: str, hint: str = None,
                passage: str = None, cot: bool = True, lang: str = "en") -> str:
    """Render a predictor prompt, optionally with a hint and/or a passage.

    The `### Section:` markers stay English in both packs — they are structure,
    not language, and the models were prompted with them in every prior run.
    """
    check_lang(lang)
    parts = [BASE_INSTRUCTION_BY_LANG[lang], "", f"### Instruction: {task_instruction}", ""]
    if hint:
        parts += [f"### Hint: {hint}", ""]
    if passage:
        parts += [f"### Context: {passage}", ""]
    parts += [f"### Input: {question}", ""]
    parts.append(_COT_SUFFIX[lang] if cot else "### Response:")
    return "\n".join(parts)


def explanation_prompt(exp_instruction: str, question: str, answer: str,
                       passage: str = None) -> str:
    if passage:
        return f"{exp_instruction}\n\n### Input: Passage:{passage}\nQ:{question}\nA:{answer}"
    return f"{exp_instruction}\n\n### Input: Q:{question}\nA:{answer}"


_COUNTERFACTUAL_OPEN = {
    "en": (
        "Can you generate a edited version of sentence-1 with opposite meaning "
        "where it states why the model generates the answer in the passage? "
        "Make sure the output sentence is purely edited from sentence-1."
    ),
    "vi": (
        "Hãy viết lại câu-1 thành một phiên bản mang nghĩa ngược lại, "
        "vẫn nói về lý do mô hình đưa ra câu trả lời trong đoạn văn. "
        "Câu xuất ra phải được chỉnh sửa thuần túy từ câu-1."
    ),
}

_COUNTERFACTUAL_MC = {
    "en": (
        "Please generate one example of obtaining the opposite meaning from given sentence. "
        "Make sure you output sentences only."
    ),
    "vi": (
        "Hãy tạo đúng một câu mang nghĩa ngược lại với câu sau. "
        "Chỉ xuất ra câu đó, không thêm gì khác."
    ),
}


def counterfactual_prompt(explanation: str, open_ended: bool = False,
                          lang: str = "en") -> str:
    check_lang(lang)
    if open_ended:
        return f"{_COUNTERFACTUAL_OPEN[lang]}\n\nSentence-1: {explanation}\n\n"
    return f"{_COUNTERFACTUAL_MC[lang]}\n\nSentences: {explanation}\n\n"


_LOCAL_OPT = {
    "en": {
        "meta": (
            "I have some texts along with their corresponding scores. "
            "The texts are the possible explanation of the following given question and answer. "
            "The texts are arranged in random order based on their scores, "
            "where higher scores indicate better quality. "
            "The scores are calculated as how relative is the texts toward the given question and answer as the explanation. "
            "The scores ranges from 0 to 1 based on your output text."
        ),
        "task": (
            "The following exemplars show how to apply your text: "
            "You replace <EXP> with your text. "
            "We say your output is bad if your output obtains lower scores than previous text, "
            "and we say your output is good if your output obtains higher scores than previous text. "
            "The output should begin with <EXP>."
        ),
        "act": (
            "Please provide new objective text to describe why the answers are given toward the questions based on your thoughts. "
            "Guess the reason no matter it is wrong or correct. "
            "Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. "
            "Every explanations should begin with <EXP>. "
            "Make sure not to repeat the input questions and answers. "
            "Please only output the explanation sentences."
        ),
        "text": "Text",
        "score": "Score",
    },
    "vi": {
        "meta": (
            "Tôi có một số đoạn văn kèm điểm số tương ứng. "
            "Các đoạn văn là lời giải thích khả dĩ cho câu hỏi và câu trả lời dưới đây. "
            "Điểm càng cao chất lượng càng tốt. "
            "Điểm được tính theo mức độ phù hợp của đoạn văn khi dùng làm lời giải thích "
            "cho câu hỏi và câu trả lời đã cho, nằm trong khoảng từ 0 đến 1."
        ),
        "task": (
            "Cách dùng đoạn văn của bạn: bạn thay <EXP> bằng nội dung của mình. "
            "Nội dung bị coi là kém nếu đạt điểm thấp hơn các đoạn trước, "
            "và được coi là tốt nếu đạt điểm cao hơn các đoạn trước. "
            "Phần xuất ra phải bắt đầu bằng <EXP>."
        ),
        "act": (
            "Hãy đưa ra MỘT lời giải thích khách quan mới vì sao câu trả lời được đưa ra "
            "cho câu hỏi, dựa trên suy nghĩ của bạn, nhắm tới điểm cao hơn. "
            "Đoán lý do bất kể câu trả lời đúng hay sai. "
            "Tuyệt đối không tự trả lời câu hỏi và không gợi ý cách trả lời tốt hơn. "
            "Mỗi lời giải thích phải bắt đầu bằng <EXP>. "
            "Không lặp lại câu hỏi và câu trả lời. "
            "Chỉ xuất ra các câu giải thích."
        ),
        "text": "Đoạn văn",
        "score": "Điểm",
    },
}


def local_optimizer_prompt(explanations: list, scores: list, question: str,
                           answer: str, lang: str = "en") -> str:
    """LLM-OPT over explanation text (per-instance refinement)."""
    check_lang(lang)
    pack = _LOCAL_OPT[lang]
    history = "".join(
        f"{pack['text']}:\n{exp}\n{pack['score']}:\n{score}\n\n"
        for exp, score in zip(explanations, scores)
    )
    return f"{pack['meta']}\n\nQ:{question}\nA:{answer}\n\n{pack['task']}\n\n{history}\n\n{pack['act']}"


_GLOBAL_OPT = {
    "en": {
        "meta": (
            "Your task is to generate the instructions <INS> for providing model explanations. "
            "Below are some previous instructions with their scores. "
            "The score is calculated as the flipping answer rates and ranges from 0 to 1."
        ),
        "task": (
            "Generate an instructions that is different from all the instructions <INS> above "
            "and has a higher score than all the instructions <INS> above. "
            "The instructions should begin with <INS> and end with </INS>. "
            "The instructions should be concise, effective, and generally applicable to all problems above."
        ),
        "ins": "Instructions",
        "score": "Score",
    },
    "vi": {
        "meta": (
            "Nhiệm vụ của bạn là tạo câu lệnh <INS> dùng để yêu cầu giải thích hành vi mô hình. "
            "Dưới đây là các câu lệnh trước kèm điểm số. "
            "Điểm là tỉ lệ lật đáp án của các lời giải thích tạo ra, nằm trong khoảng từ 0 đến 1."
        ),
        "task": (
            "Hãy tạo MỘT câu lệnh mới bằng tiếng Việt, khác mọi câu lệnh <INS> ở trên "
            "và có khả năng đạt điểm cao hơn tất cả. "
            "Câu lệnh vẫn phải cấm mô hình tự trả lời câu hỏi. "
            "Câu lệnh phải bắt đầu bằng <INS> và kết thúc bằng </INS>, "
            "ngắn gọn, hiệu quả và áp dụng được cho mọi bài toán ở trên."
        ),
        "ins": "Câu lệnh",
        "score": "Điểm",
    },
}


def global_optimizer_prompt(prompts: list, scores: list, lang: str = "en") -> str:
    """LLM-OPT over the instruction itself (OPRO-style prompt search)."""
    check_lang(lang)
    pack = _GLOBAL_OPT[lang]
    history = "".join(
        f"{pack['ins']}:\n{prompt}\n{pack['score']}:\n{score}\n\n"
        for prompt, score in zip(prompts, scores)
    )
    return f"{pack['meta']}\n\n{history}\n\n{pack['task']}\n\n"
