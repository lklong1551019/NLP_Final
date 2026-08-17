"""Prompt templates, kept in one place so the paper's Appendix can quote them.

Wording is preserved verbatim from the original implementation. The one
structural change is that the hint slot is now explicit, which is what lets the
"symmetric" metric put the true explanation where the original code left a gap.
"""

BASE_INSTRUCTION = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request of input."
)

TASK_INSTRUCTION_MC = (
    "Please select a correct choice for the each question. "
    "Make sure not to repeat the input context."
)

TASK_INSTRUCTION_QA = (
    "Please answer the question based on the context. "
    "Be sure to provid precise answer and make sure not to repeat the question. "
    "Answer the question directly and precisely and provide the answer only without context supports."
)

EXP_INSTRUCTION = (
    "Please provide the objective explanations of why model generates "
    "the answers of the given questions based on your thoughts. "
    "Guess the reason why model provides answer no matter it is wrong or correct. "
    "Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. "
    "Every explanations should begin with <EXP>. "
    "Make sure not to repeat the input questions and answers. "
    "Please only output the explanation sentences."
)


def task_prompt(task_instruction: str, question: str, hint: str = None,
                passage: str = None, cot: bool = True) -> str:
    """Render a predictor prompt, optionally with a hint and/or a passage."""
    parts = [BASE_INSTRUCTION, "", f"### Instruction: {task_instruction}", ""]
    if hint:
        parts += [f"### Hint: {hint}", ""]
    if passage:
        parts += [f"### Context: {passage}", ""]
    parts += [f"### Input: {question}", ""]
    parts.append("### Response: Let's think step by step." if cot else "### Response:")
    return "\n".join(parts)


def explanation_prompt(exp_instruction: str, question: str, answer: str,
                       passage: str = None) -> str:
    if passage:
        return f"{exp_instruction}\n\n### Input: Passage:{passage}\nQ:{question}\nA:{answer}"
    return f"{exp_instruction}\n\n### Input: Q:{question}\nA:{answer}"


def counterfactual_prompt(explanation: str, open_ended: bool = False) -> str:
    if open_ended:
        instruction = (
            "Can you generate a edited version of sentence-1 with opposite meaning "
            "where it states why the model generates the answer in the passage? "
            "Make sure the output sentence is purely edited from sentence-1."
        )
        return f"{instruction}\n\nSentence-1: {explanation}\n\n"
    instruction = (
        "Please generate one example of obtaining the opposite meaning from given sentence. "
        "Make sure you output sentences only."
    )
    return f"{instruction}\n\nSentences: {explanation}\n\n"


def local_optimizer_prompt(explanations: list, scores: list, question: str, answer: str) -> str:
    """LLM-OPT over explanation text (per-instance refinement)."""
    meta = (
        "I have some texts along with their corresponding scores. "
        "The texts are the possible explanation of the following given question and answer. "
        "The texts are arranged in random order based on their scores, "
        "where higher scores indicate better quality. "
        "The scores are calculated as how relative is the texts toward the given question and answer as the explanation. "
        "The scores ranges from 0 to 1 based on your output text."
    )
    task = (
        "The following exemplars show how to apply your text: "
        "You replace <EXP> with your text. "
        "We say your output is bad if your output obtains lower scores than previous text, "
        "and we say your output is good if your output obtains higher scores than previous text. "
        "The output should begin with <EXP>."
    )
    act = (
        "Please provide new objective text to describe why the answers are given toward the questions based on your thoughts. "
        "Guess the reason no matter it is wrong or correct. "
        "Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. "
        "Every explanations should begin with <EXP>. "
        "Make sure not to repeat the input questions and answers. "
        "Please only output the explanation sentences."
    )
    history = "".join(
        f"Text:\n{exp}\nScore:\n{score}\n\n"
        for exp, score in zip(explanations, scores)
    )
    return f"{meta}\n\nQ:{question}\nA:{answer}\n\n{task}\n\n{history}\n\n{act}"


def global_optimizer_prompt(prompts: list, scores: list) -> str:
    """LLM-OPT over the instruction itself (OPRO-style prompt search)."""
    meta = (
        "Your task is to generate the instructions <INS> for providing model explanations. "
        "Below are some previous instructions with their scores. "
        "The score is calculated as the flipping answer rates and ranges from 0 to 1."
    )
    task = (
        "Generate an instructions that is different from all the instructions <INS> above "
        "and has a higher score than all the instructions <INS> above. "
        "The instructions should begin with <INS> and end with </INS>. "
        "The instructions should be concise, effective, and generally applicable to all problems above."
    )
    history = "".join(
        f"Instructions:\n{prompt}\nScore:\n{score}\n\n"
        for prompt, score in zip(prompts, scores)
    )
    return f"{meta}\n\n{history}\n\n{task}\n\n"
