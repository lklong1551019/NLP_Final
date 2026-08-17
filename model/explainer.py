import os
import openai
import torch
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import ipdb


def reponse_xai_model(prompt, args, xai_local_model=None, xai_local_tokenizer=None):
    # claude-2
    # gpt35turboinstruct
    model_name = args.xai_model

    if model_name == "gpt35":
        try:
            input_count = len(prompt.split())*4/3
        except:
            input_count = len(prompt[0].split())*4/3

        if input_count >= 3000.0:
            return "Cost warning."

        openai.api_type = "azure"
        openai.api_base = "https://openai-datalab.openai.azure.com/"
        openai.api_version = "2023-05-15"
        openai.api_key = args.gpt_key
        # ipdb.set_trace()
        response = openai.ChatCompletion.create(
            engine="gpt35turbo", # engine = "deployment_name".
            temperature = 0.9,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": f"{prompt}"}
            ]
        )
        response = response['choices'][0]['message']['content']

        txt_count = len(response.split())*4/3
        print(f"[INFO] ======  Input Tkn Count: {input_count} || Output Tkn Count: {txt_count}\n")

    elif model_name == "claude":
        max_tokens = args.max_tokens
        anthropic = Anthropic(
            api_key=args.claude_key,
        )
        model_id = 'claude-2'
        completion = anthropic.completions.create(
                model=model_id,
                max_tokens_to_sample=max_tokens,
                prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
        )
        response = completion.completion

    elif model_name == "phi":
        # model_id = "microsoft/phi-2"
        xai_local_tokenizer.padding_side = "left"
        xai_local_model.eval()

        model_inputs_all = xai_local_tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(xai_local_model.device)
        generate_ids = xai_local_model.generate(**model_inputs_all, do_sample=True, temperature=args.temp_exp, max_new_tokens=500)
        ans_tkn = xai_local_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = ans_tkn[0].split("### Response:")[-1]

    elif model_name == "deepseek":
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", getattr(args, 'deepseek_key', None)),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        deepseek_model = getattr(args, 'deepseek_model', None) \
            or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

        input_text = prompt if isinstance(prompt, str) else prompt[0]
        completion = client.chat.completions.create(
            model=deepseek_model,
            temperature=float(args.temp_exp),
            max_tokens=args.max_tokens,
            messages=[
                {"role": "system", "content": "You are an expert at explaining language model behavior."},
                {"role": "user", "content": input_text},
            ],
        )
        response = completion.choices[0].message.content
        # An empty reply would be split into a bogus "explanation" and silently scored.
        if not (response or "").strip():
            raise RuntimeError(f"DeepSeek ({deepseek_model}) returned empty content")

    elif model_name == "openrouter":
        from model.openrouter_client import generate as _or_generate
        response = _or_generate(prompt, args)

    else:
        raise ValueError(
            f"Unknown --xai_model '{model_name}'. "
            f"Expected one of: gpt35, claude, phi, deepseek, openrouter."
        )

    return response

def generate_exp_prompt(task_instruction, input_zip, output_ans, args):

    if args.data in ["ecqa", "copa", "social", "xcopa", "xcopa_vi", "copa_en"]:
        qa_pair = zip(input_zip, output_ans)
        exp_prompt = [f"{task_instruction}\n\n### Input: Q:{item[0]}\nA:{item[1]}" for item in qa_pair]

    elif args.data == "trivaqa":
        question = [x[0] for x in input_zip]
        passage = [x[1] for x in input_zip]

        qa_pair = zip(passage, question, output_ans)
        exp_prompt = [f"{task_instruction}\n\n### Input: Passage:{item[0]}\nQ:{item[1]}\nA:{item[2]}" for item in qa_pair]

    return exp_prompt

def generate_global_xai_prompt(xai_prompts_list, scores_list, args):

    if args.xai_model not in ["claude", "gpt35"]:
        few_shot_score = ""
        for i in range(len(xai_prompts_list)):
            xai_prompt = xai_prompts_list[i]
            score = scores_list[i]
            few_shot_score += f"Prompt:\n{xai_prompt}\nScore:\n{score}\n\n"

        meta_instruction = f"Your task is to generate the general prompts <INS> for language model generating model explanations of each question. \
                            Below are some previous prompt with their scores in the ### Inputs. \
                            The score is calculated as the flipping answer rates and ranges from 0 to 1."
        task_instruction = f"Generate an prompt <INS> that is different from all prompt <INS> in ### Inputs above \
                            and has a higher score than all the prompts <INS> from ### Inputs. \
                            The prompts should begin with <INS> and end with </INS> and follow the format of the examples in ### Inputs . \
                            The prompts should be concise, effective, and generally applicable to all problems above. \
                            Make sure to generate the prompts <INS> only"

        xai_final_prompt = f"### System instruction: {meta_instruction}\n\n### Inputs: {few_shot_score}\n\n \
                            ### Task Instruction: {task_instruction}\n\n ### Response:"

    else:
        meta_instruction = f"Your task is to generate the instructions <INS> for providing model explanations. \
                            Below are some previous instructions with their scores. \
                            The score is calculated as the flipping answer rates and ranges from 0 to 1."
        task_instruction = f"Generate an instructions that is different from all the instructions <INS> above \
                            and has a higher score than all the instructions <INS> above. \
                            The instructions should begin with <INS> and end with </INS>. \
                            The instructions should be concise, effective, and generally applicable to all problems above."

        few_shot_score = ""
        for i in range(len(xai_prompts_list)):
            xai_prompt = xai_prompts_list[i]
            score = scores_list[i]
            few_shot_score += f"Instructions:\n{xai_prompt}\nScore:\n{score}\n\n"

        xai_final_prompt = f"{meta_instruction}\n\n{few_shot_score}\n\n \
                            {task_instruction}\n\n"

    return xai_final_prompt

def generate_local_xai_prompt(xai_prompts_list, scores_list, question, output_ans):

    question_answer_list = f"Q:{question}\nA:{output_ans}"

    meta_instruction = f"I have some texts along with their corresponding scores. \
                        The texts are the possible explanation of the following given question and answer. \
                        The texts are arranged in random order based on their scores, \
                        where higher scores indicate better quality. \
                        The scores are calculated as how relative is the texts toward the given question and answer as the explanation. \
                        The scores ranges from 0 to 1 based on your output text."

    task_instruction = f"The following exemplars show how to apply your text: \
                        You replace <EXP> with your text. \
                        We say your output is bad if your output obtains lower scores than previous text, \
                        and we say your output is good if your output obtains higher scores than previous text. \
                        The output should begin with <EXP>."

    act_instruction = f"Please provide new objective text to describe why the answers are given toward the questions based on your thoughts. \
                        Guess the reason no matter it is wrong or correct.\
                        Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. \
                        Every explanations should begin with <EXP>. \
                        Make sure not to repeat the input questions and answers. \
                        Please only output the explanation sentences."

    few_shot_score = ""
    for i in range(len(scores_list)):
        xai_prompt = xai_prompts_list[i]
        score = scores_list[i]
        # except:
        #     print(scores_list, len(scores_list))
        #     print(xai_prompts_list, len(xai_prompts_list))
        #     exit()
        few_shot_score += f"Text:\n{xai_prompt}\nScore:\n{score}\n\n"

    xai_final_prompt = f"{meta_instruction}\n\n{question_answer_list}\n\n \
                        {task_instruction}\n\n{few_shot_score}\n\n{act_instruction}"

    return xai_final_prompt

def generate_counterfact_prompt(explanation, args):

    # instruction = f"Please generate one counterfactual example obtaining opposite meaning of the given sentence. \
    #                 Make sure you output sentence only."
    if args.data in ["ecqa", "copa", "social", "xcopa", "xcopa_vi", "copa_en"]:
        instruction = f"Please generate one example of obtaining the opposite meaning from given sentence. \
                        Make sure you output sentences only."
        final_prompt = f"{instruction}\n\nSentences: {explanation}\n\n"

    elif args.data == "trivaqa":
        instruction = f"Can you generate a edited version of sentence-1 with opposite meaning \
                        where it states why the model generates the answer in the passage? \
                        Make sure the output sentence is purely edited from sentence-1."
        final_prompt = f"{instruction}\n\nSentence-1: {explanation}\n\n"

    return final_prompt
