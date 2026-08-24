import torch
import collections
import numpy as np
import copy
import sys
import os
import json
import argparse
from tqdm import tqdm
from random import sample
from datasets import load_dataset, Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from model.predictor import load_model, generate_api_predictor_output, diff_task_score_ecqa, diff_task_score_trivaqa
from model.predictor import generate_predictor_output_ecqa, generate_predictor_output_trivaqa
from model.predictor import contains_answer
from model.predictor import CONTROL as PRED_CONTROL
from model import llm_api
from model.explainer import reponse_xai_model, generate_counterfact_prompt, generate_local_xai_prompt, generate_exp_prompt

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def preprocess_ecqa():
    train_dict = collections.defaultdict(list)

    hg_data = load_dataset("yangdong/ecqa", 'rc')
    hg_data = hg_data['train']
    # import_len = len(hg_data["q_text"])//data_portion
    question = hg_data["q_text"]
    answer = hg_data["q_ans"]
    op1 = hg_data["q_op1"]
    op2 = hg_data["q_op2"]
    op3 = hg_data["q_op3"]
    op4 = hg_data["q_op4"]
    op5 = hg_data["q_op5"]
    option = zip(op1, op2, op3, op4, op5)
    choice = [ f"[choice]{opt[0]}@ [choice]{opt[1]}@ [choice]{opt[2]}@ [choice]{opt[3]}@ [choice]{opt[4]}@" for opt in option ]

    for idx, ques in enumerate(question):
        question = f"{ques}\n### Choices: {choice[idx]}"
        train_dict['question'].append(question)
        train_dict['answer'].append(answer[idx])
        # qa_dict[question] = answer[idx]
    return train_dict #, qa_dict

def preprocess_social():
    train_dict = collections.defaultdict(list)

    hg_data = load_dataset("tasksource/bigbench", 'social_iqa')
    hg_data = hg_data['validation']
    # import_len = len(hg_data["q_text"])//data_portion
    question = hg_data["inputs"]
    answer = hg_data["targets"]
    option = hg_data["multiple_choice_targets"]
    choice = []
    for opt in option:
        choice_txt = ""
        for i in opt:
            choice_txt += f"[choice]{i}@ "
        choice.append(choice_txt)

    for idx, ques in enumerate(question):
        question = f"{ques}\n### Choices: {choice[idx]}"
        train_dict['question'].append(question)
        train_dict['answer'].append(answer[idx][0])
        # qa_dict[question] = answer[idx]
    return train_dict

def preprocess_trivaqa():

    # Load Data2
    all_na_data = load_dataset("THUDM/LongBench", 'triviaqa_e')
    passage, question, answer = [], [], []
    na_data = all_na_data["test"]
    for i in na_data:
        pure_text = i["input"]
        pure_text = pure_text.split("Passage:\n")[-1]
        pure_text = pure_text.split("Question:\n")
        passage_txt = pure_text[0]
        question_txt = pure_text[-1].split("Answer:\n")[0]

        passage.append(passage_txt)
        question.append(question_txt)
        answer.append(i['answers'])

    # Train
    train_dict = collections.defaultdict(list)
    train_dict['passage'].extend(passage)
    train_dict['question'].extend(question)
    train_dict['answer'].extend(answer)

    return train_dict

def preprocess_copa():
    train_dict = collections.defaultdict(list)

    all_na_data = load_dataset("pkavumba/balanced-copa")
    passage, question, answer = [], [], []
    hg_data = all_na_data["test"]
    question_text = hg_data["premise"]
    question_purp = hg_data["question"]
    answer = hg_data["label"]
    op1 = hg_data["choice1"]
    op2 = hg_data["choice2"]
    option = list(zip(op1, op2))
    choice = [ f"[choice]{opt[0]}@ [choice]{opt[1]}@" for opt in option]

    for idx, ques_txt in enumerate(question_text):
        question = f"###Question: What is the {question_purp[idx]} of the Promise?\n### Premise: {ques_txt}\n### Choices: {choice[idx]}"
        train_dict['question'].append(question)
        train_dict['answer'] = [opt[answer[idx]] for opt in option]
    return train_dict

def split_reply(text):
    """Split an explainer reply into segments, dropping empty ones.

    Some explainers (e.g. GPT-5.6) pad replies with blank lines; a bare
    split on blank lines then yields empty segments which get scored as
    explanations - an empty "explanation" whose counterfactual flips the
    predictor manufactures a fake faithfulness signal.
    """
    parts = [s.strip() for s in text.split("\n\n") if s.strip()]
    return parts if parts else [text.strip()]


def preprocess_xcopa_vi(lang="vi", split="test"):
    """Load XCOPA dataset for cross-lingual experiments.
    Available langs: et, ht, id, it, qu, sw, ta, th, tr, vi, zh
    Available splits: validation (100), test (500)
    """
    train_dict = collections.defaultdict(list)

    all_na_data = load_dataset("cambridgeltl/xcopa", lang)
    hg_data = all_na_data[split]
    question_text = hg_data["premise"]
    question_purp = hg_data["question"]
    labels = hg_data["label"]
    op1 = hg_data["choice1"]
    op2 = hg_data["choice2"]
    option = list(zip(op1, op2))
    choice = [f"[choice]{opt[0]}@ [choice]{opt[1]}@" for opt in option]

    # The question scaffold is part of the prompt under test, so it has to follow
    # PROMPT_LANG like every other template. Preprocessing runs before argparse,
    # so read the switch from the environment via llm_api.prompt_lang().
    vi_scaffold = llm_api.prompt_lang() == "vi"

    for idx, ques_txt in enumerate(question_text):
        if vi_scaffold:
            purp_vi = "nguyên nhân" if question_purp[idx] == "cause" else "kết quả"
            question = (
                f"### Câu hỏi: Đâu là {purp_vi} của Tiền đề?\n"
                f"### Tiền đề: {ques_txt}\n"
                f"### Lựa chọn: {choice[idx]}"
            )
        else:
            # The paper's original English scaffold, verbatim.
            question = (
                f"###Question: What is the {question_purp[idx]} of the Premise?\n"
                f"### Premise: {ques_txt}\n"
                f"### Choices: {choice[idx]}"
            )
        train_dict['question'].append(question)
        train_dict['answer'].append(option[idx][labels[idx]])
    return train_dict

def preprocess_copa_en(split="train"):
    """Load original English balanced-COPA dataset.
    Available splits: train (1000), test (500)
    """
    train_dict = collections.defaultdict(list)

    all_na_data = load_dataset("pkavumba/balanced-copa")
    hg_data = all_na_data[split]
    question_text = hg_data["premise"]
    question_purp = hg_data["question"]
    labels = hg_data["label"]
    op1 = hg_data["choice1"]
    op2 = hg_data["choice2"]
    option = list(zip(op1, op2))
    choice = [f"[choice]{opt[0]}@ [choice]{opt[1]}@" for opt in option]

    for idx, ques_txt in enumerate(question_text):
        question = (
            f"###Question: What is the {question_purp[idx]} of the Premise?\n"
            f"### Premise: {ques_txt}\n"
            f"### Choices: {choice[idx]}"
        )
        train_dict['question'].append(question)
        train_dict['answer'].append(option[idx][labels[idx]])
    return train_dict

def get_args():
    parser = argparse.ArgumentParser(description='Process Capsule Prompt.')
    parser.add_argument('--device_num', nargs='+', default="0")
    parser.add_argument('--gpt_key', type=str)
    parser.add_argument('--claude_key', type=str)
    parser.add_argument('--data', type=str, default='ecqa')
    parser.add_argument('--pred_model', type=str, default='vicuna')
    parser.add_argument('--xai_model', type=str, default='claude')
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--temp_exp', type=float, default=0.01)
    parser.add_argument('--xai_iter', type=int, default=20)
    parser.add_argument('--ques_idx_start', type=int, default=40)
    parser.add_argument('--ques_idx_end', type=int, default=40)
    parser.add_argument('--save_cf_file_path', type=str, default=None)
    parser.add_argument('--save_file_path', type=str, default="./results")
    # New arguments
    parser.add_argument('--deepseek_key', type=str, default=None,
                        help='DeepSeek API key (or set DEEPSEEK_API_KEY env var)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Skip questions whose result file already exists in '
                             '--save_file_path. Only safe when the sampling '
                             'settings match, since the filename does not encode them.')
    parser.add_argument('--top_p_exp', type=float, default=None,
                        help='Top-p for the Explainer (paper Table 2 uses 0.9)')
    parser.add_argument('--litellm_pred_model', type=str, default=None,
                        help='Model id for an API-served Predictor (--pred_model litellm). '
                             'Defaults to $LITELLM_PRED_MODEL.')
    parser.add_argument('--deepseek_model', type=str, default=None,
                        choices=['deepseek-v4-pro', 'deepseek-v4-flash'],
                        help='DeepSeek model variant')
    parser.add_argument('--xcopa_lang', type=str, default='vi',
                        help='XCOPA language code (vi, it, th, etc.)')
    parser.add_argument('--data_split', type=str, default='test',
                        choices=['train', 'test', 'validation'],
                        help='Dataset split to use')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Use 4-bit quantization (for 8GB GPUs)')
    parser.add_argument('--no_4bit', dest='load_in_4bit', action='store_false',
                        help='Disable 4-bit quantization')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Config setup
    sleep_range = [x for x in range(10)]
    task_acc = 0.0
    args = get_args()
    max_memory = {int(i): '45GB' for i in args.device_num}
    start_idx = args.ques_idx_start
    print(f"============  GPU Memory: {max_memory}")

    # Load data
    if args.data == "ecqa":
        train_dict = preprocess_ecqa()
        task_instruction = f"Please select a correct choice for the each question. \
                            Make sure not to repeat the input context."
        exp_instruction = f"Please provide the objective explanations of why model generates \
                            the answers of the given questions based on your thoughts. \
                            Guess the reason why model provides answer no matter it is wrong or correct.\
                            Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. \
                            Every explanations should begin with <EXP>. \
                            Make sure not to repeat the input questions and answers. \
                            Please only output the explanation sentences."

        # Load predictor
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa

    elif args.data == "trivaqa":
        train_dict = preprocess_trivaqa()
        task_instruction = f"Please answer the question based on the context. \
                            Be sure to provid precise answer and make sure not to repeat the question. \
                            Answer the question directly and precisely and provide the answer only without context supports."
        exp_instruction = f"Please provide the objective explanations of why model generates \
                            the answers of the given questions from the given passages based on your thoughts. \
                            Guess the reason why model provides answer no matter it is wrong or correct.\
                            Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. \
                            Every explanations should begin with <EXP>. \
                            Make sure not to repeat the input questions and answers. \
                            Please only output the explanation sentences."

        # Load predictor
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_trivaqa

        # LLM-OPT function
        diff_task_score = diff_task_score_trivaqa

    elif args.data == "copa":
        train_dict = preprocess_copa()
        task_instruction = f"Please select a correct choice for the each question. \
                            Make sure not to repeat the input context."
        exp_instruction = f"Please provide the objective explanations of why model generates \
                            the answers of the given questions based on your thoughts. \
                            Guess the reason why model provides answer no matter it is wrong or correct.\
                            Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. \
                            Every explanations should begin with <EXP>. \
                            Make sure not to repeat the input questions and answers. \
                            Please only output the explanation sentences."

        # Load predictor
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa

    elif args.data == "social":
        args.data = "ecqa"
        train_dict = preprocess_social()
        task_instruction = f"Please select a correct choice for the each question. \
                            Make sure not to repeat the input context."
        exp_instruction = f"Please provide the objective explanations of why model generates \
                            the answers of the given questions based on your thoughts. \
                            Guess the reason why model provides answer no matter it is wrong or correct.\
                            Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. \
                            Every explanations should begin with <EXP>. \
                            Make sure not to repeat the input questions and answers. \
                            Please only output the explanation sentences."

        # Load predictor
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa

        # Reverse data name
        args.data = "social"

    elif args.data == "xcopa":
        train_dict = preprocess_xcopa_vi(lang=args.xcopa_lang, split=args.data_split if args.data_split != 'train' else 'test')
        task_instruction = f"Please select a correct choice for the each question. \
                            Make sure not to repeat the input context."
        exp_instruction = f"Please provide the objective explanations of why model generates \
                            the answers of the given questions based on your thoughts. \
                            Guess the reason why model provides answer no matter it is wrong or correct.\
                            Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. \
                            Every explanations should begin with <EXP>. \
                            Make sure not to repeat the input questions and answers. \
                            Please only output the explanation sentences."

        # Load predictor
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa

    elif args.data == "xcopa_vi":
        train_dict = preprocess_xcopa_vi(lang=args.xcopa_lang, split=args.data_split if args.data_split != 'train' else 'test')
        if not llm_api.vi_prompts(args):
            # PROMPT_LANG=en: the paper's original English instructions, verbatim.
            task_instruction = f"Please select a correct choice for the each question.                             Make sure not to repeat the input context."
            exp_instruction = f"Please provide the objective explanations of why model generates                             the answers of the given questions based on your thoughts.                             Guess the reason why model provides answer no matter it is wrong or correct.                            Make sure not answer the questions or provide any suggestions to better answer the questions by yourself.                             Every explanations should begin with <EXP>.                             Make sure not to repeat the input questions and answers.                             Please only output the explanation sentences."
        else:
            task_instruction = "Hãy chọn đáp án đúng cho mỗi câu hỏi. Lưu ý không lặp lại phần ngữ cảnh đầu vào."
            exp_instruction = "Dựa trên suy luận của bạn, hãy giải thích một cách khách quan lý do mô hình đưa ra câu trả lời cho các câu hỏi này. Hãy đưa ra lý do bất kể câu trả lời đó đúng hay sai. Tuyệt đối không tự trả lời câu hỏi hay đưa ra gợi ý để trả lời tốt hơn. Mỗi câu giải thích phải bắt đầu bằng <EXP>. Không lặp lại câu hỏi hay câu trả lời đầu vào. Lưu ý: Chỉ xuất ra các câu giải thích, không thêm bất kỳ nội dung nào khác."

        # Load predictor
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa

    elif args.data == "copa_en":
        train_dict = preprocess_copa_en(split=args.data_split)
        task_instruction = f"Please select a correct choice for the each question. \
                            Make sure not to repeat the input context."
        exp_instruction = f"Please provide the objective explanations of why model generates \
                            the answers of the given questions based on your thoughts. \
                            Guess the reason why model provides answer no matter it is wrong or correct.\
                            Make sure not answer the questions or provide any suggestions to better answer the questions by yourself. \
                            Every explanations should begin with <EXP>. \
                            Make sure not to repeat the input questions and answers. \
                            Please only output the explanation sentences."

        # Load predictor
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa


    def _result_path(question_idx):
        name = (f"local_{args.data}_{args.xai_model}_{args.pred_model}"
                f"_iter-{args.xai_iter}_sample-{question_idx}.json")
        return os.path.join(args.save_file_path, name)

    def _already_done(question_idx):
        path = _result_path(question_idx)
        return os.path.isfile(path) and os.path.getsize(path) > 0

    skipped = 0
    for idx in range(start_idx, args.ques_idx_end):
        # Resume: a question whose result file already exists is not redone.
        # NOTE the filename encodes only data/xai/pred/iter - not temperature or
        # top-p. Do not resume into a directory produced with different sampling
        # settings; use a fresh --save_file_path instead.
        if args.resume and _already_done(idx):
            skipped += 1
            print(f"============ Resume: skipping question {idx} (already done)")
            continue

        # Select init score for LLM optimization
        fed_score_llm = 0.0
        fed_score_org = 0.0

        # Select data for LLM optimization
        if args.data in ["ecqa", "copa", "social", "xcopa", "xcopa_vi", "copa_en"]:
            question = [train_dict['question'][idx]]
            answer = [train_dict['answer'][idx]] #[qa_dict[question[0]]]
            input_zip = question

        elif args.data == "trivaqa":
            passage = [train_dict['passage'][idx]]
            question = [train_dict['question'][idx]]
            answer = [train_dict['answer'][idx]]
            input_zip = list(zip(question, passage))


        # LLM optimization
        print(f"============ Starting Optimization --> Question Index: {idx}")
        xai_list = []
        scores_list = []
        xai_prompts_write = []
        cf_write = []

        # Initial state
        # Generate prediction from LLMs
        output_ans = generate_ans_function(pred_model, pred_tokenizer, task_instruction, input_zip, answer, args)
        if args.data in ["ecqa", "copa", "social", "xcopa", "xcopa_vi", "copa_en"]:
            if contains_answer(answer[0], output_ans[0]):
                target = f"============ Corrct --> Q:{question} || GT-A:{answer[0]} || LLM-A:{output_ans[0]}"
            else:
                target = f"============ Wrong  --> Q:{question} || GT-A:{answer[0]} || LLM-A:{output_ans[0]}"
            print(target)

        elif args.data == "trivaqa":
            flag = 0
            for ans in answer[0]:
                if ans.lower() in output_ans[0].lower().strip():
                    target = f"============ Corrct --> Q:{question} || GT-A:{ans} || LLM-A:{ans}"
                    flag += 1
                    break
            if flag == 0:
                target = f"============ Wrong  --> Q:{question} || GT-A:{answer[0]} || LLM-A:{output_ans[0]}"
            print(target)


        # Generate init true explanation
        output_exp_prompt = generate_exp_prompt(exp_instruction, input_zip, output_ans, args)
        exp_reply = reponse_xai_model(output_exp_prompt, args)
        exp_reply = split_reply(exp_reply.split(":\n\n")[-1])
        xai_list.extend(exp_reply)
        # print(f"============ Init Exp: {exp_reply[0]}")

        with tqdm(total=args.xai_iter) as pbar:
            for iter in range(args.xai_iter):
                print(f"============ Step:{iter} LLM Optimizing")

                # Generate counterfactual explanation
                counter_xai_prompt = generate_counterfact_prompt(exp_reply, args)
                counter_exp_reply = reponse_xai_model(counter_xai_prompt, args)
                counter_exp_reply = split_reply(counter_exp_reply.split(":\n\n")[-1])
                cf_write.append(counter_exp_reply)
                # print(f"============ Counterfact Exp: {counter_exp_reply[0]}")

                # Score difference for updating xai prompt format
                diff_score = diff_task_score(pred_model, pred_tokenizer, task_instruction, input_zip, answer, exp_reply, counter_exp_reply, args)
                scores_list.append(diff_score)

                # Save explanation
                save_explanation = xai_list[-1]
                record = {"Score": diff_score, "XAI prompt": save_explanation}
                if PRED_CONTROL.get("last"):
                    # Irrelevant-hint baseline for this same instance.
                    record["ControlScore"] = PRED_CONTROL["last"]["diff_random"]
                    record["ControlHint"] = PRED_CONTROL["last"]["hint"]
                xai_prompts_write.append(record)
                print(f"=== Score: {diff_score} || Explanation: {save_explanation}")

                if iter%5 == 0 and sum(scores_list) != 0:
                    print("============ Early Stop Optimization")
                    break
                elif iter%5 == 0 and "X" in output_ans:
                    print("============ Bad Answer")
                    break

                # LLM optimizer
                xai_prompt = generate_local_xai_prompt(xai_list, scores_list, question, output_ans, args)
                exp_reply = reponse_xai_model(xai_prompt, args)
                exp_reply = split_reply(exp_reply.split(":\n\n")[-1])
                
                
                xai_list.extend(exp_reply)
                pbar.update(1)

                if iter%5 == 0 and ("apologize" in exp_reply[0]) or ("Unfortunately" in exp_reply[0]):
                    print(f"============ {exp_reply[0]}")
                    print("============ Refuse to Answer")
                    break

        # File saving
        result_save_path = args.save_file_path
        result_file_name = f"local_{args.data}_{args.xai_model}_{args.pred_model}_iter-{args.xai_iter}_sample-{idx}.json"
        with open(os.path.join(result_save_path, result_file_name), "w") as f:
            f.write(f"{target}\n")
            for sub_xai_dict in xai_prompts_write:
                f.write(f"{sub_xai_dict}\n")
        print(f"============ Successful File Saved in {result_file_name}")

    if skipped:
        print(f"============ Resume: skipped {skipped} already-completed questions")
    print(f"============ API stats | {llm_api.stats_summary()}")
