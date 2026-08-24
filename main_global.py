import collections
import numpy as np
import copy
import sys
import os
import json
import argparse
import random
import traceback
from tqdm import tqdm
from datasets import load_dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from model.predictor import load_model, generate_api_predictor_output, diff_task_score_ecqa, diff_task_score_trivaqa
from model.predictor import generate_predictor_output_ecqa, generate_predictor_output_trivaqa
from model import llm_api
from model.explainer import reponse_xai_model, generate_counterfact_prompt, generate_exp_prompt, generate_global_xai_prompt
import time

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
    return train_dict #, qa_dict

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
    question, answer = [], []
    hg_data = all_na_data["train"]
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

    for idx, ques_txt in enumerate(question_text):
        purp_vi = "nguyên nhân" if question_purp[idx] == "cause" else "kết quả"
        question = (
            f"### Câu hỏi: Đâu là {purp_vi} của Tiền đề?\n"
            f"### Tiền đề: {ques_txt}\n"
            f"### Lựa chọn: {choice[idx]}"
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
    parser.add_argument('--temp_exp', type=float, default=0.9)
    parser.add_argument('--max_tokens', type=int, default=3000)
    parser.add_argument('--xai_iter', type=int, default=3)
    parser.add_argument('--round_xai_iter', type=int, default=10)
    parser.add_argument('--ques_sample', type=int, default=15)
    parser.add_argument('--save_file', '--save_file_path', dest='save_file', type=str, default="./results/global",
                        help='Output dir. --save_file_path is accepted as an alias so the '
                             'same flag works for main_local.py and main_global.py.')
    # New arguments
    parser.add_argument('--use_predictor_reasoning', action='store_true', default=False,
                        help='Query the target model for reasoning and pass it to the explainer.')
    parser.add_argument('--deepseek_key', type=str, default=None,
                        help='DeepSeek API key (or set DEEPSEEK_API_KEY env var)')
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
    args = get_args()
    max_memory = {int(i): '45GB' for i in args.device_num}
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
        
        # Load predictor and explainer
        if args.xai_model not in ["claude", "gpt35", "deepseek", "litellm"]:
            xai_local_model, xai_local_tokenizer = load_model(args.xai_model, max_memory, args.load_in_4bit)
        else:
            xai_local_model, xai_local_tokenizer = "", ""

        if args.pred_model not in ["claude", "gpt35", "litellm"]:
            pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
            generate_ans_function = generate_predictor_output_ecqa
        else:
            pred_model, pred_tokenizer = "", ""
            generate_ans_function = generate_api_predictor_output
            
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
        
        # Load predictor and explainer
        if args.xai_model not in ["claude", "gpt35", "deepseek", "litellm"]:
            xai_local_model, xai_local_tokenizer = load_model(args.xai_model, max_memory, args.load_in_4bit)
        else:
            xai_local_model, xai_local_tokenizer = "", ""

        if args.pred_model not in ["claude", "gpt35", "litellm"]:
            pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
            generate_ans_function = generate_predictor_output_trivaqa
        else:
            pred_model, pred_tokenizer = "", ""
            generate_ans_function = generate_api_predictor_output

        # LLM-OPT function
        diff_task_score = diff_task_score_trivaqa

    elif args.data == "copa":
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
        
        # Load predictor and explainer
        if args.xai_model not in ["claude", "gpt35", "deepseek", "litellm"]:
            xai_local_model, xai_local_tokenizer = load_model(args.xai_model, max_memory, args.load_in_4bit)
        else:
            xai_local_model, xai_local_tokenizer = "", ""

        if args.pred_model not in ["claude", "gpt35", "litellm"]:
            pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
            generate_ans_function = generate_predictor_output_ecqa
        else:
            pred_model, pred_tokenizer = "", ""
            generate_ans_function = generate_api_predictor_output
            
        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa
    # Add xcopa_vi and copa_en data branches
    elif args.data == "xcopa_vi":
        train_dict = preprocess_xcopa_vi(lang=args.xcopa_lang, split=args.data_split if args.data_split != 'train' else 'test')
        task_instruction = "Hãy chọn đáp án đúng cho mỗi câu hỏi. Lưu ý không lặp lại phần ngữ cảnh đầu vào."
        exp_instruction = "Dựa trên suy luận của bạn, hãy giải thích một cách khách quan lý do mô hình đưa ra câu trả lời cho các câu hỏi này. Hãy đưa ra lý do bất kể câu trả lời đó đúng hay sai. Tuyệt đối không tự trả lời câu hỏi hay đưa ra gợi ý để trả lời tốt hơn. Mỗi câu giải thích phải bắt đầu bằng <EXP>. Không lặp lại câu hỏi hay câu trả lời đầu vào. Lưu ý: Chỉ xuất ra các câu giải thích, không thêm bất kỳ nội dung nào khác."

        if args.xai_model not in ["claude", "gpt35", "deepseek", "litellm"]:
            xai_local_model, xai_local_tokenizer = load_model(args.xai_model, max_memory, args.load_in_4bit)
        else:
            xai_local_model, xai_local_tokenizer = "", ""

        if args.pred_model not in ["claude", "gpt35", "litellm"]:
            pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
            generate_ans_function = generate_predictor_output_ecqa
        else:
            pred_model, pred_tokenizer = "", ""
            generate_ans_function = generate_api_predictor_output

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

        if args.xai_model not in ["claude", "gpt35", "deepseek", "litellm"]:
            xai_local_model, xai_local_tokenizer = load_model(args.xai_model, max_memory, args.load_in_4bit)
        else:
            xai_local_model, xai_local_tokenizer = "", ""

        if args.pred_model not in ["claude", "gpt35", "litellm"]:
            pred_model, pred_tokenizer = load_model(args.pred_model, max_memory, args.load_in_4bit)
            generate_ans_function = generate_predictor_output_ecqa
        else:
            pred_model, pred_tokenizer = "", ""
            generate_ans_function = generate_api_predictor_output

        diff_task_score = diff_task_score_ecqa


    # Load prev prompts for LLM-OPT iteration
    cur_xai_iter = args.xai_iter * (args.round_xai_iter-1)
    result_save_path = args.save_file
    result_file_name = f"global_{args.data}_{args.xai_model}_{args.pred_model}_iter-{cur_xai_iter}_sample-{args.ques_sample}.json"
    if os.path.isfile(os.path.join(result_save_path, result_file_name)):
        xai_prompts_list = []
        scores_list = []
        xai_prompts_write = []

        print(f"============ Loading {result_file_name}")
        with open(os.path.join(result_save_path, result_file_name), "r") as f:
            prev_prompt_list = json.load(f)

        for prev_prompt in prev_prompt_list:
            xai_prompts_list.append(prev_prompt["XAI prompt"])
            scores_list.append(prev_prompt["Score"])
        scores_list.pop()
    else:
        xai_prompts_list = [exp_instruction]
        scores_list = []
        xai_prompts_write = []

    # LLM optimization
    print("============ Starting Optimization")
    error_count = 0
    max_errors = max(3, args.xai_iter // 2)
    for iter in range(args.xai_iter):
        try:
            print(f"============ Step:{iter} LLM Optimizing")

            index_value = random.sample(list(enumerate(train_dict['question'])), args.ques_sample)
            # Sample questions for optimization
            if args.data in ["ecqa", "copa", "xcopa_vi", "copa_en"]:
                question, answer = [], []
                for idx, ques in index_value:
                    question.append(ques)
                    answer.append(train_dict['answer'][idx])
                input_zip = question

            elif args.data == "trivaqa":
                passage, question, answer = [], [], []
                for idx, ques in index_value:
                    question.append(ques)
                    passage.append(train_dict['passage'][idx])
                    answer.append(train_dict['answer'][idx])
                input_zip = list(zip(question, passage))

            with tqdm(total=len(input_zip)) as pbar:
                diff_score_avg = 0.0
                for idx in range(len(input_zip)):
                    
                    # Generate prediction from LLMs
                    output_ans = generate_ans_function(pred_model, pred_tokenizer, task_instruction, [input_zip[idx]], [answer[idx]], args)

                    # Get Predictor Reasoning if enabled
                    predictor_reasoning = None
                    if getattr(args, 'use_predictor_reasoning', False):
                        from model.predictor import generate_predictor_reasoning
                        predictor_reasoning_list = generate_predictor_reasoning(pred_model, pred_tokenizer, [input_zip[idx]], output_ans, args)
                        if predictor_reasoning_list:
                            predictor_reasoning = predictor_reasoning_list[0]
                            print(f"============ Predictor Reasoning: {predictor_reasoning}")

                    # Generate true explanation
                    output_exp_prompt = generate_exp_prompt(xai_prompts_list[-1], [input_zip[idx]], output_ans, args, predictor_reasoning=predictor_reasoning)
                    if getattr(args, 'use_predictor_reasoning', False):
                        print(f"============ Init Explainer Prompt:\n{output_exp_prompt[0] if isinstance(output_exp_prompt, list) else output_exp_prompt}\n========================")
                    exp_reply = reponse_xai_model(output_exp_prompt, args, xai_local_model, xai_local_tokenizer)
                    exp_reply = exp_reply.split(":\n\n")[-1]
                    exp_reply = exp_reply.split("\n\n")
                    # print(f"============ True Exp: {exp_reply[0]}")

                    # Generate counterfactual explanation
                    counter_xai_prompt = generate_counterfact_prompt(exp_reply, args)
                    counter_exp_reply = reponse_xai_model(counter_xai_prompt, args, xai_local_model, xai_local_tokenizer)
                    counter_exp_reply = counter_exp_reply.split(":\n\n")[-1]
                    counter_exp_reply = counter_exp_reply.split("\n\n")
                    # print(f"============ Counterfact Exp: {counter_exp_reply[0]}")

                    # Score difference for updating xai prompt format
                    # ipdb.set_trace()
                    diff_score = diff_task_score(pred_model, pred_tokenizer, task_instruction, [input_zip[idx]], [answer[idx]], exp_reply, counter_exp_reply, args)
                    diff_score_avg += diff_score
                    pbar.update(1)
                
                diff_score_avg /= float(len(input_zip))
                scores_list.append(diff_score_avg)

            # LLM optimizer
            # print(f"=== Score: {len(scores_list)}")
            xai_prompt = generate_global_xai_prompt(xai_prompts_list, scores_list, args)
            updated_xai_prompt = reponse_xai_model(xai_prompt, args, xai_local_model, xai_local_tokenizer)
            
            print(f"============ XAI prompt: {xai_prompts_list[-1]} || Score: {diff_score_avg}")
            save_prompt = xai_prompts_list[-1].replace("\n", "")
            xai_prompts_write.append({"Score": diff_score_avg, "XAI prompt": save_prompt})
            xai_prompts_list.append(updated_xai_prompt)

        except Exception as e:
            # Previously a bare `except: continue`, which silently turned a bad
            # API key or model id into an empty results file that looked like a
            # successful run. Surface the error and count it.
            error_count += 1
            print(f"============ Error at Step:{iter} ({type(e).__name__}): {e}")
            traceback.print_exc()
            if error_count >= max_errors:
                raise RuntimeError(
                    f"Aborting: {error_count} consecutive/total failures. Last error: {e}"
                ) from e
            continue

    # Save file
    cur_xai_iter = args.xai_iter * (args.round_xai_iter)
    xai_prompts_write.append({"Score": "Final", "XAI prompt": updated_xai_prompt})
    result_save_path = args.save_file
    result_file_name = f"global_{args.data}_{args.xai_model}_{args.pred_model}_iter-{cur_xai_iter}_sample-{args.ques_sample}.json"
    with open(os.path.join(result_save_path, result_file_name), "w") as f:
        json.dump(xai_prompts_write, f)
    print("============ Successful File Saved")

    print(f"============ API stats | {llm_api.stats_summary()}")
