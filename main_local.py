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
from datasets import load_dataset, load_metric, Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from model.predictor import load_model, generate_api_predictor_output, diff_task_score_ecqa, diff_task_score_trivaqa
from model.predictor import generate_predictor_output_ecqa, generate_predictor_output_trivaqa
from model.explainer import reponse_xai_model, generate_counterfact_prompt, generate_local_xai_prompt, generate_exp_prompt


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

def preprocess_xcopa():
    train_dict = collections.defaultdict(list)

    all_na_data = load_dataset("xcopa", "it")
    question, answer = [], []
    hg_data = all_na_data["test"]
    question_text = hg_data["premise"]
    question_purp = hg_data["question"]
    answer = hg_data["label"]
    op1 = hg_data["choice1"]
    op2 = hg_data["choice2"]
    option = list(zip(op1, op2))
    choice = [ f"[choice]{opt[0]}@ [choice]{opt[1]}@" for opt in option]

    for idx, ques_txt in enumerate(question_text):
        question = f"###Question: What is the {question_purp[idx]} of the Promise writen in Italian?\n### Premise: {ques_txt}\n### Choices: {choice[idx]}"
        train_dict['question'].append(question)
        train_dict['answer'] = [opt[answer[idx]] for opt in option]
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
    parser.add_argument('--temp_exp', type=int, default=0.01)
    parser.add_argument('--xai_iter', type=int, default=20)
    parser.add_argument('--ques_idx_start', type=int, default=40)
    parser.add_argument('--ques_idx_end', type=int, default=40)
    parser.add_argument('--save_cf_file_path', type=str, default=None)
    parser.add_argument('--save_file_path', type=str, default="./results")
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
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory)
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
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory)
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
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory)
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
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa

        # Reverse data name
        args.data = "social"

    elif args.data == "xcopa":
        train_dict = preprocess_xcopa()
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
        pred_model, pred_tokenizer = load_model(args.pred_model, max_memory)
        if pred_tokenizer == None:
            generate_ans_function = generate_api_predictor_output
        else:
            generate_ans_function = generate_predictor_output_ecqa

        # LLM-OPT function
        diff_task_score = diff_task_score_ecqa


    for idx in range(start_idx, args.ques_idx_end):
        # Select init score for LLM optimization
        fed_score_llm = 0.0
        fed_score_org = 0.0

        # Select data for LLM optimization
        if args.data in ["ecqa", "copa", "social", "xcopa"]:
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
        if args.data in ["ecqa", "copa", "social", "xcopa"]:
            if output_ans[0] in answer:
                target = f"============ Corrct --> Q:{question} || GT-A:{answer[0]} || LLM-A:{answer[0]}"
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
        exp_reply = exp_reply.split(":\n\n")[-1]
        exp_reply = exp_reply.split("\n\n")
        xai_list.extend(exp_reply)
        # print(f"============ Init Exp: {exp_reply[0]}")

        with tqdm(total=args.xai_iter) as pbar:
            for iter in range(args.xai_iter):
                print(f"============ Step:{iter} LLM Optimizing")

                # Generate counterfactual explanation
                counter_xai_prompt = generate_counterfact_prompt(exp_reply, args)
                counter_exp_reply = reponse_xai_model(counter_xai_prompt, args)
                counter_exp_reply = counter_exp_reply.split(":\n\n")[-1]
                counter_exp_reply = counter_exp_reply.split("\n\n")
                cf_write.append(counter_exp_reply)
                # print(f"============ Counterfact Exp: {counter_exp_reply[0]}")

                # Score difference for updating xai prompt format
                diff_score = diff_task_score(pred_model, pred_tokenizer, task_instruction, input_zip, answer, exp_reply, counter_exp_reply, args)
                scores_list.append(diff_score)

                # Save explanation
                save_explanation = xai_list[-1]
                xai_prompts_write.append({"Score": diff_score, "XAI prompt": save_explanation})
                print(f"=== Score: {diff_score} || Explanation: {save_explanation}")

                if iter%5 == 0 and sum(scores_list) != 0:
                    print("============ Early Stop Optimization")
                    break
                elif iter%5 == 0 and "X" in output_ans:
                    print("============ Bad Answer")
                    break

                # LLM optimizer
                xai_prompt = generate_local_xai_prompt(xai_list, scores_list, question, output_ans)
                exp_reply = reponse_xai_model(xai_prompt, args)
                xai_list.append(exp_reply)
                pbar.update(1)

                if iter%5 == 0 and ("apologize" in exp_reply) or ("Unfortunately" in exp_reply):
                    print(f"============ {exp_reply}")
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