from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ipdb

def load_model(model_name, max_memory):

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
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = '[PAD]'
        model.eval()
    
    elif model_name == "claude":
        model = "claude"
        tokenizer = None

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

    try:
        index = ans.find("]")
        end_index = ans.find("@")
        if index!=-1:
            if ans[index+1] == " ":
                ans = ans[index+2:end_index]
            else:
                ans = ans[index+1:end_index]
        else:
            if ans_gt[0].strip() in ans:
                ans = ans_gt[0].strip()
            else:
                ans = "X"
    except:
        ans = "X"
    if len(ans) == 0 or ans is None:
        ans = "X"
    ans_llm.append(ans)
    
    return ans_llm

def generate_predictor_output_ecqa(model, tokenizer, task_instruction, input_zip, ans_gt, args):

    temperature_cot = 0.7
    ans_llm = []

    if args.pred_model == "phi":
        instruction = f"Below is an instruction that describes a task. \
                    Write a response that appropriately completes the request of input."
        final_prompt = [ f"{instruction}\n\n### Instruction: {task_instruction}\n\n \
                            ### Input: {ques}\n\n \
                            ### Response:" for ques in input_zip ]
        
        model_inputs_all = tokenizer(final_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            try:
                end_index = ans.find("@")
                index = ans.find("]")
                if index!=-1:
                    if ans[index+1] == " ":
                        ans = ans[index+2:end_index]
                    else:
                        ans = ans[index+1:end_index]
                elif "The correct answer is:\n\n" in ans:
                    ans = ans.split("The correct answer is:\n\n")[-1]
                    ans = ans.replace("*", "").strip()
                else:
                    if ans_gt[i].strip() in ans:
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
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            try:
                index = ans.find("]")
                end_index = ans.find("@")
                if end_index != -1 and index != -1:
                    if ans[index+1] == " ":
                        ans = ans[index+2:end_index]
                    else:
                        ans = ans[index+1:end_index]
                else:
                    if ans_gt[i].strip() in ans:
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

    if args.pred_model == "phi":
        model_inputs_all = tokenizer(final_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=20)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            ans_llm.append(ans)

    elif args.pred_model == "vicuna":
        model_inputs_all = tokenizer(final_prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=20)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
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
            if index!=-1:
                if ans[index+1] == " ":
                    ans = ans[index+2:end_index]
                else:
                    ans = ans[index+1:end_index]
            else:
                if ans_gt[0].strip() in ans:
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
            if index!=-1:
                if ans[index+1] == " ":
                    ans = ans[index+2:end_index]
                else:
                    ans = ans[index+1:end_index]
            else:
                if ans_gt[0].strip() in ans:
                    ans = ans_gt[0].strip()
                else:
                    ans = "X"
        except:
            ans = "X"
        if len(ans) == 0 or ans is None:
            ans = "X"
        ans_short.append(ans)

    elif args.pred_model == "phi":
        model_inputs_all = tokenizer(input_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            try:
                end_index = ans.find("@")
                index = ans.find("]")

                if index!=-1:
                    ans = ans[index+1:end_index]
                elif "The correct answer is:\n\n" in ans:
                    ans = ans.split("The correct answer is:\n\n")[-1]
                    ans = ans.replace("*", "").strip()
                else:
                    if ans_gt[i].strip() in ans:
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
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=256)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            try:
                end_index = ans.find("@")
                index = ans.find("]")
                if index!=-1:
                    if ans[index+1] == " ":
                        ans = ans[index+2:end_index]
                    else:
                        ans = ans[index+1:end_index]
                elif "The correct answer is:\n\n" in ans:
                    ans = ans.split("The correct answer is:\n\n")[-1]
                    ans = ans.replace("*", "").strip()
                else:
                    if ans_gt[i].strip() in ans:
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

    instruction = f"Below is an instruction that describes a task. \
                    Write a response that appropriately completes the request of input."
    ture_final_prompt = [f"{instruction}\n\n### Instruction: {task_instruction}\n\n \
                        ### Input: {ques}\n\n \
                        ### Response: Let's think step by step." for _, ques in true_exp_pair ]
    count_final_prompt = [f"{instruction}\n\n### Instruction: {task_instruction}\n\n \
                        ### Hint: {exp}\n\n \
                        ### Input: {ques}\n\n \
                        ### Response: Let's think step by step." for exp, ques in count_exp_pair ]
    ture_score = _ecqa_score(model, tokenizer, ture_final_prompt, answer, args)
    count_score = _ecqa_score(model, tokenizer, count_final_prompt, answer, args)
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
    if args.pred_model == "phi":
        model_inputs_all = tokenizer(input_prompt, padding="max_length", max_length=1000, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=10)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
            ans_short.append(ans)

    elif args.pred_model == "vicuna":
        model_inputs_all = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**model_inputs_all, temperature=temperature_cot, max_new_tokens=10)
        ans_tkn = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(ans_tkn)):
            ans = ans_tkn[i].split("### Response:")[-1]
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