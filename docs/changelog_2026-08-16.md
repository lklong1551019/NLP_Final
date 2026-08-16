# FaithLM Codebase Changelog
**Date**: 2026-08-16

This document tracks all modifications, bug fixes, and feature additions made to the original FaithLM codebase to adapt it for our experiments.

## 1. Local & Global Pipeline Accuracy Parsing Bug Fix
**Files modified**: `main_local.py`, `main_global.py`
* **What we changed**: Changed the answer matching condition from `if output_ans[0] in answer:` to `if answer[0].strip() in output_ans[0]:`.
* **Reason**: Modern chatty LLMs like Qwen3.5-4B generate verbose Chain-of-Thought reasoning (especially when prompted with "Let's think step by step"). The original code assumed the model output would be a perfectly short string. By checking if the massive output block was exactly equal to the ground truth string, it always evaluated to `False`, falsely labeling correct answers as `============ Wrong` and ruining the accuracy metric. Our fix checks if the ground truth answer is *inside* the model's generated text block.

## 2. Iterator Exhaustion Bug Fix
**Files modified**: `main_local.py`
* **What we changed**: Added `exp_reply = exp_reply.split("\n\n")` inside the optimization loop so `exp_reply` correctly remains a list instead of a string.
* **Reason**: The original script failed to parse the explainer's string response into a list at the end of each iteration. During the *next* iteration, the `zip(exp_reply, question)` function would iterate over the individual characters of the string instead of the whole explanation. If the string happened to be empty, `zip` yielded nothing, passing an empty prompt to the Tokenizer and causing a fatal `IndexError: list index out of range`.

## 3. Model Quantization Support for 8GB VRAM
**Files modified**: `model/predictor.py`
* **What we changed**: Integrated `BitsAndBytesConfig` (NF4 4-bit quantization) into the `load_model` function for Hugging Face `AutoModelForCausalLM` loading.
* **Reason**: The original paper likely utilized A100 clusters to load the models in float16/float32. Running Qwen3.5-4B locally on an 8GB RTX 3050Ti would immediately trigger an Out-Of-Memory (OOM) error. Quantization allows the model to comfortably fit inside the available VRAM.

## 4. DeepSeek API Integration
**Files modified**: `model/explainer.py`
* **What we changed**: Refactored the explanation generation logic to utilize the DeepSeek API (using the OpenAI SDK wrapper) as the Explainer LLM.
* **Reason**: To fulfill the experiment variant requiring DeepSeek as the teacher/explainer model instead of the original models used in the paper.

## 5. Dataset Preprocessing Fixes
**Files modified**: `main_local.py`, `main_global.py`
* **What we changed**: Corrected the preprocessing loops for XCOPA-Vi and COPA-En to `.append()` items properly into `train_dict`.
* **Reason**: The initial setup lacked the proper list appending logic to correctly parse the Hugging Face datasets into the dictionary format expected by the rest of the framework.

## 6. Automation Scripts & Parameter Tuning
**Files modified**: `scripts/run_all_experiments.sh`
* **What we changed**: Created unified shell scripts to sequentially execute the Local and Global pipelines across our 4 variants. Decreased the benchmark scale (`END_IDX=200`, `XAI_ITER=15`).
* **Reason**: The original parameter of 500 questions and 20 iterations would take roughly 28 days to complete on a single GPU. Tuning these parameters down to 200/15 brings the runtime down to a much more manageable 50 hours per variant.

## 7. Dependency Resolution
**Files modified**: `requirements.txt`
* **What we changed**: Added missing dependencies such as `anthropic` and `ipdb`.
* **Reason**: The original environment lacked a few crucial libraries required to run `model/predictor.py` and `model/explainer.py`.
