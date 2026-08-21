# FaithLM Codebase Changelog
**Date**: 2026-08-19

This document tracks all modifications, bug fixes, and feature additions made to the FaithLM codebase to adapt it for our experiments.

## 1. Natural Vietnamese Prompt Instructions
**Files modified**: `main_local.py`, `main_global.py`, `model/predictor.py`, `model/explainer.py`
* **What we changed**: Refactored all Vietnamese instruction strings (Task Instruction, Explainer Instruction, Global/Local XAI Meta and Act instructions, Counterfactual generator) from a word-by-word literal translation to a fluent, natural-sounding Vietnamese structure.
* **Reason**: To improve comprehension by the LLM (Predictor/Explainer) and yield more natural, coherent explanations in the Vietnamese language, without altering the core functional tags (e.g., `<EXP>`, `<INS>`).

## 2. Prompt Documentation & Translation Mapping
**Files modified**: `docs/prompts_translation.md` (New)
* **What we changed**: Created a comprehensive Markdown document that maps out the English and Vietnamese translations for every prompt used to interact with the models in the framework.
* **Reason**: To maintain clarity and provide a quick reference for all prompt templates being utilized across the English and Vietnamese datasets.

## 3. Codebase Cleanup and Alignment with Main
**Files modified**: `main_local.py`, `main_global.py`, `scripts/run_experiment.sh`
* **What we changed**: Reverted experimental debugging logic (such as fallback handling for empty completions and custom step comments) from `main_global.py` and `main_local.py` to strictly match the upstream `main` branch. Also reverted `scripts/run_experiment.sh` to remove hardcoded parameter passing (like `MAX_TOKENS`), relying on environment variables instead.
* **Reason**: To keep the working branch clean and prevent merge conflicts by ensuring only the prompt instruction changes were introduced, while preserving the centralized logic implemented in `main`.
