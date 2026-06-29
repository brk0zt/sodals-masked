#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Set up paths
root = Path(__file__).parent / "AssetOpsBench-main"
sys.path.insert(0, str(root / "src"))

from llm.sodals import SodalsLLMBackend

# Load first question from dataset
dataset_path = Path('c:/Users/Burak/Desktop/sodals_core/FailureSensorIQ/failuresensoriq_standard/all.jsonl')
with open(dataset_path, 'r') as f:
    question = json.loads(f.readline())

# Format prompt same way as evaluate_mcqa.py
options_text = '\n'.join([f'{opt_id}) {opt}' for opt_id, opt in zip(question['option_ids'], question['options'])])
prompt = f'''<|im_start|>system
You are an expert industrial diagnostic AI. Your task is to solve multiple-choice questions. 
First, analyze the problem step-by-step inside <think></think> tags. 
Then, provide your final choice outside the tags using exactly this format: Answer: X (where X is {', '.join(question['option_ids'])}).<|im_end|>
<|im_start|>user
Question: {question['question']}

Options:
{options_text}
<|im_end|>
<|im_start|>assistant
<think>
'''

print(f'Prompt length: {len(prompt)}')
print(f'Prompt (first 500 chars):')
print(repr(prompt[:500]))
print()
print(f'Prompt (last 500 chars):')
print(repr(prompt[-500:]))
