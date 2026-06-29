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

print(f'Question: {question["question"][:100]}...')
print(f'Options: {question["options"]}')
print()

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

print('Testing model with first MCQ...')
backend = SodalsLLMBackend('sodals/qwen-custom')
try:
    result = backend.generate(prompt)
    print(f'Output length: {len(result)}')
    print(f'Output (first 300 chars): {repr(result[:300])}')
    print()
    print(f'Output (last 300 chars): {repr(result[-300:])}')
    print()
    print(f'Full output:\n{result}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
