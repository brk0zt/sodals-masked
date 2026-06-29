#!/usr/bin/env python3
import subprocess
import json
import sys
from pathlib import Path

model_path = r"C:\Users\Burak\Desktop\sodals_core\models\model.gguf"
tokenizer_path = r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json"
binary_path = r"C:\Users\Burak\Desktop\sodals_core\target\release\candle_inference_final.exe"

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

cmd = [
    binary_path,
    "--stdin",
    "512",
    model_path,
    tokenizer_path
]

print(f"Sending MCQ prompt to binary...")
print(f"Prompt length: {len(prompt)}")
print()

result = subprocess.run(
    cmd,
    input=prompt,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    timeout=300
)

print(f"Return code: {result.returncode}")
print(f"STDOUT length: {len(result.stdout)}")
print()

try:
    data = json.loads(result.stdout)
    raw_output = data.get("raw_output", "")
    print(f"Raw output length: {len(raw_output)}")
    print(f"Raw output (first 300 chars):")
    print(repr(raw_output[:300]))
    print()
    print(f"Raw output (last 300 chars):")
    print(repr(raw_output[-300:]))
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
    print(f"STDOUT (first 500 chars): {repr(result.stdout[:500])}")
