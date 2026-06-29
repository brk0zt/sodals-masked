#!/usr/bin/env python3
import subprocess
import json

model_path = r"C:\Users\Burak\Desktop\sodals_core\models\model.gguf"
tokenizer_path = r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json"
binary_path = r"C:\Users\Burak\Desktop\sodals_core\target\release\candle_inference_final.exe"

prompt = "What is 2+2?"

cmd = [
    binary_path,
    "--stdin",
    "256",
    model_path,
    tokenizer_path
]

print(f"Running command: {' '.join(cmd)}")
print(f"Input prompt: {prompt}")
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
print(f"STDERR length: {len(result.stderr)}")
print()

print("=" * 80)
print("STDOUT (first 500 chars):")
print("=" * 80)
print(repr(result.stdout[:500]))
print()

print("=" * 80)
print("STDOUT (last 500 chars):")
print("=" * 80)
print(repr(result.stdout[-500:]))
print()

print("=" * 80)
print("STDERR (first 500 chars):")
print("=" * 80)
print(repr(result.stderr[:500]))
