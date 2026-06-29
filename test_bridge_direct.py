#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Set up logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set up paths
root = Path(__file__).parent
sys.path.insert(0, str(root / "scripts/benchmark"))

from candle_bridge import CandleModelBridge

model_path = r"C:\Users\Burak\Desktop\sodals_core\models\model.gguf"

bridge = CandleModelBridge(model_path=model_path, max_new_tokens=256)

example = {
    "task_id": "test_task",
    "input_prompt": "What is 2+2?"
}

print(f"Testing CandleModelBridge directly...")
result = bridge.run_single_inference(example)

# Write results to a UTF-8 file to avoid encoding issues
with open('bridge_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Result keys: {result.keys()}\n")
    f.write(f"Raw output length: {len(result.get('raw_output', ''))}\n")
    f.write(f"Raw output (first 200 chars):\n{repr(result['raw_output'][:200])}\n\n")
    f.write(f"Token count: {result.get('token_count', 0)}\n")
    f.write(f"Parsed answer: {result.get('parsed_answer', '')}\n")
    f.write(f"\nFull result dict:\n{result}\n")

print(f"Results written to bridge_results.txt")
print(f"Result keys: {result.keys()}")
print(f"Raw output length: {len(result.get('raw_output', ''))}")
print(f"Token count: {result.get('token_count', 0)}")
print(f"Parsed answer: {result.get('parsed_answer', '')}")
