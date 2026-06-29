#!/usr/bin/env python3
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path

root = Path(__file__).parent
src_path = root / 'AssetOpsBench-main' / 'src'
bench_path = root / 'scripts' / 'benchmark'
print('root=', root)
print('src_path=', src_path)
print('bench_path=', bench_path)
print('exists src?', src_path.exists())
print('exists bench?', bench_path.exists())

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(bench_path))
print('sys.path[0:5]=', sys.path[:5])

from llm.sodals import SodalsLLMBackend
import candle_bridge
print('llm.sodals file=', __import__('llm.sodals').__file__)
print('candle_bridge file=', candle_bridge.__file__)
print('module search path contains root and bench?')

backend = SodalsLLMBackend('sodals/qwen-custom')
print('backend bridge binary=', backend._bridge.candle_binary)

prompt = '<|im_start|>system\nYou are an expert industrial diagnostic AI...<|im_end|>\n<|im_start|>user\nWhat is 2+2?\n<|im_end|>\n<|im_start|>assistant\n<think>\n'
print('backend generate start')
res = backend.generate(prompt)

print('backend generate done len=', len(res))

with open('debug_output.txt', 'w', encoding='utf-8') as f:

    f.write(res)

print('Output successfully written to debug_output.txt')
