#!/usr/bin/env python3
import os
import sys
from pathlib import Path

root = Path(__file__).parent / 'AssetOpsBench-main'
sys.path.insert(0, str(root / 'src'))
print('sys.path[0]=', sys.path[0])
print('PYTHONPATH=', os.environ.get('PYTHONPATH'))

from llm.sodals import SodalsLLMBackend
import candle_bridge
from candle_bridge import CandleModelBridge
print('llm.sodals file=', __import__('llm.sodals').__file__)
print('candle_bridge file=', candle_bridge.__file__)
backend = SodalsLLMBackend('sodals/qwen-custom')
print('backend bridge binary=', backend._bridge.candle_binary)
print('backend bridge class from module=', backend._bridge.__class__)
print('using raw_output attr?')
res = backend._bridge.run_single_inference({'task_id':'debug_task','input_prompt':'What is 2+2?'})
print('bridge result type', type(res))
print('bridge result keys', res.keys())
print('bridge raw_output repr', repr(res.get('raw_output')))
print('bridge parsed_answer', repr(res.get('parsed_answer')))
print('backend generate result repr', repr(backend.generate('What is 2+2?')))
