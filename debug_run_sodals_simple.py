#!/usr/bin/env python3
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'AssetOpsBench-main' / 'src'))
sys.path.insert(0, str(root / 'scripts' / 'benchmark'))

os.environ['SODALS_MODEL_PATH'] = r'C:\Users\Burak\Desktop\sodals_core\models\model.gguf'

from llm.sodals import SodalsLLMBackend

print('llm.sodals file=', __import__('llm.sodals').__file__)
import candle_bridge
print('candle_bridge file=', candle_bridge.__file__)
from candle_bridge import CandleModelBridge
print('CandleModelBridge class file=', CandleModelBridge.__code__.co_filename)
print('model binary path=', CandleModelBridge(model_path=os.environ['SODALS_MODEL_PATH']).candle_binary)

backend = SodalsLLMBackend('sodals/qwen-custom')
print('backend bridge binary path=', backend._bridge.candle_binary)

prompt = 'What is 2+2?'
res = backend.generate(prompt)
print('result length=', len(res))
print('first 200 chars=', repr(res[:200]))
print('last 200 chars=', repr(res[-200:]))
