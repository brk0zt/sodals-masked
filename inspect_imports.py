#!/usr/bin/env python3
import sys
from pathlib import Path
root = Path(__file__).parent / "AssetOpsBench-main"
sys.path.insert(0, str(root / "src"))
print('sys.path[0]=', sys.path[0])
import llm
print('llm module file=', llm.__file__)
from llm.sodals import SodalsLLMBackend
print('SodalsLLMBackend module=', SodalsLLMBackend.__module__)
print('SodalsLLMBackend file=', SodalsLLMBackend.__code__.co_filename)
from candle_bridge import CandleModelBridge
print('CandleModelBridge module=', CandleModelBridge.__module__)
print('CandleModelBridge file=', CandleModelBridge.__code__.co_filename)
print('Bridge binary path=', CandleModelBridge(model_path=r'C:\Users\Burak\Desktop\sodals_core\models\model.gguf').candle_binary)
