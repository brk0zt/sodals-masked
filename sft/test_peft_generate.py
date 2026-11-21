import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# Mini dummy model
class MiniLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 16)
        self.linear = nn.Linear(16, 16)
        self.head = nn.Linear(16, 50, bias=False)
    def forward(self, input_ids, **kwargs):
        x = self.linear(self.embed(input_ids))
        logits = self.head(x)
        class Out:
            pass
        res = Out()
        res.logits = logits
        return res

mini = MiniLM()
config = LoraConfig(r=4, lora_alpha=4, target_modules=["linear"])
peft_m = get_peft_model(mini, config, adapter_name="ad1")
peft_m.add_adapter("ad2", config)

for m in peft_m.modules():
    if hasattr(m, "_active_adapter"):
        m._active_adapter = ["ad1", "ad2"]

inp = torch.tensor([[1, 2, 3]])
out = peft_m(inp)
print("Forward with multi-adapter success! Logits shape:", out.logits.shape)
