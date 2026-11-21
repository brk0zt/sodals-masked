import torch
import torch.nn as nn
from peft.tuners.lora.layer import Linear

base = nn.Linear(32, 32)
lora = Linear(base, adapter_name="expert_general_math", r=4, lora_alpha=4)
lora.update_layer(
    adapter_name="expert_abstract_algebra",
    r=4,
    lora_alpha=4,
    lora_dropout=0.0,
    init_lora_weights=True,
    use_rslora=False,
    use_dora=False
)

print("Before set_adapter:", lora._active_adapter)
lora.set_adapter(["expert_general_math", "expert_abstract_algebra"])
print("After set_adapter:", lora._active_adapter)

x = torch.randn(2, 8, 32)
out = lora(x)
print("Forward success! Output shape:", out.shape)
