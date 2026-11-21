import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import sft.smart_app_control_compat
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import PeftModel

MODEL_DIR = "C:/Users/Burak/Desktop/sodals_core/models/Qwen 3.5 9B"
ADAPTER_DIR = "C:/Users/Burak/Desktop/sodals_core/sft/qwen_lora_adapter_grpo_math_train_100"

print("1. Loading Tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("2. Loading 4-bit Base Model...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_cfg,
    device_map={"": 0},
    trust_remote_code=True,
    attn_implementation="sdpa",
)

class CPUEmbeddingWrapper(torch.nn.Module):
    def __init__(self, cpu_embedding: torch.nn.Embedding):
        super().__init__()
        self.embedding = cpu_embedding
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        orig_device = input_ids.device
        emb = self.embedding(input_ids.to("cpu"))
        return emb.to(orig_device)

print("3. Offloading embed_tokens and quantizing lm_head...")
embed_cpu = base_model.model.embed_tokens.to("cpu")
base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu)
type(base_model).device = property(lambda self: torch.device("cuda:0"))

orig_lm_weight = base_model.lm_head.weight.data
l4_head = bnb.nn.Linear4bit(4096, 248320, bias=False, compute_dtype=torch.float32, quant_type="nf4")
q_w, qstate = bnb.functional.quantize_4bit(orig_lm_weight, quant_type="nf4")
l4_head.weight = bnb.nn.Params4bit(q_w, requires_grad=False, quant_state=qstate)
base_model.lm_head = l4_head
del orig_lm_weight
torch.cuda.empty_cache()

print("4. Loading LoRA Adapter from checkpoint...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

prompt = "<|im_start|>user\nSolve for x: 2*x + 5 = 15.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")

print("5. Generating 32 tokens...")
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=32, do_sample=False)

gen_text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
print("--- GENERATED OUTPUT ---")
print(repr(gen_text))
print("------------------------")
assert len(gen_text.strip()) > 0
assert not gen_text.startswith("00000")
print("SUCCESS: Model generation is coherent and correct!")
