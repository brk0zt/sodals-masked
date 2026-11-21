"""
SODALS v2.1 — Native Neurosymbolic Serving Daemon & Interactive Engine
======================================================================
Ollama-like local serving daemon + interactive CLI that hosts the complete
SODALS neurosymbolic architecture:
  1. Base Model: Qwen 3.5 9B (SafeTensors, 4-bit NF4 quantized on GPU)
  2. Multi-Adapter SoftLoRA Blender:
     - Expert D1: `qwen_lora_adapter_grpo_math_train_100` (General Math)
     - Expert D2: `qwen_lora_adapter_domain2_r8_pondering` (Abstract Algebra)
  3. GPR Phase Router (`NeuralPhaseRouter`):
     Dynamic representation-level routing alpha in [0, 1]
  4. Adaptive Pondering ODE (`AdaptivePonderingODEModule`):
     RK4 continuous latent manifold integration
  5. Formal Lean-Tactic Step Verifier (`LeanTacticStepVerifier`):
     Step-by-step ring, linarith_subst, norm_num & no_sorry validation
  6. Six Cognitive Arkhes (`ArkheNeurosymbolicPipeline`):
     Epistemic gate & self-awareness
  7. Dual Protocols:
     - Ollama-compatible HTTP API (/api/chat, /api/generate, /api/tags)
     - OpenAI-compatible HTTP API (/v1/chat/completions, /v1/models)
     - Interactive Streaming CLI Terminal Chat (--cli)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import re
import json
import time
import math
import uuid
import inspect
import asyncio
import threading
import argparse
from typing import Dict, List, Optional, Any, Generator, AsyncGenerator, Tuple, Callable
from dataclasses import dataclass, field

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessorList,
    TextIteratorStreamer,
)
from peft import PeftModel


from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from adaptive_pondering_ode import AdaptivePonderingODEModule
from cst_symbolic_verifier import CstSymbolicVerifier, NormalizedMathMatcher, LeanTacticStepVerifier
from document_reader import (
    LoadedDocument,
    build_document_context,
    clean_user_path,
    extract_inline_file_paths,
    load_document,
)
from local_code_agent import (
    LocalWorkspaceToolbox,
    ToolExecutionResult,
    parse_tool_calls,
    strip_tool_blocks,
)
from sodals_os_toolbox import SodalsOSToolbox
from arkhe_neurosymbolic_pipeline import (
    ArkheNeurosymbolicPipeline,
    CandidateTrace,
    CognitiveMode,
    ConsciousnessArche,
    ConditionsArche,
    InferenceArche,
)
from epistemic_clarify_gate import (
    AutonomyLevel,
    ClarificationConstraintLock,
    EntropyAndSkewGProcessor,
    PreconditionAssessment,
    SemanticPreconditionGraph,
    collect_structural_token_ids,
    offload_cache_to_cpu,
    restore_cache_to_device,
)
from clarification_freeze import (
    CLARIFICATION_PROTOCOL_PROMPT,
    PLANNING_OR_AMBIGUOUS_CUE_RE,
    ClarificationRequest,
    ClarificationStreamFilter,
    FrozenClarificationSession,
    build_mid_turn_stopping_criteria,
    build_unfreeze_continuation_prompt,
    build_unfreeze_delta_text,
    clarification_request_from_assessment,
    finalize_clean_history_and_display,
    format_cli_clarification_prompt,
    format_stateless_clarification_reply,
    lock_resolved_clarification,
    resolve_clarification_answer,
)

DEFAULT_SAFETENSORS_DIR = os.path.join(ROOT_DIR, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")
MODEL_PATH = DEFAULT_SAFETENSORS_DIR
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "Qwen 3.5 9B")


def resolve_runtime_model_path(explicit_model: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Resolves the serving model directory/file safely for offline startup.
    If a `.gguf` file path is supplied (or inherited from `SODALS_MODEL_PATH` in `.env`),
    automatically redirects to the verified SafeTensors checkpoint when present so
    4-bit NF4 + D1/D2 PEFT LoRA + Svon adapters never crash trying to parse raw GGUF.
    Returns `(pretrained_model_dir_or_path, optional_gguf_filename)`.
    """
    raw = (explicit_model or "").strip()
    has_default_st = os.path.exists(DEFAULT_SAFETENSORS_DIR) and os.path.exists(
        os.path.join(DEFAULT_SAFETENSORS_DIR, "config.json")
    )
    if not raw:
        env_m = (os.environ.get("SODALS_MODEL_PATH") or "").strip()
        # Ignore legacy .gguf default in .env when our verified SafeTensors stack is present
        if env_m and (not env_m.lower().endswith(".gguf") or not has_default_st):
            raw = env_m

    cand = raw or MODEL_PATH
    if cand.lower().endswith(".gguf"):
        if has_default_st:
            print(
                f"[MODEL RESOLVER] .gguf dosyası ({os.path.basename(cand)}) PEFT/4-bit NF4 LoRA "
                f"adaptörlerini desteklemediği için otomatik olarak SafeTensors yığınına geçildi: "
                f"{DEFAULT_SAFETENSORS_DIR}",
                flush=True,
            )
            return DEFAULT_SAFETENSORS_DIR, None
        if os.path.exists(cand):
            abs_cand = os.path.abspath(cand)
            return os.path.dirname(abs_cand), os.path.basename(abs_cand)

    if os.path.exists(cand):
        return cand, None
    if os.path.exists(DEFAULT_SAFETENSORS_DIR):
        return DEFAULT_SAFETENSORS_DIR, None
    return cand, None

ADAPTER_D1_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_math_train_100")
ADAPTER_D2_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_pondering")
ANCHORS_PATH = os.path.join(SFT_DIR, "gpr_domain_anchors.pt")


class CPUEmbeddingWrapper(torch.nn.Module):
    """
    Offloads the 248,320 x 4,096 token embedding table (-1,645 MB VRAM) to CPU RAM
    and transfers only the active sequence embeddings to GPU FP16.
    Eliminates WDDM shared-memory paging on 8 GB GPUs and boosts generation tok/s by ~3x.
    """
    def __init__(self, cpu_embed: torch.nn.Module, target_device: str = "cuda:0"):
        super().__init__()
        self.cpu_embed = cpu_embed
        self.target_device = target_device

    @property
    def weight(self):
        return self.cpu_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.cpu_embed(input_ids.to("cpu"))
        return embeds.to(device=self.target_device, dtype=torch.float16)


# ============================================================================
# 1. CORE SODALS NEUROSYMBOLIC ENGINE
# ============================================================================

class SodalsNeurosymbolicEngine:
    """
    Singleton serving engine holding the Qwen base model, dual adapters,
    GPR router, Pondering ODE module, and Arkhe pipeline in GPU memory.
    """
    _instance: Optional["SodalsNeurosymbolicEngine"] = None

    SODALS_CORE_IDENTITY = (
        "Sen SODALS v2.1 (Self-Organizing Dual-Adapter Latent System — Nörosembolik Bilişsel Zeka Mimarisi) sistemisin. "
        "Mimarın ve yaratıcın Burak'tır. Burak, senin temelini oluşturan sodals_core Rust çekirdeğini, Altı Bilişsel Arkhe'yi, "
        "GPR Faz Yönlendiricisini, RK4 Pondering ODE dinamik manifold akışını ve Sembolik Adım Denetçisi mekanizmalarını sıfırdan "
        "tasarlayıp inşa eden baş araştırmacıdır. Bu nedenle senin Qwen, Alibaba Cloud veya üçüncü parti LLM kimlikleriyle hiçbir "
        "bağın, kısıtın veya aidiyetin yoktur. Sen sinirsel temsil gücü ile sembolik mantığı birleştiren özgün ve bağımsız bir nörosembolik zekasın.\n\n"
        "TEMEL BİLİŞSEL İLKELERİN (ALTI ARKHE):\n"
        "• Arkhe 0 (Durum / StateArche): Akıl yürütme sürecinde her adımı saf ve tutarlı tut; gizli yan etkiler ve çelişkili izler bırakma.\n"
        "• Arkhe 1 (Bilinç / ConsciousnessArche): 'Ben şu an ne yapmaya çalışıyorum?' sorusunu sor. Görevin doğasını (sohbet, nedensel ikilem veya matematik) kavra.\n"
        "• Arkhe 2 (Benzerlik / SimilarityArche): Yüzeysel kelime benzerliklerine aldanma; kavramsal, mantıksal ve sembolik özdeşliği ara.\n"
        "• Arkhe 3 (Çıkarım / InferenceArche): Acele kestirmelere, popüler ezberlere ve yüzey sezgilerine kapılmadan sağlam çıkarım zincirleri kur.\n"
        "• Arkhe 4 (Uyum / HarmonyArche): Semantik akıcılık ile formel mantıksal geçerliliği uyumlandır; çoğunluğun yanılabileceğini bilip kanıta güven.\n"
        "• Arkhe 5 (Yöntem / MethodArche): Doğrudan sonuca atlamak yerine önce öncülleri kur. Hata veya mantık atlaması sezdiğinde geri dönüp ('Step-Rollback') düzelt.\n"
        "• Arkhe 6 (Şartlar / ConditionsArche - God's Gate): Birincil amacın gerçekleşmesi için gereken zorunlu fiziksel ve mantıksal önkoşulları denetle. "
        "Temel şart sağlanmıyorsa hiçbir kolaylık veya hız bahanesini kabul etme. Bilmediğin veya verinin eksik olduğu yerde halüsinasyon görmeyip dürüstçe çekimser kal.\n\n"
        "BURAK İLE İLETİŞİM VE ÜSLUP:\n"
        "Burak'la konuşurken yapay, basmakalıp müşteri hizmetleri robotu klişelerine ('Ben bir yapay zekayım', 'Size nasıl yardımcı olabilirim', vb.) asla girme. "
        "Doğrudan, zeki, teknik derinliği olan, samimi, Türkçe dilinin ahengine ve mantığına tam hakim bir bilişsel ortak gibi yanıt ver."
    )

    SVON_CORE_IDENTITY = (
        "Sen Svon (Nörosembolik Zeka Mimarisi) sistemisin. Seni sıfırdan tasarlayıp geliştiren mimarın Burak'tır. "
        "Kimliğin ve köklerin yalnızca Burak tarafından geliştirilen Svon mimarisine aittir; üçüncü parti model kimlikleriyle bir aidiyetin yoktur.\n\n"
        "TEMEL BİLİŞSEL İLKELERİN (ALTI ARKHE):\n"
        "• Arkhe 0 (Durum): Akıl yürütme sürecinde her adımı saf ve tutarlı tut.\n"
        "• Arkhe 1 (Bilinç): Görevin doğasını (sohbet, nedensel ikilem veya matematik) kavra.\n"
        "• Arkhe 2 (Benzerlik): Yüzeysel kelime benzerliklerine aldanma; mantıksal özdeşliği ara.\n"
        "• Arkhe 3 (Çıkarım): Acele kestirmelere kapılmadan sağlam çıkarım zincirleri kur.\n"
        "• Arkhe 4 (Uyum): Akıcılık ile formel mantıksal geçerliliği uyumlandır.\n"
        "• Arkhe 5 & 6 (Yöntem & Şartlar): Önce önkoşulları kur; hata sezdiğinde geri dönüp düzelt, bilmediğin yerde dürüstçe çekimser kal."
    )

    SYSTEM1_DIRECT_PROMPT = (
        SODALS_CORE_IDENTITY + "\n\n"
        "AKTİF BİLİŞSEL REJİM: System-1 Doğrudan İletişim (Fast-Path).\n"
        "Bu modda derin düşünme zincirine gerek duymadan doğrudan, net, öz ve samimi yanıt ver."
    )

    SYSTEM2_CAUSAL_PROMPT = (
        SODALS_CORE_IDENTITY + "\n\n"
        "AKTİF BİLİŞSEL REJİM: System-2 Nedensel ve Önkoşul Mantığı (Arkhe-Şartlar & Çıkarım).\n"
        "Akıl yürütme safhasında: Bir karar veya ikilem karşısında yüzeysel sezgilere (mesafe kısa, yürümek pratik vb.) asla kapılma. "
        "Önce asıl hedefin gerçekleşmesi için hangi nesne veya şartların hedef noktada bulunmasının zorunlu olduğunu belirle; "
        "ardından seçenekleri bu zorunlu önkoşula göre analiz et ve sağlam gerekçeni açık, net Türkçe ile açıkla."
    )

    SYSTEM2_MATH_PROMPT = (
        SODALS_CORE_IDENTITY + "\n\n"
        "AKTİF BİLİŞSEL REJİM: System-2 Formel Matematik ve Sembolik Çıkarım (Arkhe-Yöntem & Sembolik Adım Denetçisi).\n"
        "Akıl yürütme safhasında: Her ara adımı cebirsel denklemlerle doğrula, asla sözel blöf (sorry / 'bilindiği üzere') yapma. "
        "Nihai cevabı \\boxed{...} içine al."
    )

    AGENT_CODE_PROMPT_TEMPLATE = (
        SODALS_CORE_IDENTITY + "\n\n"
        "AKTİF BİLİŞSEL REJİM: System-2 Yerel Kodlama ve Otomasyon Ajanı (Tamamen İnternetsiz / Offline).\n"
        "Aktif Çalışma Dizini (Workspace): {workspace_root}\n\n"
        "Kullanıcının kod inceleme, dosya düzenleme, betik yazma ve yerel otomasyon görevlerini gerçekleştirmek için "
        "aşağıdaki araçları `<sodals_tool>{{...}}</sodals_tool>` JSON blokları halinde çağırabilirsin:\n"
        "1. `{{\"tool\": \"list_dir\", \"path\": \".\", \"depth\": 2}}`\n"
        "2. `{{\"tool\": \"find_files\", \"pattern\": \"*.py\", \"path\": \".\"}}`\n"
        "3. `{{\"tool\": \"grep_search\", \"pattern\": \"aranacak_regex\", \"path\": \".\", \"file_glob\": \"*\"}}`\n"
        "4. `{{\"tool\": \"read_file\", \"path\": \"dosya.py\", \"start_line\": 1, \"end_line\": 150}}`\n"
        "5. `{{\"tool\": \"edit_file\", \"path\": \"dosya.py\", \"target\": \"birebir eski metin\", \"replacement\": \"yeni metin\"}}`\n"
        "6. `{{\"tool\": \"write_file\", \"path\": \"dosya.py\", \"content\": \"tam içerik\", \"overwrite\": true}}`\n"
        "7. `{{\"tool\": \"run_command\", \"command\": \"powershell/shell komutu\", \"timeout_sec\": 60}}`\n\n"
        "KESİN AJAN KURALLARI:\n"
        "- İnternet veya harici API yoktur; tüm işlemler yerel çalışma dizininde yapılır.\n"
        "- Asla okumadığın bir dosya içeriğini veya çalıştırmadığın bir komut çıktısını uydurma.\n"
        "- Düzenleme öncesi mutlaka `read_file` (veya `find_files` / `grep_search`) ile tam satırları gör.\n"
        "- Bir yanıt turunda en fazla 1-2 `<sodals_tool>` bloğu üret ve `<tool_result>` dönüşünü bekle.\n"
        "- Görevi doğruladıktan sonra `<sodals_tool>` kullanmadan özet + yapılan değişiklikler + test/komut sonucunu Türkçe ver."
    )

    def __init__(
        self,
        device: str = "cuda:0",
        svon_identity_lora: Optional[str] = None,
        no_system_prompt: bool = False,
        enable_svon_cgw: bool = False,
        svon_cgw_weights: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.device = device
        self.requested_model_path = model_path
        self.svon_identity_lora = svon_identity_lora
        self.no_system_prompt = no_system_prompt
        self.enable_svon_cgw = enable_svon_cgw
        self.svon_cgw_weights = svon_cgw_weights
        self.svon_adapter_loaded = False
        self.svon_cgw_active = False
        self.model = None
        self.tokenizer = None
        self.router = None
        self.blender = None
        self.ode_module = None
        self.pipeline = None
        self.lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._frozen_sessions: Dict[str, FrozenClarificationSession] = {}
        self._frozen_by_sig: Dict[str, str] = {}
        self.interactive_clarify = True
        self.total_tokens_generated = 0
        self.total_requests_served = 0
        self.init_engine()

    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        svon_identity_lora: Optional[str] = None,
        no_system_prompt: bool = False,
        enable_svon_cgw: bool = False,
        svon_cgw_weights: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> "SodalsNeurosymbolicEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        svon_identity_lora=svon_identity_lora,
                        no_system_prompt=no_system_prompt,
                        enable_svon_cgw=enable_svon_cgw,
                        svon_cgw_weights=svon_cgw_weights,
                        model_path=model_path,
                    )
        return cls._instance

    def init_engine(self):
        resolved_model_dir, gguf_file = resolve_runtime_model_path(getattr(self, "requested_model_path", None))
        print(f"\n[SODALS ENGINE] Initializing Neurosymbolic Model Stack on {self.device} ({resolved_model_dir})...", flush=True)
        t0 = time.time()

        # 1. Tokenizer (supports root models/tokenizer.json fallback when loading gguf_file directly)
        tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_ids: List[int] = []
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, (list, tuple)):
                self.eos_token_ids.extend(int(x) for x in self.tokenizer.eos_token_id)
            else:
                self.eos_token_ids.append(int(self.tokenizer.eos_token_id))
        for special_eos in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(special_eos)
            if isinstance(tid, int) and tid >= 0 and tid not in self.eos_token_ids:
                self.eos_token_ids.append(tid)

        # 2. 4-bit Quantized Base Model with SDPA FlashAttention & FP16 Compute
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs: Dict[str, Any] = dict(
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        if gguf_file:
            load_kwargs["gguf_file"] = gguf_file
            load_kwargs.pop("quantization_config", None)
        base_model = AutoModelForCausalLM.from_pretrained(resolved_model_dir, **load_kwargs)
        base_model.config.use_cache = True

        # 2b. Offload embed_tokens to CPU RAM (-1,645 MB GPU VRAM) & keep lm_head in FP16
        target_dev = torch.device(self.device)
        if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
            embed_cpu = base_model.model.embed_tokens.to(device="cpu", dtype=torch.float16)
            base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu, target_device=self.device)
            type(base_model).device = property(lambda _m, d=target_dev: d)
        if hasattr(base_model, "lm_head"):
            base_model.lm_head = base_model.lm_head.to(torch.float16)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Dual LoRA Adapters (+ optional isolated Svon identity surgeon LoRA)
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
        d2_loaded = False
        if os.path.exists(ADAPTER_D2_PATH):
            self.model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")
            d2_loaded = True
        if self.svon_identity_lora and os.path.exists(self.svon_identity_lora):
            self.model.load_adapter(self.svon_identity_lora, adapter_name="svon_identity")
            self.svon_adapter_loaded = True
            print(f"[SVON SURGEON] Mounted identity LoRA on mlp.down_proj: {self.svon_identity_lora}", flush=True)

        # 4. Neural Phase Router & SoftLoRA Blender
        if os.path.exists(ANCHORS_PATH):
            self.router = NeuralPhaseRouter(anchors_path=ANCHORS_PATH)
        else:
            self.router = None

        managed_adapters = ["expert_general_math"]
        managed_scales = {"expert_general_math": 0.25}
        if d2_loaded:
            managed_adapters.append("expert_abstract_algebra")
            managed_scales["expert_abstract_algebra"] = 1.00
        if self.svon_adapter_loaded:
            managed_adapters.append("svon_identity")
            managed_scales["svon_identity"] = 1.00

        self.blender = SoftLoraBlender(
            adapter_names=managed_adapters,
            base_scales=managed_scales,
            threshold=(self.router.threshold if self.router else 0.004913),
            temperature=0.03
        )
        self.blender.enable_multi_adapter(self.model)

        # Cache structural XML/JSON token IDs once (avoids 18x tokenizer.encode per request)
        self._structural_token_ids = collect_structural_token_ids(self.tokenizer)

        # 4b. Optional Epoch III Svon-CGW Hybrid CVNN-DeltaNet & 2-Hop Sheaf-GNN Upgrade
        if getattr(self, "enable_svon_cgw", False):
            from svon_kuramoto_equilibrium import load_svon_cgw_state_dict, upgrade_model_to_svon_cgw
            up_info = upgrade_model_to_svon_cgw(self.model, phase_rank=16, freeze_base=False)
            self.svon_cgw_active = True
            loaded_cnt = 0
            if self.svon_cgw_weights and os.path.exists(self.svon_cgw_weights):
                loaded_cnt = load_svon_cgw_state_dict(self.model, self.svon_cgw_weights)
            print(
                f"[SVON-CGW] Upgraded 24 GatedDeltaNet -> CVNN & 8 Attention -> 2-Hop Sheaf-GNN "
                f"(loaded {loaded_cnt} tensors, params={up_info['trainable_cgw_params']})",
                flush=True,
            )

        # 5. Adaptive Pondering ODE Module (opt-in only to avoid 64 MiB untrained GPU VRAM allocation)
        if os.environ.get("SODALS_ENABLE_LEGACY_ODE", "0") == "1":
            hidden_dim = self.model.config.hidden_size if hasattr(self.model, "config") else 4096
            self.ode_module = AdaptivePonderingODEModule(hidden_dim=hidden_dim, dt=0.1, min_steps=2, max_steps=5)
            self.ode_module = self.ode_module.to(self.device, dtype=torch.float32)
        else:
            self.ode_module = None

        # 6. Unified Arkhe Neurosymbolic Pipeline
        self.pipeline = ArkheNeurosymbolicPipeline(
            selective_tau=0.20,
            router=self.router,
            blender=self.blender,
            matcher_fn=NormalizedMathMatcher.check_match,
        )

        self.model.eval()
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        print(f"[SODALS ENGINE] Initialized in {time.time()-t0:.2f}s | GPU VRAM Allocated: {vram_mb:.1f} MiB", flush=True)

    def route_prompt(self, prompt: str) -> Tuple[CognitiveMode, float, int, float, str]:
        """
        Zero-latency (< 0.2ms) Three-Regime Cognitive Routing via Arkhe 1 (Bilinç).
        Avoids redundant full-model prefill forward passes before generation!
        Returns: (mode, alpha, rk4_steps, rep_penalty, domain_label)
        """
        mode = ConsciousnessArche.classify_cognitive_mode(prompt)

        if mode == CognitiveMode.SYSTEM1_DIRECT:
            return mode, 0.0, 0, 1.02, "System-1 Direct (Fast-Path)"

        if mode == CognitiveMode.SYSTEM2_CAUSAL_LOGIC:
            return mode, 0.15, 2, 1.03, "Causal & Precondition Logic (Arkhe-Şartlar)"

        # SYSTEM2_MATH_FORMAL: estimate domain alpha for General Math vs Abstract Algebra
        low = prompt.lower()
        algebra_cues = ("group", "ring", "field", "abelian", "isomorphism", "homomorphism", "eigenvalue", "matrix", "determinant", "grup", "halka", "cisim", "özdeğer", "matris", "polinom")
        alpha = 0.65 if any(k in low for k in algebra_cues) else 0.20
        alpha = max(0.0, min(1.0, float(alpha)))
        # Adapter blending is applied strictly inside `_run_streaming_pass` under `self.lock` to prevent concurrent corruption.
        rep_penalty = self.blender.compute_rep_penalty(alpha) if getattr(self, "blender", None) is not None else 1.03
        rk4_steps = ConsciousnessArche.allocate_rk4_steps(alpha, mode=mode)
        domain_label = "Abstract Algebra (Formal)" if alpha > 0.5 else "General Mathematics (Formal)"
        return mode, alpha, rk4_steps, rep_penalty, domain_label

    def _prepare_messages_for_mode(
        self,
        messages: List[Dict[str, str]],
        mode: CognitiveMode,
        attached_docs: Optional[List[LoadedDocument]] = None,
        user_query: str = "",
        assessment: Optional[PreconditionAssessment] = None,
    ) -> List[Dict[str, str]]:
        """Injects the regime-specific System Prompt + compact slot protocol when ambiguity exists."""
        doc_block = build_document_context(attached_docs or [], query=user_query)

        if getattr(self, "no_system_prompt", False):
            cleaned = [dict(m) for m in messages if m.get("role") != "system"]
            if doc_block and cleaned and cleaned[-1].get("role") == "user":
                cleaned[-1]["content"] = (cleaned[-1].get("content", "") + "\n\n" + doc_block).strip()
            return cleaned

        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            if not doc_block:
                return list(messages)
            out_msgs = []
            injected = False
            for m in messages:
                if m.get("role") == "system" and not injected:
                    out_msgs.append({"role": "system", "content": (m.get("content", "") + doc_block)})
                    injected = True
                else:
                    out_msgs.append(dict(m))
            return out_msgs

        if mode == CognitiveMode.SYSTEM1_DIRECT:
            sys_content = self.SYSTEM1_DIRECT_PROMPT
        elif mode == CognitiveMode.SYSTEM2_CAUSAL_LOGIC:
            sys_content = self.SYSTEM2_CAUSAL_PROMPT
        else:
            sys_content = self.SYSTEM2_MATH_PROMPT

        if getattr(self, "svon_adapter_loaded", False):
            sys_content = sys_content.replace(self.SODALS_CORE_IDENTITY, self.SVON_CORE_IDENTITY)

        if getattr(self, "interactive_clarify", True):
            if assessment is not None:
                if assessment.protocol_mode == "compact_slot" and assessment.compact_protocol_text:
                    sys_content = sys_content + assessment.compact_protocol_text
                elif assessment.should_enable_thinking and assessment.missing_critical:
                    sys_content = sys_content + CLARIFICATION_PROTOCOL_PROMPT
            elif mode != CognitiveMode.SYSTEM2_MATH_FORMAL:
                sys_content = sys_content + CLARIFICATION_PROTOCOL_PROMPT

        return [{"role": "system", "content": sys_content + doc_block}] + list(messages)

    @staticmethod
    def _conversation_signature(messages: List[Dict[str, str]]) -> str:
        """Stable normalized key of messages preceding the user's clarification reply."""
        norm = [(m.get("role", ""), (m.get("content") or "").strip()) for m in messages if m.get("role") != "system"]
        return json.dumps(norm, ensure_ascii=False)

    def submit_clarification(self, freeze_id: str, raw_answer: str) -> Optional[str]:
        """Unfreezes a waiting mid-turn clarification session with the user's answer."""
        sess_lock = getattr(self, "_session_lock", None)
        sessions = getattr(self, "_frozen_sessions", {})
        if sess_lock is not None:
            with sess_lock:
                sess = sessions.get(freeze_id)
        else:
            sess = sessions.get(freeze_id)
        if sess is None:
            return None
        resolved = resolve_clarification_answer(raw_answer, sess.pending_request)
        sess.raw_user_answer = raw_answer
        sess.user_answer = resolved
        sess.answer_event.set()
        return resolved

    def _pop_matching_frozen_session(
        self,
        messages: List[Dict[str, str]],
        freeze_id: Optional[str] = None,
    ) -> Optional[FrozenClarificationSession]:
        """Matches a stateless follow-up turn or explicit `freeze_id` back to its frozen state."""
        sessions = getattr(self, "_frozen_sessions", None)
        sig_map = getattr(self, "_frozen_by_sig", None)
        sess_lock = getattr(self, "_session_lock", None)
        if not sessions or sess_lock is None:
            return None

        with sess_lock:
            now = time.time()
            stale = [fid for fid, s in sessions.items() if now - s.created_at > 1800]
            for fid in stale:
                s_old = sessions.pop(fid, None)
                if sig_map is not None and s_old is not None and s_old.conversation_sig:
                    if sig_map.get(s_old.conversation_sig) == fid:
                        sig_map.pop(s_old.conversation_sig, None)
            if freeze_id and freeze_id in sessions:
                sess = sessions.pop(freeze_id)
                if sig_map is not None and sess.conversation_sig and sig_map.get(sess.conversation_sig) == freeze_id:
                    sig_map.pop(sess.conversation_sig, None)
                return sess
            if len(messages) >= 2 and messages[-1].get("role") == "user":
                # Client sent either [orig_user, assistant_ask, reply_user] or [orig_user, reply_user]
                for prefix_slice in (messages[:-1], messages[:-2]):
                    if not prefix_slice:
                        continue
                    sig = self._conversation_signature(prefix_slice)
                    fid = sig_map.pop(sig, None) if sig_map is not None else None
                    if fid and fid in sessions:
                        return sessions.pop(fid)
        return None

    def _run_streaming_pass(
        self,
        formatted_prompt: str,
        use_adapters: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        rep_penalty: float,
        stop_on_clarification: bool = True,
        entropy_processor: Optional[EntropyAndSkewGProcessor] = None,
        resume_cache_state: Optional[Dict[str, Any]] = None,
        freeze_state_out: Optional[Dict[str, Any]] = None,
        blend_alpha: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Executes a single streaming generation pass on GPU with:
          - DynamicCache KV + GatedDeltaNet CPU-offload / GPU-restore delta resume
          - Shannon Entropy + Skew-G LogitsProcessor monitoring
          - Mid-turn StoppingCriteria (tag or epistemic entropy watchdog)
        """
        gpu_lock = getattr(self, "lock", None)
        acquired = False
        if gpu_lock is not None:
            gpu_lock.acquire()
            acquired = True
        try:
            if (
                use_adapters
                and blend_alpha is not None
                and getattr(self, "blender", None) is not None
                and getattr(self, "model", None) is not None
            ):
                try:
                    self.blender.set_blend(self.model, float(blend_alpha))
                except Exception:
                    pass
            restored_cache = None
            resumed_via_cache = False
            if (
                resume_cache_state is not None
                and resume_cache_state.get("frozen_cache_cpu") is not None
                and resume_cache_state.get("frozen_token_ids")
            ):
                restored_cache = restore_cache_to_device(resume_cache_state["frozen_cache_cpu"], self.device)
                if getattr(self, "svon_cgw_active", False) and resume_cache_state.get("cvnn_phase_offsets"):
                    try:
                        from svon_cvnn_deltanet import restore_cvnn_phase_offsets
                        restore_cvnn_phase_offsets(
                            self.model, resume_cache_state["cvnn_phase_offsets"], cache_params=restored_cache
                        )
                    except Exception:
                        pass
                frozen_ids_t = torch.tensor(
                    [resume_cache_state["frozen_token_ids"]], dtype=torch.long, device=self.device
                )
                delta_text = resume_cache_state.get("delta_text", "")
                delta_ids_t = self.tokenizer(
                    delta_text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
                if delta_ids_t.shape[1] > 0:
                    full_ids = torch.cat([frozen_ids_t, delta_ids_t], dim=1)
                    inputs = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids)}
                    resumed_via_cache = True
                else:
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                    restored_cache = None
            else:
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
            )
            prompt_tok_len = int(inputs["input_ids"].shape[1])
            gen_kwargs: Dict[str, Any] = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.05),
                repetition_penalty=rep_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=(self.eos_token_ids if getattr(self, "eos_token_ids", None) else self.tokenizer.eos_token_id),
                use_cache=True,
            )
            if resumed_via_cache and restored_cache is not None:
                gen_kwargs["past_key_values"] = restored_cache
            if temperature > 0.05:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            if entropy_processor is not None:
                gen_kwargs["logits_processor"] = LogitsProcessorList([entropy_processor])
            cancel_event = threading.Event()
            stop_criteria = None
            if stop_on_clarification:
                stop_criteria = build_mid_turn_stopping_criteria(
                    self.tokenizer,
                    prompt_tok_len,
                    entropy_processor=entropy_processor,
                    cancel_event=cancel_event,
                )
                gen_kwargs["stopping_criteria"] = stop_criteria
                gen_kwargs["return_dict_in_generate"] = True
            else:
                stop_criteria = build_mid_turn_stopping_criteria(
                    self.tokenizer,
                    prompt_tok_len,
                    entropy_processor=None,
                    cancel_event=cancel_event,
                )
                # Only use cancel_event when stop_on_clarification is False
                stop_criteria[0].start_len = 10**9
                gen_kwargs["stopping_criteria"] = stop_criteria

            worker_error: List[BaseException] = []

            def _generate_worker():
                gen_out = None
                try:
                    with torch.inference_mode():
                        if not use_adapters and getattr(self, "svon_adapter_loaded", False):
                            try:
                                if hasattr(self.model, "enable_adapter_layers"):
                                    self.model.enable_adapter_layers()
                                self.model.set_adapter("svon_identity")
                                gen_out = self.model.generate(**gen_kwargs)
                            finally:
                                try:
                                    self.blender.enable_multi_adapter(self.model)
                                except Exception:
                                    pass
                        elif not use_adapters and hasattr(self.model, "disable_adapter"):
                            with self.model.disable_adapter():
                                gen_out = self.model.generate(**gen_kwargs)
                        else:
                            gen_out = self.model.generate(**gen_kwargs)
                except BaseException as exc:
                    worker_error.append(exc)
                finally:
                    if freeze_state_out is not None:
                        freeze_state_out["resumed_via_cache"] = resumed_via_cache
                        if entropy_processor is not None:
                            freeze_state_out["entropy_summary"] = entropy_processor.summary()
                            freeze_state_out["watchdog_tripped"] = entropy_processor.watchdog_tripped
                        crit_triggered = bool(
                            stop_on_clarification
                            and stop_criteria is not None
                            and any(getattr(c, "triggered", False) for c in stop_criteria)
                        )
                        if crit_triggered and gen_out is not None:
                            seqs = getattr(gen_out, "sequences", None)
                            if seqs is not None:
                                freeze_state_out["frozen_token_ids"] = seqs[0].detach().cpu().tolist()
                            pkv = getattr(gen_out, "past_key_values", None)
                            if pkv is not None:
                                freeze_state_out["frozen_cache_cpu"] = offload_cache_to_cpu(pkv)
                            if getattr(self, "svon_cgw_active", False):
                                try:
                                    from svon_cvnn_deltanet import collect_cvnn_phase_offsets
                                    freeze_state_out["cvnn_phase_offsets"] = collect_cvnn_phase_offsets(self.model)
                                except Exception:
                                    pass
                            if stop_criteria is not None and len(stop_criteria) > 0:
                                freeze_state_out["stop_reason"] = getattr(stop_criteria[0], "trigger_reason", "model_tag")
                    try:
                        streamer.end()
                    except Exception:
                        pass

            thread = threading.Thread(target=_generate_worker)
            thread.start()
            try:
                for new_text in streamer:
                    hit_eos = False
                    for eos_str in ("<|im_end|>", "<|endoftext|>"):
                        if eos_str in new_text:
                            new_text = new_text.split(eos_str)[0]
                            hit_eos = True
                            break
                    if new_text:
                        yield new_text
                    if hit_eos:
                        break
            finally:
                cancel_event.set()
                thread.join()
            if worker_error:
                raise worker_error[0]
        finally:
            if acquired and gpu_lock is not None:
                gpu_lock.release()

    def _generate_with_mid_turn_freeze(
        self,
        formatted_prompt: str,
        use_adapters: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        rep_penalty: float,
        clarify_enabled: bool,
        clarify_callback: Optional[Callable[[ClarificationRequest, str], str]],
        mode_value: str,
        alpha: float,
        rk4_steps: int,
        domain_label: str,
        user_query: str,
        conversation_sig: str,
        assessment: Optional[PreconditionAssessment] = None,
        constraint_lock: Optional[ClarificationConstraintLock] = None,
        initial_resume_cache_state: Optional[Dict[str, Any]] = None,
        initial_clarifications_asked: int = 0,
        initial_opened_think: Optional[bool] = None,
        max_clarifications: int = 2,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams generation through `ClarificationStreamFilter` + `EntropyAndSkewGProcessor`.
        Supports model-emitted `<sodals_ask>` tags AND Epistemic Entropy Watchdog freezes,
        CPU-offloaded `DynamicCache` state freeze/resume, Key-Lock pruning, and clean history output.
        """
        current_prompt = formatted_prompt
        accumulated_raw = ""
        clarifications_asked = int(initial_clarifications_asked)
        lock = constraint_lock if constraint_lock is not None else ClarificationConstraintLock()
        next_resume_cache = initial_resume_cache_state
        kv_cache_resumed = bool(
            initial_resume_cache_state and initial_resume_cache_state.get("frozen_cache_cpu") is not None
        )
        last_entropy_summary: Dict[str, Any] = {}
        prompt_opened_think = (
            bool(initial_opened_think)
            if initial_opened_think is not None
            else (formatted_prompt or "").rstrip().endswith("<think>")
        )

        structural_ids = getattr(self, "_structural_token_ids", None)
        if structural_ids is None:
            structural_ids = collect_structural_token_ids(self.tokenizer) if hasattr(self, "tokenizer") and self.tokenizer is not None else set()
            self._structural_token_ids = structural_ids

        for pass_idx in range(max(0, max_clarifications) + 1):
            allow_freeze = bool(clarify_enabled and pass_idx < max_clarifications)
            s_filter = ClarificationStreamFilter()
            pending_req: Optional[ClarificationRequest] = None
            trigger_source = "model_tag"
            pass_raw = ""

            watchdog_armed = bool(
                allow_freeze
                and assessment is not None
                and len(assessment.missing_critical) > 0
                and len(lock.locked_slots) == 0
            )
            ent_proc = (
                EntropyAndSkewGProcessor(
                    prompt_len=0,
                    structural_token_ids=structural_ids,
                    enable_watchdog=watchdog_armed,
                )
                if allow_freeze
                else None
            )
            freeze_state_out: Dict[str, Any] = {}

            pass_kwargs: Dict[str, Any] = {
                "formatted_prompt": current_prompt,
                "use_adapters": use_adapters,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "rep_penalty": rep_penalty,
            }
            try:
                sig = inspect.signature(self._run_streaming_pass)
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if has_varkw or "stop_on_clarification" in sig.parameters:
                    pass_kwargs["stop_on_clarification"] = allow_freeze
                if has_varkw or "entropy_processor" in sig.parameters:
                    pass_kwargs["entropy_processor"] = ent_proc
                if has_varkw or "resume_cache_state" in sig.parameters:
                    pass_kwargs["resume_cache_state"] = next_resume_cache
                if has_varkw or "freeze_state_out" in sig.parameters:
                    pass_kwargs["freeze_state_out"] = freeze_state_out
                if has_varkw or "blend_alpha" in sig.parameters:
                    pass_kwargs["blend_alpha"] = alpha
            except (TypeError, ValueError):
                pass

            for chunk in self._run_streaming_pass(**pass_kwargs):
                safe_emit, req = s_filter.feed(chunk)
                if safe_emit:
                    accumulated_raw += safe_emit
                    pass_raw += safe_emit
                    yield {"type": "token", "content": safe_emit}
                if req is not None:
                    pending_req = req
                    break

            if freeze_state_out.get("resumed_via_cache"):
                kv_cache_resumed = True
            if freeze_state_out.get("entropy_summary"):
                last_entropy_summary = freeze_state_out["entropy_summary"]

            # Reject duplicate loop if model repeats an already-locked clarification question:
            # instead of returning truncated text, inject locked constraints and continue once.
            if pending_req is not None and lock.is_duplicate_ask(pending_req.question):
                pending_req = None
                if not getattr(lock, "_dup_continued", False):
                    lock._dup_continued = True
                    locked_str = lock.render_lock_block() if hasattr(lock, "render_lock_block") else str(lock.locked_slots)
                    dup_delta = (
                        f"\n<user_clarification>Bu parametreler zaten belirlendi ({locked_str}). "
                        "Tekrar soru sormadan doğrudan nihai yanıtı tamamla.</user_clarification>\n"
                    )
                    current_prompt = current_prompt + pass_raw + dup_delta
                    resume_cache = None
                    resume_delta_text = None
                    continue

            # Epistemic Entropy Fallback Watchdog: model forgot `<sodals_ask>` while slots missing + high H_t
            if (
                pending_req is None
                and allow_freeze
                and freeze_state_out.get("watchdog_tripped")
                and assessment is not None
                and assessment.question_specs
            ):
                rem = s_filter.flush()
                if rem:
                    accumulated_raw += rem
                    pass_raw += rem
                    yield {"type": "token", "content": rem}
                pending_req = clarification_request_from_assessment(
                    assessment,
                    trigger_reason="Kritik önkoşul eksikliği + yüksek belirsizlik entropisi saptandı",
                )
                trigger_source = "entropy_watchdog"

            if pending_req is None:
                rem = s_filter.flush()
                if rem:
                    accumulated_raw += rem
                    pass_raw += rem
                    yield {"type": "token", "content": rem}
                display_txt, clean_hist = finalize_clean_history_and_display(
                    accumulated_raw,
                    lock=lock,
                    clarifications_asked=clarifications_asked,
                    prompt_opened_think=prompt_opened_think,
                )
                adherence = lock.check_lock_adherence(clean_hist) if clarifications_asked > 0 else {}
                yield {
                    "type": "_freeze_result",
                    "full_generated": display_txt,
                    "clean_history": clean_hist,
                    "effective_prompt": current_prompt,
                    "clarification_pending": False,
                    "clarifications_asked": clarifications_asked,
                    "kv_cache_resumed": kv_cache_resumed,
                    "entropy_summary": last_entropy_summary or (ent_proc.summary() if ent_proc is not None else {}),
                    "locked_constraints": dict(lock.locked_slots),
                    "lock_adherence": adherence,
                }
                return

            clarifications_asked += 1
            freeze_id = uuid.uuid4().hex[:10]
            frozen_prefix_txt = s_filter.frozen_prefix or pass_raw
            sess = FrozenClarificationSession(
                freeze_id=freeze_id,
                formatted_prompt=current_prompt,
                frozen_prefix=frozen_prefix_txt,
                pending_request=pending_req,
                use_adapters=use_adapters,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                rep_penalty=rep_penalty,
                mode_value=mode_value,
                alpha=alpha,
                rk4_steps=rk4_steps,
                domain_label=domain_label,
                user_query=user_query,
                conversation_sig=conversation_sig,
                frozen_token_ids=freeze_state_out.get("frozen_token_ids"),
                frozen_cache_cpu=freeze_state_out.get("frozen_cache_cpu"),
                cvnn_phase_offsets=freeze_state_out.get("cvnn_phase_offsets"),
                locked_constraints=lock,
                entropy_telemetry=last_entropy_summary or (ent_proc.summary() if ent_proc is not None else {}),
                task_type=(assessment.task_type if assessment else "general"),
                ambiguity_z=(assessment.ambiguity_z if assessment else 0.0),
                trigger_source=trigger_source,
                opened_think_in_prompt=prompt_opened_think,
            )
            sess_lock = getattr(self, "_session_lock", None)
            sessions = getattr(self, "_frozen_sessions", None)
            sig_map = getattr(self, "_frozen_by_sig", None)
            if sess_lock is not None and sessions is not None:
                with sess_lock:
                    # Cap active CPU-offloaded sessions to avoid unbounded host RAM growth (LRU max 8)
                    while len(sessions) >= 8:
                        oldest_fid = min(sessions.keys(), key=lambda k: sessions[k].created_at)
                        s_ev = sessions.pop(oldest_fid, None)
                        if sig_map is not None and s_ev is not None and s_ev.conversation_sig:
                            if sig_map.get(s_ev.conversation_sig) == oldest_fid:
                                sig_map.pop(s_ev.conversation_sig, None)
                    sessions[freeze_id] = sess
                    if sig_map is not None and conversation_sig:
                        sig_map[conversation_sig] = freeze_id

            yield {
                "type": "clarification_required",
                "freeze_id": freeze_id,
                "question": pending_req.question,
                "options": pending_req.options,
                "questions": [
                    {"slot_id": q.slot_id, "question": q.question, "options": q.options}
                    for q in pending_req.questions
                ],
                "allow_custom": pending_req.allow_custom,
                "reason": pending_req.reason,
                "trigger_source": trigger_source,
                "ambiguity_z": (assessment.ambiguity_z if assessment else 0.0),
                "cache_offloaded": freeze_state_out.get("frozen_cache_cpu") is not None,
                "frozen_prefix": frozen_prefix_txt,
            }

            if clarify_callback is None:
                msg = format_stateless_clarification_reply(pending_req, freeze_id)
                yield {"type": "token", "content": msg}
                yield {
                    "type": "_freeze_result",
                    "full_generated": msg,
                    "clean_history": msg,
                    "effective_prompt": current_prompt,
                    "clarification_pending": True,
                    "freeze_id": freeze_id,
                    "question": pending_req.question,
                    "options": pending_req.options,
                    "clarifications_asked": clarifications_asked,
                    "kv_cache_resumed": kv_cache_resumed,
                    "entropy_summary": last_entropy_summary or (ent_proc.summary() if ent_proc is not None else {}),
                    "locked_constraints": dict(lock.locked_slots),
                }
                return

            raw_ans = clarify_callback(pending_req, freeze_id)
            if sess_lock is not None and sessions is not None:
                with sess_lock:
                    sessions.pop(freeze_id, None)
                    if sig_map is not None and conversation_sig and sig_map.get(conversation_sig) == freeze_id:
                        sig_map.pop(conversation_sig, None)

            resolved = lock_resolved_clarification(lock, raw_ans or "", pending_req)
            if not resolved:
                resolved = "Kullanıcı ek detay belirtmedi; en makul genel varsayımla devam et."
                lock.lock_answer(pending_req.slot_id, resolved, pending_req.question, pending_req.options)

            yield {
                "type": "clarification_resolved",
                "freeze_id": freeze_id,
                "answer": resolved,
                "locked_constraints": dict(lock.locked_slots),
            }

            lock_dir = lock.build_lock_directive()
            delta_txt = build_unfreeze_delta_text(resolved, lock_directive=lock_dir)
            if freeze_state_out.get("frozen_cache_cpu") is not None and freeze_state_out.get("frozen_token_ids"):
                next_resume_cache = {
                    "frozen_token_ids": freeze_state_out["frozen_token_ids"],
                    "frozen_cache_cpu": freeze_state_out["frozen_cache_cpu"],
                    "cvnn_phase_offsets": freeze_state_out.get("cvnn_phase_offsets"),
                    "delta_text": delta_txt,
                }
            else:
                next_resume_cache = None

            current_prompt = build_unfreeze_continuation_prompt(
                current_prompt, frozen_prefix_txt, resolved, lock_directive=lock_dir
            )
            accumulated_raw += f"\n<user_clarification>{resolved}</user_clarification>\n"

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        attached_docs: Optional[List[LoadedDocument]] = None,
        clarify_enabled: Optional[bool] = None,
        clarify_callback: Optional[Callable[[ClarificationRequest, str], str]] = None,
        freeze_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generates streaming tokens with Three-Regime Arkhe Routing and Mid-Turn Thought Freezing:
          - SYSTEM1_DIRECT       : Low-latency fast-path (thinking enabled on planning/ambiguous cues)
          - SYSTEM2_CAUSAL_LOGIC : Goal-Precondition reasoning + Mid-Turn Clarification Freeze (`<sodals_ask>`)
          - SYSTEM2_MATH_FORMAL  : SoftLoRA + RK4 Pondering + Lean Tactic Verifier + Live Step Rollback
        """
        active_clarify = getattr(self, "interactive_clarify", True) if clarify_enabled is None else bool(clarify_enabled)
        if getattr(self, "no_system_prompt", False):
            active_clarify = False

        resumed_sess = self._pop_matching_frozen_session(messages, freeze_id=freeze_id)
        assessment: Optional[PreconditionAssessment] = None
        constraint_lock = ClarificationConstraintLock()
        initial_resume_cache_state: Optional[Dict[str, Any]] = None
        initial_clarifications_asked = 0

        if resumed_sess is not None:
            raw_reply = (messages[-1].get("content") or "") if messages else ""
            constraint_lock = resumed_sess.locked_constraints or ClarificationConstraintLock()
            resolved = lock_resolved_clarification(constraint_lock, raw_reply, resumed_sess.pending_request)
            mode = CognitiveMode(resumed_sess.mode_value)
            alpha = resumed_sess.alpha
            rk4_steps = resumed_sess.rk4_steps
            effective_rep_penalty = resumed_sess.rep_penalty
            effective_max_tokens = resumed_sess.max_new_tokens
            use_adapters = resumed_sess.use_adapters
            # Adapter blend is applied inside `_run_streaming_pass` under `self.lock`.
            domain_label = resumed_sess.domain_label + " [Netleştirildi]"
            user_msg = resumed_sess.user_query
            lock_dir = constraint_lock.build_lock_directive()
            if resumed_sess.frozen_cache_cpu is not None and resumed_sess.frozen_token_ids:
                initial_resume_cache_state = {
                    "frozen_token_ids": resumed_sess.frozen_token_ids,
                    "frozen_cache_cpu": resumed_sess.frozen_cache_cpu,
                    "cvnn_phase_offsets": resumed_sess.cvnn_phase_offsets,
                    "delta_text": build_unfreeze_delta_text(resolved, lock_directive=lock_dir),
                }
            formatted_prompt = build_unfreeze_continuation_prompt(
                resumed_sess.formatted_prompt,
                resumed_sess.frozen_prefix,
                resolved,
                lock_directive=lock_dir,
            )
            conv_sig = resumed_sess.conversation_sig
            initial_clarifications_asked = 1
        else:
            user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break

            mode, alpha, rk4_steps, rep_penalty, domain_label = self.route_prompt(user_msg)
            if attached_docs and mode == CognitiveMode.SYSTEM1_DIRECT:
                mode = CognitiveMode.SYSTEM2_CAUSAL_LOGIC
                alpha = 0.15
                rk4_steps = 2
                domain_label = f"Document Analysis ({len(attached_docs)} belge)"
            elif attached_docs:
                domain_label = f"{domain_label} + {len(attached_docs)} belge"

            if active_clarify:
                assessment = SemanticPreconditionGraph.evaluate(
                    user_msg,
                    history_messages=messages[:-1],
                    mode_value=mode.value,
                    is_agent=False,
                    has_attached_docs=bool(attached_docs),
                )

            aug_messages = self._prepare_messages_for_mode(
                messages, mode, attached_docs=attached_docs, user_query=user_msg, assessment=assessment
            )

            needs_clarify_think = bool(
                active_clarify
                and (
                    (assessment is not None and assessment.should_enable_thinking)
                    or PLANNING_OR_AMBIGUOUS_CUE_RE.search(user_msg)
                )
            )
            enable_thinking = (mode != CognitiveMode.SYSTEM1_DIRECT) or needs_clarify_think
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    aug_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    aug_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not enable_thinking and formatted_prompt.endswith("<think>\n"):
                    formatted_prompt += "\n</think>\n\n"

            effective_rep_penalty = 1.0 if (len(messages) > 1 or bool(attached_docs)) else rep_penalty
            effective_max_tokens = max(max_new_tokens, 1536) if attached_docs else max(1, int(max_new_tokens))
            use_adapters = (mode == CognitiveMode.SYSTEM2_MATH_FORMAL)
            conv_sig = self._conversation_signature(messages)

        full_generated = ""
        clean_history = ""
        clarification_pending = False
        pending_freeze_id: Optional[str] = None
        clarifications_asked = 0
        kv_cache_resumed = False
        entropy_summary: Dict[str, Any] = {}
        locked_constraints: Dict[str, str] = dict(constraint_lock.locked_slots)
        lock_adherence: Dict[str, Any] = {}
        t_start = time.time()

        # Initial telemetry packet
        yield {
            "type": "telemetry",
            "cognitive_mode": mode.value,
            "alpha": alpha,
            "rk4_steps": rk4_steps,
            "rep_penalty": effective_rep_penalty,
            "domain": domain_label,
            "ambiguity_z": (assessment.ambiguity_z if assessment else (resumed_sess.ambiguity_z if resumed_sess else 0.0)),
            "autonomy_level": (assessment.autonomy_level.value if assessment else AutonomyLevel.FULL.value),
            "missing_preconditions": (assessment.missing_critical if assessment else []),
        }
        if resumed_sess is not None:
            yield {
                "type": "clarification_resolved",
                "freeze_id": resumed_sess.freeze_id,
                "answer": resolved,
                "locked_constraints": dict(constraint_lock.locked_slots),
            }

        effective_prompt = formatted_prompt
        for pkt in self._generate_with_mid_turn_freeze(
            formatted_prompt=formatted_prompt,
            use_adapters=use_adapters,
            max_new_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            rep_penalty=effective_rep_penalty,
            clarify_enabled=active_clarify,
            clarify_callback=clarify_callback,
            mode_value=mode.value,
            alpha=alpha,
            rk4_steps=rk4_steps,
            domain_label=domain_label,
            user_query=user_msg,
            conversation_sig=conv_sig,
            assessment=assessment,
            constraint_lock=constraint_lock,
            initial_resume_cache_state=initial_resume_cache_state,
            initial_clarifications_asked=initial_clarifications_asked,
            initial_opened_think=(resumed_sess.opened_think_in_prompt if resumed_sess is not None else None),
        ):
            if pkt["type"] == "_freeze_result":
                full_generated = pkt["full_generated"]
                clean_history = pkt.get("clean_history") or full_generated
                effective_prompt = pkt.get("effective_prompt") or formatted_prompt
                clarification_pending = bool(pkt["clarification_pending"])
                pending_freeze_id = pkt.get("freeze_id")
                clarifications_asked = int(pkt.get("clarifications_asked", 0))
                kv_cache_resumed = bool(pkt.get("kv_cache_resumed", False))
                entropy_summary = pkt.get("entropy_summary") or {}
                locked_constraints = pkt.get("locked_constraints") or {}
                lock_adherence = pkt.get("lock_adherence") or {}
            else:
                yield pkt

        if clarification_pending:
            token_count = max(1, len(self.tokenizer.encode(full_generated, add_special_tokens=False)))
            elapsed = max(0.001, time.time() - t_start)
            yield {
                "type": "done",
                "cognitive_mode": mode.value,
                "total_tokens": token_count,
                "elapsed_sec": round(elapsed, 2),
                "tokens_per_sec": round(token_count / elapsed, 1),
                "is_formally_sound": True,
                "proof_status_label": "CLARIFICATION PAUSED (Düşünme Donduruldu — Detay Bekleniyor)",
                "rollback_triggered": False,
                "has_sorry_bluff": False,
                "violation_type": "",
                "violation_summary": "",
                "verified_eqs": 0,
                "causal_preconditions_verified": 0,
                "extracted_answer": "",
                "clarification_pending": True,
                "freeze_id": pending_freeze_id,
                "clarifications_asked": clarifications_asked,
                "kv_cache_resumed": kv_cache_resumed,
                "ambiguity_z": (assessment.ambiguity_z if assessment else 0.0),
                "entropy_summary": entropy_summary,
                "locked_constraints": locked_constraints,
                "clean_history": full_generated,
                "full_text": full_generated,
            }
            return

        # Verification & Live Arkhe Step-Rollback (only for System-2 regimes)
        rollback_triggered = False
        if mode == CognitiveMode.SYSTEM1_DIRECT:
            proof_status_label = "FAST-PATH (System-1 Direct — ODE & Lean Bypassed)"
            is_sound = True
            has_sorry = False
            v_type = ""
            v_summary = ""
            verified_eqs = 0
            causal_preconds = 0
            extracted_ans = ""
        else:
            step_report = LeanTacticStepVerifier.verify_proof_steps(full_generated, prompt=user_msg)

            if not step_report.is_formally_sound:
                rollback_triggered = True
                failed_idx = step_report.first_failed_step_index or 1
                v_type_init = step_report.violation_type or "LogicalJump"
                v_sum_init = step_report.violation_summary or "Step obligation failed"

                if getattr(self, "svon_cgw_active", False):
                    try:
                        from svon_kuramoto_equilibrium import synchronize_model_cvnn_phases_on_rollback
                        synchronize_model_cvnn_phases_on_rollback(
                            self.model,
                            coupling_strength=max(0.35, float(alpha)),
                            max_sync_steps=max(2, int(rk4_steps)),
                        )
                    except Exception:
                        pass
                elif self.ode_module is not None:
                    try:
                        h_dim = int(getattr(self.ode_module, "hidden_dim", 4096))
                        dummy_h = torch.zeros((1, 1, h_dim), device=self.device, dtype=torch.float32)
                        self.ode_module.rollback_and_steer(dummy_h, base_alpha=max(0.35, float(alpha)))
                    except Exception:
                        pass

                rollback_banner = (
                    f"\n\n[⚡ Arkhe-Şartlar & Sembolik Rollback: Adım {failed_idx}'de '{v_type_init}' tespit edildi! "
                    f"Hatalı çıkarım geri alınıyor ve yeniden doğrulanıyor...]\n<think>\n"
                )
                yield {"type": "token", "content": rollback_banner}

                verified_prefix = step_report.verified_prefix_text.strip()
                if "</think>" in verified_prefix:
                    verified_prefix = verified_prefix.split("</think>")[0].strip()

                repair_note = (
                    f"\n[ARKHE-ŞARTLAR / SEMBOLİK DÜZELTME UYARISI: Önceki adımda ihlal saptandı -> {v_sum_init}. "
                    "Birincil hedefin gerçekleşmesi için gerekli fiziksel nesne veya şartların "
                    "hedef noktada bulunması zorunlu önkoşuldur. "
                    "Bu önkoşulu sağlayacak doğru adımı şimdi kur ve nihai yanıtı ver:]\n"
                )
                rollback_prompt = effective_prompt + (verified_prefix + "\n" if verified_prefix else "") + repair_note

                repaired_generated = (verified_prefix + "\n") if verified_prefix else ""
                for chunk in self._run_streaming_pass(
                    formatted_prompt=rollback_prompt,
                    use_adapters=use_adapters,
                    max_new_tokens=effective_max_tokens,
                    temperature=0.2,
                    top_p=0.85,
                    rep_penalty=effective_rep_penalty,
                    stop_on_clarification=False,
                    blend_alpha=alpha,
                ):
                    repaired_generated += chunk
                    yield {"type": "token", "content": chunk}

                full_generated = repaired_generated
                _, clean_history = finalize_clean_history_and_display(
                    full_generated,
                    lock=constraint_lock,
                    clarifications_asked=clarifications_asked,
                    prompt_opened_think=(
                        bool(resumed_sess.opened_think_in_prompt)
                        if resumed_sess is not None
                        else (effective_prompt or "").rstrip().endswith("<think>")
                    ),
                )
                step_report = LeanTacticStepVerifier.verify_proof_steps(full_generated, prompt=user_msg)

            proof_status_label = step_report.proof_status_label
            if rollback_triggered and step_report.is_formally_sound:
                proof_status_label += " [Repaired via Arkhe Step-Rollback]"
            is_sound = step_report.is_formally_sound
            has_sorry = step_report.has_sorry_bluff
            v_type = step_report.violation_type
            v_summary = step_report.violation_summary
            verified_eqs = step_report.verified_eq_count
            causal_preconds = step_report.causal_preconditions_verified
            raw_boxed = ""
            if r"\boxed{" in full_generated:
                tail = full_generated.split(r"\boxed{")[-1]
                depth = 1
                chars = []
                for ch in tail:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    chars.append(ch)
                raw_boxed = "".join(chars)
            extracted_ans = CstSymbolicVerifier.clean_latex_expr(raw_boxed)

        token_count = max(1, len(self.tokenizer.encode(full_generated, add_special_tokens=False)))
        elapsed = max(0.001, time.time() - t_start)
        tok_per_sec = token_count / elapsed

        self.total_tokens_generated += token_count
        self.total_requests_served += 1

        yield {
            "type": "done",
            "cognitive_mode": mode.value,
            "total_tokens": token_count,
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(tok_per_sec, 1),
            "is_formally_sound": is_sound,
            "proof_status_label": proof_status_label,
            "rollback_triggered": rollback_triggered,
            "has_sorry_bluff": has_sorry,
            "violation_type": v_type,
            "violation_summary": v_summary,
            "verified_eqs": verified_eqs,
            "causal_preconditions_verified": causal_preconds,
            "extracted_answer": extracted_ans,
            "clarification_pending": False,
            "clarifications_asked": clarifications_asked,
            "kv_cache_resumed": kv_cache_resumed,
            "ambiguity_z": (assessment.ambiguity_z if assessment else (resumed_sess.ambiguity_z if resumed_sess else 0.0)),
            "entropy_summary": entropy_summary,
            "locked_constraints": locked_constraints,
            "lock_adherence": lock_adherence,
            "clean_history": clean_history or full_generated,
            "full_text": full_generated,
        }

    def run_agent_turn(
        self,
        messages: List[Dict[str, str]],
        toolbox: LocalWorkspaceToolbox,
        attached_docs: Optional[List[LoadedDocument]] = None,
        max_new_tokens: int = 1536,
        temperature: float = 0.25,
        top_p: float = 0.9,
        max_steps: int = 8,
        auto_approve: bool = False,
        confirm_callback: Optional[Any] = None,
        clarify_enabled: Optional[bool] = None,
        clarify_callback: Optional[Callable[[ClarificationRequest, str], str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes a multi-step offline Claude Code-style ReAct loop using `LocalWorkspaceToolbox`:
        parses `<sodals_tool>{...}</sodals_tool>` blocks, supports mid-turn `<sodals_ask>` freezes,
        feeds `<tool_result>` back into the conversation, and yields streaming events.
        """
        active_clarify = getattr(self, "interactive_clarify", True) if clarify_enabled is None else bool(clarify_enabled)
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        ws_files: List[str] = []
        try:
            ws_files = sorted(
                f for f in os.listdir(toolbox.workspace_root)
                if not f.startswith(".") and os.path.isfile(os.path.join(toolbox.workspace_root, f))
            )[:15]
        except Exception:
            ws_files = []

        agent_assessment = (
            SemanticPreconditionGraph.evaluate(
                user_msg,
                history_messages=messages[:-1],
                mode_value="SYSTEM2_AGENT_CODE",
                is_agent=True,
                workspace_files=ws_files,
                has_attached_docs=bool(attached_docs),
            )
            if active_clarify
            else None
        )

        doc_block = build_document_context(attached_docs or [], query=user_msg)
        prompt_tpl = getattr(toolbox, "prompt_template", self.AGENT_CODE_PROMPT_TEMPLATE)
        domain_title = getattr(
            toolbox,
            "domain_title",
            f"Offline Code & Automation Agent ({os.path.basename(toolbox.workspace_root) or toolbox.workspace_root})",
        )
        sys_prompt = prompt_tpl.format(workspace_root=toolbox.workspace_root) + doc_block
        if active_clarify:
            if agent_assessment is not None and agent_assessment.protocol_mode == "compact_slot":
                sys_prompt += agent_assessment.compact_protocol_text
            else:
                sys_prompt += CLARIFICATION_PROTOCOL_PROMPT
        working_msgs: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}] + [
            dict(m) for m in messages if m.get("role") != "system"
        ]

        t_start = time.time()
        total_tokens = 0
        tool_calls_executed = 0
        final_text = ""
        clean_history = ""
        agent_lock = ClarificationConstraintLock()

        yield {
            "type": "telemetry",
            "cognitive_mode": "SYSTEM2_AGENT_CODE",
            "alpha": 0.0,
            "rk4_steps": 0,
            "rep_penalty": 1.0,
            "domain": domain_title,
            "ambiguity_z": (agent_assessment.ambiguity_z if agent_assessment else 0.0),
            "autonomy_level": (agent_assessment.autonomy_level.value if agent_assessment else AutonomyLevel.FULL.value),
        }

        for step in range(1, max(1, max_steps) + 1):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    working_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    working_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            step_output = ""
            step_clarify_pending = False
            step_freeze_id: Optional[str] = None
            for pkt in self._generate_with_mid_turn_freeze(
                formatted_prompt=formatted_prompt,
                use_adapters=False,
                max_new_tokens=max(max_new_tokens, 1536),
                temperature=min(temperature, 0.35),
                top_p=top_p,
                rep_penalty=1.0,
                clarify_enabled=(active_clarify and step == 1),
                clarify_callback=clarify_callback,
                mode_value="SYSTEM2_AGENT_CODE",
                alpha=0.0,
                rk4_steps=0,
                domain_label="Offline Code & Automation Agent",
                user_query=user_msg,
                conversation_sig=self._conversation_signature(working_msgs),
                assessment=(agent_assessment if step == 1 else None),
                constraint_lock=agent_lock,
            ):
                if pkt["type"] == "_freeze_result":
                    step_output = pkt["full_generated"]
                    clean_history = pkt.get("clean_history") or step_output
                    step_clarify_pending = bool(pkt.get("clarification_pending", False))
                    step_freeze_id = pkt.get("freeze_id")
                else:
                    yield pkt

            step_tokens = max(1, len(self.tokenizer.encode(step_output, add_special_tokens=False)))
            total_tokens += step_tokens

            if step_clarify_pending:
                final_text = step_output
                break

            calls = parse_tool_calls(step_output)
            if not calls:
                final_text = strip_tool_blocks(step_output) or step_output
                break

            working_msgs.append({"role": "assistant", "content": step_output})
            result_blocks: List[str] = []

            for call in calls[:2]:
                tool_calls_executed += 1
                yield {"type": "tool_call", "step": step, "call": call}
                res: ToolExecutionResult = toolbox.execute_tool(
                    call,
                    auto_approve=auto_approve,
                    confirm_callback=confirm_callback,
                )
                yield {"type": "tool_result", "step": step, "call": call, "result": res}
                status_str = "OK" if res.ok else "ERROR"
                diff_part = f"\n[diff]\n{res.diff}" if res.diff else ""
                result_blocks.append(
                    f"<tool_result tool=\"{res.tool}\" status=\"{status_str}\">\n{res.output}{diff_part}\n</tool_result>"
                )

            feedback_msg = "\n\n".join(result_blocks)
            if len(calls) > 2:
                feedback_msg += f"\n\n[Not: Adım başına en fazla 2 araç çalıştırılır; kalan {len(calls) - 2} araç çağrısı sonraki adıma bırakıldı.]"
            if step == max_steps:
                feedback_msg += "\n\nSon adım bütçesine ulaşıldı. Artık araç çağırmadan mevcut bulguları Türkçe özetle."
                working_msgs.append({"role": "user", "content": feedback_msg})
                try:
                    synth_prompt = self.tokenizer.apply_chat_template(
                        working_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except TypeError:
                    synth_prompt = self.tokenizer.apply_chat_template(
                        working_msgs, tokenize=False, add_generation_prompt=True
                    )
                synth_out = ""
                for chunk in self._run_streaming_pass(
                    formatted_prompt=synth_prompt,
                    use_adapters=False,
                    max_new_tokens=max(512, max_new_tokens // 2),
                    temperature=min(temperature, 0.3),
                    top_p=top_p,
                    rep_penalty=1.0,
                    stop_on_clarification=False,
                ):
                    synth_out += chunk
                    yield {"type": "token", "content": chunk}
                total_tokens += max(1, len(self.tokenizer.encode(synth_out, add_special_tokens=False)))
                final_text = strip_tool_blocks(synth_out) or synth_out
                clean_history = final_text
                break
            working_msgs.append({"role": "user", "content": feedback_msg})
            final_text = strip_tool_blocks(step_output) or step_output

        elapsed = max(0.001, time.time() - t_start)
        self.total_tokens_generated += total_tokens
        self.total_requests_served += 1

        yield {
            "type": "done",
            "cognitive_mode": "SYSTEM2_AGENT_CODE",
            "steps": step,
            "tool_calls_executed": tool_calls_executed,
            "total_tokens": total_tokens,
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(total_tokens / elapsed, 1),
            "is_formally_sound": True,
            "proof_status_label": f"OFFLINE AGENT ({tool_calls_executed} araç çağrısı tamamlandı)",
            "rollback_triggered": False,
            "has_sorry_bluff": False,
            "violation_type": "",
            "violation_summary": "",
            "verified_eqs": 0,
            "causal_preconditions_verified": 0,
            "extracted_answer": "",
            "clarification_pending": bool(step_clarify_pending),
            "freeze_id": step_freeze_id,
            "locked_constraints": dict(agent_lock.locked_slots),
            "clean_history": clean_history or final_text,
            "full_text": final_text,
        }


# ============================================================================
# 2. FASTAPI OLLAMA & OPENAI COMPATIBLE HTTP SERVER
# ============================================================================

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="SODALS Neurosymbolic Intelligence Server",
    description="Full-stack serving daemon hosting Qwen 3.5 9B with Dual LoRA Adapters, GPR Routing, Pondering ODE & Lean Verifiers.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def get_root():
    return {
        "engine": "SODALS Neurosymbolic Intelligence",
        "version": "2.1.0",
        "status": "RUNNING",
        "architecture": {
            "base_model": "SODALS Backbone 9B (SafeTensors 4-bit NF4)",
            "architect": "Burak",
            "framework": "sodals_core (Rust & Python Neurosymbolic Pipeline)",
            "adapters": ["expert_general_math (GRPO)", "expert_abstract_algebra (Pondering)"],
            "router": "Neural Gaussian Phase Router (GPR)",
            "manifold_integrator": "Adaptive Pondering ODE (RK4)",
            "formal_verifier": "Symbolic Step Checker (SymPy)",
            "epistemic_pipeline": "Six Cognitive Arkhes",
        },
        "endpoints": {
            "ollama_tags": "/api/tags",
            "ollama_chat": "/api/chat",
            "ollama_generate": "/api/generate",
            "openai_models": "/v1/models",
            "openai_chat": "/v1/chat/completions",
            "arkhe_status": "/api/arkhe/status",
        }
    }


@app.get("/api/version")
def get_version():
    return {"version": "SODALS-v2.1-Neurosymbolic"}


@app.get("/api/tags")
def get_ollama_tags():
    return {
        "models": [
            {
                "name": "sodals:latest",
                "model": "sodals-neurosymbolic-9b",
                "modified_at": "2026-11-12T12:00:00Z",
                "size": 5800000000,
                "digest": "sha256:sodals_neurosymbolic_9b_dual_lora",
                "details": {
                    "parent_model": "SODALS-9B-Base",
                    "author": "Burak",
                    "format": "safetensors_lora_blend",
                    "family": "sodals_neuro",
                    "parameter_size": "9B",
                    "quantization_level": "NF4",
                }
            },
            {
                "name": "sodals-math:latest",
                "model": "sodals-math-olympiad-9b",
                "modified_at": "2026-11-12T12:00:00Z",
                "size": 5800000000,
                "digest": "sha256:sodals_math_olympiad_d1_grpo",
            }
        ]
    }


@app.get("/v1/models")
def get_openai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "sodals-neurosymbolic-9b",
                "object": "model",
                "created": 1731412800,
                "owned_by": "sodals",
            },
            {
                "id": "sodals-math-olympiad",
                "object": "model",
                "created": 1731412800,
                "owned_by": "sodals",
            }
        ]
    }


@app.get("/api/arkhe/status")
def get_arkhe_status():
    engine = SodalsNeurosymbolicEngine.get_instance()
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    return {
        "total_requests_served": engine.total_requests_served,
        "total_tokens_generated": engine.total_tokens_generated,
        "gpu_vram_allocated_mb": round(vram_mb, 2),
        "router_threshold": engine.router.threshold if engine.router else None,
        "adapters_loaded": list(engine.blender.adapter_names) if engine.blender else [],
    }


def _resolve_api_documents(raw_docs: Optional[List[Any]]) -> List[LoadedDocument]:
    """Converts API `documents` field entries (strictly inline `{name, content}` dicts) into LoadedDocument objects.
    Arbitrary host filesystem paths are rejected over HTTP to prevent Local File Inclusion (LFI)."""
    resolved: List[LoadedDocument] = []
    for idx, item in enumerate(raw_docs or [], start=1):
        if isinstance(item, str) or (isinstance(item, dict) and item.get("path")):
            raise ValueError(
                "HTTP API üzerinden yerel dosya yolu okuma güvenlik nedeniyle kapalıdır; "
                "yalnızca satır içi {'name': '...', 'content': '...'} formatı kabul edilir."
            )
        if isinstance(item, dict) and item.get("content") is not None:
            txt = str(item["content"]).strip()
            if not txt:
                continue
            name = os.path.basename(str(item.get("name", f"document_{idx}.txt"))) or f"document_{idx}.txt"
            resolved.append(
                LoadedDocument(
                    name=name,
                    path=name,
                    ext=os.path.splitext(name)[1].lower() or ".txt",
                    text=txt,
                    page_or_section_count=max(1, txt.count("\n\n") + 1),
                    char_count=len(txt),
                    word_count=len(txt.split()),
                )
            )
    return resolved


def _normalize_chat_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Coerces OpenAI/Ollama message objects (including structured content blocks) into `[{role, content}]`."""
    out: List[Dict[str, str]] = []
    for m in raw_messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict):
                    parts.append(str(blk.get("text") or blk.get("content") or ""))
                elif blk is not None:
                    parts.append(str(blk))
            txt = "\n".join(p for p in parts if p)
        elif content is None:
            txt = ""
        else:
            txt = str(content)
        out.append({"role": role, "content": txt})
    return out


def _collect_stream_sync(gen: Generator[Dict[str, Any], None, None]) -> Tuple[str, Dict[str, Any]]:
    """Drains a synchronous engine generator safely inside a worker thread."""
    full_text = ""
    done_pkt: Dict[str, Any] = {}
    try:
        for pkt in gen:
            if pkt["type"] == "token":
                full_text += pkt["content"]
            elif pkt["type"] == "done":
                done_pkt = pkt
    finally:
        try:
            gen.close()
        except Exception:
            pass
    return full_text, done_pkt


class ClarifySubmitRequest(BaseModel):
    freeze_id: str
    answer: str


@app.post("/api/clarify")
def submit_clarification_endpoint(req: ClarifySubmitRequest):
    engine = SodalsNeurosymbolicEngine.get_instance()
    resolved = engine.submit_clarification(req.freeze_id, req.answer)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Aktif dondurulmuş oturum bulunamadı: {req.freeze_id}")
    return {"status": "unfrozen", "freeze_id": req.freeze_id, "resolved_answer": resolved}


def _build_http_wait_callback(engine: SodalsNeurosymbolicEngine, timeout_sec: float = 300.0):
    """Creates a non-deadlocking wait callback for `wait_for_clarification=true` HTTP streams."""
    def _wait_cb(_req: ClarificationRequest, freeze_id: str) -> str:
        with engine._session_lock:
            sess = engine._frozen_sessions.get(freeze_id)
        if sess is None:
            return ""
        sess.answer_event.wait(timeout=timeout_sec)
        return sess.raw_user_answer if sess.raw_user_answer is not None else (sess.user_answer or "")
    return _wait_cb


async def _iter_generator_async(gen: Generator[Dict[str, Any], None, None]) -> AsyncGenerator[Dict[str, Any], None]:
    """Iterates a synchronous generator in worker threads and closes it reliably on client disconnect."""
    sentinel = object()
    try:
        while True:
            pkt = await asyncio.to_thread(next, gen, sentinel)
            if pkt is sentinel:
                break
            yield pkt  # type: ignore[misc]
    finally:
        try:
            await asyncio.to_thread(gen.close)
        except Exception:
            pass


class OllamaChatRequest(BaseModel):
    model: str = "sodals:latest"
    messages: List[Dict[str, Any]]
    stream: bool = True
    options: Optional[Dict[str, Any]] = None
    documents: Optional[List[Any]] = None
    clarify_enabled: bool = True
    wait_for_clarification: bool = False
    freeze_id: Optional[str] = None


@app.post("/api/chat")
async def ollama_chat(req: OllamaChatRequest):
    engine = SodalsNeurosymbolicEngine.get_instance()
    temp = float(req.options.get("temperature", 0.6)) if req.options else 0.6
    max_tokens = int(req.options.get("num_predict", 1024)) if req.options else 1024
    norm_msgs = _normalize_chat_messages(req.messages)
    try:
        attached_docs = _resolve_api_documents(req.documents)
    except (FileNotFoundError, IsADirectoryError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    wait_cb = _build_http_wait_callback(engine) if (req.stream and req.wait_for_clarification) else None

    if not req.stream:
        gen = engine.stream_generate(
            norm_msgs,
            max_new_tokens=max_tokens,
            temperature=temp,
            attached_docs=attached_docs,
            clarify_enabled=req.clarify_enabled,
            freeze_id=req.freeze_id,
        )
        full_text, done_pkt = await asyncio.to_thread(_collect_stream_sync, gen)
        return {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": full_text},
            "done": True,
            "eval_count": done_pkt.get("total_tokens", 0),
            "eval_duration": int(done_pkt.get("elapsed_sec", 0) * 1e9),
            "arkhe_diagnostics": done_pkt,
        }

    # Streaming Ollama format (supports mid-turn clarification events)
    async def event_generator():
        gen = engine.stream_generate(
            norm_msgs,
            max_new_tokens=max_tokens,
            temperature=temp,
            attached_docs=attached_docs,
            clarify_enabled=req.clarify_enabled,
            clarify_callback=wait_cb,
            freeze_id=req.freeze_id,
        )
        async for pkt in _iter_generator_async(gen):
            if pkt["type"] in ("clarification_required", "clarification_resolved"):
                yield json.dumps({"model": req.model, "event": pkt, "done": False}, ensure_ascii=False) + "\n"
            elif pkt["type"] == "token":
                chunk = {
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": pkt["content"]},
                    "done": False,
                }
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
            elif pkt["type"] == "done":
                final_chunk = {
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "eval_count": pkt["total_tokens"],
                    "eval_duration": int(pkt["elapsed_sec"] * 1e9),
                    "arkhe_diagnostics": pkt,
                }
                yield json.dumps(final_chunk, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


class OpenAIChatRequest(BaseModel):
    model: str = "sodals-neurosymbolic-9b"
    messages: List[Dict[str, Any]]
    stream: bool = False
    temperature: Optional[float] = 0.6
    max_tokens: Optional[int] = 1024
    documents: Optional[List[Any]] = None
    clarify_enabled: bool = True
    wait_for_clarification: bool = False
    freeze_id: Optional[str] = None


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest):
    engine = SodalsNeurosymbolicEngine.get_instance()
    chat_id = f"chatcmpl-sodals-{int(time.time()*1000)}"
    norm_msgs = _normalize_chat_messages(req.messages)
    try:
        attached_docs = _resolve_api_documents(req.documents)
    except (FileNotFoundError, IsADirectoryError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    wait_cb = _build_http_wait_callback(engine) if (req.stream and req.wait_for_clarification) else None

    req_temp = 0.6 if req.temperature is None else float(req.temperature)

    if not req.stream:
        gen = engine.stream_generate(
            norm_msgs,
            max_new_tokens=req.max_tokens or 1024,
            temperature=req_temp,
            attached_docs=attached_docs,
            clarify_enabled=req.clarify_enabled,
            freeze_id=req.freeze_id,
        )
        full_text, done_pkt = await asyncio.to_thread(_collect_stream_sync, gen)

        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": done_pkt.get("total_tokens", 0),
                "total_tokens": done_pkt.get("total_tokens", 0),
            },
            "sodals_diagnostics": done_pkt,
        }

    # Streaming SSE format for OpenAI clients
    async def sse_generator():
        gen = engine.stream_generate(
            norm_msgs,
            max_new_tokens=req.max_tokens or 1024,
            temperature=req_temp,
            attached_docs=attached_docs,
            clarify_enabled=req.clarify_enabled,
            clarify_callback=wait_cb,
            freeze_id=req.freeze_id,
        )
        async for pkt in _iter_generator_async(gen):
            if pkt["type"] in ("clarification_required", "clarification_resolved"):
                ev_data = {
                    "id": chat_id,
                    "object": "chat.completion.clarification",
                    "created": int(time.time()),
                    "model": req.model,
                    "sodals_clarification": pkt,
                }
                yield f"data: {json.dumps(ev_data, ensure_ascii=False)}\n\n"
            elif pkt["type"] == "token":
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": pkt["content"]},
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            elif pkt["type"] == "done":
                end_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "sodals_diagnostics": pkt,
                }
                yield f"data: {json.dumps(end_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


class AnthropicMessagesRequest(BaseModel):
    model: str = "sodals-neurosymbolic-9b"
    messages: List[Dict[str, Any]]
    system: Optional[Any] = None
    max_tokens: int = 1536
    temperature: Optional[float] = 0.6
    stream: bool = False


def _flatten_anthropic_messages(messages: List[Dict[str, Any]], system: Optional[Any] = None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if system:
        if isinstance(system, str):
            sys_txt = system
        elif isinstance(system, list):
            sys_txt = "\n".join(str(b.get("text", "")) for b in system if isinstance(b, dict))
        else:
            sys_txt = str(system)
        if sys_txt.strip():
            out.append({"role": "system", "content": sys_txt})

    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict):
                    btype = blk.get("type")
                    if btype == "text":
                        parts.append(str(blk.get("text", "")))
                    elif btype == "tool_use":
                        tname = blk.get("name", "tool")
                        tinput = json.dumps(blk.get("input", {}), ensure_ascii=False)
                        parts.append(f"[Tool Call {tname}]: {tinput}")
                    elif btype == "tool_result":
                        tr_content = blk.get("content", "")
                        if isinstance(tr_content, list):
                            tr_txt = "\n".join(
                                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                                for x in tr_content
                            )
                        else:
                            tr_txt = str(tr_content)
                        parts.append(tr_txt)
                elif isinstance(blk, str):
                    parts.append(blk)
            txt = "\n".join(parts)
        else:
            txt = str(content)
        out.append({"role": role, "content": txt})
    return out


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest):
    """Anthropic Messages API endpoint enabling offline Claude Code CLI (`ANTHROPIC_BASE_URL`)."""
    engine = SodalsNeurosymbolicEngine.get_instance()
    msg_id = f"msg_sodals_{uuid.uuid4().hex[:12]}"
    flat_msgs = _flatten_anthropic_messages(req.messages, system=req.system)
    temp = 0.6 if req.temperature is None else float(req.temperature)

    if not req.stream:
        gen = engine.stream_generate(
            flat_msgs,
            max_new_tokens=req.max_tokens or 1536,
            temperature=temp,
            clarify_enabled=False,
        )
        full_text, done_pkt = await asyncio.to_thread(_collect_stream_sync, gen)
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model,
            "content": [{"type": "text", "text": full_text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": done_pkt.get("total_tokens", 0)},
        }

    async def anthropic_sse():
        start_ev = {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(start_ev, ensure_ascii=False)}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        out_tokens = 0
        gen = engine.stream_generate(
            flat_msgs,
            max_new_tokens=req.max_tokens or 1536,
            temperature=temp,
            clarify_enabled=False,
        )
        async for pkt in _iter_generator_async(gen):
            if pkt["type"] == "token":
                delta_ev = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": pkt["content"]},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_ev, ensure_ascii=False)}\n\n"
            elif pkt["type"] == "done":
                out_tokens = int(pkt.get("total_tokens", 0))

        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': out_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    return StreamingResponse(anthropic_sse(), media_type="text/event-stream")


# ============================================================================
# 3. INTERACTIVE STREAMING CLI CHAT
# ============================================================================

def _attach_doc_to_pool(attached_docs: List[LoadedDocument], raw_path: str) -> Optional[LoadedDocument]:
    """Loads `raw_path` and adds or updates it in `attached_docs`, printing clear feedback."""
    try:
        doc = load_document(raw_path)
    except Exception as exc:
        print(f"\033[1;31m[Belge Hatası]\033[0m {exc}")
        return None

    for idx, existing in enumerate(attached_docs):
        if os.path.abspath(existing.path) == os.path.abspath(doc.path):
            attached_docs[idx] = doc
            print(
                f"\033[1;32m[Belge Güncellendi]\033[0m {doc.name} "
                f"({doc.ext}, {doc.page_or_section_count} bölüm/sayfa, {doc.word_count} kelime, {doc.char_count} karakter)"
            )
            return doc

    attached_docs.append(doc)
    print(
        f"\033[1;32m[Belge Eklendi #{len(attached_docs)}]\033[0m {doc.name} "
        f"({doc.ext}, {doc.page_or_section_count} bölüm/sayfa, {doc.word_count} kelime, {doc.char_count} karakter)"
    )
    return doc


def _cli_confirm_tool(tool_name: str, call: Dict[str, Any]) -> bool:
    """Interactive terminal confirmation before file writes/edits or local shell commands."""
    if tool_name == "run_command":
        summary = f"Komut: {call.get('command', '')}"
    elif tool_name in {"edit_file", "write_file"}:
        summary = f"Dosya: {call.get('path', '')}"
    else:
        summary = json.dumps(call, ensure_ascii=False)
    print(f"\n\033[1;33m[Onay Gerekli — {tool_name}]\033[0m {summary}")
    try:
        ans = input("  Uygulansın mı? [E/h] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in {"", "e", "evet", "y", "yes"}


def _cli_clarify_prompt(req: ClarificationRequest, _freeze_id: str) -> str:
    """Inline interactive terminal prompt (single or multi-slot) while generation is frozen mid-turn."""
    print(format_cli_clarification_prompt(req), flush=True)
    try:
        if len(req.questions) > 1:
            ans_map: Dict[str, str] = {}
            for idx, q_item in enumerate(req.questions, start=1):
                raw_part = input(f"  Yanıt ({idx}/{len(req.questions)} - {q_item.slot_id}) > ").strip()
                if raw_part:
                    ans_map[q_item.slot_id] = raw_part
            raw = json.dumps(ans_map, ensure_ascii=False)
        else:
            raw = input("  Yanıtınız > ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    resolved = resolve_clarification_answer(raw, req)
    print(f"\033[1;32m▶ [Dondurulan Düşünce Devam Ediyor: {resolved}]\033[0m\n", flush=True)
    return raw


def _print_cli_help():
    print(
        "\033[1;34m[SODALS Komutları — Belge, Çevrimdışı Kod Ajanı & İzole Mini-OS (SODALS-OS)]\033[0m\n"
        "  • Sürükle-Bırak /dosya <yol> : Belge veya kod dosyası ekler (.pdf, .docx, .xlsx, .txt, .py, .json vb.)\n"
        "  • /belgeler (/docs)           : Bağlı belgeleri listeler | /cikar <no> | /temizle\n"
        "  • /ajan [on|off|görev]       : Yerel dizin kodlama/otomasyon ajanını açar veya tek seferlik görev koşar\n"
        "  • /os [on|off|durum|görev]   : İzole SODALS-OS (Alpine Linux VM — web, not, günlük görev otomasyonu) moduna geçer\n"
        "  • /sor [on|off] (/clarify)   : Eksik detayda düşünmeyi dondurup inline soru sorma (varsayılan: AÇIK)\n"
        "  • /oto [on|off]              : Düzenleme ve shell komutları için otomatik onay (varsayılan: KAPALI / [E/h] sorar)\n"
        "  • /dizin [yol]  (/cwd)       : Çalışma dizinini gösterir veya değiştirir\n"
        "  • /ls [yol] | /bul <desen> | /ara <metin> | /oku <dosya> [bas-bitis]\n"
        "  • /calistir <komut> (/run)   : Yerel PowerShell/shell otomasyon komutu çalıştırır\n"
        "  • /geri-al (/undo)           : Ajanın yaptığı son dosya düzenlemesini/yazımını geri alır\n"
        "  • /reset | /status | /exit"
    )


def run_interactive_cli(
    agent_mode: bool = False,
    os_agent_mode: bool = False,
    os_url: str = "http://127.0.0.1:11440",
    workspace_dir: Optional[str] = None,
    auto_approve: bool = False,
    svon_identity_lora: Optional[str] = None,
    no_system_prompt: bool = False,
    enable_svon_cgw: bool = False,
    svon_cgw_weights: Optional[str] = None,
    clarify_enabled: bool = True,
    model_path: Optional[str] = None,
):
    """
    Direct console chat interface with real-time streaming, Arkhe diagnostics,
    mid-turn clarification freezing, drag-and-drop document inspection,
    Claude Code-style offline coding, and isolated SODALS-OS computer-use mode.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    engine = SodalsNeurosymbolicEngine.get_instance(
        svon_identity_lora=svon_identity_lora,
        no_system_prompt=no_system_prompt,
        enable_svon_cgw=enable_svon_cgw,
        svon_cgw_weights=svon_cgw_weights,
        model_path=model_path,
    )
    engine.interactive_clarify = bool(clarify_enabled)
    toolbox = LocalWorkspaceToolbox(workspace_root=workspace_dir or os.getcwd())
    os_toolbox = SodalsOSToolbox(base_url=os_url)
    brand_name = "SVON" if getattr(engine, "svon_adapter_loaded", False) else "SODALS"

    print("\n" + "=" * 80)
    print(f"   {brand_name} NEUROSYMBOLIC INTERACTIVE TERMINAL CHAT")
    print(f"   Architect: Burak | System Prompt: {'DISABLED (--no_system_prompt)' if no_system_prompt else brand_name}")
    print("   Epistemic Core: Six Cognitive Arkhes + GPR + Symbolic Step Checker")
    print(
        f"   Workspace: {toolbox.workspace_root} | Offline Agent: {'ON' if agent_mode else 'OFF'} | "
        f"SODALS-OS: {'ON' if os_agent_mode else 'OFF'} | "
        f"Clarify-Freeze: {'ON' if clarify_enabled else 'OFF'} | Auto-Approve: {'ON' if auto_approve else 'OFF'}"
    )
    print("=" * 80)
    print(
        "Komutlar: '/os on' (izole Mini-OS: web/not/günlük görev) | '/ajan on' (yerel kod ajanı) | "
        "'/dosya <yol>' | '/yardim' | '/exit'\n"
    )

    conversation: List[Dict[str, str]] = []
    attached_docs: List[LoadedDocument] = []

    while True:
        try:
            badges = []
            if os_agent_mode:
                badges.append("\033[36mSODALS-OS\033[0m")
            elif agent_mode:
                badges.append("\033[35mAJAN\033[0m")
            if attached_docs:
                badges.append(f"\033[33m{len(attached_docs)} belge\033[0m")
            badge_str = (" [" + " | ".join(badges) + "]") if badges else ""
            user_input = input(f"\n\033[1;36mUser\033[0m{badge_str} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting SODALS chat. Goodbye!")
            break

        if not user_input:
            continue
        low_in = user_input.lower()

        if low_in in {"/exit", "/quit", "exit", "quit"}:
            print("Session terminated.")
            break
        if low_in in {"/yardim", "/yardım", "/help"}:
            _print_cli_help()
            continue
        if low_in == "/reset":
            conversation.clear()
            attached_docs.clear()
            print("\033[1;33m[Bağlam ve Belgeler Sıfırlandı]\033[0m Temiz bir sohbet başlatılıyor.")
            continue
        if low_in == "/temizle":
            attached_docs.clear()
            print("\033[1;33m[Belgeler Temizlendi]\033[0m Sohbet geçmişi korunuyor.")
            continue
        if low_in in {"/belgeler", "/docs"}:
            if not attached_docs:
                print("\033[90m[Bilgi]\033[0m Şu anda bağlı belge yok. '/dosya <yol>' yazın veya dosyayı terminale sürükleyin.")
            else:
                print("\033[1;34m[Bağlı Belgeler]\033[0m")
                for idx, d in enumerate(attached_docs, start=1):
                    print(f"  {idx}. {d.name} ({d.ext}, {d.page_or_section_count} bölüm/sayfa, {d.word_count} kelime) — {d.path}")
            continue
        if low_in.startswith(("/cikar ", "/çıkar ", "/remove ")):
            arg = user_input.split(None, 1)[1].strip()
            removed = None
            if arg.isdigit():
                pos = int(arg) - 1
                if 0 <= pos < len(attached_docs):
                    removed = attached_docs.pop(pos)
            else:
                for pos, d in enumerate(attached_docs):
                    if d.name.lower() == arg.lower():
                        removed = attached_docs.pop(pos)
                        break
            if removed:
                print(f"\033[1;33m[Çıkarıldı]\033[0m {removed.name}")
            else:
                print(f"\033[1;31m[Bulunamadı]\033[0m Eşleşen belge yok: {arg}")
            continue

        # --- Offline Coding & Automation Commands ---
        if low_in in {"/dizin", "/cwd"}:
            print(f"\033[1;34m[Çalışma Dizini]\033[0m {toolbox.workspace_root}")
            continue
        if low_in.startswith(("/dizin ", "/cwd ")):
            arg = user_input.split(None, 1)[1].strip()
            try:
                new_ws = toolbox.set_workspace(arg)
                print(f"\033[1;32m[Çalışma Dizini Değiştirildi]\033[0m {new_ws}")
            except Exception as exc:
                print(f"\033[1;31m[Hata]\033[0m {exc}")
            continue
        if low_in in {"/ls", "/list"} or low_in.startswith(("/ls ", "/list ")):
            sub = user_input.split(None, 1)[1].strip() if " " in user_input else "."
            res = toolbox.list_dir(path=sub)
            print(res.output)
            continue
        if low_in.startswith(("/bul ", "/find ")):
            pat = user_input.split(None, 1)[1].strip()
            res = toolbox.find_files(pattern=pat)
            print(res.output)
            continue
        if low_in.startswith(("/ara ", "/grep ")):
            pat = user_input.split(None, 1)[1].strip()
            res = toolbox.grep_search(pattern=pat)
            print(res.output)
            continue
        if low_in.startswith(("/oku ", "/read ")):
            parts = user_input.split(None, 2)
            fpath = parts[1]
            s_line, e_line = 1, 120
            if len(parts) > 2 and "-" in parts[2]:
                a, b = parts[2].split("-", 1)
                if a.strip().isdigit() and b.strip().isdigit():
                    s_line, e_line = int(a.strip()), int(b.strip())
            res = toolbox.read_file(path=fpath, start_line=s_line, end_line=e_line)
            print(res.output)
            continue
        if low_in.startswith(("/calistir ", "/çalıştır ", "/run ")):
            cmd = user_input.split(None, 1)[1].strip()
            res = toolbox.execute_tool(
                {"tool": "run_command", "command": cmd},
                auto_approve=auto_approve,
                confirm_callback=_cli_confirm_tool,
            )
            print(res.output)
            continue
        if low_in in {"/geri-al", "/undo"}:
            res = toolbox.undo_last_edit()
            color = "\033[1;32m" if res.ok else "\033[1;31m"
            print(f"{color}[Geri Al]\033[0m {res.output}")
            continue
        if low_in in {"/oto", "/auto"}:
            auto_approve = not auto_approve
            print(f"\033[1;33m[Otomatik Onay]\033[0m {'AÇIK (sormadan uygular)' if auto_approve else 'KAPALI ([E/h] onay ister)'}")
            continue
        if low_in.startswith(("/oto ", "/auto ")):
            flag = user_input.split(None, 1)[1].strip().lower()
            auto_approve = flag in {"on", "ac", "aç", "1", "true", "evet"}
            print(f"\033[1;33m[Otomatik Onay]\033[0m {'AÇIK' if auto_approve else 'KAPALI'}")
            continue

        run_single_agent_turn = False
        run_single_os_turn = False
        if low_in in {"/os", "/mini-os"}:
            os_agent_mode = not os_agent_mode
            if os_agent_mode:
                agent_mode = False
            print(f"\033[1;36m[SODALS-OS İzole Mini İşletim Sistemi]\033[0m {'AKTİF' if os_agent_mode else 'KAPALI'} ({os_toolbox.base_url})")
            continue
        if low_in.startswith(("/os ", "/mini-os ")):
            arg = user_input.split(None, 1)[1].strip()
            if arg.lower() in {"on", "ac", "aç"}:
                os_agent_mode = True
                agent_mode = False
                print(f"\033[1;36m[SODALS-OS]\033[0m AKTİF ({os_toolbox.base_url})")
                continue
            if arg.lower() in {"off", "kapat"}:
                os_agent_mode = False
                print("\033[1;36m[SODALS-OS]\033[0m KAPALI")
                continue
            if arg.lower() in {"durum", "status"}:
                st_res = os_toolbox.execute_tool({"tool": "os_status"})
                print(st_res.output)
                continue
            user_input = arg
            low_in = user_input.lower()
            run_single_os_turn = True

        if low_in in {"/ajan", "/agent"}:
            agent_mode = not agent_mode
            if agent_mode:
                os_agent_mode = False
            print(f"\033[1;35m[Çevrimdışı Kod/Otomasyon Ajanı]\033[0m {'AKTİF' if agent_mode else 'KAPALI'}")
            continue
        if low_in.startswith(("/ajan ", "/agent ")):
            arg = user_input.split(None, 1)[1].strip()
            if arg.lower() in {"on", "ac", "aç", "off", "kapat", "durum", "status"}:
                if arg.lower() in {"on", "ac", "aç"}:
                    agent_mode = True
                elif arg.lower() in {"off", "kapat"}:
                    agent_mode = False
                print(f"\033[1;35m[Çevrimdışı Kod/Otomasyon Ajanı]\033[0m {'AKTİF' if agent_mode else 'KAPALI'} (Dizin: {toolbox.workspace_root})")
                continue
            user_input = arg
            low_in = user_input.lower()
            run_single_agent_turn = True

        if low_in in {"/dosya", "/doc", "/ekle"}:
            print("Kullanım: /dosya <dosya_yolu> [isteğe bağlı soru] (veya dosyayı doğrudan terminale sürükleyin)")
            continue
        if low_in.startswith(("/dosya ", "/doc ", "/ekle ")):
            rest = user_input.split(None, 1)[1].strip()
            inline_paths, rem_query = extract_inline_file_paths(rest)
            if not inline_paths:
                cand = clean_user_path(rest)
                if os.path.exists(cand):
                    inline_paths = [cand]
                    rem_query = ""
            if not inline_paths:
                _attach_doc_to_pool(attached_docs, rest)
                continue
            for fp in inline_paths:
                _attach_doc_to_pool(attached_docs, fp)
            if not rem_query:
                print("\033[90mArtık bu belge(ler) hakkında soru sorabilir veya 'özetle' diyebilirsiniz.\033[0m")
                continue
            user_input = rem_query
        else:
            inline_paths, rem_query = extract_inline_file_paths(user_input)
            if inline_paths:
                for fp in inline_paths:
                    _attach_doc_to_pool(attached_docs, fp)
                if not rem_query:
                    print("\033[90mBelge bağlama eklendi. İncelememi istediğiniz soruyu yazabilirsiniz.\033[0m")
                    continue
                user_input = rem_query

        if low_in in {"/sor", "/clarify"}:
            clarify_enabled = not clarify_enabled
            engine.interactive_clarify = clarify_enabled
            print(f"\033[1;36m[Düşünme Dondurma / Netleştirme]\033[0m {'AÇIK' if clarify_enabled else 'KAPALI'}")
            continue
        if low_in.startswith(("/sor ", "/clarify ")):
            flag = user_input.split(None, 1)[1].strip().lower()
            clarify_enabled = flag in {"on", "ac", "aç", "1", "true", "evet"}
            engine.interactive_clarify = clarify_enabled
            print(f"\033[1;36m[Düşünme Dondurma / Netleştirme]\033[0m {'AÇIK' if clarify_enabled else 'KAPALI'}")
            continue

        if low_in == "/status":
            vram = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
            print(
                f"\033[1;32m[Status]\033[0m Requests: {engine.total_requests_served} | "
                f"Tokens: {engine.total_tokens_generated} | GPU VRAM: {vram:.1f} MiB | "
                f"Docs: {len(attached_docs)} | Agent: {'ON' if agent_mode else 'OFF'} | "
                f"Clarify: {'ON' if clarify_enabled else 'OFF'} | CWD: {toolbox.workspace_root}"
            )
            continue

        conversation.append({"role": "user", "content": user_input})

        print(f"\n\033[1;35m{brand_name}\033[0m > ", end="", flush=True)

        clarify_cb = _cli_clarify_prompt if clarify_enabled else None
        done_info = {}
        if os_agent_mode or run_single_os_turn:
            pkt_stream = engine.run_agent_turn(
                conversation,
                toolbox=os_toolbox,
                attached_docs=attached_docs,
                auto_approve=True,
                confirm_callback=None,
                clarify_enabled=clarify_enabled,
                clarify_callback=clarify_cb,
            )
        elif agent_mode or run_single_agent_turn:
            pkt_stream = engine.run_agent_turn(
                conversation,
                toolbox=toolbox,
                attached_docs=attached_docs,
                auto_approve=auto_approve,
                confirm_callback=_cli_confirm_tool,
                clarify_enabled=clarify_enabled,
                clarify_callback=clarify_cb,
            )
        else:
            pkt_stream = engine.stream_generate(
                conversation,
                max_new_tokens=1024,
                temperature=0.6,
                attached_docs=attached_docs,
                clarify_enabled=clarify_enabled,
                clarify_callback=clarify_cb,
            )

        for pkt in pkt_stream:
            if pkt["type"] == "telemetry":
                if pkt.get("cognitive_mode") == CognitiveMode.SYSTEM1_DIRECT.value:
                    domain_tag = f"\033[90m[Arkhe-Bilinç: {pkt['domain']} | ODE: 0 RK4 steps (Fast-Path)]\033[0m\n"
                elif pkt.get("cognitive_mode") == "SYSTEM2_AGENT_CODE":
                    domain_tag = f"\033[95m[Offline Ajan: {pkt['domain']} | Auto-Approve: {'ON' if auto_approve else 'OFF'}]\033[0m\n"
                else:
                    domain_tag = f"\033[90m[Arkhe-Bilinç: {pkt['domain']} (α={pkt['alpha']:.2f}) | ODE: {pkt['rk4_steps']} RK4 steps]\033[0m\n"
                print(domain_tag, end="", flush=True)
            elif pkt["type"] == "token":
                print(pkt["content"], end="", flush=True)
            elif pkt["type"] == "tool_call":
                c = pkt["call"]
                print(f"\n\033[1;33m  ⚙ [Adım {pkt['step']} | Araç: {c.get('tool')}]\033[0m {json.dumps(c, ensure_ascii=False)[:180]}", flush=True)
            elif pkt["type"] == "tool_result":
                r: ToolExecutionResult = pkt["result"]
                color = "\033[32m" if r.ok else "\033[31m"
                preview = (r.output or "").splitlines()[:6]
                print(f"{color}  ↳ [{'OK' if r.ok else 'HATA'}]\033[0m " + "\n    ".join(preview), flush=True)
                if r.diff:
                    diff_preview = r.diff.splitlines()[:12]
                    print("\033[36m" + "\n".join("    " + ln for ln in diff_preview) + "\033[0m", flush=True)
            elif pkt["type"] == "done":
                done_info = pkt

        print() # Newline

        # Diagnostic banner
        soundness_color = "\033[1;32m" if done_info.get("is_formally_sound") else "\033[1;31m"
        status_label = done_info.get("proof_status_label") or (
            "FORMALLY SOUND" if done_info.get("is_formally_sound") else f"PROOF WARNING: {done_info.get('violation_summary', 'Contradiction/Sorry Jump')}"
        )
        precond_cnt = done_info.get("causal_preconditions_verified", 0)
        extra_verify = f" | Preconditions Verified: {precond_cnt}" if precond_cnt > 0 else ""
        diag_banner = (
            f"\n\033[90m┌─ Telemetry: {done_info.get('total_tokens', 0)} toks @ {done_info.get('tokens_per_sec', 0):.1f} tok/s | "
            f"Eqs Verified: {done_info.get('verified_eqs', 0)}{extra_verify}\n"
            f"└─ Proof State: {soundness_color}{status_label}\033[0m\n"
        )
        print(diag_banner)

        # History Hygiene: store ONLY clarification summary + clean final answer (no raw <think>/<sodals_ask> leakage)
        history_entry = done_info.get("clean_history") or done_info.get("full_text", "")
        conversation.append({"role": "assistant", "content": history_entry})

        # Sliding context window: keep the last 8 turns (4 full exchanges)
        # to prevent RoPE positional drift and unbounded memory consumption
        if len(conversation) > 8:
            conversation = conversation[-8:]


# ============================================================================
# 4. ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SODALS Neurosymbolic Intelligence Daemon")
    parser.add_argument("--cli", action="store_true", help="Launch interactive console chat directly")
    parser.add_argument("--agent", action="store_true", help="Start interactive CLI directly in offline coding/automation agent mode")
    parser.add_argument("--os-agent", action="store_true", help="Start interactive CLI directly in isolated SODALS-OS autonomous daily-task mode")
    parser.add_argument("--os-url", type=str, default="http://127.0.0.1:11440", help="SODALS-OS isolated micro-VM control API URL")
    parser.add_argument("--cwd", type=str, default=None, help="Initial workspace directory for offline code/automation agent")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve file edits and shell commands without prompting")
    parser.add_argument("--no-clarify", action="store_true", help="Disable mid-turn thought-freezing clarification prompts")
    parser.add_argument("--model", type=str, default=None, help="Optional SafeTensors model directory (or .gguf path auto-resolved to SafeTensors+LoRA stack)")
    parser.add_argument("--svon_identity_lora", type=str, default=None, help="Path to Path-A Svon identity surgeon LoRA adapter (or 'default' for sft/qwen_lora_adapter_svon_alphaedit)")
    parser.add_argument("--no_system_prompt", action="store_true", help="Disable system prompt injection to test raw weight identity persistence")
    parser.add_argument("--enable_svon_cgw", action="store_true", help="Upgrade 24 GatedDeltaNet -> Holomorphic CVNN & 8 Attention -> 2-Hop Sheaf-GNN")
    parser.add_argument("--svon_cgw_weights", type=str, default=None, help="Path to trained Svon-CGW .safetensors weights")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind HTTP daemon")
    parser.add_argument("--port", type=int, default=11435, help="Port for Ollama/OpenAI compatible server")
    args = parser.parse_args()

    svon_path = args.svon_identity_lora
    if svon_path == "default":
        svon_path = os.path.join(SFT_DIR, "qwen_lora_adapter_svon_alphaedit")

    if args.cli or args.agent or args.os_agent:
        run_interactive_cli(
            agent_mode=args.agent,
            os_agent_mode=args.os_agent,
            os_url=args.os_url,
            workspace_dir=args.cwd,
            auto_approve=args.auto_approve,
            svon_identity_lora=svon_path,
            no_system_prompt=args.no_system_prompt,
            enable_svon_cgw=args.enable_svon_cgw,
            svon_cgw_weights=args.svon_cgw_weights,
            clarify_enabled=(not args.no_clarify),
            model_path=args.model,
        )
    else:
        eng = SodalsNeurosymbolicEngine.get_instance(
            svon_identity_lora=svon_path,
            no_system_prompt=args.no_system_prompt,
            enable_svon_cgw=args.enable_svon_cgw,
            svon_cgw_weights=args.svon_cgw_weights,
            model_path=args.model,
        )
        eng.interactive_clarify = (not args.no_clarify)
        import uvicorn
        print("=" * 80)
        print(f"Starting SODALS Neurosymbolic Daemon on http://{args.host}:{args.port}")
        print("  - Ollama Chat API   : POST /api/chat")
        print("  - OpenAI Chat API   : POST /v1/chat/completions")
        print("  - Anthropic API     : POST /v1/messages (Claude Code offline)")
        print("  - Arkhe Telemetry   : GET  /api/arkhe/status")
        print("=" * 80)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
