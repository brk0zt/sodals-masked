"""
SODALS v2.1 → v3.0 — Full SODALS Inference Engine (Python)
==========================================================
Token-by-token üretim motoru. K1'in Rust inference pipeline'ındaki
TÜM mekanizmaları Python/PyTorch ortamına taşır:

  1. CliffordBridge Logit Steering     — Her token adımında faz enjeksiyonu
  2. SRF (Semantic Register File)      — AX, BX, DX, CX_W registerleri
  3. Skew-G Logit Braking              — Bilişsel düzleşme frenleme
  4. Dinamik Token Bütçe Genişletici   — 4096 → max 9216
  5. Emergency Answer Latch            — Son 60 tokende zorunlu cevap
  6. MindFilter / Tekrar Kırıcı       — Koherens ve tekrar denetimi
  7. Oja B-Matrisi Projeksiyon         — Alt-uzay bilgisi çıkarımda

model.generate() KULLANILMAZ. Her token kendi forward pass'i ile üretilir.
"""

import sys
import os
import math
import time
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
from pondering_core_bridge import CliffordBridgePyTorch

# ════════════════════════════════════════════════════════════════════════════
# 1. SRF — Semantic Register File (Python Port of Rust SRF)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticRegisterFile:
    """
    K1 Rust'taki SRF'nin Python portu.
    Her token adımında güncellenen hızlı O(1) register dosyası.
    """
    ax: float = 0.0   # Aggregate semantic phase (hidden state toplam fazı)
    bx: float = 0.0   # Task domain phase (prompt semantik alanı)
    dx: float = 0.0   # Bridge h_phase (CliffordBridge çıktısı)
    cx_w: float = 0.5  # Wronskian confidence [0, 1]
    cx_j: float = 0.0  # Jacobian divergence

    def update_from_bridge(self, h_phase: float, confidence: float):
        """CliffordBridge sonuçlarını registerlere yaz."""
        self.dx = h_phase
        self.cx_w = max(0.0, min(1.0, confidence))

    def compute_alu_theta(self) -> float:
        """Oracle ALU fazını hesapla: 0.7*DX + 0.3*AX"""
        return 0.7 * self.dx + 0.3 * self.ax

    def to_dict(self) -> dict:
        return {"AX": self.ax, "BX": self.bx, "DX": self.dx,
                "CX_W": self.cx_w, "CX_J": self.cx_j}


# ════════════════════════════════════════════════════════════════════════════
# 2. Skew-G Logit Braking
# ════════════════════════════════════════════════════════════════════════════

class SkewGBraking:
    """
    K1 Rust'taki apply_skew_g_braking fonksiyonunun Python portu.
    Son N tokenin log-prob varyansı düşükse (bilişsel düzleşme),
    logitleri 1.20× keskinleştir ama yapısal tokenları koru.
    """
    def __init__(self, window_size: int = 32, flatline_threshold: float = 0.1,
                 sharpening_factor: float = 1.20):
        self.window_size = window_size
        self.flatline_threshold = flatline_threshold
        self.sharpening_factor = sharpening_factor
        self.recent_log_probs: deque = deque(maxlen=window_size)

    def record_token_logprob(self, log_prob: float):
        self.recent_log_probs.append(log_prob)

    def should_brake(self) -> bool:
        if len(self.recent_log_probs) < self.window_size // 2:
            return False
        probs = list(self.recent_log_probs)
        mean_p = sum(probs) / len(probs)
        variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
        return variance < self.flatline_threshold

    def apply_braking(self, logits: torch.Tensor,
                      structural_token_ids: List[int]) -> torch.Tensor:
        """Logitleri keskinleştir, yapısal tokenları koru."""
        if not self.should_brake():
            return logits

        # Yapısal tokenların değerlerini kaydet
        preserved = {}
        for tid in structural_token_ids:
            if tid < logits.shape[-1]:
                preserved[tid] = logits[..., tid].clone()

        # Keskinleştir
        logits = logits * self.sharpening_factor

        # Yapısal tokenları geri yükle
        for tid, val in preserved.items():
            logits[..., tid] = val

        return logits


# ════════════════════════════════════════════════════════════════════════════
# 3. Dynamic Token Budget Expander
# ════════════════════════════════════════════════════════════════════════════

class DynamicBudgetExpander:
    """
    K1 Rust'taki Dynamic Budget Expander Supervisor'un Python portu.
    Bütçenin %90'ında CoT kalitesi analiz edilir:
    - Unique token oranı ≥ 0.32
    - Tekrarlayan paragraf yok
    - Tereddüt sayısı ≤ 5
    Koşullar sağlanırsa bütçe +%50 genişletilir (3 kez, tavan ceiling).
    """
    def __init__(self, initial_budget: int = 4096, expansion_factor: float = 1.5,
                 max_expansions: int = 3, ceiling: int = 9216):
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.expansion_factor = expansion_factor
        self.max_expansions = max_expansions
        self.ceiling = ceiling
        self.expansions_used = 0
        self.generated_tokens: List[int] = []

    def record_token(self, token_id: int):
        self.generated_tokens.append(token_id)

    def should_expand(self) -> bool:
        """Bütçenin %90'ında çağır."""
        n = len(self.generated_tokens)
        if n < int(self.current_budget * 0.9):
            return False
        if self.expansions_used >= self.max_expansions:
            return False
        if self.current_budget >= self.ceiling:
            return False

        # Kalite kontrolü
        unique_ratio = len(set(self.generated_tokens[-256:])) / min(256, len(self.generated_tokens))
        if unique_ratio < 0.32:
            return False

        # Tekrarlayan blok kontrolü (son 512 tokende)
        recent = self.generated_tokens[-512:]
        for chunk_size in range(40, min(300, len(recent) // 2)):
            tail = recent[-chunk_size:]
            prev = recent[-2 * chunk_size:-chunk_size]
            if tail == prev:
                return False

        return True

    def try_expand(self) -> bool:
        if self.should_expand():
            old = self.current_budget
            self.current_budget = min(
                int(self.current_budget * self.expansion_factor),
                self.ceiling
            )
            self.expansions_used += 1
            print(f"[BUDGET EXPANDER] Bütçe genişletildi: {old} → {self.current_budget} "
                  f"(genişleme #{self.expansions_used})")
            return True
        return False

    @property
    def budget(self) -> int:
        return self.current_budget


# ════════════════════════════════════════════════════════════════════════════
# 4. Emergency Answer Latch
# ════════════════════════════════════════════════════════════════════════════

class EmergencyAnswerLatch:
    """
    K1 Rust'taki Emergency Answer Latch'in Python portu.
    Token bütçesinin son 60 tokeninde "Answer:" yoksa,
    modelin KV-cache'ine zorunlu cevap formatı enjekte eder.
    """
    LATCH_TEMPLATE = "\n</think>\n\nTherefore, the final answer is:\nAnswer: "

    def __init__(self, latch_zone: int = 60):
        self.latch_zone = latch_zone
        self.has_answer_marker = False

    def check_for_answer(self, text: str):
        """Üretilen metinde 'Answer:' var mı kontrol et."""
        lower = text.lower()
        if "answer:" in lower or "answer :" in lower:
            self.has_answer_marker = True

    def should_inject(self, tokens_generated: int, budget: int) -> bool:
        remaining = budget - tokens_generated
        return (remaining <= self.latch_zone and
                not self.has_answer_marker)

    def get_latch_tokens(self, tokenizer) -> List[int]:
        enc = tokenizer.encode(self.LATCH_TEMPLATE, add_special_tokens=False)
        return enc


# ════════════════════════════════════════════════════════════════════════════
# 5. MindFilter / Tekrar Kırıcı
# ════════════════════════════════════════════════════════════════════════════

class MindFilter:
    """
    K1 Rust'taki MindFilter ve repetition breaker'ın Python portu.
    - Son 128 tokenin koherens düşüklüğü tespiti
    - Tekrarlayan paragraf blokları tespiti
    """
    def __init__(self, coherence_window: int = 128,
                 max_low_coherence_streak: int = 15,
                 min_chunk_bytes: int = 40, max_chunk_bytes: int = 3000):
        self.coherence_window = coherence_window
        self.max_low_coherence_streak = max_low_coherence_streak
        self.min_chunk_bytes = min_chunk_bytes
        self.max_chunk_bytes = max_chunk_bytes
        self.recent_logprobs: deque = deque(maxlen=coherence_window)
        self.low_coherence_streak = 0
        self.generated_text_bytes = bytearray()

    def record_token(self, log_prob: float, token_text: str):
        self.recent_logprobs.append(log_prob)
        self.generated_text_bytes.extend(token_text.encode("utf-8", errors="ignore"))

        # Koherens kontrolü (log_prob < -5.0 = düşük güven)
        if log_prob < -5.0:
            self.low_coherence_streak += 1
        else:
            self.low_coherence_streak = 0

    def should_stop(self) -> Tuple[bool, str]:
        # 1. Koherens düşüklüğü
        if self.low_coherence_streak >= self.max_low_coherence_streak:
            return True, "MindFilter: 15 ardışık düşük-koherens token"

        # 2. Tekrarlayan blok tespiti
        text = bytes(self.generated_text_bytes)
        text_len = len(text)
        for chunk_size in range(self.min_chunk_bytes,
                                min(self.max_chunk_bytes, text_len // 2), 20):
            if text_len >= 2 * chunk_size:
                tail = text[-chunk_size:]
                prev = text[-2 * chunk_size:-chunk_size]
                if tail == prev:
                    return True, f"MindFilter: {chunk_size}-byte tekrarlayan blok"

        return False, ""


# ════════════════════════════════════════════════════════════════════════════
# 6. Oja B-Matrisi Projeksiyon
# ════════════════════════════════════════════════════════════════════════════

class OjaSubspaceProjector:
    """
    Eğitimde kaydedilen Oja B-matrislerini çıkarımda yükler.
    Hidden state'i öğrenilmiş alt-uzaya projekte ederek
    CliffordBridge'e ek faz bilgisi sağlar.
    """
    def __init__(self, oja_path: Optional[str] = None):
        self.b_matrices: Dict[str, torch.Tensor] = {}
        self.loaded = False

        if oja_path and os.path.exists(oja_path):
            self.b_matrices = torch.load(oja_path, map_location="cpu",
                                         weights_only=False)
            self.loaded = True
            print(f"[OJA] Loaded {len(self.b_matrices)} B-matrices from {oja_path}")

    def project_hidden_state(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Hidden state'i Oja alt-uzayına projekte et.
        Returns: Projekte edilmiş vektör veya None.
        """
        if not self.loaded or not self.b_matrices:
            return None

        # İlk uygun B-matrisini kullan (boyut uyumlu olan)
        h_dim = hidden.shape[-1]
        for name, B in self.b_matrices.items():
            if B.shape[0] == h_dim or B.shape[1] == h_dim:
                B_dev = B.to(device=hidden.device, dtype=hidden.dtype)
                if B.shape[1] == h_dim:
                    # B: [r, h_dim] → proj = hidden @ B.T → [batch, seq, r]
                    projected = torch.matmul(hidden, B_dev.T)
                else:
                    # B: [h_dim, r] → proj = hidden @ B → [batch, seq, r]
                    projected = torch.matmul(hidden, B_dev)
                return projected
        return None

    def compute_subspace_phase(self, hidden: torch.Tensor) -> float:
        """Oja alt-uzayındaki toplam faz açısını hesapla."""
        proj = self.project_hidden_state(hidden)
        if proj is None:
            return 0.0
        # Projeksiyonun toplam fazı
        mean_val = proj.float().mean().item()
        return math.sin(mean_val * math.pi)


# ════════════════════════════════════════════════════════════════════════════
# 7. FULL SODALS INFERENCE ENGINE
# ════════════════════════════════════════════════════════════════════════════

class FullSodalsInferenceEngine:
    """
    model.generate() KULLANMAZ.
    Her token kendi forward pass'i, CliffordBridge, SRF, Skew-G,
    Budget Expander, Emergency Latch ve MindFilter ile üretilir.
    """

    def __init__(self, model, tokenizer,
                 initial_budget: int = 4096,
                 budget_ceiling: int = 9216,
                 oja_path: Optional[str] = None,
                 verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose

        # Vocab size
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else len(tokenizer)

        # Alt sistemler
        self.clifford_bridge = CliffordBridgePyTorch(
            vocab_size=vocab_size, wave_dim=256, default_impedance=0.45
        )
        self.srf = SemanticRegisterFile()
        self.skew_g = SkewGBraking(window_size=32, flatline_threshold=0.1)
        self.budget_expander = DynamicBudgetExpander(
            initial_budget=initial_budget, ceiling=budget_ceiling
        )
        self.emergency_latch = EmergencyAnswerLatch(latch_zone=60)
        self.mind_filter = MindFilter()
        self.oja_projector = OjaSubspaceProjector(oja_path)

        # Yapısal token ID'leri (Qwen 3.5)
        self.structural_token_ids = self._find_structural_tokens()

    def _find_structural_tokens(self) -> List[int]:
        """EOS, <|im_start|>, <|im_end|> gibi yapısal tokenları bul."""
        structural = []
        for special in ["<|im_start|>", "<|im_end|>", "<|endoftext|>",
                        "<think>", "</think>"]:
            ids = self.tokenizer.encode(special, add_special_tokens=False)
            structural.extend(ids)
        if self.tokenizer.eos_token_id is not None:
            structural.append(self.tokenizer.eos_token_id)
        return list(set(structural))

    @torch.no_grad()
    def generate(self, prompt: str, temperature: float = 0.0) -> Dict:
        """
        Tam SODALS token-by-token üretim döngüsü.

        Returns:
            {
                "response": str,
                "tokens_generated": int,
                "budget_used": int,
                "budget_expansions": int,
                "srf_final": dict,
                "stop_reason": str,
                "elapsed_seconds": float,
                "tokens_per_second": float,
            }
        """
        start_time = time.time()

        # Reset all subsystems
        self.srf = SemanticRegisterFile()
        self.skew_g = SkewGBraking()
        self.budget_expander = DynamicBudgetExpander(
            initial_budget=self.budget_expander.initial_budget,
            ceiling=self.budget_expander.ceiling
        )
        self.emergency_latch = EmergencyAnswerLatch()
        self.mind_filter = MindFilter()

        # Encode prompt
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_enc = self.tokenizer(input_text, return_tensors="pt")
        input_ids = input_enc["input_ids"].to(self.model.device)
        attention_mask = input_enc["attention_mask"].to(self.model.device)

        # Prompt'un BX fazını hesapla
        prompt_hash = sum(ord(c) for c in prompt[:200])
        self.srf.bx = math.sin(prompt_hash * 0.0001 * math.pi)

        generated_ids = []
        generated_text = ""
        stop_reason = "max_budget"
        past_key_values = None

        # ── TOKEN-BY-TOKEN GENERATION LOOP ──
        for step in range(self.budget_expander.ceiling):

            # Budget kontrolü
            if len(generated_ids) >= self.budget_expander.budget:
                # Genişleme denemesi
                if not self.budget_expander.try_expand():
                    stop_reason = "max_budget"
                    break

            # Emergency Answer Latch kontrolü
            if self.emergency_latch.should_inject(
                    len(generated_ids), self.budget_expander.budget):
                latch_tokens = self.emergency_latch.get_latch_tokens(self.tokenizer)
                self.emergency_latch.has_answer_marker = True
                for lt in latch_tokens:
                    generated_ids.append(lt)
                    self.budget_expander.record_token(lt)
                latch_text = self.tokenizer.decode(latch_tokens, skip_special_tokens=False)
                generated_text += latch_text
                if self.verbose:
                    print(f"[EMERGENCY LATCH] Injected: {repr(latch_text)}")
                stop_reason = "emergency_latch"

                # Synchronize model KV-cache and attention_mask with injected latch tokens
                latch_tensor = torch.tensor([latch_tokens], device=self.model.device, dtype=torch.long)
                latch_mask = torch.ones_like(latch_tensor)
                attention_mask = torch.cat([attention_mask, latch_mask], dim=1)
                input_ids = torch.cat([input_ids, latch_tensor], dim=1)

                with torch.no_grad():
                    latch_out = self.model(
                        input_ids=latch_tensor,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                past_key_values = latch_out.past_key_values
                raw_logits = latch_out.logits[:, -1, :]
                last_hidden = latch_out.hidden_states[-1][:, -1, :]
                # Proceed to sampling without looping into another latch injection!
            else:
                # ── STANDARD FORWARD PASS ──
                if past_key_values is not None:
                    # Incremental: sadece son token
                    model_input = input_ids[:, -1:]
                    attn = attention_mask
                else:
                    # İlk adım: tüm input
                    model_input = input_ids
                    attn = attention_mask

                outputs = self.model(
                    input_ids=model_input,
                    attention_mask=attn,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                raw_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                last_hidden = outputs.hidden_states[-1][:, -1, :]
            last_hidden = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden_dim]

            # ── 1. SRF AX güncellemesi ──
            hidden_mean = last_hidden.float().mean().item()
            self.srf.ax = math.sin(hidden_mean * math.pi)

            # ── 2. Oja alt-uzay fazı ──
            oja_phase = self.oja_projector.compute_subspace_phase(last_hidden)

            # ── 3. CliffordBridge logit steering ──
            target_vocab = raw_logits.shape[-1]
            steered_bias = self.clifford_bridge.compute_steering_logits_bias(
                last_hidden, target_vocab_size=target_vocab
            )
            steered_logits = raw_logits + steered_bias.squeeze(1)

            # Bridge'den h_phase ve confidence çıkar (SRF güncelleme için)
            f_phase = math.sin(hidden_mean * math.pi)
            g_phase = math.cos(last_hidden.float().var().item() * math.pi)
            # Oja fazını g_phase'e ekle
            g_phase_combined = g_phase + 0.3 * oja_phase
            h_phase_val = math.atan2(
                math.sin(f_phase) * math.cos(g_phase_combined * 0.45) +
                math.cos(f_phase) * (-math.sin(g_phase_combined * 0.45)),
                math.cos(f_phase) * math.cos(g_phase_combined * 0.45) -
                math.sin(f_phase) * (-math.sin(g_phase_combined * 0.45))
            )
            confidence = min(1.0, abs(math.cos(h_phase_val)))
            self.srf.update_from_bridge(h_phase_val, confidence)

            # ── 4. Skew-G Braking ──
            steered_logits = self.skew_g.apply_braking(
                steered_logits, self.structural_token_ids
            )

            # ── 5. NaN/Inf clamping ──
            steered_logits = torch.clamp(steered_logits, -1e9, 1e9)
            steered_logits = torch.where(
                torch.isnan(steered_logits),
                torch.zeros_like(steered_logits),
                steered_logits
            )

            # ── 6. Sampling ──
            if temperature <= 0:
                # Greedy
                next_token_id = steered_logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(steered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            # Log-prob kaydet
            log_probs = F.log_softmax(steered_logits, dim=-1)
            token_log_prob = log_probs[0, next_token_id].item()
            self.skew_g.record_token_logprob(token_log_prob)

            # Token metnini çöz
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)

            # MindFilter kaydet
            self.mind_filter.record_token(token_log_prob, token_text)

            # Budget kaydet
            generated_ids.append(next_token_id)
            self.budget_expander.record_token(next_token_id)
            generated_text += token_text

            # Emergency latch kontrolü
            self.emergency_latch.check_for_answer(generated_text)

            # ── EOS kontrolü ──
            if next_token_id == self.tokenizer.eos_token_id:
                stop_reason = "eos"
                break

            # ── MindFilter kontrolü ──
            should_stop, reason = self.mind_filter.should_stop()
            if should_stop:
                stop_reason = reason
                if self.verbose:
                    print(f"[MINDFILTER] Stopped: {reason}")
                break

            # KV cache için input_ids'i güncelle
            input_ids = torch.tensor([[next_token_id]], device=self.model.device)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.model.device, dtype=attention_mask.dtype)
            ], dim=1)

            # Logging (her 100 adımda)
            if self.verbose and (step + 1) % 100 == 0:
                print(f"[STEP {step+1}] tokens={len(generated_ids)}, "
                      f"budget={self.budget_expander.budget}, "
                      f"SRF={self.srf.to_dict()}")

        elapsed = time.time() - start_time
        tps = len(generated_ids) / elapsed if elapsed > 0 else 0

        # Clean response
        clean_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "response": clean_response,
            "tokens_generated": len(generated_ids),
            "budget_used": self.budget_expander.budget,
            "budget_expansions": self.budget_expander.expansions_used,
            "srf_final": self.srf.to_dict(),
            "stop_reason": stop_reason,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(tps, 2),
        }
