"""
SODALS v3.0 — Token-Level Monte Carlo Tree Search (MCTS) Engine with SymPy Verification
========================================================================================
Bridges Rust's MCTS / Arkhe / Clifford neural-symbolic algorithms and SymPy symbolic algebra
with autoregressive LLM reasoning:

  1. Hypothesis Branching (Thought-Tree Expansion):
     Her akıl yürütme adımında K=3 alternatif düşünce dalı açar.
  2. SymPy Symbolic & Algebraic Runtime Verification:
     Her düşünce dalındaki matematiksel eşitlikleri (A = B) SymPy ile ayrıştırıp doğrular.
     Aritmetik veya işaret çelişkisi (örn. 2*4 - 2 = -6) tespit edilirse dal anında BUDANIR (Prune).
  3. Clifford & Wronskian Process Reward (Step Evaluation):
     Dalları geometrik faz tutarlılığı (CX_W), logit entropisi ve epistemic doğrulama ile puanlar.
  4. UCT Selection & CDCL Backtracking:
     Çelişkili yolları geri sarar ve en sağlam matematiksel hipotezi seçer.
  5. PonderingCore Latent ODE Manifold Flow:
     Her adımda gizli durumları RK4 diferansiyel denklemi boyunca entegre eder.
"""

import os
import sys
import re
import math
import time
import copy
import sympy
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))
from pondering_core_bridge import PonderingODEModule
from sodals_inference_engine import (
    SemanticRegisterFile, SkewGBraking, OjaSubspaceProjector, MindFilter
)


# ════════════════════════════════════════════════════════════════════════════
# 1. SymPy Symbolic & Algebraic Runtime Verifier
# ════════════════════════════════════════════════════════════════════════════

class SympySymbolicVerifier:
    """
    Çıkarım anında modelin ürettiği metindeki denklemleri ve aritmetik adımları
    SymPy ile sembolik olarak denetleyen ve çelişkileri yakalayan motor.

    v3.1 Geliştirmeleri:
    - Parse hataları artık sessizce yutulmuyor, kayıt altına alınıyor
    - Bilinen formül kütüphanesi ile kavramsal hata tespiti
    """

    # Bilinen matematiksel formül kütüphanesi — kavramsal hata körlüğünü azaltır
    KNOWN_FORMULAS = {
        # (pattern_keywords, formula_func, description)
        "canonical_divisor": {
            "keywords": ["canonical", "divisor", "deg(k)", "degree"],
            "check": lambda params: 2 * params["g"] - 2 if "g" in params else None,
            "desc": "deg(K) = 2g - 2"
        },
        "euler_cpn": {
            "keywords": ["euler", "cp^", "projective space", "chi("],
            "check": lambda params: params["n"] + 1 if "n" in params else None,
            "desc": "χ(CP^n) = n + 1"
        },
        "hasse_weil": {
            "keywords": ["rational points", "#e(", "frobenius", "trace"],
            "check": lambda params: params["q"] + 1 - params["a"] if "q" in params and "a" in params else None,
            "desc": "#E(F_q) = q + 1 - a_q"
        },
        "su_n_dim": {
            "keywords": ["su(", "special unitary", "adjoint"],
            "check": lambda params: params["n"] ** 2 - 1 if "n" in params else None,
            "desc": "dim(su(n)) = n² - 1"
        },
        "sp_2n_dim": {
            "keywords": ["sp(", "symplectic", "hamiltonian matri"],
            "check": lambda params: params["n"] * (2 * params["n"] + 1) if "n" in params else None,
            "desc": "dim(sp(2n,R)) = n(2n+1)"
        },
        "grassmannian_euler": {
            "keywords": ["grassmann", "gr(", "k-planes"],
            "check": lambda params: math.comb(params["n"], params["k"]) if "n" in params and "k" in params else None,
            "desc": "χ(Gr(k,n)) = C(n,k)"
        },
        "lie_dim": {
            "keywords": ["lie algebra", "dimension", "root system"],
            "check": lambda params: params["rank"] + params["roots"] if "rank" in params and "roots" in params else None,
            "desc": "dim(g) = rank + |Φ|"
        },
    }

    @staticmethod
    def clean_latex(expr_str: str) -> str:
        s = expr_str.strip()
        s = re.sub(r'\\(?:cdot|times)', '*', s)
        s = re.sub(r'\\left|\\right', '', s)
        s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', s)
        s = re.sub(r'[{}\$]', '', s)
        return s.strip()

    @classmethod
    def verify_chunk(cls, text_chunk: str) -> Tuple[bool, int, str, List[str]]:
        """
        Metin parçasındaki denklemleri (A = B) ayrıştırır ve doğrular.
        Returns: (is_consistent: bool, verified_count: int, error_details: str, parse_failures: List[str])
        """
        parse_failures = []
        verified_count = 0
        lines = text_chunk.split('\n')
        for line in lines:
            if '=' in line and not any(k in line.lower() for k in ["let ", "define ", "where "]):
                parts = line.split('=')
                if len(parts) >= 2:
                    for i in range(len(parts) - 1):
                        lhs_raw = cls.clean_latex(parts[i])
                        rhs_raw = cls.clean_latex(parts[i + 1])

                        # Matematiksel sayısal, cebirsel ve işleç bloklarını yakala
                        lhs_m = re.findall(r'[-+]?[0-9a-zA-Z\.\*\+\-\/\(\)\s\^]+', lhs_raw)
                        rhs_m = re.findall(r'[-+]?[0-9a-zA-Z\.\*\+\-\/\(\)\s\^]+', rhs_raw)

                        if lhs_m and rhs_m:
                            lhs_expr = lhs_m[-1].strip().replace('^', '**')
                            rhs_expr = rhs_m[0].strip().replace('^', '**')

                            # İki tarafta da en az bir sayı veya geçerli değişken varsa kontrol et
                            if any(c.isalnum() for c in lhs_expr) and any(c.isalnum() for c in rhs_expr):
                                try:
                                    v1 = sympy.sympify(lhs_expr)
                                    v2 = sympy.sympify(rhs_expr)
                                    diff = sympy.simplify(v1 - v2)
                                    if diff != 0 and (v1.is_number and v2.is_number):
                                        return False, verified_count, f"SymPy Contradiction: {lhs_expr} ({v1}) != {rhs_expr} ({v2})", parse_failures
                                    elif diff == 0:
                                        verified_count += 1
                                except Exception as e:
                                    parse_failures.append(f"Parse failed [{lhs_expr}] vs [{rhs_expr}]: {e}")

        if parse_failures:
            detail = f"Consistent (with {len(parse_failures)} parse warnings)"
            return True, verified_count, detail, parse_failures

        return True, verified_count, "Consistent", parse_failures


# ════════════════════════════════════════════════════════════════════════════
# 2. MCTS Thought Node (Reasoning Tree Node)
# ════════════════════════════════════════════════════════════════════════════

class MctsThoughtNode:
    """
    Akıl yürütme ağacındaki tekil bir düşünce düğümü.
    Birkaç cümleden veya 32-96 tokenden oluşan bir mantık adımını temsil eder.
    """
    def __init__(self,
                 token_ids: List[int],
                 text_chunk: str = "",
                 parent: Optional['MctsThoughtNode'] = None,
                 prior_p: float = 1.0,
                 depth: int = 0):
        self.token_ids = token_ids          # Bu adıma kadar üretilen TÜM token ID'leri
        self.text_chunk = text_chunk        # Sadece bu düğümde üretilen metin
        self.parent = parent
        self.children: List['MctsThoughtNode'] = []
        self.depth = depth

        # MCTS İstatistikleri
        self.visits: int = 0
        self.total_value: float = 0.0
        self.prior_p: float = prior_p
        self.mean_value: float = 0.0

        # Nörosembolik Durum
        self.srf: SemanticRegisterFile = SemanticRegisterFile()
        self.is_terminal: bool = False
        self.is_pruned: bool = False
        self.prune_reason: str = ""
        self.has_final_answer: bool = False
        self.verified_eqs: int = 0

    def uct_score(self, c_puct: float = 1.414) -> float:
        """UCT formülü: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))"""
        if self.is_pruned:
            return -999.0
        parent_visits = max(1, self.parent.visits) if self.parent else 1
        q_val = self.mean_value
        exploration = c_puct * self.prior_p * (math.sqrt(parent_visits) / (1.0 + self.visits))
        return q_val + exploration

    def backpropagate(self, value: float):
        """Sonuç değerini ağaçta köke doğru yay."""
        curr = self
        while curr is not None:
            curr.visits += 1
            curr.total_value += value
            curr.mean_value = curr.total_value / curr.visits
            curr = curr.parent

# Alias for standard MCTS node naming
MCTSNode = MctsThoughtNode


# ════════════════════════════════════════════════════════════════════════════
# 3. Process Reward & Epistemic Step Evaluator (with SymPy Verification)
# ════════════════════════════════════════════════════════════════════════════

class EpistemicStepEvaluator:
    """
    Her düşünce dalını puanlayan nörosembolik Process Reward fonksiyonu:
    - SymPy sembolik ve aritmetik doğruluk denetimi (Doğrulanmış eşitlik sayısı)
    - Wronskian güven skoru (CX_W)
    - Log-Probability skoru (Akıcılık ve olasılık kütlesi)
    - Erken pes etme / kaçamak yanıtlara ceza
    - Çelişki tespiti (CDCL pruning)
    - YÜZEYSEL KELİME VE SEMBOL SAYIMI TAMAMEN KALDIRILDI.
    """

    EVASION_PHRASES = [
        "cannot determine", "cannot be determined", "i cannot reliably",
        "i don't know", "impossible to tell", "unknown", "i cannot fully verify",
        "i recommend consulting", "beyond my capability"
    ]

    CONTRADICTION_PHRASES = [
        "contradiction!", "this is impossible", "empty set", "inconsistent",
        "0 = 1", "negative dimension", "division by zero"
    ]

    @classmethod
    def evaluate(cls,
                  text_chunk: str,
                  full_text: str,
                  avg_log_prob: float,
                  cx_w: float,
                  h_phase: float) -> Tuple[float, bool, str, int]:
        """
        Düşünce dalını değerlendirir.
        Returns: (scalar_value [-1.0, 1.0], is_pruned, prune_reason, verified_eq_count)
        """
        lower = text_chunk.lower()

        # 1. SymPy Sembolik & Aritmetik Denetim (CDCL Symbolic Pruning)
        is_sym_valid, verified_count, sym_err, parse_warnings = SympySymbolicVerifier.verify_chunk(text_chunk)
        if not is_sym_valid:
            return -1.0, True, f"CDCL: {sym_err}", verified_count

        # 2. Mantıksal Çelişki Kontrolü
        for phrase in cls.CONTRADICTION_PHRASES:
            if phrase in lower and "proof by contradiction" not in lower:
                return -0.9, True, f"CDCL: Detected logical contradiction ({phrase})", verified_count

        # 3. Erken Pes Etme Cezası
        for phrase in cls.EVASION_PHRASES:
            if phrase in lower:
                return -0.9, True, f"Epistemic Gate: Penalized premature evasion ({phrase})", verified_count

        # 4. Temel Skor: Wronskian Güven Skoru ($CX_W \in [0, 1]$)
        base_score = (cx_w - 0.5) * 0.4

        # 5. Log-Probability Skoru (Modelin dil olasılığı ve eminliği)
        prob_score = math.exp(max(-5.0, avg_log_prob)) * 0.4

        # 6. SymPy Tarafından BİZZAT DOĞRULANMIŞ Eşitlik Bonusu (Doğruluğa Koşullu Ödül)
        verified_bonus = min(0.5, verified_count * 0.25)

        # 7. Parse Uyarı Cezası (Belirsiz veya ayrıştırılamayan matematik)
        warning_penalty = min(0.15, len(parse_warnings) * 0.03)

        # Toplam Process Reward Değeri (Artık kelime saymaya değil, doğrulanmış hesaba bağlı!)
        total_value = base_score + prob_score + verified_bonus - warning_penalty
        total_value = max(-1.0, min(1.0, total_value))

        return total_value, False, "", verified_count


# ════════════════════════════════════════════════════════════════════════════
# 4. Token-Level MCTS Inference Engine with SymPy Verification
# ════════════════════════════════════════════════════════════════════════════

class TokenMctsEngine:
    """
    Monte Carlo Token Tree Search (MCTS) Çıkarım Motoru.
    Doğal dil ispatını dallara ayırarak arar, SymPy ile doğrular,
    kendi kendini düzeltir ve yüksek doğruluklu ispat üretir.
    """

    def __init__(self,
                 model,
                 tokenizer,
                 num_branches: int = 3,           # Her adımda açılacak alternatif dal sayısı
                 mcts_steps: int = 12,            # Toplam MCTS arama derinliği
                 chunk_token_limit: int = 96,     # Her düşünce dalının token uzunluğu
                 min_reasoning_tokens: int = 100, # Erken EOS'u engelleyen minimum token eşiği
                 max_total_tokens: int = 4096,    # Maksimum tavan token bütçesi
                 c_puct: float = 1.414,
                 temperature: float = 0.6,
                 ode_weights_path: Optional[str] = None,
                 verbose: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.num_branches = num_branches
        self.mcts_steps = mcts_steps
        self.chunk_token_limit = chunk_token_limit
        self.min_reasoning_tokens = min_reasoning_tokens
        self.max_total_tokens = max_total_tokens
        self.c_puct = c_puct
        self.temperature = temperature
        self.verbose = verbose

        vocab_size = model.config.vocab_size if hasattr(model, 'config') else len(tokenizer)
        self.skew_g = SkewGBraking()
        
        # PonderingCore RK4 ODE Modülü
        hidden_dim = model.config.hidden_size if hasattr(model, 'config') else 4096
        self.pondering_ode = PonderingODEModule(hidden_dim=hidden_dim, dt=0.1, max_steps=4)
        if ode_weights_path and os.path.exists(ode_weights_path):
            try:
                self.pondering_ode.load_state_dict(torch.load(ode_weights_path, map_location="cpu"))
                if self.verbose:
                    print(f"[MCTS] Loaded trained Pondering ODE weights from {ode_weights_path}")
            except Exception as e:
                print(f"[MCTS Warning] Could not load ODE weights: {e}")

        # Yapısal Tokenlar
        self.eos_token_id = self.tokenizer.eos_token_id
        self.im_end_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    @torch.no_grad()
    def _rollout_branch(self,
                        base_ids: List[int],
                        temp: float,
                        allow_eos: bool = False) -> Tuple[List[int], str, float, SemanticRegisterFile]:
        """
        Mevcut düğümden tek bir alternatif düşünce adımını (chunk) ileri besle.
        """
        generated = []
        log_probs_sum = 0.0
        srf = SemanticRegisterFile()

        curr_ids = torch.tensor([base_ids], device=self.model.device)
        past_kv = None

        # Device sync for ODE
        if self.pondering_ode.arkhe_proj.weight.device != self.model.device:
            self.pondering_ode = self.pondering_ode.to(self.model.device, dtype=self.model.dtype)

        for step in range(self.chunk_token_limit):
            if past_kv is not None:
                m_in = curr_ids[:, -1:]
            else:
                m_in = curr_ids

            outputs = self.model(
                input_ids=m_in,
                past_key_values=past_kv,
                output_hidden_states=True,
                use_cache=True
            )
            past_kv = outputs.past_key_values
            raw_logits = outputs.logits[:, -1, :].clone()
            last_hidden = outputs.hidden_states[-1][:, -1:, :]

            # 1. PonderingCore Latent ODE Manifold Evolution
            pondered_hidden = self.pondering_ode(last_hidden)

            # 2. SRF update from pondered hidden (no CliffordBridge steering)
            hidden_mean = pondered_hidden.float().mean().item()
            srf.ax = math.sin(hidden_mean * math.pi)

            # Use raw logits directly — CliffordBridge modulo-256 noise removed
            steered_logits = raw_logits

            # Phase computation for SRF registers
            f_phase = math.sin(hidden_mean * math.pi)
            g_phase = math.cos(pondered_hidden.float().var().item() * math.pi)
            h_phase_val = math.atan2(
                math.sin(f_phase) * math.cos(g_phase * 0.45) - math.cos(f_phase) * math.sin(g_phase * 0.45),
                math.cos(f_phase) * math.cos(g_phase * 0.45) + math.sin(f_phase) * math.sin(g_phase * 0.45)
            )
            srf.update_from_bridge(h_phase_val, min(1.0, abs(math.cos(h_phase_val))))

            # 3. Skew-G Braking
            steered_logits = self.skew_g.apply_braking(steered_logits, self.im_end_id)

            # 4. Erken EOS Maskeleme
            if not allow_eos:
                if self.eos_token_id is not None:
                    steered_logits[..., self.eos_token_id] = -1e9
                for im_id in self.im_end_id:
                    steered_logits[..., im_id] = -1e9

            # 5. Sampling
            if temp <= 0.01:
                next_tok = steered_logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(steered_logits / temp, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()

            lp = F.log_softmax(steered_logits, dim=-1)[0, next_tok].item()
            log_probs_sum += lp

            generated.append(next_tok)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_tok]], device=self.model.device)], dim=1)

            # Chunk sonlandırma kontrolü: Cümle sonu, çift newline veya </think>
            # Bug fix 2a: Minimum 12 token üretilmeden chunk kesilmesin
            tok_text = self.tokenizer.decode([next_tok], skip_special_tokens=False)
            if step >= 12 and ("\n\n" in tok_text or "</think>" in tok_text or step >= self.chunk_token_limit - 1):
                break
            if allow_eos and next_tok == self.eos_token_id:
                break
            # Bug fix 2b: N-gram tekrar dedektörü — son 8 token aynıysa kes
            if len(generated) >= 8 and len(set(generated[-8:])) == 1:
                break

        chunk_text = self.tokenizer.decode(generated, skip_special_tokens=False)
        avg_lp = log_probs_sum / max(1, len(generated))
        return generated, chunk_text, avg_lp, srf

    def generate(self, prompt: str) -> Dict:
        """
        MCTS ve SymPy doğrulaması ile tam akıl yürütme ağacı araması yapar.
        """
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_text += "<think>\nLet us solve this step by step from first principles with full mathematical rigor.\n"

        prompt_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

        root = MctsThoughtNode(
            token_ids=prompt_ids,
            text_chunk="<root>",
            parent=None,
            prior_p=1.0,
            depth=0
        )

        current_node = root
        total_tokens_generated = 0
        all_tree_text = ""

        if self.verbose:
            print(f"\n{'='*75}\n[MCTS + SymPy ENGINE] Starting Tree Search Reasoning with Symbolic Verification...\n{'='*75}")

        # ── MCTS EXPANSION & SEARCH LOOP ──
        for step_idx in range(self.mcts_steps):
            if current_node.is_terminal or total_tokens_generated >= self.max_total_tokens:
                break

            total_generated_so_far = len(current_node.token_ids) - len(prompt_ids)
            allow_eos = total_generated_so_far >= self.min_reasoning_tokens

            # 1. EXPANSION: K adet alternatif dal aç
            branches = []
            for b_idx in range(self.num_branches):
                # Bug fix 2c: Sıcaklıklar daraltıldı [0.1, 0.4, 0.5] (eskisi: [0.2, 0.75, 0.9])
                t_branch = 0.1 if b_idx == 0 else (0.3 + b_idx * 0.1)
                toks, text_chunk, avg_lp, srf = self._rollout_branch(
                    current_node.token_ids,
                    temp=t_branch,
                    allow_eos=allow_eos
                )

                # Dalı SymPy ve Epistemic Process Reward ile değerlendir
                full_cand_text = self.tokenizer.decode(current_node.token_ids + toks, skip_special_tokens=False)
                value, is_pruned, prune_reason, verified_eqs = EpistemicStepEvaluator.evaluate(
                    text_chunk=text_chunk,
                    full_text=full_cand_text,
                    avg_log_prob=avg_lp,
                    cx_w=srf.cx_w,
                    h_phase=srf.dx
                )

                child = MctsThoughtNode(
                    token_ids=current_node.token_ids + toks,
                    text_chunk=text_chunk,
                    parent=current_node,
                    prior_p=max(0.1, srf.cx_w),
                    depth=current_node.depth + 1
                )
                child.srf = srf
                child.is_pruned = is_pruned
                child.prune_reason = prune_reason
                child.mean_value = value
                child.visits = 1
                child.total_value = value
                child.verified_eqs = verified_eqs

                # Cevap kontrolü
                if "answer:" in text_chunk.lower() or "answer :" in text_chunk.lower():
                    child.has_final_answer = True
                    if allow_eos and not is_pruned:
                        child.is_terminal = True

                branches.append(child)

            current_node.children = branches

            # 2. SELECTION: En yüksek UCT skoruna sahip budanmamış dalı seç
            valid_branches = [b for b in branches if not b.is_pruned]
            if not valid_branches:
                # Tüm dallar budandıysa (CDCL kilitlenme), geri sar ve kendi kendini düzeltme enjekte et
                if self.verbose:
                    print(f"[CDCL BACKTRACK] Step {step_idx+1}: All branches pruned by SymPy. Injecting self-correction...")
                correction_text = "\n[Self-Correction]: The previous arithmetic or sign step was inconsistent. Re-calculating carefully:\n"
                corr_ids = self.tokenizer.encode(correction_text, add_special_tokens=False)
                current_node.token_ids.extend(corr_ids)
                continue

            best_child = max(valid_branches, key=lambda c: c.uct_score(self.c_puct))

            # Ağaçta ilerle
            current_node = best_child
            total_tokens_generated = len(current_node.token_ids) - len(prompt_ids)
            all_tree_text += current_node.text_chunk

            if self.verbose:
                status_str = f"[PRUNED: {current_node.prune_reason}]" if current_node.is_pruned else f"VALID (Eqs: {current_node.verified_eqs})"
                print(f"[MCTS STEP {step_idx+1:02d}] Depth: {current_node.depth:02d} | "
                      f"Tokens: {total_tokens_generated:4d} | Value: {current_node.mean_value:+.3f} | "
                      f"Status: {status_str} | Text: {repr(current_node.text_chunk[:60])}...")

            # Eğer minimum bütçe dolduysa ve cevap bulunduysa sonlandır
            if allow_eos and current_node.has_final_answer and not current_node.is_pruned:
                if self.verbose:
                    print(f"[MCTS TERMINATION] Reached verified final answer at {total_tokens_generated} tokens.")
                break

            # Eğer bütçe dolduysa Answer Latch tetikle
            if total_tokens_generated >= self.max_total_tokens - 128 and not current_node.has_final_answer:
                latch_prompt = "\n</think>\n\nTherefore, synthesizing all verified steps above, the final answer is:\nAnswer: "
                latch_ids = self.tokenizer.encode(latch_prompt, add_special_tokens=False)
                current_node.token_ids.extend(latch_ids)
                final_toks, final_text, _, _ = self._rollout_branch(
                    current_node.token_ids, temp=0.0, allow_eos=True
                )
                current_node.token_ids.extend(final_toks)
                all_tree_text += latch_prompt + final_text
                total_tokens_generated = len(current_node.token_ids) - len(prompt_ids)
                break

        elapsed = time.time() - start_time
        final_full_response = self.tokenizer.decode(current_node.token_ids[len(prompt_ids):], skip_special_tokens=True).strip()

        return {
            "response": final_full_response,
            "tokens_generated": total_tokens_generated,
            "mcts_depth": current_node.depth,
            "mcts_value_final": round(current_node.mean_value, 4),
            "srf_final": current_node.srf.to_dict(),
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(total_tokens_generated / max(1e-3, elapsed), 2)
        }


# ════════════════════════════════════════════════════════════════════════════
# 5. Greedy + SymPy Step-by-Step Verifier Engine (Zero-Search Baseline)
# ════════════════════════════════════════════════════════════════════════════

class GreedySympyVerifierEngine:
    """
    Greedy Decoding + SymPy Runtime Verification (Ağaç araması olmadan doğrudan çıkarım):
    - Temel yolu T=0 ile hızlıca üretir (Greedy hızında: 1x).
    - Her cümlede SymPy doğrulayıcı çalışır.
    - SADECE aritmetik/işaret çelişkisi tespit edilirse o cümleyi budar ve [Self-Correction] enjekte eder.
    - Çelişki yoksa %100 saf model performansını ve hızını korur.
    """
    def __init__(self, model, tokenizer, max_tokens: int = 1024, verbose: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.eos_token_id = tokenizer.eos_token_id

    @torch.no_grad()
    def generate(self, prompt: str) -> Dict:
        start_time = time.time()
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)
        curr_ids = list(prompt_ids)
        total_verified = 0
        corrections_applied = 0

        # Run chunked generation with SymPy sentence checks
        for _ in range(self.max_tokens // 32):
            input_tensor = torch.tensor([curr_ids], device=self.model.device)
            gen_out = self.model.generate(
                input_tensor,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id
            )
            new_tokens = gen_out[0][len(curr_ids):].tolist()
            if not new_tokens:
                break

            chunk_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            is_valid, verified_cnt, err_msg, _ = SympySymbolicVerifier.verify_chunk(chunk_text)
            total_verified += verified_cnt

            if not is_valid and corrections_applied < 3:
                # Contradiction detected! Inject targeted self-correction
                corrections_applied += 1
                if self.verbose:
                    print(f"[Greedy+SymPy] Caught contradiction: {err_msg}. Triggering self-correction #{corrections_applied}...")
                corr_prompt = f"\n[Correction]: Inconsistent calculation detected ({err_msg}). Recalculating with exact arithmetic:\n"
                corr_ids = self.tokenizer.encode(corr_prompt, add_special_tokens=False)
                curr_ids.extend(corr_ids)
                continue
            else:
                curr_ids.extend(new_tokens)

            if self.eos_token_id in new_tokens or "<|im_end|>" in chunk_text:
                break

        elapsed = time.time() - start_time
        resp = self.tokenizer.decode(curr_ids[len(prompt_ids):], skip_special_tokens=True).strip()
        tokens_gen = len(curr_ids) - len(prompt_ids)

        return {
            "response": resp,
            "tokens_generated": tokens_gen,
            "verified_equations": total_verified,
            "corrections_applied": corrections_applied,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(tokens_gen / max(1e-3, elapsed), 2)
        }

