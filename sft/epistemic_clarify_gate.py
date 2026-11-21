"""
SODALS / Svon — Epistemic Clarification Gate, Precondition Graph & Entropy Watchdog
====================================================================================
Implements the audited v2 cognitive clarification control layer connecting SODALS's
empirically validated internal signals and architecture principles without breaking
Epoch I frozen modules (`arkhe_neurosymbolic_pipeline.py`, `cst_symbolic_verifier.py`):

  1. `SemanticPreconditionGraph` (`CST` + Arkhe 6 `ConditionsArche`):
     < 0.5 ms precondition coverage scoring across task domains (travel/route,
     interactive math, offline coding/automation agent, planning/design). Avoids
     token bloat on clear prompts and populates workspace file options for agents.
  2. `AutonomyGate` (`autonomy_gate.rs` port):
     3-regime Z-score gate (`FULL` / `DEFERRED` / `HALT_AND_PROMPT`) governing
     thinking activation, compact slot-aware protocol injection, and fallback freeze.
  3. `EntropyAndSkewGProcessor` (`LogitsProcessor`):
     Empirically backed by `eval_internal_signals_logprob_vs_probe_results.json`
     (`neg_mean_entropy` AUROC 0.7583 silent failures / 0.8791 OOD). Tracks
     Shannon entropy & Top1-Top2 margin in O(1) memory, protects structural JSON/XML
     tokens under `SkewGBraking`, and trips the epistemic fallback watchdog if the
     model forgets `<sodals_ask>` while remaining highly uncertain.
  4. `ClarificationConstraintLock` (`fluid_algebra.rs` Viscosity / Key-Lock):
     Locks user-resolved slot values, eliminates unselected branch options via
     impedance constraints, blocks duplicate loops, and verifies adherence.
  5. `offload_cache_to_cpu` / `restore_cache_to_device`:
     Moves frozen Qwen 3.5 `DynamicCache` (Attention KV + GatedDeltaNet recurrent
     & conv states) to CPU RAM during human wait and restores to GPU on unfreeze.
"""

from __future__ import annotations

import copy
import math
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import torch

try:
    from transformers import LogitsProcessor
except Exception:  # pragma: no cover - fallback for lightweight environments
    class LogitsProcessor:  # type: ignore[no-redef]
        pass


class AutonomyLevel(Enum):
    """Three-regime physics autonomy gate aligned with `src/engine/autonomy_gate.rs`."""

    FULL = "FULL"
    DEFERRED = "DEFERRED"
    HALT_AND_PROMPT = "HALT_AND_PROMPT"


@dataclass(frozen=True)
class ClarificationQuestionSpec:
    """Single slot question inside a single- or multi-slot clarification request."""

    slot_id: str
    question: str
    options: Tuple[str, ...] = ()
    allow_custom: bool = True
    critical: bool = True


@dataclass
class PreconditionAssessment:
    """Result of evaluating task preconditions and ambiguity Z-score."""

    task_type: str
    ambiguity_z: float
    autonomy_level: AutonomyLevel
    filled_slots: Dict[str, str]
    missing_critical: List[str]
    missing_optional: List[str]
    should_enable_thinking: bool
    protocol_mode: str  # "none" | "compact_slot"
    compact_protocol_text: str = ""
    question_specs: List[ClarificationQuestionSpec] = field(default_factory=list)
    reason: str = ""


# ----------------------------------------------------------------------------
# Domain Precondition Patterns (Macro vs Granular Detection)
# ----------------------------------------------------------------------------

_MACRO_CITIES = (
    "antalya", "istanbul", "ankara", "izmir", "muğla", "mugla", "bursa",
    "kapadokya", "cappadocia", "trabzon", "gaziantep", "adana", "konya", "mersin",
)

_GRANULAR_LOCATION_RE = re.compile(
    r"(?:"
    r"konyaalt[ıi]|lara\b|kundu\b|alanya\b|belek\b|side\b|manavgat\b|kemer\b|ka[şs]\b|kalkan\b|muratpa[şs]a|kalei[çc]i|"
    r"kad[ıi]k[öo]y|be[şs]ikta[şs]|taksim\b|beyo[ğg]lu|[üu]sk[üu]dar|fatih\b|sultanahmet|şişli|sisli|sarıyer|sariyer|"
    r"[çc]ankaya|k[ıi]z[ıi]lay|bah[çc]elievler|ulus\b|alsancak|bornova|kar[şs][ıi]yaka|[çc]e[şs]me|ala[çc]at[ıi]|"
    r"bodrum|fethiye|marmaris|dat[çc]a|g[öo]cek|g[öo]reme|[üu]rg[üu]p|avanos|"
    r"\b(?:il[çc]e(?:si|de|ye)?|semt(?:i|te|inde)?|mahalle(?:si|de)?|merkez(?:de|i)?|sahil(?:de|e)?|havaliman[ıi]|otogar|district|neighborhood|near\s+[a-zığüşöç]+)\b"
    r")",
    re.I,
)

_TRAVEL_CUE_RE = re.compile(
    r"\b(?:otel(?:im|de|e|i)?|gezi(?:lecek)?|rota|turist|tatil|kahvalt[ıi]|ak[şs]am\s+yeme[ğg]i|restoran|itinerary|travel|sightseeing|things\s+to\s+do|nereleri\s+gez)\b",
    re.I,
)

_DURATION_CUE_RE = re.compile(
    r"(?:"
    r"\b\d+\s*(?:g[üu]n|saat|hafta|gece|day|days|hour|hours|hr|hrs)\b|"
    r"\b(?:yar[ıi]m\s+g[üu]n|t[üu]m\s+g[üu]n|hafta\s+sonu|bir\s+ka[çc]\s+saat|bug[üu]n|yar[ıi]n|ak[şs]am|sabah|k[ıi]s[ıi]tl[ıi]\s+vakit)\b"
    r")",
    re.I,
)

_TRAVEL_PREF_RE = re.compile(
    r"\b(?:y[üu]r[üu]yerek|y[üu]r[üu]y[üu][şs]|ara[çc]|araba|otob[üu]s|tramvay|taksi|tarih(?:i)?|m[üu]ze|plaj|deniz|do[ğg]a|şelale|aile|[çc]ocuk)\b",
    re.I,
)

_CITY_DISTRICT_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "antalya": (
        "Konyaaltı",
        "Lara / Kundu",
        "Kaleiçi / Muratpaşa (Merkez)",
        "Alanya",
        "Belek",
        "Side / Manavgat",
        "Kemer / Kaş",
    ),
    "istanbul": (
        "Fatih / Sultanahmet",
        "Beyoğlu / Taksim",
        "Beşiktaş",
        "Kadıköy",
        "Üsküdar",
    ),
    "ankara": (
        "Çankaya / Kızılay",
        "Ulus / Altındağ (Tarihi Merkez)",
        "Yenimahalle / Bahçelievler",
        "Gölbaşı / Eymir",
    ),
    "izmir": (
        "Konak / Alsancak",
        "Karşıyaka / Bostanlı",
        "Bornova",
        "Çeşme / Alaçatı / Urla",
    ),
}

_MATH_CONCEPTUAL_RE = re.compile(
    r"\b(?:form[üu]l[üu]?|teorem|ispat(?:la)?|kan[ıi]t|tan[ıi]m[ıi]?|nedir\s*\??$|kural[ıi]|what\s+is\s+the\s+formula|prove|derive)\b",
    re.I,
)

_MATH_CALC_VERB_RE = re.compile(
    r"\b(?:hesapla(?:y[ıi]n)?|bul(?:unuz)?|ka[çc](?:t[ıi]r)?|calculate|compute|determine|find\s+the)\b",
    re.I,
)

_MATH_PARAM_SHAPE_RE = re.compile(
    r"\b(?:silindir(?:in)?|k[üu]re(?:nin)?|koni(?:nin)?|[üu][çc]gen(?:in)?|dikd[öo]rtgen(?:in)?|kare(?:nin)?|daire(?:nin)?|[çc]ember(?:in)?|hacim|hacmi(?:ni)?|y[üu]zey\s+alan[ıi]|alan[ıi](?:n[ıi])?|[çc]evre(?:si|sini)?|faiz|h[ıi]z|bas[ıi]n[çc]|yo[ğg]unluk|cylinder|sphere|cone|triangle|circle|volume|area|perimeter)\b",
    re.I,
)

_MATH_GIVEN_VALUES_RE = re.compile(
    r"(?:\d|[a-zA-Z]\s*=|\b(?:yarıçap|yaricap|radius|yükseklik|yukseklik|height|çap|cap|taban|kenar|kütlesi|ivme)\s*(?:ise|olsun|=|:))",
    re.I,
)

_PLANNING_BROAD_RE = re.compile(
    r"\b(?:program[ıi]?\s*(?:[çc][ıi]kar|[öo]ner|haz[ıi]rla)|g[üu]n[üu]m[üu]\s*ayarla|plan(?:la|[ıi]\s*yap)|mimari\s*tasarla|sistem\s*kur|web\s*sunucu(?:su)?|veritaban[ıi]\s*se[çc]|b[üu]t[çc]e\s*plan|spor\s*program|ders\s*[çc]al[ıi][şs]ma\s*program)\b",
    re.I,
)

_CODE_MODIFY_RE = re.compile(
    r"\b(?:d[üu]zenle|d[üu]zelt|g[üu]ncelle|refactor|h[ıi]zland[ıi]r|optimize\s*et|ekle|sil|temizle|kald[ıi]r|test\s*yaz|edit|fix|modify|delete|remove|clean\s*up)\b",
    re.I,
)

_RISKY_DESTRUCTIVE_RE = re.compile(
    r"\b(?:sil|temizle|sıfırla|reset|drop|truncate|rm\s+-rf|remove-item|overwrite|üzerine\s+yaz)\b",
    re.I,
)

_GREETING_OR_IDENTITY_RE = re.compile(
    r"^\s*(?:merhaba|selam|g[üu]nayd[ıi]n|iyi\s+(?:ak[şs]amlar|geceler)|nas[ıi]ls[ıi]n|sen\s+kimsin|kimsin|mimar[ıi]n\s+kim|burak\s+kim|ad[ıi]n\s+ne|hi|hello|who\s+are\s+you)[\s\?!\.]*$",
    re.I,
)


def _detect_macro_city(text: str) -> Optional[str]:
    low = (text or "").lower()
    for city in _MACRO_CITIES:
        if city in low:
            if city == "mugla":
                return "muğla"
            return city
    return None


class SemanticPreconditionGraph:
    """
    Lightweight Continuous Semantic Topology (`CST`) + Arkhe 6 (`ConditionsArche`)
    precondition scorer. Determines in <0.5 ms which critical task parameters are
    missing vs already satisfied in the prompt/history.
    """

    @staticmethod
    def evaluate(
        user_query: str,
        history_messages: Sequence[Dict[str, str]] = (),
        mode_value: str = "SYSTEM2_CAUSAL_LOGIC",
        is_agent: bool = False,
        workspace_files: Optional[Sequence[str]] = None,
        has_attached_docs: bool = False,
        full_threshold: float = 0.20,
        halt_threshold: float = 0.65,
    ) -> PreconditionAssessment:
        q = (user_query or "").strip()
        if not q or _GREETING_OR_IDENTITY_RE.match(q):
            return PreconditionAssessment(
                task_type="direct_social_or_identity",
                ambiguity_z=0.0,
                autonomy_level=AutonomyLevel.FULL,
                filled_slots={},
                missing_critical=[],
                missing_optional=[],
                should_enable_thinking=False,
                protocol_mode="none",
                reason="Greeting or identity query; zero missing preconditions.",
            )

        hist_user_text = " ".join(
            (m.get("content") or "")
            for m in history_messages
            if m.get("role") in {"user", "assistant"}
        )
        combined = f"{hist_user_text}\n{q}".strip()

        # 1. Offline Coding & Automation Agent Gate (`AutonomyGate` + Workspace Candidates)
        # Checked BEFORE travel/route so software terms like "rota" (API route) never trigger hotel prompts.
        if is_agent and _CODE_MODIFY_RE.search(q):
            has_explicit_file = bool(re.search(r"[A-Za-z0-9_\-]+\.(?:py|rs|ts|tsx|js|json|toml|md|yml|yaml|cu|cpp|h)\b", q))
            is_risky = bool(_RISKY_DESTRUCTIVE_RE.search(q))
            if not has_explicit_file and not has_attached_docs:
                cands = tuple(workspace_files[:5]) if workspace_files else ()
                z = 0.78 if is_risky else 0.58
                lvl = AutonomyLevel.HALT_AND_PROMPT if z >= halt_threshold else AutonomyLevel.DEFERRED
                q_spec = ClarificationQuestionSpec(
                    slot_id="hedef_dosya",
                    question="Bu işlemin tam olarak hangi dosya/modül üzerinde uygulanmasını ve hedef kapsamını belirtir misiniz?",
                    options=cands if cands else ("Tüm workspace'i önce salt-okunur tara", "Yalnızca ana giriş dosyasını hedefle"),
                    allow_custom=True,
                    critical=True,
                )
                return PreconditionAssessment(
                    task_type="code_modify",
                    ambiguity_z=z,
                    autonomy_level=lvl,
                    filled_slots={},
                    missing_critical=["hedef_dosya"],
                    missing_optional=[],
                    should_enable_thinking=True,
                    protocol_mode="compact_slot",
                    compact_protocol_text=_build_compact_slot_protocol("hedef_dosya / kapsam", q_spec),
                    question_specs=[q_spec],
                    reason="Agent edit/action prompt lacks an explicit target file or symbol.",
                )

        # 2. Travel / Route / Hotel Recommendation Domain (skipped in coding agent mode)
        if (not is_agent) and _TRAVEL_CUE_RE.search(q):
            macro_city = _detect_macro_city(combined)
            has_granular = bool(_GRANULAR_LOCATION_RE.search(combined))
            has_duration = bool(_DURATION_CUE_RE.search(combined))
            has_pref = bool(_TRAVEL_PREF_RE.search(combined))

            filled: Dict[str, str] = {}
            missing_crit: List[str] = []
            missing_opt: List[str] = []
            q_specs: List[ClarificationQuestionSpec] = []
            z = 0.0

            if has_granular:
                filled["konum"] = _GRANULAR_LOCATION_RE.search(combined).group(0)
            else:
                missing_crit.append("konum")
                z += 0.55 if macro_city else 0.70
                opts = _CITY_DISTRICT_OPTIONS.get(macro_city or "", ("Şehir Merkezi", "Sahil / Turistik Bölge", "Tarihi Bölge"))
                q_txt = (
                    f"Oteliniz {macro_city.capitalize()}'nın hangi bölge veya ilçesinde bulunuyor?"
                    if macro_city
                    else "Konumunuz veya konakladığınız il/ilçe neresi?"
                )
                q_specs.append(
                    ClarificationQuestionSpec(
                        slot_id="konum",
                        question=q_txt,
                        options=opts,
                        allow_custom=True,
                        critical=True,
                    )
                )

            if has_duration:
                filled["sure"] = _DURATION_CUE_RE.search(combined).group(0)
            else:
                missing_opt.append("sure")
                z += 0.22
                q_specs.append(
                    ClarificationQuestionSpec(
                        slot_id="sure",
                        question="Gezi için yaklaşık ne kadar vaktiniz var ve ulaşım şekliniz nedir?",
                        options=("Yarım gün / Yürüyerek", "1 tam gün / Toplu taşıma veya araç", "2-3 gün / Araçlı"),
                        allow_custom=True,
                        critical=False,
                    )
                )

            if has_pref:
                filled["tercih"] = _TRAVEL_PREF_RE.search(combined).group(0)

            level = AutonomyLevel.FULL
            if z >= halt_threshold or missing_crit:
                level = AutonomyLevel.HALT_AND_PROMPT if z >= halt_threshold else AutonomyLevel.DEFERRED
            elif z >= full_threshold:
                level = AutonomyLevel.DEFERRED

            if level == AutonomyLevel.FULL:
                return PreconditionAssessment(
                    task_type="route_travel",
                    ambiguity_z=round(z, 3),
                    autonomy_level=AutonomyLevel.FULL,
                    filled_slots=filled,
                    missing_critical=[],
                    missing_optional=missing_opt,
                    should_enable_thinking=False,
                    protocol_mode="none",
                    reason="Travel location and core constraints are already specified.",
                )

            slot_list_str = ", ".join(missing_crit + missing_opt[:1])
            compact_proto = _build_compact_slot_protocol(slot_list_str, q_specs[0])
            return PreconditionAssessment(
                task_type="route_travel",
                ambiguity_z=round(min(1.0, z), 3),
                autonomy_level=level,
                filled_slots=filled,
                missing_critical=missing_crit,
                missing_optional=missing_opt,
                should_enable_thinking=True,
                protocol_mode="compact_slot",
                compact_protocol_text=compact_proto,
                question_specs=q_specs,
                reason=f"Missing travel preconditions: {', '.join(missing_crit + missing_opt)}",
            )

        # 3. Interactive Mathematics / Quantitative Parameter Check
        if mode_value == "SYSTEM2_MATH_FORMAL" or _MATH_PARAM_SHAPE_RE.search(q):
            is_conceptual = bool(_MATH_CONCEPTUAL_RE.search(q))
            asks_numeric = bool(_MATH_CALC_VERB_RE.search(q)) or bool(_MATH_PARAM_SHAPE_RE.search(q))
            # Check current turn for any numeric/symbolic values; check history only for explicit
            # geometric parameter bindings so unrelated numbers in prior turns (e.g. years) do not suppress asks.
            has_vals = bool(_MATH_GIVEN_VALUES_RE.search(q)) or bool(
                re.search(
                    r"(?:[a-zA-Z]\s*=\s*\d|\b(?:yarıçap|yaricap|radius|yükseklik|yukseklik|height|çap|cap|taban|kenar|kütlesi|ivme)\s*(?:ise|olsun|=|:)\s*\d)",
                    hist_user_text,
                    re.I,
                )
            )

            if (not is_conceptual) and asks_numeric and _MATH_PARAM_SHAPE_RE.search(q) and (not has_vals):
                shape_word = _MATH_PARAM_SHAPE_RE.search(q).group(0)
                q_spec = ClarificationQuestionSpec(
                    slot_id="matematik_parametre",
                    question=f"'{shape_word}' hesaplaması için bilinen ölçü/değerler (ör. yarıçap r, yükseklik h, kenar uzunluğu) ve birim nedir?",
                    options=("Birim cm olarak veriyorum", "Birim m olarak veriyorum", "Sembolik (harfli) genel formül çıkar"),
                    allow_custom=True,
                    critical=True,
                )
                return PreconditionAssessment(
                    task_type="calculation_math",
                    ambiguity_z=0.75,
                    autonomy_level=AutonomyLevel.HALT_AND_PROMPT,
                    filled_slots={},
                    missing_critical=["matematik_parametre"],
                    missing_optional=[],
                    should_enable_thinking=True,
                    protocol_mode="compact_slot",
                    compact_protocol_text=_build_compact_slot_protocol("ölçü/parametre değerleri", q_spec),
                    question_specs=[q_spec],
                    reason="Numeric/geometric calculation requested without given dimensions or symbolic request.",
                )

            return PreconditionAssessment(
                task_type="calculation_math",
                ambiguity_z=0.05,
                autonomy_level=AutonomyLevel.FULL,
                filled_slots={"params": "present_or_conceptual"},
                missing_critical=[],
                missing_optional=[],
                should_enable_thinking=True,
                protocol_mode="none",
                reason="Math problem is either self-contained or conceptual.",
            )

        # 4. General Planning / Architecture / Schedule Design
        if _PLANNING_BROAD_RE.search(q) and not has_attached_docs:
            has_scope_detail = bool(re.search(r"(?:\d+|python|rust|fastapi|sqlite|postgres|react|hafta|g[üu]n|saat|b[üu]t[çc]e|tl|usd|\$)", combined, re.I))
            if not has_scope_detail:
                q_spec = ClarificationQuestionSpec(
                    slot_id="kapsam_kisit",
                    question="Plan/tasarımı optimize edebilmem için temel hedef, süre veya teknik/bütçe kısıtınız nedir?",
                    options=("Hızlı/minimal kapsam (MVP)", "Kapsamlı & adım adım plan", "Mevcut kaynaklarıma göre optimize et"),
                    allow_custom=True,
                    critical=True,
                )
                return PreconditionAssessment(
                    task_type="planning_design",
                    ambiguity_z=0.52,
                    autonomy_level=AutonomyLevel.DEFERRED,
                    filled_slots={},
                    missing_critical=["kapsam_kisit"],
                    missing_optional=[],
                    should_enable_thinking=True,
                    protocol_mode="compact_slot",
                    compact_protocol_text=_build_compact_slot_protocol("hedef / süre / kısıt", q_spec),
                    question_specs=[q_spec],
                    reason="Broad planning/design prompt lacks concrete scope or constraints.",
                )

        return PreconditionAssessment(
            task_type="general",
            ambiguity_z=0.08 if mode_value != "SYSTEM1_DIRECT" else 0.0,
            autonomy_level=AutonomyLevel.FULL,
            filled_slots={},
            missing_critical=[],
            missing_optional=[],
            should_enable_thinking=(mode_value != "SYSTEM1_DIRECT"),
            protocol_mode="none",
            reason="No missing critical preconditions detected.",
        )


def _build_compact_slot_protocol(missing_desc: str, example_spec: ClarificationQuestionSpec) -> str:
    opts_json = ", ".join(f'"{o}"' for o in example_spec.options[:6])
    return (
        f"\n\n[ARKHE-6 ÖNKOŞUL PROTOKOLÜ (<sodals_ask>)]:\n"
        f"- Bu görevde eksik kritik parametre olabilir: {missing_desc}.\n"
        f"- Eksikse ASLA tahminle genel cevap verme; </think> etiketini kapatmadan düşüncen içinde TEK bir blok üret:\n"
        f"  <sodals_ask>{{\"question\": \"{example_spec.question}\", \"options\": [{opts_json}]}}</sodals_ask>\n"
        f"- Aynı anda birden fazla eksik varsa `\"questions\": [{{\"id\": \"...\", \"question\": \"...\", \"options\": [...]}}]` kullan.\n"
        f"- Blok tamamlanınca üretim dondurulur; `<user_clarification>` gelince aynı cümleden devam edeceksin."
    )


# ----------------------------------------------------------------------------
# Key-Lock Constraint Manager (`fluid_algebra.rs` Impedance & Pruning)
# ----------------------------------------------------------------------------

class ClarificationConstraintLock:
    """
    Maintains resolved clarification slot locks (`Key -> Value`) and rejected option
    viscosity barriers so post-unfreeze generation cannot drift back to eliminated
    alternatives or enter duplicate clarification loops.
    """

    def __init__(self) -> None:
        self.locked_slots: Dict[str, str] = {}
        self.rejected_alternatives: List[str] = []
        self.asked_questions: List[str] = []

    def lock_answer(
        self,
        slot_id: str,
        chosen_answer: str,
        question: str = "",
        options: Sequence[str] = (),
    ) -> None:
        clean_ans = (chosen_answer or "").strip()
        if clean_ans:
            self.locked_slots[slot_id or f"slot_{len(self.locked_slots)+1}"] = clean_ans
        if question:
            self.asked_questions.append(question.strip())

        chosen_low = clean_ans.lower()
        for opt in options:
            opt_s = (opt or "").strip()
            if opt_s and opt_s.lower() != chosen_low and opt_s not in self.rejected_alternatives:
                self.rejected_alternatives.append(opt_s)

    def is_duplicate_ask(self, question: str) -> bool:
        q_norm = re.sub(r"\W+", " ", (question or "").lower()).strip()
        if not q_norm:
            return False
        for prev in self.asked_questions:
            p_norm = re.sub(r"\W+", " ", prev.lower()).strip()
            if q_norm == p_norm or (len(q_norm) > 12 and (q_norm in p_norm or p_norm in q_norm)):
                return True
        return False

    def build_lock_directive(self) -> str:
        if not self.locked_slots:
            return ""
        locked_str = "; ".join(f"{k}={v}" for k, v in self.locked_slots.items())
        parts = [f"[KİLİTLİ ÖNKOŞULLAR: {locked_str}]"]
        if self.rejected_alternatives:
            rej_preview = ", ".join(self.rejected_alternatives[:5])
            parts.append(f"[ELENEN DALLAR (odak dışı): {rej_preview}]")
        return " ".join(parts)

    def build_clean_summary(self) -> str:
        if not self.locked_slots:
            return ""
        pairs = [f"{v}" for v in self.locked_slots.values() if v]
        return f"[Netleştirilen Bağlam: {', '.join(pairs)}]" if pairs else ""

    def check_lock_adherence(self, output_text: str) -> Dict[str, Any]:
        low = (output_text or "").lower()
        matched = []
        missing = []
        for k, v in self.locked_slots.items():
            tokens = [t for t in re.split(r"[\s/,]+", v.lower()) if len(t) >= 3]
            if not tokens or any(t in low for t in tokens):
                matched.append(k)
            else:
                missing.append(k)
        return {
            "locked_count": len(self.locked_slots),
            "adhered_slots": matched,
            "missing_slots": missing,
            "adherence_ok": len(missing) == 0,
        }


# ----------------------------------------------------------------------------
# Entropy Watchdog + Structural-Safe SkewG Braking (`LogitsProcessor`)
# ----------------------------------------------------------------------------

def collect_structural_token_ids(tokenizer: Any) -> Set[int]:
    """Identifies structural XML/JSON tokens preserved from SkewG sharpening."""
    structural: Set[int] = set()
    probes = (
        "<",
        ">",
        "</",
        "<think>",
        "</think>",
        "<sodals_ask>",
        "</sodals_ask>",
        "sod",
        "ask",
        "_",
        "{",
        "}",
        "\"",
        ":",
        "[",
        "]",
        ",",
        "\n",
    )
    for p in probes:
        try:
            for tid in tokenizer.encode(p, add_special_tokens=False):
                structural.add(int(tid))
        except Exception:
            continue
    return structural


class EntropyAndSkewGProcessor(LogitsProcessor):
    """
    Combines:
      1. Exact Next-Token Shannon Entropy `H_t = -sum p log p` + Top1-Top2 margin
         monitoring (`eval_internal_signals_logprob_vs_probe_results.json`).
      2. Epistemic Ambiguity Watchdog: trips if `H_t` stays high while critical
         task preconditions are missing and the model has not emitted `<sodals_ask>`.
      3. `SkewGBraking`: sharpens flat cognitive plateaus by 1.20x while preserving
         `<sodals_ask>` JSON/XML structural tokens.
    """

    def __init__(
        self,
        prompt_len: int,
        structural_token_ids: Optional[Set[int]] = None,
        enable_watchdog: bool = False,
        entropy_halt_threshold: float = 2.15,
        margin_halt_threshold: float = 0.25,
        watchdog_min_steps: int = 18,
        watchdog_max_steps: int = 96,
        skew_window: int = 32,
        skew_flatline_var: float = 0.10,
        skew_sharpen_factor: float = 1.20,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.structural_token_ids = set(structural_token_ids or ())
        self.enable_watchdog = bool(enable_watchdog)
        self.entropy_halt_threshold = float(entropy_halt_threshold)
        self.margin_halt_threshold = float(margin_halt_threshold)
        self.watchdog_min_steps = int(watchdog_min_steps)
        self.watchdog_max_steps = int(watchdog_max_steps)
        self.skew_window = int(skew_window)
        self.skew_flatline_var = float(skew_flatline_var)
        self.skew_sharpen_factor = float(skew_sharpen_factor)

        self.steps: int = 0
        self.sum_entropy: float = 0.0
        self.peak_entropy: float = 0.0
        self.last_entropy: float = 0.0
        self.sum_margin: float = 0.0
        self.min_margin: float = 1.0
        self.recent_entropies: Deque[float] = deque(maxlen=16)
        self.recent_margins: Deque[float] = deque(maxlen=16)
        self.recent_logprobs: Deque[float] = deque(maxlen=self.skew_window)
        self.watchdog_tripped: bool = False
        self.watchdog_step: Optional[int] = None
        self.ask_tag_opened: bool = False
        self._struct_idx_cache: Dict[Tuple[str, int], Optional[torch.Tensor]] = {}

    def _get_structural_index_tensor(self, device: torch.device, vocab_lim: int) -> Optional[torch.Tensor]:
        if not self.structural_token_ids:
            return None
        key = (str(device), int(vocab_lim))
        if key not in self._struct_idx_cache:
            valid = sorted(tid for tid in self.structural_token_ids if 0 <= tid < vocab_lim)
            self._struct_idx_cache[key] = (
                torch.tensor(valid, dtype=torch.long, device=device) if valid else None
            )
        return self._struct_idx_cache[key]

    def notify_ask_tag_opened(self) -> None:
        self.ask_tag_opened = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.numel() == 0:
            return scores

        row = scores[0].float()
        log_probs = torch.log_softmax(row, dim=-1)
        probs = torch.exp(log_probs)
        ent_terms = torch.where(probs > 0.0, probs * log_probs, torch.zeros_like(probs))
        ent_t = -ent_terms.sum()

        k_top = min(2, probs.numel())
        top2_vals, _ = torch.topk(probs, k=k_top)
        p1_t = top2_vals[0]
        p2_t = top2_vals[1] if k_top > 1 else torch.zeros_like(p1_t)
        max_lp_t = log_probs.max()

        # Single device->host scalar transfer per decode step (replaces 4-5 blocking .item() syncs)
        entropy, p1, p2, max_lp = torch.stack((ent_t, p1_t, p2_t, max_lp_t)).tolist()
        if not math.isfinite(entropy):
            entropy = 0.0
        margin = max(0.0, float(p1) - float(p2))

        self.steps += 1
        self.last_entropy = float(entropy)
        self.sum_entropy += float(entropy)
        if entropy > self.peak_entropy:
            self.peak_entropy = float(entropy)
        self.sum_margin += margin
        if margin < self.min_margin:
            self.min_margin = margin

        self.recent_entropies.append(float(entropy))
        self.recent_margins.append(margin)
        if math.isfinite(max_lp):
            self.recent_logprobs.append(float(max_lp))

        # Epistemic Fallback Watchdog: missing slots + sustained high entropy inside early reasoning
        if (
            self.enable_watchdog
            and (not self.watchdog_tripped)
            and (not self.ask_tag_opened)
            and (self.watchdog_min_steps <= self.steps <= self.watchdog_max_steps)
            and len(self.recent_entropies) >= 8
        ):
            roll_ent = sum(self.recent_entropies) / len(self.recent_entropies)
            roll_mrg = sum(self.recent_margins) / len(self.recent_margins)
            if roll_ent >= self.entropy_halt_threshold and roll_mrg <= self.margin_halt_threshold:
                self.watchdog_tripped = True
                self.watchdog_step = self.steps

        # Centered Skew-G Braking on flatlined reasoning plateaus (vectorized structural token preservation)
        if len(self.recent_logprobs) >= max(8, self.skew_window // 2):
            lp_list = list(self.recent_logprobs)
            mean_lp = sum(lp_list) / len(lp_list)
            var_lp = sum((x - mean_lp) ** 2 for x in lp_list) / len(lp_list)
            if var_lp < self.skew_flatline_var:
                finite_mask = torch.isfinite(scores)
                if finite_mask.any():
                    struct_idx = self._get_structural_index_tensor(scores.device, scores.shape[-1])
                    preserved = scores.index_select(-1, struct_idx).clone() if struct_idx is not None else None
                    row_max = scores.masked_fill(~finite_mask, float("-inf")).max(dim=-1, keepdim=True).values
                    centered = scores - row_max
                    sharpened = row_max + centered * self.skew_sharpen_factor
                    scores = torch.where(finite_mask, sharpened, scores)
                    if struct_idx is not None and preserved is not None:
                        scores.index_copy_(-1, struct_idx, preserved)

        return scores

    def summary(self) -> Dict[str, Any]:
        n = max(1, self.steps)
        return {
            "steps_observed": self.steps,
            "mean_entropy": round(self.sum_entropy / n, 4),
            "peak_entropy": round(self.peak_entropy, 4),
            "mean_margin": round(self.sum_margin / n, 4),
            "min_margin": round(self.min_margin if self.steps > 0 else 1.0, 4),
            "watchdog_tripped": self.watchdog_tripped,
            "watchdog_step": self.watchdog_step,
        }


# ----------------------------------------------------------------------------
# CPU RAM Offload / Restore Helpers for Qwen 3.5 DynamicCache
# ----------------------------------------------------------------------------

def _move_tensor_tree(obj: Any, device: Any, seen: Optional[Set[int]] = None) -> Any:
    if obj is None:
        return None
    target_dev = torch.device(device) if not isinstance(device, torch.device) else device
    if isinstance(obj, torch.Tensor):
        return obj.to(device=target_dev, non_blocking=False)
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return obj
    seen.add(oid)

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, torch.Tensor):
                obj[i] = item.to(device=target_dev, non_blocking=False)
            elif isinstance(item, tuple):
                obj[i] = _move_tensor_tree(item, target_dev, seen)
            else:
                _move_tensor_tree(item, target_dev, seen)
        return obj

    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(item, target_dev, seen) for item in obj)

    if isinstance(obj, dict):
        for k, val in list(obj.items()):
            if isinstance(val, torch.Tensor):
                obj[k] = val.to(device=target_dev, non_blocking=False)
            elif isinstance(val, tuple):
                obj[k] = _move_tensor_tree(val, target_dev, seen)
            else:
                _move_tensor_tree(val, target_dev, seen)
        return obj

    attr_names: List[str] = []
    if hasattr(obj, "__dict__"):
        attr_names.extend(obj.__dict__.keys())
    for cls in type(obj).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for s in slots:
            if s not in ("__dict__", "__weakref__") and s not in attr_names:
                attr_names.append(s)

    for attr in attr_names:
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(val, torch.Tensor):
            try:
                setattr(obj, attr, val.to(device=target_dev, non_blocking=False))
            except Exception:
                pass
        elif attr == "device" and isinstance(val, torch.device):
            try:
                setattr(obj, attr, target_dev)
            except Exception:
                pass
        elif isinstance(val, tuple):
            try:
                setattr(obj, attr, _move_tensor_tree(val, target_dev, seen))
            except Exception:
                pass
        elif isinstance(val, (list, dict)) or hasattr(val, "keys") or hasattr(val, "values") or "Cache" in type(val).__name__ or "Layer" in type(val).__name__:
            _move_tensor_tree(val, target_dev, seen)
    return obj


def offload_cache_to_cpu(cache: Any) -> Any:
    """
    Moves all KV/conv/recurrent state tensors in a HuggingFace `DynamicCache`
    into CPU RAM while generation is paused awaiting human clarification,
    freeing GPU VRAM on 8 GB cards without losing latent/KV state.
    """
    if cache is None:
        return None
    _move_tensor_tree(cache, torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def restore_cache_to_device(cache: Any, device: Any) -> Any:
    """Restores a CPU-offloaded `DynamicCache` back onto `device` before Pass 2 resume."""
    if cache is None:
        return None
    target = torch.device(device) if not isinstance(device, torch.device) else device
    _move_tensor_tree(cache, target)
    return cache
