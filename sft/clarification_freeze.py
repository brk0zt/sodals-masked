"""
SODALS / Svon — Mid-Turn Thought-Freeze & Interactive Clarification Engine (v2)
=============================================================================
Implements interactive ambiguity resolution mid-turn with genuine state freezing:
  1. `SemanticPreconditionGraph` + `AutonomyGate` (`epistemic_clarify_gate.py`)
     evaluates missing preconditions before generation (< 0.5 ms).
  2. During Pass 1, `EntropyAndSkewGProcessor` measures next-token Shannon
     entropy & Top1-Top2 margin without extra forward passes, while `_MidTurnStopOnAsk`
     stops `model.generate()` cleanly either when:
       - the model emits `</sodals_ask>` / `</ask_user>`, OR
       - the Epistemic Entropy Watchdog trips while critical slots are missing.
  3. Frozen generation captures `frozen_token_ids`, CPU-offloaded `DynamicCache`
     (Attention KV + GatedDeltaNet recurrent/conv states), Svon-CGW cumulative phase
     offsets, and `ClarificationConstraintLock` (Key-Lock + viscosity pruning).
  4. Unfreezing resumes via delta tokens appended onto the restored `DynamicCache`
     (zero full-prompt re-prefill!), balances `<think>...</think>` in streamed output,
     and saves ONLY `[Netleştirilen Bağlam: ...] + final answer` into chat history.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from epistemic_clarify_gate import (
    ClarificationConstraintLock,
    ClarificationQuestionSpec,
    EntropyAndSkewGProcessor,
    PreconditionAssessment,
)

OPEN_TAGS = ("<sodals_ask>", "<ask_user>")
CLOSE_TAGS = ("</sodals_ask>", "</ask_user>")

_ASK_PATTERNS = [
    re.compile(r"<sodals_ask>(.*?)</sodals_ask>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<ask_user>(.*?)</ask_user>", re.DOTALL | re.IGNORECASE),
    re.compile(r"```(?:sodals_ask|ask_user)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE),
]

CLARIFICATION_PROTOCOL_PROMPT = (
    "\n\nNETLEŞTİRME VE DÜŞÜNMEYİ DONDURMA PROTOKOLÜ (<sodals_ask>):\n"
    "- Yanıt vermeden önce eksik KRİTİK parametre (konum/ilçe, süre, ölçü/sayısal değer, hedef dosya, kısıt vb.) var mı kontrol et.\n"
    "- Kritik detay eksikse ASLA genelgeçer tahmin yapma ve </think> etiketini kapatmadan düşüncen içinde TEK bir blok üret:\n"
    "  <sodals_ask>{\"question\": \"Kısa ve net soru\", \"options\": [\"Seçenek 1\", \"Seçenek 2\"]}</sodals_ask>\n"
    "- Birden fazla eksik parametre varsa `{\"questions\": [{\"id\": \"slot1\", \"question\": \"...\", \"options\": [...]}]}` biçimini kullanabilirsin.\n"
    "- `<user_clarification>` yanıtı geldiğinde dondurulan düşüncenin tam kaldığı yerden devam edip en optimum cevabı ver."
)

# Kept exported for compatibility & heuristic supplemental signal; primary gating uses SemanticPreconditionGraph.
PLANNING_OR_AMBIGUOUS_CUE_RE = re.compile(
    r"(?:"
    r"\b(?:öner(?:ir\s*misin|i)?|tavsiye|rota|gezi|plan(?:la)?|program|nereye|nerede|ne\s+yap(?:ayım|malı)|"
    r"hangi(?:si|leri)?|kur(?:mak|alım)?|tasarla|mimari|ekle(?:yelim)?|hazırla|bütçe|otel|tatil|kahvaltı|akşam\s+yemeği)\b|"
    r"\b(?:recommend|suggest|itinerary|plan|design|build|architecture|which\s+one|what\s+should\s+i)\b"
    r")",
    re.I,
)


@dataclass
class ClarificationQuestionItem:
    """One slot question inside a single- or multi-slot clarification card."""

    slot_id: str
    question: str
    options: List[str] = field(default_factory=list)
    allow_custom: bool = True


@dataclass
class ClarificationRequest:
    """Structured mid-turn clarification payload extracted from `<sodals_ask>` or watchdog."""

    question: str
    options: List[str] = field(default_factory=list)
    allow_custom: bool = True
    reason: str = ""
    raw_block: str = ""
    start_idx: int = -1
    end_idx: int = -1
    slot_id: str = "parametre"
    questions: List[ClarificationQuestionItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.questions and self.question:
            self.questions = [
                ClarificationQuestionItem(
                    slot_id=self.slot_id or "parametre",
                    question=self.question,
                    options=list(self.options),
                    allow_custom=self.allow_custom,
                )
            ]
        elif self.questions and not self.question:
            self.question = self.questions[0].question
            self.options = list(self.questions[0].options)
            self.slot_id = self.questions[0].slot_id
            self.allow_custom = self.questions[0].allow_custom


@dataclass
class FrozenClarificationSession:
    """Holds paused mid-turn state (CPU-offloaded DynamicCache + tokens + Key-Lock) awaiting answer."""

    freeze_id: str
    formatted_prompt: str
    frozen_prefix: str
    pending_request: ClarificationRequest
    use_adapters: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    rep_penalty: float
    mode_value: str
    alpha: float
    rk4_steps: int
    domain_label: str
    user_query: str
    conversation_sig: str = ""
    frozen_token_ids: Optional[List[int]] = None
    frozen_cache_cpu: Any = None
    cvnn_phase_offsets: Optional[Dict[int, Any]] = None
    locked_constraints: ClarificationConstraintLock = field(default_factory=ClarificationConstraintLock)
    entropy_telemetry: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "general"
    ambiguity_z: float = 0.0
    trigger_source: str = "model_tag"  # "model_tag" | "entropy_watchdog"
    opened_think_in_prompt: bool = False
    created_at: float = field(default_factory=time.time)
    answer_event: threading.Event = field(default_factory=threading.Event)
    user_answer: Optional[str] = None
    raw_user_answer: Optional[str] = None


def clarification_request_from_assessment(
    assessment: PreconditionAssessment,
    trigger_reason: str = "",
) -> Optional[ClarificationRequest]:
    """Builds a structured single- or multi-slot `ClarificationRequest` from `SemanticPreconditionGraph`."""
    if not assessment.question_specs:
        return None
    items = [
        ClarificationQuestionItem(
            slot_id=sp.slot_id,
            question=sp.question,
            options=list(sp.options),
            allow_custom=sp.allow_custom,
        )
        for sp in assessment.question_specs[:3]
    ]
    return ClarificationRequest(
        question=items[0].question,
        options=list(items[0].options),
        allow_custom=items[0].allow_custom,
        reason=trigger_reason or assessment.reason,
        slot_id=items[0].slot_id,
        questions=items,
    )


def parse_clarification_request(text: str) -> Optional[ClarificationRequest]:
    """Parses the first `<sodals_ask>...</sodals_ask>` block (single or multi-question) from generated text."""
    body = text or ""
    for pat in _ASK_PATTERNS:
        m = pat.search(body)
        if not m:
            continue
        inner = (m.group(1) or "").strip()
        if not inner:
            continue

        parsed: Optional[Dict[str, Any]] = None
        s_i = inner.find("{")
        e_i = inner.rfind("}")
        if s_i != -1 and e_i > s_i:
            snippet = inner[s_i : e_i + 1]
            for cand in (snippet, re.sub(r",\s*([}\]])", r"\1", snippet)):
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict) and (obj.get("question") or obj.get("questions")):
                        parsed = obj
                        break
                except json.JSONDecodeError:
                    continue

        if parsed is not None:
            raw_qs = parsed.get("questions")
            q_items: List[ClarificationQuestionItem] = []
            if isinstance(raw_qs, list):
                for idx, entry in enumerate(raw_qs[:3]):
                    if isinstance(entry, dict) and entry.get("question"):
                        opts = [str(o).strip() for o in (entry.get("options") or []) if str(o).strip()][:8]
                        q_items.append(
                            ClarificationQuestionItem(
                                slot_id=str(entry.get("id") or entry.get("slot_id") or f"slot_{idx+1}").strip(),
                                question=str(entry["question"]).strip(),
                                options=opts,
                                allow_custom=bool(entry.get("allow_custom", True)),
                            )
                        )
            if not q_items and parsed.get("question"):
                opts = [str(o).strip() for o in (parsed.get("options") or []) if str(o).strip()][:8]
                q_items.append(
                    ClarificationQuestionItem(
                        slot_id=str(parsed.get("id") or parsed.get("slot_id") or "parametre").strip(),
                        question=str(parsed["question"]).strip(),
                        options=opts,
                        allow_custom=bool(parsed.get("allow_custom", True)),
                    )
                )
            if q_items:
                return ClarificationRequest(
                    question=q_items[0].question,
                    options=list(q_items[0].options),
                    allow_custom=q_items[0].allow_custom,
                    reason=str(parsed.get("reason") or "").strip(),
                    raw_block=m.group(0),
                    start_idx=m.start(),
                    end_idx=m.end(),
                    slot_id=q_items[0].slot_id,
                    questions=q_items,
                )

        # Fallback: plain text question inside `<sodals_ask>Oteliniz nerede?</sodals_ask>`
        return ClarificationRequest(
            question=inner.strip().strip('"').strip("'"),
            options=[],
            allow_custom=True,
            raw_block=m.group(0),
            start_idx=m.start(),
            end_idx=m.end(),
        )
    return None


def resolve_clarification_answer(raw_input: str, req: ClarificationRequest) -> str:
    """
    Resolves single answers (`'1'` -> option[0], or free text) AND multi-slot replies
    (JSON dict, `'1, 2'` / `'1; 2'` paired with `req.questions`, or raw multi-part text).
    """
    cleaned = (raw_input or "").strip()
    if not cleaned:
        return ""

    if len(req.questions) > 1:
        # Support JSON map `{"konum": "Konyaaltı", "sure": "2 gün"}` or `{"0": "1", ...}`
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    parts = []
                    for q_item in req.questions:
                        val = str(obj.get(q_item.slot_id, "")).strip()
                        if val:
                            parts.append(f"{q_item.slot_id}={resolve_single_item_answer(val, q_item)}")
                    if parts:
                        return "; ".join(parts)
            except json.JSONDecodeError:
                pass
        # Support delimiter-separated multi-answers matching `len(req.questions)`
        sep_parts = [p.strip() for p in re.split(r"\s*[;|]\s*|\s*,\s*", cleaned) if p.strip()]
        if len(sep_parts) == len(req.questions):
            resolved_pairs = [
                f"{q_item.slot_id}={resolve_single_item_answer(val, q_item)}"
                for q_item, val in zip(req.questions, sep_parts)
            ]
            return "; ".join(resolved_pairs)

    if req.questions:
        return resolve_single_item_answer(cleaned, req.questions[0])
    return cleaned


def _strip_slot_prefix(raw_val: str, slot_id: str) -> str:
    cleaned = (raw_val or "").strip()
    prefix = f"{slot_id}="
    if slot_id and cleaned.lower().startswith(prefix.lower()):
        return cleaned[len(prefix) :].strip()
    return cleaned


def resolve_single_item_answer(raw_val: str, item: ClarificationQuestionItem) -> str:
    cleaned = _strip_slot_prefix(raw_val, item.slot_id)
    if cleaned.isdigit() and item.options:
        idx = int(cleaned) - 1
        if 0 <= idx < len(item.options):
            return item.options[idx]
    return cleaned


def lock_resolved_clarification(
    lock: ClarificationConstraintLock,
    raw_input: str,
    req: ClarificationRequest,
) -> str:
    """Resolves `raw_input` against `req` and registers hard Key-Lock entries."""
    cleaned = (raw_input or "").strip()
    if not cleaned:
        return ""

    if len(req.questions) > 1:
        resolved_parts: List[str] = []
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    for q_item in req.questions:
                        val = str(obj.get(q_item.slot_id, "")).strip()
                        if val:
                            chosen = resolve_single_item_answer(val, q_item)
                            lock.lock_answer(q_item.slot_id, chosen, q_item.question, q_item.options)
                            resolved_parts.append(f"{q_item.slot_id}={chosen}")
                    if resolved_parts:
                        return "; ".join(resolved_parts)
            except json.JSONDecodeError:
                pass

        sep_parts = [p.strip() for p in re.split(r"\s*[;|]\s*|\s*,\s*", cleaned) if p.strip()]
        if len(sep_parts) == len(req.questions):
            for q_item, val in zip(req.questions, sep_parts):
                chosen = resolve_single_item_answer(val, q_item)
                lock.lock_answer(q_item.slot_id, chosen, q_item.question, q_item.options)
                resolved_parts.append(f"{q_item.slot_id}={chosen}")
            return "; ".join(resolved_parts)

    chosen = resolve_clarification_answer(cleaned, req)
    slot_id = req.questions[0].slot_id if req.questions else (req.slot_id or "parametre")
    opts = req.questions[0].options if req.questions else req.options
    q_txt = req.questions[0].question if req.questions else req.question
    lock.lock_answer(slot_id, chosen, q_txt, opts)
    return chosen


def build_unfreeze_delta_text(user_answer: str, lock_directive: str = "") -> str:
    """
    Constructs ONLY the continuation delta tokens fed after the frozen
    `DynamicCache` / `<sodals_ask>` prefix so KV+DeltaNet cache does NOT re-prefill.
    """
    clean_ans = (user_answer or "").strip()
    lock_line = f"{lock_directive}\n" if lock_directive else ""
    return (
        f"\n<user_clarification>{clean_ans}</user_clarification>\n"
        f"{lock_line}"
        f"Kullanıcı netleştirdi: {clean_ans}. Kilitli önkoşullara sadık kalarak "
        f"düşünmeyi sonlandırıp (`</think>`) en optimum ve doğrudan yanıtı veriyorum:\n"
    )


def build_unfreeze_continuation_prompt(
    formatted_prompt: str,
    frozen_prefix: str,
    user_answer: str,
    lock_directive: str = "",
) -> str:
    """Text-fallback unfreeze prompt used when DynamicCache continuation is unavailable."""
    clean_prefix = (frozen_prefix or "").rstrip()
    return f"{formatted_prompt}{clean_prefix}{build_unfreeze_delta_text(user_answer, lock_directive=lock_directive)}"


def finalize_clean_history_and_display(
    raw_streamed_text: str,
    lock: Optional[ClarificationConstraintLock],
    clarifications_asked: int,
    prompt_opened_think: bool = False,
) -> Tuple[str, str]:
    """
    Enforces `<think>...</think>` tag balance & conversation history hygiene:
      - When `clarifications_asked == 0`: preserves output intact for exact compatibility.
      - When `clarifications_asked > 0`:
          * `display_text`: balances missing opening `<think>` if prompt opened `<think>`.
          * `clean_history`: strips `<think>...</think>`, `<sodals_ask>...</sodals_ask>`,
            `<user_clarification>...</user_clarification>`, and dangling closing tags,
            keeping ONLY `[Netleştirilen Bağlam: ...] + final clean answer`.
    """
    txt = raw_streamed_text or ""
    if clarifications_asked <= 0:
        return txt, txt

    display_txt = txt
    # Remove any leaked internal `<user_clarification>` / `<sodals_ask>` blocks from final text
    display_txt = re.sub(r"<user_clarification>.*?</user_clarification>", "", display_txt, flags=re.DOTALL | re.IGNORECASE)
    display_txt = re.sub(r"<(?:sodals_ask|ask_user)>.*?</(?:sodals_ask|ask_user)>", "", display_txt, flags=re.DOTALL | re.IGNORECASE)

    # Balance `<think>` / `</think>` for displayed text
    has_open = "<think>" in display_txt
    has_close = "</think>" in display_txt
    if has_close and not has_open and prompt_opened_think:
        display_txt = "<think>\n" + display_txt.lstrip()
    elif has_open and not has_close:
        display_txt = display_txt.rstrip() + "\n</think>"

    # Clean history: keep ONLY clarification summary + final user-facing answer
    clean_body = re.sub(r"<think>.*?</think>", "", display_txt, flags=re.DOTALL | re.IGNORECASE)
    clean_body = clean_body.replace("<think>", "").replace("</think>", "").strip()
    summary_prefix = lock.build_clean_summary() if lock is not None else ""
    if summary_prefix and clean_body:
        clean_history = f"{summary_prefix}\n\n{clean_body}"
    else:
        clean_history = clean_body or summary_prefix or txt
    return display_txt.strip(), clean_history.strip()


def format_cli_clarification_prompt(req: ClarificationRequest) -> str:
    """Formats an inline terminal clarification box (supports single or multi-slot)."""
    lines = ["\n\033[1;36m⏸ [Yanıt Donduruldu — Netleştirme Gerekiyor]\033[0m"]
    if req.reason:
        lines.append(f"  \033[90m(Neden: {req.reason})\033[0m")

    q_items = req.questions or [
        ClarificationQuestionItem(slot_id=req.slot_id, question=req.question, options=req.options, allow_custom=req.allow_custom)
    ]
    for q_idx, item in enumerate(q_items, start=1):
        prefix = f"{q_idx}. " if len(q_items) > 1 else "? "
        lines.append(f"\033[1;37m{prefix}{item.question}\033[0m")
        for idx, opt in enumerate(item.options, start=1):
            lines.append(f"  \033[33m{idx})\033[0m {opt}")
    if len(q_items) > 1:
        lines.append("  \033[90m[Her soru için numara veya yanıtınızı sırayla girebilirsiniz]\033[0m")
    else:
        hint = "Numara seçin veya kendi yanıtınızı yazın" if q_items[0].options else "Yanıtınızı yazın"
        lines.append(f"  \033[90m[{hint}]\033[0m")
    return "\n".join(lines)


def format_stateless_clarification_reply(req: ClarificationRequest, freeze_id: str) -> str:
    """Formats a user-readable clarification message when no blocking callback is attached."""
    lines: List[str] = []
    q_items = req.questions or [
        ClarificationQuestionItem(slot_id=req.slot_id, question=req.question, options=req.options, allow_custom=req.allow_custom)
    ]
    if len(q_items) == 1:
        lines.append(f"**Netleştirme Gerekiyor:** {q_items[0].question}")
        if req.reason:
            lines.append(f"_{req.reason}_")
        if q_items[0].options:
            lines.append("")
            for idx, opt in enumerate(q_items[0].options, start=1):
                lines.append(f"{idx}. {opt}")
    else:
        lines.append("**Netleştirme Gerekiyor (Çoklu Parametre):**")
        if req.reason:
            lines.append(f"_{req.reason}_")
        for q_i, item in enumerate(q_items, start=1):
            lines.append(f"\n**{q_i}) {item.question}**")
            for idx, opt in enumerate(item.options, start=1):
                lines.append(f"   - {idx}: {opt}")
    lines.append(f"\n_(Durum donduruldu `freeze_id={freeze_id}`; yanıtınızı gönderdiğiniz anda kaldığım yerden devam edeceğim.)_")
    return "\n".join(lines)


def build_mid_turn_stopping_criteria(
    tokenizer: Any,
    prompt_token_len: int,
    entropy_processor: Optional[EntropyAndSkewGProcessor] = None,
    cancel_event: Optional[threading.Event] = None,
):
    """
    Creates a Transformers `StoppingCriteriaList` that halts `model.generate()` on:
      1. Closing `</sodals_ask>` / `</ask_user>` tag or fenced ```sodals_ask...``` completion, OR
      2. `EntropyAndSkewGProcessor.watchdog_tripped` (sustained high Shannon entropy +
         missing preconditions without an explicit `<sodals_ask>`), OR
      3. External `cancel_event` (cleanly terminates GPU worker if stream is closed early).
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _MidTurnStopOnAsk(StoppingCriteria):
        def __init__(
            self,
            tok: Any,
            start_len: int,
            ent_proc: Optional[EntropyAndSkewGProcessor],
            cancel_ev: Optional[threading.Event],
        ):
            self.tok = tok
            self.start_len = int(start_len)
            self.ent_proc = ent_proc
            self.cancel_ev = cancel_ev
            self.triggered = False
            self.trigger_reason = ""  # "model_tag" | "entropy_watchdog"
            self._fenced_open = False

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if self.ent_proc is not None and self.ent_proc.watchdog_tripped:
                self.triggered = True
                self.trigger_reason = "entropy_watchdog"
                return True

            cur_len = int(input_ids.shape[1])
            if cur_len <= self.start_len:
                return bool(self.cancel_ev is not None and self.cancel_ev.is_set())

            # Decode only a tight trailing window covering opening/closing ask tags
            tail_start = max(self.start_len, cur_len - 28)
            tail_txt = self.tok.decode(input_ids[0, tail_start:], skip_special_tokens=False)
            tail_low = tail_txt.lower()

            if "<sodals_ask>" in tail_low or "<ask_user>" in tail_low:
                if self.ent_proc is not None:
                    self.ent_proc.notify_ask_tag_opened()
            if "```sodals_ask" in tail_low or "```ask_user" in tail_low:
                self._fenced_open = True
                if self.ent_proc is not None:
                    self.ent_proc.notify_ask_tag_opened()

            for tag in CLOSE_TAGS:
                if tag in tail_low:
                    self.triggered = True
                    self.trigger_reason = "model_tag"
                    return True

            if self._fenced_open:
                full_gen = self.tok.decode(input_ids[0, self.start_len :], skip_special_tokens=False)
                if _ASK_PATTERNS[2].search(full_gen):
                    self.triggered = True
                    self.trigger_reason = "model_tag"
                    return True

            if self.cancel_ev is not None and self.cancel_ev.is_set():
                return True
            return False

    return StoppingCriteriaList(
        [_MidTurnStopOnAsk(tokenizer, prompt_token_len, entropy_processor, cancel_event)]
    )


class ClarificationStreamFilter:
    """
    Buffers potential partial `<sodals_ask` / `<ask_user` tags during streaming so
    raw XML markers never leak to the client, and cleanly separates prefix reasoning
    from a completed `ClarificationRequest`.
    """

    def __init__(self) -> None:
        self._raw_total: str = ""
        self._emitted_len: int = 0
        self.pending_request: Optional[ClarificationRequest] = None
        self.frozen_prefix: str = ""

    @property
    def raw_text(self) -> str:
        return self._raw_total

    def feed(self, chunk: str) -> Tuple[str, Optional[ClarificationRequest]]:
        if self.pending_request is not None:
            return "", self.pending_request

        self._raw_total += chunk or ""
        req = parse_clarification_request(self._raw_total)
        if req is not None:
            self.pending_request = req
            self.frozen_prefix = self._raw_total[: req.end_idx]
            safe_end = min(req.start_idx, len(self._raw_total))
            emit = ""
            if safe_end > self._emitted_len:
                emit = self._raw_total[self._emitted_len : safe_end]
                self._emitted_len = safe_end
            return emit, req

        unemitted = self._raw_total[self._emitted_len :]
        unemitted_low = unemitted.lower()
        hold_from = len(self._raw_total)

        for open_marker in ("<sodals_ask>", "<ask_user>", "```sodals_ask", "```ask_user"):
            pos = unemitted_low.find(open_marker)
            if pos != -1:
                hold_from = min(hold_from, self._emitted_len + pos)

        if hold_from == len(self._raw_total):
            for sep in ("<", "```"):
                pos = unemitted_low.rfind(sep)
                if pos != -1:
                    tail = unemitted_low[pos:]
                    if any(m.startswith(tail) for m in ("<sodals_ask>", "<ask_user>", "```sodals_ask", "```ask_user")):
                        hold_from = min(hold_from, self._emitted_len + pos)
                        break

        emit = ""
        if hold_from > self._emitted_len:
            emit = self._raw_total[self._emitted_len : hold_from]
            self._emitted_len = hold_from
        return emit, None

    def flush(self) -> str:
        if self.pending_request is not None:
            return ""
        if len(self._raw_total) > self._emitted_len:
            rem = self._raw_total[self._emitted_len :]
            self._emitted_len = len(self._raw_total)
            return rem
        return ""
