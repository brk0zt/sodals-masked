"""
Unit & Integration Tests for Mid-Turn Thought-Freezing Clarification (`clarification_freeze.py` + `sodals_serve.py`)
"""

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

SFT_DIR = Path(__file__).resolve().parents[2]
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

import torch

from clarification_freeze import (
    CLARIFICATION_PROTOCOL_PROMPT,
    PLANNING_OR_AMBIGUOUS_CUE_RE,
    ClarificationStreamFilter,
    build_mid_turn_stopping_criteria,
    build_unfreeze_continuation_prompt,
    parse_clarification_request,
    resolve_clarification_answer,
)
from sodals_serve import (
    DEFAULT_SAFETENSORS_DIR,
    SodalsNeurosymbolicEngine,
    _flatten_anthropic_messages,
    resolve_runtime_model_path,
)


class _MiniTokenizer:
    """Minimal mock tokenizer supporting `encode` and `apply_chat_template`."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in (text or "")]

    def decode(self, ids, skip_special_tokens: bool = False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        suffix = "<assistant><think>\n" if enable_thinking else "<assistant><think>\n\n</think>\n\n"
        return "\n".join(parts) + "\n" + suffix


class TestClarificationFreeze(unittest.TestCase):
    """Validates mid-turn `<sodals_ask>` parsing, streaming isolation, StoppingCriteria, and unfreeze resume."""

    def test_parse_clarification_and_option_resolution(self):
        raw = (
            "<think>\nAntalya çok geniş bir il; otel Konyaaltı'nda mı Alanya'da mı bilinmeden rota verilemez.\n"
            "<sodals_ask>{\"question\": \"Oteliniz Antalya'nın hangi ilçesinde?\", "
            "\"options\": [\"Konyaaltı\", \"Lara\", \"Alanya\",],}</sodals_ask>"
        )
        req = parse_clarification_request(raw)
        self.assertIsNotNone(req)
        self.assertIn("hangi ilçesinde", req.question)
        self.assertEqual(req.options, ["Konyaaltı", "Lara", "Alanya"])
        self.assertEqual(resolve_clarification_answer("1", req), "Konyaaltı")
        self.assertEqual(resolve_clarification_answer("Belek merkez", req), "Belek merkez")

    def test_stream_filter_holds_split_tags_and_contains_lt_inside_json(self):
        flt = ClarificationStreamFilter()
        chunks = [
            "<think>\nÖnce otelin ",
            "konumunu sormam gerek.<sod",
            "als_ask>{\"question\": \"Süre (< 3 gün mü) ve semt?\", ",
            "\"options\": [\"Konyaaltı\", \"Alanya\"]}</sodals_",
            "ask>BUNLAR_YAZILMAMALI",
        ]
        emitted = []
        frozen_req = None
        for ch in chunks:
            out, req = flt.feed(ch)
            if out:
                emitted.append(out)
            if req is not None:
                frozen_req = req
                break

        self.assertIsNotNone(frozen_req)
        full_emitted = "".join(emitted)
        self.assertEqual(full_emitted, "<think>\nÖnce otelin konumunu sormam gerek.")
        self.assertNotIn("<sodals_ask>", full_emitted)
        self.assertNotIn("BUNLAR_YAZILMAMALI", full_emitted)
        self.assertIn("</sodals_ask>", flt.frozen_prefix)

    def test_stopping_criteria_triggers_on_closing_tag(self):
        tok = _MiniTokenizer()
        prompt_ids = tok.encode("prompt:")
        criteria = build_mid_turn_stopping_criteria(tok, len(prompt_ids))

        cont_ids = torch.tensor([prompt_ids + tok.encode("<think>düşünüyorum")], dtype=torch.long)
        self.assertFalse(criteria(cont_ids, None))

        ask_ids = torch.tensor(
            [prompt_ids + tok.encode("<think><sodals_ask>{\"question\":\"Nerede?\"}</sodals_ask>")],
            dtype=torch.long,
        )
        self.assertTrue(criteria(ask_ids, None))

    def test_engine_mid_turn_freeze_callback_and_unfreeze_resume(self):
        eng = object.__new__(SodalsNeurosymbolicEngine)
        eng.tokenizer = _MiniTokenizer()
        eng.device = "cpu"
        eng.no_system_prompt = False
        eng.svon_adapter_loaded = False
        eng.svon_cgw_active = False
        eng.interactive_clarify = True
        eng.ode_module = None
        eng.blender = SimpleNamespace(compute_rep_penalty=lambda _a: 1.0, set_blend=lambda _m, _a: None)
        eng.lock = threading.Lock()
        eng._session_lock = threading.Lock()
        eng._frozen_sessions = {}
        eng._frozen_by_sig = {}
        eng.total_tokens_generated = 0
        eng.total_requests_served = 0

        captured_prompts = []

        def _fake_stream(formatted_prompt, **_kwargs):
            captured_prompts.append(formatted_prompt)
            if len(captured_prompts) == 1:
                yield "<think>\nOtelin hangi ilçede olduğunu bilmeden Antalya rotası kurulamaz."
                yield "<sodals_ask>{\"question\": \"Oteliniz Antalya'nın hangi ilçesinde?\", \"options\": [\"Konyaaltı\", \"Alanya\"]}</sodals_ask>BU_YAZILMAMALI"
            else:
                yield "Otel Konyaaltı'nda olduğu için önce Konyaaltı Sahili'nden başlıyoruz.</think>\n\n1. Konyaaltı Sahili & Akvaryum\n2. Kaleiçi"

        eng._run_streaming_pass = _fake_stream

        events = list(
            eng.stream_generate(
                [{"role": "user", "content": "Antalya'da otelime yeni geldim. Bana gezilecek yerlerle ilgili bir rota öner"}],
                clarify_callback=lambda _req, _fid: "1",
            )
        )

        # Ensure prompt 1 requested thinking + clarification protocol
        self.assertIn("<sodals_ask>", captured_prompts[0])
        self.assertIn("<think>", captured_prompts[0])

        # Ensure prompt 2 resumed straight from the exact frozen state + <user_clarification>Konyaaltı</user_clarification>
        self.assertEqual(len(captured_prompts), 2)
        self.assertIn("Otelin hangi ilçede olduğunu bilmeden Antalya rotası kurulamaz.", captured_prompts[1])
        self.assertIn("<user_clarification>Konyaaltı</user_clarification>", captured_prompts[1])

        ev_types = [e["type"] for e in events]
        self.assertIn("clarification_required", ev_types)
        self.assertIn("clarification_resolved", ev_types)
        # Ensure display text balances `<think>...</think>` while `clean_history` strips internal tags cleanly
        done = [e for e in events if e["type"] == "done"][0]
        self.assertEqual(done["clarifications_asked"], 1)
        self.assertFalse(done["clarification_pending"])
        self.assertIn("Konyaaltı Sahili", done["full_text"])
        self.assertIn("<think>", done["full_text"])
        self.assertIn("</think>", done["full_text"])
        self.assertNotIn("BU_YAZILMAMALI", done["full_text"])
        self.assertIn("Konyaaltı", done["locked_constraints"].values())
        self.assertNotIn("<think>", done["clean_history"])
        self.assertNotIn("</think>", done["clean_history"])
        self.assertIn("[Netleştirilen Bağlam: Konyaaltı]", done["clean_history"])
        self.assertIn("1. Konyaaltı Sahili & Akvaryum", done["clean_history"])

    def test_multi_slot_questions_and_entropy_watchdog_fallback(self):
        multi_raw = (
            "<think>İki detay eksik.<sodals_ask>{\"questions\": ["
            "{\"id\": \"konum\", \"question\": \"Otel hangi ilçede?\", \"options\": [\"Konyaaltı\", \"Alanya\"]},"
            "{\"id\": \"sure\", \"question\": \"Kaç günlük rota?\", \"options\": [\"Yarım gün\", \"2 gün\"]}"
            "]}</sodals_ask>"
        )
        req = parse_clarification_request(multi_raw)
        self.assertIsNotNone(req)
        self.assertEqual(len(req.questions), 2)
        self.assertEqual(
            resolve_clarification_answer("1, 2", req),
            "konum=Konyaaltı; sure=2 gün",
        )

        # Test Epistemic Entropy Watchdog fallback when model forgets to emit `<sodals_ask>`
        eng = object.__new__(SodalsNeurosymbolicEngine)
        eng.tokenizer = _MiniTokenizer()
        eng.device = "cpu"
        eng.no_system_prompt = False
        eng.svon_adapter_loaded = False
        eng.svon_cgw_active = False
        eng.interactive_clarify = True
        eng.ode_module = None
        eng.blender = SimpleNamespace(compute_rep_penalty=lambda _a: 1.0, set_blend=lambda _m, _a: None)
        eng.lock = threading.Lock()
        eng._session_lock = threading.Lock()
        eng._frozen_sessions = {}
        eng._frozen_by_sig = {}
        eng.total_tokens_generated = 0
        eng.total_requests_served = 0

        calls = {"n": 0}

        def _fake_watchdog_pass(formatted_prompt, use_adapters, max_new_tokens, temperature, top_p, rep_penalty, **kwargs):
            calls["n"] += 1
            out_dict = kwargs.get("freeze_state_out")
            if calls["n"] == 1:
                yield "Antalya için birçok ilçe seçeneği var ama otel bilgisi belirsiz."
                if isinstance(out_dict, dict):
                    out_dict["watchdog_tripped"] = True
                    out_dict["entropy_summary"] = {"mean_entropy": 2.8, "watchdog_tripped": True}
            else:
                yield "Konyaaltı odaklı rota hazır.</think>\n\n1. Konyaaltı Plajı\n2. Kaleiçi"

        eng._run_streaming_pass = _fake_watchdog_pass

        events = list(
            eng.stream_generate(
                [{"role": "user", "content": "Antalya'da otelime yeni geldim. Bana gezilecek yerlerle ilgili bir rota öner"}],
                clarify_callback=lambda _req, _fid: "1",
            )
        )
        req_ev = [e for e in events if e["type"] == "clarification_required"][0]
        self.assertEqual(req_ev["trigger_source"], "entropy_watchdog")
        self.assertIn("Konyaaltı", req_ev["options"])
        done = [e for e in events if e["type"] == "done"][0]
        self.assertEqual(done["clarifications_asked"], 1)
        self.assertIn("Konyaaltı", done["locked_constraints"].values())
        self.assertIn("[Netleştirilen Bağlam: Konyaaltı]", done["clean_history"])

    def test_gguf_model_path_resolver_and_anthropic_flattener(self):
        # Passing a .gguf path cleanly resolves to the SafeTensors stack when available (zero GGUF load crash)
        resolved_dir, gguf_file = resolve_runtime_model_path(
            r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B_Q4_K_M.gguf"
        )
        if os.path.exists(os.path.join(DEFAULT_SAFETENSORS_DIR, "config.json")):
            self.assertEqual(resolved_dir, DEFAULT_SAFETENSORS_DIR)
            self.assertIsNone(gguf_file)

        flat = _flatten_anthropic_messages(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Merhaba Antalya"}]}],
            system=[{"type": "text", "text": "Kısa yanıt ver"}],
        )
        self.assertEqual(flat[0], {"role": "system", "content": "Kısa yanıt ver"})
        self.assertEqual(flat[1], {"role": "user", "content": "Merhaba Antalya"})

    def test_stateless_http_freeze_and_resume_with_submit_clarification(self):
        eng = object.__new__(SodalsNeurosymbolicEngine)
        eng.tokenizer = _MiniTokenizer()
        eng.device = "cpu"
        eng.no_system_prompt = False
        eng.svon_adapter_loaded = False
        eng.svon_cgw_active = False
        eng.interactive_clarify = True
        eng.ode_module = None
        eng.blender = SimpleNamespace(compute_rep_penalty=lambda _a: 1.0, set_blend=lambda _m, _a: None)
        eng.lock = threading.Lock()
        eng._session_lock = threading.Lock()
        eng._frozen_sessions = {}
        eng._frozen_by_sig = {}
        eng.total_tokens_generated = 0
        eng.total_requests_served = 0

        prompts_seen = []

        def _fake_stream(formatted_prompt, **_kwargs):
            prompts_seen.append(formatted_prompt)
            if len(prompts_seen) == 1:
                yield "<think>\nİlçe bilgisi eksik."
                yield "<sodals_ask>{\"question\": \"Oteliniz hangi bölgede?\", \"options\": [\"Konyaaltı\", \"Lara\"]}</sodals_ask>"
            else:
                yield "Lara bölgesine göre rota hazır.</think>\n\n1. Düden Şelalesi"

        eng._run_streaming_pass = _fake_stream

        turn1 = list(
            eng.stream_generate(
                [{"role": "user", "content": "Antalya'da otelime yeni geldim. Bana gezilecek yerlerle ilgili bir rota öner"}],
                clarify_callback=None,
            )
        )
        done1 = [e for e in turn1 if e["type"] == "done"][0]
        self.assertTrue(done1["clarification_pending"])
        freeze_id = done1["freeze_id"]

        # Test explicit `submit_clarification` unfreezes answer_event too
        resolved = eng.submit_clarification(freeze_id, "2")
        self.assertEqual(resolved, "Lara")

        # Next stateless turn from client with `[orig_user, assistant_ask, "2"]` auto-resumes frozen prefix!
        turn2 = list(
            eng.stream_generate(
                [
                    {"role": "user", "content": "Antalya'da otelime yeni geldim. Bana gezilecek yerlerle ilgili bir rota öner"},
                    {"role": "assistant", "content": done1["full_text"]},
                    {"role": "user", "content": "2"},
                ],
                clarify_callback=None,
            )
        )
        done2 = [e for e in turn2 if e["type"] == "done"][0]
        self.assertFalse(done2["clarification_pending"])
        self.assertIn("<user_clarification>Lara</user_clarification>", prompts_seen[1])
        self.assertIn("Düden Şelalesi", done2["full_text"])

    def test_submit_clarification_multislot_idempotence_and_no_self_rejection(self):
        from clarification_freeze import (
            ClarificationQuestionItem,
            ClarificationRequest,
            FrozenClarificationSession,
            lock_resolved_clarification,
        )
        from epistemic_clarify_gate import ClarificationConstraintLock
        from sodals_serve import SodalsNeurosymbolicEngine, _build_http_wait_callback

        req = ClarificationRequest(
            question="Oteliniz hangi bölgede?",
            options=["Konyaaltı", "Lara", "Alanya"],
            slot_id="konum",
            questions=[
                ClarificationQuestionItem(slot_id="konum", question="Oteliniz hangi bölgede?", options=["Konyaaltı", "Lara", "Alanya"]),
                ClarificationQuestionItem(slot_id="sure", question="Kaç gün kalacaksınız?", options=["1 gün", "2 gün", "3+ gün"]),
            ],
        )
        eng = object.__new__(SodalsNeurosymbolicEngine)
        eng._session_lock = __import__("threading").Lock()
        eng._frozen_sessions = {}
        eng._frozen_by_sig = {}
        sess = FrozenClarificationSession(
            freeze_id="multi123",
            formatted_prompt="",
            frozen_prefix="",
            pending_request=req,
            use_adapters=False,
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.9,
            rep_penalty=1.0,
            mode_value="SYSTEM2_CAUSAL_LOGIC",
            alpha=0.0,
            rk4_steps=0,
            domain_label="Travel",
            user_query="Antalya rotası",
        )
        eng._frozen_sessions["multi123"] = sess
        eng.submit_clarification("multi123", "1, 2")

        wait_cb = _build_http_wait_callback(eng, timeout_sec=1.0)
        raw_back = wait_cb(req, "multi123")
        lock = ClarificationConstraintLock()
        resolved = lock_resolved_clarification(lock, raw_back, req)
        self.assertEqual(resolved, "konum=Konyaaltı; sure=2 gün")
        self.assertEqual(lock.locked_slots["konum"], "Konyaaltı")
        self.assertEqual(lock.locked_slots["sure"], "2 gün")
        self.assertNotIn("Konyaaltı", lock.rejected_alternatives)
        self.assertNotIn("2 gün", lock.rejected_alternatives)
        self.assertIn("Lara", lock.rejected_alternatives)

        # Even passing already-prefixed text into lock_resolved_clarification must stay idempotent!
        lock2 = ClarificationConstraintLock()
        resolved2 = lock_resolved_clarification(lock2, "konum=Konyaaltı; sure=2 gün", req)
        self.assertEqual(resolved2, "konum=Konyaaltı; sure=2 gün")
        self.assertEqual(lock2.locked_slots["konum"], "Konyaaltı")
        self.assertNotIn("Konyaaltı", lock2.rejected_alternatives)

    def test_stateless_resume_strips_pass2_think_from_clean_history(self):
        from sodals_serve import SodalsNeurosymbolicEngine

        class _DummyTok:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
                return "<|im_start|>assistant\n<think>\n" if enable_thinking else "<|im_start|>assistant\n"

            def encode(self, txt, add_special_tokens=False):
                return list(range(max(1, len(txt.split()))))

        eng = object.__new__(SodalsNeurosymbolicEngine)
        eng.tokenizer = _DummyTok()
        eng.interactive_clarify = True
        eng.no_system_prompt = False
        eng.svon_adapter_loaded = False
        eng._session_lock = __import__("threading").Lock()
        eng._frozen_sessions = {}
        eng._frozen_by_sig = {}
        eng.total_tokens_generated = 0
        eng.total_requests_served = 0
        eng.ode_module = None
        eng.blender = None
        eng.model = None
        eng.route_prompt = lambda _q: (
            __import__("arkhe_neurosymbolic_pipeline").CognitiveMode.SYSTEM1_DIRECT,
            0.0,
            0,
            1.0,
            "Travel",
        )
        call_n = {"n": 0}

        def _fake_pass(formatted_prompt, use_adapters, max_new_tokens, temperature, top_p, rep_penalty, stop_on_clarification=True):
            call_n["n"] += 1
            if call_n["n"] == 1:
                yield 'Önce konum lazım. <sodals_ask>{"question": "Hangi ilçe?", "options": ["Konyaaltı", "Lara"]}</sodals_ask>'
            else:
                yield "Konyaaltı seçildi, rotayı planlıyorum.\n</think>\n\n1. Konyaaltı Sahili\n2. Kaleiçi"

        eng._run_streaming_pass = _fake_pass
        turn1 = list(
            eng.stream_generate(
                [{"role": "user", "content": "Antalya'da otelime yeni geldim. Rota öner"}],
                clarify_callback=None,
            )
        )
        done1 = [e for e in turn1 if e["type"] == "done"][0]
        turn2 = list(
            eng.stream_generate(
                [
                    {"role": "user", "content": "Antalya'da otelime yeni geldim. Rota öner"},
                    {"role": "assistant", "content": done1["full_text"]},
                    {"role": "user", "content": "1"},
                ],
                clarify_callback=None,
            )
        )
        done2 = [e for e in turn2 if e["type"] == "done"][0]
        self.assertNotIn("Konyaaltı seçildi, rotayı planlıyorum", done2["clean_history"])
        self.assertNotIn("</think>", done2["clean_history"])
        self.assertIn("1. Konyaaltı Sahili", done2["clean_history"])

    def test_anthropic_flatten_preserves_tool_use_and_list_tool_result(self):
        from sodals_serve import _flatten_anthropic_messages

        flat = _flatten_anthropic_messages(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Dosyayı okuyorum."},
                        {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "app.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "print('ok')"}]},
                    ],
                },
            ],
            system=[{"type": "text", "text": "System instructions"}],
        )
        self.assertEqual(flat[0]["role"], "system")
        self.assertIn("read_file", flat[1]["content"])
        self.assertIn("app.py", flat[1]["content"])
        self.assertIn("print('ok')", flat[2]["content"])


if __name__ == "__main__":
    unittest.main()
