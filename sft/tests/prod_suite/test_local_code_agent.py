"""
SODALS v2.1 — GPU-Free Unit Tests for Offline Coding & Automation Agent
=======================================================================
Tests:
  1. `LocalWorkspaceToolbox`: `list_dir`, `find_files`, `grep_search`, `read_file`,
     and workspace-escape `PermissionError` guard.
  2. Safe editing + undo stack: `write_file`, `edit_file` (unique/ambiguous/missing checks),
     unified diff output, and `undo_last_edit`.
  3. Offline `run_command` execution, dangerous-command blocklist, and interactive confirmation gate.
  4. `parse_tool_calls` (`<think>` exclusion, nested `{}` braces in JSON code payloads,
     `<sodals_tool>` / `<tool_call>` / fenced formats) and `strip_tool_blocks`.
  5. `SodalsNeurosymbolicEngine.run_agent_turn` multi-step ReAct loop (read -> edit -> run -> report)
     completely offline and GPU-free.
"""

import json
import os
import sys
import tempfile
import threading
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.dirname(os.path.dirname(TESTS_DIR))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32":
    import smart_app_control_compat  # noqa: F401
from local_code_agent import (
    LocalWorkspaceToolbox,
    parse_tool_calls,
    strip_tool_blocks,
)
import sodals_serve


class _DummyAgentTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        self.last_messages = messages
        return "<prompt>"

    def encode(self, text, add_special_tokens=False):
        return text.split() or ["tok"]


class TestLocalCodeAgent(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = self.tmp_dir.name
        self.toolbox = LocalWorkspaceToolbox(workspace_root=self.root)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_navigation_search_read_and_boundary_guard(self):
        sub = os.path.join(self.root, "src")
        os.makedirs(sub, exist_ok=True)
        fpath = os.path.join(sub, "calc.py")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("def compute():\n    x = 10\n    return x * 2\n")

        ls_res = self.toolbox.list_dir(".", depth=2)
        self.assertTrue(ls_res.ok)
        self.assertIn("calc.py", ls_res.output)

        find_res = self.toolbox.find_files("*.py")
        self.assertTrue(find_res.ok)
        self.assertIn("src/calc.py", find_res.output)

        grep_res = self.toolbox.grep_search("return x \\* 2")
        self.assertTrue(grep_res.ok)
        self.assertIn("src/calc.py:3:", grep_res.output)

        read_res = self.toolbox.read_file("src/calc.py", start_line=1, end_line=3)
        self.assertTrue(read_res.ok)
        self.assertIn("2:     x = 10", read_res.output)

        outside = os.path.abspath(os.path.join(self.root, "..", "outside_escape.txt"))
        with self.assertRaises(PermissionError):
            self.toolbox.resolve_path(outside)

    def test_write_edit_diff_and_undo(self):
        w_res = self.toolbox.write_file("hello.py", "def f():\n    return 1\n", overwrite=False)
        self.assertTrue(w_res.ok)
        self.assertIn("return 1", w_res.diff)

        # Should refuse overwrite=False if file already exists
        w_fail = self.toolbox.write_file("hello.py", "overwrite", overwrite=False)
        self.assertFalse(w_fail.ok)

        # Edit missing target -> fail
        e_miss = self.toolbox.edit_file("hello.py", "return 999", "return 2")
        self.assertFalse(e_miss.ok)

        # Valid edit -> success + diff
        e_ok = self.toolbox.edit_file("hello.py", "return 1", "return 42")
        self.assertTrue(e_ok.ok)
        self.assertIn("-    return 1", e_ok.diff)
        self.assertIn("+    return 42", e_ok.diff)

        # Undo edit restores `return 1`, second undo removes newly created file
        u1 = self.toolbox.undo_last_edit()
        self.assertTrue(u1.ok)
        with open(os.path.join(self.root, "hello.py"), "r", encoding="utf-8") as f:
            self.assertIn("return 1", f.read())

        u2 = self.toolbox.undo_last_edit()
        self.assertTrue(u2.ok)
        self.assertFalse(os.path.exists(os.path.join(self.root, "hello.py")))

    def test_run_command_confirmation_and_safety_blocklist(self):
        # Confirmation denial blocks mutating tool
        denied = self.toolbox.execute_tool(
            {"tool": "run_command", "command": f'"{sys.executable}" -c "print(123)"'},
            auto_approve=False,
            confirm_callback=lambda name, call: False,
        )
        self.assertFalse(denied.ok)
        self.assertTrue(denied.aborted_by_user)

        # Confirmation approval runs offline Python command cleanly
        approved = self.toolbox.execute_tool(
            {"tool": "run_command", "command": f'& "{sys.executable}" -c "print(6*7)"'},
            auto_approve=False,
            confirm_callback=lambda name, call: True,
        )
        self.assertTrue(approved.ok)
        self.assertIn("42", approved.output)

        # Safety blocklist rejects destructive system command even if auto_approve=True
        blocked = self.toolbox.execute_tool(
            {"tool": "run_command", "command": "format C: /q"},
            auto_approve=True,
        )
        self.assertFalse(blocked.ok)
        self.assertIn("Güvenlik kilidi", blocked.output)

    def test_parse_tool_calls_excludes_think_and_supports_nested_braces(self):
        sample = (
            "<think>\n"
            'Örnek düşünce: <sodals_tool>{"tool": "read_file", "path": "ignored.py"}</sodals_tool>\n'
            "</think>\n"
            '<sodals_tool>{"tool": "write_file", "path": "d.py", "content": "d = {\'a\': 1}\\nprint(d)", "overwrite": true}</sodals_tool>'
        )
        calls = parse_tool_calls(sample)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "write_file")
        self.assertEqual(calls[0]["path"], "d.py")
        self.assertIn("{'a': 1}", calls[0]["content"])
        self.assertEqual(strip_tool_blocks(sample), "")

    def test_run_agent_turn_end_to_end_offline_loop(self):
        app_path = os.path.join(self.root, "app.py")
        with open(app_path, "w", encoding="utf-8") as f:
            f.write("def val():\n    return 1\n\nif __name__ == '__main__':\n    print(val())\n")

        engine = object.__new__(sodals_serve.SodalsNeurosymbolicEngine)
        engine.lock = threading.Lock()
        engine.tokenizer = _DummyAgentTokenizer()
        engine.total_tokens_generated = 0
        engine.total_requests_served = 0

        cmd_payload = json.dumps({"tool": "run_command", "command": f'& "{sys.executable}" app.py'})
        scripted_responses = [
            '<think>Önce dosyayı okuyayım.</think>\n<sodals_tool>{"tool": "read_file", "path": "app.py", "start_line": 1, "end_line": 10}</sodals_tool>',
            '<sodals_tool>{"tool": "edit_file", "path": "app.py", "target": "    return 1", "replacement": "    return 99"}</sodals_tool>',
            f"<sodals_tool>{cmd_payload}</sodals_tool>",
            "Düzenleme yapıldı ve `app.py` çalıştırılarak `99` çıktısı doğrulandı.",
        ]
        call_idx = {"i": 0}
        seen_kwargs = []

        def _fake_pass(formatted_prompt, use_adapters, max_new_tokens, temperature, top_p, rep_penalty):
            seen_kwargs.append((use_adapters, rep_penalty, max_new_tokens))
            idx = call_idx["i"]
            call_idx["i"] += 1
            yield scripted_responses[idx]

        engine._run_streaming_pass = _fake_pass

        packets = list(
            engine.run_agent_turn(
                [{"role": "user", "content": "app.py içindeki dönüş değerini 99 yap ve test et"}],
                toolbox=self.toolbox,
                auto_approve=True,
                max_steps=6,
            )
        )
        tool_results = [p["result"] for p in packets if p["type"] == "tool_result"]
        done = next(p for p in packets if p["type"] == "done")

        self.assertEqual(len(tool_results), 3)
        self.assertTrue(all(r.ok for r in tool_results))
        self.assertIn("99", tool_results[2].output)
        self.assertEqual(done["tool_calls_executed"], 3)
        self.assertEqual(done["steps"], 4)
        self.assertIn("99", done["full_text"])
        self.assertTrue(all((ua is False and rp == 1.0 and mt >= 1536) for ua, rp, mt in seen_kwargs))

    def test_wave1_security_and_parser_guards(self):
        # 1. Non-numeric timeout_sec returns ToolExecutionResult(ok=False) instead of crashing
        bad_to = self.toolbox.execute_tool(
            {"tool": "run_command", "command": "echo ok", "timeout_sec": "not-a-number"},
            auto_approve=True,
        )
        self.assertFalse(bad_to.ok)
        self.assertIn("timeout_sec", bad_to.output)

        # 2. Unclosed <think> trailing tool block is NOT parsed/executed
        unclosed = '<think>taslak <sodals_tool>{"tool": "read_file", "path": "x.py"}</sodals_tool>'
        self.assertEqual(parse_tool_calls(unclosed), [])

        # 3. edit_file refuses non-UTF-8 files to prevent silent replacement-character corruption
        bad_utf8 = os.path.join(self.root, "latin.txt")
        with open(bad_utf8, "wb") as fb:
            fb.write(b"abc \xff\xfe def")
        res = self.toolbox.edit_file("latin.txt", "abc", "xyz")
        self.assertFalse(res.ok)
        self.assertIn("UTF-8", res.output)


if __name__ == "__main__":
    unittest.main()
