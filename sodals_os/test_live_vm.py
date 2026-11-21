#!/usr/bin/env python3
r"""
Live End-to-End Verification Suite for SODALS-OS (Alpine Linux QEMU Micro-VM)
=============================================================================
Verifies all guarantees against the LIVE running VM at http://127.0.0.1:11440:
  1. Real Linux kernel + unprivileged `svon` user + ZERO host C:\ / 9p / virtfs mounts
  2. Live public web page open (`https://example.com`) + text & link extraction
  3. SSRF / private-LAN block (`http://127.0.0.1`, `http://192.168.1.1`)
  4. Persistent Markdown + SQLite note creation & search
  5. Autonomous daily task creation (`web_digest`) + immediate execution + auto-saved daily note
  6. Sandboxed Linux shell inside `/home/svon/workspace` (cannot see host Windows C:\)
  7. Live Web Control Panel (`GET /?token=...`) returns HTTP 302 -> 200 HTML
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = REPO_ROOT / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from sodals_os_toolbox import SodalsOSToolbox  # noqa: E402


def _wait_ready(base_url: str, timeout_sec: int = 60) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with opener.open(f"{base_url}/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"SODALS-OS did not respond on {base_url}/health within {timeout_sec}s")


def main() -> int:
    tb = SodalsOSToolbox()
    assert tb.token, "API token missing in sodals_os/secrets/api_token.txt"
    _wait_ready(tb.base_url, timeout_sec=60)

    print("=== 1. OS STATUS & ISOLATION AUDIT ===")
    st = tb.execute_tool({"tool": "os_status"})
    assert st.ok, f"os_status failed: {st.output}"
    st_json = json.loads(st.output)
    print(json.dumps(st_json, ensure_ascii=False, indent=2))
    assert "Linux" in st_json["kernel"], f"Expected Linux kernel, got {st_json['kernel']}"
    assert st_json["isolation"]["host_c_drive_accessible"] is False
    assert st_json["isolation"]["host_filesystem_mounts"] == []
    assert st_json["isolation"]["unprivileged_user"] is True

    print("\n=== 2. LIVE WEB PAGE OPEN (https://example.com) ===")
    w = tb.execute_tool({"tool": "os_web_open", "url": "https://example.com"})
    assert w.ok, f"os_web_open failed: {w.output}"
    assert "Example Domain" in w.output, w.output
    print(w.output[:400])

    print("\n=== 3. SSRF / HOST LAN PROTECTION CHECK ===")
    w_bad1 = tb.execute_tool({"tool": "os_web_open", "url": "http://127.0.0.1:11440/health"})
    assert not w_bad1.ok, "127.0.0.1 should be blocked!"
    w_bad2 = tb.execute_tool({"tool": "os_web_open", "url": "http://192.168.1.1"})
    assert not w_bad2.ok, "192.168.1.1 should be blocked!"
    print(f"Blocked 127.0.0.1: {w_bad1.output}")
    print(f"Blocked 192.168.1.1: {w_bad2.output}")

    print("\n=== 4. NOTES VAULT SAVE & SEARCH ===")
    ns = tb.execute_tool(
        {
            "tool": "os_note_save",
            "title": "SODALS-OS Kurulum Doğrulama",
            "content": "İzole Alpine Linux mikro-VM başarıyla kuruldu ve test edildi.",
            "tags": ["sistem", "test"],
        }
    )
    assert ns.ok, ns.output
    note_id = json.loads(ns.output)["note"]["id"]
    nl = tb.execute_tool({"tool": "os_note_list", "query": "Alpine"})
    assert nl.ok and note_id in nl.output, nl.output
    print(f"Saved & verified note: {note_id}")

    print("\n=== 5. AUTONOMOUS DAILY TASK (web_digest) CREATE & RUN ===")
    ta = tb.execute_tool(
        {
            "tool": "os_task_add",
            "name": "Günlük Web Özeti Testi",
            "task_type": "web_digest",
            "schedule": "daily:08:30",
            "config": {"urls": ["https://example.com"]},
        }
    )
    assert ta.ok, ta.output
    tid = json.loads(ta.output)["task"]["id"]
    tr = tb.execute_tool({"tool": "os_task_run", "task_id": tid})
    assert tr.ok, tr.output
    tr_json = json.loads(tr.output)
    print(json.dumps(tr_json, ensure_ascii=False, indent=2))
    assert tr_json.get("note_id"), "Expected daily task to save a note!"

    print("\n=== 6. SANDBOXED LINUX SHELL & HOST C:\\ ISOLATION ===")
    sh = tb.execute_tool({"tool": "os_shell", "command": "uname -a && id && (ls /mnt/c 2>&1 || true)"})
    assert sh.ok, sh.output
    print(sh.output)
    assert "uid=1000(svon)" in sh.output or "svon" in sh.output
    assert "No such file or directory" in sh.output

    print("\n=== 7. LIVE WEB DASHBOARD CHECK ===")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"{tb.base_url}/",
        headers={"Cookie": f"sodals_os_token={tb.token}"},
    )
    with opener.open(req, timeout=10) as resp:
        html_body = resp.read().decode("utf-8", errors="ignore")
        assert resp.status == 200 and "SODALS-OS" in html_body
    print("Dashboard HTTP 200 OK!")

    # Clean up test artifacts so the VM stays tidy for the user
    tb.execute_tool({"tool": "os_task_delete", "task_id": tid})
    tb.execute_tool({"tool": "os_note_delete", "note_id": note_id})
    tb.execute_tool({"tool": "os_note_delete", "note_id": tr_json["note_id"]})
    print("\n=== ALL 7 LIVE SODALS-OS TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
