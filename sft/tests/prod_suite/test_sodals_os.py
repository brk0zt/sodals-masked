import os
import struct
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SFT_DIR = REPO_ROOT / "sft"
OS_DIR = REPO_ROOT / "sodals_os"
GUEST_DIR = OS_DIR / "guest"

for p in (str(SFT_DIR), str(OS_DIR), str(GUEST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from iso_builder import SECTOR_SIZE, build_nocloud_iso  # noqa: E402


class TestSodalsOSIsoBuilder(unittest.TestCase):
    def test_nocloud_iso_pvd_svd_and_joliet_entries(self):
        with tempfile.TemporaryDirectory() as td:
            out_iso = Path(td) / "cidata.iso"
            meta = b"instance-id: sodals-os-test\nlocal-hostname: sodals-os\n"
            user = b"#cloud-config\nhostname: sodals-os\n"
            build_nocloud_iso({"meta-data": meta, "user-data": user}, out_iso, volume_id="cidata")

            raw = out_iso.read_bytes()
            self.assertEqual(len(raw) % SECTOR_SIZE, 0)

            # Check PVD at sector 16
            pvd = raw[16 * SECTOR_SIZE : 17 * SECTOR_SIZE]
            self.assertEqual(pvd[0], 1)
            self.assertEqual(pvd[1:6], b"CD001")
            self.assertEqual(pvd[40:46], b"cidata")

            # Check Joliet SVD at sector 17
            svd = raw[17 * SECTOR_SIZE : 18 * SECTOR_SIZE]
            self.assertEqual(svd[0], 2)
            self.assertEqual(svd[1:6], b"CD001")
            self.assertEqual(svd[88:91], b"%/E")

            # Parse Joliet root directory at sector 24 and verify exact lowercase filenames + payloads
            root_sec = raw[24 * SECTOR_SIZE : 25 * SECTOR_SIZE]
            pos = 0
            found = {}
            while pos < SECTOR_SIZE and root_sec[pos] != 0:
                rec_len = root_sec[pos]
                rec = root_sec[pos : pos + rec_len]
                lba = struct.unpack("<I", rec[2:6])[0]
                size = struct.unpack("<I", rec[10:14])[0]
                name_len = rec[32]
                name_bytes = rec[33 : 33 + name_len]
                if name_bytes not in (b"\x00", b"\x01"):
                    decoded_name = name_bytes.decode("utf-16-be").removesuffix(";1")
                    found[decoded_name] = raw[lba * SECTOR_SIZE : lba * SECTOR_SIZE + size]
                pos += rec_len

            self.assertEqual(found.get("meta-data"), meta)
            self.assertEqual(found.get("user-data"), user)


class TestSodalsOSDaemonAndToolbox(unittest.TestCase):
    def test_daemon_notes_tasks_ssrf_and_toolbox_bridge(self):
        with tempfile.TemporaryDirectory() as td:
            os.environ["SODALS_OS_HOME"] = td
            os.environ["SODALS_OS_TOKEN"] = "unit_test_secret_token_123"
            # Reload os_daemon cleanly pointing to temp dir
            import importlib
            import os_daemon

            importlib.reload(os_daemon)
            from sodals_os_toolbox import SodalsOSToolbox

            # Verify SSRF private LAN blocking
            ok_local, err_local = os_daemon._validate_public_url("http://127.0.0.1:8000")
            self.assertFalse(ok_local)
            self.assertIn("engellendi", err_local)

            ok_lan, err_lan = os_daemon._validate_public_url("http://192.168.1.1/admin")
            self.assertFalse(ok_lan)
            self.assertIn("engellendi", err_lan)

            ok_file, _ = os_daemon._validate_public_url("file:///C:/Windows/win.ini")
            self.assertFalse(ok_file)

            # Start in-memory HTTP daemon on ephemeral port
            srv = ThreadingHTTPServer(("127.0.0.1", 0), os_daemon.SodalsOSHandler)
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                tb = SodalsOSToolbox(base_url=f"http://127.0.0.1:{port}", token="unit_test_secret_token_123")

                # 1. Status check
                st = tb.execute_tool({"tool": "os_status"})
                self.assertTrue(st.ok, st.output)
                self.assertIn("SODALS-OS", st.output)

                # 2. Save & search & read note
                ns = tb.execute_tool(
                    {
                        "tool": "os_note_save",
                        "title": "Gunluk Plan",
                        "content": "- Sabah raporu oku\n- Notlari duzenle",
                        "tags": ["daily", "plan"],
                    }
                )
                self.assertTrue(ns.ok, ns.output)
                nl = tb.execute_tool({"tool": "os_note_list", "query": "Sabah"})
                self.assertTrue(nl.ok, nl.output)
                self.assertIn("Gunluk Plan", nl.output)

                # 3. Create & immediately run a daily note_briefing task
                ta = tb.execute_tool(
                    {
                        "tool": "os_task_add",
                        "name": "Sabah Checklist",
                        "task_type": "note_briefing",
                        "schedule": "daily:08:30",
                        "config": {"template": "- [ ] Haberleri kontrol et"},
                    }
                )
                self.assertTrue(ta.ok, ta.output)
                import json

                tid = json.loads(ta.output)["task"]["id"]
                tr = tb.execute_tool({"tool": "os_task_run", "task_id": tid})
                self.assertTrue(tr.ok, tr.output)
                self.assertIn("Günlük plan notu oluşturuldu", tr.output)

                # 4. Workspace file isolation check (cannot escape workspace)
                fw = tb.execute_tool({"tool": "os_file_write", "path": "hello.txt", "content": "Merhaba Mini-OS"})
                self.assertTrue(fw.ok, fw.output)
                fr = tb.execute_tool({"tool": "os_file_read", "path": "hello.txt"})
                self.assertTrue(fr.ok, fr.output)
                self.assertIn("Merhaba Mini-OS", fr.output)

                esc = tb.execute_tool({"tool": "os_file_read", "path": "../../Windows/win.ini"})
                self.assertFalse(esc.ok)

                # 5. Unauthorized token rejected
                bad_tb = SodalsOSToolbox(base_url=f"http://127.0.0.1:{port}", token="wrong_token")
                bad_st = bad_tb.execute_tool({"tool": "os_status"})
                self.assertFalse(bad_st.ok)

                # 6. Wave 3 security regressions: note_id path traversal, invalid schedule, XSS esc(), bootstrap auth
                trav = tb.execute_tool({
                    "tool": "os_note_save",
                    "title": "Evil",
                    "content": "x",
                    "note_id": "../../pwned",
                })
                self.assertFalse(trav.ok)

                bad_sched = tb.execute_tool({
                    "tool": "os_task_add",
                    "name": "Bad",
                    "task_type": "note_briefing",
                    "schedule": "daily:99:99",
                    "config": {},
                })
                self.assertFalse(bad_sched.ok)

                bad_type = tb.execute_tool({
                    "tool": "os_task_add",
                    "name": "BadType",
                    "task_type": "arbitrary_exec",
                    "schedule": "manual",
                    "config": {},
                })
                self.assertFalse(bad_type.ok)

                # 7. Single-quote XSS protection & no inline interpolated onclick handlers in DASHBOARD_HTML
                self.assertIn("&#39;", os_daemon.DASHBOARD_HTML)
                self.assertNotIn("viewNote('${esc(", os_daemon.DASHBOARD_HTML)
                self.assertNotIn("delNote('${esc(", os_daemon.DASHBOARD_HTML)

                # 8. One-time bootstrap login code works ONCE and rejects replay
                import urllib.request
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                req_bs = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/auth/bootstrap",
                    data=b"{}",
                    headers={
                        "Authorization": "Bearer unit_test_secret_token_123",
                        "Content-Type": "application/json",
                        "X-SODALS-CSRF": "1",
                    },
                    method="POST",
                )
                with opener.open(req_bs, timeout=5) as r_bs:
                    bs_data = json.loads(r_bs.read().decode("utf-8"))
                self.assertTrue(bs_data.get("ok"))
                code = bs_data["code"]
                self.assertTrue(os_daemon._consume_bootstrap_code(code))
                self.assertFalse(os_daemon._consume_bootstrap_code(code))  # replay blocked!

                # 9. Redirect SSRF handler blocks redirect to localhost/LAN
                redir = os_daemon._SafeRedirectHandler()
                dummy_req = urllib.request.Request("https://example.com/")
                with self.assertRaises(urllib.error.URLError):
                    redir.redirect_request(dummy_req, None, 302, "Found", {}, "http://127.0.0.1:11440/api/status")

                # 10. Multi-sector ISO-9660 builder supports > 40 files cleanly
                with tempfile.TemporaryDirectory() as iso_td:
                    many = {f"file_{i:02d}.txt": f"payload-{i}".encode("utf-8") for i in range(45)}
                    out_iso2 = Path(iso_td) / "many.iso"
                    build_nocloud_iso(many, out_iso2, volume_id="cidata")
                    self.assertGreater(out_iso2.stat().st_size, 25 * SECTOR_SIZE)
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
