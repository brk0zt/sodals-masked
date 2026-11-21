"""
SODALS v2.1 — Quick Interactive Chat & Serving Launcher
=======================================================
Usage:
  python sodals_chat.py            # Opens direct interactive CLI chat (supports /dosya, /ajan, drag-and-drop)
  python sodals_chat.py --agent    # Starts directly in offline Claude Code-style coding & automation agent mode
  python sodals_chat.py --cwd PATH # Sets workspace folder for the offline coding agent
  python sodals_chat.py --server   # Starts Ollama/OpenAI compatible background server on http://127.0.0.1:11435
  python sodals_chat.py --port 8000
"""

import sys
import os
import subprocess

SFT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft")
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

# Auto-redirect to verified Python 3.13 once (guarded against infinite fork loops!)
try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    if os.environ.get("SODALS_PY_REDIRECTED") != "1":
        _CANDIDATE_PYTHONS = (r"C:\Python\Python313\python.exe",)
        TARGET_PYTHON = next(
            (p for p in _CANDIDATE_PYTHONS if os.path.exists(p) and os.path.normcase(sys.executable) != os.path.normcase(p)),
            "",
        )
        if TARGET_PYTHON:
            env = os.environ.copy()
            env["SODALS_PY_REDIRECTED"] = "1"
            cmd = [TARGET_PYTHON, os.path.abspath(__file__)] + sys.argv[1:]
            res = subprocess.run(cmd, env=env)
            sys.exit(res.returncode)

from sodals_serve import main

if __name__ == "__main__":
    if "--server" in sys.argv:
        sys.argv.remove("--server")
    elif not any(
        a in ("--cli", "--host", "--port", "-h", "--help") or a.startswith(("--host=", "--port="))
        for a in sys.argv[1:]
    ):
        sys.argv.append("--cli")
    main()
