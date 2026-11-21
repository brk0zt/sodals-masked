"""
Runtime Environment & Reproducibility Metadata Collector
======================================================
Records exact Python interpreter path (warning when running from Windows.old),
package versions (torch, transformers, peft, bitsandbytes, sympy, numpy), and
GPU device info into every benchmark/training JSON artifact.
"""

import datetime
import os
import platform
import sys
from typing import Any, Dict


def _safe_pkg_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "not_installed"


def collect_runtime_metadata() -> Dict[str, Any]:
    """Returns deterministic runtime and package version metadata for JSON outputs."""
    exe_path = os.path.abspath(sys.executable) if sys.executable else ""
    in_windows_old = "windows.old" in exe_path.lower()

    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sys_executable": exe_path,
        "interpreter_in_windows_old": in_windows_old,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": _safe_pkg_version("torch"),
            "transformers": _safe_pkg_version("transformers"),
            "peft": _safe_pkg_version("peft"),
            "bitsandbytes": _safe_pkg_version("bitsandbytes"),
            "sympy": _safe_pkg_version("sympy"),
            "numpy": _safe_pkg_version("numpy"),
        },
    }

    try:
        import torch
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["cuda_version"] = str(torch.version.cuda)
            meta["gpu_name"] = str(torch.cuda.get_device_name(0))
        else:
            meta["cuda_version"] = None
            meta["gpu_name"] = None
    except Exception:
        meta["cuda_available"] = False
        meta["cuda_version"] = None
        meta["gpu_name"] = None

    return meta
