#!/usr/bin/env python3
"""
SODALS Full-Project Kaggle Migration & Unified Test Suite Manager
=================================================================
Packages and syncs the ENTIRE `sodals_core` project (code, tests, LoRA adapters,
auxiliary cognitive weights, holographic .qmem crystals, Rust core source,
SODALS-OS mini-OS source, ArXiv papers, benchmarks) plus the separate heavy
model weights pack (GGUF + jlens + ssm_weights) to PRIVATE Kaggle datasets,
and deploys a unified `T4 x2` Kaggle Test Suite Runner kernel.

Security invariants:
  - Always creates PRIVATE Kaggle datasets (`public=False`).
  - Hard-blocks secrets, vaults, tokens, personal folders (`kişisel/`),
    QEMU/qcow2 VM binaries, build caches (`target/`, `vcpkg/`, `.git/`), and
    giant raw debug logs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SFT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SFT_DIR.parent
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from kaggle_cli_manager import (  # noqa: E402
    DEFAULT_ACCELERATOR,
    _dataset_exists,
    get_kaggle_api,
    resolve_kaggle_username,
)

PROJECT_DATASET_SLUG = "sodals-core-full-project"
MODELS_DATASET_SLUG = "sodals-models-gguf-and-weights"
RUNNER_KERNEL_SLUG = "sodals-full-test-suite-t4-x2"

# Hard-blocked paths / patterns (never uploaded to Kaggle)
FORBIDDEN_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "target",
    "target_cuda",
    "vcpkg",
    "blobs",
    "kişisel",
    "kisisel",
    "secrets",
    "qemu",
    "images",
    "run",
    "logs",
    "tmp",
    "kaggle_bundle",
    "kaggle_outputs",
    "run_logs_contaminated",
    "archive",
    "scratch",
    "dumps",
    "trace",
}

FORBIDDEN_FILE_NAMES = {
    "sodals.vault",
    "sodals.salt",
    "qyptos_native.vault",
    "api_token.txt",
    "access_token",
    "kaggle.json",
    ".env",
}

FORBIDDEN_EXTENSIONS = {
    ".log",
    ".whl",
    ".qcow2",
    ".iso",
    ".exe",
    ".dll",
    ".pyd",
    ".pdb",
    ".vault",
    ".salt",
}

# Production LoRA adapter folders inside sft/ to include (tokenizer.json deduplicated)
PROD_ADAPTER_DIRS = [
    "qwen_lora_adapter_balanced_r4",
    "qwen_lora_adapter_grpo_math_train_100",
    "qwen_lora_adapter_domain2_r8_pondering",
    "qwen_lora_adapter_domain2_r8_surgical_patch",
    "qwen_lora_adapter_interleaved_d1_d2_r8",
    "qwen_lora_adapter_sdpo_anchored_r8",
    "qwen_lora_gpr_soft_moa",
    "qwen_lora_gpr_soft_moa_warmstart",
    "qwen_lora_adapter_with_pondering",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_forbidden_file(path: Path) -> bool:
    low_name = path.name.lower()
    if low_name in FORBIDDEN_FILE_NAMES:
        return True
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return True
    if low_name.startswith(("build_output", "check_final", "check_errors", "min_out", "tmp_", "debug_", "stderr_", "test_stdout", "dump_logits")):
        return True
    return False


def _copy_tree_filtered(
    src: Path,
    dst: Path,
    *,
    allowed_exts: Optional[set[str]] = None,
    skip_dirs: Optional[set[str]] = None,
    max_file_mb: float = 150.0,
    skip_filenames: Optional[set[str]] = None,
) -> List[Tuple[str, int]]:
    copied: List[Tuple[str, int]] = []
    if not src.exists():
        return copied
    extra_skip = skip_dirs or set()
    extra_skip_files = {x.lower() for x in (skip_filenames or set())}
    max_bytes = int(max_file_mb * 1024 * 1024)

    for dp, dns, fns in os.walk(src):
        dns[:] = [
            d
            for d in dns
            if d not in FORBIDDEN_DIR_NAMES
            and d not in extra_skip
            and not d.startswith(("checkpoints", "kaggle_bundle", "kaggle_outputs"))
        ]
        for fn in fns:
            fp = Path(dp) / fn
            if _is_forbidden_file(fp) or fn.lower() in extra_skip_files:
                continue
            if allowed_exts is not None and fp.suffix.lower() not in allowed_exts:
                continue
            sz = fp.stat().st_size
            if sz > max_bytes:
                continue
            rel = fp.relative_to(src)
            out_p = dst / rel
            out_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, out_p)
            copied.append((str(rel).replace("\\", "/"), sz))
    return copied


def stage_full_project(stage_root: Path, username: str) -> Dict[str, Any]:
    """Stages the complete clean `sodals_core` project (code, tests, adapters, Rust, SODALS-OS, ArXiv, qmems)."""
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []

    def _record(prefix: str, items: List[Tuple[str, int]]) -> None:
        for rel, sz in items:
            manifest_entries.append({"path": f"{prefix}/{rel}".lstrip("/"), "bytes": sz})

    # 1. Complete sft/ codebase, tests, and datasets (excluding checkpoints & lora folders handled below)
    sft_items = _copy_tree_filtered(
        SFT_DIR,
        stage_root / "sft",
        skip_dirs={d.name for d in SFT_DIR.iterdir() if d.is_dir() and d.name.startswith("qwen_lora_")},
        max_file_mb=100.0,
    )
    _record("sft", sft_items)

    # 2. Production LoRA adapters & auxiliary cognitive weights (deduplicate 19MB tokenizer.json copies)
    for ad_name in PROD_ADAPTER_DIRS:
        ad_src = SFT_DIR / ad_name
        if not ad_src.is_dir():
            continue
        ad_items = _copy_tree_filtered(
            ad_src,
            stage_root / "sft" / ad_name,
            skip_filenames={"tokenizer.json", "tokenizer.json.bak"},
            max_file_mb=130.0,
        )
        _record(f"sft/{ad_name}", ad_items)

    # 3. Latest Epoch II svon_identity + Epoch III svon_cgw checkpoints (if present locally)
    ep2_ad = SFT_DIR / "kaggle_outputs" / "svon-epoch-ii-and-iii-t4-x2" / "svon_out" / "qwen_lora_adapter_svon_alphaedit"
    if ep2_ad.is_dir():
        _record(
            "sft/qwen_lora_adapter_svon_alphaedit",
            _copy_tree_filtered(ep2_ad, stage_root / "sft" / "qwen_lora_adapter_svon_alphaedit", max_file_mb=100.0),
        )
    ep3_dir = SFT_DIR / "kaggle_outputs" / "svon-epoch-ii-and-iii-t4-x2" / "svon_out" / "svon_cgw_checkpoints"
    if ep3_dir.is_dir():
        _record(
            "sft/svon_cgw_checkpoints",
            _copy_tree_filtered(ep3_dir, stage_root / "sft" / "svon_cgw_checkpoints", max_file_mb=100.0),
        )

    # 4. Complete Rust core source (src/, Cargo.toml, Cargo.lock, patched-crates/) — source & configs only
    rust_exts = {".rs", ".toml", ".lock", ".cu", ".c", ".h", ".md", ".json", ".py", ".sh", ".ebnf"}
    _record("src", _copy_tree_filtered(REPO_ROOT / "src", stage_root / "src", allowed_exts=rust_exts, max_file_mb=10.0))
    if (REPO_ROOT / "patched-crates").is_dir():
        _record(
            "patched-crates",
            _copy_tree_filtered(REPO_ROOT / "patched-crates", stage_root / "patched-crates", allowed_exts=rust_exts, max_file_mb=10.0),
        )

    # 5. SODALS-OS Mini-OS source (strictly excluding secrets/, qemu/, images/, logs/, run/)
    os_src = REPO_ROOT / "sodals_os"
    if os_src.is_dir():
        _record("sodals_os", _copy_tree_filtered(os_src, stage_root / "sodals_os", max_file_mb=10.0))

    # 6. ArXiv papers, scripts, SWE-bench Lite data, and root launchers/configs
    for folder_name, max_mb in [("arxiv", 20.0), ("scripts", 20.0), ("patches", 10.0)]:
        fsrc = REPO_ROOT / folder_name
        if fsrc.is_dir():
            _record(folder_name, _copy_tree_filtered(fsrc, stage_root / folder_name, max_file_mb=max_mb))

    if (REPO_ROOT / "data" / "SWE-bench_Lite").is_dir():
        _record(
            "data/SWE-bench_Lite",
            _copy_tree_filtered(REPO_ROOT / "data" / "SWE-bench_Lite", stage_root / "data" / "SWE-bench_Lite", max_file_mb=20.0),
        )
    if (REPO_ROOT / "data" / "swe_lite.json").is_file():
        dst_swe = stage_root / "data" / "swe_lite.json"
        dst_swe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "data" / "swe_lite.json", dst_swe)
        manifest_entries.append({"path": "data/swe_lite.json", "bytes": dst_swe.stat().st_size})

    # 7. Holographic .qmem knowledge crystals (root + crates/sodals_neuro)
    qmem_dst = stage_root / "qmem"
    qmem_dst.mkdir(parents=True, exist_ok=True)
    qmem_seen = set()
    for q_search in [REPO_ROOT, REPO_ROOT / "src" / "crates" / "sodals_neuro"]:
        if not q_search.is_dir():
            continue
        for qp in sorted(q_search.glob("*.qmem")):
            if qp.name in qmem_seen:
                continue
            qmem_seen.add(qp.name)
            shutil.copy2(qp, qmem_dst / qp.name)
            manifest_entries.append({"path": f"qmem/{qp.name}", "bytes": qp.stat().st_size})

    # 8. Root project files (launchers, Cargo, requirements, verified math500/hle baselines)
    root_files = [
        "Cargo.toml",
        "Cargo.lock",
        "config.toml",
        "requirements_sft.txt",
        "sodals_chat.py",
        "sodals_agent.bat",
        "sodals_os.bat",
        "start-claude-offline.bat",
        "math500_full500_sdpo_a_results.json",
        "math500_full500_lora_4096_results.json",
        "lmeval_base_math500_results.json",
        "hle_sodals_Qwen3.5-9B.json",
        "adaptive_knowledge.ebnf",
    ]
    for rf in root_files:
        src_rf = REPO_ROOT / rf
        if src_rf.is_file():
            shutil.copy2(src_rf, stage_root / rf)
            manifest_entries.append({"path": rf, "bytes": src_rf.stat().st_size})

    total_bytes = sum(x["bytes"] for x in manifest_entries)
    summary = {
        "dataset_id": f"{username}/{PROJECT_DATASET_SLUG}",
        "staged_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": len(manifest_entries),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "included_prod_adapters": [d for d in PROD_ADAPTER_DIRS if (stage_root / "sft" / d).is_dir()],
        "included_qmems": sorted(qmem_seen),
    }
    (stage_root / "PROJECT_MANIFEST.json").write_text(
        json.dumps({"summary": summary, "files": manifest_entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    meta = {
        "title": "SODALS Core Full Project",
        "id": f"{username}/{PROJECT_DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    (stage_root / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def stage_models_pack(stage_root: Path, username: str) -> Dict[str, Any]:
    """Stages heavy model binaries (GGUF + jlens + ssm_weights + tokenizer) as a separate static private dataset."""
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []

    def _link_or_copy(src: Path, rel_dst: str) -> None:
        if not src.exists():
            return
        dst = stage_root / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)  # Zero-disk-duplication hardlink on NTFS!
        except OSError:
            shutil.copy2(src, dst)
        entries.append({"path": rel_dst, "bytes": src.stat().st_size})

    _link_or_copy(REPO_ROOT / "models" / "Qwen3.5-9B_Q4_K_M.gguf", "models/Qwen3.5-9B_Q4_K_M.gguf")
    _link_or_copy(REPO_ROOT / "models" / "tokenizer.json", "models/tokenizer.json")
    _link_or_copy(REPO_ROOT / "models" / "tokenizer_config.json", "models/tokenizer_config.json")
    _link_or_copy(REPO_ROOT / "jlens_weights.safetensors", "weights/jlens_weights.safetensors")

    ssm_dir = REPO_ROOT / "ssm_weights"
    if ssm_dir.is_dir():
        for p in sorted(ssm_dir.rglob("*")):
            if p.is_file() and not _is_forbidden_file(p):
                rel = "weights/ssm_weights/" + str(p.relative_to(ssm_dir)).replace("\\", "/")
                _link_or_copy(p, rel)

    total_bytes = sum(x["bytes"] for x in entries)
    summary = {
        "dataset_id": f"{username}/{MODELS_DATASET_SLUG}",
        "staged_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": len(entries),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "files": entries,
    }
    (stage_root / "MODELS_MANIFEST.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "title": "SODALS Models GGUF and Weights",
        "id": f"{username}/{MODELS_DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    (stage_root / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def build_unified_runner_notebook(username: str, kernel_dir: Path, include_models: bool = True) -> Path:
    """Creates the unified Kaggle T4 x2 test runner notebook mounting the full project & models datasets."""
    kernel_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "id": "sodals-header",
            "metadata": {},
            "source": [
                "# SODALS Core — Unified Kaggle T4 x2 Test & Benchmark Suite\n",
                "Mounts the complete `sodals-core-full-project` (code, prod_suite tests, D1/D2/Soft-MoA/Svon adapters, Rust core, SODALS-OS, ArXiv) and `sodals-models-gguf-and-weights` datasets.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "sodals-env-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 0: Kaggle Environment Fix (remove incompatible preinstalled torchao 0.10.0 & align transformers/peft)\n",
                "import subprocess, sys\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)\n",
                "subprocess.run([\n",
                "    sys.executable, '-m', 'pip', 'install', '-q', '-U',\n",
                "    'transformers>=5.16.0', 'peft>=0.15.0', 'safetensors>=0.4.5',\n",
                "    'accelerate>=1.0.0', 'bitsandbytes>=0.45.0', 'jinja2>=3.1.6',\n",
                "], check=False)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "sodals-stage-workspace",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys, json, shutil, subprocess, zipfile, torch\n",
                "from pathlib import Path\n",
                "\n",
                "print('PyTorch:', torch.__version__, '| GPUs:', torch.cuda.device_count())\n",
                "for i in range(torch.cuda.device_count()):\n",
                "    p = torch.cuda.get_device_properties(i)\n",
                "    print(f'  GPU {i}: {p.name} ({p.total_memory/1024**3:.2f} GiB)')\n",
                "\n",
                "def find_dataset(slug):\n",
                f"    for base in [Path('/kaggle/input') / slug, Path('/kaggle/input/datasets/{username}') / slug]:\n",
                "        if base.exists():\n",
                "            return base\n",
                "    for p in Path('/kaggle/input').rglob('*MANIFEST.json'):\n",
                "        if slug in str(p):\n",
                "            return p.parent\n",
                "    return None\n",
                "\n",
                f"PROJ_SRC = find_dataset('{PROJECT_DATASET_SLUG}')\n",
                f"MODELS_SRC = find_dataset('{MODELS_DATASET_SLUG}')\n",
                "print('PROJ_SRC  :', PROJ_SRC)\n",
                "print('MODELS_SRC:', MODELS_SRC)\n",
                "assert PROJ_SRC is not None, 'Full project dataset not mounted!'\n",
                "\n",
                "WORK = Path('/kaggle/working/sodals_core')\n",
                "if WORK.exists():\n",
                "    shutil.rmtree(WORK)\n",
                "WORK.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Fast staging: symlink large read-only subtrees (adapters, qmem, data) and copy lightweight code trees\n",
                "SYMLINK_TOP_DIRS = {'qmem', 'data', 'arxiv', 'src', 'patched-crates'}\n",
                "for item in PROJ_SRC.iterdir():\n",
                "    dst = WORK / item.name\n",
                "    if item.suffix == '.zip':\n",
                "        with zipfile.ZipFile(item) as z:\n",
                "            z.extractall(WORK / item.stem)\n",
                "    elif item.is_dir() and item.name in SYMLINK_TOP_DIRS:\n",
                "        os.symlink(str(item), str(dst))\n",
                "    elif item.is_dir() and item.name == 'sft':\n",
                "        dst.mkdir(parents=True, exist_ok=True)\n",
                "        for sitem in item.iterdir():\n",
                "            sdst = dst / sitem.name\n",
                "            if sitem.is_dir() and (sitem.name.startswith('qwen_lora_') or sitem.name == 'svon_cgw_checkpoints'):\n",
                "                os.symlink(str(sitem), str(sdst))\n",
                "            elif sitem.is_dir():\n",
                "                shutil.copytree(sitem, sdst)\n",
                "            else:\n",
                "                shutil.copy2(sitem, sdst)\n",
                "    elif item.is_dir():\n",
                "        shutil.copytree(item, dst)\n",
                "    else:\n",
                "        shutil.copy2(item, dst)\n",
                "\n",
                "if MODELS_SRC is not None:\n",
                "    if (MODELS_SRC / 'models').exists():\n",
                "        os.symlink(str(MODELS_SRC / 'models'), str(WORK / 'models'))\n",
                "    if (MODELS_SRC / 'weights').exists():\n",
                "        os.symlink(str(MODELS_SRC / 'weights'), str(WORK / 'weights'))\n",
                "    print('Mounted GGUF & model weights via symlink ->', WORK / 'models')\n",
                "\n",
                "print(json.dumps(json.loads((WORK / 'PROJECT_MANIFEST.json').read_text())['summary'], indent=2))\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "sodals-stage1-unit-tests",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 1: Full CPU Unit Test Suite (prod_suite + SODALS-OS + Clarification Freeze + Svon-CGW + AlphaEdit)\n",
                "import subprocess, sys\n",
                "env = os.environ.copy()\n",
                "env['PYTHONPATH'] = f'{WORK}/sft:{WORK}/sodals_os:{WORK}/sodals_os/guest:' + env.get('PYTHONPATH', '')\n",
                "res = subprocess.run(\n",
                "    [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(WORK / 'sft' / 'tests' / 'prod_suite'), '-p', 'test_*.py', '-v'],\n",
                "    cwd=str(WORK),\n",
                "    env=env,\n",
                "    check=False,\n",
                ")\n",
                "print('Stage 1 prod_suite exit code:', res.returncode)\n",
                "assert res.returncode == 0, f'Stage 1 unit tests failed with exit code {res.returncode}'\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "sodals-stage2-integrity",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 2: GGUF & Adapter Integrity Verification\n",
                "from pathlib import Path\n",
                "gguf_path = WORK / 'models' / 'Qwen3.5-9B_Q4_K_M.gguf'\n",
                "if gguf_path.exists():\n",
                "    with open(gguf_path, 'rb') as f:\n",
                "        magic = f.read(4)\n",
                "    print(f'GGUF Verified: {gguf_path.name} ({gguf_path.stat().st_size/1024**3:.2f} GiB) | magic={magic!r}')\n",
                "    assert magic == b'GGUF'\n",
                "else:\n",
                "    print('GGUF dataset not mounted in this run.')\n",
                "\n",
                "adapters = sorted(p.name for p in (WORK / 'sft').glob('qwen_lora_*') if p.is_dir())\n",
                "print(f'Available LoRA Adapters ({len(adapters)}):', adapters)\n",
                "qmems = sorted(p.name for p in (WORK / 'qmem').glob('*.qmem')) if (WORK / 'qmem').exists() else []\n",
                "print(f'Available Holographic .qmem Crystals ({len(qmems)}):', qmems)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "sodals-stage3-gpu-benchmarks",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 3: Configurable GPU Benchmark Suite (toggle flags to run heavy evals on T4 x2)\n",
                "RUN_FULL_STACK_COMPAT = True\n",
                "RUN_INDEPENDENT_400 = False  # Set True to run N=100 x G=4 SymbolicStepChecker on T4 x2\n",
                "\n",
                "if RUN_FULL_STACK_COMPAT:\n",
                "    out_json = Path('/kaggle/working/full_stack_compatibility_report.json')\n",
                "    cmd = [\n",
                "        sys.executable, '-B', str(WORK / 'sft' / 'eval_local_full_stack_compatibility.py'),\n",
                "        '--model_path', 'Qwen/Qwen3.5-9B',\n",
                "        '--fp16', '--online',\n",
                "        '--d1_adapter_path', str(WORK / 'sft' / 'qwen_lora_adapter_balanced_r4'),\n",
                "        '--d2_adapter_path', str(WORK / 'sft' / 'qwen_lora_adapter_domain2_r8_pondering'),\n",
                "        '--output_json', str(out_json),\n",
                "    ]\n",
                "    svon_id = WORK / 'sft' / 'qwen_lora_adapter_svon_alphaedit'\n",
                "    if svon_id.is_dir():\n",
                "        cmd += ['--adapter_svon', str(svon_id)]\n",
                "    cgw_w = WORK / 'sft' / 'svon_cgw_checkpoints' / 'svon_cgw_hybrid.safetensors'\n",
                "    if cgw_w.is_file():\n",
                "        cmd += ['--cgw_weights', str(cgw_w)]\n",
                "    print('Running:', ' '.join(cmd))\n",
                "    r = subprocess.run(cmd, cwd=str(WORK / 'sft'), env=env, check=False)\n",
                "    print('Full-Stack Compatibility exit code:', r.returncode)\n",
                "    if out_json.exists():\n",
                "        print(out_json.read_text(encoding='utf-8'))\n",
                "\n",
                "if RUN_INDEPENDENT_400:\n",
                "    cmd = [\n",
                "        sys.executable, '-B', str(WORK / 'sft' / 'eval_symbolic_checker_independent_400.py'),\n",
                "        '--model_path', 'Qwen/Qwen3.5-9B',\n",
                "        '--fp16', '--online',\n",
                "        '--batch_g4', '--max_new_tokens', '1024',\n",
                "    ]\n",
                "    subprocess.run(cmd, cwd=str(WORK / 'sft'), env=env, check=False)\n",
            ],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = kernel_dir / f"{RUNNER_KERNEL_SLUG}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")

    ds_sources = [f"{username}/{PROJECT_DATASET_SLUG}"]
    if include_models:
        ds_sources.append(f"{username}/{MODELS_DATASET_SLUG}")

    kmeta = {
        "id": f"{username}/{RUNNER_KERNEL_SLUG}",
        "title": "SODALS Full Test Suite T4 x2",
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ds_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    return nb_path


def _push_private_dataset(api: Any, folder: Path, dataset_ref: str, notes: str, wait_sec: int = 300) -> Dict[str, Any]:
    existing = _dataset_exists(api, dataset_ref)
    if existing is None:
        print(f"[KAGGLE] Creating PRIVATE dataset: {dataset_ref}", flush=True)
        resp = api.dataset_create_new(
            folder=str(folder),
            public=False,
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip",
        )
        action = "created"
    else:
        print(f"[KAGGLE] Updating PRIVATE dataset: {dataset_ref} (current={existing})", flush=True)
        resp = api.dataset_create_version(
            folder=str(folder),
            version_notes=notes,
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip",
        )
        action = "versioned"

    if getattr(resp, "error", None):
        raise RuntimeError(f"Kaggle dataset upload error ({dataset_ref}): {resp.error}")

    if action == "versioned":
        print(f"[KAGGLE] Waiting 15s for Kaggle version indexing on `{dataset_ref}`...", flush=True)
        time.sleep(15)

    deadline = time.time() + max(30, wait_sec)
    status = "unknown"
    while time.time() < deadline:
        status = (_dataset_exists(api, dataset_ref) or "pending").lower()
        print(f"[KAGGLE] Dataset `{dataset_ref}` status = {status}", flush=True)
        if status in ("ready", "complete"):
            break
        if status in ("error", "failed"):
            raise RuntimeError(f"Dataset `{dataset_ref}` failed: {status}")
        time.sleep(10)
    return {"dataset_ref": dataset_ref, "action": action, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload full SODALS project & GGUF/weights to private Kaggle datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("dry-run", help="Stage project & models locally to inspect exact file counts and MB sizes")
    p_proj = sub.add_parser("push-project", help="Upload full sodals_core project dataset (private)")
    p_proj.add_argument("--notes", type=str, default=None)

    p_mod = sub.add_parser("push-models", help="Upload heavy GGUF + jlens + ssm_weights dataset (private)")
    p_mod.add_argument("--notes", type=str, default=None)

    p_run = sub.add_parser("push-runner", help="Push unified Kaggle T4 x2 test suite notebook")
    p_run.add_argument("--no-run", action="store_true", help="Upload notebook without auto-executing immediately")
    p_run.add_argument("--without-models", action="store_true", help="Mount only project dataset (before models upload finishes)")

    args = parser.parse_args()
    stage_base = Path(tempfile.gettempdir()) / "sodals_kaggle_full_sync"

    if args.cmd == "dry-run":
        try:
            user = resolve_kaggle_username(None)
        except Exception:
            user = "brkozt"
        p_sum = stage_full_project(stage_base / "project", user)
        m_sum = stage_models_pack(stage_base / "models", user)
        print(json.dumps({"project": p_sum, "models": m_sum}, indent=2, ensure_ascii=False))
        return 0

    api, creds = get_kaggle_api()
    username = resolve_kaggle_username(None)
    if username == "YOUR_KAGGLE_USERNAME":
        username = str(creds.get("username") or username)
    ts_note = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    if args.cmd == "push-project":
        p_dir = stage_base / "project"
        p_sum = stage_full_project(p_dir, username)
        print(f"[STAGE] Full project staged: {p_sum['file_count']} files, {p_sum['total_mb']} MB", flush=True)
        res = _push_private_dataset(
            api, p_dir, f"{username}/{PROJECT_DATASET_SLUG}", args.notes or f"Full SODALS project sync ({ts_note})"
        )
        print(json.dumps({"staged": p_sum, "upload": res}, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "push-models":
        m_dir = stage_base / "models"
        m_sum = stage_models_pack(m_dir, username)
        print(f"[STAGE] Models pack staged: {m_sum['file_count']} files, {m_sum['total_mb']} MB", flush=True)
        res = _push_private_dataset(
            api, m_dir, f"{username}/{MODELS_DATASET_SLUG}", args.notes or f"SODALS GGUF + JLens + SSM weights ({ts_note})", wait_sec=600
        )
        print(json.dumps({"staged": m_sum, "upload": res}, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "push-runner":
        k_dir = stage_base / "runner_kernel"
        build_unified_runner_notebook(username, k_dir, include_models=(not args.without_models))
        print(f"[KAGGLE] Pushing unified runner kernel: {username}/{RUNNER_KERNEL_SLUG}", flush=True)
        kn_resp = api.kernels_push(folder=str(k_dir), acc=DEFAULT_ACCELERATOR, no_run=args.no_run)
        if getattr(kn_resp, "error", None):
            raise RuntimeError(f"Kernel push error: {kn_resp.error}")
        print(
            json.dumps(
                {
                    "kernel_ref": f"{username}/{RUNNER_KERNEL_SLUG}",
                    "url": getattr(kn_resp, "url", None),
                    "version": getattr(kn_resp, "versionNumber", None),
                    "no_run": args.no_run,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
