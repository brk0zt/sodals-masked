#!/usr/bin/env python3
"""
SODALS v2.1 — 12-Hour Kaggle T4 x2 Batch Chain Launcher
=======================================================
Packages and submits a single unattended 12-hour Kaggle T4 x2 batch kernel
(`brkozt/sodals-ind400-dev500-chain-t4-x2`) that executes in sequence:
  1. Phase 2B Independent-400 (`N=100 x G=4` SymbolicStepChecker 2x2 audit)
  2. Phase 3 Dev-500 Baseline (`D1 balanced_r4`, FP16)
  3. Phase 3 Dev-500 Full Stack (`D1 + D2 + Svon Identity + Svon-CGW`, paired McNemar vs Baseline)

Mounts the existing private `brkozt/sodals-core-full-project` dataset and overlays
the latest Linux/FP16-enabled `eval_math_train_disjoint_dev500.py` directly into
the working directory so no 844 MB dataset re-upload is required.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from kaggle_cli_manager import DEFAULT_ACCELERATOR, get_kaggle_api, resolve_kaggle_username  # noqa: E402

PROJECT_DATASET_SLUG = "sodals-core-full-project"
CHAIN_KERNEL_SLUG = "sodals-independent-400-and-dev-500-chain-t4-x2"


def build_chain_notebook(username: str, kernel_dir: Path) -> Path:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    overlay_files = [
        "eval_math_train_disjoint_dev500.py",
        "eval_symbolic_checker_independent_400.py",
        "eval_pristine30_q171_q200_arkhe_pipeline.py",
        "cloud_runtime_utils.py",
    ]
    overlay_b64 = {
        fname: base64.b64encode((SFT_DIR / fname).read_text(encoding="utf-8").encode("utf-8")).decode("ascii")
        for fname in overlay_files
    }

    cells = [
        {
            "cell_type": "markdown",
            "id": "chain-header",
            "metadata": {},
            "source": [
                "# SODALS Core — 12-Hour Unattended Kaggle T4 x2 Evaluation Chain\n",
                "Runs **Independent-400** (`N=100 x G=4`) -> **Dev-500 Baseline** -> **Dev-500 Svon + CGW** in lossless FP16 across Dual Tesla T4 (32 GB VRAM) with per-prompt checkpointing and an 11.5h soft deadline guard.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "chain-env-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 0: Align Kaggle Python environment (remove torchao 0.10.0 conflict, upgrade transformers/peft)\n",
                "import subprocess, sys, time\n",
                "SESSION_T0 = time.time()\n",
                "SOFT_DEADLINE_SEC = 11 * 3600 + 25 * 60  # 11h25m soft guard inside Kaggle's 12h limit\n",
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
            "id": "chain-stage-workspace",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64, json, os, shutil, sys, zipfile, torch\n",
                "from pathlib import Path\n",
                "\n",
                "print('PyTorch:', torch.__version__, '| GPUs:', torch.cuda.device_count())\n",
                "assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'GPU not active!'\n",
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
                "print('PROJ_SRC:', PROJ_SRC)\n",
                "assert PROJ_SRC is not None, 'sodals-core-full-project dataset not mounted!'\n",
                "\n",
                "WORK = Path('/kaggle/working/sodals_core')\n",
                "if WORK.exists():\n",
                "    shutil.rmtree(WORK)\n",
                "WORK.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Stage sft/ cleanly into writable workspace (symlink heavy adapter folders if unpacked)\n",
                "sft_zip = PROJ_SRC / 'sft.zip'\n",
                "if sft_zip.is_file():\n",
                "    with zipfile.ZipFile(sft_zip) as z:\n",
                "        z.extractall(WORK / 'sft')\n",
                "elif (PROJ_SRC / 'sft').is_dir():\n",
                "    dst_sft = WORK / 'sft'\n",
                "    dst_sft.mkdir(parents=True, exist_ok=True)\n",
                "    for sitem in (PROJ_SRC / 'sft').iterdir():\n",
                "        sdst = dst_sft / sitem.name\n",
                "        if sitem.is_dir() and (sitem.name.startswith('qwen_lora_') or sitem.name == 'svon_cgw_checkpoints'):\n",
                "            os.symlink(str(sitem), str(sdst))\n",
                "        elif sitem.is_dir():\n",
                "            shutil.copytree(sitem, sdst)\n",
                "        else:\n",
                "            shutil.copy2(sitem, sdst)\n",
                "\n",
                "# Overlay latest Linux/FP16/online-patched evaluation scripts into staged sft/\n",
                f"_OVERLAY_B64 = {json.dumps(overlay_b64)}\n",
                "for _fname, _b64 in _OVERLAY_B64.items():\n",
                "    (WORK / 'sft' / _fname).write_bytes(base64.b64decode(_b64))\n",
                "\n",
                "RES_DIR = Path('/kaggle/working/chain_results')\n",
                "RES_DIR.mkdir(parents=True, exist_ok=True)\n",
                "env = os.environ.copy()\n",
                "env['HF_HUB_OFFLINE'] = '0'\n",
                "env['TRANSFORMERS_OFFLINE'] = '0'\n",
                "env['PYTHONPATH'] = f'{WORK}/sft:' + env.get('PYTHONPATH', '')\n",
                "print('Staged sft/ successfully. Adapters:', sorted(p.name for p in (WORK / 'sft').glob('qwen_lora_*') if p.is_dir()))\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "chain-run-all",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Independent-400 -> Dev-500 Baseline -> Dev-500 Svon+CGW with soft 11h25m time-budget guard\n",
                "import gc, json, subprocess, sys, time, torch, zipfile\n",
                "from pathlib import Path\n",
                "\n",
                "SFT = WORK / 'sft'\n",
                "ind400_out = RES_DIR / 'symbolic_checker_independent_400_results.json'\n",
                "dev500_base_out = RES_DIR / 'dev500_baseline_results.json'\n",
                "dev500_svon_out = RES_DIR / 'dev500_svon_cgw_results.json'\n",
                "\n",
                "stages = [\n",
                "    (\n",
                "        'stage1_independent_400',\n",
                "        [\n",
                "            sys.executable, '-B', str(SFT / 'eval_symbolic_checker_independent_400.py'),\n",
                "            '--model_path', 'Qwen/Qwen3.5-9B',\n",
                "            '--fp16', '--online',\n",
                "            '--adapter_path', str(SFT / 'qwen_lora_adapter_balanced_r4'),\n",
                "            '--batch_g4', '--max_new_tokens', '1024',\n",
                "            '--output_json', str(ind400_out),\n",
                "        ],\n",
                "    ),\n",
                "    (\n",
                "        'stage2_dev500_baseline',\n",
                "        [\n",
                "            sys.executable, '-B', str(SFT / 'eval_math_train_disjoint_dev500.py'),\n",
                "            '--model_path', 'Qwen/Qwen3.5-9B',\n",
                "            '--fp16', '--online',\n",
                "            '--dev_json', str(SFT / 'math_train_disjoint_dev500.json'),\n",
                "            '--adapter_path', str(SFT / 'qwen_lora_adapter_balanced_r4'),\n",
                "            '--scale', '0.25', '--max_new_tokens', '1024',\n",
                "            '--output_json', str(dev500_base_out),\n",
                "        ],\n",
                "    ),\n",
                "    (\n",
                "        'stage3_dev500_svon_cgw',\n",
                "        [\n",
                "            sys.executable, '-B', str(SFT / 'eval_math_train_disjoint_dev500.py'),\n",
                "            '--model_path', 'Qwen/Qwen3.5-9B',\n",
                "            '--fp16', '--online',\n",
                "            '--dev_json', str(SFT / 'math_train_disjoint_dev500.json'),\n",
                "            '--adapter_path', str(SFT / 'qwen_lora_adapter_balanced_r4'),\n",
                "            '--d2_adapter_path', str(SFT / 'qwen_lora_adapter_domain2_r8_pondering'),\n",
                "            '--svon_identity_lora', str(SFT / 'qwen_lora_adapter_svon_alphaedit'),\n",
                "            '--enable_svon_cgw',\n",
                "            '--svon_cgw_weights', str(SFT / 'svon_cgw_checkpoints' / 'svon_cgw_hybrid.safetensors'),\n",
                "            '--baseline_json', str(dev500_base_out),\n",
                "            '--scale', '0.25', '--max_new_tokens', '1024',\n",
                "            '--output_json', str(dev500_svon_out),\n",
                "        ],\n",
                "    ),\n",
                "]\n",
                "\n",
                "stage_summary = {}\n",
                "for name, cmd in stages:\n",
                "    remaining = SOFT_DEADLINE_SEC - (time.time() - SESSION_T0)\n",
                "    if remaining < 300:\n",
                "        print(f'[TIME GUARD] Skipping {name} (remaining={remaining:.0f}s < 300s)', flush=True)\n",
                "        stage_summary[name] = {'status': 'skipped_time_guard', 'remaining_sec': round(remaining, 1)}\n",
                "        break\n",
                "    print('=' * 80, flush=True)\n",
                "    print(f'[CHAIN START] {name} | remaining_budget={remaining/3600:.2f}h', flush=True)\n",
                "    print('CMD:', ' '.join(cmd), flush=True)\n",
                "    print('=' * 80, flush=True)\n",
                "    t0 = time.time()\n",
                "    try:\n",
                "        proc = subprocess.run(cmd, cwd=str(SFT), env=env, timeout=max(300, int(remaining)), check=False)\n",
                "        stage_summary[name] = {\n",
                "            'status': 'ok' if proc.returncode == 0 else f'exit_{proc.returncode}',\n",
                "            'elapsed_min': round((time.time() - t0) / 60.0, 2),\n",
                "        }\n",
                "    except subprocess.TimeoutExpired:\n",
                "        print(f'[TIME GUARD] {name} reached soft deadline; partial checkpoints preserved cleanly.', flush=True)\n",
                "        stage_summary[name] = {\n",
                "            'status': 'partial_timeout_preserved',\n",
                "            'elapsed_min': round((time.time() - t0) / 60.0, 2),\n",
                "        }\n",
                "        break\n",
                "    gc.collect()\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n",
                "\n",
                "summary_path = RES_DIR / 'chain_execution_summary.json'\n",
                "summary_path.write_text(json.dumps(stage_summary, indent=2, ensure_ascii=False), encoding='utf-8')\n",
                "zip_path = Path('/kaggle/working/ind400_dev500_chain_results.zip')\n",
                "with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:\n",
                "    for fp in sorted(RES_DIR.glob('*.json')):\n",
                "        zf.write(fp, arcname=fp.name)\n",
                "print('[CHAIN COMPLETE] Summary:', json.dumps(stage_summary, indent=2), flush=True)\n",
                "print('[CHAIN COMPLETE] Packed ->', zip_path, f'({zip_path.stat().st_size/1024:.1f} KB)', flush=True)\n",
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
    nb_path = kernel_dir / f"{CHAIN_KERNEL_SLUG}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")

    kmeta = {
        "id": f"{username}/{CHAIN_KERNEL_SLUG}",
        "title": "SODALS Independent-400 and Dev-500 Chain T4 x2",
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{username}/{PROJECT_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    return nb_path


def main() -> int:
    api, creds = get_kaggle_api()
    username = resolve_kaggle_username("brkozt")
    if username == "YOUR_KAGGLE_USERNAME":
        username = str(creds.get("username") or "brkozt")

    k_dir = Path(tempfile.gettempdir()) / "sodals_kaggle_ind400_dev500_chain"
    build_chain_notebook(username, k_dir)
    kernel_ref = f"{username}/{CHAIN_KERNEL_SLUG}"
    print(f"[KAGGLE] Pushing 12h batch chain kernel: {kernel_ref}", flush=True)
    kn_resp = api.kernels_push(folder=str(k_dir), acc=DEFAULT_ACCELERATOR, no_run=False)
    if getattr(kn_resp, "error", None):
        raise RuntimeError(f"Kernel push error: {kn_resp.error}")

    print(
        json.dumps(
            {
                "kernel_ref": kernel_ref,
                "url": getattr(kn_resp, "url", None),
                "version": getattr(kn_resp, "versionNumber", None),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Poll briefly to verify Kaggle accepted and transitioned the kernel to RUNNING (or QUEUED)
    for attempt in range(1, 13):
        st = api.kernels_status(kernel_ref)
        status_str = str(getattr(st, "status", "UNKNOWN"))
        print(f"[VERIFY {attempt:02d}/12] {kernel_ref} -> {status_str}", flush=True)
        if "RUNNING" in status_str.upper():
            return 0
        if "ERROR" in status_str.upper() or "CANCEL" in status_str.upper():
            return 1
        time.sleep(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
