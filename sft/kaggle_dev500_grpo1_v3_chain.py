#!/usr/bin/env python3
"""
SODALS v2.1 — Kaggle T4 x2 Batch Chain v3: Dev-500 Baseline + GRPO-1 LR Movement Sweep
======================================================================================
Submits a clean, unattended Kaggle T4 x2 kernel (`brkozt/sodals-dev500-and-grpo1-lr-sweep-t4-x2`)
that guarantees:
  - Strictly `batch_size=1` greedy evaluation (zero left-padding numerical drift).
  - Full `eos_token_id` (`<|im_end|>` + `<|endoftext|>`) and `--no_thinking` for stock
    `Qwen/Qwen3.5-9B` so answers terminate cleanly in ~250-450 tokens instead of truncating at 1024.
  - Stage 1: Dev-500 Baseline (`D1 balanced_r4`, FP16, N=500)
  - Stage 2: Movement-Conditioned GRPO-1 10-Step LR Sweep (`1e-4`, `3e-4`, `5e-4`; ZPD-only,
    checks `||dW||_F / ||W_0||_F >= 1.0%` or `KL >= 0.02`)
  - Stage 3: Full GRPO-1 training with winning LR + paired Dev-500 McNemar evaluation (time-guarded)
  - Deletes `/kaggle/working/sodals_core` before exit so output download is lightweight (<10 MB).
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
V3_KERNEL_SLUG = "sodals-dev500-and-grpo1-lr-sweep-t4-x2"


def build_v3_notebook(username: str, kernel_dir: Path) -> Path:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    overlay_files = [
        "eval_math_train_disjoint_dev500.py",
        "train_grpo_math_train_100.py",
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
            "id": "v3-header",
            "metadata": {},
            "source": [
                "# SODALS Core v3 — Dev-500 Baseline (`batch_size=1`, clean EOS/no-thinking) + GRPO-1 LR Sweep\n",
                "Runs on Kaggle T4 x2 (32 GB VRAM, FP16) with per-prompt checkpointing, movement-conditioned LR selection, and clean lightweight output packing.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "v3-env-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess, sys, time\n",
                "SESSION_T0 = time.time()\n",
                "SOFT_DEADLINE_SEC = 11 * 3600 + 25 * 60  # 11h25m soft guard inside Kaggle 12h limit\n",
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
            "id": "v3-stage-workspace",
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
                f"_OVERLAY_B64 = {json.dumps(overlay_b64)}\n",
                "for _fname, _b64 in _OVERLAY_B64.items():\n",
                "    (WORK / 'sft' / _fname).write_bytes(base64.b64decode(_b64))\n",
                "\n",
                "RES_DIR = Path('/kaggle/working/v3_results')\n",
                "RES_DIR.mkdir(parents=True, exist_ok=True)\n",
                "env = os.environ.copy()\n",
                "env['HF_HUB_OFFLINE'] = '0'\n",
                "env['TRANSFORMERS_OFFLINE'] = '0'\n",
                "env['PYTHONPATH'] = f'{WORK}/sft:' + env.get('PYTHONPATH', '')\n",
                "print('Staged sft/ cleanly with v3 overlays.')\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "v3-run-chain",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import gc, json, shutil, subprocess, sys, time, torch, zipfile\n",
                "from pathlib import Path\n",
                "\n",
                "SFT = WORK / 'sft'\n",
                "dev500_base_out = RES_DIR / 'dev500_baseline_v3_results.json'\n",
                "dev500_grpo1_out = RES_DIR / 'dev500_grpo1_v3_results.json'\n",
                "stage_summary = {}\n",
                "\n",
                "def run_guarded(name, cmd):\n",
                "    remaining = SOFT_DEADLINE_SEC - (time.time() - SESSION_T0)\n",
                "    if remaining < 300:\n",
                "        print(f'[TIME GUARD] Skipping {name} (remaining={remaining:.0f}s)', flush=True)\n",
                "        stage_summary[name] = {'status': 'skipped_time_guard', 'remaining_sec': round(remaining, 1)}\n",
                "        return False\n",
                "    print('=' * 80, flush=True)\n",
                "    print(f'[V3 START] {name} | remaining={remaining/3600:.2f}h', flush=True)\n",
                "    print('CMD:', ' '.join(cmd), flush=True)\n",
                "    print('=' * 80, flush=True)\n",
                "    t0 = time.time()\n",
                "    try:\n",
                "        proc = subprocess.run(cmd, cwd=str(SFT), env=env, timeout=max(300, int(remaining)), check=False)\n",
                "        stage_summary[name] = {\n",
                "            'status': 'ok' if proc.returncode == 0 else f'exit_{proc.returncode}',\n",
                "            'elapsed_min': round((time.time() - t0) / 60.0, 2),\n",
                "        }\n",
                "        ok = (proc.returncode == 0)\n",
                "    except subprocess.TimeoutExpired:\n",
                "        print(f'[TIME GUARD] {name} hit soft deadline; partial checkpoints preserved.', flush=True)\n",
                "        stage_summary[name] = {\n",
                "            'status': 'partial_timeout_preserved',\n",
                "            'elapsed_min': round((time.time() - t0) / 60.0, 2),\n",
                "        }\n",
                "        ok = False\n",
                "    gc.collect()\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n",
                "    return ok\n",
                "\n",
                "# Stage 1: Clean Dev-500 Baseline (batch_size=1, --no_thinking, full EOS, max_new_tokens=1024)\n",
                "run_guarded('stage1_dev500_baseline_v3', [\n",
                "    sys.executable, '-B', str(SFT / 'eval_math_train_disjoint_dev500.py'),\n",
                "    '--model_path', 'Qwen/Qwen3.5-9B', '--fp16', '--online', '--no_thinking',\n",
                "    '--dev_json', str(SFT / 'math_train_disjoint_dev500.json'),\n",
                "    '--adapter_path', str(SFT / 'qwen_lora_adapter_balanced_r4'),\n",
                "    '--scale', '0.25', '--max_new_tokens', '1024',\n",
                "    '--output_json', str(dev500_base_out),\n",
                "])\n",
                "\n",
                "# Stage 2: GRPO-1 10-Step LR Movement Sweep (ZPD-only, abstain OFF)\n",
                "sweep_lrs = ['1e-4', '3e-4', '5e-4']\n",
                "sweep_metrics = {}\n",
                "winning_lr = None\n",
                "for lr_str in sweep_lrs:\n",
                "    sw_json = RES_DIR / f'grpo1_sweep_lr_{lr_str}.json'\n",
                "    sw_adp = Path('/tmp') / f'grpo1_sweep_adp_{lr_str}'\n",
                "    ok = run_guarded(f'stage2_grpo1_sweep_lr_{lr_str}', [\n",
                "        sys.executable, '-B', str(SFT / 'train_grpo_math_train_100.py'),\n",
                "        '--model_path', 'Qwen/Qwen3.5-9B', '--fp16', '--online', '--no_thinking',\n",
                "        '--adapter_in', str(SFT / 'qwen_lora_adapter_balanced_r4'),\n",
                "        '--adapter_out', str(sw_adp),\n",
                "        '--output_json', str(sw_json),\n",
                "        '--num_prompts', '40', '--sweep_steps', '10', '--group_size', '4',\n",
                "        '--max_new_tokens', '1024', '--lr', lr_str, '--beta', '0.03', '--fresh_scheduler',\n",
                "    ])\n",
                "    if sw_json.exists():\n",
                "        d = json.loads(sw_json.read_text(encoding='utf-8'))\n",
                "        steps = d.get('step_details', [])\n",
                "        last_drift = steps[-1].get('rel_frob_drift_pct', 0.0) if steps else 0.0\n",
                "        max_kl = max((s.get('mean_kl_div', 0.0) for s in steps), default=0.0)\n",
                "        mean_kl = sum(s.get('mean_kl_div', 0.0) for s in steps) / max(1, len(steps))\n",
                "        moved = (last_drift >= 1.0) or (max_kl >= 0.02)\n",
                "        stable = (max_kl < 0.25)\n",
                "        sweep_metrics[lr_str] = {\n",
                "            'steps': len(steps), 'rel_frob_drift_pct': last_drift,\n",
                "            'mean_kl': round(mean_kl, 5), 'max_kl': round(max_kl, 5),\n",
                "            'moved_gate': moved, 'stable': stable,\n",
                "        }\n",
                "        if moved and stable and winning_lr is None:\n",
                "            winning_lr = lr_str\n",
                "    if not ok:\n",
                "        break\n",
                "\n",
                "stage_summary['sweep_selection'] = {'winning_lr': winning_lr, 'metrics': sweep_metrics}\n",
                "print('[SWEEP SELECTION]', json.dumps(stage_summary['sweep_selection'], indent=2), flush=True)\n",
                "\n",
                "# Stage 3: Full GRPO-1 Training + Paired Dev-500 Evaluation if winning_lr found and >=2.5h remains\n",
                "rem_after_sweep = SOFT_DEADLINE_SEC - (time.time() - SESSION_T0)\n",
                "if winning_lr is not None and rem_after_sweep >= 9000:\n",
                "    grpo1_adp_out = RES_DIR / 'qwen_lora_adapter_grpo1_v3'\n",
                "    grpo1_train_json = RES_DIR / 'grpo1_full_train_v3.json'\n",
                "    trained_ok = run_guarded('stage3a_grpo1_full_train', [\n",
                "        sys.executable, '-B', str(SFT / 'train_grpo_math_train_100.py'),\n",
                "        '--model_path', 'Qwen/Qwen3.5-9B', '--fp16', '--online', '--no_thinking',\n",
                "        '--adapter_in', str(SFT / 'qwen_lora_adapter_balanced_r4'),\n",
                "        '--adapter_out', str(grpo1_adp_out),\n",
                "        '--output_json', str(grpo1_train_json),\n",
                "        '--num_prompts', '40', '--group_size', '4',\n",
                "        '--max_new_tokens', '1024', '--lr', winning_lr, '--beta', '0.03', '--fresh_scheduler',\n",
                "    ])\n",
                "    if trained_ok and grpo1_adp_out.exists():\n",
                "        run_guarded('stage3b_dev500_grpo1_eval', [\n",
                "            sys.executable, '-B', str(SFT / 'eval_math_train_disjoint_dev500.py'),\n",
                "            '--model_path', 'Qwen/Qwen3.5-9B', '--fp16', '--online', '--no_thinking',\n",
                "            '--dev_json', str(SFT / 'math_train_disjoint_dev500.json'),\n",
                "            '--adapter_path', str(grpo1_adp_out),\n",
                "            '--baseline_json', str(dev500_base_out),\n",
                "            '--scale', '0.25', '--max_new_tokens', '1024',\n",
                "            '--output_json', str(dev500_grpo1_out),\n",
                "        ])\n",
                "\n",
                "(RES_DIR / 'v3_execution_summary.json').write_text(json.dumps(stage_summary, indent=2, ensure_ascii=False), encoding='utf-8')\n",
                "zip_path = Path('/kaggle/working/v3_results_summary.zip')\n",
                "with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:\n",
                "    for fp in sorted(RES_DIR.rglob('*')):\n",
                "        if fp.is_file():\n",
                "            zf.write(fp, arcname=str(fp.relative_to(RES_DIR)))\n",
                "\n",
                "# Clean staged workspace so Kaggle output download stays lightweight (<10 MB)!\n",
                "if WORK.exists():\n",
                "    shutil.rmtree(WORK, ignore_errors=True)\n",
                "print('[V3 COMPLETE] Summary:', json.dumps(stage_summary, indent=2), flush=True)\n",
                "print('[V3 COMPLETE] Packed ->', zip_path, f'({zip_path.stat().st_size/1024:.1f} KB)', flush=True)\n",
            ],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = kernel_dir / "v3_dev500_grpo1_chain.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")

    meta = {
        "id": f"{username}/{V3_KERNEL_SLUG}",
        "title": "SODALS Dev500 And GRPO1 LR Sweep T4 X2",
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{username}/{PROJECT_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return nb_path


def main() -> int:
    api, creds = get_kaggle_api()
    username = resolve_kaggle_username("brkozt")
    if username == "YOUR_KAGGLE_USERNAME":
        username = str(creds.get("username") or "brkozt")
    with tempfile.TemporaryDirectory(prefix="sodals_kaggle_v3_") as tmp:
        kdir = Path(tmp) / V3_KERNEL_SLUG
        build_v3_notebook(username, kdir)
        kernel_ref = f"{username}/{V3_KERNEL_SLUG}"
        print(f"[PUSH] Submitting {kernel_ref} (acc={DEFAULT_ACCELERATOR})...", flush=True)
        resp = api.kernels_push(folder=str(kdir), acc=DEFAULT_ACCELERATOR, no_run=False)
        if getattr(resp, "error", None):
            raise RuntimeError(f"Kernel push error: {resp.error}")
        print(
            json.dumps(
                {
                    "kernel_ref": kernel_ref,
                    "url": getattr(resp, "url", None),
                    "version": getattr(resp, "versionNumber", None),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        for attempt in range(1, 7):
            time.sleep(5)
            st = api.kernels_status(kernel_ref)
            status_str = str(getattr(st, "status", "UNKNOWN"))
            print(f"[VERIFY {attempt:02d}/6] {kernel_ref} -> {status_str}", flush=True)
            if "RUNNING" in status_str.upper():
                return 0
            if "ERROR" in status_str.upper() or "CANCEL" in status_str.upper():
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
