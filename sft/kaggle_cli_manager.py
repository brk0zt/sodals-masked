"""
Automated Kaggle CLI Orchestrator for Svon Epoch II & III (`GPU T4 x2`)
=======================================================================
Automates staging, dataset upload/versioning, T4 x2 kernel submission, status
polling, log retrieval, and output artifact downloading/extraction via the
official `kaggle` CLI / SDK.

Credentials expected in ONE of:
  1. `%USERPROFILE%\\.kaggle\\kaggle.json` (Windows default: `C:\\Users\\Burak\\.kaggle\\kaggle.json`)
  2. `%KAGGLE_CONFIG_DIR%\\kaggle.json`
  3. Env vars `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`)

Usage:
  python sft/kaggle_cli_manager.py stage
  python sft/kaggle_cli_manager.py check
  python sft/kaggle_cli_manager.py push --wait_dataset_sec 180
  python sft/kaggle_cli_manager.py watch --poll_sec 30
  python sft/kaggle_cli_manager.py download
  python sft/kaggle_cli_manager.py run-all
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SFT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = SFT_DIR / "kaggle_bundle"
DATASET_STAGE_DIR = BUNDLE_DIR / "dataset_pkg"
KERNEL_STAGE_DIR = BUNDLE_DIR / "kernel_pkg"
DEFAULT_OUTPUT_DIR = SFT_DIR / "kaggle_outputs"

DATASET_TITLE = "Svon Epoch II and III Code Bundle"
DATASET_SLUG_NAME = "svon-epoch-ii-and-iii-code-bundle"
KERNEL_TITLE = "Svon Epoch II and III T4 x2"
KERNEL_SLUG_NAME = "svon-epoch-ii-and-iii-t4-x2"
NOTEBOOK_FILENAME = "kaggle_svon_epoch2_epoch3_t4x2.ipynb"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"  # Kaggle API enum for GPU T4 x2 (dual 16 GB)

PLACEHOLDER_USER = "YOUR_KAGGLE_USERNAME"

DATASET_TOP_LEVEL_FILES: Tuple[str, ...] = (
    "cloud_runtime_utils.py",
    "env_reproducibility.py",
    "soft_lora_blender.py",
    "svon_identity_tracer.py",
    "svon_alphaedit_surgeon.py",
    "eval_svon_identity_and_regression.py",
    "eval_local_full_stack_compatibility.py",
    "svon_cvnn_deltanet.py",
    "svon_sheaf_gnn_attention.py",
    "svon_kuramoto_equilibrium.py",
    "train_svon_cgw_architecture.py",
    "run_kaggle_svon_epoch2_epoch3.py",
    "math_train_hard_mined_zpd_prompts.json",
    "sodals_serve.py",
)

DATASET_REL_FILES: Tuple[str, ...] = (
    "tests/prod_suite/test_svon_alphaedit_surgeon.py",
    "tests/prod_suite/test_svon_cgw_hybrid_gnn_cvnn.py",
)

TERMINAL_KERNEL_STATUSES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


def find_kaggle_credentials(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Discovers Kaggle credentials from `access_token`, `kaggle.json`, or environment variables."""
    cand_dirs: List[Path] = []
    if config_dir:
        cand_dirs.append(Path(config_dir))
    env_cfg = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_cfg:
        cand_dirs.append(Path(env_cfg))
    cand_dirs.append(Path.home() / ".kaggle")

    env_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_tok:
        return {
            "found": True,
            "auth_type": "access_token",
            "source": "environment:KAGGLE_API_TOKEN",
            "config_dir": str(cand_dirs[0]),
            "username": os.environ.get("KAGGLE_USERNAME", "").strip() or None,
            "has_key": True,
            "access_token": env_tok,
        }

    for d in cand_dirs:
        tok_file = d / "access_token"
        try:
            if tok_file.is_file():
                tok = tok_file.read_text(encoding="utf-8").strip()
                if tok:
                    return {
                        "found": True,
                        "auth_type": "access_token",
                        "source": str(tok_file),
                        "config_dir": str(d),
                        "username": os.environ.get("KAGGLE_USERNAME", "").strip() or None,
                        "has_key": True,
                        "access_token": tok,
                    }
        except (PermissionError, OSError):
            pass

        cfg_file = d / "kaggle.json"
        try:
            if cfg_file.is_file():
                data = json.loads(cfg_file.read_text(encoding="utf-8"))
                user = str(data.get("username") or "").strip()
                key = str(data.get("key") or "").strip()
                if user and key:
                    return {
                        "found": True,
                        "auth_type": "kaggle_json",
                        "source": str(cfg_file),
                        "config_dir": str(d),
                        "username": user,
                        "has_key": True,
                    }
        except (PermissionError, OSError):
            pass
        except Exception as exc:
                return {
                    "found": False,
                    "source": str(cfg_file),
                    "error": f"Failed to parse {cfg_file}: {exc}",
                    "username": None,
                }

    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_key = os.environ.get("KAGGLE_KEY", "").strip()
    if env_user and env_key:
        return {
            "found": True,
            "auth_type": "legacy_env",
            "source": "environment",
            "config_dir": env_cfg or str(Path.home() / ".kaggle"),
            "username": env_user,
            "has_key": True,
        }

    return {
        "found": False,
        "source": str(Path.home() / ".kaggle" / "access_token"),
        "config_dir": str(cand_dirs[0]),
        "username": env_user or None,
        "has_key": False,
    }


def resolve_kaggle_username(explicit_username: Optional[str] = None, config_dir: Optional[str] = None) -> str:
    """Returns explicit username, credential username, or placeholder if absent."""
    if explicit_username and explicit_username.strip():
        return explicit_username.strip()
    creds = find_kaggle_credentials(config_dir=config_dir)
    if creds.get("username"):
        return str(creds["username"])
    return PLACEHOLDER_USER


def build_dataset_metadata(username: str, dataset_slug: str = DATASET_SLUG_NAME) -> Dict[str, Any]:
    """Builds a valid Kaggle `dataset-metadata.json` payload."""
    return {
        "title": DATASET_TITLE,
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "MIT"}],
        "keywords": ["nlp", "qwen", "lora", "gnn", "complex-neural-networks"],
    }


def build_kernel_metadata(
    username: str,
    dataset_slug: str = DATASET_SLUG_NAME,
    kernel_slug: str = KERNEL_SLUG_NAME,
    accelerator: str = DEFAULT_ACCELERATOR,
    is_private: bool = True,
) -> Dict[str, Any]:
    """Builds a valid Kaggle `kernel-metadata.json` payload configured for GPU T4 x2."""
    return {
        "id": f"{username}/{kernel_slug}",
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK_FILENAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(is_private),
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [f"{username}/{dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _copy_if_changed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def resolve_writable_dir(explicit_dir: Optional[Path], default_dir: Path, fallback_subpath: str) -> Path:
    """Returns requested/default directory, falling back to Temp if read-only/blocked."""
    temp_root = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "Temp"
    candidates = [Path(explicit_dir)] if explicit_dir is not None else [default_dir, temp_root / fallback_subpath]
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return cand
        except OSError as exc:
            last_err = exc
    raise PermissionError(f"No writable directory found among {candidates}: {last_err}")


def stage_bundle(
    username: Optional[str] = None,
    dataset_slug: str = DATASET_SLUG_NAME,
    kernel_slug: str = KERNEL_SLUG_NAME,
    accelerator: str = DEFAULT_ACCELERATOR,
    is_private: bool = True,
    config_dir: Optional[str] = None,
    export_path_b_on_pass: bool = False,
    bundle_dir: Optional[Path] = None,
    include_prod_adapters: bool = False,
    d1_dir_name: str = "qwen_lora_adapter_grpo_math_train_100",
    d2_dir_name: str = "qwen_lora_adapter_domain2_r8_pondering",
) -> Dict[str, Any]:
    """Stages code files, notebook, and metadata JSONs into a writable bundle directory."""
    resolved_user = resolve_kaggle_username(username, config_dir=config_dir)
    root_bundle = resolve_writable_dir(bundle_dir, BUNDLE_DIR, "sodals_kaggle_bundle")
    ds_stage_dir = root_bundle / "dataset_pkg"
    kn_stage_dir = root_bundle / "kernel_pkg"

    if ds_stage_dir.exists():
        shutil.rmtree(ds_stage_dir)
    ds_stage_dir.mkdir(parents=True, exist_ok=True)
    kn_stage_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    missing_files: List[str] = []

    for fn in DATASET_TOP_LEVEL_FILES:
        src = SFT_DIR / fn
        if not src.is_file():
            missing_files.append(str(src))
            continue
        _copy_if_changed(src, ds_stage_dir / fn)
        copied_files.append(fn)

    for rel in DATASET_REL_FILES:
        src = SFT_DIR / rel
        if not src.is_file():
            missing_files.append(str(src))
            continue
        _copy_if_changed(src, ds_stage_dir / rel)
        copied_files.append(rel)

    if include_prod_adapters:
        for slot, dirname in (("d1", d1_dir_name), ("d2", d2_dir_name)):
            src_ad = SFT_DIR / dirname
            if src_ad.is_dir():
                for fn in ("adapter_config.json", "adapter_model.safetensors"):
                    fp = src_ad / fn
                    if fp.is_file():
                        rel_dst = f"prod_adapters/{slot}/{fn}"
                        _copy_if_changed(fp, ds_stage_dir / rel_dst)
                        copied_files.append(rel_dst)

    if missing_files:
        raise FileNotFoundError(f"Missing required SFT source files for Kaggle bundle: {missing_files}")

    ds_meta = build_dataset_metadata(resolved_user, dataset_slug=dataset_slug)
    ds_meta_path = ds_stage_dir / "dataset-metadata.json"
    ds_meta_path.write_text(json.dumps(ds_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    nb_src = SFT_DIR / NOTEBOOK_FILENAME
    if not nb_src.is_file():
        raise FileNotFoundError(f"Notebook template not found: {nb_src}")
    nb_dst = kn_stage_dir / NOTEBOOK_FILENAME
    _copy_if_changed(nb_src, nb_dst)

    if export_path_b_on_pass:
        nb_obj = json.loads(nb_dst.read_text(encoding="utf-8"))
        for cell in nb_obj.get("cells", []):
            if cell.get("id") == "svon-epoch2-run":
                raw_lines = [ln.rstrip("\r\n") for ln in "".join(cell.get("source", [])).splitlines()]
                code_only = "\n".join(ln for ln in raw_lines if not ln.lstrip().startswith("#"))
                if "--export_path_b_on_pass" not in code_only:
                    inserted = False
                    new_lines: List[str] = []
                    for ln in raw_lines:
                        if (not inserted) and ln.strip() == "]":
                            new_lines.append('    "--export_path_b_on_pass",')
                            inserted = True
                        new_lines.append(ln)
                    if not inserted:
                        new_lines.append('cmd_ep2.append("--export_path_b_on_pass")')
                    cell["source"] = [ln + "\n" for ln in new_lines[:-1]] + ([new_lines[-1]] if new_lines else [])
        nb_dst.write_text(json.dumps(nb_obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    kn_meta = build_kernel_metadata(
        resolved_user,
        dataset_slug=dataset_slug,
        kernel_slug=kernel_slug,
        accelerator=accelerator,
        is_private=is_private,
    )
    kn_meta_path = kn_stage_dir / "kernel-metadata.json"
    kn_meta_path.write_text(json.dumps(kn_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "username": resolved_user,
        "dataset_id": ds_meta["id"],
        "kernel_id": kn_meta["id"],
        "bundle_dir": str(root_bundle),
        "dataset_dir": str(ds_stage_dir),
        "kernel_dir": str(kn_stage_dir),
        "dataset_metadata_path": str(ds_meta_path),
        "kernel_metadata_path": str(kn_meta_path),
        "staged_file_count": len(copied_files),
        "staged_files": copied_files,
        "placeholder_username": resolved_user == PLACEHOLDER_USER,
    }


def get_kaggle_api(config_dir: Optional[str] = None):
    """Lazily authenticates and returns a `KaggleApi` instance after checking credentials."""
    creds = find_kaggle_credentials(config_dir=config_dir)
    if not creds.get("found"):
        raise RuntimeError(
            "Kaggle credentials not found. Place `~/.kaggle/access_token` (`KGAT_...`) or `kaggle.json` at:\n"
            f"  {creds['source']}\n"
            "or set KAGGLE_API_TOKEN / KAGGLE_CONFIG_DIR."
        )
    if creds.get("config_dir"):
        os.environ["KAGGLE_CONFIG_DIR"] = str(creds["config_dir"])
    if creds.get("access_token") and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = str(creds["access_token"])

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    if not creds.get("username"):
        introspected = api.get_config_value(api.CONFIG_NAME_USER)
        if introspected:
            creds["username"] = str(introspected).strip()

    sanitized_creds = {k: v for k, v in creds.items() if k != "access_token"}
    return api, sanitized_creds


def _normalize_status_str(raw_status: Any) -> str:
    val = getattr(raw_status, "name", None) or str(raw_status)
    return val.split(".")[-1].strip().upper()


def _dataset_exists(api: Any, dataset_ref: str) -> Optional[str]:
    try:
        return str(api.dataset_status(dataset_ref))
    except Exception:
        return None


def check_remote_state(api: Any, username: str, dataset_slug: str, kernel_slug: str) -> Dict[str, Any]:
    """Queries Kaggle for current dataset status, kernel status, and quota summary."""
    dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"
    ds_status = _dataset_exists(api, dataset_ref)

    kn_status: Optional[str] = None
    kn_failure: Optional[str] = None
    try:
        resp = api.kernels_status(kernel_ref)
        kn_status = _normalize_status_str(resp.status)
        kn_failure = getattr(resp, "failure_message", None)
    except Exception:
        kn_status = None

    return {
        "authenticated_as": username,
        "dataset_ref": dataset_ref,
        "dataset_status": ds_status or "not_found",
        "kernel_ref": kernel_ref,
        "kernel_status": kn_status or "not_found",
        "kernel_failure_message": kn_failure,
    }


def push_dataset_and_kernel(
    api: Any,
    stage_info: Dict[str, Any],
    wait_dataset_sec: int = 180,
    version_notes: Optional[str] = None,
    no_run: bool = False,
) -> Dict[str, Any]:
    """Uploads/versions the dataset (`dir_mode='zip'`) and pushes the T4 x2 kernel."""
    if stage_info.get("placeholder_username"):
        raise ValueError("Cannot push with placeholder username; configure kaggle.json or pass --username.")

    dataset_ref = stage_info["dataset_id"]
    dataset_dir = stage_info["dataset_dir"]
    kernel_dir = stage_info["kernel_dir"]
    notes = version_notes or f"Automated Svon bundle sync ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"

    existing = _dataset_exists(api, dataset_ref)
    if existing is None:
        print(f"[KAGGLE PUSH] Creating new private dataset: {dataset_ref}", flush=True)
        ds_resp = api.dataset_create_new(
            folder=dataset_dir,
            public=False,
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip",
        )
        ds_action = "created"
    else:
        print(f"[KAGGLE PUSH] Creating new version of dataset: {dataset_ref} (current={existing})", flush=True)
        ds_resp = api.dataset_create_version(
            folder=dataset_dir,
            version_notes=notes,
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip",
        )
        ds_action = "versioned"

    if getattr(ds_resp, "error", None):
        raise RuntimeError(f"Dataset push error: {ds_resp.error}")

    if ds_action == "versioned":
        # Kaggle briefly reports the previous version's "ready" state right after dataset_create_version.
        # Wait before polling so the kernel mounts the newly uploaded version, not the stale one.
        print(f"[KAGGLE PUSH] Waiting 15s for Kaggle to begin indexing new version of `{dataset_ref}`...", flush=True)
        time.sleep(15)

    deadline = time.time() + max(10, wait_dataset_sec)
    ds_status = "unknown"
    while time.time() < deadline:
        ds_status = (_dataset_exists(api, dataset_ref) or "pending").lower()
        print(f"[KAGGLE PUSH] Dataset `{dataset_ref}` status = {ds_status}", flush=True)
        if ds_status in ("ready", "complete"):
            break
        if ds_status in ("error", "failed"):
            raise RuntimeError(f"Dataset `{dataset_ref}` entered error state: {ds_status}")
        time.sleep(5)
    else:
        raise TimeoutError(f"Timed out waiting for dataset `{dataset_ref}` to become ready (last={ds_status})")

    print(f"[KAGGLE PUSH] Pushing T4 x2 kernel: {stage_info['kernel_id']}", flush=True)
    kn_resp = api.kernels_push(
        folder=kernel_dir,
        acc=DEFAULT_ACCELERATOR,
        no_run=no_run,
    )
    if getattr(kn_resp, "error", None):
        raise RuntimeError(f"Kernel push error: {kn_resp.error}")

    return {
        "dataset_action": ds_action,
        "dataset_ref": dataset_ref,
        "dataset_status": ds_status,
        "kernel_ref": stage_info["kernel_id"],
        "kernel_url": getattr(kn_resp, "url", None),
        "kernel_version": getattr(kn_resp, "versionNumber", None),
    }


def watch_kernel(
    api: Any,
    kernel_ref: str,
    poll_sec: int = 30,
    max_wait_min: int = 360,
    log_out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Polls Kaggle kernel status until completion/error and fetches logs."""
    deadline = time.time() + max(60, max_wait_min * 60)
    last_status = "UNKNOWN"
    failure_message: Optional[str] = None

    while time.time() < deadline:
        resp = api.kernels_status(kernel_ref)
        status = _normalize_status_str(resp.status)
        failure_message = getattr(resp, "failure_message", None) or None
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        msg = f"[KAGGLE WATCH {ts}] {kernel_ref} -> {status}"
        if failure_message:
            msg += f" | failure={failure_message}"
        print(msg, flush=True)
        last_status = status
        if status in TERMINAL_KERNEL_STATUSES:
            break
        time.sleep(max(5, poll_sec))
    else:
        raise TimeoutError(f"Kernel `{kernel_ref}` did not reach terminal status within {max_wait_min} min (last={last_status})")

    logs_text = ""
    try:
        logs_text = str(api.kernels_logs(kernel_ref) or "")
        if log_out_path is not None and logs_text:
            log_out_path.parent.mkdir(parents=True, exist_ok=True)
            log_out_path.write_text(logs_text, encoding="utf-8")
    except Exception as exc:
        logs_text = f"<failed to retrieve logs: {exc}>"

    return {
        "kernel_ref": kernel_ref,
        "status": last_status,
        "failure_message": failure_message,
        "log_path": str(log_out_path) if log_out_path else None,
        "log_tail": "\n".join(logs_text.splitlines()[-40:]) if logs_text else "",
    }


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> List[str]:
    """Extracts zip contents safely while preventing path traversal."""
    dest_real = os.path.realpath(str(dest_dir))
    os.makedirs(dest_real, exist_ok=True)
    extracted: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = os.path.realpath(os.path.join(dest_real, member.filename))
            if os.path.commonpath([dest_real, target]) != dest_real:
                raise ValueError(f"Refusing unsafe zip member path: {member.filename}")
            if member.is_dir():
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(member.filename)
    return extracted


def download_kernel_outputs(api: Any, kernel_ref: str, out_dir: Path) -> Dict[str, Any]:
    """Downloads Kaggle kernel outputs and extracts `svon_kaggle_artifacts.zip` if present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files, status_str = api.kernels_output(kernel_ref, path=str(out_dir), force=True, quiet=False)

    zip_candidates = list(out_dir.rglob("svon_kaggle_artifacts.zip"))
    extracted_files: List[str] = []
    extract_dir = out_dir / "extracted"
    if zip_candidates:
        extracted_files = safe_extract_zip(zip_candidates[0], extract_dir)

    expected_keys = (
        "svon_tweezer_neuron_map.json",
        "svon_identity_regression_report.json",
        "cloud_pipeline_summary.json",
        "svon_cgw_hybrid.safetensors",
        "svon_cgw_hybrid.qmem",
    )
    found_artifacts = {
        key: [str(p) for p in out_dir.rglob(key)]
        for key in expected_keys
    }

    return {
        "kernel_ref": kernel_ref,
        "output_dir": str(out_dir),
        "api_status": status_str,
        "downloaded_files": [str(f) for f in files],
        "extracted_count": len(extracted_files),
        "found_artifacts": found_artifacts,
    }


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--username", type=str, default=None, help="Kaggle username override")
    p.add_argument("--config_dir", type=str, default=None, help="Directory holding kaggle.json")
    p.add_argument("--dataset_slug", type=str, default=DATASET_SLUG_NAME)
    p.add_argument("--kernel_slug", type=str, default=KERNEL_SLUG_NAME)
    p.add_argument("--include_prod_adapters", action="store_true", help="Bundle D1 (GRPO) & D2 (Algebra) production adapters")
    p.add_argument("--d1_dir_name", type=str, default="qwen_lora_adapter_grpo_math_train_100")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated Kaggle CLI pipeline for Svon Epoch II & III (T4 x2)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stage = sub.add_parser("stage", help="Stage SFT files & generate dataset-metadata.json + kernel-metadata.json")
    _add_common_args(p_stage)
    p_stage.add_argument("--public", action="store_true", help="Make kernel public instead of private")
    p_stage.add_argument("--export_path_b_on_pass", action="store_true", help="Patch notebook to export Path B FP16 on gate pass")

    p_check = sub.add_parser("check", help="Validate kaggle.json authentication and remote dataset/kernel state")
    _add_common_args(p_check)

    p_push = sub.add_parser("push", help="Stage, push/version dataset, wait ready, and submit T4 x2 kernel")
    _add_common_args(p_push)
    p_push.add_argument("--public", action="store_true")
    p_push.add_argument("--export_path_b_on_pass", action="store_true")
    p_push.add_argument("--wait_dataset_sec", type=int, default=180)
    p_push.add_argument("--version_notes", type=str, default=None)
    p_push.add_argument("--no_run", action="store_true", help="Quick-save kernel without triggering remote execution")

    p_watch = sub.add_parser("watch", help="Poll remote Kaggle kernel status until completion and fetch logs")
    _add_common_args(p_watch)
    p_watch.add_argument("--poll_sec", type=int, default=30)
    p_watch.add_argument("--max_wait_min", type=int, default=360)
    p_watch.add_argument("--out_dir", type=str, default=None)

    p_dl = sub.add_parser("download", help="Download kernel outputs and unpack svon_kaggle_artifacts.zip")
    _add_common_args(p_dl)
    p_dl.add_argument("--out_dir", type=str, default=None)

    p_all = sub.add_parser("run-all", help="Full pipeline: check -> stage -> push -> watch -> download")
    _add_common_args(p_all)
    p_all.add_argument("--public", action="store_true")
    p_all.add_argument("--export_path_b_on_pass", action="store_true")
    p_all.add_argument("--wait_dataset_sec", type=int, default=180)
    p_all.add_argument("--version_notes", type=str, default=None)
    p_all.add_argument("--poll_sec", type=int, default=30)
    p_all.add_argument("--max_wait_min", type=int, default=360)
    p_all.add_argument("--out_dir", type=str, default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "stage":
        info = stage_bundle(
            username=args.username,
            dataset_slug=args.dataset_slug,
            kernel_slug=args.kernel_slug,
            is_private=not args.public,
            config_dir=args.config_dir,
            export_path_b_on_pass=args.export_path_b_on_pass,
            include_prod_adapters=args.include_prod_adapters,
            d1_dir_name=args.d1_dir_name,
        )
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    try:
        api, creds = get_kaggle_api(config_dir=args.config_dir)
    except Exception as exc:
        print(f"[KAGGLE AUTH ERROR] {exc}", file=sys.stderr)
        return 2

    username = resolve_kaggle_username(args.username, config_dir=args.config_dir)
    if username == PLACEHOLDER_USER:
        username = str(creds.get("username") or PLACEHOLDER_USER)

    if args.command == "check":
        state = check_remote_state(api, username, args.dataset_slug, args.kernel_slug)
        print(json.dumps({"credentials": creds, "remote": state}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "push":
        stage_info = stage_bundle(
            username=username,
            dataset_slug=args.dataset_slug,
            kernel_slug=args.kernel_slug,
            is_private=not args.public,
            config_dir=args.config_dir,
            export_path_b_on_pass=args.export_path_b_on_pass,
            include_prod_adapters=args.include_prod_adapters,
            d1_dir_name=args.d1_dir_name,
        )
        res = push_dataset_and_kernel(
            api,
            stage_info,
            wait_dataset_sec=args.wait_dataset_sec,
            version_notes=args.version_notes,
            no_run=args.no_run,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    out_dir = resolve_writable_dir(
        Path(args.out_dir) if getattr(args, "out_dir", None) else None,
        DEFAULT_OUTPUT_DIR / args.kernel_slug,
        f"sodals_kaggle_outputs/{args.kernel_slug}",
    )
    kernel_ref = f"{username}/{args.kernel_slug}"

    if args.command == "watch":
        res = watch_kernel(
            api,
            kernel_ref,
            poll_sec=args.poll_sec,
            max_wait_min=args.max_wait_min,
            log_out_path=out_dir / "kernel_execution.log",
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["status"] == "COMPLETE" else 1

    if args.command == "download":
        res = download_kernel_outputs(api, kernel_ref, out_dir)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.command == "run-all":
        stage_info = stage_bundle(
            username=username,
            dataset_slug=args.dataset_slug,
            kernel_slug=args.kernel_slug,
            is_private=not args.public,
            config_dir=args.config_dir,
            export_path_b_on_pass=args.export_path_b_on_pass,
            include_prod_adapters=args.include_prod_adapters,
            d1_dir_name=args.d1_dir_name,
        )
        push_res = push_dataset_and_kernel(
            api,
            stage_info,
            wait_dataset_sec=args.wait_dataset_sec,
            version_notes=args.version_notes,
            no_run=False,
        )
        watch_res = watch_kernel(
            api,
            kernel_ref,
            poll_sec=args.poll_sec,
            max_wait_min=args.max_wait_min,
            log_out_path=out_dir / "kernel_execution.log",
        )
        if watch_res["status"] != "COMPLETE":
            print(json.dumps({"push": push_res, "watch": watch_res}, indent=2, ensure_ascii=False))
            return 1
        dl_res = download_kernel_outputs(api, kernel_ref, out_dir)
        print(json.dumps({"push": push_res, "watch": watch_res, "download": dl_res}, indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
